import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")

st.title("💳 Credit Card Default Prediction")
st.write("Upload a CSV, align columns to the model schema, run predictions, and optionally evaluate with metrics.")

# ---------- Utilities ----------
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_expected_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    try:
        booster = model.get_booster()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    return None

def align_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    incoming_map = {c.lower(): c for c in df.columns}
    aligned = pd.DataFrame(index=df.index)
    for feat in expected:
        key = feat.lower()
        if key in incoming_map:
            src_col = incoming_map[key]
            aligned[feat] = pd.to_numeric(df[src_col], errors="coerce")
        else:
            aligned[feat] = 0.0
    return aligned.fillna(0.0)

def normalize_01(x: np.ndarray) -> np.ndarray:
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + 1e-12)

def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------- Load model ----------
model_path = "model.pkl"
try:
    model = load_model(model_path)
except Exception as e:
    st.error("Failed to load model.pkl. Make sure it is present in the app directory.")
    st.exception(e)
    st.stop()

expected = get_expected_features(model)
if expected is None:
    st.info("Model doesn't expose feature names. We'll coerce numeric columns and predict with best effort.")

# ---------- File upload ----------
uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read CSV. Please check the file format/encoding.")
        st.exception(e)
        st.stop()

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head(20), use_container_width=True)

    # ---- Target column selection ----
    st.subheader("🎯 Target Column (for evaluation)")
    target_col = st.selectbox(
        "Select target column (optional):",
        options=["<None>"] + list(data.columns),
        index=0
    )
    if target_col == "<None>":
        target_col = None

    # Build feature dataframe (exclude target if chosen)
    feature_df = data.drop(columns=[target_col], errors="ignore")

    if expected is not None:
        X_new = align_features(feature_df, expected)
    else:
        X_new = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ---------- Predict ----------
    with st.spinner("Running predictions..."):
        try:
            preds = model.predict(X_new)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_new)[:, 1]
            else:
                if hasattr(model, "decision_function"):
                    raw = model.decision_function(X_new)
                    probs = normalize_01(raw)
                else:
                    probs = preds.astype(float)
        except Exception as e:
            st.error("Prediction failed even after schema alignment. Please verify your CSV columns/types.")
            st.exception(e)
            st.stop()

    # ---------- Display ----------
    out = data.copy()
    out["Prediction"] = preds
    out["Default_Probability"] = probs

    st.subheader("✅ Predictions")
    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=to_csv_download(out),
        file_name="predictions.csv",
        mime="text/csv"
    )

    # ---------- Metrics + Confusion Matrix ----------
    if target_col is not None:
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, roc_auc_score, confusion_matrix
            )
            import matplotlib.pyplot as plt
            import seaborn as sns

            y_true = pd.to_numeric(data[target_col], errors="coerce").fillna(0).astype(int)
            y_pred = pd.Series(preds).astype(int)

            unique_classes = np.unique(y_true)
            if len(unique_classes) > 2:
                avg = "weighted"   # multiclass case
            else:
                avg = "binary"

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
            prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_true, y_pred, average=avg, zero_division=0)

            auc = None
            try:
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, probs, multi_class="ovr" if len(unique_classes) > 2 else "raise")
            except Exception:
                pass

            st.subheader("📈 Model Evaluation")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("F1", f"{f1:.3f}")
            c3.metric("Precision", f"{prec:.3f}")
            c4.metric("Recall", f"{rec:.3f}")
            c5.metric("AUC", f"{auc:.3f}" if auc is not None else "n/a")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=[str(c) for c in unique_classes],
                        yticklabels=[str(c) for c in unique_classes],
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        except Exception as e:
            st.info("Could not compute metrics on the uploaded file.")
            st.exception(e)

else:
    st.info("Upload a CSV to begin.")

