import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit app settings
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")
st.title("💳 Credit Card Default Prediction App (XGBoost)")

st.write("""
Upload a dataset (CSV) with customer credit card details, 
and this app will predict whether each customer will **default next month**.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview")
    st.dataframe(data.head())

    # Drop target if present (to avoid leakage)
    X_new = data.drop(columns=["default.payment.next.month"], errors="ignore")

    # Predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]  # probability of default

    # Add results
    data["Prediction"] = predictions
    data["Default_Probability"] = probabilities

    st.write("### ✅ Predictions")
    st.dataframe(data.head(20))

    # Download predictions
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

    # 📊 Model Evaluation (only if actual labels exist)
    if "default.payment.next.month" in data.columns:
        y_true = data["default.payment.next.month"]
        y_pred = predictions
        y_prob = probabilities

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)

        st.write("### 📈 Model Evaluation Metrics")
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1-score", f"{f1:.2f}")
        st.metric("ROC-AUC", f"{roc_auc:.2f}")

        # 📉 Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.write("### 📉 Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        st.pyplot(fig)
    else:
        st.info("ℹ️ No target column found — showing only predictions (evaluation not possible).")
