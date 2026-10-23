
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_pipeline.joblib"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("Fraudulent Transaction Detection System")
st.write("Upload transaction data to identify potentially fraudulent transactions.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run src/train_model.py first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        missing_columns = [
            column for column in FEATURE_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            st.error(
                "The CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        X = data[FEATURE_COLUMNS].copy()

        if X.isnull().any().any():
            st.error("The uploaded data contains missing values.")
            st.stop()

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            st.error("All transaction feature columns must contain numeric values.")
            st.stop()

        if not X.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
            st.error("The uploaded data contains infinite or invalid values.")
            st.stop()

        if st.button("Detect Fraud", type="primary"):
            predictions = model.predict(X)

            results = data.copy()
            results["Prediction"] = predictions
            results["Result"] = results["Prediction"].map({
                0: "Legitimate",
                1: "Potential Fraud"
            })

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[:, 1]
                results["Fraud Probability (%)"] = (
                    probabilities * 100
                ).round(2)

            total = len(results)
            fraud_count = int((predictions == 1).sum())
            legitimate_count = int((predictions == 0).sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Transactions Analyzed", f"{total:,}")
            col2.metric("Potential Fraud", f"{fraud_count:,}")
            col3.metric("Legitimate Predictions", f"{legitimate_count:,}")

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download Results as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

            st.warning(
                "Predictions are model estimates, not proof of fraud. "
                "Review flagged transactions before taking action."
            )

    except Exception as error:
        st.error(f"Unable to process the CSV: {error}")
