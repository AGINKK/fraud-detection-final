# Fraudulent Transaction Detection System

## Project Overview

The Fraudulent Transaction Detection System is a machine learning project that identifies potentially fraudulent credit card transactions. It uses Logistic Regression to classify transactions as legitimate or potentially fraudulent and provides an interactive Streamlit web application for predictions.

## Objectives

- Analyze credit card transaction data.
- Explore and understand class imbalance.
- Train a machine learning classification model.
- Evaluate the model using suitable performance metrics.
- Predict transactions through a CSV upload interface.
- Download prediction results for further analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Dataset

**Source:** [Credit Card Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions and 31 columns.

- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
- `V1`–`V28`: Anonymized numerical features obtained through Principal Component Analysis (PCA).
- `Amount`: Transaction amount.
- `Class`: Target label — `0` for legitimate transactions and `1` for fraudulent transactions.

The dataset contains a severe class imbalance, with fraudulent transactions representing approximately 0.17% of all records.

The raw dataset is excluded from this repository because of its size. Download it from Kaggle and place `creditcard.csv` inside `data/raw/`.

## Machine Learning Model

The project uses **Logistic Regression** with the following pipeline:

1. StandardScaler for feature scaling.
2. Logistic Regression with `class_weight="balanced"` to account for class imbalance.
3. A stratified train-test split, with 80% of the data used for training and 20% for testing.

## Model Evaluation

Results from the current test set:

| Metric | Score |
|---|---:|
| Accuracy | 97.55% |
| Precision | 6.10% |
| Recall | 91.84% |
| F1-score | 11.44% |

These results show that the model detects most fraudulent transactions, but its low precision means many transactions flagged as fraudulent are actually legitimate. Further tuning and evaluation are needed before real-world use.

The project also saves a confusion matrix and detailed evaluation metrics in the `reports/` directory.

## Streamlit Application

The web application allows users to:

- Upload a transaction CSV file.
- Detect potentially fraudulent transactions.
- View prediction summaries and individual results.
- View estimated fraud probabilities when supported by the model.
- Download prediction results as a CSV file.

Predictions are estimates and should not be treated as proof of fraud.

## Project Structure

```text
fraud-detection-final/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── raw/
├── models/
│   └── fraud_detection_pipeline.joblib
├── notebooks/
│   └── 01_data_understanding.ipynb
├── reports/
│   ├── confusion_matrix.png
│   └── evaluation_metrics.json
└── src/
    └── train_model.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd fraud-detection-final
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from Kaggle and place `creditcard.csv` in:

```text
data/raw/creditcard.csv
```

### 5. Train the model

```bash
python src/train_model.py
```

This creates the trained model and evaluation reports.

### 6. Launch the application

```bash
python -m streamlit run app.py
```

## Future Improvements

- Tune the classification threshold to improve precision and recall.
- Compare Logistic Regression with Random Forest and other suitable classifiers.
- Evaluate additional metrics such as Precision-Recall AUC.
- Investigate false positives and false negatives.
- Improve the Streamlit dashboard with additional visualizations.

## Disclaimer

This project is intended for educational and demonstration purposes. It has not been validated for production financial decisions.

## Author

**AGIN KK**

GitHub: [AGINKK](https://github.com/AGINKK)
