# tests/test_pipeline.py
import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


def _sample_raw():
    """One raw customer row, including the blank-TotalCharges edge case."""
    return pd.DataFrame([{
        "customerID": "0001-ABCDE",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35,
        "TotalCharges": " ",  # blank, exactly as in the raw dataset
        "Churn": "No",
    }])


def test_preprocess_cleans_data():
    clean = preprocess_data(_sample_raw())
    assert "customerID" not in clean.columns          # ID dropped
    assert clean["TotalCharges"].iloc[0] == 0         # blank -> 0
    assert clean["TotalCharges"].dtype.kind in "fi"   # now numeric
    assert clean["Churn"].iloc[0] == 0                # target mapped to 0/1


def test_build_features_is_all_numeric():
    feats = build_features(preprocess_data(_sample_raw()))
    assert feats.select_dtypes(exclude="number").empty   # every column numeric
    assert "Churn" in feats.columns                       # target survives