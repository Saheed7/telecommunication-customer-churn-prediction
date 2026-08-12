# src/serving/inference.py
import json
from pathlib import Path

import mlflow.xgboost
import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

MODEL_DIR = Path("models/production")

_model = None
_feature_columns = None


def load_model():
    """Load the promoted model + feature columns from models/production (once)."""
    global _model, _feature_columns
    if _model is not None:
        return _model, _feature_columns

    model_path = MODEL_DIR / "model"
    if not model_path.exists():
        raise RuntimeError(
            f"No model at {model_path}. Run: python -m scripts.export_model"
        )

    _model = mlflow.xgboost.load_model(str(model_path))
    with open(MODEL_DIR / "feature_columns.json") as f:
        _feature_columns = json.load(f)["feature_columns"]

    return _model, _feature_columns


def predict_churn(raw: dict) -> dict:
    """Take one raw customer record (dict) and return a churn prediction."""
    model, feature_columns = load_model()

    df = pd.DataFrame([raw])
    df["Churn"] = "No"                       # dummy col so preprocess_data runs
    df = preprocess_data(df)
    df = build_features(df)
    df = df.drop(columns=["Churn"])

    df = df.reindex(columns=feature_columns, fill_value=0)

    proba = float(model.predict_proba(df)[:, 1][0])
    prediction = int(proba >= 0.5)
    return {
        "churn_prediction": prediction,
        "churn_probability": round(proba, 4),
        "label": "Will churn" if prediction else "Will stay",
    }

    
