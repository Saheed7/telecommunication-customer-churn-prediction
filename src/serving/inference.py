# src/serving/inference.py
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

EXPERIMENT_NAME = "Telco Churn - XGBoost"

# module-level cache so we load the model once, not on every request
_model = None
_feature_columns = None


def _setup_mlflow():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlruns_path = Path.cwd() / "mlruns"
    mlflow.set_tracking_uri(mlruns_path.as_uri())


def load_model():
    """Load the most recent trained model + its feature columns (once)."""
    global _model, _feature_columns
    if _model is not None:
        return _model, _feature_columns

    _setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"No experiment named '{EXPERIMENT_NAME}'. Train first.")

    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found. Run training first.")
    run_id = runs[0].info.run_id

    _model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    fc_path = client.download_artifacts(run_id, "feature_columns.json")
    import json
    with open(fc_path) as f:
        _feature_columns = json.load(f)["feature_columns"]

    return _model, _feature_columns


def predict_churn(raw: dict) -> dict:
    """Take one raw customer record (dict) and return a churn prediction."""
    model, feature_columns = load_model()

    # 1. raw dict -> one-row DataFrame
    df = pd.DataFrame([raw])

    # 2. same preprocessing + feature engineering as training
    #    (add a dummy Churn col so preprocess_data can run, then drop it)
    df["Churn"] = "No"
    df = preprocess_data(df)
    df = build_features(df)
    df = df.drop(columns=["Churn"])

    # 3. align to the exact training columns (fill any missing one-hot cols with 0)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # 4. predict
    proba = float(model.predict_proba(df)[:, 1][0])
    prediction = int(proba >= 0.5)
    return {
        "churn_prediction": prediction,
        "churn_probability": round(proba, 4),
        "label": "Will churn" if prediction else "Will stay",
    }