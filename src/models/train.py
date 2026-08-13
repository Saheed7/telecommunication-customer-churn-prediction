# Training module 
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# ---- Configuration (all the knobs in one place) ----
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
TARGET = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5
EXPERIMENT_NAME = "Telco Churn - XGBoost"

PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


def setup_mlflow():
    """Point MLflow at a local ./mlruns folder (Windows-safe URI + file-store opt-in)."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlruns_path = Path.cwd() / "mlruns"
    mlruns_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def train():
    # 1. Build the dataset by running our pipeline modules in order
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    df = build_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # stratify keeps the churn ratio identical in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2. Handle class imbalance: tell XGBoost to weight churners more heavily.
    #    Computed from TRAIN ONLY so no test information leaks in.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Use tuned hyperparameters if available, else fall back to defaults
    best_params_path = Path("models/best_params.json")
    if best_params_path.exists():
        import json
        with open(best_params_path) as f:
            tuned = json.load(f)
        params = {**PARAMS, **tuned, "scale_pos_weight": scale_pos_weight}
        print(f"Using tuned hyperparameters from {best_params_path}")
    else:
        params = {**PARAMS, "scale_pos_weight": scale_pos_weight}
        print("No tuned params found; using defaults.")


    # 3. Everything inside this block is recorded as ONE MLflow run
    setup_mlflow()
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("threshold", THRESHOLD)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= THRESHOLD).astype(int)

        metrics = {
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)

        # Save the model AND the exact feature-column order (critical for serving)
        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.log_dict({"feature_columns": list(X.columns)}, "feature_columns.json")

        print("\n===== TRAINING COMPLETE =====")
        print(f"scale_pos_weight: {scale_pos_weight:.3f}")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.3f}")
        print("\n" + classification_report(y_test, preds, digits=3))
        print(f"MLflow run logged to: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    train()