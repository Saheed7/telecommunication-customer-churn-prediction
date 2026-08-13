# src/models/tune.py
import os
import json
from pathlib import Path

import mlflow
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
TARGET = "Churn"
RANDOM_STATE = 42
N_TRIALS = 30
CV_FOLDS = 3
TUNING_EXPERIMENT = "Telco Churn - Tuning"
BEST_PARAMS_PATH = Path("models/best_params.json")


def _setup_mlflow():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    (Path.cwd() / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri((Path.cwd() / "mlruns").as_uri())
    mlflow.set_experiment(TUNING_EXPERIMENT)


def tune():
    # Build the dataset with the same pipeline as training
    df = build_features(preprocess_data(load_data(DATA_PATH)))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    _setup_mlflow()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
        }
        model = XGBClassifier(**params)
        # cross-validated ROC-AUC on the TRAINING set only (no test leakage)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        # log each trial to MLflow so the dashboard can compare them
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"})
            mlflow.log_metric("cv_roc_auc", score)
        return score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print("\n===== TUNING COMPLETE =====")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best CV ROC-AUC: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved best params -> {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    tune()