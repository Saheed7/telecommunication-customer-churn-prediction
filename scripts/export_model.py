# scripts/export_model.py
import os
import json
import shutil
from pathlib import Path

import mlflow

EXPERIMENT_NAME = "Telco Churn - XGBoost"
OUTPUT_DIR = Path("models/production")


def export_model():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri((Path.cwd() / "mlruns").as_uri())

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError("No experiment found. Run training first.")

    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found. Run training first.")
    run_id = runs[0].info.run_id

    # fresh output folder
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # download the model artifacts and the feature-column list
    model_src = client.download_artifacts(run_id, "model")
    fc_src = client.download_artifacts(run_id, "feature_columns.json")

    shutil.copytree(model_src, OUTPUT_DIR / "model")
    shutil.copy(fc_src, OUTPUT_DIR / "feature_columns.json")

    print(f"Exported model from run {run_id[:8]} -> {OUTPUT_DIR}")
    print("Contents:", [p.name for p in OUTPUT_DIR.iterdir()])


if __name__ == "__main__":
    export_model()