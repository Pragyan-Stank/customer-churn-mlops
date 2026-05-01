# src/model/register.py

import mlflow
from mlflow import MlflowClient
import json
import yaml


# -------------------------------
# Load config
# -------------------------------
with open("src/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)

MLFLOW_CONFIG = config["mlflow"]
REGISTRY_CONFIG = config["registry"]


# -------------------------------
# Register Logic
# -------------------------------
def register_model():

    # -------------------------------
    # Load evaluation output
    # -------------------------------
    with open("artifacts/eval_output.json", "r") as f:
        eval_output = json.load(f)

    decision = eval_output["decision"]
    best_run_id = eval_output["best_run_id"]

    print(f"Decision: {decision.upper()}")

    # -------------------------------
    # STOP if model failed
    # -------------------------------
    if decision != "pass":
        print("Model did not meet criteria. Skipping registration.")
        return

    # -------------------------------
    # MLflow setup
    # -------------------------------
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    model_name = REGISTRY_CONFIG["model_name"]
    alias = REGISTRY_CONFIG["alias"]

    # -------------------------------
    # Register model
    # -------------------------------
    model_uri = f"runs:/{best_run_id}/customer_churn_model"

    print(f"Registering model from run: {best_run_id}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Registered Model Version: {result.version}")

    # -------------------------------
    # Set alias (champion)
    # -------------------------------
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=result.version
    )

    print(f"Alias '{alias}' updated successfully!")


# -------------------------------
# Entry point (DVC compatible)
# -------------------------------
if __name__ == "__main__":
    register_model()