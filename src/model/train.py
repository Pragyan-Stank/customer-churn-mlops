# src/model/train.py

import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from mlflow import MlflowClient
import json
import os
import yaml


# -------------------------------
# Load config
# -------------------------------
with open("src/config/params.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_CONFIG = config["model"]
TRAINING_CONFIG = config["training"]
MLFLOW_CONFIG = config["mlflow"]


# -------------------------------
# Objective function (Optuna)
# -------------------------------
def objective(trial, X_train, y_train, X_val, y_val):

    with mlflow.start_run(nested=True):

        mlflow.set_tag("trial_number", trial.number)

        # 🔹 Hyperparameters from config ranges
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                MODEL_CONFIG["n_estimators_min"],
                MODEL_CONFIG["n_estimators_max"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                MODEL_CONFIG["max_depth_min"],
                MODEL_CONFIG["max_depth_max"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                MODEL_CONFIG["learning_rate_min"],
                MODEL_CONFIG["learning_rate_max"],
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                MODEL_CONFIG["subsample_min"],
                MODEL_CONFIG["subsample_max"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                MODEL_CONFIG["colsample_bytree_min"],
                MODEL_CONFIG["colsample_bytree_max"],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                MODEL_CONFIG["min_child_weight_min"],
                MODEL_CONFIG["min_child_weight_max"],
            ),
        }

        mlflow.log_params(params)

        # 🔹 Model
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=TRAINING_CONFIG["random_seed"],
            use_label_encoder=False,
        )

        # 🔹 Training with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 🔹 Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_pred_proba)

        mlflow.log_metric("val_auc", val_auc)

        # 🔹 Log model
        mlflow.xgboost.log_model(model, artifact_path="customer_churn_model")

        return val_auc


# -------------------------------
# Main training function
# -------------------------------
def run_training(X_train, y_train, X_val, y_val):

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    with mlflow.start_run(run_name="optuna_training"):

        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=TRAINING_CONFIG["n_trials"]
        )

        best_trial_number = study.best_trial.number
        best_val_auc = study.best_value

        mlflow.log_params({
        f"best_{k}": v for k, v in study.best_params.items()
        })
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.set_tag("best_trial_number", best_trial_number)

    # -------------------------------
    # Find best run_id
    # -------------------------------
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(MLFLOW_CONFIG["experiment_name"])

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.trial_number = '{best_trial_number}'"
    )

    if len(runs) == 0:
        raise Exception("No run found for best trial!")

    best_run_id = runs[0].info.run_id

    # -------------------------------
    # Save result for next stage
    # -------------------------------
    os.makedirs("artifacts", exist_ok=True)

    result = {
        "best_run_id": best_run_id,
        "best_val_auc": best_val_auc
    }

    with open("artifacts/train_output.json", "w") as f:
        json.dump(result, f)

    print("Training complete!")
    print(result)


# -------------------------------
# Entry point (for DVC / CLI)
# -------------------------------
if __name__ == "__main__":

    import pickle

    with open("data/processed/train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)

    with open("data/processed/test.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)

    run_training(X_train, y_train, X_test, y_test)