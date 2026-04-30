# src/model/train.py

import mlflow
import tensorflow as tf
from tensorflow import keras
import optuna
import numpy as np
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

    tf.random.set_seed(TRAINING_CONFIG["random_seed"])

    with mlflow.start_run(nested=True):

        mlflow.set_tag("trial_number", trial.number)

        # 🔹 Hyperparameters from config
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                MODEL_CONFIG["learning_rate_min"],
                MODEL_CONFIG["learning_rate_max"],
                log=True
            ),
            "units": trial.suggest_int(
                "units",
                MODEL_CONFIG["units_min"],
                MODEL_CONFIG["units_max"]
            ),
            "dropout": trial.suggest_float(
                "dropout",
                MODEL_CONFIG["dropout_min"],
                MODEL_CONFIG["dropout_max"]
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                MODEL_CONFIG["batch_size"]
            ),
        }

        mlflow.log_params(params)

        # 🔹 Model
        model = keras.Sequential([
            keras.layers.Input(shape=(11,)),
            keras.layers.Dense(params['units'], activation='relu'),
            keras.layers.Dropout(params['dropout']),
            keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )

        # 🔹 Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            restore_best_weights=True
        )

        # 🔹 Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MODEL_CONFIG["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stop],
            verbose=0
        )

        val_auc = max(history.history["val_auc"])
        mlflow.log_metric("val_auc", val_auc)

        # 🔹 Log model
        mlflow.tensorflow.log_model(model, name="customer_churn_model")

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

    # TEMP: load processed data (we’ll replace with DVC later)
    import pickle

    with open("data/processed/train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)

    with open("data/processed/test.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)

    run_training(X_train, y_train, X_test, y_test)