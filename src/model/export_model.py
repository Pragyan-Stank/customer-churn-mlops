"""
Export champion model from MLflow to local disk with metadata.
Used in DVC pipeline / CI before Docker build.
"""

import mlflow
import tensorflow as tf
import os
import json
from mlflow.tracking import MlflowClient

# -----------------------
# Config
# -----------------------
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "customer-churn-model"
MODEL_ALIAS = "champion"

EXPORT_DIR = "model"
MODEL_FILE = "churn_model.keras"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# -----------------------
# Get model version info
# -----------------------
model_version = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias=MODEL_ALIAS
)

run_id = model_version.run_id
version = model_version.version

print(f"Exporting model:")
print(f"Model: {MODEL_NAME}")
print(f"Alias: {MODEL_ALIAS}")
print(f"Version: {version}")
print(f"Run ID: {run_id}")

# -----------------------
# Load model
# -----------------------
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.tensorflow.load_model(model_uri)

# -----------------------
# Save model
# -----------------------
os.makedirs(EXPORT_DIR, exist_ok=True)

model_path = os.path.join(EXPORT_DIR, MODEL_FILE)
model.save(model_path)

# -----------------------
# Save metadata
# -----------------------
metadata = {
    "model_name": MODEL_NAME,
    "alias": MODEL_ALIAS,
    "version": version,
    "run_id": run_id,
    "model_uri": model_uri
}

with open(os.path.join(EXPORT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n✅ Model exported to: {model_path}")
print("✅ Metadata saved to: model/metadata.json")