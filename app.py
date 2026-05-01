from fastapi import FastAPI
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from typing import List
import os

app = FastAPI()

# Load model from local disk (exported once via export_model.py)
# This avoids MLflow artifact path issues inside Docker
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded successfully from: {MODEL_PATH}")


class InputData(BaseModel):
    features: List[float] = Field(..., min_items=11, max_items=11)


@app.get("/")
def home():
    return {"message": "Customer Churn Model API is running"}


@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)

    return {
        "prediction": float(prediction[0][0])
    }