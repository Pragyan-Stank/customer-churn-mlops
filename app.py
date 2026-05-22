from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from typing import List
import os

app = FastAPI()

# CORS: Allow frontend to communicate with the API
# In production, replace "*" with your frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pickle

# Load model and scaler from local disk (exported once via export_model.py / preprocess.py)
# This avoids MLflow artifact path issues inside Docker
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully from: {MODEL_PATH}")

SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print(f"Scaler loaded successfully from: {SCALER_PATH}")


class InputData(BaseModel):
    features: List[float] = Field(..., min_items=11, max_items=11)


@app.get("/")
def home():
    return {"message": "Customer Churn Model API is running"}


@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    
    # Scale features before prediction
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)

    return {
        "prediction": float(prediction[0][0])
    }