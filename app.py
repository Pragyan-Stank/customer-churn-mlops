from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List
import os
import pickle
import joblib

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

# Load model and scaler from local disk (exported once via export_model.py / preprocess.py)
# This avoids MLflow artifact path issues inside Docker
MODEL_PATH = os.getenv("MODEL_PATH", "model/churn_model.pkl")
model = joblib.load(MODEL_PATH)
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

    # predict_proba returns [[prob_class_0, prob_class_1]]
    # We return class 1 (churn) probability
    prediction = model.predict_proba(arr_scaled)[:, 1]

    return {
        "prediction": float(prediction[0])
    }