from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow
import tensorflow
import numpy as np
from typing import List

app = FastAPI()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.tensorflow.load_model(
    "models:/customer-churn-model@champion"
)

print("Model loaded successfully!")

class InputData(BaseModel):
    features: List[float] = Field(..., min_items=11, max_items=11)


@app.post("/predict")
def predict(data: InputData):   # ← IMPORTANT CHANGE
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)

    return {
        "prediction": float(prediction[0][0])
    }