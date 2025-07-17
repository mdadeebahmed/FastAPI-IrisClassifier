# app/main.py

from fastapi import FastAPI
from joblib import load
import numpy as np
from .schemas import IrisInput

app = FastAPI(title="Iris Flower Classifier API")

# Load the trained model
model = load("app/iris_model.joblib")

# Define class names (order matches model training)
class_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict")
def predict_species(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)[0]
    return {"prediction": class_names[prediction]}
