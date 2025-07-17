# Iris Classifier FastAPI Application

This project is a FastAPI-based web application for predicting the species of an Iris flower based on its sepal and petal dimensions. The application serves a trained machine learning model (Logistic Regression) built using scikit-learn and is exposed via an API endpoint.

## Features

- Accepts user input via POST request (JSON) for sepal and petal measurements.
- Returns the predicted Iris species.
- Built with FastAPI for quick and efficient web deployment.

## How to Run

1. **Install dependencies**
```
pip install -r requirements.txt
```

2. **Start the FastAPI server**
```
uvicorn main:app --reload
```

3. **Access the interactive API docs**
Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```

## Sample Request

Send a POST request to `/predict` with a JSON body like:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Sample Response
```json
{
  "prediction": "setosa"
}
```

## Files

- `main.py`: FastAPI application entry point.
- `model.pkl`: Trained scikit-learn model.
- `schemas.py`: Pydantic model for input validation.
- `README.md`: Project documentation.
- `requirements.txt`: Required Python packages.

## Learnings

- Built and deployed a machine learning model with FastAPI.
- Understood API design and data validation using Pydantic.
- Gained experience in full-cycle deployment of ML applications.
