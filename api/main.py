from fastapi import FastAPI
from api.schemas import CustomerData
from api.predictor import predict_churn

app = FastAPI(title="Customer Churn Predictor")

@app.get("/")
def root():
    return {"message": "Customer churn prediction API is up!"}

@app.post("/predict")
def predict(data: CustomerData):
    prob = predict_churn(data.dict())
    return {"churn_probability": round(prob, 4)}
