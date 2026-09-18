from fastapi import FastAPI
from pydantic import BaseModel

from src.risk_prediction import predict_risk


app = FastAPI(
    title="Customer Retention Risk API",
    version="1.0.0"
)


class CustomerData(BaseModel):
    call_failure: int
    complains: int
    subscription_length: int
    charge_amount: int
    seconds_of_use: int
    frequency_of_use: int
    frequency_of_sms: int
    distinct_called_numbers: int
    age_group: int
    tariff_plan: int
    status: int
    age: int
    customer_value: float


@app.get("/")
def home():
    return {
        "message": "Customer Retention Risk API",
        "status": "running"
    }


@app.post("/predict")
def predict(customer: CustomerData):
    result = predict_risk(
        customer.model_dump()
    )

    return result
