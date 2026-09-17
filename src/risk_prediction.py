import pandas as pd
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
    "has_complaint",
    "activity_score"
]


def load_model():
    return joblib.load(MODEL_PATH)


def prepare_customer_data(customer):
    df = pd.DataFrame([customer])

    if "complains" in df.columns:
        df["has_complaint"] = (
            df["complains"] > 0
        ).astype(int)

    activity_columns = [
        "frequency_of_use",
        "frequency_of_sms",
        "distinct_called_numbers"
    ]

    df["activity_score"] = df[activity_columns].sum(axis=1)

    df = df[FEATURES]

    return df


def predict_risk(customer):
    model = load_model()

    customer_data = prepare_customer_data(customer)

    probability = model.predict_proba(
        customer_data
    )[0][1]

    if probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": round(
            probability * 100,
            2
        ),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    customer = {
        "call_failure": 10,
        "complains": 1,
        "subscription_length": 12,
        "charge_amount": 20,
        "seconds_of_use": 3000,
        "frequency_of_use": 30,
        "frequency_of_sms": 15,
        "distinct_called_numbers": 20,
        "age_group": 3,
        "tariff_plan": 1,
        "status": 1,
        "age": 30,
        "customer_value": 80
    }

    result = predict_risk(customer)

    print(result)
