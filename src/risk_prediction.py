import pandas as pd
import joblib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def prepare_customer_data(customer):
    df = pd.DataFrame([customer])

    if "seconds_of_use" in df.columns and "subscription_length" in df.columns:
        df["avg_usage_per_month"] = (
            df["seconds_of_use"] /
            df["subscription_length"].replace(0, 1)
        )

    if "frequency_of_use" in df.columns and "subscription_length" in df.columns:
        df["monthly_call_frequency"] = (
            df["frequency_of_use"] /
            df["subscription_length"].replace(0, 1)
        )

    if "frequency_of_sms" in df.columns and "subscription_length" in df.columns:
        df["monthly_sms_frequency"] = (
            df["frequency_of_sms"] /
            df["subscription_length"].replace(0, 1)
        )

    if "complains" in df.columns:
        df["has_complaint"] = (
            df["complains"] > 0
        ).astype(int)

    if "call_failure" in df.columns:
        df["has_call_failures"] = (
            df["call_failure"] > 0
        ).astype(int)

    activity_columns = [
        column
        for column in [
            "frequency_of_use",
            "frequency_of_sms",
            "distinct_called_numbers"
        ]
        if column in df.columns
    ]

    if activity_columns:
        df["activity_score"] = df[activity_columns].sum(axis=1)

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
        "customer_value": 80
    }

    result = predict_risk(customer)

    print(result)
