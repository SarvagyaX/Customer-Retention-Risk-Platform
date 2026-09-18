import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from insights import get_retention_insights


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


st.set_page_config(
    page_title="Customer Retention Risk Platform",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


st.title("Customer Retention Risk Platform")

st.write(
    "Customer churn analytics, risk prediction, and retention insights."
)


total_customers = len(df)

churned_customers = int(
    df["churn"].sum()
)

churn_rate = (
    df["churn"].mean() * 100
)

value_at_risk = df.loc[
    df["churn"] == 1,
    "customer_value"
].sum()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Customer Value at Risk",
    f"{value_at_risk:,.2f}"
)


st.divider()


st.subheader("Retention Insights")

insights = get_retention_insights(df)

for insight in insights:
    st.write(f"• {insight}")


st.divider()


left, right = st.columns(2)


with left:
    st.subheader("Churn by Tariff Plan")

    tariff_data = (
        df.groupby("tariff_plan")["churn"]
        .mean()
        .mul(100)
    )

    st.bar_chart(tariff_data)


with right:
    st.subheader("Churn by Age Group")

    age_data = (
        df.groupby("age_group")["churn"]
        .mean()
        .mul(100)
    )

    st.bar_chart(age_data)


st.divider()


st.subheader("Customer Risk Prediction")


col1, col2, col3 = st.columns(3)


with col1:

    call_failure = st.number_input(
        "Call Failures",
        min_value=0,
        value=10
    )

    complains = st.number_input(
        "Complaints",
        min_value=0,
        value=1
    )

    subscription_length = st.number_input(
        "Subscription Length",
        min_value=1,
        value=12
    )

    charge_amount = st.number_input(
        "Charge Amount",
        min_value=0,
        value=20
    )


with col2:

    seconds_of_use = st.number_input(
        "Seconds of Use",
        min_value=0,
        value=3000
    )

    frequency_of_use = st.number_input(
        "Frequency of Use",
        min_value=0,
        value=30
    )

    frequency_of_sms = st.number_input(
        "Frequency of SMS",
        min_value=0,
        value=15
    )

    distinct_called_numbers = st.number_input(
        "Distinct Called Numbers",
        min_value=0,
        value=20
    )


with col3:

    age_group = st.number_input(
        "Age Group",
        min_value=1,
        value=3
    )

    tariff_plan = st.number_input(
        "Tariff Plan",
        min_value=1,
        value=1
    )

    status = st.number_input(
        "Status",
        min_value=0,
        value=1
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        value=80.0
    )


if st.button("Predict Customer Risk"):

    customer = {
        "call_failure": call_failure,
        "complains": complains,
        "subscription_length": subscription_length,
        "charge_amount": charge_amount,
        "seconds_of_use": seconds_of_use,
        "frequency_of_use": frequency_of_use,
        "frequency_of_sms": frequency_of_sms,
        "distinct_called_numbers": distinct_called_numbers,
        "age_group": age_group,
        "tariff_plan": tariff_plan,
        "status": status,
        "age": age,
        "customer_value": customer_value
    }

    from src.risk_prediction import predict_risk

    result = predict_risk(customer)

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Churn Probability",
        f"{result['churn_probability']:.2f}%"
    )

    result_col2.metric(
        "Risk Level",
        result["risk_level"]
    )
