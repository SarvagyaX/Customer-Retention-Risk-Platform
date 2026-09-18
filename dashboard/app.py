import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from insights import get_retention_insights
from src.risk_prediction import predict_risk


DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


st.set_page_config(
    page_title="Customer Retention Risk Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.75;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .insight-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 0.7rem;
    }

    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 1rem;
    }

    .footer {
        text-align: center;
        opacity: 0.6;
        padding-top: 2rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Retention Risk Platform</h1>
        <p>Customer churn analytics, risk prediction, and retention intelligence.</p>
    </div>
    """,
    unsafe_allow_html=True
)


total_customers = len(df)
churned_customers = int(df["churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate = df["churn"].mean() * 100
value_at_risk = df.loc[df["churn"] == 1, "customer_value"].sum()


st.markdown(
    '<div class="section-title">Business Overview</div>',
    unsafe_allow_html=True
)


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
    "Retention Rate",
    f"{100 - churn_rate:.2f}%"
)

col4.metric(
    "Customer Value at Risk",
    f"{value_at_risk:,.2f}"
)


st.divider()


st.markdown(
    '<div class="section-title">Retention Intelligence</div>',
    unsafe_allow_html=True
)


insights = get_retention_insights(df)

for insight in insights:
    st.markdown(
        f'<div class="insight-box">💡 {insight}</div>',
        unsafe_allow_html=True
    )


st.divider()


st.markdown(
    '<div class="section-title">Churn Analysis</div>',
    unsafe_allow_html=True
)


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.subheader("Churn by Tariff Plan")

    tariff_data = (
        df.groupby("tariff_plan")["churn"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(
        tariff_data,
        height=350
    )


with chart_col2:
    st.subheader("Churn by Age Group")

    age_data = (
        df.groupby("age_group")["churn"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(
        age_data,
        height=350
    )


st.divider()


st.markdown(
    '<div class="section-title">Customer Risk Assessment</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter customer activity and account information to estimate churn risk."
)


input_col1, input_col2, input_col3 = st.columns(3)


with input_col1:

    st.markdown("**Usage & Communication**")

    call_failure = st.number_input(
        "Call Failures",
        min_value=0,
        value=10
    )

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


with input_col2:

    st.markdown("**Account Information**")

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

    customer_value = st.number_input(
        "Customer Value",
        min_value=0.0,
        value=80.0
    )


with input_col3:

    st.markdown("**Customer Profile**")

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

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


st.write("")


if st.button(
    "🔍 Predict Customer Risk",
    use_container_width=True
):

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

    result = predict_risk(customer)

    probability = result["churn_probability"]
    risk_level = result["risk_level"]


    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )

    st.subheader("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    result_col1.metric(
        "Churn Probability",
        f"{probability:.2f}%"
    )


    result_col2.metric(
        "Risk Level",
        risk_level
    )


    result_col3.metric(
        "Customer Value",
        f"{customer_value:,.2f}"
    )


    if risk_level == "High":
        st.error(
            "High-risk customer identified. This customer may require priority retention attention."
        )

    elif risk_level == "Medium":
        st.warning(
            "Medium-risk customer identified. Consider monitoring activity and engagement."
        )

    else:
        st.success(
            "Low-risk customer identified. The current customer profile shows relatively low predicted churn risk."
        )


    st.progress(
        min(probability / 100, 1.0)
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


st.markdown(
    '<div class="footer">Customer Retention Risk Platform • Machine Learning + SQL + Business Analytics</div>',
    unsafe_allow_html=True
)
