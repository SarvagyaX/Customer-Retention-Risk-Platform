import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

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
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #071426;
    }

    [data-testid="stHeader"] {
        background: rgba(7,20,38,0);
    }

    [data-testid="stSidebar"] {
        background: #08172d;
        border-right: 1px solid #17365f;
    }

    [data-testid="stSidebar"] * {
        color: #e7f0ff;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #f4f8ff;
        margin-bottom: 0.2rem;
    }

    .gradient-title {
        background: linear-gradient(90deg,#ffffff,#35c9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #9eb4d1;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .hero {
        background: linear-gradient(135deg,#0d2d57,#102d55 55%,#123e69);
        border: 1px solid #214e7e;
        border-radius: 22px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(32,207,255,0.12);
        border: 1px solid rgba(32,207,255,0.35);
        color: #58d7ff;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .kpi-card {
        border-radius: 18px;
        padding: 1.25rem;
        min-height: 135px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .kpi-blue {
        background: linear-gradient(135deg,#1557d6,#2678ee);
    }

    .kpi-red {
        background: linear-gradient(135deg,#d72d62,#ef4777);
    }

    .kpi-green {
        background: linear-gradient(135deg,#079b72,#21bf88);
    }

    .kpi-orange {
        background: linear-gradient(135deg,#df7b0c,#f5a623);
    }

    .kpi-label {
        color: #dbe9ff;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .kpi-value {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.45rem;
    }

    .kpi-sub {
        color: rgba(255,255,255,0.75);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    .section-card {
        background: #0b1d35;
        border: 1px solid #193a61;
        border-radius: 18px;
        padding: 1.3rem;
        margin-bottom: 1rem;
    }

    .section-title {
        color: #f4f8ff;
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.8rem;
    }

    .insight {
        background: #102743;
        border: 1px solid #234b76;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        color: #dce9fa;
    }

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        padding: 1.5rem 0.5rem;
    }

    .pipeline-box {
        background: linear-gradient(135deg,#102f55,#164a77);
        border: 1px solid #2875a8;
        border-radius: 14px;
        padding: 1rem;
        min-width: 145px;
        text-align: center;
        color: #eef7ff;
        font-weight: 700;
    }

    .pipeline-arrow {
        color: #32c9ff;
        font-size: 1.5rem;
        font-weight: 800;
    }

    .risk-card {
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #24517d;
        background: linear-gradient(135deg,#0b203b,#102d4c);
        box-shadow: 0 12px 35px rgba(0,0,0,0.22);
    }

    .risk-low {
        border-color: #18b77a;
    }

    .risk-medium {
        border-color: #f3a51b;
    }

    .risk-high {
        border-color: #ed4b59;
    }

    .risk-value {
        font-size: 2.8rem;
        font-weight: 850;
        color: white;
    }

    .risk-pill {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 800;
        margin-top: 0.5rem;
    }

    .low-pill {
        background: #0b9d68;
        color: white;
    }

    .medium-pill {
        background: #e39a12;
        color: white;
    }

    .high-pill {
        background: #df4050;
        color: white;
    }

    .footer {
        text-align: center;
        color: #7891af;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: #0d223d;
        border: 1px solid #1b3d64;
        padding: 1rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9db5d3;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f8ff;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #2ac8ff;
        background: linear-gradient(90deg,#3154ff,#08bfe9);
        color: white;
        font-weight: 750;
        min-height: 48px;
        box-shadow: 0 8px 22px rgba(19,150,220,0.25);
    }

    .stButton > button:hover {
        border-color: #7be5ff;
        color: white;
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div {
        background: #10243d;
        border-color: #244b72;
    }

    div[data-baseweb="input"] > div {
        background: #10243d;
        border-color: #244b72;
    }

    .small-note {
        color: #8ea7c4;
        font-size: 0.82rem;
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


if "page" not in st.session_state:
    st.session_state.page = "Overview"


with st.sidebar:
    st.markdown(
        """
        <div style="padding:1rem 0 1.5rem 0;">
            <div style="font-size:1.35rem;font-weight:800;color:#ffffff;">
                📊 Retention Risk
            </div>
            <div style="color:#79a0c8;font-size:0.85rem;">
                Analytics Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Analytics",
            "Risk Prediction",
            "Data Explorer"
        ],
        index=[
            "Overview",
            "Analytics",
            "Risk Prediction",
            "Data Explorer"
        ].index(st.session_state.page)
    )

    st.session_state.page = page

    st.markdown("---")

    st.markdown(
        """
        <div style="padding:0.8rem 0;">
            <div style="color:#21c98a;font-weight:700;">● Live Demo</div>
            <div style="color:#7891af;font-size:0.8rem;margin-top:0.3rem;">
                ML-powered customer retention analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


total_customers = len(df)
churned_customers = int(df["churn"].sum())
churn_rate = df["churn"].mean() * 100
retention_rate = 100 - churn_rate
value_at_risk = df.loc[df["churn"] == 1, "customer_value"].sum()


if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">● LIVE ML PLATFORM</span>
            <div class="main-title">
                Customer Retention <span class="gradient-title">Risk Platform</span>
            </div>
            <div class="subtitle">
                Customer churn analytics, risk prediction, and retention intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(
        f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">TOTAL CUSTOMERS</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-sub">Customer records analyzed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    k2.markdown(
        f"""
        <div class="kpi-card kpi-red">
            <div class="kpi-label">CHURNED CUSTOMERS</div>
            <div class="kpi-value">{churned_customers:,}</div>
            <div class="kpi-sub">{churn_rate:.2f}% of customers</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    k3.markdown(
        f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">RETENTION RATE</div>
            <div class="kpi-value">{retention_rate:.2f}%</div>
            <div class="kpi-sub">Customers retained</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    k4.markdown(
        f"""
        <div class="kpi-card kpi-orange">
            <div class="kpi-label">CUSTOMER VALUE AT RISK</div>
            <div class="kpi-value">{value_at_risk:,.2f}</div>
            <div class="kpi-sub">Value associated with churned customers</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    insight_col, chart_col = st.columns([0.95, 1.35], gap="large")

    with insight_col:
        st.markdown(
            '<div class="section-title">💡 Retention Insights</div>',
            unsafe_allow_html=True
        )

        insights = get_retention_insights(df)

        for insight in insights:
            st.markdown(
                f'<div class="insight">💡 {insight}</div>',
                unsafe_allow_html=True
            )

    with chart_col:
        st.markdown(
            '<div class="section-title">📊 Customer Churn Distribution</div>',
            unsafe_allow_html=True
        )

        churn_distribution = pd.DataFrame(
            {
                "Status": ["Retained", "Churned"],
                "Customers": [
                    total_customers - churned_customers,
                    churned_customers
                ]
            }
        )

        fig = px.pie(
            churn_distribution,
            names="Status",
            values="Customers",
            hole=0.58,
            color="Status",
            color_discrete_map={
                "Retained": "#20c98a",
                "Churned": "#ef476f"
            }
        )

        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce9fa"),
            legend=dict(
                orientation="h",
                y=-0.05
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown(
        '<div class="section-title">🔄 ML Risk Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="pipeline">
                <div class="pipeline-box">👤 Customer Data</div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-box">🧹 Data Cleaning</div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-box">⚙️ Feature Engineering</div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-box">🤖 ML Model</div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-box">🚦 Risk Level</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "Analytics":

    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">ANALYTICS</span>
            <div class="main-title">Customer Analytics</div>
            <div class="subtitle">
                Explore churn patterns across customer segments and activity levels.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    filter1, filter2, filter3 = st.columns(3)

    tariff_options = ["All"] + sorted(
        df["tariff_plan"].dropna().unique().tolist()
    )

    age_options = ["All"] + sorted(
        df["age_group"].dropna().unique().tolist()
    )

    complaint_options = ["All"] + sorted(
        df["complains"].dropna().unique().tolist()
    )

    with filter1:
        selected_tariff = st.selectbox(
            "Tariff Plan",
            tariff_options
        )

    with filter2:
        selected_age = st.selectbox(
            "Age Group",
            age_options
        )

    with filter3:
        selected_complaint = st.selectbox(
            "Complaint Status",
            complaint_options
        )

    filtered_df = df.copy()

    if selected_tariff != "All":
        filtered_df = filtered_df[
            filtered_df["tariff_plan"] == selected_tariff
        ]

    if selected_age != "All":
        filtered_df = filtered_df[
            filtered_df["age_group"] == selected_age
        ]

    if selected_complaint != "All":
        filtered_df = filtered_df[
            filtered_df["complains"] == selected_complaint
        ]

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Customers",
        f"{len(filtered_df):,}"
    )

    a2.metric(
        "Churned",
        f"{int(filtered_df['churn'].sum()):,}"
    )

    a3.metric(
        "Churn Rate",
        f"{filtered_df['churn'].mean() * 100:.2f}%"
        if len(filtered_df) > 0
        else "0.00%"
    )

    a4.metric(
        "Customer Value",
        f"{filtered_df['customer_value'].sum():,.2f}"
    )

    st.write("")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            '<div class="section-title">📱 Churn by Tariff Plan</div>',
            unsafe_allow_html=True
        )

        tariff_data = (
            filtered_df.groupby("tariff_plan")["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )

        tariff_data.columns = [
            "Tariff Plan",
            "Churn Rate"
        ]

        fig = px.bar(
            tariff_data,
            x="Tariff Plan",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[
                "#20c98a",
                "#f5a623",
                "#ef476f"
            ],
            text_auto=".1f"
        )

        fig.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce9fa"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:
        st.markdown(
            '<div class="section-title">👥 Churn by Age Group</div>',
            unsafe_allow_html=True
        )

        age_data = (
            filtered_df.groupby("age_group")["churn"]
            .mean()
            .mul(100)
            .reset_index()
        )

        age_data.columns = [
            "Age Group",
            "Churn Rate"
        ]

        fig = px.bar(
            age_data,
            x="Age Group",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[
                "#20c98a",
                "#f5a623",
                "#ef476f"
            ],
            text_auto=".1f"
        )

        fig.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce9fa"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">⚠️ Complaint vs Churn</div>',
        unsafe_allow_html=True
    )

    complaint_data = (
        filtered_df.groupby("complains")["churn"]
        .mean()
        .mul(100)
        .reset_index()
    )

    complaint_data.columns = [
        "Complaints",
        "Churn Rate"
    ]

    fig = px.bar(
        complaint_data,
        x="Complaints",
        y="Churn Rate",
        color="Churn Rate",
        color_continuous_scale=[
            "#20c98a",
            "#f5a623",
            "#ef476f"
        ],
        text_auto=".1f"
    )

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce9fa"),
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif page == "Risk Prediction":

    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">AI POWERED</span>
            <div class="main-title">Customer Risk Prediction</div>
            <div class="subtitle">
                Enter customer information to estimate churn probability and risk level.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, middle, right = st.columns(3, gap="large")

    with left:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📞 Usage & Communication</div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

    with middle:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">💳 Account Information</div>
            </div>
            """,
            unsafe_allow_html=True
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

        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=80.0
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">👤 Customer Profile</div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

    predict_button = st.button(
        "🔮 Predict Customer Risk",
        use_container_width=True
    )

    if predict_button:

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

        if risk_level == "High":
            risk_class = "risk-high"
            pill_class = "high-pill"
            message = "High-risk customer identified. Priority retention attention may be appropriate."
        elif risk_level == "Medium":
            risk_class = "risk-medium"
            pill_class = "medium-pill"
            message = "Medium-risk customer identified. Monitor customer activity and engagement."
        else:
            risk_class = "risk-low"
            pill_class = "low-pill"
            message = "Low-risk customer identified based on the current customer profile."

        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <div style="color:#8fa9c6;font-weight:700;">
                    PREDICTION RESULT
                </div>
                <div class="risk-value">
                    {probability:.2f}%
                </div>
                <div style="color:#a7bad0;">
                    Predicted churn probability
                </div>
                <div class="risk-pill {pill_class}">
                    {risk_level} RISK
                </div>
                <div style="margin-top:1rem;color:#cbd9e8;">
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={
                    "suffix": "%",
                    "font": {"color": "#ffffff"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#8da5c0"
                    },
                    "bar": {
                        "color": "#35c9ff"
                    },
                    "bgcolor": "#102640",
                    "bordercolor": "#24496e",
                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "#123e38"
                        },
                        {
                            "range": [40, 70],
                            "color": "#493b18"
                        },
                        {
                            "range": [70, 100],
                            "color": "#4b2029"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            margin=dict(l=30, r=30, t=20, b=10)
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        r2.metric(
            "Risk Level",
            risk_level
        )

        r3.metric(
            "Customer Value",
            f"{customer_value:,.2f}"
        )

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🚦 Risk Guide</div>
                <div style="color:#20c98a;font-weight:700;">● Low Risk</div>
                <div class="small-note">Probability below 40%</div>
                <br>
                <div style="color:#f5a623;font-weight:700;">● Medium Risk</div>
                <div class="small-note">Probability from 40% to below 70%</div>
                <br>
                <div style="color:#ef476f;font-weight:700;">● High Risk</div>
                <div class="small-note">Probability 70% or higher</div>
            </div>
            """,
            unsafe_allow_html=True
        )


elif page == "Data Explorer":

    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">DATA EXPLORER</span>
            <div class="main-title">Customer Dataset Explorer</div>
            <div class="subtitle">
                Filter and inspect customer records used by the analytics platform.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    status_filter = st.selectbox(
        "Customer Status",
        ["All", "Retained", "Churned"]
    )

    explorer_df = df.copy()

    if status_filter == "Retained":
        explorer_df = explorer_df[
            explorer_df["churn"] == 0
        ]

    elif status_filter == "Churned":
        explorer_df = explorer_df[
            explorer_df["churn"] == 1
        ]

    st.metric(
        "Records Shown",
        f"{len(explorer_df):,}"
    )

    display_columns = [
        "call_failure",
        "complains",
        "subscription_length",
        "charge_amount",
        "seconds_of_use",
        "frequency_of_use",
        "frequency_of_sms",
        "age_group",
        "tariff_plan",
        "customer_value",
        "churn"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in explorer_df.columns
    ]

    st.dataframe(
        explorer_df[available_columns],
        use_container_width=True,
        height=500
    )


st.markdown(
    """
    <div class="footer">
        Customer Retention Risk Platform • Machine Learning • SQL • Business Analytics
    </div>
    """,
    unsafe_allow_html=True
)
