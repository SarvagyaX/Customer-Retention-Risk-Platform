# Customer Retention Risk Platform

An end-to-end customer churn analytics and retention risk platform built with Python, SQL, Machine Learning, FastAPI, Streamlit, Plotly, and GitHub Actions.

The platform analyzes customer behavior, identifies churn patterns, estimates individual customer churn probability, and presents retention-focused business insights through an interactive dashboard.

## Live Demo

[Open the Customer Retention Risk Platform](PASTE_YOUR_STREAMLIT_URL_HERE)

## Project Overview

Customer churn is an important business problem because identifying customers who may leave can help organizations prioritize retention efforts.

This project builds an end-to-end workflow that transforms raw customer data into:

- Clean and processed datasets
- Engineered customer behavior features
- Exploratory churn analysis
- SQL-based business analysis
- Machine learning churn predictions
- Customer-level risk scoring
- FastAPI prediction endpoint
- Interactive Streamlit dashboard
- Automated model training and validation using GitHub Actions

## Key Features

### Customer Analytics

The dashboard provides:

- Total customer count
- Churned customer count
- Churn rate
- Retention rate
- Customer value associated with churned customers
- Churn analysis by tariff plan
- Churn analysis by age group
- Complaint versus churn analysis
- Customer activity analysis

### Customer Risk Prediction

Users can enter customer information and receive:

- Churn probability
- Low, Medium, or High risk classification
- Visual risk gauge
- Customer value context
- Risk interpretation

The current risk thresholds are:

| Risk Level | Churn Probability |
|---|---:|
| Low | < 40% |
| Medium | 40% - < 70% |
| High | >= 70% |

These thresholds are application-level decision thresholds and are separate from the underlying model probability estimation.

### Interactive Analytics

The dashboard includes interactive filtering and visualizations for:

- Tariff plan
- Age group
- Complaint status
- Customer churn
- Customer value
- Customer activity

### Data Explorer

The Data Explorer allows users to filter customer records by retention status and inspect the underlying customer-level data used by the platform.

## Machine Learning

The project uses a Logistic Regression classification model with feature scaling.

### Machine Learning Workflow

```text
Raw Customer Data
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Train/Test Split
       |
       v
StandardScaler
       |
       v
Logistic Regression
       |
       v
Churn Probability
       |
       v
Customer Risk Level
