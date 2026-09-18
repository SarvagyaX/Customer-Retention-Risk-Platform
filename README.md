# Customer Retention Risk Platform

An end-to-end customer churn analytics and retention risk platform built with Python, SQL, Machine Learning, FastAPI, Streamlit, Plotly, and GitHub Actions.

The platform analyzes customer behavior, identifies churn patterns, estimates individual customer churn probability, and presents retention-focused business insights through an interactive dashboard.

## Live Demo

[https://customer-retention-risk-platform.streamlit.app/](https://customer-retention-risk-platform.streamlit.app/)

## Project Overview

Customer churn is an important business problem because identifying customers who may leave can help organizations prioritize retention efforts.

This project builds an end-to-end workflow that transforms raw customer data into:

* Clean and processed datasets
* Engineered customer behavior features
* Exploratory churn analysis
* SQL-based business analysis
* Machine learning churn predictions
* Customer-level risk scoring
* FastAPI prediction endpoint
* Interactive Streamlit dashboard
* Automated model training and validation using GitHub Actions

## Key Features

### Customer Analytics

The dashboard provides:

* Total customer count
* Churned customer count
* Churn rate
* Retention rate
* Customer value associated with churned customers
* Churn analysis by tariff plan
* Churn analysis by age group
* Complaint versus churn analysis
* Customer activity analysis

### Customer Risk Prediction

Users can enter customer information and receive:

* Churn probability
* Low, Medium, or High risk classification
* Visual risk gauge
* Customer value context
* Risk interpretation

The current application thresholds are:

| Risk Level | Churn Probability |
| ---------- | ----------------: |
| Low        |             < 40% |
| Medium     |       40% - < 70% |
| High       |            >= 70% |

These thresholds are application-level decision thresholds and are separate from the underlying model probability estimation.

### Interactive Analytics

The dashboard includes interactive filtering and visualizations for:

* Tariff plan
* Age group
* Complaint status
* Customer churn
* Customer value
* Customer activity

### Data Explorer

The Data Explorer allows users to filter customer records by retention status and inspect customer-level data used by the platform.

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
```

### Model Pipeline

The model pipeline consists of:

1. StandardScaler
2. Logistic Regression

The dataset is divided into training and testing sets using an 80/20 split with stratification and a fixed random state of 42.

## Feature Engineering

Additional customer behavior features are created from the original variables.

### Engineered Features

* Average usage per month
* Monthly call frequency
* Monthly SMS frequency
* Complaint indicator
* Call failure indicator
* Activity score
* Customer value group

These features provide additional behavioral information for churn analysis and prediction.

## Model Performance

The current Logistic Regression model was evaluated on a held-out test set.

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 90.35% |
| Precision | 82.69% |
| Recall    | 48.31% |
| F1 Score  | 60.99% |
| ROC-AUC   | 92.74% |

The model demonstrates strong overall discrimination based on ROC-AUC, while recall is lower than accuracy. This distinction is important for retention applications because missing potential churners can be more important than overall classification accuracy.

## Business Insights

The dashboard currently identifies patterns such as:

* Overall customer churn rate
* Churn behavior among customers with complaints
* Customer value associated with churned customers
* Higher churn among low-activity customer groups
* Churn differences across tariff plans
* Churn differences across age groups

These insights can be used as starting points for customer retention analysis.

## SQL Analytics

The project includes SQL-based business analysis using SQLite.

The SQL workflow analyzes:

* Overall customer metrics
* Churn by tariff plan
* Churn by age group
* Churn by subscription length
* Churn by complaint status
* Churn by call failures
* Customer value segments
* Customer activity segments
* Customer value associated with churn

The SQLite database is generated automatically from the processed customer dataset.

## API

A FastAPI service provides a prediction endpoint for customer risk scoring.

### Endpoint

```text
POST /predict
```

The API accepts customer information and returns:

```text
Churn Probability
Risk Level
```

The API is tested automatically as part of the GitHub Actions workflow.

## Automated Workflow

GitHub Actions automates the main project pipeline.

The workflow performs:

```text
Data Cleaning
      |
      v
Feature Engineering
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
SQL Analysis
      |
      v
Risk Prediction Test
      |
      v
API Import Test
      |
      v
API Prediction Test
```

Generated datasets, the SQLite database, and the trained model are updated by the workflow.

## Technology Stack

### Programming

* Python
* SQL

### Data Science

* Pandas
* NumPy
* Scikit-learn
* Matplotlib

### Machine Learning

* Logistic Regression
* StandardScaler
* Feature Engineering
* Classification Metrics

### Database

* SQLite
* SQL

### API

* FastAPI
* Uvicorn
* Pydantic

### Dashboard

* Streamlit
* Plotly

### Automation

* GitHub Actions

### Development

* GitHub

## Project Structure

```text
Customer-Retention-Risk-Platform/
│
├── api/
│   ├── main.py
│   └── test_api.py
│
├── dashboard/
│   ├── app.py
│   └── insights.py
│
├── data/
│   ├── customer_churn_raw.csv
│   ├── customer_churn_processed.csv
│   ├── customer_churn_features.csv
│   └── customer_retention.db
│
├── models/
│   └── churn_logistic_regression.joblib
│
├── notebooks/
│
├── reports/
│   ├── eda/
│   └── model_evaluation/
│
├── sql/
│   ├── schema.sql
│   └── business_queries.sql
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda_analysis.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── risk_prediction.py
│   └── sql_analysis.py
│
├── .github/
│   └── workflows/
│       └── train_model.yml
│
├── requirements.txt
└── README.md
```

## Data Pipeline

The data pipeline follows these stages:

### 1. Raw Data

The original customer churn dataset is stored in:

```text
data/customer_churn_raw.csv
```

### 2. Data Cleaning

The cleaning pipeline:

* Standardizes column names
* Removes duplicate records
* Handles missing numeric values
* Handles missing categorical values
* Saves the processed dataset

Output:

```text
data/customer_churn_processed.csv
```

### 3. Feature Engineering

Behavioral and customer-value features are created from the processed dataset.

Output:

```text
data/customer_churn_features.csv
```

### 4. Model Training

The Logistic Regression model is trained using the engineered dataset.

Output:

```text
models/churn_logistic_regression.joblib
```

### 5. SQL Analysis

The processed dataset is loaded into SQLite for business analysis.

Output:

```text
data/customer_retention.db
```

### 6. Dashboard

The Streamlit dashboard consumes the generated dataset and trained model to provide interactive analytics and customer risk prediction.

## Dataset

The project uses the Iranian Customer Churn dataset.

The raw dataset contains 3,150 customer records.

After data cleaning and duplicate removal, 2,850 records are used in the current modeling and dashboard workflow.

The target variable is:

```text
churn
```

where:

```text
0 = Retained
1 = Churned
```

## Running the Project

Install the required Python packages:

```text
pip install -r requirements.txt
```

Run data cleaning:

```text
python src/data_cleaning.py
```

Run feature engineering:

```text
python src/feature_engineering.py
```

Train the model:

```text
python src/model_training.py
```

Evaluate the model:

```text
python src/model_evaluation.py
```

Run SQL analysis:

```text
python src/sql_analysis.py
```

Run risk prediction:

```text
python src/risk_prediction.py
```

Run the Streamlit dashboard:

```text
streamlit run dashboard/app.py
```

Run the FastAPI application:

```text
uvicorn api.main:app --reload
```

## API Example

Example customer input:

```json
{
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
```

Example response structure:

```json
{
    "churn_probability": 0.13,
    "risk_level": "Low"
}
```

The exact prediction depends on the customer input and trained model.

## Project Objective

The main objective of this project is to demonstrate an end-to-end data science workflow that connects:

```text
Data Engineering
       +
Data Analytics
       +
Machine Learning
       +
SQL
       +
API Development
       +
Interactive Visualization
       +
Workflow Automation
```

The project focuses on turning customer data into retention-oriented analytics rather than only building a standalone machine learning model.

## Future Improvements

Potential future improvements include:

* Model comparison using additional classification algorithms
* Hyperparameter tuning
* Explainable AI for individual predictions
* Feature importance analysis
* Customer segmentation
* Retention campaign recommendation
* Model monitoring
* Database-backed production deployment
* Authentication for the prediction API
* Cloud deployment for the API
* Automated model retraining with new customer data

## Disclaimer

This project is intended for educational and portfolio purposes.

The churn predictions represent model estimates based on the available dataset and should not be treated as guaranteed customer behavior.

## Author

**SARVAGYA**

B.Tech Graduate | Data Science | Machine Learning | Data Analytics
