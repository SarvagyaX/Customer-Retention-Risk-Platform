-- ============================================================
-- Customer Retention Risk Platform
-- Database Schema
-- Source: UCI Iranian Churn Dataset
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    call_failure INTEGER,
    complains INTEGER,
    subscription_length INTEGER,
    charge_amount INTEGER,
    seconds_of_use INTEGER,
    frequency_of_use INTEGER,
    frequency_of_sms INTEGER,
    distinct_called_numbers INTEGER,
    age_group INTEGER,
    tariff_plan INTEGER,
    status INTEGER,
    churn INTEGER,
    customer_value REAL
);
