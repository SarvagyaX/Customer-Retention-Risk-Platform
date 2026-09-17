-- ============================================================
-- Customer Retention Risk Platform
-- Business Analytics Queries
-- ============================================================


-- 1. Overall customer and churn summary
SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    COUNT(*) - SUM(churn) AS retained_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers;


-- 2. Churn rate by age group
SELECT
    age_group,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY age_group
ORDER BY churn_rate_percent DESC;


-- 3. Churn rate by tariff plan
SELECT
    tariff_plan,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY tariff_plan
ORDER BY churn_rate_percent DESC;


-- 4. Churn rate by subscription length
SELECT
    subscription_length,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY subscription_length
ORDER BY subscription_length;


-- 5. Customers with complaints
SELECT
    complaints,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY complaints
ORDER BY complaints DESC;


-- 6. Customer value analysis
SELECT
    CASE
        WHEN customer_value < 50 THEN 'Low Value'
        WHEN customer_value < 100 THEN 'Medium Value'
        WHEN customer_value < 150 THEN 'High Value'
        ELSE 'Very High Value'
    END AS customer_value_segment,
    COUNT(*) AS customers,
    ROUND(AVG(customer_value), 2) AS average_customer_value,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY customer_value_segment
ORDER BY average_customer_value DESC;


-- 7. High-value customers who churned
SELECT
    customer_id,
    customer_value,
    subscription_length,
    frequency_of_use,
    frequency_of_sms,
    complaints,
    churn
FROM customers
WHERE churn = 1
  AND customer_value >= 100
ORDER BY customer_value DESC;


-- 8. Customers showing multiple warning signals
SELECT
    customer_id,
    customer_value,
    complaints,
    call_failure,
    subscription_length,
    frequency_of_use,
    churn
FROM customers
WHERE complaints > 0
  AND call_failure > 0
ORDER BY customer_value DESC;


-- 9. Customer activity and churn
SELECT
    CASE
        WHEN frequency_of_use < 20 THEN 'Low Activity'
        WHEN frequency_of_use < 50 THEN 'Medium Activity'
        ELSE 'High Activity'
    END AS activity_segment,
    COUNT(*) AS customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY activity_segment
ORDER BY churn_rate_percent DESC;


-- 10. Customer value at risk
SELECT
    COUNT(*) AS churned_customers,
    ROUND(SUM(customer_value), 2) AS total_customer_value_at_risk,
    ROUND(AVG(customer_value), 2) AS average_value_of_churned_customers
FROM customers
WHERE churn = 1;
