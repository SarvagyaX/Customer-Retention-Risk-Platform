SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    COUNT(*) - SUM(churn) AS retained_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers;


SELECT
    age_group,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY age_group
ORDER BY churn_rate_percent DESC;


SELECT
    tariff_plan,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY tariff_plan
ORDER BY churn_rate_percent DESC;


SELECT
    subscription_length,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY subscription_length
ORDER BY subscription_length;


SELECT
    complains,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY complains
ORDER BY complains DESC;


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
GROUP BY
    CASE
        WHEN customer_value < 50 THEN 'Low Value'
        WHEN customer_value < 100 THEN 'Medium Value'
        WHEN customer_value < 150 THEN 'High Value'
        ELSE 'Very High Value'
    END
ORDER BY average_customer_value DESC;


SELECT
    customer_value,
    subscription_length,
    frequency_of_use,
    frequency_of_sms,
    complains,
    churn
FROM customers
WHERE churn = 1
  AND customer_value >= 100
ORDER BY customer_value DESC;


SELECT
    customer_value,
    complains,
    call_failure,
    subscription_length,
    frequency_of_use,
    churn
FROM customers
WHERE complains > 0
  AND call_failure > 0
ORDER BY customer_value DESC;


SELECT
    CASE
        WHEN frequency_of_use < 20 THEN 'Low Activity'
        WHEN frequency_of_use < 50 THEN 'Medium Activity'
        ELSE 'High Activity'
    END AS activity_segment,
    COUNT(*) AS customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY
    CASE
        WHEN frequency_of_use < 20 THEN 'Low Activity'
        WHEN frequency_of_use < 50 THEN 'Medium Activity'
        ELSE 'High Activity'
    END
ORDER BY churn_rate_percent DESC;


SELECT
    COUNT(*) AS churned_customers,
    ROUND(SUM(customer_value), 2) AS total_customer_value_at_risk,
    ROUND(AVG(customer_value), 2) AS average_value_of_churned_customers
FROM customers
WHERE churn = 1;


SELECT
    status,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY status
ORDER BY churn_rate_percent DESC;


SELECT
    call_failure,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY call_failure
ORDER BY churn_rate_percent DESC;


SELECT
    CASE
        WHEN frequency_of_sms = 0 THEN 'No SMS Activity'
        WHEN frequency_of_sms < 20 THEN 'Low SMS Activity'
        WHEN frequency_of_sms < 50 THEN 'Medium SMS Activity'
        ELSE 'High SMS Activity'
    END AS sms_activity_segment,
    COUNT(*) AS customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY
    CASE
        WHEN frequency_of_sms = 0 THEN 'No SMS Activity'
        WHEN frequency_of_sms < 20 THEN 'Low SMS Activity'
        WHEN frequency_of_sms < 50 THEN 'Medium SMS Activity'
        ELSE 'High SMS Activity'
    END
ORDER BY churn_rate_percent DESC;


SELECT
    customer_value,
    subscription_length,
    frequency_of_use,
    frequency_of_sms,
    complains,
    call_failure,
    churn
FROM customers
WHERE churn = 1
  AND (
      complains > 0
      OR call_failure > 0
  )
ORDER BY customer_value DESC;


SELECT
    CASE
        WHEN customer_value < 50 THEN 'Low Value'
        WHEN customer_value < 100 THEN 'Medium Value'
        WHEN customer_value < 150 THEN 'High Value'
        ELSE 'Very High Value'
    END AS customer_value_segment,
    COUNT(*) AS customers,
    ROUND(SUM(customer_value), 2) AS total_customer_value,
    ROUND(
        SUM(
            CASE
                WHEN churn = 1 THEN customer_value
                ELSE 0
            END
        ),
        2
    ) AS customer_value_at_risk
FROM customers
GROUP BY
    CASE
        WHEN customer_value < 50 THEN 'Low Value'
        WHEN customer_value < 100 THEN 'Medium Value'
        WHEN customer_value < 150 THEN 'High Value'
        ELSE 'Very High Value'
    END
ORDER BY customer_value_at_risk DESC;
