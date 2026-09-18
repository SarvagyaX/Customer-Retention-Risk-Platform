import sqlite3
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_processed.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "customer_retention.db"


def create_database():
    df = pd.read_csv(DATA_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    df.to_sql(
        "customers",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()

    print(f"Database created at: {DATABASE_PATH}")


def run_analysis():
    connection = sqlite3.connect(DATABASE_PATH)

    queries = {
        "overall_metrics": """
            SELECT
                COUNT(*) AS total_customers,
                SUM(churn) AS churned_customers,
                COUNT(*) - SUM(churn) AS retained_customers,
                ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
            FROM customers
        """,
        "churn_by_tariff": """
            SELECT
                tariff_plan,
                COUNT(*) AS total_customers,
                SUM(churn) AS churned_customers,
                ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
            FROM customers
            GROUP BY tariff_plan
            ORDER BY churn_rate_percent DESC
        """,
        "churn_by_age": """
            SELECT
                age_group,
                COUNT(*) AS total_customers,
                SUM(churn) AS churned_customers,
                ROUND(AVG(churn) * 100, 2) AS churn_rate_percent
            FROM customers
            GROUP BY age_group
            ORDER BY churn_rate_percent DESC
        """,
        "value_at_risk": """
            SELECT
                COUNT(*) AS churned_customers,
                ROUND(SUM(customer_value), 2) AS total_customer_value_at_risk,
                ROUND(AVG(customer_value), 2) AS average_value_of_churned_customers
            FROM customers
            WHERE churn = 1
        """
    }

    for name, query in queries.items():
        print(f"\n{name}")

        result = pd.read_sql_query(
            query,
            connection
        )

        print(result.to_string(index=False))

    connection.close()


def main():
    create_database()
    run_analysis()


if __name__ == "__main__":
    main()
