import pandas as pd


def get_retention_insights(df):
    insights = []

    churn_rate = df["churn"].mean() * 100

    if churn_rate >= 20:
        insights.append(
            f"Overall churn rate is {churn_rate:.2f}%."
        )
    else:
        insights.append(
            f"Overall churn rate is {churn_rate:.2f}%."
        )

    if "complains" in df.columns:
        complaint_churn = (
            df.groupby("complains")["churn"]
            .mean()
            .mul(100)
        )

        if len(complaint_churn) > 1:
            highest_group = complaint_churn.idxmax()
            highest_rate = complaint_churn.max()

            insights.append(
                f"Customers in complaint group {highest_group} "
                f"have a churn rate of {highest_rate:.2f}%."
            )

    if "customer_value" in df.columns:
        churned_value = df.loc[
            df["churn"] == 1,
            "customer_value"
        ].sum()

        insights.append(
            f"Customer value associated with churned customers "
            f"is {churned_value:,.2f}."
        )

    if "frequency_of_use" in df.columns:
        activity_churn = (
            df.groupby(
                pd.cut(
                    df["frequency_of_use"],
                    bins=[-1, 20, 50, float("inf")],
                    labels=[
                        "Low Activity",
                        "Medium Activity",
                        "High Activity"
                    ]
                )
            )["churn"]
            .mean()
            .mul(100)
        )

        highest_activity_group = activity_churn.idxmax()
        highest_activity_rate = activity_churn.max()

        insights.append(
            f"{highest_activity_group} customers have the "
            f"highest churn rate at {highest_activity_rate:.2f}%."
        )

    return insights
