import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "customer_churn_processed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "customer_churn_features.csv"


def load_processed_data():
    """Load the cleaned customer dataset."""
    return pd.read_csv(INPUT_PATH)


def create_features(df):
    """Create additional customer behaviour and risk features."""

    df = df.copy()

    # Customer usage intensity
    if "seconds_of_use" in df.columns and "subscription_length" in df.columns:
        df["avg_usage_per_month"] = (
            df["seconds_of_use"] /
            df["subscription_length"].replace(0, 1)
        )

    # Calling activity relative to customer tenure
    if "frequency_of_use" in df.columns and "subscription_length" in df.columns:
        df["monthly_call_frequency"] = (
            df["frequency_of_use"] /
            df["subscription_length"].replace(0, 1)
        )

    # SMS activity relative to customer tenure
    if "frequency_of_sms" in df.columns and "subscription_length" in df.columns:
        df["monthly_sms_frequency"] = (
            df["frequency_of_sms"] /
            df["subscription_length"].replace(0, 1)
        )

    # Complaint indicator
  if "complains" in df.columns:
    df["has_complaint"] = (df["complains"] > 0).astype(int)

    # Call failure indicator
   if "call_failure" in df.columns:
    df["has_call_failures"] = (df["call_failure"] > 0).astype(int)

    # Customer activity score
    activity_columns = [
        column for column in [
            "frequency_of_use",
            "frequency_of_sms",
            "distinct_called_numbers"
        ]
        if column in df.columns
    ]

    if activity_columns:
        df["activity_score"] = df[activity_columns].sum(axis=1)

    # Customer value group
    if "customer_value" in df.columns:
        df["customer_value_group"] = pd.qcut(
            df["customer_value"],
            q=4,
            labels=["Low", "Medium", "High", "Very High"],
            duplicates="drop"
        )

    return df


def save_features(df):
    """Save the feature-engineered dataset."""
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature dataset saved to: {OUTPUT_PATH}")


def main():
    df = load_processed_data()

    print(f"Input shape: {df.shape}")

    df = create_features(df)

    print(f"Feature-engineered shape: {df.shape}")

    save_features(df)


if __name__ == "__main__":
    main()
