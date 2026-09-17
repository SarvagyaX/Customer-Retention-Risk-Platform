import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_processed.csv"


def load_raw_data():
    """Load the original customer dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def clean_column_names(df):
    """Standardize column names for easier analysis."""
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def remove_duplicates(df):
    """Remove duplicate customer records."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df):
    """Handle missing values using simple column-aware rules."""
    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    return df


def clean_data():
    """Run the complete data-cleaning pipeline."""

    df = load_raw_data()

    print(f"Original dataset shape: {df.shape}")

    df = clean_column_names(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    print(f"Cleaned dataset shape: {df.shape}")

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    clean_data()
