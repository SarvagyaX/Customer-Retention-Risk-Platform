import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "customer_churn_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


def load_data():
    """Load the cleaned customer dataset."""
    return pd.read_csv(INPUT_PATH)


def create_output_directory():
    """Create the EDA output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_summary(df):
    """Generate a basic dataset summary."""

    summary = pd.DataFrame({
        "metric": [
            "total_customers",
            "total_features",
            "churned_customers",
            "retained_customers",
            "churn_rate"
        ],
        "value": [
            len(df),
            len(df.columns),
            int((df["churn"] == 1).sum()),
            int((df["churn"] == 0).sum()),
            round(df["churn"].mean() * 100, 2)
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    return summary


def plot_churn_distribution(df):
    """Create churn distribution chart."""

    counts = df["churn"].value_counts().sort_index()

    labels = ["Retained", "Churned"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)

    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "churn_distribution.png",
        dpi=150
    )
    plt.close()


def plot_churn_by_age_group(df):
    """Analyze churn across age groups."""

    if "age_group" not in df.columns:
        return

    churn_rate = (
        df.groupby("age_group")["churn"]
        .mean()
        .mul(100)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        churn_rate.index.astype(str),
        churn_rate.values
    )

    plt.title("Churn Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "churn_by_age_group.png",
        dpi=150
    )
    plt.close()


def plot_churn_by_tariff(df):
    """Analyze churn across tariff plans."""

    if "tariff_plan" not in df.columns:
        return

    churn_rate = (
        df.groupby("tariff_plan")["churn"]
        .mean()
        .mul(100)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        churn_rate.index.astype(str),
        churn_rate.values
    )

    plt.title("Churn Rate by Tariff Plan")
    plt.xlabel("Tariff Plan")
    plt.ylabel("Churn Rate (%)")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "churn_by_tariff_plan.png",
        dpi=150
    )
    plt.close()


def plot_customer_value(df):
    """Compare customer value between churn groups."""

    if "customer_value" not in df.columns:
        return

    churned = df.loc[df["churn"] == 1, "customer_value"]
    retained = df.loc[df["churn"] == 0, "customer_value"]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        [retained, churned],
        labels=["Retained", "Churned"]
    )

    plt.title("Customer Value by Churn Status")
    plt.xlabel("Customer Status")
    plt.ylabel("Customer Value")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "customer_value_vs_churn.png",
        dpi=150
    )
    plt.close()


def run_eda():
    """Run the complete exploratory analysis."""

    create_output_directory()

    df = load_data()

    print("Dataset shape:", df.shape)

    summary = generate_summary(df)

    print("\nDataset Summary")
    print(summary.to_string(index=False))

    plot_churn_distribution(df)
    plot_churn_by_age_group(df)
    plot_churn_by_tariff(df)
    plot_customer_value(df)

    print("\nEDA completed successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
