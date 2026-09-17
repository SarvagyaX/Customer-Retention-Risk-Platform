import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    drop_columns = [
        "churn",
        "customer_value_group"
    ]

    drop_columns = [
        column for column in drop_columns
        if column in df.columns
    ]

    X = df.drop(columns=drop_columns)
    y = df["churn"]

    return X, y


def load_model():
    return joblib.load(MODEL_PATH)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics, predictions, probabilities


def save_metrics(metrics):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["metric", "value"]
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False
    )


def save_confusion_matrix(y_test, predictions):
    matrix = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.xticks(
        [0, 1],
        ["Retained", "Churned"]
    )

    plt.yticks(
        [0, 1],
        ["Retained", "Churned"]
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close()


def save_roc_curve(y_test, probabilities):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=150
    )

    plt.close()


def main():
    df = load_data()

    X, y = prepare_data(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = load_model()

    metrics, predictions, probabilities = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_metrics(metrics)
    save_confusion_matrix(
        y_test,
        predictions
    )
    save_roc_curve(
        y_test,
        probabilities
    )

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
