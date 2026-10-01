from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # --------------------------------------------------
    # 2. Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\n--- Dataset Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Test fraud cases: {y_test.sum()}")

    # --------------------------------------------------
    # 3. Build Logistic Regression pipeline
    # --------------------------------------------------
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------
    # 4. Train model
    # --------------------------------------------------
    print("\nTraining Logistic Regression...")

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. Get fraud probabilities
    # --------------------------------------------------
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 6. Calculate PR-AUC
    # --------------------------------------------------
    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\n--- Precision-Recall Performance ---")
    print(f"PR-AUC: {pr_auc:.4f}")

    # --------------------------------------------------
    # 7. Threshold analysis
    # --------------------------------------------------
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

    print("\n--- Threshold Analysis ---")
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.3f}"
            f"{recall:<12.3f}"
            f"{f1:<12.3f}"
        )

    # --------------------------------------------------
    # 8. Generate Precision-Recall curve
    # --------------------------------------------------
    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall_values,
        precision_values,
        linewidth=2,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()