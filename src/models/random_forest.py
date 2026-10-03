from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


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
    # 2. Train / validation / test split
    # --------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print("\n--- Dataset Split ---")
    print(f"Training samples:    {len(X_train)}")
    print(f"Validation samples:  {len(X_validation)}")
    print(f"Test samples:        {len(X_test)}")

    print("\n--- Fraud Distribution ---")
    print(f"Training fraud:      {y_train.sum()}")
    print(f"Validation fraud:    {y_validation.sum()}")
    print(f"Test fraud:          {y_test.sum()}")

    # --------------------------------------------------
    # 3. Build Random Forest
    # --------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------
    # 4. Train model
    # --------------------------------------------------
    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. Validation probabilities
    # --------------------------------------------------
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------
    # 6. Find best threshold using validation F1
    # --------------------------------------------------
    thresholds = [i / 100 for i in range(1, 100)]

    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\n--- Validation Threshold Selection ---")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation F1:  {best_f1:.4f}")

    # --------------------------------------------------
    # 7. Test predictions
    # --------------------------------------------------
    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # --------------------------------------------------
    # 8. Final test metrics
    # --------------------------------------------------
    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # --------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------
    print("\n--- Test Confusion Matrix ---")
    print(confusion_matrix(y_test, test_predictions))

    # --------------------------------------------------
    # 10. Classification report
    # --------------------------------------------------
    print("\n--- Test Classification Report ---")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
    