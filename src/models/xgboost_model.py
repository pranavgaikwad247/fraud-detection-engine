from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
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
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Clean column names by removing accidental spaces
    df.columns = df.columns.str.strip()

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ---------------------------------------------------------
    # 60% Training / 20% Validation / 20% Test
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Calculate class imbalance
    # ---------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\n--- Class Imbalance ---")
    print(f"Legitimate transactions: {negative_count}")
    print(f"Fraudulent transactions: {positive_count}")
    print(f"Scale Pos Weight:        {scale_pos_weight:.2f}")

    # ---------------------------------------------------------
    # Create XGBoost model
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Validation predictions
    # ---------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # ---------------------------------------------------------
    # Find best threshold using VALIDATION data only
    # ---------------------------------------------------------

    thresholds = [i / 100 for i in range(1, 100)]

    best_threshold = 0.50
    best_f1 = 0.0

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

    # ---------------------------------------------------------
    # FINAL TEST EVALUATION
    # ---------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

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

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    print("\n--- Test Confusion Matrix ---")
    print(confusion_matrix(y_test, test_predictions))

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------

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