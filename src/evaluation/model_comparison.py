from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def find_best_threshold(model, X_validation, y_validation):
    """
    Find the probability threshold that gives the
    highest F1 score on the validation set.
    """

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

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

    return best_threshold, best_f1


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Train and evaluate a model using:
    - Training set for training
    - Validation set for threshold selection
    - Test set for final evaluation
    """

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    best_threshold, validation_f1 = find_best_threshold(
        model,
        X_validation,
        y_validation,
    )

    test_probabilities = model.predict_proba(X_test)[:, 1]

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

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation F1:  {validation_f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1:        {f1:.4f}")
    print(f"Test ROC-AUC:   {roc_auc:.4f}")
    print(f"Test PR-AUC:    {pr_auc:.4f}")

    return {
        "Model": name,
        "Threshold": best_threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

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

    results = []

    # =========================================================
    # 1. NAIVE BASELINE
    # =========================================================

    print("\n" + "=" * 60)
    print("1. NAIVE BASELINE")
    print("=" * 60)

    # Predict every transaction as legitimate
    naive_predictions = [0] * len(y_test)
    naive_probabilities = [0.0] * len(y_test)

    naive_precision = precision_score(
        y_test,
        naive_predictions,
        zero_division=0,
    )

    naive_recall = recall_score(
        y_test,
        naive_predictions,
        zero_division=0,
    )

    naive_f1 = f1_score(
        y_test,
        naive_predictions,
        zero_division=0,
    )

    naive_roc_auc = roc_auc_score(
        y_test,
        naive_probabilities,
    )

    naive_pr_auc = average_precision_score(
        y_test,
        naive_probabilities,
    )

    results.append(
        {
            "Model": "Naive Baseline",
            "Threshold": "-",
            "Precision": naive_precision,
            "Recall": naive_recall,
            "F1": naive_f1,
            "ROC-AUC": naive_roc_auc,
            "PR-AUC": naive_pr_auc,
        }
    )

    print("Predicts every transaction as legitimate.")
    print(f"Precision: {naive_precision:.4f}")
    print(f"Recall:    {naive_recall:.4f}")
    print(f"F1:        {naive_f1:.4f}")

    # =========================================================
    # 2. LOGISTIC REGRESSION
    # =========================================================

    print("\n" + "=" * 60)
    print("2. LOGISTIC REGRESSION")
    print("=" * 60)

    logistic_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    # =========================================================
    # 3. RANDOM FOREST
    # =========================================================

    print("\n" + "=" * 60)
    print("3. RANDOM FOREST")
    print("=" * 60)

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    # =========================================================
    # 4. XGBOOST
    # =========================================================

    print("\n" + "=" * 60)
    print("4. XGBOOST")
    print("=" * 60)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    xgboost_model = XGBClassifier(
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

    results.append(
        evaluate_model(
            "XGBoost",
            xgboost_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
        )
    )

    # =========================================================
    # MODEL COMPARISON TABLE
    # =========================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": lambda x: (
                    f"{x:.2f}"
                    if isinstance(x, (float, int))
                    else str(x)
                ),
                "Precision": lambda x: f"{x:.4f}",
                "Recall": lambda x: f"{x:.4f}",
                "F1": lambda x: f"{x:.4f}",
                "ROC-AUC": lambda x: f"{x:.4f}",
                "PR-AUC": lambda x: f"{x:.4f}",
            },
        )
    )

    # =========================================================
    # F1 COMPARISON
    # =========================================================

    model_results = results_df[
        results_df["Model"] != "Naive Baseline"
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        model_results["Model"],
        model_results["F1"],
    )

    plt.title("F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)

    for index, value in enumerate(model_results["F1"]):
        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.show()

    # =========================================================
    # PR-AUC COMPARISON
    # =========================================================

    plt.figure(figsize=(9, 6))

    plt.bar(
        model_results["Model"],
        model_results["PR-AUC"],
    )

    plt.title("PR-AUC Comparison")
    plt.xlabel("Model")
    plt.ylabel("PR-AUC")
    plt.ylim(0, 1)

    for index, value in enumerate(model_results["PR-AUC"]):
        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()