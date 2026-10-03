from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "fraud_random_forest.joblib"
THRESHOLD_PATH = MODEL_DIR / "fraud_threshold.joblib"


def find_best_threshold(model, X_validation, y_validation):
    """Find the threshold that maximizes F1 on validation data."""

    probabilities = model.predict_proba(X_validation)[:, 1]

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in [i / 100 for i in range(1, 100)]:
        predictions = (probabilities >= threshold).astype(int)

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


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

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------

    print("\nTraining final Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    # ---------------------------------------------------------
    # Find threshold using validation data
    # ---------------------------------------------------------

    print("\nFinding optimal threshold...")

    threshold, validation_f1 = find_best_threshold(
        model,
        X_validation,
        y_validation,
    )

    print(f"Best threshold: {threshold:.2f}")
    print(f"Validation F1:  {validation_f1:.4f}")

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

    print("\n--- Model Saved ---")
    print(f"Model:     {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()