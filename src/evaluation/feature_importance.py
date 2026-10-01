from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Clean column names by removing accidental spaces
    df.columns = df.columns.str.strip()

    print("\n--- Columns Found ---")
    print(df.columns.tolist())

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split into training, validation, and test sets
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

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Get feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    )

    # Sort features from most important to least important
    importance = importance.sort_values(ascending=False)

    print("\n--- Feature Importance ---")
    print(importance.to_string())

    print("\n--- Top 10 Features ---")
    print(importance.head(10))

    # Plot top 15 features
    plt.figure(figsize=(10, 6))

    importance.head(15).sort_values().plot(
        kind="barh"
    )

    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()