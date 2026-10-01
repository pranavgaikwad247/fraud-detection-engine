from pathlib import Path

import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ---------------------------------------------------------
    # 2. Split dataset
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Train Random Forest
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Random Forest training complete.")

    # ---------------------------------------------------------
    # 4. Create SHAP explainer
    # ---------------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.TreeExplainer(model)

    print("SHAP explainer created successfully.")

    # ---------------------------------------------------------
    # 5. Find actual fraud transactions
    # ---------------------------------------------------------

    fraud_indices = y_test[y_test == 1].index

    print("\n--- Fraud Transactions in Test Set ---")
    print(f"Number of actual fraud transactions: {len(fraud_indices)}")

    # Select the first actual fraud transaction
    fraud_index = fraud_indices[0]

    transaction = X_test.loc[[fraud_index]]
    actual_class = y_test.loc[fraud_index]

    print("\n--- Selected Fraud Transaction ---")
    print(f"Dataset index: {fraud_index}")
    print(f"Actual class:  {actual_class}")

    # ---------------------------------------------------------
    # 6. Get model prediction
    # ---------------------------------------------------------

    prediction_probability = model.predict_proba(
        transaction
    )[0, 1]

    prediction = int(prediction_probability >= 0.50)

    print(f"Fraud probability: {prediction_probability:.4f}")
    print(f"Model prediction: {prediction}")

    # ---------------------------------------------------------
    # 7. Calculate SHAP values
    # ---------------------------------------------------------

    print("\nCalculating SHAP values...")

    shap_values = explainer.shap_values(transaction)

    print("SHAP values calculated.")

    # ---------------------------------------------------------
    # 8. Extract fraud-class SHAP values
    # ---------------------------------------------------------

    if isinstance(shap_values, list):
        fraud_shap_values = shap_values[1][0]

    elif len(shap_values.shape) == 3:
        fraud_shap_values = shap_values[0, :, 1]

    else:
        fraud_shap_values = shap_values[0]

    # ---------------------------------------------------------
    # 9. Create explanation table
    # ---------------------------------------------------------

    explanation = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "SHAP Value": fraud_shap_values,
        }
    )

    explanation["Absolute Impact"] = (
        explanation["SHAP Value"].abs()
    )

    explanation = explanation.sort_values(
        by="Absolute Impact",
        ascending=False,
    )

    explanation["Direction"] = explanation[
        "SHAP Value"
    ].apply(
        lambda value:
        "Toward Fraud"
        if value > 0
        else "Toward Legitimate"
    )

    # ---------------------------------------------------------
    # 10. Display explanation
    # ---------------------------------------------------------

    print("\n--- Top 10 Feature Contributions ---")

    print(
        explanation[
            [
                "Feature",
                "SHAP Value",
                "Absolute Impact",
                "Direction",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()