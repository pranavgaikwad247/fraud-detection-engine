from pathlib import Path

import joblib
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_random_forest.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "fraud_threshold.joblib"


FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


class FraudPredictor:
    """Loads the fraud model and performs predictions and explanations."""

    def __init__(self):
        print("Loading fraud detection model...")

        self.model = joblib.load(MODEL_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)

        self.explainer = shap.TreeExplainer(self.model)

        print("Model loaded successfully.")
        print(f"Fraud threshold: {self.threshold}")

    def _prepare_transaction(self, transaction: dict):
        """Validate and convert transaction into model input."""

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in transaction
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        return pd.DataFrame(
            [[transaction[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

    def predict(self, transaction: dict) -> dict:
        """Predict whether a transaction is potentially fraudulent."""

        data = self._prepare_transaction(transaction)

        fraud_probability = self.model.predict_proba(data)[0][1]

        prediction = int(
            fraud_probability >= self.threshold
        )

        if prediction == 1:
            decision = "POTENTIAL_FRAUD"
        else:
            decision = "LEGITIMATE"

        return {
            "fraud_probability": round(
                float(fraud_probability),
                4,
            ),
            "prediction": prediction,
            "decision": decision,
            "threshold": self.threshold,
        }

    def explain(self, transaction: dict) -> list:
        """Return the strongest SHAP factors for the transaction."""

        data = self._prepare_transaction(transaction)

        shap_values = self.explainer.shap_values(data)

        # Random Forest binary classification:
        # SHAP may return either a list or a 3D ndarray
        if isinstance(shap_values, list):

            fraud_shap_values = shap_values[1][0]

        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:

            fraud_shap_values = shap_values[0, :, 1]

        else:

            fraud_shap_values = shap_values[0]

        explanation = pd.DataFrame({
            "feature": FEATURES,
            "shap_value": fraud_shap_values,
        })

        explanation["absolute_impact"] = (
            explanation["shap_value"].abs()
        )

        explanation["direction"] = explanation[
            "shap_value"
        ].apply(
            lambda value:
            "Toward Fraud"
            if value > 0
            else "Toward Legitimate"
        )

        explanation = explanation.sort_values(
            "absolute_impact",
            ascending=False,
        ).head(10)

        return [
            {
                "feature": row["feature"],
                "shap_value": round(
                    float(row["shap_value"]),
                    6,
                ),
                "absolute_impact": round(
                    float(row["absolute_impact"]),
                    6,
                ),
                "direction": row["direction"],
            }
            for _, row in explanation.iterrows()
        ]