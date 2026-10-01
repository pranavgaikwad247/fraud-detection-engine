from pathlib import Path

import joblib
import pandas as pd


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
    """Loads the trained fraud detection model and makes predictions."""

    def __init__(self):
        print("Loading fraud detection model...")

        self.model = joblib.load(MODEL_PATH)
        self.threshold = joblib.load(THRESHOLD_PATH)

        print("Model loaded successfully.")
        print(f"Fraud threshold: {self.threshold}")

    def predict(self, transaction: dict) -> dict:
        """Predict whether a transaction is potentially fraudulent."""

        # Check that all required features exist
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in transaction
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Keep features in the exact order used during training
        data = pd.DataFrame(
            [[transaction[feature] for feature in FEATURES]],
            columns=FEATURES,
        )

        # Get fraud probability
        fraud_probability = self.model.predict_proba(data)[0][1]

        # Apply our saved threshold
        prediction = int(fraud_probability >= self.threshold)

        if prediction == 1:
            decision = "POTENTIAL_FRAUD"
        else:
            decision = "LEGITIMATE"

        return {
            "fraud_probability": round(float(fraud_probability), 4),
            "prediction": prediction,
            "decision": decision,
            "threshold": self.threshold,
        }