from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Select a known fraud transaction
    fraud_transaction = df[df["Class"] == 1].iloc[0]

    actual_class = int(fraud_transaction["Class"])

    transaction = {
        feature: float(fraud_transaction[feature])
        for feature in FEATURES
    }

    print("\n--- Dataset Transaction ---")
    print(f"Actual class: {actual_class}")
    print(f"Amount:       {transaction['Amount']}")

    print("\nSending transaction to API...")

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=transaction,
        timeout=10,
    )

    print(f"HTTP status: {response.status_code}")

    print("\n--- API Response ---")
    print(response.json())


if __name__ == "__main__":
    main()