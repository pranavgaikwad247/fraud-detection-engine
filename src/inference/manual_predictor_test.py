from src.inference.predictor import FraudPredictor


def main():
    predictor = FraudPredictor()

    transaction = {
        "Time": 406,
        "V1": -1.36,
        "V2": -0.07,
        "V3": 2.54,
        "V4": 1.38,
        "V5": -0.34,
        "V6": 0.46,
        "V7": -0.09,
        "V8": 0.17,
        "V9": 0.26,
        "V10": -0.53,
        "V11": 0.79,
        "V12": -0.76,
        "V13": 0.43,
        "V14": -0.90,
        "V15": 0.17,
        "V16": 0.28,
        "V17": -0.04,
        "V18": -0.18,
        "V19": 0.04,
        "V20": 0.25,
        "V21": -0.02,
        "V22": 0.27,
        "V23": -0.11,
        "V24": 0.06,
        "V25": 0.14,
        "V26": 0.10,
        "V27": 0.02,
        "V28": 0.01,
        "Amount": 10.00,
    }

    result = predictor.predict(transaction)

    print("\n--- Prediction ---")
    print(f"Fraud probability: {result['fraud_probability']}")
    print(f"Prediction:        {result['prediction']}")
    print(f"Decision:          {result['decision']}")
    print(f"Threshold:         {result['threshold']}")


if __name__ == "__main__":
    main()