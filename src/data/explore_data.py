from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    class_counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    class_counts.plot(kind="bar")

    plt.title("Transaction Class Distribution")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()