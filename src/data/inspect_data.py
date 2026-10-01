from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Class Distribution ---")
    print(df["Class"].value_counts())

    print("\n--- Class Distribution (%) ---")
    print(df["Class"].value_counts(normalize=True) * 100)

    print("\n--- Statistical Summary ---")
    print(df.describe().T)


if __name__ == "__main__":
    main()