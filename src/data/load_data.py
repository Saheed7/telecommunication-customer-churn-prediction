# src/data/load_data.py
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Read the raw Telco churn CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    # Load the data and print some basic information. 
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    print("Loaded:", df.shape)
    print(df["Churn"].value_counts())