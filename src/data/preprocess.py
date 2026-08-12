#Preprocessing module 
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco churn data:
      1. Convert TotalCharges to numeric and fill blank (tenure-0) rows with 0.
      2. Drop the customerID identifier column.
      3. Map the Churn target from Yes/No to 1/0.
    Returns a cleaned DataFrame (features still in original categorical form).
    """
    df = df.copy()  # never mutate the caller's DataFrame

    # 1. TotalCharges is stored as text with 11 blanks (all tenure = 0).
    #    'coerce' turns blanks into NaN, then we fill those with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 2. customerID is a unique identifier, not a predictive feature.
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Target must be numeric for the model.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


if __name__ == "__main__":
    from src.data.load_data import load_data

    raw = load_data("data/raw/Telco-Customer-Churn.csv")
    clean = preprocess_data(raw)

    print("Raw shape:      ", raw.shape)
    print("Cleaned shape:  ", clean.shape, "(customerID dropped)")
    print("TotalCharges dtype:", clean["TotalCharges"].dtype)
    print("Churn values:   ", sorted(clean["Churn"].unique()))
    print("Any nulls left? ", clean.isna().sum().sum())