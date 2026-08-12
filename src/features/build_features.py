# Build Features Stage
import pandas as pd

# Two-value columns → single 0/1 column via a FIXED mapping.
# Hardcoding this guarantees the same encoding at training and serving time.
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Three-or-more-value columns → one-hot encoded.
MULTICATEGORY_COLUMNS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns into numeric form:
      - binary columns  -> 0/1 via BINARY_MAP (deterministic)
      - multi-category  -> one-hot encoded with drop_first=True
    Returns a fully numeric DataFrame (features + Churn target).
    """
    df = df.copy()

    # 1. Binary columns -> deterministic 0/1
    for col in BINARY_COLUMNS:
        df[col] = df[col].map(BINARY_MAP)

    # 2. Multi-category columns -> one-hot (drop_first avoids redundant columns)
    df = pd.get_dummies(df, columns=MULTICATEGORY_COLUMNS, drop_first=True)

    # 3. get_dummies produces True/False columns; convert them to 1/0 ints
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


if __name__ == "__main__":
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data

    raw = load_data("data/raw/Telco-Customer-Churn.csv")
    clean = preprocess_data(raw)
    features = build_features(clean)

    print("Cleaned shape: ", clean.shape)
    print("Encoded shape: ", features.shape, "(30 features + Churn target)")
    print("All numeric?   ", features.select_dtypes(exclude="number").empty)
    print("Sample of new columns:", [c for c in features.columns if "Contract" in c])