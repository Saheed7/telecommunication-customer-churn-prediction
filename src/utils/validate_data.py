# Data Vaalidation Phase
import pandas as pd

# Every raw Telco file should contain exactly these columns
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def validate_data(df: pd.DataFrame) -> bool:
    """
    Run data-quality checks on the raw Telco churn data.
    Returns True only if every check passes. Prints a readable report.
    """
    checks = {}

    # Structure
    checks["all expected columns present"] = set(EXPECTED_COLUMNS).issubset(df.columns)
    checks["dataset is not empty"] = len(df) > 0

    # Identifier integrity
    checks["customerID has no missing values"] = df["customerID"].notna().all()
    checks["customerID is unique"] = df["customerID"].is_unique

    # Categorical values fall within their allowed sets
    checks["gender values valid"] = set(df["gender"].dropna().unique()) <= {"Male", "Female"}
    checks["Churn values valid"] = set(df["Churn"].dropna().unique()) <= {"Yes", "No"}
    checks["SeniorCitizen values valid"] = set(df["SeniorCitizen"].dropna().unique()) <= {0, 1}

    # Numeric sanity
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    checks["tenure is numeric and non-negative"] = tenure.notna().all() and (tenure >= 0).all()

    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    checks["MonthlyCharges is numeric and positive"] = monthly.notna().all() and (monthly > 0).all()

    # Report
    print("\n===== DATA VALIDATION REPORT =====")
    all_passed = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_passed = all_passed and passed
    print("==================================")
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print(f"data_quality_pass = {int(all_passed)}\n")

    return all_passed


if __name__ == "__main__":
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    validate_data(df)