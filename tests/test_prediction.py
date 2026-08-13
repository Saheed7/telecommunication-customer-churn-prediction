# tests/test_prediction.py
from src.serving.inference import predict_churn

SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 140.70,
}


def test_predict_churn_output_shape():
    result = predict_churn(SAMPLE_CUSTOMER)
    assert set(result.keys()) == {"churn_prediction", "churn_probability", "label"}


def test_predict_churn_valid_values():
    result = predict_churn(SAMPLE_CUSTOMER)
    assert result["churn_prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["label"] in ("Will churn", "Will stay")