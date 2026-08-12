# src/app/gradio_ui.py
import gradio as gr

from src.serving.inference import predict_churn


def _predict(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
             MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
             DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
             Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    # Pack the form values into the same dict predict_churn already expects
    raw = {
        "gender": gender, "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner, "Dependents": Dependents, "tenure": int(tenure),
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract,
        "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges), "TotalCharges": float(TotalCharges),
    }
    r = predict_churn(raw)
    return f"{r['label']}  —  churn probability {r['churn_probability'] * 100:.1f}%"


def build_gradio_ui():
    yesno = ["Yes", "No"]
    net3 = ["No", "Yes", "No internet service"]
    # NOTE: this list order MUST match _predict's argument order above
    inputs = [
        gr.Dropdown(["Female", "Male"], label="gender", value="Female"),
        gr.Radio([0, 1], label="SeniorCitizen", value=0),
        gr.Dropdown(yesno, label="Partner", value="No"),
        gr.Dropdown(yesno, label="Dependents", value="No"),
        gr.Slider(0, 72, step=1, label="tenure (months)", value=12),
        gr.Dropdown(yesno, label="PhoneService", value="Yes"),
        gr.Dropdown(["No", "Yes", "No phone service"], label="MultipleLines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService", value="Fiber optic"),
        gr.Dropdown(net3, label="OnlineSecurity", value="No"),
        gr.Dropdown(net3, label="OnlineBackup", value="No"),
        gr.Dropdown(net3, label="DeviceProtection", value="No"),
        gr.Dropdown(net3, label="TechSupport", value="No"),
        gr.Dropdown(net3, label="StreamingTV", value="No"),
        gr.Dropdown(net3, label="StreamingMovies", value="No"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(yesno, label="PaperlessBilling", value="Yes"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    label="PaymentMethod", value="Electronic check"),
        gr.Number(label="MonthlyCharges", value=70.35),
        gr.Number(label="TotalCharges", value=140.70),
    ]
    return gr.Interface(
        fn=_predict,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction"),
        title="Telco Customer Churn Predictor",
        description="Enter a customer's details to get a churn prediction.",
    )