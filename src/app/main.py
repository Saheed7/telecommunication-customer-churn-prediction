## src/app/main.py
from fastapi import FastAPI

# This 'app' object IS the  API. uvicorn will look for it by name.
app = FastAPI(title="Telco Churn Prediction API")


@app.get("/")
def health_check():
    """A simple endpoint to confirm the API is alive."""
    return {"status": "ok", "message": "Churn prediction API is running"}



# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.inference import predict_churn

app = FastAPI(title="Telco Churn Prediction API")


# Defines the shape of a valid prediction request
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        return predict_churn(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# src/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr                                    # NEW

from src.serving.inference import predict_churn
from src.app.gradio_ui import build_gradio_ui          # NEW

app = FastAPI(title="Telco Churn Prediction API")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        return predict_churn(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW: mount the Gradio form onto the same app at /ui
app = gr.mount_gradio_app(app, build_gradio_ui(), path="/ui")