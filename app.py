
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model and preprocessing objects
model = joblib.load("logistic_regression_model.pkl")
encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Customer Churn Prediction API")


class CustomerData(BaseModel):
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    tenure_months: float
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}
    

@app.post("/predict")
def predict_churn(customer: CustomerData):

    categorical_data = pd.DataFrame([{
        "Gender": customer.gender,
        "Senior Citizen": customer.senior_citizen,
        "Partner": customer.partner,
        "Dependents": customer.dependents,
        "Phone Service": customer.phone_service,
        "Multiple Lines": customer.multiple_lines,
        "Internet Service": customer.internet_service,
        "Online Security": customer.online_security,
        "Online Backup": customer.online_backup,
        "Device Protection": customer.device_protection,
        "Tech Support": customer.tech_support,
        "Streaming TV": customer.streaming_tv,
        "Streaming Movies": customer.streaming_movies,
        "Contract": customer.contract,
        "Paperless Billing": customer.paperless_billing,
        "Payment Method": customer.payment_method
    }])

    numerical_data = pd.DataFrame([{
        "Tenure Months": customer.tenure_months,
        "Monthly Charges": customer.monthly_charges,
        "Total Charges": customer.total_charges
    }])

    numerical_scaled = scaler.transform(numerical_data)
    encoded = encoder.transform(categorical_data)

    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    features = np.hstack([numerical_scaled, encoded])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    churn_index = list(model.classes_).index(1)
    churn_probability = float(probabilities[churn_index])

    return {
        "prediction": prediction,
        "result": (
            "Likely to churn" if prediction == 1
            else "Unlikely to churn"
        ),
        "churn_probability_percent": round(
            churn_probability * 100, 2
        )
    }
