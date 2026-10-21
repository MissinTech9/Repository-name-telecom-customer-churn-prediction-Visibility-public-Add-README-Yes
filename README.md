# Repository-name-telecom-customer-churn-prediction-Visibility-public-Add-README-Yes
# Telecom Customer Churn Prediction

## About the Project

I built this project to predict telecom customers who may stop using a company's services and estimate their probability of leaving.

The idea is to help telecom companies identify customers who may leave so they can understand the situation and find ways to keep them.

## What the Project Does

- Uses customer details such as contract type, length of time with the company, internet service, and monthly charges.
- Predicts whether a customer is likely to leave or stay.
- Estimates the customer's probability of leaving.
- Provides predictions through a FastAPI application.

## Tools I Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Jupyter Notebook

## Sample Prediction

I tested the API with sample customer details, and this was the result:

- **Prediction:** Likely to churn
- **Probability of leaving:** 82.95%

This means the model estimated an 82.95% probability that the sample customer would leave. It is a prediction, not a guarantee.

## How to Run the Project

1. Download or clone this repository.
2. Install the required packages:

   `pip install -r requirements.txt`

3. Start the API:

   `python -m uvicorn app:app --reload`

4. Open `http://127.0.0.1:8000/docs` in your browser.
5. Use the `/predict` endpoint to test the model with customer details.

## Project Status

The API is running and has been tested locally. Online deployment is the next step.

## Author

Faith Egbe
