from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ----------------------------
# Define the request model
# ----------------------------
class CustomerFeatures(BaseModel):
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

# ----------------------------
# Initialize FastAPI app
# ----------------------------
app = FastAPI(title="Telco Churn API", version="0.1.0")

# ----------------------------
# Load trained model and encoders
# ----------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(features: CustomerFeatures):
    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([features.dict()])

    # Apply saved encoders
    for col, le in encoders.items():
        if col in df:
            df[col] = le.transform(df[col])

    # Predict
    prediction = model.predict(df)[0]

    # Return result
    return {"churn": bool(prediction)}

# ----------------------------
# Optional: Run with uvicorn for local testing
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
