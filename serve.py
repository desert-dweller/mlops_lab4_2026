from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from schemas import WineFeatures

app = FastAPI(title="Wine Quality API", description="Predicts if a red wine is premium.")

# Load the registered model from MLflow
# Note: In production, you would point to a specific version or alias (e.g., models:/WinePremiumPredictor/Production)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_URI = "models:/WinePremiumPredictor/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
def predict_quality(features: WineFeatures):
    # Convert the Pydantic model to a dictionary, then to a DataFrame
    input_data = pd.DataFrame([features.dict()])
    input_data.columns = input_data.columns.str.replace('_', ' ')
    
    # Generate prediction
    prediction = model.predict(input_data)
    
    # Return standard JSON response
    is_premium = bool(prediction[0])
    return {
        "is_premium": is_premium,
        "message": "Premium wine detected!" if is_premium else "Standard quality wine."
    }