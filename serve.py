from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from schemas import WineFeatures
from dotenv import load_dotenv

app = FastAPI(title="Wine Quality API", description="Predicts if a red wine is premium.")

# 1. Load your DagsHub credentials from the .env file
load_dotenv()

# 2. Load the model directly from the DagsHub registry! 
# (Make sure you actually registered the model in the DagsHub UI first)
MODEL_URI = "models:/WinePremiumPredictor/1" 
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
def predict_quality(features: WineFeatures):
    input_data = pd.DataFrame([features.dict()])
    input_data.columns = input_data.columns.str.replace('_', ' ')
    
    prediction = model.predict(input_data)
    is_premium = bool(prediction[0])
    
    return {
        "is_premium": is_premium,
        "message": "Premium wine detected!" if is_premium else "Standard quality wine."
    }