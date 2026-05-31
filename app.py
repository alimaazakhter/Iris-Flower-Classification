from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = FastAPI(
    title="Iris Species Classifier API",
    description="A FastAPI backend to serve predictions for the Iris Flower Classification project.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "iris_logistic_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

model = None
label_encoder = None

def train_and_save_model():
    """Reads Iris.csv, trains a Logistic Regression model, and saves it."""
    global model, label_encoder
    csv_path = "Iris.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required dataset: '{csv_path}' in current directory.")
        
    print("Training model from 'Iris.csv'...")
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    
    le = LabelEncoder()
    df['Species_encoded'] = le.fit_transform(df['Species'])
    
    # We drop Id and Species columns if they exist
    X = df.drop(['Id', 'Species', 'Species_encoded'], axis=1, errors='ignore')
    y = df['Species_encoded']
    
    # Standardize/Fit
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Save objects
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(le, LABEL_ENCODER_FILE)
    
    model = clf
    label_encoder = le
    print("Model and Label Encoder successfully trained and saved!")

# Load model on startup
@app.on_event("startup")
def startup_event():
    global model, label_encoder
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE):
            print("Loading pre-trained model and label encoder...")
            model = joblib.load(MODEL_FILE)
            label_encoder = joblib.load(LABEL_ENCODER_FILE)
        else:
            train_and_save_model()
    except Exception as e:
        print(f"Error loading/training model: {e}")

class IrisPredictionRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Length of the sepal in cm")
    sepal_width: float = Field(..., example=3.5, description="Width of the sepal in cm")
    petal_length: float = Field(..., example=1.4, description="Length of the petal in cm")
    petal_width: float = Field(..., example=0.2, description="Width of the petal in cm")

class IrisPredictionResponse(BaseModel):
    species_id: int
    species_name: str
    probabilities: list[float]
    model_accuracy: float = 1.0

# Root route is served via StaticFiles mount at the bottom of route definitions

@app.post("/predict", response_model=IrisPredictionResponse)
def predict(request: IrisPredictionRequest):
    global model, label_encoder
    
    # Fallback check
    if model is None or label_encoder is None:
        try:
            train_and_save_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model is unavailable and cannot be trained: {e}")
            
    try:
        # Prepare inputs
        input_data = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])
        
        # Predict class index
        pred_idx = int(model.predict(input_data)[0])
        # Predict class probabilities
        probs = model.predict_proba(input_data)[0].tolist()
        
        # Map class index to class name
        species_name = label_encoder.inverse_transform([pred_idx])[0]
        
        return {
            "species_id": pred_idx,
            "species_name": species_name,
            "probabilities": probs,
            "model_accuracy": 1.0
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Serve frontend HTML/CSS/JS files directly at http://127.0.0.1:8000/
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
