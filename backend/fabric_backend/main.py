from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import io
import os
from typing import Dict

# Fabric class mapping
FABRIC_MAPPING = {
    0: "Cotton",
    1: "Cotton/Poly blend",
    2: "Poly/Spandex"
}

# Required spectral bands (C113 to C212 = 100 bands)
REQUIRED_BANDS = [f"C{str(i).zfill(3)}" for i in range(113, 213)]

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once at startup for performance"""
    global model
    model_path = Path(__file__).parent / "model" / "textile_classifier_rf.pkl"
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model file not found at {model_path}")
        raise RuntimeError("Model file not found. Please ensure textile_classifier_rf.pkl exists in the model directory.")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    yield

app = FastAPI(title="FabricSense-AI API", version="1.0.0", lifespan=lifespan)

# CORS — reads allowed origins from env var for production flexibility
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FabricSense-AI API",
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Predict fabric type from hyperspectral CSV data
    
    Args:
        file: CSV file containing hyperspectral data (C000-C223)
    
    Returns:
        JSON with predicted_fabric, confidence, and samples_processed
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read CSV file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Extract required bands (C113 to C212)
        missing_bands = [band for band in REQUIRED_BANDS if band not in df.columns]
        if missing_bands:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required spectral bands: {', '.join(missing_bands[:5])}{'...' if len(missing_bands) > 5 else ''}"
            )
        
        # Select only the required 100 bands
        X = df[REQUIRED_BANDS].copy()
        
        # DEBUG LOGS
        print(f"📊 CSV columns: {len(df.columns)} total")
        print(f"📊 Selected bands: C113 to C212 ({len(REQUIRED_BANDS)} bands)")
        print(f"📊 Input shape: {X.shape}")
        
        # Convert to numeric and handle any non-numeric values
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Check for NaN values after conversion
        if X.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains non-numeric values in spectral bands"
            )
        
        # Perform prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # DEBUG LOGS
        print(f"🔮 Raw predictions: {predictions}")
        print(f"🔮 Prediction probabilities shape: {probabilities.shape}")
        
        # Get the most common prediction and its average confidence
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        most_common_prediction = unique_predictions[np.argmax(counts)]
        
        # Calculate average confidence for the most common prediction
        prediction_indices = predictions == most_common_prediction
        avg_confidence = probabilities[prediction_indices, most_common_prediction].mean()
        
        # Map numeric prediction to fabric name
        predicted_fabric = FABRIC_MAPPING.get(int(most_common_prediction), "Unknown")
        
        # DEBUG LOGS
        print(f"✅ Final prediction: {predicted_fabric} (class {most_common_prediction})")
        print(f"✅ Confidence: {avg_confidence:.2%}")
        print(f"✅ Samples processed: {len(df)}")
        
        return {
            "predicted_fabric": predicted_fabric,
            "confidence": round(float(avg_confidence), 2),
            "samples_processed": len(df)
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check including model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "required_bands": len(REQUIRED_BANDS),
        "fabric_classes": list(FABRIC_MAPPING.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
