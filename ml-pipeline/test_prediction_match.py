"""
Test script to verify terminal and API predictions match
"""
import pandas as pd
import joblib
import requests
import sys
from pathlib import Path

# Load the trained model
model_path = Path(__file__).parent / "model" / "textile_classifier_rf.pkl"
model = joblib.load(model_path)

# Map fab_id to fabric type
fab_map = {0: 'Cotton', 1: 'Cotton/Poly blend', 2: 'Poly/Spandex'}

# Load test data
test_file = Path(__file__).parent / "model" / "test_dataset1.csv"
new_sample = pd.read_csv(test_file)

# Select top 100 bands (same as training)
top_bands = [f'C{i:03d}' for i in range(113, 213)]
X_new = new_sample[top_bands]

# TERMINAL PREDICTION
print("=" * 60)
print("TERMINAL PREDICTION (Direct Model)")
print("=" * 60)
preds = model.predict(X_new)
probs = model.predict_proba(X_new)
terminal_prediction = fab_map[preds[0]]
terminal_confidence = probs[0][preds[0]]

print(f"Predicted Fabric: {terminal_prediction}")
print(f"Confidence: {terminal_confidence:.2%}")
print(f"Raw class: {preds[0]}")
print()

# API PREDICTION
print("=" * 60)
print("API PREDICTION (Backend Server)")
print("=" * 60)

try:
    with open(test_file, 'rb') as f:
        files = {'file': (str(test_file.name), f, 'text/csv')}
        response = requests.post('http://localhost:8000/predict', files=files)
    
    if response.status_code == 200:
        api_result = response.json()
        api_prediction = api_result['predicted_fabric']
        api_confidence = api_result['confidence']
        
        print(f"Predicted Fabric: {api_prediction}")
        print(f"Confidence: {api_confidence:.2%}")
        print(f"Samples Processed: {api_result['samples_processed']}")
        print()
        
        # COMPARISON
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        
        match = terminal_prediction == api_prediction
        confidence_diff = abs(terminal_confidence - api_confidence)
        
        print(f"Predictions Match: {'✅ YES' if match else '❌ NO'}")
        print(f"Terminal: {terminal_prediction} ({terminal_confidence:.2%})")
        print(f"API:      {api_prediction} ({api_confidence:.2%})")
        print(f"Confidence Difference: {confidence_diff:.4f}")
        
        if match and confidence_diff < 0.01:
            print("\n✅ SUCCESS: Terminal and API predictions are identical!")
            sys.exit(0)
        else:
            print("\n❌ MISMATCH: Predictions differ!")
            sys.exit(1)
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.json())
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to backend server at http://localhost:8000")
    print("Make sure the backend is running: py -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    sys.exit(1)
