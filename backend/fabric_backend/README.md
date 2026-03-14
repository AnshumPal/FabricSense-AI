# FabricSense-AI Backend API

Production-ready FastAPI backend for textile fabric classification using hyperspectral data.

## Quick Start

```bash
# 1. Copy model file
copy ..\model\textile_classifier_rf.pkl model\textile_classifier_rf.pkl

# 2. Run setup
setup.bat

# 3. Start server
uvicorn main:app --reload
```

Server runs at: http://localhost:8000

## API Endpoint

### POST /predict

Upload CSV with hyperspectral data (C000-C223), get fabric prediction.

**Response:**
```json
{
  "predicted_fabric": "Cotton/Poly blend",
  "confidence": 0.91,
  "samples_processed": 120
}
```

## Fabric Classes

- 0 → Cotton
- 1 → Cotton/Poly blend
- 2 → Poly/Spandex

## Documentation

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Requirements

- Python 3.8+
- CSV with columns C113-C212 (100 bands)
- Trained model: textile_classifier_rf.pkl
