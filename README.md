# FabricSense AI

Hyperspectral textile classification system. Upload a CSV of spectral reflectance data and the model predicts the fabric type (Cotton, Cotton/Poly Blend, or Poly-Spandex) using a trained Random Forest classifier.

---

## Project Structure

```
/
├── frontend/          # Next.js app (deploy to Vercel)
└── backend/
    └── fabric_backend/  # FastAPI server (deploy to Railway / Render)
```

---

## Running Locally

### Backend

```bash
cd backend/fabric_backend
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

> The model file `textile_classifier_rf.pkl` must be placed in `backend/fabric_backend/model/`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:3000`.

---

## Environment Variables

### Frontend (`frontend/.env.local`)

| Variable | Description | Example |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend API base URL | `http://localhost:8000` |

### Backend (`backend/fabric_backend/.env`)

| Variable | Description | Example |
|---|---|---|
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:3000,https://your-app.vercel.app` |

---

## Deployment

### Frontend → Vercel

1. Push repo to GitHub
2. Import project in [vercel.com](https://vercel.com)
3. Set root directory to `frontend`
4. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url`

### Backend → Railway / Render

1. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
2. Set working directory to `backend/fabric_backend`
3. Add environment variable: `CORS_ORIGINS=https://your-app.vercel.app`
4. Upload `textile_classifier_rf.pkl` to `model/` directory

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Upload CSV, returns fabric prediction |

### Prediction Response

```json
{
  "predicted_fabric": "Cotton",
  "confidence": 0.87,
  "samples_processed": 1
}
```
