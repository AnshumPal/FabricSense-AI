# FabricSense AI - Hyperspectral Textile Classification

Automated textile classification using hyperspectral imaging and machine learning for recycling and sustainable fashion.

## 🎯 Project Goal

Simplify automated textile sorting in recycling facilities by achieving 95%+ classification accuracy at the pixel level using deep learning and hyperspectral imaging.

### Problem
- Garment tags are often missing or incorrect, making manual sorting unreliable
- Manual sorting is labor-intensive and results in excessive textile waste

### Solution
- Build a deep learning + hyperspectral imaging pipeline to classify textiles automatically
- Initial lab tests show >95% accuracy
- Future integration: Real-time sorting with conveyor belt systems

---

## 📁 Project Structure

```
textileProject/
├── frontend/                    # Next.js web interface
│   ├── app/                    # Next.js app router (pages)
│   ├── components/             # React UI components
│   ├── hooks/                  # Custom React hooks
│   ├── lib/                    # Utility functions
│   ├── public/                 # Static assets
│   ├── styles/                 # Global CSS/Tailwind
│   ├── package.json
│   ├── next.config.mjs
│   ├── tsconfig.json
│   └── .env.local              # Frontend config (API URL)
│
├── backend/
│   └── fabric_backend/         # FastAPI Python backend
│       ├── main.py             # API server entry point
│       ├── requirements.txt    # Python dependencies
│       ├── model/              # Symlink to ../ml-pipeline/model/
│       └── .env                # Backend config (MODEL_PATH, PORT)
│
├── ml-pipeline/                # ML model & training
│   ├── model/                  # Trained RF classifier
│   │   └── textile_classifier_rf.pkl
│   ├── datasets/               # Data directory (ready for training data)
│   ├── notebooks/              # Jupyter exploration notebooks
│   ├── scripts/                # Training & evaluation scripts
│   ├── test_prediction_match.py
│   └── .env                    # ML config
│
├── docs/                       # Documentation
│   ├── README.md               # Project overview
│   ├── DEPLOYMENT_COMPLETE.md
│   ├── FABRICSENSE_AI_AUDIT_REPORT.md
│   └── PREDICTION_FIX_SUMMARY.md
│
├── .vscode/                    # VS Code settings
├── .gitignore
└── README.md                   # This file

```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ & pnpm
- **Python** 3.9+
- **pip** (Python package manager)

### 1️⃣ Frontend Setup

```bash
cd frontend
pnpm install          # Install dependencies
pnpm dev              # Start dev server at http://localhost:3000
```

Configure backend URL in `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 2️⃣ Backend Setup

```bash
cd backend/fabric_backend
pip install -r requirements.txt
python main.py        # Starts at http://localhost:8000
```

Or with uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

Check health: `curl http://localhost:8000/health`

### 3️⃣ ML Pipeline (Optional - for model training)

```bash
cd ml-pipeline
python scripts/train.py              # Train new model (if training scripts exist)
python test_prediction_match.py      # Test predictions
```

---

## 📊 Features

### Frontend (Next.js + React)
- 🎨 Modern UI with Tailwind CSS + Radix UI components
- 📤 CSV file upload with drag-and-drop
- 📈 Real-time spectral data visualization (Recharts)
- 🎯 Fabric classification results with confidence scores
- 🌙 Dark/Light theme support

### Backend (FastAPI)
- ⚡ Fast async API endpoints
- 📊 Hyperspectral data processing (100 spectral bands C113-C212)
- 🤖 Random Forest model inference
- 📝 Automatic CORS handling
- 🏥 Health check endpoints

### ML Pipeline
- 📚 Dataset preparation & preprocessing
- 🏋️ Random Forest classifier training
- ✅ Model validation & testing
- 📦 Joblib serialized model format

---

## 🔌 API Endpoints

### Core Endpoints

#### `POST /predict`
Predict fabric type from hyperspectral CSV data

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@hyperspectral_data.csv"
```

**Response:**
```json
{
  "predicted_fabric": "Cotton",
  "confidence": 0.95,
  "samples_processed": 1
}
```

#### `GET /health`
Check API and model status

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "required_bands": 100,
  "fabric_classes": ["Cotton", "Cotton/Poly blend", "Poly/Spandex"]
}
```

---

## 📋 CSV Input Format

The API expects CSV files with:
- **Header row:** Wavelength identifiers (e.g., `C113,C114,...,C212`)
- **Data rows:** Reflectance values (numeric, 0-1 range)

Example:
```csv
C113,C114,C115,...,C212
0.12,0.15,0.18,...,0.34
0.11,0.14,0.17,...,0.33
```

---

## 🎓 Supported Fabric Types

The model classifies fabrics into these categories:
- **Cotton** - Natural plant fiber (breathable, moisture absorbent)
- **Cotton/Poly Blend** - Hybrid synthetic-natural textile
- **Poly/Spandex** - Synthetic stretch fabric (elastic, shape-retaining)

---

## 🛠 Development Commands

### Frontend
```bash
cd frontend
pnpm dev       # Development server
pnpm build     # Production build
pnpm start     # Start production server
pnpm lint      # Run ESLint
```

### Backend
```bash
cd backend/fabric_backend
python main.py                          # Dev server
uvicorn main:app --reload               # With auto-reload
python -m pip install -r requirements.txt # Install deps
```

### ML Pipeline
```bash
cd ml-pipeline
python scripts/train.py                 # Train model (if available)
python test_prediction_match.py         # Compare terminal & API predictions
```

---

## 🗂 File Organization Tips

### Adding New Frontend Features
1. Create component in `frontend/components/`
2. Import in `frontend/app/page.tsx`
3. Style with Tailwind CSS classes

### Adding New API Endpoints
1. Add function in `backend/fabric_backend/main.py`
2. Use `@app.post()` or `@app.get()` decorator
3. Update frontend to call new endpoint

### Updating ML Model
1. Train new model in `ml-pipeline/scripts/`
2. Save to `ml-pipeline/model/`
3. Update backend `FABRIC_MAPPING` if classes change
4. Test with `ml-pipeline/test_prediction_match.py`

---

## 🔐 Environment Variables

### Frontend (`frontend/.env.local`)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (`backend/fabric_backend/.env`)
```
MODEL_PATH=../../ml-pipeline/model/textile_classifier_rf.pkl
PORT=8000
```

### ML Pipeline (`ml-pipeline/.env`)
```
API_URL=http://localhost:8000
MODEL_PATH=./model/textile_classifier_rf.pkl
```

---

## 📚 Dataset Information

**Source:** DeepTextile Dataset (BSD License)  
**URL:** https://github.com/danikagupta/DeepTextile

**Composition:**
- 15 fabric types across 5 categories
- 224 hyperspectral bands per pixel (SPECIM FX-17 camera)
- Currently using bands C113-C212 (100 bands) for efficiency
- Aggregated pixel data (3×3 grids) for faster processing

---

## ✅ Project Milestones

- [x] Create project structure
- [x] Implement preprocessing pipeline
- [x] Train baseline classifiers (SVM, CNN, Random Forest)
- [x] Optimize model for fabric blends
- [ ] Integrate prototype with conveyor belt hardware
- [ ] Achieve >95% real-time sorting accuracy
- [ ] Deploy to cloud (AWS/Azure)

---

## 🐛 Troubleshooting

### Frontend Can't Connect to Backend
- Ensure backend is running: `python main.py`
- Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local`
- Verify CORS is enabled in `backend/fabric_backend/main.py`
- Check browser console for error details

### Model File Not Found
- Verify path: `ml-pipeline/model/textile_classifier_rf.pkl` exists
- Check `MODEL_PATH` in backend code or `.env` file
- From backend directory: `python -c "from pathlib import Path; print(Path('../../ml-pipeline/model').exists())"`

### CSV Upload Fails
- Ensure CSV has header row with column names (C113-C212)
- Verify all values are numeric
- Check file is not empty
- Use sample data from `ml-pipeline/datasets/test_dataset.csv`

### Port Already in Use
**Frontend:** `pnpm dev` uses port 3000 (can change with `-p` flag)  
**Backend:** Set `PORT` in `.env` or change `--port 8000` argument

---

## 📖 Documentation

Complete documentation available in `docs/`:
- `DEPLOYMENT_COMPLETE.md` - Deployment guide
- `FABRICSENSE_AI_AUDIT_REPORT.md` - Architecture audit
- `PREDICTION_FIX_SUMMARY.md` - Model fix summary

---

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes in appropriate folder (frontend/, backend/, ml-pipeline/)
3. Test locally before committing
4. Push and create a pull request

---

## 📄 License

This project uses the DeepTextile Dataset (BSD License).

---

## 📞 Support

For issues or questions:
1. Check `docs/` folder for detailed guides
2. Review server logs (frontend & backend consoles)
3. Check file paths match the new structure
4. Verify environment variables are set correctly

---

**Last Updated:** March 2026  
**Project Status:** Active Development
