# Project Reorganization - File Updates Required

After running `REORGANIZATION.bat`, you must update these files to fix broken paths:

---

## 1. **backend/fabric_backend/main.py**

**Location:** Line ~29 in the `lifespan()` function

**Current code:**
```python
model_path = Path(__file__).parent / "model" / "textile_classifier_rf.pkl"
```

**Update to:**
```python
model_path = Path(__file__).parent.parent.parent / "ml-pipeline" / "model" / "textile_classifier_rf.pkl"
```

---

## 2. **frontend/app/page.tsx**

**Location:** Line ~36 in the `handlePredict()` function

**Current code:**
```typescript
const response = await fetch("http://localhost:8000/predict", {
```

**Update to (Option A - Environment Variable):**
```typescript
const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const response = await fetch(`${apiUrl}/predict`, {
```

Then create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 3. **ml-pipeline/test_prediction_match.py**

**Location:** Line ~47

**Current code:**
```python
response = requests.post('http://localhost:8000/predict', files=files)
```

**Update to (Option A - Environment Variable):**
```python
import os
api_url = os.getenv('API_URL', 'http://localhost:8000')
response = requests.post(f'{api_url}/predict', files=files)
```

Then create `ml-pipeline/.env`:
```
API_URL=http://localhost:8000
```

---

## 4. **backend/fabric_backend/requirements.txt**

✓ **No changes needed** - However, verify it's in:
```
backend/fabric_backend/requirements.txt
```

---

## 5. **frontend/package.json**

**Location:** `scripts` section

✓ **No changes needed** - Scripts are relative paths that work from frontend/ directory:
```json
"scripts": {
  "dev": "next dev",
  "build": "next build",
  "start": "next start",
  "lint": "eslint ."
}
```

---

## 6. **frontend/next.config.mjs**

✓ **No changes needed** - No backend references

---

## 7. **.gitignore** (at root)

**Add/Update to cover new structure:**
```
# Build outputs
.next/
dist/
build/

# Dependencies
node_modules/
frontend/node_modules/

# Virtual environments
.venv/
backend/.venv/
ml-pipeline/.venv/
venv/

# Environment files
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# Models/Data
ml-pipeline/model/*.pkl
ml-pipeline/datasets/*.csv
```

---

## 8. Create These New Files

### **frontend/.env.local**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **backend/fabric_backend/.env** (optional)
```
MODEL_PATH=../../ml-pipeline/model/textile_classifier_rf.pkl
PORT=8000
```

### **ml-pipeline/.env** (optional)
```
API_URL=http://localhost:8000
MODEL_PATH=./model/textile_classifier_rf.pkl
```

---

## Summary of Changes

| File | Change | Optional? |
|------|--------|-----------|
| backend/fabric_backend/main.py | Update model path (relative) | ❌ REQUIRED |
| frontend/app/page.tsx | Use environment variable for API URL | ✓ Recommended |
| ml-pipeline/test_prediction_match.py | Use environment variable for API URL | ✓ Recommended |
| .gitignore | Update patterns for new structure | ✓ Recommended |
| frontend/.env.local | Create with API_URL | ✓ Optional |
| backend/.env | Create with config | ✓ Optional |
| ml-pipeline/.env | Create with config | ✓ Optional |

---

## Verification Steps

After making these updates, verify everything works:

### **1. Frontend**
```bash
cd frontend
pnpm install
pnpm dev
# Should start at http://localhost:3000
```

### **2. Backend**
```bash
cd backend/fabric_backend
python -m pip install -r requirements.txt
python main.py
# Should start at http://localhost:8000
# Check: http://localhost:8000/health
```

### **3. ML Pipeline**
```bash
cd ml-pipeline
python test_prediction_match.py
# Should successfully load model and make predictions
```

### **4. Check Model Path**
```bash
# From backend/fabric_backend directory:
python -c "from pathlib import Path; p = Path(__file__).parent.parent.parent / 'ml-pipeline' / 'model'; print(f'Model path resolves to: {p}'); print(f'Exists: {p.exists()}')"
```
