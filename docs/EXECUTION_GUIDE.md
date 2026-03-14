# Project Reorganization - Complete Execution Guide

## 📋 Overview

This guide walks you through reorganizing your FabricSense-AI project from a flat structure into a clean, modular architecture with separate `frontend/`, `backend/`, `ml-pipeline/`, and `docs/` folders.

**Estimated time:** 15-20 minutes  
**Risk level:** Low (all changes are reversible via git)

---

## 🎯 What You'll Get

```
BEFORE (Flat Structure):
textileProject/
├── app/
├── components/
├── fabric_backend/
├── model/
├── various .tsx, .json, .py files
└── Mixed markdown docs

AFTER (Organized Structure):
textileProject/
├── frontend/           (All Next.js code)
├── backend/            (FastAPI server)
├── ml-pipeline/        (Models & training)
├── docs/               (Documentation)
├── .vscode/, .gitignore, README.md
```

---

## ⚙️ Step-by-Step Execution

### **Step 1: Backup Your Project (Recommended)**

```bash
# Option A: Git commit everything first
git add -A
git commit -m "Pre-reorganization backup"

# Option B: Create a zip backup
# Right-click textileProject → Send to → Compressed folder
```

### **Step 2: Run the Reorganization Batch Script**

```bash
# Open Command Prompt in your project root
# cd c:\Users\ASUS\OneDrive\Desktop\textileProject

# Run the reorganization script
REORGANIZATION.bat

# Follow the prompts (press any key when complete)
```

✅ **What this does:**
- Creates new directories: `frontend/`, `backend/`, `ml-pipeline/`, `docs/`
- Moves all Next.js files to `frontend/`
- Moves Python backend to `backend/`
- Moves ML model and test scripts to `ml-pipeline/`
- Moves markdown docs to `docs/`

---

### **Step 3: Update File Paths**

Follow `FILE_UPDATES_REQUIRED.md` to fix these files:

#### **A. Update Backend Model Path**

**File:** `backend/fabric_backend/main.py`  
**Line:** ~29 in `lifespan()` function

```python
# BEFORE:
model_path = Path(__file__).parent / "model" / "textile_classifier_rf.pkl"

# AFTER:
model_path = Path(__file__).parent.parent.parent / "ml-pipeline" / "model" / "textile_classifier_rf.pkl"
```

#### **B. Externalize Frontend API URL**

**File:** `frontend/app/page.tsx`  
**Line:** ~36 in `handlePredict()` function

```typescript
// BEFORE:
const response = await fetch("http://localhost:8000/predict", {

// AFTER:
const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const response = await fetch(`${apiUrl}/predict`, {
```

Then create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### **C. Update ML Test Script (Optional)**

**File:** `ml-pipeline/test_prediction_match.py`  
**Line:** ~47

```python
# BEFORE:
response = requests.post('http://localhost:8000/predict', files=files)

# AFTER:
import os
api_url = os.getenv('API_URL', 'http://localhost:8000')
response = requests.post(f'{api_url}/predict', files=files)
```

---

### **Step 4: Create Configuration Files**

#### **Create: `frontend/.env.local`**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### **Create: `backend/fabric_backend/.env`** (Optional)
```
MODEL_PATH=../../ml-pipeline/model/textile_classifier_rf.pkl
PORT=8000
```

#### **Create: `ml-pipeline/.env`** (Optional)
```
API_URL=http://localhost:8000
MODEL_PATH=./model/textile_classifier_rf.pkl
```

---

### **Step 5: Replace Root README**

```bash
# Delete old README
del README.md

# Rename new README
rename README_NEW.md README.md
```

Or manually copy content from `README_NEW.md` to `README.md`

---

### **Step 6: Update/Create .gitignore**

Replace `.gitignore` content with patterns from `FILE_UPDATES_REQUIRED.md` to properly ignore:
- Build files (`.next/`, `dist/`)
- Dependencies (`node_modules/`, `.venv/`)
- Environment files (`.env`, `.env.local`)
- Cache directories (`__pycache__/`, `.pytest_cache/`)

---

### **Step 7: Verify the Structure**

Open file explorer and verify:
```
textileProject/
├── frontend/
│   ├── app/              ✓ Exists
│   ├── components/       ✓ Exists
│   ├── hooks/            ✓ Exists
│   ├── package.json      ✓ Exists
│   ├── next.config.mjs   ✓ Exists
│   └── .env.local        ✓ Created
├── backend/
│   └── fabric_backend/
│       ├── main.py       ✓ Updated
│       └── requirements.txt
├── ml-pipeline/
│   ├── model/
│   │   └── textile_classifier_rf.pkl
│   └── test_prediction_match.py
├── docs/
│   ├── README.md
│   ├── DEPLOYMENT_COMPLETE.md
│   └── ... other docs
└── README.md             ✓ New
```

---

## 🧪 Testing Phase

### **Test 1: Frontend Starts**

```bash
cd frontend
pnpm install              # Install dependencies (first time)
pnpm dev                  # Start dev server

# Expected: 
# ✓ Compiled successfully
# ✓ Running at http://localhost:3000
# ✓ No path errors in terminal
```

### **Test 2: Backend Starts**

```bash
# In a new terminal window
cd backend\fabric_backend
pip install -r requirements.txt   # First time only
python main.py

# Expected:
# ✓ Model loaded successfully
# ✓ Uvicorn running on http://localhost:8000
# ✓ No model path errors
```

Or with uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

### **Test 3: API Health Check**

```bash
# In yet another terminal
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "required_bands": 100,
#   "fabric_classes": ["Cotton", "Cotton/Poly blend", "Poly/Spandex"]
# }
```

### **Test 4: Frontend → Backend Connection**

1. Go to `http://localhost:3000` in browser
2. Upload a test CSV from `ml-pipeline/` (if available)
3. Click "Predict"
4. Should show results (not error messages)

### **Test 5: Model Path Resolution**

```bash
# From backend\fabric_backend directory
python -c "from pathlib import Path; p = Path(__file__).parent.parent.parent / 'ml-pipeline' / 'model' / 'textile_classifier_rf.pkl'; print(f'Model exists: {p.exists()}'); print(f'Path: {p}')"

# Expected: True and full path to model file
```

---

## ✅ Cleanup Steps

### **Delete Auto-Generated Files** (Safe)

```batch
@echo off
REM This saves ~500-800 MB of disk space

rmdir /s /q .next 2>nul
rmdir /s /q frontend\.next 2>nul
rmdir /s /q node_modules 2>nul
rmdir /s /q frontend\node_modules 2>nul

echo Auto-generated files deleted!
```

### **Delete Optional Docs** (If Not Needed)

```batch
del "docs\DEPLOYMENT_COMPLETE.md" 2>nul
del "docs\FABRICSENSE_AI_AUDIT_REPORT.md" 2>nul
del "docs\PREDICTION_FIX_SUMMARY.md" 2>nul
del "docs\Readme.txt" 2>nul

echo Optional documentation removed!
```

---

## 📝 Git Integration

After successful testing, commit the reorganization:

```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "Reorganize project structure: separate frontend, backend, ml-pipeline, and docs"

# Push (if using remote)
git push origin main
```

### What to Ignore (Already in .gitignore)
```
.next/
node_modules/
.venv/
__pycache__/
.env
.env.local
```

---

## 🆘 Troubleshooting

### Issue: Frontend Can't Connect to Backend

**Symptom:** "Failed to fetch" or "Connection refused" error

**Solution:**
1. Verify backend is running: `python main.py`
2. Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local`
3. Clear browser cache: Ctrl+Shift+Delete
4. Check browser console (F12) for exact error

### Issue: Model File Not Found

**Symptom:** "Model file not found" in backend console

**Solution:**
1. Run verification test:
   ```bash
   cd backend\fabric_backend
   python -c "from pathlib import Path; print(Path(__file__).parent.parent.parent / 'ml-pipeline' / 'model')"
   ```
2. Verify path matches actual location:
   ```bash
   dir ml-pipeline\model\textile_classifier_rf.pkl
   ```
3. Update `main.py` model path if different

### Issue: Port Already in Use

**Symptom:** "Port 8000 already in use" or "Port 3000 already in use"

**Solution - Frontend:**
```bash
cd frontend
pnpm dev -p 3001          # Use different port
```

**Solution - Backend:**
```bash
# Edit backend/.env or use command line:
python main.py --port 8001
```

### Issue: Python Dependency Errors

**Symptom:** "ModuleNotFoundError" or missing packages

**Solution:**
```bash
cd backend\fabric_backend
python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📚 Next Steps

After successful reorganization:

1. **Update documentation** - Update any team docs/wikis with new structure
2. **Update CI/CD** - If using GitHub Actions/Jenkins, update build paths
3. **Update deployment** - Update deployment scripts (if any)
4. **Team onboarding** - Share this README with team members
5. **Version control** - Push changes and create pull request if using git flow

---

## 📊 Quick Reference

| Task | Command | Location |
|------|---------|----------|
| Start frontend | `pnpm dev` | `frontend/` |
| Start backend | `python main.py` | `backend/fabric_backend/` |
| Install frontend deps | `pnpm install` | `frontend/` |
| Install backend deps | `pip install -r requirements.txt` | `backend/fabric_backend/` |
| View API docs | Visit `http://localhost:8000/docs` | (if Swagger enabled) |
| Run tests | `python test_prediction_match.py` | `ml-pipeline/` |
| Build frontend | `pnpm build` | `frontend/` |

---

## ⚠️ Important Notes

1. **Environment Variables:** Don't commit `.env` or `.env.local` files to git (already in .gitignore)
2. **Model File:** Verify `ml-pipeline/model/textile_classifier_rf.pkl` exists before running backend
3. **Virtual Environments:** Create fresh venv if needed: `python -m venv backend/.venv` then `backend\.venv\Scripts\activate`
4. **Port Conflicts:** Close other apps using ports 3000 and 8000 if needed

---

## ✨ Summary Checklist

- [ ] Backup original project
- [ ] Run `REORGANIZATION.bat`
- [ ] Update `backend/fabric_backend/main.py` (model path)
- [ ] Update `frontend/app/page.tsx` (API URL)
- [ ] Create `.env.local` files
- [ ] Replace `README.md` with new version
- [ ] Update `.gitignore` patterns
- [ ] Test frontend startup (`pnpm dev`)
- [ ] Test backend startup (`python main.py`)
- [ ] Test API health check (curl)
- [ ] Test frontend→backend connection
- [ ] Delete optional docs (if desired)
- [ ] Delete auto-generated files (`.next/`, `node_modules/`)
- [ ] Commit to git
- [ ] Share with team

---

**Congratulations! Your project is now organized and ready for development.** 🎉

For questions, see `FILE_UPDATES_REQUIRED.md` or `SAFE_DELETIONS.md`.
