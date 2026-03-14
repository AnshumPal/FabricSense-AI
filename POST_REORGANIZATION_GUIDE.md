# Post-Reorganization Verification & Next Steps

**Run this AFTER executing REORGANIZE_NOW.bat successfully**

---

## ✅ STEP 1: Verify Folder Structure Created

Open file explorer or terminal and verify all these folders exist:

```bash
# From project root, check:
dir frontend\app
dir frontend\components
dir backend\fabric_backend
dir ml-pipeline\model
dir docs
```

**Expected:** All folders exist without errors.

---

## 📝 STEP 2: Update Code Paths (CRITICAL)

### **File 1: backend/fabric_backend/main.py**

**Location:** Line ~29 (inside `lifespan()` function)

**BEFORE:**
```python
model_path = Path(__file__).parent / "model" / "textile_classifier_rf.pkl"
```

**AFTER:**
```python
model_path = Path(__file__).parent.parent.parent / "ml-pipeline" / "model" / "textile_classifier_rf.pkl"
```

**Why:** The model file moved from `root/model/` to `ml-pipeline/model/`, so backend needs new path.

---

### **File 2: frontend/app/page.tsx**

**Location:** Line ~36 (inside `handlePredict()` function)

**BEFORE:**
```typescript
const response = await fetch("http://localhost:8000/predict", {
```

**AFTER:**
```typescript
const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const response = await fetch(`${apiUrl}/predict`, {
```

**Why:** Makes API URL configurable via environment variables.

**Then create:** `frontend/.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### **File 3: ml-pipeline/test_prediction_match.py** (Optional)

**Location:** Line ~47

**BEFORE:**
```python
response = requests.post('http://localhost:8000/predict', files=files)
```

**AFTER:**
```python
import os
api_url = os.getenv('API_URL', 'http://localhost:8000')
response = requests.post(f'{api_url}/predict', files=files)
```

**Then create:** `ml-pipeline/.env`
```
API_URL=http://localhost:8000
```

---

## 🧪 STEP 3: Install Dependencies

### **Frontend**
```bash
cd frontend
pnpm install
# Or: npm install
```

Takes ~2-5 minutes depending on internet.

### **Backend**
```bash
cd backend\fabric_backend
pip install -r requirements.txt
```

Takes ~1-2 minutes.

---

## 🚀 STEP 4: Test Each Component

### **Test 1: Frontend Starts**

```bash
cd frontend
pnpm dev
```

**Expected output:**
```
> next dev
⚡ starting dev server...
✓ compiled
▲ Next.js 16.1.6
- Local: http://localhost:3000
```

✅ Leave this running, open new terminal for next test.

---

### **Test 2: Backend Starts**

```bash
cd backend\fabric_backend
python main.py
```

**Expected output:**
```
✓ Model loaded successfully from C:\...\ml-pipeline\model\textile_classifier_rf.pkl
⚙️ uvicorn running on http://localhost:8000
```

✅ Leave this running, open new terminal for next test.

---

### **Test 3: API Health Check**

```bash
# From a new terminal (cmd/PowerShell)
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "required_bands": 100,
  "fabric_classes": ["Cotton", "Cotton/Poly blend", "Poly/Spandex"]
}
```

✅ If you see this, backend is working!

---

### **Test 4: Frontend → Backend Communication**

1. Open browser: `http://localhost:3000`
2. You should see the FabricSense AI interface
3. Upload a test CSV file (if available)
4. Click "Predict"
5. Should see results without errors

✅ If this works, everything is connected!

---

## 📋 STEP 5: Verify All Files Moved Correctly

Run this from project root:

```bash
# Check frontend files
if exist frontend\package.json echo ✓ Frontend package.json exists
if exist frontend\app echo ✓ Frontend app/ exists
if exist frontend\components echo ✓ Frontend components/ exists

# Check backend files
if exist backend\fabric_backend\main.py echo ✓ Backend main.py exists
if exist backend\fabric_backend\requirements.txt echo ✓ Backend requirements.txt exists

# Check ML files
if exist ml-pipeline\model echo ✓ ML model/ exists
if exist ml-pipeline\test_prediction_match.py echo ✓ ML test script exists

# Check docs files
if exist docs echo ✓ Docs folder exists
if exist docs\DEPLOYMENT_COMPLETE.md echo ✓ Docs are archived

# Check root files
if exist README.md echo ✓ New README.md exists
if exist .gitignore echo ✓ .gitignore exists
if exist .vscode echo ✓ .vscode folder exists
```

**All should return ✓**

---

## 🗑️ STEP 6: Optional Cleanup (Safe Deletions)

After verifying everything works, you can delete:

### **Auto-Generated Cache (Safe to delete)**
```bash
# Optional - these regenerate automatically
rmdir /s /q frontend\.next
rmdir /s /q .next
del package-lock.json  (if using pnpm)
```

**Disk space saved:** ~200-500 MB

### **Old Batch Scripts (Optional)**
```bash
# If you have multiple batch scripts and want to keep only REORGANIZE_NOW.bat
del REORGANIZATION.bat
```

### **Never Delete**
```bash
# DO NOT DELETE THESE:
# - .git/
# - .vscode/
# - .gitignore
# - node_modules/ (if still in root, soon it will be in frontend/)
```

---

## 📊 Final Folder Structure

After all steps, your structure should look like:

```
textileProject/
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   ├── styles/
│   ├── public/
│   ├── package.json ✓
│   ├── tsconfig.json ✓
│   ├── next.config.mjs ✓
│   ├── .env.local ✓
│   └── .next/ (will auto-generate)
│
├── backend/
│   └── fabric_backend/
│       ├── main.py ✓ (UPDATED)
│       ├── requirements.txt ✓
│       ├── model/
│       └── utils/
│
├── ml-pipeline/
│   ├── model/ ✓
│   ├── test_prediction_match.py ✓
│   ├── datasets/ (empty)
│   ├── notebooks/ (empty)
│   ├── scripts/ (empty)
│   └── .env (optional)
│
├── docs/ ✓
│   ├── README_ARCHIVE.md
│   ├── DEPLOYMENT_COMPLETE.md
│   ├── FABRICSENSE_AI_AUDIT_REPORT.md
│   └── ... (other guides)
│
├── .vscode/ ✓
├── .git/ ✓
├── .gitignore ✓
├── .gitignore (keep as-is)
├── node_modules/ (in root is ok for now)
├── README.md ✓ (NEW)
└── REORGANIZE_NOW.bat
```

---

## ✅ Completion Checklist

- [ ] Ran `REORGANIZE_NOW.bat` successfully
- [ ] All folders created without errors
- [ ] Updated `backend/fabric_backend/main.py` (model path)
- [ ] Updated `frontend/app/page.tsx` (API URL)
- [ ] Created `frontend/.env.local`
- [ ] Frontend starts: `cd frontend && pnpm dev`
- [ ] Backend starts: `cd backend\fabric_backend && python main.py`
- [ ] API responds to `curl http://localhost:8000/health`
- [ ] Frontend loads at `http://localhost:3000`
- [ ] Frontend can upload and predict (if test CSV available)
- [ ] All verification scripts returned ✓
- [ ] Optionally cleaned up `.next/` and caches

---

## 🔍 Troubleshooting

### **Issue: "Can't find module" error in frontend**

**Causes:**
- File didn't move properly
- Import paths are wrong

**Fix:**
```bash
cd frontend
pnpm install --force
pnpm dev
```

---

### **Issue: Backend says "Model not found"**

**Cause:** Model path is wrong

**Fix:**
1. Check: `dir ml-pipeline\model\textile_classifier_rf.pkl`
2. Should exist and show the file
3. Verify `backend/fabric_backend/main.py` has correct path:
   ```python
   Path(__file__).parent.parent.parent / "ml-pipeline" / "model"
   ```

---

### **Issue: "Port 3000 already in use"**

**Fix:**
```bash
cd frontend
pnpm dev -p 3001  # Use different port
```

---

### **Issue: "Port 8000 already in use"**

**Fix:**
```bash
cd backend\fabric_backend
python -c "import sys; sys.path.insert(0, '.'); from main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
```

---

### **Issue: CORS error (frontend can't reach backend)**

**Fix:**
1. Verify backend is running (check port 8000)
2. Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local`
3. Verify it's `http://localhost:8000`
4. Restart frontend: `pnpm dev`

---

## 📞 Getting Help

1. Check `docs/` folder for original guides
2. Review this file again (post-execution guide)
3. Check browser console (F12) for error messages
4. Check terminal output for stack traces

---

## 🎉 Success!

If you completed all steps and tests pass, your project is now:

✅ **Organized** - Clear separation of frontend, backend, and ML  
✅ **Scalable** - Easy to add new features to each part  
✅ **Maintainable** - Anyone can understand the structure  
✅ **Professional** - Industry-standard project layout  

Congratulations! 🚀

---

**Date:** March 12, 2026  
**Status:** Post-Reorganization Ready
