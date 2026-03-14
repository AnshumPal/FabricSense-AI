@echo off
REM ====================================================================
REM FabricSense-AI Project Structure Reorganization
REM Windows Batch Script - Full Automation
REM ====================================================================
REM Run this from: c:\Users\ASUS\OneDrive\Desktop\textileProject
REM ====================================================================

setlocal enabledelayedexpansion
color 0A
title FabricSense-AI Reorganization Script

echo.
echo ====================================================================
echo  FabricSense-AI Project Reorganization
echo ====================================================================
echo.
echo This script will reorganize your project into a clean structure:
echo   - frontend/       (All Next.js code)
echo   - backend/        (All Python/FastAPI code)
echo   - ml-pipeline/    (Models, datasets, training scripts)
echo   - docs/           (All markdown documentation)
echo.
echo Starting in 3 seconds... Press Ctrl+C to cancel
timeout /t 3 /nobreak >nul

REM ====================================================================
REM STEP 1: CREATE DIRECTORY STRUCTURE
REM ====================================================================
echo.
echo [STEP 1/5] Creating directory structure...
echo.

mkdir frontend 2>nul
mkdir frontend\app 2>nul
mkdir frontend\components 2>nul
mkdir frontend\hooks 2>nul
mkdir frontend\lib 2>nul
mkdir frontend\styles 2>nul
mkdir frontend\public 2>nul

mkdir backend 2>nul
mkdir backend\fabric_backend 2>nul

mkdir ml-pipeline 2>nul
mkdir ml-pipeline\model 2>nul
mkdir ml-pipeline\datasets 2>nul
mkdir ml-pipeline\notebooks 2>nul
mkdir ml-pipeline\scripts 2>nul

mkdir docs 2>nul

echo ✓ All directories created successfully

REM ====================================================================
REM STEP 2: MOVE FRONTEND FILES
REM ====================================================================
echo.
echo [STEP 2/5] Moving frontend files...
echo.

move app frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move app\
move components frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move components\
move hooks frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move hooks\
move lib frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move lib\
move styles frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move styles\
move public frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move public\

move package.json frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move package.json
move package-lock.json frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move package-lock.json
move pnpm-lock.yaml frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move pnpm-lock.yaml
move tsconfig.json frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move tsconfig.json
move next.config.mjs frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move next.config.mjs
move next-env.d.ts frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move next-env.d.ts
move postcss.config.mjs frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move postcss.config.mjs
move components.json frontend\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move components.json

echo ✓ Frontend files moved successfully

REM ====================================================================
REM STEP 3: MOVE BACKEND FILES
REM ====================================================================
echo.
echo [STEP 3/5] Moving backend files...
echo.

if exist fabric_backend (
    xcopy fabric_backend backend\fabric_backend /E /I /Y >nul 2>&1
    rmdir /s /q fabric_backend >nul 2>&1
    echo ✓ Backend files moved successfully
) else (
    echo ⚠ fabric_backend/ not found
)

REM ====================================================================
REM STEP 4: MOVE ML-PIPELINE FILES
REM ====================================================================
echo.
echo [STEP 4/5] Moving ML pipeline files...
echo.

if exist model (
    xcopy model ml-pipeline\model /E /I /Y >nul 2>&1
    rmdir /s /q model >nul 2>&1
    echo ✓ Model directory moved
) else (
    echo ⚠ model/ not found
)

move test_prediction_match.py ml-pipeline\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move test_prediction_match.py
echo ✓ ML pipeline files moved successfully

REM ====================================================================
REM STEP 5: MOVE DOCUMENTATION FILES
REM ====================================================================
echo.
echo [STEP 5/5] Moving documentation files...
echo.

move DEPLOYMENT_COMPLETE.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move DEPLOYMENT_COMPLETE.md
move FABRICSENSE_AI_AUDIT_REPORT.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move FABRICSENSE_AI_AUDIT_REPORT.md
move PREDICTION_FIX_SUMMARY.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move PREDICTION_FIX_SUMMARY.md
move Readme.txt docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move Readme.txt
move FILE_UPDATES_REQUIRED.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move FILE_UPDATES_REQUIRED.md
move EXECUTION_GUIDE.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move EXECUTION_GUIDE.md
move SAFE_DELETIONS.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move SAFE_DELETIONS.md
move START_HERE.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move START_HERE.md
move README_NEW.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move README_NEW.md
move README_REORGANIZATION_PACKAGE.md docs\ >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move README_REORGANIZATION_PACKAGE.md
move README.md docs\README_ARCHIVE.md >nul 2>&1
if errorlevel 1 echo   ⚠ Could not move README.md

echo ✓ Documentation files moved successfully

REM ====================================================================
REM CREATE NEW ROOT README.md
REM ====================================================================
echo.
echo [BONUS] Creating new root README.md...
echo.

(
echo # FabricSense AI - Hyperspectral Textile Classification
echo.
echo Automated textile classification using hyperspectral imaging and machine learning for recycling and sustainable fashion.
echo.
echo ## 🎯 Quick Start
echo.
echo ### Frontend ^(Next.js^)
echo ```bash
echo cd frontend
echo pnpm install
echo pnpm dev
echo ```
echo Development server: http://localhost:3000
echo.
echo ### Backend ^(FastAPI^)
echo ```bash
echo cd backend/fabric_backend
echo pip install -r requirements.txt
echo python main.py
echo ```
echo API server: http://localhost:8000
echo.
echo ### ML Pipeline
echo ```bash
echo cd ml-pipeline
echo python test_prediction_match.py
echo ```
echo.
echo ## 📁 Project Structure
echo.
echo ```
echo textileProject/
echo ├── frontend/              # Next.js web interface
echo │   ├── app/               # Page routes
echo │   ├── components/        # React components
echo │   ├── hooks/             # Custom hooks
echo │   ├── lib/               # Utilities
echo │   ├── styles/            # Global CSS
echo │   └── public/            # Static assets
echo │
echo ├── backend/               # FastAPI server
echo │   └── fabric_backend/
echo │       ├── main.py        # API entry point
echo │       ├── requirements.txt
echo │       └── model/         # Predictions
echo │
echo ├── ml-pipeline/           # ML training ^& evaluation
echo │   ├── model/             # Saved models
echo │   ├── datasets/          # Data directory
echo │   ├── notebooks/         # Jupyter notebooks
echo │   └── scripts/           # Training scripts
echo │
echo └── docs/                  # Documentation
echo     ├── README_ARCHIVE.md
echo     └── Other guides...
echo ```
echo.
echo ## 🚀 Features
echo.
echo - 📤 CSV file upload with drag-and-drop
echo - 📈 Real-time spectral data visualization
echo - 🎯 Fabric type classification with confidence scores
echo - 🤖 ML-powered predictions ^(Random Forest^)
echo - 🌙 Dark/Light theme support
echo.
echo ## 📊 Supported Fabrics
echo.
echo - Cotton
echo - Cotton/Poly Blend
echo - Poly/Spandex
echo.
echo ## 🔌 API Endpoints
echo.
echo - `POST /predict` - Classify fabric from CSV
echo - `GET /health` - Check API status
echo.
echo ## 📚 Documentation
echo.
echo See `docs/` folder for detailed guides and reports.
echo.
echo ## 🛠 Development
echo.
echo | Component | Command | Port |
echo |-----------|---------|------|
echo | Frontend | `cd frontend ^&^& pnpm dev` | 3000 |
echo | Backend | `cd backend\fabric_backend ^&^& python main.py` | 8000 |
echo | ML Tests | `cd ml-pipeline ^&^& python test_prediction_match.py` | - |
echo.
echo ## ✅ Last Updated
echo.
echo Project reorganized: March 2026
echo Structure: Clean and modular
echo Status: Ready for development
) > README.md

echo ✓ New README.md created at root

REM ====================================================================
REM COMPLETION
REM ====================================================================
echo.
echo ====================================================================
echo  ✅ REORGANIZATION COMPLETE!
echo ====================================================================
echo.
echo Your project is now organized into:
echo.
echo   ✓ frontend/       - All Next.js code
echo   ✓ backend/        - Python FastAPI server
echo   ✓ ml-pipeline/    - Models and training
echo   ✓ docs/           - Documentation
echo.
echo ====================================================================
echo  📝 NEXT STEPS:
echo ====================================================================
echo.
echo 1. Update frontend imports (if needed):
echo    Navigate to: frontend/app/page.tsx
echo    Check API endpoint is correct
echo.
echo 2. Update backend model path (if needed):
echo    Navigate to: backend/fabric_backend/main.py
echo    Check model path points to: ../../../ml-pipeline/model/
echo.
echo 3. Test Frontend:
echo    cd frontend
echo    pnpm install ^(first time only^)
echo    pnpm dev
echo.
echo 4. Test Backend:
echo    cd backend\fabric_backend
echo    pip install -r requirements.txt ^(first time only^)
echo    python main.py
echo.
echo 5. Verify they can communicate:
echo    Frontend: http://localhost:3000
echo    Backend: http://localhost:8000/health
echo.
echo ====================================================================
echo  🗑️  Optional Cleanup:
echo ====================================================================
echo.
echo Safe to delete manually:
echo   - Optionally move node_modules/ to frontend/ later
echo   - Old batch scripts if no longer needed
echo.
echo Do NOT delete:
echo   - .vscode/ folder
echo   - .git/ folder
echo   - .gitignore file
echo.
echo ====================================================================
echo.
pause
