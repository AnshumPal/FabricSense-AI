@echo off
REM ====================================================================
REM FabricSense-AI Project Reorganization Script
REM Windows Batch Script - Run from project root
REM ====================================================================
REM This script reorganizes the project from flat structure to:
REM   textileProject/
REM   ├── frontend/
REM   ├── backend/
REM   ├── ml-pipeline/
REM   ├── docs/
REM   └── .vscode/, .gitignore, README.md
REM ====================================================================

echo.
echo ====================================================================
echo Starting Project Reorganization...
echo ====================================================================
echo.

REM Step 1: Create new directory structure
echo [1/4] Creating new directory structure...
mkdir frontend 2>nul
mkdir frontend\app 2>nul
mkdir frontend\components 2>nul
mkdir frontend\hooks 2>nul
mkdir frontend\lib 2>nul
mkdir frontend\styles 2>nul
mkdir frontend\public 2>nul
mkdir backend 2>nul
mkdir ml-pipeline 2>nul
mkdir ml-pipeline\model 2>nul
mkdir ml-pipeline\datasets 2>nul
mkdir ml-pipeline\notebooks 2>nul
mkdir ml-pipeline\scripts 2>nul
mkdir docs 2>nul
echo ✓ Directories created

echo.
echo [2/4] Moving frontend files...
REM Move Next.js app and config files to frontend/
move app frontend\ >nul 2>&1
move components frontend\ >nul 2>&1
move hooks frontend\ >nul 2>&1
move lib frontend\ >nul 2>&1
move styles frontend\ >nul 2>&1
move public frontend\ >nul 2>&1
move components.json frontend\ >nul 2>&1
move next.config.mjs frontend\ >nul 2>&1
move next-env.d.ts frontend\ >nul 2>&1
move tsconfig.json frontend\ >nul 2>&1
move package.json frontend\ >nul 2>&1
move package-lock.json frontend\ >nul 2>&1
move pnpm-lock.yaml frontend\ >nul 2>&1
move postcss.config.mjs frontend\ >nul 2>&1
echo ✓ Frontend files moved

echo.
echo [3/4] Moving backend and ML files...
REM Move backend
move fabric_backend backend\fabric_backend >nul 2>&1
if exist fabric_backend (
    rmdir /s /q fabric_backend >nul 2>&1
)

REM Move ML model and training files
move model backend.. >nul 2>&1
if exist model (
    xcopy model ml-pipeline\model /E /I /Y >nul 2>&1
    rmdir /s /q model >nul 2>&1
)

REM Move test script to ml-pipeline
move test_prediction_match.py ml-pipeline\ >nul 2>&1

echo ✓ Backend and ML files moved

echo.
echo [4/4] Moving documentation files...
REM Move markdown documentation to docs/
move DEPLOYMENT_COMPLETE.md docs\ >nul 2>&1
move FABRICSENSE_AI_AUDIT_REPORT.md docs\ >nul 2>&1
move PREDICTION_FIX_SUMMARY.md docs\ >nul 2>&1
move Readme.txt docs\ >nul 2>&1
REM Keep main README.md at root (will replace with new one)

echo ✓ Documentation moved

echo.
echo ====================================================================
echo NEXT STEPS:
echo ====================================================================
echo.
echo 1. Update file paths in these files:
echo    - backend/fabric_backend/main.py (line ~29: update model path)
echo    - frontend/app/page.tsx (line ~36: make API URL configurable)
echo    - ml-pipeline/test_prediction_match.py (line ~47: check API URL)
echo.
echo 2. Create new root README.md at textileProject/
echo.
echo 3. Create .env files for environment variables:
echo    - frontend/.env.local (for API URL)
echo    - backend/.env (for model path and PORT)
echo.
echo 4. Update .gitignore to cover new structure
echo.
echo 5. Run from new locations:
echo    Frontend: cd frontend && pnpm install && pnpm dev
echo    Backend:  cd backend\fabric_backend && pip install -r requirements.txt && python main.py
echo    ML:       cd ml-pipeline && python scripts/train.py
echo.
echo 6. Safe to delete:
echo    - .next/ (regenerated on build)
echo    - node_modules/ (reinstalled via pnpm install)
echo    - Audit reports from docs/ if no longer needed
echo.
echo ====================================================================
echo Reorganization complete!
echo ====================================================================
pause
