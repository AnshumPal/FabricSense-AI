@echo off
echo Setting up FabricSense-AI Backend...
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Installing dependencies...
pip install -r requirements.txt

echo Step 4: Copying model file...
copy ..\model\textile_classifier_rf.pkl model\textile_classifier_rf.pkl

echo.
echo Setup complete!
echo.
echo To run the server:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run server: uvicorn main:app --reload
echo.
pause
