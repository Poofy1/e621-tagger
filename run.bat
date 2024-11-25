@echo off

:: Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install requirements
pip install flask torch torchvision transformers pillow

:: Create required directories
mkdir uploads 2>nul
mkdir checkpoints 2>nul
mkdir data 2>nul
mkdir templates 2>nul

:: Start Flask app in background and open browser
start /B python app.py
timeout /t 2 /nobreak >nul
start http://localhost:5000