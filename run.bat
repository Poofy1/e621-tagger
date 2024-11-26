@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

:: Install all required packages
echo Installing requirements...
pip install --quiet ^
    flask ^
    torch ^
    torchvision ^
    Pillow ^
    pandas ^
    pyarrow ^
    numpy ^
    tqdm ^
    transformers ^
    scikit-learn ^
    matplotlib ^
    fastparquet

if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

:: Create required directories
echo Creating directories...
if not exist uploads mkdir uploads
if not exist checkpoints mkdir checkpoints
if not exist data mkdir data
if not exist templates mkdir templates

:: Check if model file exists
if not exist "checkpoints\best_model.pth" (
    echo Warning: Model file not found in checkpoints folder
    echo Please ensure you have the model file in the correct location
    pause
)

:: Start Flask app
echo Starting Flask application...
python webui.py
if errorlevel 1 (
    echo Failed to start Flask application
    pause
    exit /b 1
)

endlocal