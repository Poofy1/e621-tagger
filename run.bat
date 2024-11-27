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

:: Update pip
echo Updating pip...
python -m pip install --quiet --upgrade pip
if errorlevel 1 (
    echo Failed to update pip
    pause
    exit /b 1
)

:: Install all required packages
echo Installing requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

:: Start Flask app
echo Starting webui...
python webui.py
if errorlevel 1 (
    echo Failed to start webui
    pause
    exit /b 1
)

endlocal