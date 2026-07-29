@echo off
setlocal enabledelayedexpansion

:: 1. Pull git repo - if not pulled
echo Checking for git repository...
git pull origin main
if %errorlevel% neq 0 (
    echo Git pull failed or not a git repository. Proceeding anyway...
)

:: 2. Install python 3.14 if not installed
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. 
    echo Please install Python 3.14 from https://www.python.org/ and try again.
    pause
    exit /b
)
echo Python detected.

:: 3. Create new venv and pip install all from requirements.txt
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Find if nvidia smi is available - then download right torch version
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Installing Torch with CUDA support...
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected or nvidia-smi not found. Installing standard Torch...
    pip install --force-reinstall torch torchvision torchaudio
)

:: 5. Start app.py
echo Starting the application...
python Wildlife_Animal_Classifier\app.py

pause
