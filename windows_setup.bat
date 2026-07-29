@echo off
setlocal enabledelayedexpansion
set "BASE_DIR=%~dp0"

:: 1. Pull git repo - if not pulled
echo Checking for git repository...
git remote add origin https://github.com/IncreasingLoss/animal_multiclass_cnn.git 2>nul
git remote set-url origin https://github.com/IncreasingLoss/animal_multiclass_cnn.git 2>nul
git pull origin main
if %errorlevel% neq 0 (
    echo Git pull failed or not a git repository. Proceeding anyway...
)


:: 2. Check for Python (3.12.2 to 3.14)
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. 
    echo Please install Python (version 3.12.2 or newer) from https://www.python.org/ and try again.
    pause
    exit /b
)
echo Python detected.

:: 3. Create new venv and pip install all from requirements.txt
if not exist %BASE_DIR%.venv (
    echo Creating virtual environment...
    %BASE_DIR%python -m venv %BASE_DIR%.venv
)

echo Activating virtual environment...
call %BASE_DIR%.venv\Scripts\activate

echo Installing dependencies from requirements.txt...
%BASE_DIR%.venv\Scripts\python.exe -m pip install --upgrade pip
%BASE_DIR%.venv\Scripts\python.exe -m pip install -r requirements.txt

:: 3.5 Pull models from Hugging Face if missing
echo Checking for models...
if not exist %BASE_DIR%models (
    %BASE_DIR%.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IncreasingLoss/animal_multiclass_cnn', local_dir='%BASE_DIR%models')"
)

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
