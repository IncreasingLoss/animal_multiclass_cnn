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

:: 3. Create new venv and activate it
if not exist %BASE_DIR%.venv (
    echo Creating virtual environment...
    %BASE_DIR%python -m venv %BASE_DIR%.venv
)

echo Activating virtual environment...
call %BASE_DIR%.venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

:: 4. Detect CUDA and install PyTorch first to ensure correct wheel
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Determining best Torch wheel...
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') do set "DRV_VER=%%i"
    echo Detected Driver Version: %DRV_VER%
    
    :: Robust version detection logic
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"

    :: Check for CUDA 13 support (Driver >= 560)
    echo %DRV_VER% | findstr /R /C:"[5][6-9][0-9]" >nul
    if %errorlevel% equ 0 (
        echo Driver version supports CUDA 13.x...
        set "TORCH_URL=https://download.pytorch.org/whl/cu132"
    ) else (
        :: Check for CUDA 12 support (Driver >= 525)
        echo %DRV_VER% | findstr /R /C:"[5][2-9][0-9]" >nul
        if %errorlevel% equ 0 (
            echo Driver version supports CUDA 12.x...
            set "TORCH_URL=https://download.pytorch.org/whl/cu124"
        ) else (
            :: Check for CUDA 11 support (Driver >= 450)
            echo %DRV_VER% | findstr /R /C:"[4][5-9][0-9]" >nul
            if %errorlevel% equ 0 (
                echo Driver version supports CUDA 11.x...
                set "TORCH_URL=https://download.pytorch.org/whl/cu118"
            ) else (
                echo Driver version is older than 450 or unrecognized for standard Torch wheels. 
                echo Falling back to cu121 as a common compatible target...
                set "TORCH_URL=https://download.pytorch.org/whl/cu121"
            )
        )
    )
    
    :: Final validation: check if the selected URL is actually accessible before proceeding
    echo Verifying %TORCH_URL%...
    curl -Is %TORCH_URL% | find "200" >nul
    if %errorlevel% neq 0 (
        echo Warning: Selected URL %TORCH_URL% returned non-200. Falling back to cu121.
        set "TORCH_URL=https://download.pytorch.org/whl/cu121"
    ) else (
        echo Verified link: %TORCH_URL%
    )
) else (
    echo No NVIDIA GPU detected or nvidia-smi not found. Using CPU version.
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
)

echo Installing Torch from %TORCH_URL%...
python -m pip install --force-reinstall torch torchvision torchaudio --index-url %TORCH_URL%

:: 5. Install other specific dependencies
echo Installing additional dependencies...
python -m pip install gradio numpy ultralytics requests pypinyin pillow

:: 6. Pull models from Hugging Face if missing
echo Checking for models...
if not exist %BASE_DIR%models (
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IncreasingLoss/animal_multiclass_cnn', local_dir='%BASE_DIR%models')"
)

:: 7. Install remaining requirements from file
echo Installing remaining dependencies from requirements.txt...
python -m pip install -r requirements.txt

:: 8. Start app.py
echo Starting the application...
python Wildlife_Animal_Classifier\app.py

pause
