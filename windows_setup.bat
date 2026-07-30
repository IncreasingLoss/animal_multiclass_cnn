@echo off
setlocal enabledelayedexpansion

:: =====================================================================
::  animal_multiclass_cnn - Windows setup & launch script
::  Safe to re-run: skips steps that are already done
:: =====================================================================

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"
echo [i] Working directory: %BASE_DIR%

where winget >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] winget was not found on this system.
    echo     Automatic installs of Python/Git below will fail if they are missing.
    echo     You can get winget via the "App Installer" package in the Microsoft Store.
)

:: ---------------------------------------------------------------------
:: 1) Check / install Git, then clone (first run) or pull (later runs)
:: ---------------------------------------------------------------------
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Git not found. Installing via winget...
    winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo [X] Failed to install Git automatically.
        echo     Please install it manually from https://git-scm.com and re-run.
        pause
        exit /b 1
    )
    call :RefreshPath
    where git >nul 2>&1
    if !errorlevel! neq 0 (
        echo [!] Git installed but this window can't see it yet.
        echo     Close this window, open a NEW terminal, and run the script again.
        pause
        exit /b 1
    )
)

if not exist ".git" (
    echo [i] No repo here yet - cloning...
    git clone https://github.com/IncreasingLoss/animal_multiclass_cnn.git .
    if !errorlevel! neq 0 (
        echo [X] Git clone failed. Check your internet connection and the repo URL.
        pause
        exit /b 1
    )
) else (
    echo [i] Repo already present - pulling latest changes...
    git remote set-url origin https://github.com/IncreasingLoss/animal_multiclass_cnn.git 2>nul
    git pull origin main
    if !errorlevel! neq 0 (
        echo [!] Git pull failed - continuing with the local copy already on disk.
    )
)

:: ---------------------------------------------------------------------
:: 2) Check / install Python (needs 3.12.2 - 3.14)
:: ---------------------------------------------------------------------
set "PY_OK=0"
set "PY_VER="
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
)

if defined PY_VER (
    for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
        set "PY_MAJOR=%%a"
        set "PY_MINOR=%%b"
    )
    if "!PY_MAJOR!"=="3" (
        if !PY_MINOR! GEQ 12 if !PY_MINOR! LEQ 14 set "PY_OK=1"
    )
)

if "!PY_OK!"=="0" (
    if defined PY_VER (
        echo [!] Found Python !PY_VER!, but this project needs 3.12.2 - 3.14.
    ) else (
        echo [!] Python not found in PATH.
    )
    echo [i] Installing Python 3.13 via winget...
    winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo [X] Winget failed to install Python.
        echo     Please install 3.12.2+ manually from https://www.python.org and re-run.
        pause
        exit /b 1
    )
    call :RefreshPath
    where python >nul 2>&1
    if !errorlevel! neq 0 (
        echo [!] Python installed but this window can't see it yet.
        echo     Close this window, open a NEW terminal, and run the script again.
        pause
        exit /b 1
    )
    echo [i] Python is now available.
) else (
    echo [i] Python !PY_VER! OK.
)

:: ---------------------------------------------------------------------
:: 3) Create / activate the virtual environment
:: ---------------------------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    echo [i] Creating virtual environment...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [X] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [i] Virtual environment already exists.
)

echo [i] Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo [i] Upgrading pip...
python -m pip install --upgrade pip

:: ---------------------------------------------------------------------
:: 4) Detect CUDA and pick the matching PyTorch wheel index
:: ---------------------------------------------------------------------
set "TORCH_URL=https://download.pytorch.org/whl/cpu"

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [i] NVIDIA GPU detected. Reading CUDA version...
    set "CUDA_VER="
    for /f "tokens=9" %%v in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set "CUDA_VER=%%v"
    echo [i] Reported CUDA Version: !CUDA_VER!

    if defined CUDA_VER (
        for /f "tokens=1,2 delims=." %%a in ("!CUDA_VER!") do (
            set "CUDA_MAJOR=%%a"
            set "CUDA_MINOR=%%b"
        )
        if !CUDA_MAJOR! GEQ 13 (
            set "TORCH_URL=https://download.pytorch.org/whl/cu130"
        ) else (
            if !CUDA_MAJOR! EQU 12 (
                if !CUDA_MINOR! GEQ 8 (
                    set "TORCH_URL=https://download.pytorch.org/whl/cu128"
                ) else (
                    if !CUDA_MINOR! GEQ 6 (
                        set "TORCH_URL=https://download.pytorch.org/whl/cu126"
                    ) else (
                        set "TORCH_URL=https://download.pytorch.org/whl/cu124"
                    )
                )
            ) else (
                set "TORCH_URL=https://download.pytorch.org/whl/cu118"
            )
        )
    ) else (
        echo [!] Could not parse a CUDA version - defaulting to cu128.
        set "TORCH_URL=https://download.pytorch.org/whl/cu128"
    )
    echo [i] Selected wheel index: !TORCH_URL!

    echo [i] Verifying !TORCH_URL! is reachable...
    curl -Is !TORCH_URL! 2>nul | find "200" >nul
    if !errorlevel! neq 0 (
        echo [!] !TORCH_URL! didn't respond as expected - falling back to cu128.
        set "TORCH_URL=https://download.pytorch.org/whl/cu128"
    )
) else (
    echo [i] No NVIDIA GPU detected - using the CPU build.
)

:: Which build tag are we targeting? (cpu / cu118 / cu124 / cu126 / cu128 / cu130)
set "TORCH_TAG=!TORCH_URL:*whl/=!"

:: What's already installed, if anything?
set "TORCH_INSTALLED_TAG="
del "%TEMP%\_torchver.txt" >nul 2>&1
python -c "import torch; print(torch.__version__)" >"%TEMP%\_torchver.txt" 2>nul
if !errorlevel! equ 0 (
    set /p TORCH_FULL_VER=<"%TEMP%\_torchver.txt"
    set "TORCH_INSTALLED_TAG="
    for /f "tokens=2 delims=+" %%t in ("!TORCH_FULL_VER!") do set "TORCH_INSTALLED_TAG=%%t"
    if not defined TORCH_INSTALLED_TAG set "TORCH_INSTALLED_TAG=cpu"
)
del "%TEMP%\_torchver.txt" >nul 2>&1

set "NEED_TORCH=1"
if defined TORCH_INSTALLED_TAG (
    if "!TORCH_INSTALLED_TAG!"=="!TORCH_TAG!" (
        echo [i] PyTorch already installed with the right build ^(!TORCH_TAG!^) - skipping.
        set "NEED_TORCH=0"
    ) else (
        echo [i] Installed PyTorch build is "!TORCH_INSTALLED_TAG!" but "!TORCH_TAG!" is needed - reinstalling.
    )
) else (
    echo [i] PyTorch not installed yet.
)

if "!NEED_TORCH!"=="1" (
    echo [i] Installing PyTorch ^(this can take a while^)...
    python -m pip install torch torchvision torchaudio --index-url !TORCH_URL!
    if !errorlevel! neq 0 (
        echo [!] Install from !TORCH_URL! failed - retrying with the CPU build...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if !errorlevel! neq 0 (
            echo [X] PyTorch install failed entirely. Aborting.
            pause
            exit /b 1
        )
    )
)

:: ---------------------------------------------------------------------
:: 5) Other project dependencies
:: ---------------------------------------------------------------------
echo [i] Installing additional dependencies...
python -m pip install gradio numpy ultralytics requests pypinyin pillow
if !errorlevel! neq 0 (
    echo [X] Dependency installation failed.
    pause
    exit /b 1
)

:: ---------------------------------------------------------------------
:: 6) requirements.txt - installed BEFORE the model download, since the
::    download step needs huggingface_hub, which normally lives in here
:: ---------------------------------------------------------------------
if exist "requirements.txt" (
    echo [i] Installing remaining dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [X] Failed installing from requirements.txt.
        pause
        exit /b 1
    )
) else (
    echo [i] No requirements.txt found - skipping.
)

:: Safety net in case huggingface_hub isn't listed anywhere above
python -c "import huggingface_hub" >nul 2>&1
if %errorlevel% neq 0 (
    echo [i] Installing huggingface_hub for model download...
    python -m pip install huggingface_hub
)

:: ---------------------------------------------------------------------
:: 7) Pull the model from Hugging Face if missing
:: ---------------------------------------------------------------------
if not exist "models" (
    echo [i] Downloading model from Hugging Face...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IncreasingLoss/animal_multiclass_cnn', local_dir='models')"
    if !errorlevel! neq 0 (
        echo [X] Model download failed. Check your connection / Hugging Face access and re-run.
        pause
        exit /b 1
    )
) else (
    echo [i] "models" folder already exists - skipping download.
)

:: ---------------------------------------------------------------------
:: 8) Launch the app
:: ---------------------------------------------------------------------
echo [i] Starting the application...
python "Wildlife_Animal_Classifier\app.py"

pause
exit /b 0

:: =====================================================================
::  Helper: refresh PATH in this session from the registry, so a program
::  installed a few lines above (Python/Git) can be found without a
::  brand-new terminal window.
:: =====================================================================
:RefreshPath
set "SYS_PATH="
set "USR_PATH="
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%B"
set "PATH=%SYS_PATH%;%USR_PATH%"
exit /b 0