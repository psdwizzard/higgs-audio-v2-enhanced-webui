@echo off
setlocal EnableDelayedExpansion
title Higgs Audio Discord Bot - Installer
color 0B
cd /d "%~dp0.."
echo ============================================
echo   Higgs Audio Discord Bot - Installer
echo ============================================
echo.
echo Working directory: %CD%
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

REM Check for ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg not found in PATH.
    echo ffmpeg is required for MP3 conversion.
    echo Download from https://ffmpeg.org/download.html
    echo.
) else (
    echo [OK] ffmpeg found
    echo.
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. CUDA may not be available.
    echo The bot will fall back to CPU which is much slower.
    echo.
) else (
    echo [OK] NVIDIA GPU detected
    echo.
)

set VENV_DIR=discord_bot\bot_venv
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

REM Create venv if it doesn't exist
if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment...
    python -m venv --system-site-packages "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created at %VENV_DIR%
    echo.
) else (
    echo [OK] Virtual environment already exists
    echo.
)

REM Install PyTorch with CUDA in the venv
echo Checking PyTorch CUDA support...
"%VENV_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'no cuda'" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch with CUDA support (this may take a few minutes^)...
    "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo.
        echo WARNING: CUDA 12.4 install failed, trying CUDA 11.8...
        "%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if errorlevel 1 (
            echo ERROR: Failed to install PyTorch with CUDA.
            echo You can try manually: %VENV_PIP% install torch --index-url https://download.pytorch.org/whl/cu124
            pause
            exit /b 1
        )
    )
    echo.
) else (
    echo [OK] PyTorch with CUDA already installed
    echo.
)

REM Pin transformers to compatible version
echo Installing bot dependencies...
"%VENV_PIP%" install transformers==4.47.1 discord.py>=2.3.0 python-dotenv>=1.0.0 aiohttp>=3.9.0
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo.
echo [OK] All dependencies installed
echo.

REM Verify CUDA
echo Verifying CUDA...
"%VENV_PYTHON%" -c "import torch; print(f'  torch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else '  CUDA: NOT AVAILABLE - will use CPU')"
echo.

REM Check if .env exists, offer to run setup
if not exist ".env" (
    echo No .env file found. You need to configure
    echo your Discord credentials before running the bot.
    echo.
    set /p RUNSETUP="Run setup wizard now? (y/n): "
    if /i "!RUNSETUP!"=="y" (
        call "%~dp0setup_bot.bat"
    ) else (
        echo.
        echo You can run setup_bot.bat later to configure credentials.
    )
) else (
    echo [OK] .env file already exists
)

echo.
echo ============================================
echo   Installation complete!
echo.
echo   Next steps:
echo     1. Run setup_bot.bat  (if you haven't yet)
echo     2. Run start_bot.bat  to launch the bot
echo ============================================
pause
