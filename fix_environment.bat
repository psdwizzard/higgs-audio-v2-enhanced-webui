@echo off
REM Script to fix environment issues by recreating the venv with correct packages

echo ============================================================
echo  Higgs Audio WebUI - Environment Fix Script
echo ============================================================
echo.
echo This will:
echo  1. Remove the existing virtual environment
echo  2. Create a fresh virtual environment
echo  3. Install all dependencies with correct versions
echo.
pause

REM Change to the script directory
cd /d "%~dp0"

REM Remove existing virtual environment
if exist "higgs_audio_env" (
    echo [INFO] Removing existing virtual environment...
    rmdir /s /q higgs_audio_env
    echo [INFO] Old environment removed.
)

REM Check if Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python is not available in PATH.
    echo Please install Python 3.10 and add it to your PATH.
    pause
    exit /b 1
)

echo [INFO] Python version:
python --version

REM Create fresh virtual environment
echo.
echo [INFO] Creating fresh virtual environment...
python -m venv higgs_audio_env
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [INFO] Virtual environment created successfully.

REM Activate virtual environment
echo.
echo [INFO] Activating virtual environment...
call higgs_audio_env\Scripts\activate.bat

REM Verify we're in the virtual environment
echo [INFO] Verifying virtual environment...
where python
echo.

REM Upgrade pip in venv
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support first (CUDA 12.1)
echo.
echo [INFO] Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
IF ERRORLEVEL 1 (
    echo [WARNING] Failed to install PyTorch with CUDA support.
    echo Trying to install CPU version...
    python -m pip install torch torchvision torchaudio
)

REM Install numpy with specific version constraint (before other packages)
echo.
echo [INFO] Installing compatible numpy version...
python -m pip install "numpy<2.0"

REM Install pandas with compatible version
echo [INFO] Installing pandas...
python -m pip install "pandas>=2.0.0"

REM Install requirements
echo.
echo [INFO] Installing remaining requirements...
python -m pip install -r requirements.txt

REM Install Gradio and Faster-Whisper
echo.
echo [INFO] Installing Gradio and Faster-Whisper...
python -m pip install gradio faster-whisper

REM Verify installations
echo.
echo ============================================================
echo [INFO] Verifying installations...
echo ============================================================
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some packages may not have installed correctly.
)

echo.
echo ============================================================
echo [SUCCESS] Environment setup complete!
echo ============================================================
echo.
echo You can now run the application using run_gui.bat
echo.
pause
