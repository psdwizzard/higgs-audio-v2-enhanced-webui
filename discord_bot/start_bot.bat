@echo off
title Higgs Audio Discord Bot
cd /d "%~dp0.."
echo ============================================
echo  Higgs Audio Discord Bot
echo ============================================
echo.
echo Working directory: %CD%

set VENV_PYTHON=discord_bot\bot_venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Please run bot_install.bat first.
    pause
    exit /b 1
)

echo Using: %VENV_PYTHON%
echo.

:start
echo Starting bot...
"%VENV_PYTHON%" run_bot.py

echo.
echo Bot exited with code %ERRORLEVEL%.
echo Restarting in 5 seconds... (Ctrl+C to stop)
timeout /t 5 /nobreak >nul
goto start
