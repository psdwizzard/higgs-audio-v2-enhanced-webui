@echo off
title Dropbox Setup - Higgs Audio Bot
cd /d "%~dp0.."

set VENV_PYTHON=discord_bot\bot_venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found.
    echo Please run bot_install.bat first.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m discord_bot.dropbox_setup

pause
