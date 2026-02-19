@echo off
setlocal EnableDelayedExpansion
title Higgs Audio Discord Bot - Setup
color 0A
cd /d "%~dp0.."
echo ============================================
echo   Higgs Audio Discord Bot - Setup Wizard
echo ============================================
echo.
echo This will create your .env file with the
echo credentials needed to run the bot.
echo.
echo Your keys will be saved locally and never
echo sent anywhere.
echo.
echo ============================================
echo.

REM Use venv python if available, otherwise system python
set BOT_PYTHON=discord_bot\bot_venv\Scripts\python.exe
if not exist "%BOT_PYTHON%" set BOT_PYTHON=python

REM Check if .env already exists
if exist ".env" (
    echo WARNING: A .env file already exists!
    echo.
    set /p OVERWRITE="Overwrite it? (y/n): "
    if /i not "!OVERWRITE!"=="y" (
        echo.
        echo Setup cancelled. Your existing .env was not changed.
        pause
        exit /b
    )
    echo.
)

echo --- Step 1 of 3: Discord Bot Token ---
echo.
echo Go to https://discord.com/developers/applications
echo Select your app ^> "Bot" in sidebar ^> Click "Reset Token"
echo.
echo IMPORTANT: This is the BOT token, NOT the Client Secret.
echo It usually starts with a long base64 string with dots.
echo.
set /p BOT_TOKEN="Paste your bot token here: "
echo.

echo --- Step 2 of 3: Guild (Server) ID ---
echo.
echo In Discord: Settings ^> Advanced ^> Enable Developer Mode
echo Then right-click your server name ^> Copy Server ID
echo.
echo You can enter multiple IDs separated by commas.
echo.
set /p GUILD_IDS="Paste your server ID(s): "
echo.

echo --- Step 3 of 3: Admin User ID ---
echo.
echo Right-click your own username in Discord ^> Copy User ID
echo This controls who can run /audiobook and /cancel.
echo.
echo You can enter multiple IDs separated by commas.
echo.
set /p ADMIN_USERS="Paste your user ID(s): "
echo.

REM Use Python to write the file cleanly (avoids batch trailing-space issues)
"%BOT_PYTHON%" -c "t='!BOT_TOKEN!';g='!GUILD_IDS!';a='!ADMIN_USERS!';f=open('.env','w');f.write(f'DISCORD_BOT_TOKEN={t.strip()}\nGUILD_IDS={g.strip()}\nADMIN_USERS={a.strip()}\n');f.close();print('Written OK')"
if errorlevel 1 (
    echo ERROR: Failed to write .env file.
    pause
    exit /b 1
)

echo.
echo ============================================
echo.
echo  .env file created successfully!
echo.
echo  You can now run the bot with:
echo    start_bot.bat
echo.
echo  To change these later, just run this
echo  setup wizard again or edit .env directly.
echo.
echo ============================================
pause
