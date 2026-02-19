@echo off
setlocal EnableDelayedExpansion
title Higgs Audio Discord Bot - Add Server
color 0E
cd /d "%~dp0.."
echo ============================================
echo   Add Bot to a New Discord Server
echo ============================================
echo.

if not exist ".env" (
    echo ERROR: No .env file found. Run setup_bot.bat first.
    pause
    exit /b 1
)

set BOT_PYTHON=discord_bot\bot_venv\Scripts\python.exe
if not exist "%BOT_PYTHON%" set BOT_PYTHON=python

REM Extract Application ID from the bot token automatically
for /f "delims=" %%a in ('"%BOT_PYTHON%" discord_bot\utils\get_app_id.py 2^>nul') do set APP_ID=%%a

if "%APP_ID%"=="" (
    echo ERROR: Could not read bot token from .env
    pause
    exit /b 1
)

echo ============================================
echo.
echo   1. Open this link in your browser:
echo.
echo   https://discord.com/oauth2/authorize?client_id=%APP_ID%^&scope=bot+applications.commands^&permissions=277025508352
echo.
echo   2. Pick the server and click "Authorize"
echo   3. Come back here when done
echo.
echo ============================================
echo.

set /p GUILD_ID="Paste the new Server ID (right-click server > Copy Server ID): "
echo.

REM Append the guild ID to GUILD_IDS in .env
"%BOT_PYTHON%" -c "import re;gid='!GUILD_ID!'.strip();lines=open('.env').read();m=re.search(r'GUILD_IDS=(.*)',lines);old=m.group(1).strip() if m else '';new_ids=old+','+gid if old else gid;lines=re.sub(r'GUILD_IDS=.*','GUILD_IDS='+new_ids,lines) if m else lines+'\nGUILD_IDS='+gid+'\n';open('.env','w').write(lines);print(f'GUILD_IDS is now: {new_ids}')"

echo.
echo [OK] Server added! Restart the bot for slash
echo      commands to sync to the new server.
echo.
echo      Admins on that server (anyone with the
echo      Administrator permission) can use
echo      /audiobook and /cancel automatically.
echo.
pause
