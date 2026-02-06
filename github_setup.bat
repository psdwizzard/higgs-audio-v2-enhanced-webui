@echo off
echo ========================================
echo  GitHub Repository Setup for psdwizzard
echo ========================================
echo.
echo This script will help you complete the setup of your new repository:
echo https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui
echo.

echo [STEP 1] Create the repository on GitHub:
echo   1. Go to: https://github.com/new
echo   2. Repository name: higgs-audio-v2-enhanced-webui
echo   3. Description: 🎵 Professional Higgs Audio v2 WebUI with multi-speaker generation, volume normalization, and advanced features
echo   4. Make it Public (recommended)
echo   5. Don't initialize with README (we have everything ready)
echo   6. Click "Create repository"
echo.

set /p created="Have you created the repository on GitHub? (y/N): "
if /i not "%created%"=="y" (
    echo Please create the repository first, then run this script again.
    echo Opening GitHub in your browser...
    start https://github.com/new
    pause
    exit /b
)

echo.
echo [STEP 2] Pushing your enhanced code to GitHub...
echo.

git branch -M main
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] 🎉 Repository created successfully!
    echo.
    echo Your enhanced Higgs Audio v2 WebUI is now live at:
    echo https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui
    echo.
    echo Features included:
    echo ✅ Multi-speaker generation with unlimited speakers
    echo ✅ Professional volume normalization system
    echo ✅ Advanced generation parameters exposed
    echo ✅ Enhanced voice library with per-voice settings
    echo ✅ Public sharing capabilities
    echo ✅ Cache management tools
    echo ✅ Comprehensive documentation
    echo.
    echo Next steps:
    echo 1. Visit your repository to see the README
    echo 2. Share with the community
    echo 3. Create releases for major updates
    echo 4. Welcome contributors through issues
    echo.
    start https://github.com/psdwizzard/higgs-audio-v2-enhanced-webui
) else (
    echo.
    echo [ERROR] Failed to push to repository.
    echo Make sure:
    echo 1. You created the repository on GitHub
    echo 2. You're logged into GitHub
    echo 3. The repository name is correct
    echo.
    echo You can try pushing manually with:
    echo git push -u origin main
)

echo.
pause 