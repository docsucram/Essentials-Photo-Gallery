@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo  Step 1: Building Essentials Photo Gallery (PyInstaller)
echo =======================================================
python build_exe.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Standalone build ready in dist\EssentialsPhotoGallery\
echo.
pause
