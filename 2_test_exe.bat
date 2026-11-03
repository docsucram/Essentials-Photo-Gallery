@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo  Step 2: Testing Standalone EssentialsPhotoGallery.exe
echo =======================================================

if not exist "dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe" (
    echo [ERROR] "dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe" not found!
    echo Please run "1_build_exe.bat" first.
    pause
    exit /b 1
)

echo Starting dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe ...
start "" "dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe"
