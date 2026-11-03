@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo  Step 3: Compiling Installer with Inno Setup
echo =======================================================

if not exist "dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe" (
    echo [ERROR] "dist\EssentialsPhotoGallery\EssentialsPhotoGallery.exe" not found!
    echo Please run 1_build_exe.bat before building the installer.
    echo.
    pause
    exit /b 1
)

set "ISCC_EXE="

if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\iscc.exe" (
    set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\iscc.exe"
    goto :compile
)

if exist "C:\Program Files\Inno Setup 6\iscc.exe" (
    set "ISCC_EXE=C:\Program Files\Inno Setup 6\iscc.exe"
    goto :compile
)

if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
    set "ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\iscc.exe"
    goto :compile
)

where iscc >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "ISCC_EXE=iscc"
    goto :compile
)

echo [ERROR] Could not locate Inno Setup compiler iscc.exe.
echo Checked:
echo   - %LOCALAPPDATA%\Programs\Inno Setup 6\iscc.exe
echo   - C:\Program Files\Inno Setup 6\iscc.exe
echo   - C:\Program Files (x86)\Inno Setup 6\iscc.exe
echo   - System PATH
echo.
pause
exit /b 1

:compile
echo Found Inno Setup Compiler:
echo "%ISCC_EXE%"
echo.
echo Compiling installer.iss ...
echo.

"%ISCC_EXE%" "installer.iss"
if errorlevel 1 (
    echo.
    echo [ERROR] Inno Setup compilation failed!
    echo.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [SUCCESS] Installer successfully generated!
echo Location: dist_installer\EssentialsPhotoGallery_Setup_x64.exe
echo =======================================================
echo.
pause
