@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo  Full Pipeline: Build Standalone + Build Installer
echo =======================================================

call 1_build_exe.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build step failed. Aborting installer compilation.
    pause
    exit /b %ERRORLEVEL%
)

call 3_build_installer.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Installer step failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [DONE] Full build and packaging process complete!
echo.
pause
