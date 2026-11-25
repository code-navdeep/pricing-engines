@echo off
title Pricing Engines

REM Change to script directory
cd /d "%~dp0"

echo Stopping Pricing Engines...
echo ========================================

REM Stop both calculators' Docker containers
docker-compose down
if errorlevel 1 (
    echo.
    echo FAILED: the calculators were not stopped. Is Docker Desktop running?
    echo.
    pause
    exit /b 1
)

echo.
echo Both calculators stopped
echo Ports 8050 and 8051 freed
echo.
echo You can now safely close this window.
echo.
pause
