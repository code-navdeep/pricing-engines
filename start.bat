@echo off
title Pricing Engines

REM Change to script directory
cd /d "%~dp0"

echo Starting Pricing Engines...
echo ======================================

REM Check that Docker is installed
where docker >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed.
    echo Install Docker Desktop from https://www.docker.com/products/docker-desktop and try again.
    pause
    exit /b 1
)

REM Check that Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Docker is running
echo Building and starting both calculators...
echo.

REM Start both calculators (futures and options)
docker-compose up -d --build
if errorlevel 1 (
    echo.
    echo FAILED: the calculators did not start. See the messages above.
    pause
    exit /b 1
)

REM Wait for apps to be ready
echo Waiting for calculators to start...
timeout /t 6 /nobreak >nul

REM Open both calculators in the browser
start http://localhost:8050
timeout /t 1 /nobreak >nul
start http://localhost:8051

echo.
echo Both calculators are running!
echo Futures Calculator: http://localhost:8050
echo Options Calculator: http://localhost:8051
echo.
echo To stop: run stop.bat
echo ======================================
echo.
pause
