@echo off
setlocal
title Akashic Records - Stop All Services (Docker)

cd /d "%~dp0"

echo ============================================================
echo   Akashic Records - Docker Shutdown
echo ============================================================
echo.

docker compose down
if %errorlevel% neq 0 (
    echo [ERROR] Failed to stop docker services.
    exit /b 1
)

echo Services stopped.
endlocal
