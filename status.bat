@echo off
setlocal
title Akashic Records - Service Status (Docker)

cd /d "%~dp0"

echo ============================================================
echo   Akashic Records - Docker Service Status
echo ============================================================
echo.

docker compose ps
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Unable to query docker compose status.
    exit /b 1
)

echo.
echo Health checks:
curl -fsS http://localhost:8088/health >nul 2>&1
if %errorlevel% equ 0 (
    echo - Akashic API: RUNNING
) else (
    echo - Akashic API: NOT RESPONDING
)

curl -fsS http://localhost:6333/collections >nul 2>&1
if %errorlevel% equ 0 (
    echo - Qdrant: RUNNING
) else (
    echo - Qdrant: NOT RESPONDING
)

endlocal
