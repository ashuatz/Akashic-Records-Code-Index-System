@echo off
setlocal
title Akashic Records - Start All Services (Docker)

cd /d "%~dp0"

echo ============================================================
echo   Akashic Records - Docker Startup
echo ============================================================
echo.

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo Created .env from .env.example
    echo.
)

docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] docker compose is not available.
    echo Install Docker Desktop and ensure "docker compose" works.
    exit /b 1
)

echo Starting services (qdrant + akashic)...
docker compose up -d --build
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services.
    exit /b 1
)

echo.
echo Services started.
echo - Akashic API: http://localhost:8088/health
echo - Qdrant:      http://localhost:6333/collections
echo.
echo To include Ollama container:
echo   docker compose --profile ollama up -d
echo.

endlocal
