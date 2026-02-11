@echo off
chcp 65001 >nul
REM ============================================
REM [03] Scriptable Render Pipeline Packages
REM - SRP Core, URP, HDRP
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [03] SRP Packages
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/4] Indexing SRP Core...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.render-pipelines.core" --skip-errors

echo [2/4] Indexing URP...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.render-pipelines.universal" --skip-errors

echo [3/4] Indexing HDRP...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.render-pipelines.high-definition" --skip-errors

echo [4/4] Indexing Denoising...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.rendering.denoising" --skip-errors

echo.
echo [03] SRP Packages Complete!
echo %date% %time% > "scripts\indexing\status\03_srp_packages.done"
pause
exit /b 0

:check_services
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Qdrant not running!
    pause
    exit /b 1
)
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not running!
    pause
    exit /b 1
)
echo [OK] Services running.
exit /b 0

