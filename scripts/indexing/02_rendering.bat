@echo off
chcp 65001 >nul
REM ============================================
REM [02] Rendering Systems
REM - Graphics, GfxDevice, Camera
REM - Shaders, GI
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [02] Rendering Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/6] Indexing Runtime/Graphics...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Graphics" --skip-errors

echo [2/6] Indexing Runtime/GfxDevice...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\GfxDevice" --skip-errors

echo [3/6] Indexing Runtime/Camera...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Camera" --skip-errors

echo [4/6] Indexing Runtime/Shaders...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Shaders" --skip-errors

echo [5/6] Indexing Runtime/GI...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\GI" --skip-errors

echo [6/6] Indexing Shaders (top-level)...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Shaders" --skip-errors

echo.
echo [02] Rendering Complete!
echo %date% %time% > "scripts\indexing\status\02_rendering.done"
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

