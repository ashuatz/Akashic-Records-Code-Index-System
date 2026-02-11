@echo off
chcp 65001 >nul
REM ============================================
REM [10] Input & XR
REM - Input, XR, VR, AR
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [10] Input & XR
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Runtime/Input...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Input" --skip-errors

echo [2/5] Indexing Modules/XR...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\XR" --skip-errors

echo [3/5] Indexing Modules/VR...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\VR" --skip-errors

echo [4/5] Indexing Modules/AR...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\AR" --skip-errors

echo [5/5] Indexing Modules/InputLegacy...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\InputLegacy" --skip-errors

echo.
echo [10] Input & XR Complete!
echo %date% %time% > "scripts\indexing\status\10_input_xr.done"
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

