@echo off
chcp 65001 >nul
REM ============================================
REM [13] Editor Core
REM - Editor main folder
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [13] Editor Core
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/1] Indexing Editor folder...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Editor" --skip-errors

echo.
echo [13] Editor Complete!
echo %date% %time% > "scripts\indexing\status\13_editor.done"
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

