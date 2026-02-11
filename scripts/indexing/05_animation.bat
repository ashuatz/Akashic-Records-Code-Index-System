@echo off
chcp 65001 >nul
REM ============================================
REM [05] Animation & Director
REM - Animation Module, Director
REM - Timeline, Playables
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [05] Animation Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/3] Indexing Modules/Animation...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Animation" --skip-errors

echo [2/3] Indexing Modules/Director...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Director" --skip-errors

echo [3/3] Indexing Runtime/Director...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Director" --skip-errors

echo.
echo [05] Animation Complete!
echo %date% %time% > "scripts\indexing\status\05_animation.done"
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

