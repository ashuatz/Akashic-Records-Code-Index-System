@echo off
chcp 65001 >nul
REM ============================================
REM [06] Physics Systems
REM - Physics, Cloth, Vehicles
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [06] Physics Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/4] Indexing Modules/Physics...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Physics" --skip-errors

echo [2/4] Indexing Modules/Physics2D...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Physics2D" --skip-errors

echo [3/4] Indexing Modules/Cloth...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Cloth" --skip-errors

echo [4/4] Indexing Modules/Vehicles...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Vehicles" --skip-errors

echo.
echo [06] Physics Complete!
echo %date% %time% > "scripts\indexing\status\06_physics.done"
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

