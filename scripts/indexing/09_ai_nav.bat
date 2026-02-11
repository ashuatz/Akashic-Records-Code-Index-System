@echo off
chcp 65001 >nul
REM ============================================
REM [09] AI & Navigation
REM - AI, NavMesh, Pathfinding
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [09] AI & Navigation
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/3] Indexing Modules/AI...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\AI" --skip-errors

echo [2/3] Indexing Modules/AIEditor...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\AIEditor" --skip-errors

echo [3/3] Indexing Modules/NavMeshRuntime (if exists)...
if exist "%UNITY_SOURCE_ROOT%\Modules\NavMeshRuntime" (
    python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\NavMeshRuntime" --skip-errors
)

echo.
echo [09] AI & Navigation Complete!
echo %date% %time% > "scripts\indexing\status\09_ai_nav.done"
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

