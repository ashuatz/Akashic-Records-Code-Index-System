@echo off
chcp 65001 >nul
REM ============================================
REM [11] ECS & Jobs
REM - Entities, Jobs, Burst, Collections
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [11] ECS & Jobs
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Runtime/Jobs...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Jobs" --skip-errors

echo [2/5] Indexing Runtime/Burst...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Burst" --skip-errors

echo [3/5] Indexing Packages/com.unity.entities...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.entities" --skip-errors

echo [4/5] Indexing Packages/com.unity.collections...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.collections" --skip-errors

echo [5/5] Indexing Packages/com.unity.mathematics...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.mathematics" --skip-errors

echo.
echo [11] ECS & Jobs Complete!
echo %date% %time% > "scripts\indexing\status\11_ecs_jobs.done"
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

