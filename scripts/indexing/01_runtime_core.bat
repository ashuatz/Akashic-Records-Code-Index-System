@echo off
chcp 65001 >nul
REM ============================================
REM [01] Runtime Core Systems
REM - Core, BaseClasses, Utilities, Math
REM - Containers, Memory, Threads
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [01] Runtime Core Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/6] Indexing Runtime/Core...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Core" --skip-errors

echo [2/6] Indexing Runtime/BaseClasses...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\BaseClasses" --skip-errors

echo [3/6] Indexing Runtime/Utilities...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Utilities" --skip-errors

echo [4/6] Indexing Runtime/Math...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Math" --skip-errors

echo [5/6] Indexing Runtime/Containers...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Containers" --skip-errors

echo [6/6] Indexing Runtime/Threads...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Threads" --skip-errors

echo.
echo [01] Runtime Core Complete!
echo %date% %time% > "scripts\indexing\status\01_runtime_core.done"
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

