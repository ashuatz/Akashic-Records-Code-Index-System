@echo off
chcp 65001 >nul
REM ============================================
REM [04] Scripting Systems
REM - Scripting, Mono, ScriptingBackend
REM - Export, GameCode
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [04] Scripting Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Runtime/Scripting...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Scripting" --skip-errors

echo [2/5] Indexing Runtime/Mono...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Mono" --skip-errors

echo [3/5] Indexing Runtime/ScriptingBackend...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\ScriptingBackend" --skip-errors

echo [4/5] Indexing Runtime/Export...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Export" --skip-errors

echo [5/5] Indexing Runtime/GameCode...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\GameCode" --skip-errors

echo.
echo [04] Scripting Complete!
echo %date% %time% > "scripts\indexing\status\04_scripting.done"
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

