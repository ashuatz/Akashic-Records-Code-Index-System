@echo off
chcp 65001 >nul
REM ============================================
REM [07] UI Systems
REM - IMGUI, UIElements, TextCore
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [07] UI Systems
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Modules/IMGUI...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\IMGUI" --skip-errors

echo [2/5] Indexing Modules/UIElements...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\UIElements" --skip-errors

echo [3/5] Indexing Modules/UIElementsNative...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\UIElementsNative" --skip-errors

echo [4/5] Indexing Modules/TextCore...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\TextCore" --skip-errors

echo [5/5] Indexing Modules/TextCoreFontEngine...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\TextCoreFontEngine" --skip-errors

echo.
echo [07] UI Systems Complete!
echo %date% %time% > "scripts\indexing\status\07_ui.done"
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

