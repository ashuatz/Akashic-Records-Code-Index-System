@echo off
chcp 65001 >nul
REM ============================================
REM [08] Audio & Video
REM - Audio, Video, DSPGraph
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [08] Audio & Video
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/4] Indexing Modules/Audio...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Audio" --skip-errors

echo [2/4] Indexing Modules/DSPGraph...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\DSPGraph" --skip-errors

echo [3/4] Indexing Modules/Video...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Video" --skip-errors

echo [4/4] Indexing Runtime/Video...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Video" --skip-errors

echo.
echo [08] Audio & Video Complete!
echo %date% %time% > "scripts\indexing\status\08_audio.done"
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

