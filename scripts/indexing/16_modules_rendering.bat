@echo off
chcp 65001 >nul
REM ============================================
REM [16] Modules - Rendering Related
REM - GI, Particles, VFX
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [16] Modules - Rendering
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Modules/GI...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\GI" --skip-errors

echo [2/5] Indexing Modules/ParticleSystem...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\ParticleSystem" --skip-errors

echo [3/5] Indexing Modules/VFX...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\VFX" --skip-errors

echo [4/5] Indexing Modules/SpriteMask...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\SpriteMask" --skip-errors

echo [5/5] Indexing Modules/SpriteShape...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\SpriteShape" --skip-errors

echo.
echo [16] Modules Rendering Complete!
echo %date% %time% > "scripts\indexing\status\16_modules_rendering.done"
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

