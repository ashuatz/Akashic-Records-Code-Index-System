@echo off
chcp 65001 >nul
REM ============================================
REM [15] Terrain & 2D
REM - Terrain, 2D, Tilemap, Sprite
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [15] Terrain & 2D
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Modules/Terrain...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\Terrain" --skip-errors

echo [2/5] Indexing Modules/TerrainPhysics...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\TerrainPhysics" --skip-errors

echo [3/5] Indexing Runtime/2D...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\2D" --skip-errors

echo [4/5] Indexing Packages/com.unity.2d.sprite...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.2d.sprite" --skip-errors

echo [5/5] Indexing Packages/com.unity.2d.tilemap...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Packages\com.unity.2d.tilemap" --skip-errors

echo.
echo [15] Terrain & 2D Complete!
echo %date% %time% > "scripts\indexing\status\15_terrain_2d.done"
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

