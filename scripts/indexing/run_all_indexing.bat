@echo off
REM ============================================
REM Unity Codebase Full Indexing Script
REM Runs all 16 partitions sequentially
REM ============================================

echo ============================================
echo Unity Codebase Full Indexing
echo ============================================
echo.

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)

REM Check services
echo Checking services...
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Qdrant is not running! Start it first.
    pause
    exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not running! Start it first.
    pause
    exit /b 1
)

echo [OK] Services are running.
echo.

REM Run each partition
set PARTITIONS=01_runtime_core 02_rendering 03_srp_packages 04_scripting 05_animation 06_physics 07_ui 08_audio 09_ai_nav 10_input_xr 11_ecs_jobs 12_asset_serialization 13_editor 14_networking 15_terrain_2d 16_modules_rendering

for %%P in (%PARTITIONS%) do (
    echo.
    echo ============================================
    echo Running: %%P
    echo ============================================

    if exist "scripts\indexing\status\%%P.done" (
        echo [SKIP] Already completed.
    ) else (
        call "scripts\indexing\%%P.bat"
        if errorlevel 1 (
            echo [WARNING] %%P had errors, continuing...
        ) else (
            echo [DONE] %%P completed.
        )
    )
)

echo.
echo ============================================
echo All partitions processed!
echo ============================================
echo.

REM Show final stats
sqlite3 data\metadata.db "SELECT COUNT(DISTINCT file_path) as files, COUNT(*) as chunks FROM chunks;"

pause

