@echo off
REM ============================================
REM Unity Codebase Parallel Indexing
REM Runs multiple partitions in parallel
REM WARNING: Requires 64GB+ RAM
REM ============================================

echo ============================================
echo Unity Codebase Parallel Indexing
echo WARNING: This requires 64GB+ RAM!
echo ============================================
echo.

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)

REM Group 1: Runtime Core (run first, largest)
echo Starting Group 1: Core systems...
if not exist "scripts\indexing\status\01_runtime_core.done" (
    start "Indexing: Runtime Core" cmd /c "scripts\indexing\01_runtime_core.bat"
)
if not exist "scripts\indexing\status\02_rendering.done" (
    start "Indexing: Rendering" cmd /c "scripts\indexing\02_rendering.bat"
)
if not exist "scripts\indexing\status\03_srp_packages.done" (
    start "Indexing: SRP Packages" cmd /c "scripts\indexing\03_srp_packages.bat"
)

echo Waiting for Group 1 to complete...
:wait_group1
timeout /t 30 /nobreak >nul
if not exist "scripts\indexing\status\01_runtime_core.done" goto wait_group1
if not exist "scripts\indexing\status\02_rendering.done" goto wait_group1
if not exist "scripts\indexing\status\03_srp_packages.done" goto wait_group1

REM Group 2: Scripting and Animation
echo Starting Group 2: Scripting and Animation...
if not exist "scripts\indexing\status\04_scripting.done" (
    start "Indexing: Scripting" cmd /c "scripts\indexing\04_scripting.bat"
)
if not exist "scripts\indexing\status\05_animation.done" (
    start "Indexing: Animation" cmd /c "scripts\indexing\05_animation.bat"
)
if not exist "scripts\indexing\status\06_physics.done" (
    start "Indexing: Physics" cmd /c "scripts\indexing\06_physics.bat"
)

echo Waiting for Group 2 to complete...
:wait_group2
timeout /t 30 /nobreak >nul
if not exist "scripts\indexing\status\04_scripting.done" goto wait_group2
if not exist "scripts\indexing\status\05_animation.done" goto wait_group2
if not exist "scripts\indexing\status\06_physics.done" goto wait_group2

REM Group 3: UI, Audio, AI
echo Starting Group 3: UI, Audio, AI...
if not exist "scripts\indexing\status\07_ui.done" (
    start "Indexing: UI" cmd /c "scripts\indexing\07_ui.bat"
)
if not exist "scripts\indexing\status\08_audio.done" (
    start "Indexing: Audio" cmd /c "scripts\indexing\08_audio.bat"
)
if not exist "scripts\indexing\status\09_ai_nav.done" (
    start "Indexing: AI Nav" cmd /c "scripts\indexing\09_ai_nav.bat"
)

echo Waiting for Group 3 to complete...
:wait_group3
timeout /t 30 /nobreak >nul
if not exist "scripts\indexing\status\07_ui.done" goto wait_group3
if not exist "scripts\indexing\status\08_audio.done" goto wait_group3
if not exist "scripts\indexing\status\09_ai_nav.done" goto wait_group3

REM Group 4: Input, ECS, Serialization
echo Starting Group 4: Input, ECS, Serialization...
if not exist "scripts\indexing\status\10_input_xr.done" (
    start "Indexing: Input XR" cmd /c "scripts\indexing\10_input_xr.bat"
)
if not exist "scripts\indexing\status\11_ecs_jobs.done" (
    start "Indexing: ECS Jobs" cmd /c "scripts\indexing\11_ecs_jobs.bat"
)
if not exist "scripts\indexing\status\12_asset_serialization.done" (
    start "Indexing: Asset Serialization" cmd /c "scripts\indexing\12_asset_serialization.bat"
)

echo Waiting for Group 4 to complete...
:wait_group4
timeout /t 30 /nobreak >nul
if not exist "scripts\indexing\status\10_input_xr.done" goto wait_group4
if not exist "scripts\indexing\status\11_ecs_jobs.done" goto wait_group4
if not exist "scripts\indexing\status\12_asset_serialization.done" goto wait_group4

REM Group 5: Editor, Networking, Terrain
echo Starting Group 5: Editor, Networking, Terrain...
if not exist "scripts\indexing\status\13_editor.done" (
    start "Indexing: Editor" cmd /c "scripts\indexing\13_editor.bat"
)
if not exist "scripts\indexing\status\14_networking.done" (
    start "Indexing: Networking" cmd /c "scripts\indexing\14_networking.bat"
)
if not exist "scripts\indexing\status\15_terrain_2d.done" (
    start "Indexing: Terrain 2D" cmd /c "scripts\indexing\15_terrain_2d.bat"
)
if not exist "scripts\indexing\status\16_modules_rendering.done" (
    start "Indexing: Modules Rendering" cmd /c "scripts\indexing\16_modules_rendering.bat"
)

echo Waiting for Group 5 to complete...
:wait_group5
timeout /t 30 /nobreak >nul
if not exist "scripts\indexing\status\13_editor.done" goto wait_group5
if not exist "scripts\indexing\status\14_networking.done" goto wait_group5
if not exist "scripts\indexing\status\15_terrain_2d.done" goto wait_group5
if not exist "scripts\indexing\status\16_modules_rendering.done" goto wait_group5

echo.
echo ============================================
echo ALL INDEXING COMPLETE!
echo ============================================

sqlite3 data\metadata.db "SELECT COUNT(DISTINCT file_path) as files, COUNT(*) as chunks FROM chunks;"

pause

