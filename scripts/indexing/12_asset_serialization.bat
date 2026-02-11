@echo off
chcp 65001 >nul
REM ============================================
REM [12] Asset & Serialization
REM - AssetBundle, Serialize, Resources
REM ============================================

cd /d "%~dp0\..\.."

if "%UNITY_SOURCE_ROOT%"=="" (
    echo [ERROR] UNITY_SOURCE_ROOT is not set.
    echo Set it first: set UNITY_SOURCE_ROOT=D:\path\to\unity
    exit /b 1
)


echo ============================================
echo [12] Asset & Serialization
echo ============================================

call :check_services
if errorlevel 1 exit /b 1

echo [1/5] Indexing Modules/AssetBundle...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\AssetBundle" --skip-errors

echo [2/5] Indexing Modules/AssetDatabase...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Modules\AssetDatabase" --skip-errors

echo [3/5] Indexing Runtime/Serialize...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Serialize" --skip-errors

echo [4/5] Indexing Runtime/Resources...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Resources" --skip-errors

echo [5/5] Indexing Runtime/Streaming...
python scripts/ingest.py ingest --path "%UNITY_SOURCE_ROOT%\Runtime\Streaming" --skip-errors

echo.
echo [12] Asset & Serialization Complete!
echo %date% %time% > "scripts\indexing\status\12_asset_serialization.done"
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

