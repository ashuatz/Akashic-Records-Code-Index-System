@echo off
chcp 65001 >nul
REM ============================================
REM Unity Rendering Code Indexing
REM - URP, HDRP, SRP Core
REM - Camera, Culling, Graphics
REM - Shaders
REM ============================================

cd /d "%~dp0"

echo ============================================
echo Unity Rendering Indexing Script
echo ============================================
echo.

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

echo [1/5] Indexing SRP Core (Render Pipeline Core)...
python scripts/ingest.py ingest --path "C:\TA\unity\Packages\com.unity.render-pipelines.core" --skip-errors

echo [2/5] Indexing URP (Universal Render Pipeline)...
python scripts/ingest.py ingest --path "C:\TA\unity\Packages\com.unity.render-pipelines.universal" --skip-errors

echo [3/5] Indexing Runtime Camera + Graphics...
python scripts/ingest.py ingest --path "C:\TA\unity\Runtime\Camera" --skip-errors
python scripts/ingest.py ingest --path "C:\TA\unity\Runtime\Graphics" --skip-errors
python scripts/ingest.py ingest --path "C:\TA\unity\Runtime\GfxDevice" --skip-errors
python scripts/ingest.py ingest --path "C:\TA\unity\Runtime\GI" --skip-errors

echo [4/5] Indexing Modules/Rendering...
python scripts/ingest.py ingest --path "C:\TA\unity\Modules\Rendering" --skip-errors

echo [5/5] Indexing Shaders...
python scripts/ingest.py ingest --path "C:\TA\unity\Shaders" --skip-errors
python scripts/ingest.py ingest --path "C:\TA\unity\Runtime\Shaders" --skip-errors

echo.
echo ============================================
echo Rendering Indexing Complete!
echo ============================================
echo.

python -c "import sqlite3; conn=sqlite3.connect('data/metadata.db'); cur=conn.cursor(); cur.execute('SELECT COUNT(*) FROM chunks'); chunks=cur.fetchone()[0]; cur.execute('SELECT COUNT(*) FROM files'); files=cur.fetchone()[0]; print(f'Total indexed: {files} files, {chunks} chunks')"

echo.
pause
