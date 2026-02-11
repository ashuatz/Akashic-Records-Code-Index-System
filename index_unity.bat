@echo off
title Unity Codebase Indexing
cd /d C:\TA\AI\AkashicRecords_CodeIndexSystem

echo ============================================================
echo   Akashic Records - Unity Codebase Indexing
echo ============================================================
echo.
echo This will index the Unity codebase at C:\TA\unity
echo Estimated files: ~56,000
echo.
echo Options:
echo   1. Continue indexing (incremental - skip unchanged files)
echo   2. Full re-index (start from scratch)
echo.
set /p choice="Select option (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting incremental indexing...
    python scripts/ingest.py ingest --path "C:\TA\unity" --incremental --skip-errors
) else if "%choice%"=="2" (
    echo.
    echo Starting full indexing...
    python scripts/ingest.py ingest --path "C:\TA\unity" --skip-errors
) else (
    echo Invalid option
)

echo.
pause
