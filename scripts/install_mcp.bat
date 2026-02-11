@echo off
REM Akashic Records MCP Server - Installation Script for Claude Code
REM Usage: install_mcp.bat [--user|--local|--project]

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set SERVER_PATH=%PROJECT_DIR%\src\mcp_server.py
set HTTP_SERVER_PATH=%PROJECT_DIR%\src\mcp_server_http.py
set SCOPE=user

REM Parse arguments
:parse_args
if "%~1"=="" goto :main
if "%~1"=="--user" set SCOPE=user
if "%~1"=="--local" set SCOPE=local
if "%~1"=="--project" set SCOPE=project
if "%~1"=="-s" (
    set SCOPE=%~2
    shift
)
shift
goto :parse_args

:main
echo.
echo ============================================================
echo   Akashic Records MCP Server - Installation
echo ============================================================
echo.
echo Scope: %SCOPE%
echo Server: %SERVER_PATH%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Check if claude CLI is available
claude --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Claude CLI is not installed
    echo Please install: npm install -g @anthropic-ai/claude-code
    exit /b 1
)

REM Remove existing if present
echo Removing existing 'akashic' server if present...
claude mcp remove akashic -s %SCOPE% >nul 2>&1

REM Add stdio mode MCP server
echo.
echo Adding Akashic MCP Server (stdio mode)...
claude mcp add -s %SCOPE% akashic -- python "%SERVER_PATH%"

if errorlevel 1 (
    echo [ERROR] Failed to add MCP server
    exit /b 1
)

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo MCP Server 'akashic' has been added to Claude Code.
echo.
echo To verify: claude mcp list
echo To remove: claude mcp remove akashic -s %SCOPE%
echo.
echo Available tools:
echo   - search_code: Natural language code search
echo   - get_symbol: Find symbol definitions
echo   - get_file_context: Get file content
echo   - list_indexed_files: List indexed files
echo   - get_references: Find symbol references
echo   - get_dependencies: Get dependency graph
echo.

pause
