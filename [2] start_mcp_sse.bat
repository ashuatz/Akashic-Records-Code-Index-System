@echo off
setlocal
title Akashic MCP SSE Server (Local Python)

cd /d "%~dp0"

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo Created .env from .env.example
)

echo Starting Akashic MCP SSE server...
python src/mcp_server_sse.py

endlocal
