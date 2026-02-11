#!/bin/bash
# Akashic Records MCP Server - Installation Script for Claude Code
# Usage: ./install_mcp.sh [--user|--local|--project]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_PATH="$PROJECT_DIR/src/mcp_server.py"
SCOPE="user"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            SCOPE="user"
            shift
            ;;
        --local)
            SCOPE="local"
            shift
            ;;
        --project)
            SCOPE="project"
            shift
            ;;
        -s)
            SCOPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "============================================================"
echo "  Akashic Records MCP Server - Installation"
echo "============================================================"
echo ""
echo "Scope: $SCOPE"
echo "Server: $SERVER_PATH"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)

# Check if claude CLI is available
if ! command -v claude &> /dev/null; then
    echo "[ERROR] Claude CLI is not installed"
    echo "Please install: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# Remove existing if present
echo "Removing existing 'akashic' server if present..."
claude mcp remove akashic -s "$SCOPE" 2>/dev/null || true

# Add stdio mode MCP server
echo ""
echo "Adding Akashic MCP Server (stdio mode)..."
claude mcp add -s "$SCOPE" akashic -- "$PYTHON_CMD" "$SERVER_PATH"

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "MCP Server 'akashic' has been added to Claude Code."
echo ""
echo "To verify: claude mcp list"
echo "To remove: claude mcp remove akashic -s $SCOPE"
echo ""
echo "Available tools:"
echo "  - search_code: Natural language code search"
echo "  - get_symbol: Find symbol definitions"
echo "  - get_file_context: Get file content"
echo "  - list_indexed_files: List indexed files"
echo "  - get_references: Find symbol references"
echo "  - get_dependencies: Get dependency graph"
echo ""
