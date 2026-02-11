"""
Akashic Records MCP Server - HTTP/SSE Mode (Official MCP Protocol)

Uses the official mcp library with SseServerTransport for Claude Desktop compatibility.

Usage:
    python src/mcp_server_sse.py

Environment Variables:
    MCP_HOST: Bind address (default: 0.0.0.0)
    MCP_PORT: Port number (default: 8089)
    MCP_API_KEY: Authentication key (default: akashic-secret-key)
    COMPILE_COMMANDS_PATH: Path to compile_commands.json (default: ./data/compile_commands.json)
"""
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code_store import CodeStore
from semantic_analyzer import ClangdAnalyzer
try:
    from runtime_config import load_dotenv, resolve_settings_path
except ImportError:  # pragma: no cover - package import fallback
    from .runtime_config import load_dotenv, resolve_settings_path

load_dotenv()

# Configuration
HOST = os.getenv("MCP_HOST", os.getenv("AKASHIC_SERVER_HOST", "0.0.0.0"))
PORT = int(os.getenv("MCP_PORT", os.getenv("AKASHIC_SERVER_PORT", "8089")))
API_KEY = os.getenv("MCP_API_KEY", "akashic-secret-key")
COMPILE_COMMANDS_PATH = os.getenv(
    "COMPILE_COMMANDS_PATH",
    str(Path(__file__).parent.parent / "data" / "compile_commands.json"),
)

# Initialize MCP server
mcp_server = Server("akashic-records")

# SSE transport
sse_transport = SseServerTransport("/messages")

# Global instances
_code_store: CodeStore | None = None
_clangd_analyzer: ClangdAnalyzer | None = None


async def get_code_store() -> CodeStore:
    """Get or create CodeStore instance"""
    global _code_store
    if _code_store is None:
        settings_path = resolve_settings_path()
        _code_store = CodeStore(str(settings_path))
        await _code_store.connect()
    return _code_store


async def get_clangd_analyzer() -> ClangdAnalyzer | None:
    """Get or create ClangdAnalyzer instance"""
    global _clangd_analyzer
    if _clangd_analyzer is None:
        compile_commands_path = Path(COMPILE_COMMANDS_PATH)
        if not compile_commands_path.is_absolute():
            compile_commands_path = (Path(__file__).parent.parent / compile_commands_path).resolve()

        if compile_commands_path.exists():
            _clangd_analyzer = ClangdAnalyzer(compile_commands_path=compile_commands_path)
            if await _clangd_analyzer.verify_installation():
                print(f"ClangdAnalyzer initialized: {compile_commands_path}")
            else:
                _clangd_analyzer = None
    return _clangd_analyzer


# =============================================================================
# MCP Tool Definitions
# =============================================================================

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_code",
            description="Search code using natural language query with semantic vector search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "language": {"type": "string", "enum": ["csharp", "cpp", "python"], "description": "Filter by language"},
                    "limit": {"type": "integer", "default": 10, "description": "Maximum results"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_symbol",
            description="Find symbol definition by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name"},
                    "symbol_type": {"type": "string", "description": "Type filter (class, function, etc.)"}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="get_file_context",
            description="Get code context from a specific file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "start_line": {"type": "integer", "description": "Start line number"},
                    "end_line": {"type": "integer", "description": "End line number"}
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="list_indexed_files",
            description="List all indexed files with optional pattern filter",
            inputSchema={
                "type": "object",
                "properties": {
                    "path_pattern": {"type": "string", "description": "Glob pattern to filter files"},
                    "language": {"type": "string", "description": "Language filter"}
                }
            }
        ),
        Tool(
            name="get_cpp_symbols",
            description="Search for C++ symbols by name using clangd semantic analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name to search"},
                    "file_path": {"type": "string", "description": "Specific file to search in"},
                    "kind": {"type": "string", "enum": ["function", "class", "method", "struct", "enum", "field", "variable"]}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="find_cpp_references",
            description="Find all references to a symbol at a specific position using clangd LSP",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to C++ source file"},
                    "line": {"type": "integer", "description": "Line number (1-indexed)"},
                    "column": {"type": "integer", "description": "Column number (0-indexed)"}
                },
                "required": ["file_path", "line", "column"]
            }
        ),
        Tool(
            name="go_to_definition",
            description="Go to definition of symbol at position using clangd LSP",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to C++ source file"},
                    "line": {"type": "integer", "description": "Line number (1-indexed)"},
                    "column": {"type": "integer", "description": "Column number (0-indexed)"}
                },
                "required": ["file_path", "line", "column"]
            }
        ),
        Tool(
            name="analyze_cpp_file",
            description="Extract all symbols from a C++ file using clangd",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to C++ file"}
                },
                "required": ["file_path"]
            }
        )
    ]


# =============================================================================
# MCP Tool Handlers
# =============================================================================

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    try:
        result = await dispatch_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def dispatch_tool(name: str, args: dict) -> dict:
    """Dispatch tool call to appropriate handler"""

    if name == "search_code":
        store = await get_code_store()
        # Ensure limit is an integer
        limit = args.get("limit", 10)
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

        results = await store.search(
            args.get("query", ""),
            limit=limit,
            language=args.get("language")
        )
        return {
            "results": [
                {
                    "file_path": r.chunk.file_path,
                    "symbol_name": r.chunk.symbol_name,
                    "symbol_type": r.chunk.symbol_type,
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "language": r.chunk.language,
                    "code": r.chunk.code,
                    "score": r.score,
                    "rerank_score": r.rerank_score
                }
                for r in results
            ],
            "count": len(results)
        }

    elif name == "get_symbol":
        store = await get_code_store()
        result = await store.get_symbol(args.get("name", ""), args.get("symbol_type"))
        if result:
            return {
                "found": True,
                "symbol": {
                    "name": result.name,
                    "type": result.type,
                    "file_path": result.file_path,
                    "start_line": result.start_line,
                    "code": result.code
                }
            }
        return {"found": False, "symbol": None}

    elif name == "get_file_context":
        file_path = args.get("file_path", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        # Normalize path: handle both / and \ separators
        file_path = file_path.replace("\\", "/")

        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        if start_line and end_line:
            lines = lines[max(0, start_line-1):end_line]

        return {
            "file_path": file_path,
            "content": "".join(lines),
            "total_lines": len(lines)
        }

    elif name == "list_indexed_files":
        store = await get_code_store()
        files = await store.list_files(
            path_pattern=args.get("path_pattern"),
            language=args.get("language")
        )
        return {"files": files, "count": len(files)}

    elif name == "get_cpp_symbols":
        analyzer = await get_clangd_analyzer()
        if not analyzer:
            return {"error": "clangd not available"}

        # Normalize file_path if provided
        search_file_path = args.get("file_path")
        if search_file_path:
            search_file_path = search_file_path.replace("\\", "/")

        symbols = await analyzer.search_symbols(
            args.get("name", ""),
            file_path=search_file_path,
            kind=args.get("kind")
        )
        return {
            "symbols": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "column_start": s.column_start,
                    "signature": s.signature,
                    "parent_name": s.parent_name
                }
                for s in symbols
            ],
            "count": len(symbols)
        }

    elif name == "find_cpp_references":
        analyzer = await get_clangd_analyzer()
        if not analyzer:
            return {"error": "clangd not available"}

        # Normalize path
        ref_file_path = args.get("file_path", "").replace("\\", "/")

        refs = await analyzer.find_references(
            ref_file_path,
            args.get("line", 1),
            args.get("column", 0)
        )

        # Enrich with code context
        results = []
        for ref in refs:
            context_lines = []
            try:
                with open(ref.file_path, 'r', encoding='utf-8', errors='replace') as f:
                    all_lines = f.readlines()
                    start = max(0, ref.line - 3)
                    end = min(len(all_lines), ref.line + 2)
                    context_lines = all_lines[start:end]
            except:
                pass

            results.append({
                "file_path": ref.file_path,
                "line": ref.line,
                "column": ref.column,
                "kind": ref.kind,
                "code_context": "".join(context_lines)
            })

        return {"references": results, "count": len(results)}

    elif name == "go_to_definition":
        analyzer = await get_clangd_analyzer()
        if not analyzer:
            return {"error": "clangd not available"}

        # Normalize path
        def_file_path = args.get("file_path", "").replace("\\", "/")

        definition = await analyzer.go_to_definition(
            def_file_path,
            args.get("line", 1),
            args.get("column", 0)
        )

        if not definition:
            return {"found": False, "definition": None}

        # Enrich with code context
        context_lines = []
        try:
            with open(definition.file_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
                start = max(0, definition.line - 3)
                end = min(len(all_lines), definition.line + 20)
                context_lines = all_lines[start:end]
        except:
            pass

        return {
            "found": True,
            "definition": {
                "file_path": definition.file_path,
                "line": definition.line,
                "column": definition.column,
                "code_context": "".join(context_lines)
            }
        }

    elif name == "analyze_cpp_file":
        analyzer = await get_clangd_analyzer()
        if not analyzer:
            return {"error": "clangd not available"}

        # Normalize path
        analyze_file_path = args.get("file_path", "").replace("\\", "/")

        symbols, references = await analyzer.analyze_file(analyze_file_path)
        return {
            "symbols": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "column_start": s.column_start,
                    "signature": s.signature,
                    "parent_name": s.parent_name
                }
                for s in symbols
            ],
            "count": len(symbols)
        }

    else:
        return {"error": f"Unknown tool: {name}"}


# =============================================================================
# HTTP Handlers
# =============================================================================

async def handle_sse(request: Request):
    """Handle SSE connection for MCP"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_server.run(
            streams[0], streams[1], mcp_server.create_initialization_options()
        )

    # Return empty response after SSE connection closes
    return JSONResponse({"status": "disconnected"})


async def handle_messages(request: Request):
    """Handle POST messages for MCP"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    await sse_transport.handle_post_message(request.scope, request.receive, request._send)

    # Return accepted response
    return JSONResponse({"status": "accepted"}, status_code=202)


async def health_check(request: Request):
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "akashic-records",
        "version": "1.0.0"
    })


async def server_info(request: Request):
    """Server info endpoint"""
    return JSONResponse({
        "name": "Akashic Records MCP Server",
        "version": "1.0.0",
        "description": "AI-optimized code search for large codebases",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages",
            "health": "/health"
        },
        "tools": [
            "search_code", "get_symbol", "get_file_context", "list_indexed_files",
            "get_cpp_symbols", "find_cpp_references", "go_to_definition", "analyze_cpp_file"
        ]
    })


# Create Starlette app with CORS
app = Starlette(
    debug=False,
    routes=[
        Route("/", server_info),
        Route("/health", health_check),
        Route("/sse", handle_sse),
        Route("/messages", handle_messages, methods=["POST"]),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)


if __name__ == "__main__":
    import uvicorn

    print(f"""
================================================================
       Akashic Records MCP Server (HTTP/SSE Mode)
================================================================
  Host: {HOST}
  Port: {PORT}
  API Key: {API_KEY}
----------------------------------------------------------------
  Endpoints:
    - GET  /        : Server info
    - GET  /health  : Health check
    - GET  /sse     : SSE connection (MCP)
    - POST /messages: Message handling (MCP)
----------------------------------------------------------------
  Claude Desktop config:
    "akashic": {{
      "type": "sse",
      "url": "http://{HOST}:{PORT}/sse",
      "headers": {{
        "Authorization": "Bearer {API_KEY}"
      }}
    }}
================================================================
""")

    uvicorn.run(app, host=HOST, port=PORT)
