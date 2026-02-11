"""
Akashic Records MCP Server - HTTP/SSE Mode

MCP server that exposes endpoints over HTTP for network access.
Supports Server-Sent Events (SSE) for real-time communication.

Usage:
    python src/mcp_server_http.py --host 10.14.0.33 --port 8088
"""

import asyncio
import argparse
import json
import logging
import sys
import os
import fnmatch
from pathlib import Path
from typing import Any, Optional, List
from datetime import datetime

from aiohttp import web
import aiohttp_cors

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code_store import CodeStore, SearchResult
from semantic_analyzer import ClangdAnalyzer, SemanticAnalyzer
try:
    from runtime_config import load_dotenv, resolve_settings_path
except ImportError:  # pragma: no cover - package import fallback
    from .runtime_config import load_dotenv, resolve_settings_path

load_dotenv()

# Configure logging
log_dir = Path(__file__).parent.parent / "data"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "akashic_mcp_http.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Global CodeStore instance
_code_store: Optional[CodeStore] = None

# Global Semantic Analyzer instance
_clangd_analyzer: Optional[ClangdAnalyzer] = None

# Default compile_commands.json path
DEFAULT_COMPILE_COMMANDS_PATH = str(Path(__file__).parent.parent / "data" / "compile_commands.json")


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_MCP_HTTP_ROUTES = _env_bool("AKASHIC_ENABLE_MCP_HTTP_ROUTES", True)


async def get_code_store() -> CodeStore:
    """Get or initialize the global CodeStore instance."""
    global _code_store
    if _code_store is None:
        settings_path = resolve_settings_path()
        _code_store = CodeStore(str(settings_path))
        await _code_store.connect()
        logger.info("CodeStore initialized successfully")
    return _code_store


async def get_clangd_analyzer() -> Optional[ClangdAnalyzer]:
    """Get or initialize the global ClangdAnalyzer instance."""
    global _clangd_analyzer
    if _clangd_analyzer is None:
        compile_commands_path = Path(
            os.environ.get("COMPILE_COMMANDS_PATH", DEFAULT_COMPILE_COMMANDS_PATH)
        )
        if not compile_commands_path.is_absolute():
            compile_commands_path = (Path(__file__).parent.parent / compile_commands_path).resolve()

        if compile_commands_path.exists():
            _clangd_analyzer = ClangdAnalyzer(compile_commands_path=compile_commands_path)
            if await _clangd_analyzer.verify_installation():
                logger.info(f"ClangdAnalyzer initialized with compile_commands at: {compile_commands_path}")
            else:
                logger.warning("clangd is not available")
                _clangd_analyzer = None
        else:
            logger.warning(f"compile_commands.json not found at: {compile_commands_path}")
    return _clangd_analyzer


# =============================================================================
# Helper Functions
# =============================================================================

import re

async def _find_first_identifier_column(file_path: str, line: int) -> int:
    """
    Find the column of the first meaningful identifier on a line.
    Used when column is 0 to auto-detect symbol position for references/definition.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return 0

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if line < 1 or line > len(lines):
            return 0

        line_content = lines[line - 1]

        # Pattern to find function/method definitions or declarations
        # Matches: ReturnType ClassName::MethodName or ReturnType FunctionName
        patterns = [
            # Class::Method pattern (e.g., "void ShaderUtil::OpenCompiledShader")
            r'\b(\w+)::(\w+)\s*\(',
            # Function definition (e.g., "void FunctionName(")
            r'\b(?:void|int|bool|float|double|auto|const|static|virtual|inline|\w+\s*\*?)\s+(\w+)\s*\(',
            # Any identifier followed by (
            r'\b(\w+)\s*\(',
            # Any identifier
            r'\b(\w{3,})\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, line_content)
            if match:
                # Get the last group (the actual identifier name)
                for g in reversed(match.groups()):
                    if g:
                        # Find the position of this identifier in the line
                        idx = line_content.find(g)
                        if idx >= 0:
                            return idx
        return 0
    except Exception as e:
        logger.warning(f"Failed to find identifier column: {e}")
        return 0


# =============================================================================
# MCP Tool Handlers
# =============================================================================

async def handle_search_code(params: dict) -> dict:
    """Handle search_code tool call."""
    query = params.get("query", "")
    language = params.get("language")
    limit = params.get("limit", 10)

    # Ensure limit is an integer
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    if not query:
        return {"error": "Query is required"}

    store = await get_code_store()
    results = await store.search(query, limit=limit, language=language)

    formatted_results = []
    for r in results:
        formatted_results.append({
            "file_path": r.chunk.file_path,
            "symbol_name": r.chunk.symbol_name,
            "symbol_type": r.chunk.symbol_type,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "language": r.chunk.language,
            "code": r.chunk.code,
            "score": r.score,
            "rerank_score": r.rerank_score
        })

    return {"results": formatted_results, "count": len(formatted_results)}


async def handle_get_symbol(params: dict) -> dict:
    """Handle get_symbol tool call."""
    name = params.get("name", "")
    symbol_type = params.get("symbol_type")

    if not name:
        return {"error": "Symbol name is required"}

    store = await get_code_store()
    result = await store.get_symbol(name, symbol_type)

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


async def handle_get_file_context(params: dict) -> dict:
    """Handle get_file_context tool call."""
    file_path = params.get("file_path", "")
    start_line = params.get("start_line")
    end_line = params.get("end_line")

    if not file_path:
        return {"error": "File path is required"}

    # Normalize path: handle both / and \ separators
    file_path = file_path.replace("\\", "/")
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        if start_line and end_line:
            # 1-indexed to 0-indexed
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            content = ''.join(lines[start_idx:end_idx])
        else:
            content = ''.join(lines)

        return {
            "file_path": str(path),
            "content": content,
            "total_lines": len(lines)
        }
    except Exception as e:
        return {"error": str(e)}


async def handle_list_indexed_files(params: dict) -> dict:
    """Handle list_indexed_files tool call."""
    path_pattern = params.get("path_pattern", "*")
    language = params.get("language")

    store = await get_code_store()

    cursor = store.db_conn.cursor()

    if language:
        cursor.execute(
            "SELECT DISTINCT file_path FROM files WHERE language = ?",
            (language,)
        )
    else:
        cursor.execute("SELECT DISTINCT file_path FROM files")

    files = [row[0] for row in cursor.fetchall()]

    # Apply glob pattern filter
    if path_pattern and path_pattern != "*":
        files = [f for f in files if fnmatch.fnmatch(f, path_pattern)]

    return {"files": files, "count": len(files)}


async def handle_get_references(params: dict) -> dict:
    """Handle get_references tool call."""
    symbol = params.get("symbol", "")
    file_path = params.get("file_path")

    # Normalize path if provided
    if file_path:
        file_path = file_path.replace("\\", "/")

    if not symbol:
        return {"error": "Symbol name is required"}

    store = await get_code_store()
    cursor = store.db_conn.cursor()

    # Search in code content for symbol references
    if file_path:
        cursor.execute("""
            SELECT file_path, start_line, end_line, code
            FROM chunks
            WHERE code LIKE ? AND file_path = ?
            LIMIT 50
        """, (f'%{symbol}%', file_path))
    else:
        cursor.execute("""
            SELECT file_path, start_line, end_line, code
            FROM chunks
            WHERE code LIKE ?
            LIMIT 50
        """, (f'%{symbol}%',))

    references = []
    for row in cursor.fetchall():
        references.append({
            "file_path": row[0],
            "start_line": row[1],
            "end_line": row[2],
            "context": row[3][:200] + "..." if len(row[3]) > 200 else row[3]
        })

    return {"references": references, "count": len(references)}


async def handle_get_dependencies(params: dict) -> dict:
    """Handle get_dependencies tool call."""
    symbol = params.get("symbol", "")
    direction = params.get("direction", "both")

    if not symbol:
        return {"error": "Symbol name is required"}

    store = await get_code_store()
    cursor = store.db_conn.cursor()

    dependencies = {"outgoing": [], "incoming": []}

    if direction in ["outgoing", "both"]:
        cursor.execute("""
            SELECT to_symbol_id, kind FROM dependencies
            WHERE from_symbol_id IN (SELECT id FROM symbols WHERE name LIKE ?)
        """, (f'%{symbol}%',))
        for row in cursor.fetchall():
            dependencies["outgoing"].append({"symbol": row[0], "kind": row[1]})

    if direction in ["incoming", "both"]:
        cursor.execute("""
            SELECT from_symbol_id, kind FROM dependencies
            WHERE to_symbol_id IN (SELECT id FROM symbols WHERE name LIKE ?)
        """, (f'%{symbol}%',))
        for row in cursor.fetchall():
            dependencies["incoming"].append({"symbol": row[0], "kind": row[1]})

    return dependencies


async def handle_analyze_cpp_file(params: dict) -> dict:
    """Handle analyze_cpp_file tool call - extract symbols using clangd."""
    file_path = params.get("file_path", "")

    if not file_path:
        return {"error": "File path is required"}

    # Normalize path
    file_path = file_path.replace("\\", "/")
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    # Check if it's a C++ file
    cpp_extensions = {'.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx'}
    if path.suffix.lower() not in cpp_extensions:
        return {"error": f"Not a C++ file: {file_path}"}

    analyzer = await get_clangd_analyzer()
    if not analyzer:
        return {"error": "clangd is not available. Make sure clangd is installed and compile_commands.json exists."}

    try:
        symbols, references = await analyzer.analyze_file(file_path)

        return {
            "file_path": file_path,
            "symbols": [s.to_dict() for s in symbols],
            "references": [r.to_dict() for r in references],
            "symbol_count": len(symbols),
            "reference_count": len(references)
        }
    except Exception as e:
        logger.error(f"Failed to analyze file: {e}")
        return {"error": str(e)}


async def handle_get_cpp_symbols(params: dict) -> dict:
    """
    Handle get_cpp_symbols tool call - search for C++ symbols by name.

    Returns ALL occurrences: definitions, declarations, and usages.
    Uses SQLite text search on indexed code chunks.
    """
    name = params.get("name", "")
    file_path = params.get("file_path")
    kind = params.get("kind")  # Not used for text search, kept for API compatibility
    limit = params.get("limit", 50)

    # Ensure limit is an integer
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50

    # Normalize path if provided
    if file_path:
        file_path = file_path.replace("\\", "/")

    if not name:
        return {"error": "Symbol name is required"}

    store = await get_code_store()
    cursor = store.db_conn.cursor()

    results = []

    if file_path:
        # Search in specific file
        query = """
            SELECT file_path, start_line, end_line, code, symbol_name, symbol_type, language
            FROM chunks
            WHERE file_path = ? AND code LIKE ?
            ORDER BY start_line
            LIMIT ?
        """
        cursor.execute(query, (file_path, f'%{name}%', limit))
    else:
        # Search across all C++ files
        # Use word boundary matching for more accurate results
        query = """
            SELECT file_path, start_line, end_line, code, symbol_name, symbol_type, language
            FROM chunks
            WHERE code LIKE ? AND language = 'cpp'
            ORDER BY
                CASE
                    WHEN file_path LIKE '%.h' OR file_path LIKE '%.hpp' THEN 0
                    ELSE 1
                END,
                file_path, start_line
            LIMIT ?
        """
        cursor.execute(query, (f'%{name}%', limit))

    for row in cursor.fetchall():
        fp, start_line, end_line, code, symbol_name, symbol_type, language = row

        # Find exact line numbers where the symbol appears
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines):
            if name in line:
                actual_line = start_line + i

                # Determine usage type from context
                usage_type = "usage"
                line_stripped = line.strip()

                # Check if it's a definition/declaration
                if f"class {name}" in line or f"struct {name}" in line:
                    usage_type = "class_definition"
                elif f"{name}(" in line and ("::" in line or line_stripped.startswith("void") or
                     line_stripped.startswith("int") or line_stripped.startswith("bool") or
                     line_stripped.startswith("auto") or line_stripped.startswith("static")):
                    if "{" in line or ";" in line:
                        usage_type = "function_definition" if "{" in line else "function_declaration"
                elif f"{name} =" in line or f"{name}=" in line:
                    usage_type = "assignment"
                elif f".{name}(" in line or f"->{name}(" in line or f"::{name}(" in line:
                    usage_type = "method_call"
                elif f"{name}(" in line:
                    usage_type = "function_call"

                # Get context (the line itself)
                context = line.strip()[:200]  # Limit context length

                results.append({
                    "name": name,
                    "kind": usage_type,
                    "file_path": fp,
                    "line_start": actual_line,
                    "line_end": actual_line,
                    "column_start": line.find(name),
                    "context": context,
                    "language": language
                })

    # Remove duplicates (same file + line)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["file_path"], r["line_start"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    logger.info(f"Found {len(unique_results)} occurrences of '{name}'")
    return {"symbols": unique_results, "count": len(unique_results)}


async def handle_find_cpp_references(params: dict) -> dict:
    """Handle find_cpp_references tool call - find all references using clangd LSP."""
    file_path = params.get("file_path", "")
    line = params.get("line", 0)
    column = params.get("column", 0)

    # Normalize path
    file_path = file_path.replace("\\", "/")

    if not file_path:
        return {"error": "File path is required"}
    if not line:
        return {"error": "Line number is required"}

    analyzer = await get_clangd_analyzer()
    if not analyzer:
        return {"error": "clangd is not available"}

    try:
        # If column is 0, try to find the first identifier on the line
        if column == 0:
            column = await _find_first_identifier_column(file_path, line)
            logger.info(f"Auto-detected column {column} for line {line}")

        references = await analyzer.find_references(file_path, line, column)

        # Add code context to each reference
        enriched_refs = []
        for ref in references:
            ref_dict = ref.to_dict()
            # Read code context from file
            try:
                ref_path = Path(ref.file_path)
                if ref_path.exists():
                    with open(ref_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        ref_line = ref.line - 1  # 0-indexed
                        # Get 2 lines before and after for context
                        start = max(0, ref_line - 2)
                        end = min(len(lines), ref_line + 3)
                        context_lines = lines[start:end]
                        ref_dict["context"] = "".join(context_lines).strip()
                        ref_dict["context_start_line"] = start + 1
            except Exception:
                pass
            enriched_refs.append(ref_dict)

        return {
            "references": enriched_refs,
            "count": len(enriched_refs)
        }
    except Exception as e:
        logger.error(f"Failed to find references: {e}")
        return {"error": str(e)}


async def handle_go_to_definition(params: dict) -> dict:
    """Handle go_to_definition tool call - find definition using clangd LSP."""
    file_path = params.get("file_path", "")
    line = params.get("line", 0)
    column = params.get("column", 0)

    # Normalize path
    file_path = file_path.replace("\\", "/")

    if not file_path:
        return {"error": "File path is required"}
    if not line:
        return {"error": "Line number is required"}

    analyzer = await get_clangd_analyzer()
    if not analyzer:
        return {"error": "clangd is not available"}

    try:
        # If column is 0, try to find the first identifier on the line
        if column == 0:
            column = await _find_first_identifier_column(file_path, line)
            logger.info(f"Auto-detected column {column} for definition at line {line}")

        definition = await analyzer.go_to_definition(file_path, line, column)
        if definition:
            def_dict = definition.to_dict()
            # Add code context
            try:
                def_path = Path(definition.file_path)
                if def_path.exists():
                    with open(def_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        def_line = definition.line - 1  # 0-indexed
                        # Get 5 lines before and 15 lines after (capture function body)
                        start = max(0, def_line - 2)
                        end = min(len(lines), def_line + 20)
                        context_lines = lines[start:end]
                        def_dict["context"] = "".join(context_lines).strip()
                        def_dict["context_start_line"] = start + 1
            except Exception:
                pass
            return {"definition": def_dict, "found": True}
        return {"definition": None, "found": False}
    except Exception as e:
        logger.error(f"Failed to go to definition: {e}")
        return {"error": str(e)}


# =============================================================================
# HTTP Endpoints
# =============================================================================

async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "healthy",
        "service": "akashic-mcp-server",
        "timestamp": datetime.utcnow().isoformat()
    })


async def handle_index(request: web.Request) -> web.Response:
    """Serve frontend index.html."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return web.FileResponse(str(index_path))
    return web.Response(text="Frontend not found", status=404)


async def handle_indexing_manager(request: web.Request) -> web.Response:
    """Serve indexing manager UI."""
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    manager_path = frontend_dir / "indexing_manager.html"
    if manager_path.exists():
        return web.FileResponse(str(manager_path))
    return web.Response(text="Indexing Manager not found", status=404)


async def handle_tools_list(request: web.Request) -> web.Response:
    """List available MCP tools."""
    tools = [
        {
            "name": "search_code",
            "description": "Search code using natural language query",
            "parameters": {
                "query": {"type": "string", "required": True},
                "language": {"type": "string", "required": False, "enum": ["csharp", "cpp", "python"]},
                "limit": {"type": "integer", "required": False, "default": 10}
            }
        },
        {
            "name": "get_symbol",
            "description": "Find symbol definition by name",
            "parameters": {
                "name": {"type": "string", "required": True},
                "symbol_type": {"type": "string", "required": False}
            }
        },
        {
            "name": "get_file_context",
            "description": "Get code context from a specific file",
            "parameters": {
                "file_path": {"type": "string", "required": True},
                "start_line": {"type": "integer", "required": False},
                "end_line": {"type": "integer", "required": False}
            }
        },
        {
            "name": "list_indexed_files",
            "description": "List all indexed files",
            "parameters": {
                "path_pattern": {"type": "string", "required": False},
                "language": {"type": "string", "required": False}
            }
        },
        {
            "name": "get_references",
            "description": "Find all references to a symbol",
            "parameters": {
                "symbol": {"type": "string", "required": True},
                "file_path": {"type": "string", "required": False}
            }
        },
        {
            "name": "get_dependencies",
            "description": "Get dependency relationships for a symbol",
            "parameters": {
                "symbol": {"type": "string", "required": True},
                "direction": {"type": "string", "required": False, "enum": ["outgoing", "incoming", "both"]}
            }
        },
        {
            "name": "analyze_cpp_file",
            "description": "Analyze a C++ file using clangd to extract symbols (classes, functions, methods, etc.)",
            "parameters": {
                "file_path": {"type": "string", "required": True, "description": "Path to the C++ file to analyze"}
            }
        },
        {
            "name": "get_cpp_symbols",
            "description": "Search for C++ symbols by name using clangd semantic analysis",
            "parameters": {
                "name": {"type": "string", "required": True, "description": "Symbol name to search for"},
                "file_path": {"type": "string", "required": False, "description": "Optional: specific file to search in"},
                "kind": {"type": "string", "required": False, "enum": ["function", "class", "method", "struct", "enum", "field", "variable", "namespace"]}
            }
        },
        {
            "name": "find_cpp_references",
            "description": "Find all references to a symbol at a specific position using clangd LSP",
            "parameters": {
                "file_path": {"type": "string", "required": True, "description": "Path to the C++ source file"},
                "line": {"type": "integer", "required": True, "description": "Line number (1-indexed)"},
                "column": {"type": "integer", "required": True, "description": "Column number (0-indexed)"}
            }
        },
        {
            "name": "go_to_definition",
            "description": "Go to the definition of a symbol at a specific position using clangd LSP",
            "parameters": {
                "file_path": {"type": "string", "required": True, "description": "Path to the C++ source file"},
                "line": {"type": "integer", "required": True, "description": "Line number (1-indexed)"},
                "column": {"type": "integer", "required": True, "description": "Column number (0-indexed)"}
            }
        }
    ]
    return web.json_response({"tools": tools})


async def handle_tool_call(request: web.Request) -> web.Response:
    """Execute a tool call."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    tool_name = data.get("tool")
    params = data.get("params", {})

    if not tool_name:
        return web.json_response({"error": "Tool name is required"}, status=400)

    handlers = {
        "search_code": handle_search_code,
        "get_symbol": handle_get_symbol,
        "get_file_context": handle_get_file_context,
        "list_indexed_files": handle_list_indexed_files,
        "get_references": handle_get_references,
        "get_dependencies": handle_get_dependencies,
        "analyze_cpp_file": handle_analyze_cpp_file,
        "get_cpp_symbols": handle_get_cpp_symbols,
        "find_cpp_references": handle_find_cpp_references,
        "go_to_definition": handle_go_to_definition
    }

    handler = handlers.get(tool_name)
    if not handler:
        return web.json_response({"error": f"Unknown tool: {tool_name}"}, status=400)

    try:
        result = await handler(params)
        return web.json_response({"result": result})
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_search(request: web.Request) -> web.Response:
    """Convenience endpoint for search_code."""
    query = request.query.get("q", "")
    language = request.query.get("lang")
    limit = int(request.query.get("limit", 10))

    result = await handle_search_code({
        "query": query,
        "language": language,
        "limit": limit
    })
    return web.json_response(result)


async def handle_indexing_status(request: web.Request) -> web.Response:
    """Get indexing partition status by checking .done files."""
    status_dir = Path(__file__).resolve().parent.parent / "scripts" / "indexing" / "status"

    partitions = [
        "01_runtime_core", "02_rendering", "03_srp_packages", "04_scripting",
        "05_animation", "06_physics", "07_ui", "08_audio",
        "09_ai_nav", "10_input_xr", "11_ecs_jobs", "12_asset_serialization",
        "13_editor", "14_networking", "15_terrain_2d", "16_modules_rendering"
    ]

    results = {}
    for p in partitions:
        done_file = status_dir / f"{p}.done"
        if done_file.exists():
            try:
                with open(done_file, 'r') as f:
                    timestamp = f.read().strip()
                results[p] = {"status": "done", "timestamp": timestamp}
            except:
                results[p] = {"status": "done", "timestamp": "unknown"}
        else:
            results[p] = {"status": "pending", "timestamp": None}

    # Get total indexed stats
    try:
        store = await get_code_store()
        cursor = store.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM chunks")
        files = cursor.fetchone()[0]
    except:
        chunks = 0
        files = 0

    completed = sum(1 for p in results.values() if p["status"] == "done")

    return web.json_response({
        "partitions": results,
        "summary": {
            "total": len(partitions),
            "completed": completed,
            "pending": len(partitions) - completed,
            "indexed_files": files,
            "indexed_chunks": chunks
        }
    })


# =============================================================================
# MCP Protocol (SSE) Support
# =============================================================================

# MCP Tools metadata for protocol
MCP_TOOLS = [
    {
        "name": "search_code",
        "description": "Search code using natural language query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "language": {"type": "string", "enum": ["csharp", "cpp", "python"], "description": "Filter by language"},
                "limit": {"type": "integer", "default": 10, "description": "Maximum results"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_symbol",
        "description": "Find symbol definition by name",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Symbol name"},
                "symbol_type": {"type": "string", "description": "Type filter"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_file_context",
        "description": "Get code context from a specific file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "start_line": {"type": "integer", "description": "Start line"},
                "end_line": {"type": "integer", "description": "End line"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "list_indexed_files",
        "description": "List all indexed files",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path_pattern": {"type": "string", "description": "Glob pattern"},
                "language": {"type": "string", "description": "Language filter"}
            },
            "required": []
        }
    },
    {
        "name": "get_cpp_symbols",
        "description": "Search for C++ symbols by name using clangd",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Symbol name"},
                "file_path": {"type": "string", "description": "Specific file"},
                "kind": {"type": "string", "enum": ["function", "class", "method", "struct", "enum"]}
            },
            "required": ["name"]
        }
    },
    {
        "name": "find_cpp_references",
        "description": "Find all references to a symbol at position using clangd LSP",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to C++ file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "column": {"type": "integer", "description": "Column number (0-indexed)"}
            },
            "required": ["file_path", "line", "column"]
        }
    },
    {
        "name": "go_to_definition",
        "description": "Go to definition of symbol at position using clangd LSP",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to C++ file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "column": {"type": "integer", "description": "Column number (0-indexed)"}
            },
            "required": ["file_path", "line", "column"]
        }
    },
    {
        "name": "analyze_cpp_file",
        "description": "Extract all symbols from a C++ file using clangd",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to C++ file"}
            },
            "required": ["file_path"]
        }
    }
]

# Active SSE connections for message passing
_sse_connections: dict[str, asyncio.Queue] = {}
_connection_counter = 0


async def handle_mcp_sse(request: web.Request) -> web.StreamResponse:
    """
    MCP SSE endpoint for Claude Desktop connection.
    Implements MCP protocol over Server-Sent Events.
    """
    global _connection_counter
    _connection_counter += 1
    connection_id = f"conn_{_connection_counter}"

    logger.info(f"New MCP SSE connection: {connection_id}")

    # Create response with SSE headers
    response = web.StreamResponse(
        status=200,
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': '*',
        }
    )
    await response.prepare(request)

    # Create message queue for this connection
    message_queue: asyncio.Queue = asyncio.Queue()
    _sse_connections[connection_id] = message_queue

    try:
        # Send endpoint info for client to POST messages
        endpoint_event = {
            "endpoint": f"/mcp/message?session={connection_id}"
        }
        await response.write(f"event: endpoint\ndata: {json.dumps(endpoint_event)}\n\n".encode('utf-8'))

        # Keep connection alive and send messages from queue
        while True:
            try:
                # Wait for message with timeout for keepalive
                message = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                await response.write(f"event: message\ndata: {json.dumps(message)}\n\n".encode('utf-8'))
            except asyncio.TimeoutError:
                # Send keepalive ping
                await response.write(f": keepalive\n\n".encode('utf-8'))
            except asyncio.CancelledError:
                break

    except ConnectionResetError:
        logger.info(f"SSE connection closed: {connection_id}")
    finally:
        _sse_connections.pop(connection_id, None)
        logger.info(f"SSE connection removed: {connection_id}")

    return response


async def handle_mcp_message(request: web.Request) -> web.Response:
    """
    Handle MCP JSON-RPC messages from client.
    Messages are sent via POST, responses via SSE.
    """
    session_id = request.query.get("session", "")

    if session_id not in _sse_connections:
        return web.json_response({"error": "Invalid session"}, status=400)

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    logger.info(f"MCP message received: {data.get('method', 'unknown')}")

    # Handle JSON-RPC request
    jsonrpc_id = data.get("id")
    method = data.get("method", "")
    params = data.get("params", {})

    response_data = await process_mcp_request(method, params, jsonrpc_id)

    # Send response via SSE
    queue = _sse_connections.get(session_id)
    if queue:
        await queue.put(response_data)

    # Also return 202 Accepted
    return web.Response(status=202)


async def process_mcp_request(method: str, params: dict, jsonrpc_id) -> dict:
    """Process MCP JSON-RPC request and return response."""

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "akashic-records",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "notifications/initialized":
        # No response needed for notifications
        return {"jsonrpc": "2.0", "id": jsonrpc_id, "result": {}}

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "result": {
                "tools": MCP_TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        handlers = {
            "search_code": handle_search_code,
            "get_symbol": handle_get_symbol,
            "get_file_context": handle_get_file_context,
            "list_indexed_files": handle_list_indexed_files,
            "get_references": handle_get_references,
            "get_dependencies": handle_get_dependencies,
            "analyze_cpp_file": handle_analyze_cpp_file,
            "get_cpp_symbols": handle_get_cpp_symbols,
            "find_cpp_references": handle_find_cpp_references,
            "go_to_definition": handle_go_to_definition
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }

        try:
            result = await handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, ensure_ascii=False)
                        }
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": jsonrpc_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": jsonrpc_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


# =============================================================================
# Server Setup
# =============================================================================

async def on_startup(app: web.Application):
    """Initialize CodeStore on server startup."""
    logger.info("Initializing CodeStore...")
    await get_code_store()
    logger.info("Server startup complete")


async def on_cleanup(app: web.Application):
    """Cleanup on server shutdown."""
    global _code_store
    if _code_store:
        await _code_store.close()
        _code_store = None
    logger.info("Server cleanup complete")


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()

    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        )
    })

    # Add API routes
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/tools", handle_tools_list)
    app.router.add_post("/api/call", handle_tool_call)
    app.router.add_get("/api/search", handle_search)
    app.router.add_get("/api/indexing-status", handle_indexing_status)

    # MCP Protocol routes (for Claude Desktop)
    if ENABLE_MCP_HTTP_ROUTES:
        app.router.add_get("/mcp/sse", handle_mcp_sse)
        app.router.add_post("/mcp/message", handle_mcp_message)
    else:
        logger.info("MCP HTTP routes are disabled (AKASHIC_ENABLE_MCP_HTTP_ROUTES=false)")

    # Serve frontend
    app.router.add_get("/", handle_index)
    app.router.add_get("/indexing", handle_indexing_manager)

    # Serve static files
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.router.add_static("/static/", str(frontend_dir), name="static")

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        try:
            cors.add(route)
        except ValueError:
            pass  # Skip static routes

    # Add lifecycle hooks
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def main():
    default_host = os.environ.get("MCP_HOST", os.environ.get("AKASHIC_SERVER_HOST", "0.0.0.0"))
    default_port = int(os.environ.get("MCP_PORT", os.environ.get("AKASHIC_SERVER_PORT", "8088")))

    parser = argparse.ArgumentParser(description="Akashic MCP Server (HTTP Mode)")
    parser.add_argument("--host", default=default_host, help=f"Host to bind (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument(
        "--settings",
        default=os.environ.get("AKASHIC_SETTINGS_PATH", "config/settings.yaml"),
        help="Path to settings.yaml (or use AKASHIC_SETTINGS_PATH env)",
    )
    args = parser.parse_args()
    os.environ["AKASHIC_SETTINGS_PATH"] = args.settings

    app = create_app()

    logger.info(f"Starting Akashic MCP Server on {args.host}:{args.port}")
    print(f"\n{'='*60}")
    print(f"  Akashic Records MCP Server (HTTP Mode)")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"{'='*60}")
    print(f"\nAPI Endpoints:")
    print(f"  GET  /health       - Health check")
    print(f"  GET  /api/tools    - List available tools")
    print(f"  POST /api/call     - Execute tool call")
    print(f"  GET  /api/search   - Quick search (query: q, lang, limit)")
    if ENABLE_MCP_HTTP_ROUTES:
        print(f"\nMCP Protocol (Claude Desktop):")
        print(f"  GET  /mcp/sse      - SSE connection endpoint")
        print(f"  POST /mcp/message  - JSON-RPC message endpoint")
        print(f"\nClaude Desktop config (claude_desktop_config.json):")
        print(f'  {{"mcpServers":{{"akashic":{{"url":"http://{args.host}:{args.port}/mcp/sse","transport":"sse"}}}}}}')
    else:
        print(f"\nMCP Protocol endpoints are disabled (AKASHIC_ENABLE_MCP_HTTP_ROUTES=false)")
    print(f"\nExample API call:")
    print(f"  curl 'http://{args.host}:{args.port}/api/search?q=MonoBehaviour&limit=5'")
    print(f"{'='*60}\n")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
