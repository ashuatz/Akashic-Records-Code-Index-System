"""
Akashic Records MCP Server

MCP (Model Context Protocol) server for AI-powered code search and indexing.
Provides tools for natural language code search, symbol lookup, and file context retrieval.
"""

import asyncio
import logging
import sys
import os
import fnmatch
from pathlib import Path
from typing import Any, Optional, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code_store import CodeStore, code_store_context, SearchResult, SymbolInfo
try:
    from runtime_config import load_dotenv, resolve_settings_path
except ImportError:  # pragma: no cover - package import fallback
    from .runtime_config import load_dotenv, resolve_settings_path

load_dotenv()

# Configure logging to file only (stderr must be clean for MCP)
log_dir = Path(__file__).parent.parent / "data"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "akashic_mcp_server.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
    ]
)
logger = logging.getLogger(__name__)


# Global CodeStore instance
_code_store: Optional[CodeStore] = None


async def get_code_store() -> CodeStore:
    """Get or initialize the global CodeStore instance."""
    global _code_store
    if _code_store is None:
        settings_path = resolve_settings_path()
        _code_store = CodeStore(str(settings_path))
        await _code_store.connect()
        logger.info("CodeStore initialized successfully")
    return _code_store


# Initialize server
server = Server("akashic-code-index")


def format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results for display"""
    if not results:
        return "No results found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n=== Result {i} ===")
        output.append(f"File: {result.get('file_path', 'Unknown')}")
        output.append(f"Lines: {result.get('start_line', '?')}-{result.get('end_line', '?')}")

        if 'symbol_type' in result:
            output.append(f"Type: {result['symbol_type']}")
        if 'symbol_name' in result:
            output.append(f"Symbol: {result['symbol_name']}")
        if 'language' in result:
            output.append(f"Language: {result['language']}")
        if 'score' in result:
            output.append(f"Relevance: {result['score']:.2f}")

        if 'content' in result:
            output.append("\nCode:")
            output.append("```")
            output.append(result['content'])
            output.append("```")

    return "\n".join(output)


def format_symbol_results(results: list[dict[str, Any]]) -> str:
    """Format symbol search results for display"""
    if not results:
        return "Symbol not found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(f"\n=== Match {i} ===")
        output.append(f"Symbol: {result.get('symbol_name', 'Unknown')}")
        output.append(f"Type: {result.get('symbol_type', 'Unknown')}")
        output.append(f"File: {result.get('file_path', 'Unknown')}")
        output.append(f"Line: {result.get('line_number', '?')}")

        if 'signature' in result:
            output.append(f"\nSignature:")
            output.append(f"```")
            output.append(result['signature'])
            output.append("```")

        if 'content' in result:
            output.append("\nDefinition:")
            output.append("```")
            output.append(result['content'])
            output.append("```")

    return "\n".join(output)


def format_references_results(refs: list[dict[str, Any]]) -> str:
    """Format reference results for display"""
    if not refs:
        return "No references found."

    output = [f"Found {len(refs)} references:\n"]

    # Group by file for better readability
    by_file: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        file_path = ref.get('file_path', 'Unknown')
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(ref)

    for file_path, file_refs in sorted(by_file.items()):
        output.append(f"\n{file_path}")
        for ref in sorted(file_refs, key=lambda r: r.get('line', 0)):
            line = ref.get('line', '?')
            column = ref.get('column', '?')
            kind = ref.get('kind', 'reference')
            output.append(f"  Line {line}:{column} - {kind}")

    return "\n".join(output)


def format_dependencies_results(deps: list[dict[str, Any]]) -> str:
    """Format dependency results for display"""
    if not deps:
        return "No dependencies found."

    output = [f"Found {len(deps)} dependencies:\n"]

    # Group by relationship kind
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for dep in deps:
        kind = dep.get('kind', 'unknown')
        if kind not in by_kind:
            by_kind[kind] = []
        by_kind[kind].append(dep)

    for kind, kind_deps in sorted(by_kind.items()):
        output.append(f"\n{kind.upper()}:")
        for dep in kind_deps:
            from_symbol = dep.get('from_symbol', '?')
            to_symbol = dep.get('to_symbol', '?')
            file_path = dep.get('file_path', '')
            if file_path:
                output.append(f"  {from_symbol} -> {to_symbol} ({file_path})")
            else:
                output.append(f"  {from_symbol} -> {to_symbol}")

    return "\n".join(output)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_code",
            description=(
                "Search code using natural language query. "
                "Supports semantic search across indexed codebase. "
                "Examples: 'collision detection', '물리 엔진 충돌', 'authentication logic'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language (csharp, cpp, python)",
                        "enum": ["csharp", "cpp", "python"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_symbol",
            description=(
                "Find symbol definition (similar to VS Code Ctrl+T). "
                "Search for classes, methods, functions, properties by name. "
                "Supports partial matching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Symbol name to find (supports partial match)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Filter by symbol type",
                        "enum": ["class", "method", "function", "property", "interface", "enum"]
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="get_file_context",
            description=(
                "Get code context from a specific file. "
                "Retrieve entire file or specific line range."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file (absolute or relative to index root)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed)",
                        "minimum": 1
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (inclusive)",
                        "minimum": 1
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="list_indexed_files",
            description=(
                "List all indexed files in the codebase. "
                "Supports filtering by glob pattern and language."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter (e.g., '**/Physics/*', '*.cs')"
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language",
                        "enum": ["csharp", "cpp", "python"]
                    }
                }
            }
        ),
        Tool(
            name="get_references",
            description=(
                "Find all references to a symbol (function calls, variable usages). "
                "Returns locations where the symbol is used."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to find references for"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional: limit to specific file"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_dependencies",
            description=(
                "Get dependency relationships for a symbol "
                "(what it inherits/calls/uses, or what depends on it)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "description": (
                            "outgoing = what this symbol depends on, "
                            "incoming = what depends on this symbol"
                        )
                    }
                },
                "required": ["symbol"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    try:
        logger.info(f"Tool called: {name} with args: {arguments}")

        # Get CodeStore instance
        store = await get_code_store()

        if name == "search_code":
            query = arguments["query"]
            language = arguments.get("language")
            limit = arguments.get("limit", 10)

            if not query.strip():
                return [TextContent(
                    type="text",
                    text="Error: Query cannot be empty"
                )]

            # Use async search from CodeStore
            results: List[SearchResult] = await store.search(
                query=query,
                limit=limit,
                language=language,
                use_reranker=True
            )

            # Format results
            formatted = format_search_results([
                {
                    'file_path': r.chunk.file_path,
                    'start_line': r.chunk.start_line,
                    'end_line': r.chunk.end_line,
                    'symbol_type': r.chunk.symbol_type,
                    'symbol_name': r.chunk.symbol_name,
                    'language': r.chunk.language,
                    'score': r.rerank_score if r.rerank_score is not None else r.score,
                    'content': r.chunk.code
                }
                for r in results
            ])

            return [TextContent(type="text", text=formatted)]

        elif name == "get_symbol":
            symbol_name = arguments["name"]
            symbol_type = arguments.get("symbol_type")

            if not symbol_name.strip():
                return [TextContent(
                    type="text",
                    text="Error: Symbol name cannot be empty"
                )]

            # Use get_symbol from CodeStore
            symbol_info: Optional[SymbolInfo] = await store.get_symbol(
                name=symbol_name,
                symbol_type=symbol_type
            )

            if symbol_info:
                results = [{
                    'symbol_name': symbol_info.name,
                    'symbol_type': symbol_info.type,
                    'file_path': symbol_info.file_path,
                    'line_number': symbol_info.start_line,
                    'content': symbol_info.code
                }]
            else:
                results = []

            formatted = format_symbol_results(results)

            return [TextContent(type="text", text=formatted)]

        elif name == "get_file_context":
            file_path = arguments["file_path"]
            start_line = arguments.get("start_line")
            end_line = arguments.get("end_line")

            if start_line is not None and end_line is not None:
                if start_line > end_line:
                    return [TextContent(
                        type="text",
                        text="Error: start_line cannot be greater than end_line"
                    )]

            # Read file directly
            try:
                file_obj = Path(file_path)
                if not file_obj.exists():
                    return [TextContent(
                        type="text",
                        text=f"Error: File not found: {file_path}"
                    )]

                content = file_obj.read_text(encoding='utf-8', errors='ignore')

                # Extract range if specified
                if start_line is not None or end_line is not None:
                    lines = content.split('\n')
                    start_idx = (start_line - 1) if start_line else 0
                    end_idx = end_line if end_line else len(lines)
                    content = '\n'.join(lines[start_idx:end_idx])

            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error reading file: {e}"
                )]

            # Format with line numbers if range specified
            if start_line is not None:
                lines = content.split('\n')
                numbered_lines = [
                    f"{start_line + i:4d} | {line}"
                    for i, line in enumerate(lines)
                ]
                formatted_content = '\n'.join(numbered_lines)
            else:
                formatted_content = content

            return [TextContent(
                type="text",
                text=f"File: {file_path}\n\n```\n{formatted_content}\n```"
            )]

        elif name == "list_indexed_files":
            path_pattern = arguments.get("path_pattern")
            language = arguments.get("language")

            # Query SQLite for indexed files
            if store.db_conn is None:
                return [TextContent(
                    type="text",
                    text="Error: Database not connected"
                )]

            cursor = store.db_conn.cursor()

            if language:
                cursor.execute(
                    "SELECT DISTINCT file_path FROM chunks WHERE language = ?",
                    (language,)
                )
            else:
                cursor.execute("SELECT DISTINCT file_path FROM chunks")

            files = [row[0] for row in cursor.fetchall()]

            # Filter by pattern if specified
            if path_pattern and files:
                files = [f for f in files if fnmatch.fnmatch(f, path_pattern)]

            if not files:
                return [TextContent(
                    type="text",
                    text="No files found matching the criteria."
                )]

            # Group by directory for better readability
            output = [f"Found {len(files)} indexed files:\n"]
            for file_path in sorted(files)[:100]:  # Limit to 100 files
                output.append(f"  {file_path}")

            if len(files) > 100:
                output.append(f"\n  ... and {len(files) - 100} more files")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "get_references":
            symbol = arguments["symbol"]
            file_path = arguments.get("file_path")

            if not symbol.strip():
                return [TextContent(
                    type="text",
                    text="Error: Symbol name cannot be empty"
                )]

            # Query SQLite references table
            if store.db_conn is None:
                return [TextContent(
                    type="text",
                    text="Error: Database not connected"
                )]

            cursor = store.db_conn.cursor()

            # Query references joined with symbols
            if file_path:
                cursor.execute("""
                    SELECT r.file_path, r.line_number, r.column_number, r.reference_kind
                    FROM references r
                    JOIN symbols s ON r.symbol_id = s.id
                    WHERE s.name LIKE ? AND r.file_path = ?
                    ORDER BY r.file_path, r.line_number
                """, (f"%{symbol}%", file_path))
            else:
                cursor.execute("""
                    SELECT r.file_path, r.line_number, r.column_number, r.reference_kind
                    FROM references r
                    JOIN symbols s ON r.symbol_id = s.id
                    WHERE s.name LIKE ?
                    ORDER BY r.file_path, r.line_number
                """, (f"%{symbol}%",))

            rows = cursor.fetchall()

            # Format results
            refs = [
                {
                    'file_path': row[0],
                    'line': row[1],
                    'column': row[2],
                    'kind': row[3]
                }
                for row in rows
            ]

            formatted = format_references_results(refs)
            return [TextContent(type="text", text=formatted)]

        elif name == "get_dependencies":
            symbol = arguments["symbol"]
            direction = arguments.get("direction", "both")

            if not symbol.strip():
                return [TextContent(
                    type="text",
                    text="Error: Symbol name cannot be empty"
                )]

            # Query SQLite dependencies table
            if store.db_conn is None:
                return [TextContent(
                    type="text",
                    text="Error: Database not connected"
                )]

            cursor = store.db_conn.cursor()

            deps = []

            # Query outgoing dependencies (what this symbol depends on)
            if direction in ["outgoing", "both"]:
                cursor.execute("""
                    SELECT d.dependency_kind, s_from.name, s_to.name, s_to.file_path
                    FROM dependencies d
                    JOIN symbols s_from ON d.from_symbol_id = s_from.id
                    JOIN symbols s_to ON d.to_symbol_id = s_to.id
                    WHERE s_from.name LIKE ?
                """, (f"%{symbol}%",))

                for row in cursor.fetchall():
                    deps.append({
                        'kind': row[0],
                        'from_symbol': row[1],
                        'to_symbol': row[2],
                        'file_path': row[3]
                    })

            # Query incoming dependencies (what depends on this symbol)
            if direction in ["incoming", "both"]:
                cursor.execute("""
                    SELECT d.dependency_kind, s_from.name, s_to.name, s_from.file_path
                    FROM dependencies d
                    JOIN symbols s_from ON d.from_symbol_id = s_from.id
                    JOIN symbols s_to ON d.to_symbol_id = s_to.id
                    WHERE s_to.name LIKE ?
                """, (f"%{symbol}%",))

                for row in cursor.fetchall():
                    deps.append({
                        'kind': f"{row[0]} (incoming)",
                        'from_symbol': row[1],
                        'to_symbol': row[2],
                        'file_path': row[3]
                    })

            formatted = format_dependencies_results(deps)
            return [TextContent(type="text", text=formatted)]

        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool: {name}"
            )]

    except KeyError as e:
        error_msg = f"Error: Missing required argument: {e}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]

    except Exception as e:
        error_msg = f"Error executing tool '{name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Akashic Records MCP Server")

    try:
        # Initialize CodeStore on startup
        await get_code_store()

        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server initialized, waiting for connections...")
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        global _code_store
        if _code_store:
            await _code_store.close()
        logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
