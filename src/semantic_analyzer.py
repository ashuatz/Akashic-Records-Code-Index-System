"""
Semantic Analysis Module for Akashic Records

Provides semantic analysis using clangd (C++) and Roslyn (C#) for extracting
symbols and references. Enables advanced code navigation features like
"Go to Definition" and "Find References".

Supports:
- C++ via clangd LSP server
- C# via Roslyn/OmniSharp LSP server

Usage:
    # C++ Analysis
    analyzer = ClangdAnalyzer(compile_commands_path="build/compile_commands.json")
    symbols, references = await analyzer.analyze_file("src/main.cpp")

    # C# Analysis
    analyzer = RoslynAnalyzer(solution_path="MyProject.sln")
    symbols, references = await analyzer.analyze()
"""

import asyncio
import json
import logging
import os
import subprocess
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class SymbolKind(str, Enum):
    """Symbol kinds matching LSP SymbolKind enum."""
    FILE = "file"
    MODULE = "module"
    NAMESPACE = "namespace"
    PACKAGE = "package"
    CLASS = "class"
    METHOD = "method"
    PROPERTY = "property"
    FIELD = "field"
    CONSTRUCTOR = "constructor"
    ENUM = "enum"
    INTERFACE = "interface"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONSTANT = "constant"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    KEY = "key"
    NULL = "null"
    ENUM_MEMBER = "enummember"
    STRUCT = "struct"
    EVENT = "event"
    OPERATOR = "operator"
    TYPE_PARAMETER = "typeparameter"


class ReferenceKind(str, Enum):
    """Reference kinds."""
    DEFINITION = "definition"
    DECLARATION = "declaration"
    CALL = "call"
    READ = "read"
    WRITE = "write"
    REFERENCE = "reference"


@dataclass
class Symbol:
    """Represents a code symbol (function, class, variable, etc.)."""
    name: str
    kind: str  # function, class, variable, method, field, ...
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int = 0
    signature: Optional[str] = None
    parent_name: Optional[str] = None  # For methods: class name
    doc_comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class Reference:
    """Represents a reference to a symbol."""
    symbol_name: str
    file_path: str
    line: int
    column: int
    kind: str  # call, read, write, definition, reference
    context: Optional[str] = None  # Surrounding code context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# Clangd Analyzer (C++)
# ============================================================================

DEFAULT_CLANGD_PATH = "clangd"


def resolve_clangd_path(clangd_path: Optional[str] = None) -> str:
    """Resolve clangd executable from arg -> env -> PATH -> fallback command name."""
    if clangd_path:
        return clangd_path

    env_path = os.getenv("AKASHIC_CLANGD_PATH")
    if env_path:
        return env_path

    detected = shutil.which("clangd")
    if detected:
        return detected

    return DEFAULT_CLANGD_PATH


class ClangdAnalyzer:
    """
    C++ semantic analyzer using clangd LSP server.

    Requires:
    - clangd installed and in PATH
    - compile_commands.json for the project

    Features:
    - Symbol extraction (classes, functions, methods, fields)
    - Reference finding (calls, definitions, uses)
    - Type information and signatures
    """

    def __init__(self, compile_commands_path: Optional[str] = None,
                 clangd_path: Optional[str] = None, timeout: int = 60):
        """
        Initialize clangd analyzer.

        Args:
            compile_commands_path: Path to compile_commands.json directory
            clangd_path: Path to clangd executable (optional)
            timeout: Timeout for clangd operations in seconds
        """
        self.compile_commands_path = Path(compile_commands_path) if compile_commands_path else None
        self.clangd_path = resolve_clangd_path(clangd_path)
        self.timeout = timeout
        self._verified = False

    async def verify_installation(self) -> bool:
        """Verify clangd is installed and accessible."""
        if self._verified:
            return True

        try:
            result = await asyncio.create_subprocess_exec(
                self.clangd_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=5)

            if result.returncode == 0:
                version_info = stdout.decode('utf-8', errors='ignore')
                logger.info(f"clangd found: {version_info.splitlines()[0]}")
                self._verified = True
                return True
            else:
                logger.error(f"clangd check failed: {stderr.decode('utf-8', errors='ignore')}")
                return False

        except asyncio.TimeoutError:
            logger.error("clangd --version timed out")
            return False
        except FileNotFoundError:
            logger.error(f"clangd not found at: {self.clangd_path}")
            logger.info("Install clangd: https://clangd.llvm.org/installation.html")
            return False
        except Exception as e:
            logger.error(f"Failed to verify clangd: {e}")
            return False

    async def analyze_file(self, file_path: str) -> Tuple[List[Symbol], List[Reference]]:
        """
        Analyze a C++ file and extract symbols and references.

        Args:
            file_path: Path to C++ source file

        Returns:
            Tuple of (symbols, references) lists

        Raises:
            RuntimeError: If clangd is not available or analysis fails
        """
        if not await self.verify_installation():
            raise RuntimeError("clangd is not available")

        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Use clangd's --check mode for batch analysis
            symbols = await self._extract_symbols_check_mode(file_path)
            references = await self._extract_references_check_mode(file_path)

            logger.info(f"Analyzed {file_path}: {len(symbols)} symbols, {len(references)} references")
            return symbols, references

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return [], []

    async def analyze_directory(self, dir_path: str,
                                 extensions: Optional[List[str]] = None) -> Tuple[List[Symbol], List[Reference]]:
        """
        Analyze all C++ files in a directory.

        Args:
            dir_path: Directory path to analyze
            extensions: File extensions to process (default: ['.cpp', '.cc', '.cxx', '.h', '.hpp'])

        Returns:
            Tuple of (symbols, references) lists for all files
        """
        if extensions is None:
            extensions = ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx']

        dir_path = Path(dir_path).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_symbols = []
        all_references = []

        # Collect all files
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(f"*{ext}"))

        logger.info(f"Analyzing {len(files)} C++ files in {dir_path}")

        # Analyze files in parallel with concurrency limit
        sem = asyncio.Semaphore(4)  # Limit concurrent clangd processes

        async def analyze_with_semaphore(file_path: Path):
            async with sem:
                return await self.analyze_file(str(file_path))

        tasks = [analyze_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Analysis failed: {result}")
                continue

            symbols, references = result
            all_symbols.extend(symbols)
            all_references.extend(references)

        logger.info(f"Total: {len(all_symbols)} symbols, {len(all_references)} references")
        return all_symbols, all_references

    async def _extract_symbols_check_mode(self, file_path: Path) -> List[Symbol]:
        """Extract symbols using clangd LSP protocol."""
        try:
            return await self._extract_symbols_lsp(file_path)
        except Exception as e:
            logger.warning(f"LSP symbol extraction failed for {file_path}: {e}")
            return []

    async def _extract_symbols_lsp(self, file_path: Path) -> List[Symbol]:
        """Extract symbols using clangd LSP with JSON-RPC."""
        import struct

        args = [self.clangd_path]
        if self.compile_commands_path:
            args.extend(["--compile-commands-dir", str(self.compile_commands_path)])
        args.extend(["--log=error"])  # Reduce log noise

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_uri = f"file:///{str(file_path).replace(chr(92), '/').replace(':', '%3A')}"

            # Initialize LSP
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": f"file:///{str(self.compile_commands_path).replace(chr(92), '/').replace(':', '%3A')}" if self.compile_commands_path else None,
                    "capabilities": {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True
                            }
                        }
                    }
                }
            }

            # Send initialize
            await self._send_lsp_message(proc.stdin, init_request)
            response = await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)

            # Send initialized notification
            initialized_notif = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
            await self._send_lsp_message(proc.stdin, initialized_notif)

            # Open document
            open_notif = {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": file_uri,
                        "languageId": "cpp",
                        "version": 1,
                        "text": content
                    }
                }
            }
            await self._send_lsp_message(proc.stdin, open_notif)

            # Wait for indexing (clangd sends background indexing notifications)
            await asyncio.sleep(2)

            # Request document symbols
            symbols_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/documentSymbol",
                "params": {
                    "textDocument": {"uri": file_uri}
                }
            }
            await self._send_lsp_message(proc.stdin, symbols_request)

            # Read response (may need to skip notifications)
            symbols_response = None
            for _ in range(10):  # Try up to 10 messages
                response = await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)
                if response and response.get("id") == 2:
                    symbols_response = response
                    break

            # Shutdown
            shutdown_request = {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": None}
            await self._send_lsp_message(proc.stdin, shutdown_request)

            exit_notif = {"jsonrpc": "2.0", "method": "exit", "params": None}
            await self._send_lsp_message(proc.stdin, exit_notif)

            # Parse symbols from response
            if symbols_response and "result" in symbols_response:
                return self._parse_lsp_symbols(symbols_response["result"], str(file_path))

            return []

        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()

    async def _send_lsp_message(self, stdin, message: dict):
        """Send LSP JSON-RPC message."""
        content = json.dumps(message)
        header = f"Content-Length: {len(content.encode('utf-8'))}\r\n\r\n"
        stdin.write(header.encode('utf-8') + content.encode('utf-8'))
        await stdin.drain()

    async def _read_lsp_message(self, stdout) -> Optional[dict]:
        """Read LSP JSON-RPC message."""
        # Read headers
        headers = {}
        while True:
            line = await stdout.readline()
            if not line:
                return None
            line = line.decode('utf-8').strip()
            if not line:
                break
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip().lower()] = value.strip()

        # Read content - ensure we read ALL bytes (important for large responses)
        content_length = int(headers.get('content-length', 0))
        if content_length > 0:
            # Use readexactly to ensure complete read
            try:
                content = await stdout.readexactly(content_length)
            except asyncio.IncompleteReadError as e:
                # If incomplete, use what we got
                content = e.partial
                logger.warning(f"Incomplete LSP read: got {len(content)} of {content_length} bytes")

            try:
                return json.loads(content.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.warning(f"LSP JSON decode error: {e}")
                # Try to salvage partial JSON by finding complete objects
                decoded = content.decode('utf-8', errors='replace')
                # If it's a response with result, try to extract it
                if '"result"' in decoded:
                    # Find the last complete JSON object
                    try:
                        # Try parsing progressively shorter strings
                        for end_pos in range(len(decoded), 0, -1):
                            try:
                                return json.loads(decoded[:end_pos])
                            except json.JSONDecodeError:
                                continue
                    except Exception:
                        pass
                return None
        return None

    def _parse_lsp_symbols(self, symbols_data: List[dict], file_path: str) -> List[Symbol]:
        """Parse LSP DocumentSymbol response into Symbol objects."""
        symbols = []

        # LSP SymbolKind mapping
        kind_map = {
            1: "file", 2: "module", 3: "namespace", 4: "package",
            5: "class", 6: "method", 7: "property", 8: "field",
            9: "constructor", 10: "enum", 11: "interface", 12: "function",
            13: "variable", 14: "constant", 15: "string", 16: "number",
            17: "boolean", 18: "array", 19: "object", 20: "key",
            21: "null", 22: "enummember", 23: "struct", 24: "event",
            25: "operator", 26: "typeparameter"
        }

        def process_symbol(sym: dict, parent_name: str = None):
            name = sym.get("name", "")
            kind_num = sym.get("kind", 0)
            kind = kind_map.get(kind_num, "unknown")

            range_info = sym.get("range", sym.get("location", {}).get("range", {}))
            start = range_info.get("start", {})
            end = range_info.get("end", {})

            symbol = Symbol(
                name=name,
                kind=kind,
                file_path=file_path,
                line_start=start.get("line", 0) + 1,  # LSP is 0-indexed
                line_end=end.get("line", 0) + 1,
                column_start=start.get("character", 0),
                column_end=end.get("character", 0),
                signature=sym.get("detail"),
                parent_name=parent_name
            )
            symbols.append(symbol)

            # Process children (for hierarchical symbols)
            for child in sym.get("children", []):
                process_symbol(child, name)

        for sym in (symbols_data or []):
            process_symbol(sym)

        return symbols

    def _parse_ast_symbols(self, ast_node: Dict[str, Any], file_path: str,
                           parent_name: Optional[str] = None) -> List[Symbol]:
        """Recursively parse AST and extract symbols."""
        symbols = []

        if not isinstance(ast_node, dict):
            return symbols

        kind = ast_node.get('kind', '')

        # Symbol kinds we're interested in
        symbol_kinds = {
            'FunctionDecl': 'function',
            'CXXMethodDecl': 'method',
            'CXXConstructorDecl': 'constructor',
            'CXXDestructorDecl': 'destructor',
            'ClassDecl': 'class',
            'StructDecl': 'struct',
            'EnumDecl': 'enum',
            'FieldDecl': 'field',
            'VarDecl': 'variable',
            'NamespaceDecl': 'namespace',
        }

        if kind in symbol_kinds:
            name = ast_node.get('name', '<anonymous>')
            loc = ast_node.get('loc', {})
            range_info = ast_node.get('range', {})

            if loc and 'line' in loc:
                symbol = Symbol(
                    name=name,
                    kind=symbol_kinds[kind],
                    file_path=file_path,
                    line_start=loc.get('line', 0),
                    line_end=range_info.get('end', {}).get('line', loc.get('line', 0)),
                    column_start=loc.get('col', 0),
                    column_end=range_info.get('end', {}).get('col', 0),
                    signature=ast_node.get('type', {}).get('qualType'),
                    parent_name=parent_name
                )
                symbols.append(symbol)

                # Update parent for nested symbols
                if kind in ['ClassDecl', 'StructDecl', 'NamespaceDecl']:
                    parent_name = name

        # Recurse into children
        for child in ast_node.get('inner', []):
            symbols.extend(self._parse_ast_symbols(child, file_path, parent_name))

        return symbols

    async def _extract_references_check_mode(self, file_path: Path) -> List[Reference]:
        """Extract references (placeholder - requires full LSP implementation)."""
        # Full reference extraction requires LSP protocol implementation
        # This is a placeholder that returns empty list
        # TODO: Implement LSP client for full reference extraction
        return []

    async def search_symbols(
        self,
        name: str,
        file_path: Optional[str] = None,
        kind: Optional[str] = None
    ) -> List[Symbol]:
        """
        Search for symbols by name using clangd's workspace symbol search.

        Args:
            name: Symbol name to search (supports partial match)
            file_path: Optional file path to limit search scope
            kind: Optional symbol kind filter (function, class, method, etc.)

        Returns:
            List of Symbol objects matching the query
        """
        if not await self.verify_installation():
            raise RuntimeError("clangd is not available")

        # If file_path is provided, analyze that specific file
        if file_path:
            file_path = Path(file_path).resolve()
            if not file_path.exists():
                return []

            try:
                symbols, _ = await self.analyze_file(str(file_path))
                # Filter by name and kind
                results = []
                for s in symbols:
                    if name.lower() in s.name.lower():
                        if kind is None or s.kind.lower() == kind.lower():
                            results.append(s)
                return results
            except Exception as e:
                logger.error(f"Failed to search symbols in {file_path}: {e}")
                return []

        # Without file_path, use workspace symbol search via LSP
        # This is a simplified version - returns empty for now
        # Full implementation would require maintaining an LSP connection
        logger.warning("Workspace-wide symbol search requires file_path parameter")
        return []

    async def find_references(self, file_path: str, line: int, column: int) -> List[Reference]:
        """
        Find all references to the symbol at the given position using clangd LSP.

        Args:
            file_path: Path to the source file
            line: 1-indexed line number
            column: 0-indexed column number

        Returns:
            List of Reference objects
        """
        if not await self.verify_installation():
            raise RuntimeError("clangd is not available")

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        args = [self.clangd_path]
        if self.compile_commands_path:
            args.extend(["--compile-commands-dir", str(self.compile_commands_path)])
        args.extend(["--log=error"])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_uri = f"file:///{str(file_path).replace(chr(92), '/').replace(':', '%3A')}"

            # Initialize LSP
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": f"file:///{str(self.compile_commands_path).replace(chr(92), '/').replace(':', '%3A')}" if self.compile_commands_path else None,
                    "capabilities": {
                        "textDocument": {
                            "references": {"dynamicRegistration": False}
                        }
                    }
                }
            }

            await self._send_lsp_message(proc.stdin, init_request)
            await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)

            # Initialized notification
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

            # Open document
            open_notif = {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": file_uri,
                        "languageId": "cpp",
                        "version": 1,
                        "text": content
                    }
                }
            }
            await self._send_lsp_message(proc.stdin, open_notif)

            # Wait for indexing
            await asyncio.sleep(3)

            # Request references (LSP uses 0-indexed lines)
            refs_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/references",
                "params": {
                    "textDocument": {"uri": file_uri},
                    "position": {"line": line - 1, "character": column},
                    "context": {"includeDeclaration": True}
                }
            }
            await self._send_lsp_message(proc.stdin, refs_request)

            # Read response
            refs_response = None
            for _ in range(15):
                response = await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)
                if response and response.get("id") == 2:
                    refs_response = response
                    break

            # Shutdown
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": None})
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "method": "exit", "params": None})

            # Parse references
            references = []
            if refs_response and "result" in refs_response and refs_response["result"]:
                for loc in refs_response["result"]:
                    uri = loc.get("uri", "")
                    # Convert URI back to path
                    ref_path = uri.replace("file:///", "").replace("%3A", ":").replace("/", "\\")
                    range_info = loc.get("range", {})
                    start = range_info.get("start", {})

                    references.append(Reference(
                        symbol_name="",  # Would need additional lookup
                        file_path=ref_path,
                        line=start.get("line", 0) + 1,
                        column=start.get("character", 0),
                        kind="reference"
                    ))

            return references

        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()

    async def go_to_definition(self, file_path: str, line: int, column: int) -> Optional[Reference]:
        """
        Go to the definition of the symbol at the given position.

        Args:
            file_path: Path to the source file
            line: 1-indexed line number
            column: 0-indexed column number

        Returns:
            Reference to the definition, or None if not found
        """
        if not await self.verify_installation():
            raise RuntimeError("clangd is not available")

        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        args = [self.clangd_path]
        if self.compile_commands_path:
            args.extend(["--compile-commands-dir", str(self.compile_commands_path)])
        args.extend(["--log=error"])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            file_uri = f"file:///{str(file_path).replace(chr(92), '/').replace(':', '%3A')}"

            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "processId": None,
                    "rootUri": f"file:///{str(self.compile_commands_path).replace(chr(92), '/').replace(':', '%3A')}" if self.compile_commands_path else None,
                    "capabilities": {}
                }
            }

            await self._send_lsp_message(proc.stdin, init_request)
            await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

            # Open document
            open_notif = {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": file_uri,
                        "languageId": "cpp",
                        "version": 1,
                        "text": content
                    }
                }
            }
            await self._send_lsp_message(proc.stdin, open_notif)
            await asyncio.sleep(3)

            # Request definition
            def_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "textDocument/definition",
                "params": {
                    "textDocument": {"uri": file_uri},
                    "position": {"line": line - 1, "character": column}
                }
            }
            await self._send_lsp_message(proc.stdin, def_request)

            # Read response
            def_response = None
            for _ in range(15):
                response = await asyncio.wait_for(self._read_lsp_message(proc.stdout), timeout=30)
                if response and response.get("id") == 2:
                    def_response = response
                    break

            # Shutdown
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": None})
            await self._send_lsp_message(proc.stdin, {"jsonrpc": "2.0", "method": "exit", "params": None})

            # Parse definition
            if def_response and "result" in def_response and def_response["result"]:
                result = def_response["result"]
                # Can be a single Location or array of Location
                if isinstance(result, list):
                    result = result[0] if result else None

                if result:
                    uri = result.get("uri", "")
                    ref_path = uri.replace("file:///", "").replace("%3A", ":").replace("/", "\\")
                    range_info = result.get("range", {})
                    start = range_info.get("start", {})

                    return Reference(
                        symbol_name="",
                        file_path=ref_path,
                        line=start.get("line", 0) + 1,
                        column=start.get("character", 0),
                        kind="definition"
                    )

            return None

        finally:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()


# ============================================================================
# Roslyn Analyzer (C#)
# ============================================================================

class RoslynAnalyzer:
    """
    C# semantic analyzer using Roslyn/OmniSharp.

    Requires:
    - dotnet SDK installed
    - OmniSharp server OR custom Roslyn tool

    Features:
    - Symbol extraction (classes, methods, properties, fields)
    - Reference finding
    - Type information
    """

    def __init__(self, solution_path: str, omnisharp_path: str = "omnisharp",
                 timeout: int = 120):
        """
        Initialize Roslyn analyzer.

        Args:
            solution_path: Path to .sln solution file
            omnisharp_path: Path to OmniSharp executable
            timeout: Timeout for analysis in seconds
        """
        self.solution_path = Path(solution_path).resolve()
        self.omnisharp_path = omnisharp_path
        self.timeout = timeout
        self._verified = False

        if not self.solution_path.exists():
            raise FileNotFoundError(f"Solution not found: {self.solution_path}")

    async def verify_installation(self) -> bool:
        """Verify OmniSharp or dotnet is installed."""
        if self._verified:
            return True

        # Check for dotnet SDK
        try:
            proc = await asyncio.create_subprocess_exec(
                "dotnet", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode == 0:
                version = stdout.decode('utf-8', errors='ignore').strip()
                logger.info(f"dotnet SDK found: {version}")
                self._verified = True
                return True

        except Exception as e:
            logger.error(f"dotnet SDK not found: {e}")
            logger.info("Install .NET SDK: https://dotnet.microsoft.com/download")
            return False

        return False

    async def analyze(self) -> Tuple[List[Symbol], List[Reference]]:
        """
        Analyze the C# solution and extract symbols and references.

        Returns:
            Tuple of (symbols, references) lists

        Raises:
            RuntimeError: If required tools are not available
        """
        if not await self.verify_installation():
            raise RuntimeError("Required tools not available")

        try:
            # Use custom Roslyn analysis tool
            symbols = await self._analyze_with_roslyn_tool()
            references = await self._extract_references_roslyn()

            logger.info(f"Analyzed {self.solution_path}: {len(symbols)} symbols, {len(references)} references")
            return symbols, references

        except Exception as e:
            logger.error(f"Failed to analyze {self.solution_path}: {e}")
            return [], []

    async def _analyze_with_roslyn_tool(self) -> List[Symbol]:
        """
        Analyze using custom Roslyn tool.

        This would require a separate C# console app that uses Roslyn APIs.
        For now, returns empty list as placeholder.
        """
        # TODO: Implement custom Roslyn analysis tool
        # The tool would:
        # 1. Load solution with Roslyn Workspace API
        # 2. Extract all symbols using SemanticModel
        # 3. Output as JSON

        logger.warning("Roslyn analysis tool not implemented yet")
        return []

    async def _extract_references_roslyn(self) -> List[Reference]:
        """Extract references using Roslyn (placeholder)."""
        # TODO: Implement with Roslyn FindReferences API
        return []

    async def analyze_with_omnisharp(self) -> Tuple[List[Symbol], List[Reference]]:
        """
        Analyze using OmniSharp LSP server (alternative approach).

        Requires OmniSharp server running.
        """
        # TODO: Implement OmniSharp LSP client
        logger.warning("OmniSharp analysis not implemented yet")
        return [], []


# ============================================================================
# Unified Analyzer Interface
# ============================================================================

class SemanticAnalyzer:
    """
    Unified semantic analyzer that dispatches to language-specific analyzers.

    Usage:
        analyzer = SemanticAnalyzer(
            cpp_compile_commands="build/compile_commands.json",
            csharp_solution="MyProject.sln"
        )
        symbols, references = await analyzer.analyze_path("src/")
    """

    def __init__(self, cpp_compile_commands: Optional[str] = None,
                 csharp_solution: Optional[str] = None,
                 clangd_path: str = "clangd",
                 omnisharp_path: str = "omnisharp"):
        """
        Initialize unified analyzer.

        Args:
            cpp_compile_commands: Path to C++ compile_commands.json directory
            csharp_solution: Path to C# solution file
            clangd_path: Path to clangd executable
            omnisharp_path: Path to OmniSharp executable
        """
        self.cpp_analyzer = None
        self.csharp_analyzer = None

        if cpp_compile_commands:
            self.cpp_analyzer = ClangdAnalyzer(
                compile_commands_path=cpp_compile_commands,
                clangd_path=clangd_path
            )

        if csharp_solution:
            self.csharp_analyzer = RoslynAnalyzer(
                solution_path=csharp_solution,
                omnisharp_path=omnisharp_path
            )

    async def analyze_path(self, path: str) -> Tuple[List[Symbol], List[Reference]]:
        """
        Analyze a file or directory, dispatching to appropriate analyzer.

        Args:
            path: File or directory path

        Returns:
            Tuple of (symbols, references) lists
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Determine language from extension or directory contents
        if path_obj.is_file():
            ext = path_obj.suffix.lower()

            if ext in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx']:
                if not self.cpp_analyzer:
                    raise RuntimeError("C++ analyzer not configured")
                return await self.cpp_analyzer.analyze_file(str(path))

            elif ext == '.cs':
                # For single C# file, would need project context
                logger.warning("Single C# file analysis requires solution context")
                return [], []

            else:
                logger.warning(f"Unsupported file type: {ext}")
                return [], []

        else:
            # Directory - try both analyzers
            all_symbols = []
            all_references = []

            if self.cpp_analyzer:
                cpp_symbols, cpp_refs = await self.cpp_analyzer.analyze_directory(str(path))
                all_symbols.extend(cpp_symbols)
                all_references.extend(cpp_refs)

            if self.csharp_analyzer:
                cs_symbols, cs_refs = await self.csharp_analyzer.analyze()
                all_symbols.extend(cs_symbols)
                all_references.extend(cs_refs)

            return all_symbols, all_references


# ============================================================================
# Utility Functions
# ============================================================================

def export_symbols_to_json(symbols: List[Symbol], output_path: str):
    """Export symbols to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in symbols], f, indent=2)
    logger.info(f"Exported {len(symbols)} symbols to {output_path}")


def export_references_to_json(references: List[Reference], output_path: str):
    """Export references to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in references], f, indent=2)
    logger.info(f"Exported {len(references)} references to {output_path}")


async def main():
    """CLI entry point for testing."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python semantic_analyzer.py <path> [--cpp-compile-commands <path>] [--csharp-solution <path>]")
        print("\nExample:")
        print("  python semantic_analyzer.py src/ --cpp-compile-commands build/")
        return

    path = sys.argv[1]
    cpp_compile_commands = None
    csharp_solution = None

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--cpp-compile-commands' and i + 1 < len(sys.argv):
            cpp_compile_commands = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--csharp-solution' and i + 1 < len(sys.argv):
            csharp_solution = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Create analyzer
    analyzer = SemanticAnalyzer(
        cpp_compile_commands=cpp_compile_commands,
        csharp_solution=csharp_solution
    )

    # Analyze
    print(f"\nAnalyzing: {path}")
    symbols, references = await analyzer.analyze_path(path)

    print(f"\nResults:")
    print(f"  Symbols: {len(symbols)}")
    print(f"  References: {len(references)}")

    # Show sample symbols
    if symbols:
        print("\nSample symbols:")
        for symbol in symbols[:10]:
            print(f"  {symbol.kind:15} {symbol.name:30} @ {symbol.file_path}:{symbol.line_start}")

    # Export results
    if symbols or references:
        output_dir = Path("semantic_analysis_output")
        output_dir.mkdir(exist_ok=True)

        if symbols:
            export_symbols_to_json(symbols, str(output_dir / "symbols.json"))

        if references:
            export_references_to_json(references, str(output_dir / "references.json"))


if __name__ == '__main__':
    asyncio.run(main())
