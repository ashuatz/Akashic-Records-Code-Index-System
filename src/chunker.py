"""
Code Chunking Module for Akashic Records

Uses Tree-sitter for language-aware code chunking at symbol boundaries.
Falls back to fixed-size chunks for unknown languages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import tree_sitter_c_sharp as ts_csharp
import tree_sitter_cpp as ts_cpp
import tree_sitter_python as ts_python
import tree_sitter_javascript as ts_javascript
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language, Parser, Node, Tree


logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code extracted from a file."""
    file_path: str
    code: str
    symbol_name: Optional[str]
    symbol_type: Optional[str]
    start_line: int
    end_line: int
    language: str


class CodeChunker:
    """
    Language-aware code chunker using Tree-sitter.

    Chunks code at natural boundaries (classes, methods, functions) while
    respecting token limits. Falls back to fixed-size chunks for unknown languages.
    """

    # Language file extension mappings
    LANGUAGE_EXTENSIONS = {
        '.cs': 'csharp',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mts': 'typescript',
        '.cts': 'typescript',
    }

    # AST node types to extract as chunks
    CHUNK_NODE_TYPES = {
        'csharp': {
            'class_declaration',
            'interface_declaration',
            'struct_declaration',
            'enum_declaration',
            'method_declaration',
            'property_declaration',
            'constructor_declaration',
            'destructor_declaration',
            'namespace_declaration',
        },
        'cpp': {
            'class_specifier',
            'struct_specifier',
            'enum_specifier',
            'function_definition',
            'namespace_definition',
            'template_declaration',
        },
        'python': {
            'class_definition',
            'function_definition',
            'decorated_definition',
        },
        'javascript': {
            'class_declaration',
            'function_declaration',
            'method_definition',
            'arrow_function',
            'function_expression',
            'export_statement',
        },
        'typescript': {
            'class_declaration',
            'interface_declaration',
            'type_alias_declaration',
            'enum_declaration',
            'function_declaration',
            'method_definition',
            'arrow_function',
            'function_expression',
            'export_statement',
        },
    }

    # Name identifier field names per language
    NAME_FIELDS = {
        'csharp': 'name',
        'cpp': 'declarator',
        'python': 'name',
        'javascript': 'name',
        'typescript': 'name',
    }

    def __init__(self, max_tokens: int = 4000, overlap: int = 200):
        """
        Initialize the code chunker.

        Args:
            max_tokens: Maximum tokens per chunk (approximate, uses chars * 0.3)
            overlap: Number of tokens to overlap between chunks
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        # Rough approximation: 1 token ~= 3.3 chars
        self.max_chars = int(max_tokens * 3.3)
        self.overlap_chars = int(overlap * 3.3)

        # Initialize Tree-sitter parsers
        self.parsers = {}
        self._init_parsers()

    def _init_parsers(self):
        """Initialize Tree-sitter parsers for supported languages."""
        try:
            # C#
            csharp_lang = Language(ts_csharp.language())
            csharp_parser = Parser(csharp_lang)
            self.parsers['csharp'] = csharp_parser

            # C++
            cpp_lang = Language(ts_cpp.language())
            cpp_parser = Parser(cpp_lang)
            self.parsers['cpp'] = cpp_parser

            # Python
            python_lang = Language(ts_python.language())
            python_parser = Parser(python_lang)
            self.parsers['python'] = python_parser

            # JavaScript
            js_lang = Language(ts_javascript.language())
            js_parser = Parser(js_lang)
            self.parsers['javascript'] = js_parser

            # TypeScript
            ts_lang = Language(ts_typescript.language_typescript())
            ts_parser = Parser(ts_lang)
            self.parsers['typescript'] = ts_parser

            logger.info(f"Initialized parsers for: {', '.join(self.parsers.keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter parsers: {e}")
            raise

    def detect_language(self, file_path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier or 'unknown'
        """
        ext = file_path.suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, 'unknown')

    def chunk_file(self, file_path: Path) -> list[CodeChunk]:
        """
        Chunk a file into code chunks.

        Args:
            file_path: Path to the file to chunk

        Returns:
            List of code chunks
        """
        try:
            # Read file with error handling
            content = self._read_file(file_path)
            if not content:
                return []

            # Detect language
            language = self.detect_language(file_path)

            # Chunk content
            return self.chunk_content(content, language, str(file_path))

        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            return []

    def chunk_content(self, content: str, language: str, file_path: str) -> list[CodeChunk]:
        """
        Chunk code content into structured chunks.

        Args:
            content: Source code content
            language: Programming language
            file_path: Original file path (for metadata)

        Returns:
            List of code chunks
        """
        # Unknown language - use fixed-size chunking
        if language not in self.parsers:
            return self._chunk_fixed_size(content, language, file_path)

        try:
            # Parse with Tree-sitter
            parser = self.parsers[language]
            tree = parser.parse(content.encode('utf-8'))

            # Extract chunks from AST
            chunks = self._extract_ast_chunks(tree, content, language, file_path)

            # If no chunks extracted or file is very small, return whole file
            if not chunks:
                return [CodeChunk(
                    file_path=file_path,
                    code=content,
                    symbol_name=None,
                    symbol_type='file',
                    start_line=1,
                    end_line=content.count('\n') + 1,
                    language=language
                )]

            # Split large chunks if needed
            final_chunks = []
            for chunk in chunks:
                if len(chunk.code) > self.max_chars:
                    final_chunks.extend(self._split_large_chunk(chunk))
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}, falling back to fixed-size")
            return self._chunk_fixed_size(content, language, file_path)

    def _read_file(self, file_path: Path) -> str:
        """
        Read file content with encoding error handling.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
            except (UnicodeDecodeError, OSError):
                continue

        logger.warning(f"Could not read {file_path} with any encoding")
        return ""

    def _extract_ast_chunks(self, tree: Tree, content: str, language: str, file_path: str) -> list[CodeChunk]:
        """
        Extract code chunks from AST tree.

        Args:
            tree: Parsed Tree-sitter tree
            content: Original source code
            language: Programming language
            file_path: Original file path

        Returns:
            List of code chunks
        """
        chunks = []
        chunk_types = self.CHUNK_NODE_TYPES.get(language, set())

        if not chunk_types:
            return []

        # Traverse tree and extract chunkable nodes
        def traverse(node: Node, parent_context: Optional[str] = None):
            # Check if this node type should be chunked
            if node.type in chunk_types:
                chunk = self._node_to_chunk(node, content, language, file_path, parent_context)
                if chunk:
                    chunks.append(chunk)

                    # Update parent context for nested nodes
                    new_context = chunk.symbol_name or parent_context
                    for child in node.children:
                        traverse(child, new_context)
            else:
                # Continue traversing
                for child in node.children:
                    traverse(child, parent_context)

        traverse(tree.root_node)

        # Sort by start line
        chunks.sort(key=lambda c: c.start_line)

        return chunks

    def _node_to_chunk(self, node: Node, content: str, language: str,
                       file_path: str, parent_context: Optional[str] = None) -> Optional[CodeChunk]:
        """
        Convert an AST node to a code chunk.

        Args:
            node: Tree-sitter node
            content: Original source code
            language: Programming language
            file_path: Original file path
            parent_context: Parent symbol name (e.g., class name for methods)

        Returns:
            CodeChunk or None if extraction fails
        """
        try:
            # Extract code text
            start_byte = node.start_byte
            end_byte = node.end_byte
            code = content[start_byte:end_byte]

            # Extract symbol name
            symbol_name = self._extract_symbol_name(node, language, content)

            # Add parent context if available
            if parent_context and symbol_name:
                symbol_name = f"{parent_context}.{symbol_name}"

            # Calculate line numbers
            start_line = node.start_point[0] + 1  # Tree-sitter uses 0-based
            end_line = node.end_point[0] + 1

            return CodeChunk(
                file_path=file_path,
                code=code,
                symbol_name=symbol_name,
                symbol_type=node.type,
                start_line=start_line,
                end_line=end_line,
                language=language
            )

        except Exception as e:
            logger.warning(f"Failed to convert node to chunk: {e}")
            return None

    def _extract_symbol_name(self, node: Node, language: str, content: str) -> Optional[str]:
        """
        Extract symbol name from AST node.

        Args:
            node: Tree-sitter node
            language: Programming language
            content: Original source code

        Returns:
            Symbol name or None
        """
        try:
            name_field = self.NAME_FIELDS.get(language, 'name')

            # Try to get name from direct field
            name_node = node.child_by_field_name(name_field)
            if name_node:
                return content[name_node.start_byte:name_node.end_byte]

            # Fallback: search for identifier child
            for child in node.children:
                if 'identifier' in child.type.lower():
                    return content[child.start_byte:child.end_byte]

            return None

        except Exception as e:
            logger.debug(f"Could not extract symbol name: {e}")
            return None

    def _chunk_fixed_size(self, content: str, language: str, file_path: str) -> list[CodeChunk]:
        """
        Chunk content using fixed-size chunks with overlap.

        Args:
            content: Source code content
            language: Programming language
            file_path: Original file path

        Returns:
            List of fixed-size code chunks
        """
        chunks = []
        lines = content.split('\n')

        if not lines:
            return []

        # Calculate lines per chunk based on average line length
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 80
        lines_per_chunk = max(1, int(self.max_chars / avg_line_length))
        overlap_lines = max(1, int(self.overlap_chars / avg_line_length))

        start_idx = 0
        chunk_idx = 0

        while start_idx < len(lines):
            end_idx = min(start_idx + lines_per_chunk, len(lines))
            chunk_lines = lines[start_idx:end_idx]
            code = '\n'.join(chunk_lines)

            chunks.append(CodeChunk(
                file_path=file_path,
                code=code,
                symbol_name=None,
                symbol_type='fixed_chunk',
                start_line=start_idx + 1,
                end_line=end_idx,
                language=language
            ))

            chunk_idx += 1
            start_idx = end_idx - overlap_lines if end_idx < len(lines) else end_idx

        return chunks

    def _split_large_chunk(self, chunk: CodeChunk) -> list[CodeChunk]:
        """
        Split a large chunk into smaller chunks at line boundaries.

        Args:
            chunk: Large code chunk to split

        Returns:
            List of smaller chunks
        """
        lines = chunk.code.split('\n')

        if len(lines) <= 1:
            return [chunk]

        # Calculate lines per sub-chunk
        avg_line_length = len(chunk.code) / len(lines)
        lines_per_chunk = max(1, int(self.max_chars / avg_line_length))
        overlap_lines = max(1, int(self.overlap_chars / avg_line_length))

        sub_chunks = []
        start_idx = 0

        while start_idx < len(lines):
            end_idx = min(start_idx + lines_per_chunk, len(lines))
            sub_lines = lines[start_idx:end_idx]
            code = '\n'.join(sub_lines)

            # Append part number to symbol name
            symbol_name = chunk.symbol_name
            if symbol_name and len(sub_chunks) > 0:
                symbol_name = f"{symbol_name}_part{len(sub_chunks) + 1}"

            sub_chunks.append(CodeChunk(
                file_path=chunk.file_path,
                code=code,
                symbol_name=symbol_name,
                symbol_type=chunk.symbol_type,
                start_line=chunk.start_line + start_idx,
                end_line=chunk.start_line + end_idx - 1,
                language=chunk.language
            ))

            start_idx = end_idx - overlap_lines if end_idx < len(lines) else end_idx

        return sub_chunks


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses rough approximation: 1 token ~= 3.3 characters.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return int(len(text) / 3.3)


if __name__ == '__main__':
    # Simple test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            chunker = CodeChunker(max_tokens=1000)
            chunks = chunker.chunk_file(test_file)

            print(f"\nChunked {test_file} into {len(chunks)} chunks:\n")
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:")
                print(f"  Symbol: {chunk.symbol_name or 'N/A'} ({chunk.symbol_type})")
                print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
                print(f"  Size: {len(chunk.code)} chars (~{estimate_tokens(chunk.code)} tokens)")
                print(f"  Preview: {chunk.code[:100]}...")
                print()
        else:
            print(f"File not found: {test_file}")
    else:
        print("Usage: python chunker.py <file_path>")
