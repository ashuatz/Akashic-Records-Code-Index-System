# Akashic Records - AI Code Index System

## Project Overview

**Purpose**: MCP server for AI-optimized code search across large game engine codebases (30GB-150GB)

**Core Problem Solved**: AI wastes context investigating dependencies. With Akashic, AI searches in natural language and gets only relevant code.

## Architecture (v3.0 - MCP Only)

```
┌─────────────────────────────────────────────────────────┐
│  Claude / ChatGPT / Gemini (External AI)                │
│                        │                                 │
│                        ▼ MCP call                        │
├─────────────────────────────────────────────────────────┤
│               AKASHIC MCP SERVER                         │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │  1. Embedding (nomic-embed-code)                 │   │
│   │  2. Vector Search (Qdrant)                       │   │
│   │  3. Rerank (BGE-Reranker-Base)                   │   │
│   │  4. Return: Code + Metadata                      │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Why no intermediate LLM?** External AI (Claude, etc.) handles query rewriting and summarization directly.

## Directory Structure

```
AkashicRecords/
├── src/
│   ├── __init__.py              # Package marker
│   ├── code_store.py            # Core storage/search logic
│   ├── semantic_analyzer.py     # clangd/Roslyn LSP integration
│   ├── metadata_store.py        # SQLite: symbols, references, dependencies
│   ├── chunker.py               # Tree-sitter based code chunking
│   ├── embedder.py              # Embedding generation (llama.cpp or sentence-transformers)
│   ├── indexer.py               # Codebase indexing pipeline
│   ├── mcp_server.py            # MCP server (stdio mode)
│   └── mcp_server_http.py       # MCP server (HTTP/SSE mode)
├── tools/
│   └── RoslynAnalyzer/          # C# semantic analysis tool (.NET)
│       ├── RoslynAnalyzer.csproj
│       └── Program.cs
├── data/
│   ├── metadata.db              # SQLite (symbols, references, dependencies, BM25)
│   └── qdrant/                  # Qdrant vector storage
├── config/
│   └── settings.yaml            # Configuration file
├── scripts/
│   ├── init_collection.py       # Qdrant collection initialization
│   └── ingest.py                # Codebase indexing CLI
├── Doc/                         # Documentation
│   ├── README.md
│   ├── Setup-Guide.md
│   ├── Indexing-Pipeline-Detail.md
│   ├── Codebase-Ingestion-Guide.md
│   ├── akashic-skill.md
│   ├── AI-Code-Indexing-Detailed-Implementation.md
│   └── AI-Code-Indexing-Deep-Analysis.md
├── requirements.txt
└── README.md
```

## Key Components

| Component | Role | Location | Status |
|-----------|------|----------|--------|
| **nomic-embed-text** | Code → 768-dim vector | Ollama (localhost:11434) | ✅ Active |
| **BGE-Reranker-Base** | Search result reranking | src/code_store.py | ✅ Active |
| **Qdrant** | Vector similarity search | localhost:6333 | ✅ Active |
| **SQLite** | Symbol table, BM25 index | data/metadata.db | ✅ Active |
| **clangd** | C++ semantic analysis (LSP) | src/semantic_analyzer.py | ✅ **Integrated** |
| **Roslyn** | C# semantic analysis | tools/RoslynAnalyzer/ | ⏳ Planned |
| **Tree-sitter** | Fast code structure parsing | src/chunker.py | ✅ Active |

### clangd Integration Details
- **Path**: `C:\Users\sangyoon\AppData\Local\Microsoft\WinGet\Packages\LLVM.clangd_...\bin\clangd.exe`
- **Version**: 21.1.8
- **compile_commands.json**: `C:\TA\unity\compile_commands.json` (7,265 entries)
- **Features**: Symbol extraction (classes, functions, methods, structs, enums, fields, variables)

## Indexing Pipeline (4 Phases)

### Phase 0: Prerequisites
- compile_commands.json for C++ (clangd)
- .csproj/.sln files for C# (Roslyn)

### Phase 1: Semantic Analysis
- `semantic_analyzer.py` communicates with clangd/Roslyn via LSP
- Extracts: symbols, references, dependencies
- Stores in SQLite `metadata.db`

### Phase 2: Chunking + Embedding
- `chunker.py` uses Tree-sitter for structure-aware chunking
- `embedder.py` generates 768-dim vectors via nomic-embed-code
- Max chunk size: 8000 tokens

### Phase 3: Storage
- Vectors → Qdrant (code_index collection)
- Metadata → SQLite (symbols, references, dependencies tables)

## SQLite Schema

```sql
-- Symbol definitions (classes, functions, methods)
CREATE TABLE symbols (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,          -- class, function, method, field, etc.
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    signature TEXT,
    parent_id TEXT,
    language TEXT
);

-- Symbol references (where symbols are used)
CREATE TABLE references (
    id INTEGER PRIMARY KEY,
    symbol_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line INTEGER,
    column INTEGER,
    kind TEXT,                   -- call, read, write
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- Symbol dependencies (what uses what)
CREATE TABLE dependencies (
    from_symbol_id TEXT NOT NULL,
    to_symbol_id TEXT NOT NULL,
    kind TEXT,                   -- inherits, uses, implements
    PRIMARY KEY (from_symbol_id, to_symbol_id)
);
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_code(query, language?, limit?)` | Natural language code search |
| `get_symbol(name, symbol_type?)` | Symbol lookup (VS Ctrl+T style) |
| `get_references(symbol, file?)` | Find all symbol usages |
| `get_file_context(file_path, start_line?, end_line?)` | Get code context from a file |
| `list_indexed_files(path_pattern?, language?)` | List indexed files |
| `get_dependencies(symbol, direction?)` | Symbol dependency graph |
| `analyze_cpp_file(file_path)` | **[clangd]** Extract C++ symbols via LSP |
| `get_cpp_symbols(name, file_path?, kind?)` | **[clangd]** Search C++ symbols by name |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Start Ollama with embedding model
ollama pull nomic-embed-text
ollama serve  # Runs on localhost:11434

# 4. Initialize collection
python scripts/init_collection.py

# 5. Index codebase (use --skip-errors for large codebases)
python scripts/ingest.py ingest --path "C:\TA\unity" --skip-errors

# 6. Start MCP HTTP server
python src/mcp_server_http.py --host 0.0.0.0 --port 8088
```

### Batch Scripts (Windows)
- `start_server.bat` - Launch MCP HTTP server
- `index_unity.bat` - Index Unity codebase

## Hardware Requirements

| Tier | GPU | RAM | Storage |
|------|-----|-----|---------|
| Minimum (CPU-only) | None | 32GB | 500GB NVMe |
| Recommended (30GB code) | RTX 3060 12GB | 64GB | 1TB NVMe |
| Large Scale (150GB code) | RTX 3060 12GB | 128GB | 2TB NVMe |

## Documentation

- **Setup-Guide.md**: Complete installation guide
- **Indexing-Pipeline-Detail.md**: Semantic analysis pipeline details
- **Codebase-Ingestion-Guide.md**: How to index codebases
- **akashic-skill.md**: Claude skill for auto-invocation
- **AI-Code-Indexing-Detailed-Implementation.md**: v3.0 technical spec
- **AI-Code-Indexing-Deep-Analysis.md**: Research and critic review
