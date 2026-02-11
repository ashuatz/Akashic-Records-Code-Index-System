# Akashic Records - Implementation Checklist

## Overview

This document tracks the implementation status of all components required for the Akashic Records MCP server.

---

## Deployment Readiness (2026-02-11) - COMPLETE

| Item | Status | File |
|------|--------|------|
| `.env` template | ?? | `.env.example` |
| Runtime env loader | ?? | `src/runtime_config.py` |
| Docker image | ?? | `Dockerfile` |
| Compose orchestration | ?? | `docker-compose.yml` |
| Runtime dependency split | ?? | `requirements.server.txt` |
| Git ignore hygiene | ?? | `.gitignore` |
| Windows compose scripts | ?? | `start_all.bat`, `stop_all.bat`, `status.bat` |

---

## Stage 1: Project Structure ✅ COMPLETE

| Item | Status | File |
|------|--------|------|
| `src/__init__.py` | ✅ | `src/__init__.py` |
| `src/code_store.py` | ✅ | `src/code_store.py` |
| `src/semantic_analyzer.py` | ✅ | `src/semantic_analyzer.py` |
| `src/metadata_store.py` | ✅ | `src/metadata_store.py` |
| `src/chunker.py` | ✅ | `src/chunker.py` |
| `src/mcp_server.py` | ✅ | `src/mcp_server.py` |
| `requirements.txt` | ✅ | `requirements.txt` |
| `config/settings.yaml` | ✅ | `config/settings.yaml` |

**Note**: `embedder.py` functionality is integrated into `code_store.py` (EmbeddingClient class)

---

## Stage 2: SQLite Metadata Store ✅ COMPLETE

| Item | Status | Location |
|------|--------|----------|
| `symbols` table | ✅ | `scripts/init_collection.py` |
| `references` table | ✅ | `scripts/init_collection.py` |
| `dependencies` table | ✅ | `scripts/init_collection.py` |
| `file_hashes` table | ✅ | `scripts/init_collection.py` |
| `files` table | ✅ | `scripts/init_collection.py` |
| `chunks` table | ✅ | `scripts/init_collection.py` |
| FTS5 virtual tables | ✅ | `scripts/init_collection.py` |
| DB Abstraction Layer | ✅ | `src/metadata_store.py` |

---

## Stage 3: Chunker + Embedder ✅ COMPLETE

| Item | Status | File |
|------|--------|------|
| Tree-sitter chunking | ✅ | `src/chunker.py` |
| C# support | ✅ | `src/chunker.py` |
| C++ support | ✅ | `src/chunker.py` |
| Python support | ✅ | `src/chunker.py` |
| JavaScript/TypeScript | ✅ | `src/chunker.py` |
| Configurable chunk size | ✅ | `src/chunker.py` (max_tokens param) |
| llama.cpp embedding client | ✅ | `src/code_store.py` (EmbeddingClient) |
| Batch embedding | ✅ | `src/code_store.py` |

---

## Stage 4: Qdrant + Reranker ✅ COMPLETE

| Item | Status | File |
|------|--------|------|
| Qdrant client | ✅ | `src/code_store.py` |
| Collection management | ✅ | `scripts/init_collection.py` |
| BGE-Reranker integration | ✅ | `src/code_store.py` (Reranker class) |
| Vector search | ✅ | `src/code_store.py` |
| Hybrid search (vector + BM25) | ✅ | `src/code_store.py` |

---

## Stage 5: Indexing Orchestrator ✅ COMPLETE

| Item | Status | File |
|------|--------|------|
| Worker pool | ✅ | `src/orchestrator.py` |
| Hash-based change detection | ✅ | `src/orchestrator.py` |
| Resumable state | ✅ | `src/orchestrator.py` |
| File discovery | ✅ | `src/orchestrator.py` |
| Retry with backoff | ✅ | `src/orchestrator.py` |
| File System Watcher | ✅ | `src/file_watcher.py` |

---

## Stage 6: MCP Server ✅ COMPLETE

| Tool | Status | Description |
|------|--------|-------------|
| `search_code` | ✅ | Natural language code search |
| `get_symbol` | ✅ | Symbol definition lookup |
| `get_references` | ✅ | Find all symbol references |
| `get_dependencies` | ✅ | Symbol dependency graph |
| `get_file_context` | ✅ | File content retrieval |
| `list_indexed_files` | ✅ | List indexed files |

**Mode**: stdio (standard for Claude Code integration)

---

## Additional Components (Multi-LLM Recommendations)

| Item | Status | File |
|------|--------|------|
| Benchmark Harness | ✅ | `scripts/benchmark.py` |
| Test Queries | ✅ | `benchmark/queries.yaml` |
| Visualization | ✅ | `scripts/visualize_benchmark.py` |
| Semantic Analyzer (clangd) | ✅ | `src/semantic_analyzer.py` |
| Semantic Analyzer (Roslyn) | ✅ | `src/semantic_analyzer.py` |

---

## Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/init_collection.py` | ✅ | Initialize Qdrant + SQLite |
| `scripts/ingest.py` | ✅ | Index codebase CLI |
| `scripts/benchmark.py` | ✅ | Run relevance benchmarks |
| `scripts/run_benchmark_suite.py` | ✅ | Run benchmark suites |
| `scripts/visualize_benchmark.py` | ✅ | Visualize benchmark results |

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. Start Embedding server (llama.cpp)
./llama-server --model nomic-embed-text-v1.5.Q6_K.gguf --embedding --port 8081

# 4. Initialize databases
python scripts/init_collection.py --reset

# 5. Index a codebase
python scripts/ingest.py ingest --path "D:/Code/MyProject"

# 6. Start MCP server (for Claude Code)
python src/mcp_server.py
```

---

## Configuration (config/settings.yaml)

Key settings to configure:

```yaml
qdrant:
  host: localhost
  port: 6333
  collection: code_index

embedding:
  url: http://localhost:8081
  model: nomic-embed-text
  dimensions: 768

reranker:
  enabled: true
  model: BAAI/bge-reranker-base

indexing:
  chunk:
    max_tokens: 4000  # Adjustable for benchmarking
    overlap_tokens: 200
```

---

## Status Summary

| Stage | Status |
|-------|--------|
| Stage 1: Project Structure | ✅ COMPLETE |
| Stage 2: SQLite Metadata | ✅ COMPLETE |
| Stage 3: Chunker + Embedder | ✅ COMPLETE |
| Stage 4: Qdrant + Reranker | ✅ COMPLETE |
| Stage 5: Orchestrator | ✅ COMPLETE |
| Stage 6: MCP Server | ✅ COMPLETE |
| **OVERALL** | **✅ 100% COMPLETE** |

---

*Last Updated: 2026-01-30*
