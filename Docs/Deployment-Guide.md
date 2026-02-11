# Deployment Guide

This guide covers production-style setup for this repository.

## 1. Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Optional: host Ollama endpoint for embeddings

## 2. Environment Setup

Copy and edit environment values:

```bash
cp .env.example .env
```

Important keys:
- `AKASHIC_SERVER_PORT`
- `AKASHIC_ENABLE_MCP_HTTP_ROUTES`
- `AKASHIC_QDRANT_HOST`
- `AKASHIC_QDRANT_PORT`
- `AKASHIC_EMBEDDING_URL`
- `AKASHIC_METADATA_DB_PATH`
- `COMPILE_COMMANDS_PATH`

## 3. Start with Docker

```bash
docker compose up -d --build
```

Services:
- `akashic`: MCP HTTP server
- `qdrant`: vector store
- `ollama` (optional profile): local embedding server container

Enable optional Ollama container:

```bash
docker compose --profile ollama up -d
```

## 4. Verify

```bash
docker compose ps
curl http://localhost:8088/health
curl http://localhost:6333/collections
```

MCP endpoints are optional:
- enabled: `GET /mcp/sse`, `POST /mcp/message`
- disable with: `AKASHIC_ENABLE_MCP_HTTP_ROUTES=false`
- dedicated SSE server mode: `src/mcp_server_sse.py` uses `GET /sse`, `POST /messages`

## 5. Windows Helper Scripts

- Start: `start_all.bat`
- Stop: `stop_all.bat`
- Status: `status.bat`

These scripts are now compose-based and no longer contain machine-specific absolute paths.

## 6. Data and Volumes

Mounted local directory:
- `./data` -> `/app/data`

Docker named volumes:
- `qdrant_storage`
- `ollama_data` (optional service)

## 7. Initialize and Ingest

Initialize collection/schema:

```bash
python scripts/init_collection.py --settings config/settings.yaml
```

Run ingestion:

```bash
python scripts/ingest.py ingest --path "D:/Code/MyProject" --recursive
```

Both scripts load `.env` and support `AKASHIC_SETTINGS_PATH`.

## 8. Stop and Cleanup

Stop services:

```bash
docker compose down
```

Stop and remove volumes:

```bash
docker compose down -v
```
