# Configuration Reference

Configuration is now layered in this order:

1. `config/settings.yaml` defaults
2. `.env` file values
3. process environment values
4. explicit CLI args (when provided)

Implementation entry point: `src/runtime_config.py`.

## Core Files

- Base config: `config/settings.yaml`
- Environment template: `.env.example`
- Runtime loader: `src/runtime_config.py`

## Environment Variables

### Server

- `AKASHIC_SETTINGS_PATH`: custom settings yaml path
- `AKASHIC_SERVER_HOST`: server host override
- `AKASHIC_SERVER_PORT`: server port override
- `AKASHIC_ENABLE_MCP_HTTP_ROUTES`: enable `/mcp/sse` + `/mcp/message` on HTTP server (`true`/`false`)
- `MCP_HOST`: MCP bind host
- `MCP_PORT`: MCP bind port
- `MCP_API_KEY`: SSE auth key

### Embedding

- `AKASHIC_EMBEDDING_URL`
- `AKASHIC_EMBEDDING_ENDPOINT`
- `AKASHIC_EMBEDDING_MODEL`
- `AKASHIC_EMBEDDING_DIMENSIONS`
- `AKASHIC_EMBEDDING_TIMEOUT`

### Qdrant

- `AKASHIC_QDRANT_HOST`
- `AKASHIC_QDRANT_PORT`
- `AKASHIC_QDRANT_COLLECTION`

### Reranker

- `AKASHIC_RERANKER_ENABLED`
- `AKASHIC_RERANKER_MODEL`
- `AKASHIC_RERANKER_TOP_K_CANDIDATES`
- `AKASHIC_RERANKER_TOP_K_RESULTS`

### Metadata and Logging

- `AKASHIC_METADATA_DB_PATH`
- `AKASHIC_LOG_LEVEL`
- `AKASHIC_LOG_FILE`
- `COMPILE_COMMANDS_PATH`
- `AKASHIC_CLANGD_PATH`

## Path Handling

- Relative paths are resolved from project root.
- `metadata.db` and log file paths are normalized to absolute internal runtime paths.
- In Docker, paths resolve from `/app`.

## Notes

- If reranker dependencies are missing, disable reranker via `AKASHIC_RERANKER_ENABLED=false`.
- Docker runtime dependency set is in `requirements.server.txt`.
- If your client only uses REST (`/api/call`), you can disable MCP HTTP routes by setting `AKASHIC_ENABLE_MCP_HTTP_ROUTES=false`.
