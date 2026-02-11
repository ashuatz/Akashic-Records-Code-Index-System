# Akashic Records Code Index System

Production-ready MCP code index server with:
- `docker compose` one-command startup
- centralized `.env` configuration
- cleaned Git tracking for runtime artifacts

Korean version: `README.ko.md`

## What This Tool Is

This project is an MCP-based code search service for AI assistants.

It helps assistants:
- search large codebases by intent (not just keyword matching),
- retrieve focused code context,
- follow symbol/definition/reference workflows through tools.

It is designed to reduce prompt/context overhead and make answers more consistent.
This README intentionally excludes internal Unity source details and implementation-sensitive content.

## Quick Start

1. Copy environment file:
```bash
cp .env.example .env
```

2. Start services:
```bash
docker compose up -d --build
```

3. Check health:
```bash
docker compose ps
curl http://localhost:8088/health
```

Windows shortcuts:
- `start_all.bat`
- `stop_all.bat`
- `status.bat`

## Service Endpoints

- MCP HTTP server: `http://localhost:8088`
- Health check: `http://localhost:8088/health`
- Qdrant: `http://localhost:6333/collections`
- Optional MCP routes: `GET /mcp/sse`, `POST /mcp/message`

## Known Issues

- HTTP API authentication is not yet enforced for `/api/*` endpoints.
  - Current impact: network-exposed deployments can invoke tools without auth.
  - Temporary mitigation: run behind a private network, reverse proxy auth, or disable external ingress.

## Do You Need MCP Endpoints?

- If you only use REST tool calls (`/api/call`): MCP routes are not required.
- If you connect Claude/Desktop via MCP over SSE on this server: keep `/mcp/sse` and `/mcp/message`.
- If you run `src/mcp_server_sse.py`: that server uses `/sse` and `/messages` separately.

To disable MCP routes on the HTTP server:

```env
AKASHIC_ENABLE_MCP_HTTP_ROUTES=false
```

## Configuration

Use `.env` for deployment-specific values.

Key variables:
- `AKASHIC_SETTINGS_PATH`
- `AKASHIC_SERVER_HOST`
- `AKASHIC_SERVER_PORT`
- `AKASHIC_ENABLE_MCP_HTTP_ROUTES`
- `AKASHIC_QDRANT_HOST`
- `AKASHIC_QDRANT_PORT`
- `AKASHIC_EMBEDDING_URL`
- `AKASHIC_METADATA_DB_PATH`
- `COMPILE_COMMANDS_PATH`
- `AKASHIC_CLANGD_PATH`

Reference:
- `.env.example`
- `Docs/Configuration-Reference.md`

## Skill Composition (Claude Code)

If you want to use this tool as a Claude Code skill, use a minimal structure:

```text
~/.claude/skills/akashic-mcp/
|- SKILL.md
`- references/
   `- api-reference.md   # optional
```

`SKILL.md` should contain:
- `name`: stable, lowercase-hyphen name (for example: `akashic-mcp`)
- `description`: when this skill should trigger
- workflow guidance: which tool to call first, then follow-up tools

Recommended trigger intents:
- "where is X implemented"
- "find references to X"
- "show me code context around X"
- "go to definition of X"

Recommended tool flow:
1. `search_code` for discovery
2. `get_symbol` or `get_cpp_symbols` for exact target
3. `get_file_context` for focused reading
4. `find_cpp_references` / `go_to_definition` for navigation

Keep skill content lightweight:
- include only reusable workflow instructions,
- move long API details into `references/*`,
- avoid embedding repository-private or sensitive implementation specifics.

See also: `Docs/Skills/Skill-Template.md`

## Local Python Run

```bash
pip install -r requirements.txt
python src/mcp_server_http.py
```

`src/*` servers and `scripts/*` tools now load `.env` automatically.

## Optional Folders (Not Required For Runtime)

- `scripts/indexing/*`
  - Unity-source-specific batch helpers for bulk ingestion.
  - Not required to run MCP HTTP server in production.
  - These scripts now require `UNITY_SOURCE_ROOT` when you use them.
- `benchmark/*`
  - Offline quality/performance evaluation harness (MRR/Recall/NDCG/latency).
  - Not required for runtime deployment.

## Documentation

- `Docs/README.md`
- `Docs/Deployment-Guide.md`
- `Docs/Configuration-Reference.md`
