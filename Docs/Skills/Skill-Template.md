# Akashic MCP Skill Template

This template is for Claude Code skill setup without internal project-specific details.

## 1. Folder Layout

```text
~/.claude/skills/akashic-mcp/
|- SKILL.md
`- references/
   `- api-reference.md   # optional
```

## 2. `SKILL.md` Template

```markdown
---
name: akashic-mcp
description: >
  Use this skill when searching indexed code with natural language,
  locating symbol definitions/references, or retrieving targeted file context
  through Akashic MCP endpoints.
---

# Akashic MCP Skill

## When to use
- User asks where a feature is implemented
- User asks for symbol definitions or references
- User needs focused code context from indexed sources

## Recommended workflow
1. `search_code` for discovery
2. `get_symbol` / `get_cpp_symbols` for exact target
3. `get_file_context` for local context
4. `find_cpp_references` / `go_to_definition` for navigation

## Endpoint notes
- REST tool call: `POST /api/call`
- Optional MCP routes: `GET /mcp/sse`, `POST /mcp/message`
```

## 3. Example MCP Registration

### HTTP tool endpoint

```bash
claude mcp add -s user -t http akashic http://<HOST>:<PORT>/api
```

### SSE endpoint (when enabled)

```bash
claude mcp add -s user -t sse akashic http://<HOST>:<PORT>/mcp/sse
```

`/mcp/*` routes are enabled only when `AKASHIC_ENABLE_MCP_HTTP_ROUTES=true`.

## 4. Safety Notes

- Do not include internal IPs or private paths in shared skill files.
- Keep `SKILL.md` short and move long parameter docs to `references/*`.
