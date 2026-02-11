"""
Runtime configuration helpers for Akashic Records.

Centralizes:
- .env loading
- settings.yaml resolution
- environment variable overrides
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_str(value: str) -> str:
    return value


def load_dotenv(env_path: Optional[str] = None, override: bool = False) -> Path:
    """Load simple KEY=VALUE pairs from .env into process environment."""
    candidate = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    if not candidate.exists():
        return candidate

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip optional surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return candidate


def resolve_settings_path(settings_path: Optional[str] = None) -> Path:
    """Resolve settings path from arg/env/default, relative to project root."""
    path_str = settings_path or os.environ.get("AKASHIC_SETTINGS_PATH")
    path = Path(path_str) if path_str else DEFAULT_SETTINGS_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _set_nested(config: Dict[str, Any], section: str, key: str, value: Any) -> None:
    if section not in config or not isinstance(config[section], dict):
        config[section] = {}
    config[section][key] = value


def _resolve_project_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply known environment variable overrides to settings dict."""

    overrides: Dict[str, tuple[str, str, Callable[[str], Any]]] = {
        "AKASHIC_SERVER_HOST": ("server", "host", _parse_str),
        "AKASHIC_SERVER_PORT": ("server", "port", _parse_int),
        "AKASHIC_EMBEDDING_URL": ("embedding", "url", _parse_str),
        "AKASHIC_EMBEDDING_ENDPOINT": ("embedding", "endpoint", _parse_str),
        "AKASHIC_EMBEDDING_MODEL": ("embedding", "model", _parse_str),
        "AKASHIC_EMBEDDING_DIMENSIONS": ("embedding", "dimensions", _parse_int),
        "AKASHIC_EMBEDDING_TIMEOUT": ("embedding", "timeout", _parse_int),
        "AKASHIC_QDRANT_HOST": ("qdrant", "host", _parse_str),
        "AKASHIC_QDRANT_PORT": ("qdrant", "port", _parse_int),
        "AKASHIC_QDRANT_COLLECTION": ("qdrant", "collection", _parse_str),
        "AKASHIC_RERANKER_ENABLED": ("reranker", "enabled", _parse_bool),
        "AKASHIC_RERANKER_MODEL": ("reranker", "model", _parse_str),
        "AKASHIC_RERANKER_TOP_K_CANDIDATES": ("reranker", "top_k_candidates", _parse_int),
        "AKASHIC_RERANKER_TOP_K_RESULTS": ("reranker", "top_k_results", _parse_int),
        "AKASHIC_METADATA_DB_PATH": ("metadata", "db_path", _parse_str),
        "AKASHIC_LOG_LEVEL": ("logging", "level", _parse_str),
        "AKASHIC_LOG_FILE": ("logging", "file", _parse_str),
    }

    for env_key, (section, key, parser) in overrides.items():
        raw_value = os.environ.get(env_key)
        if raw_value is None:
            continue
        try:
            _set_nested(config, section, key, parser(raw_value))
        except Exception:
            # Ignore malformed overrides; keep YAML value
            continue

    # Normalize known path settings to absolute paths for predictable behavior
    metadata = config.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("db_path"):
        metadata["db_path"] = _resolve_project_path(str(metadata["db_path"]))

    logging_cfg = config.get("logging", {})
    if isinstance(logging_cfg, dict) and logging_cfg.get("file"):
        logging_cfg["file"] = _resolve_project_path(str(logging_cfg["file"]))

    return config


def load_settings(settings_path: Optional[str] = None) -> Dict[str, Any]:
    """Load settings.yaml and apply .env + env var overrides."""
    load_dotenv()
    resolved = resolve_settings_path(settings_path)

    with open(resolved, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Invalid settings format: expected dict in {resolved}")

    return apply_env_overrides(config)

