from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG: List[Dict[str, Any]] = [
    {
        "id": "core-tools",
        "url": "http://127.0.0.1:8000/sse",
        "description": "Core tool server (SSE)",
    }
]


def load_server_configs(config_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Load server configurations. If YAML is not available or file missing, return defaults.
    The expected YAML format is:

    tool_servers:
      - id: core-tools
        url: http://127.0.0.1:8000/sse
        description: "Core tools"
    """
    if config_path is None:
        # default to sibling YAML file if present
        here = os.path.dirname(__file__)
        config_path = os.path.join(here, "tool_servers.yaml")

    if not os.path.exists(config_path) or yaml is None:
        return DEFAULT_CONFIG

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    servers = data.get("tool_servers")
    if isinstance(servers, list) and servers:
        return servers  # type: ignore[return-value]

    return DEFAULT_CONFIG
