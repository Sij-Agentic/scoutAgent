import json
import time
from pathlib import Path
from typing import Any, Optional

from scout_agent.custom_logging import get_logger
from scout_agent.config import get_config

logger = get_logger("data_cache.file_cache")


class FileCache:
    """Simple TTL-based JSON file cache under data_dir.

    Layout (relative to config.data_dir):
      reddit_cache/
        search/{key}.json
        threads/{post_id}.json
    """

    def __init__(self, base_subdir: str = "reddit_cache") -> None:
        cfg = get_config()
        self.base_dir = Path(cfg.data_dir) / base_subdir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, category: str, key: str) -> Path:
        p = self.base_dir / category
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{key}.json"

    @staticmethod
    def is_fresh(path: Path, ttl_seconds: int) -> bool:
        if not path.exists():
            return False
        try:
            mtime = path.stat().st_mtime
        except Exception:
            return False
        age = time.time() - mtime
        return age <= ttl_seconds

    def load(self, category: str, key: str, ttl_seconds: Optional[int]) -> Optional[Any]:
        path = self._path(category, key)
        if ttl_seconds is not None and not self.is_fresh(path, ttl_seconds):
            return None
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache {path}: {e}")
            return None

    def save(self, category: str, key: str, data: Any) -> Path:
        path = self._path(category, key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return path
        except Exception as e:
            logger.error(f"Failed to write cache {path}: {e}")
            raise
