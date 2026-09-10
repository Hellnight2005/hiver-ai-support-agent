import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any


class DiskCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, key_str: str) -> str:
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get(self, key_str: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{self._compute_key(key_str)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, key_str: str, data: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{self._compute_key(key_str)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
