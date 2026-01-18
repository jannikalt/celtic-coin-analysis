"""I/O helpers for JSON and directory management."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return the Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dict as JSON to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """Load and return JSON as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
