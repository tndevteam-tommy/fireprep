"""Helpers for golden-file regression tests.

Set ``UNPREPARED_UPDATE_GOLDENS=1`` to (re)record goldens instead of comparing.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_GOLDEN_DIR = os.path.join(HERE, "goldens")
PIXEL_GOLDEN_DIR = os.path.join(HERE, "pixel_goldens")     # local only (renderer dependent), git-ignored


def updating() -> bool:
    return os.environ.get("UNPREPARED_UPDATE_GOLDENS") == "1"


def normalize(value: Any) -> Any:
    """JSON-friendly, platform-stable form of a snapshot (floats rounded, tuples -> lists)."""
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True).encode()).hexdigest()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")


def describe_diff(a, b, path="") -> list:
    """Human readable list of differences between two normalized structures."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing' if k not in a else 'extra'} in golden")
            else:
                out.extend(describe_diff(a[k], b[k], f"{path}/{k}"))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(describe_diff(x, y, f"{path}[{i}]"))
    elif a != b:
        out.append(f"{path}: golden={a!r} now={b!r}")
    return out
