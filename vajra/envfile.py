"""Minimal .env loader (no extra dependency): KEY=VALUE lines -> os.environ.

Never overrides variables already present in the environment.
"""

from __future__ import annotations

import os


def load_dotenv(path: str = ".env") -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("\"'")
            if k and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
    return loaded


def write_dotenv(path: str, values: dict[str, str]) -> None:
    """Merge values into an existing .env (preserves comments/order)."""
    existing: dict[str, str] = {}
    order: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, v = s.split("=", 1)
                    existing[k.strip()] = v.strip()
                    order.append(k.strip())
    for k in values:
        if k not in order:
            order.append(k)
    existing.update(values)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# VAJRA alert credentials (DO NOT COMMIT)\n")
        for k in order:
            f.write(f"{k}={existing[k]}\n")
