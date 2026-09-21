"""Slab store: rolling 6-hour ring buffer of gridded fields.

The doc's Tier-0 output: 1 km grid, 5-minute slabs, rolling 6 h.
MVP keeps fields in memory (float32); production writes to Zarr.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from ..config import FRAME_MINUTES


class SlabStore:
    """Ring buffer of per-field gridded slabs keyed by frame index."""

    def __init__(self, max_hours: float = 6.0) -> None:
        self.max_frames = int(max_hours * 60 / FRAME_MINUTES)
        self.frames: Deque[int] = deque(maxlen=self.max_frames)
        self.data: dict[str, Deque[np.ndarray]] = {}
        self.strokes: Deque[list[tuple[float, float]]] = deque(maxlen=self.max_frames)
        self.terrain: np.ndarray | None = None

    def add(self, frame: int, fields: dict[str, np.ndarray]) -> None:
        if frame in self.frames:
            return
        for k, v in fields.items():
            if k == "strokes":
                continue
            if k == "terrain":
                self.terrain = v
                continue
            if v is None:
                continue
            if k not in self.data:
                self.data[k] = deque(maxlen=self.max_frames)
            self.data[k].append(v.astype(np.float32))
        if "strokes" in fields:
            self.strokes.append(fields["strokes"])
        self.frames.append(frame)

    def latest(self, field: str) -> np.ndarray | None:
        q = self.data.get(field)
        return q[-1] if q else None

    def history(self, field: str, n: int) -> list[np.ndarray]:
        q = self.data.get(field, [])
        return list(q)[-n:]

    def n_frames(self) -> int:
        return len(self.frames)

    def latest_frame(self) -> int | None:
        return self.frames[-1] if self.frames else None
