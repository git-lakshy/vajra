"""Tier-1: Convective Initiation head (satellite-first).

Works where radar cannot see (Himalayan valley folds, radar-poor NE):
uses INSAT-style IR brightness temperature time-differencing:
  - cloud-top cooling rate (dT/dt < -4 K / 15 min)
  - TB10.8 crossing freezing isotherms
  - water-vapour/IR trend analogues
Outputs P(CI within 30/60/90 min) on the shared grid.
"""

from __future__ import annotations

import numpy as np

from ..config import THRESH
from ..ingest.store import SlabStore


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cloud_top_cooling_rate(store: SlabStore) -> np.ndarray | None:
    """K per 15 min from the last 3 frames of IR BT (5-min cadence)."""
    bts = store.history("bt", 3)
    if len(bts) < 3:
        return None
    return (bts[-1] - bts[-3]) / 3.0 * (15.0 / 5.0)


def ci_probability(store: SlabStore, lead_minutes: int) -> np.ndarray:
    """Gridded P(convective initiation within lead_minutes)."""
    bt = store.latest("bt")
    ny, nx = bt.shape if bt is not None else (512, 512)
    prior = np.full((ny, nx), 0.02, dtype=np.float32)

    cool = cloud_top_cooling_rate(store)
    if bt is None or cool is None:
        return prior

    # cooling trigger: stronger cooling => higher CI prob
    cool_trig = _sigmoid(((-cool) - (-THRESH["cooling_rate_K_per_15min"])) / 1.5)
    # deep-convection environment: current cold tops
    deep = _sigmoid((THRESH["bt_cold_K"] - bt) / 6.0)
    # glaciation proxy: BT well below freezing level
    glac = _sigmoid((THRESH["bt_freezing_K"] - bt) / 8.0)

    # horizon scaling: longer leads => higher chance of initiation
    horizon = {30: 0.55, 60: 0.75, 90: 0.9}.get(lead_minutes, 0.5)

    p = prior + horizon * 0.9 * cool_trig * (0.4 + 0.6 * glac) * (0.3 + 0.7 * deep)
    # cells already raining heavily: CI has happened -> decay prob of *new* CI
    refl = store.latest("refl")
    if refl is not None:
        existing = _sigmoid((refl - 45.0) / 5.0)
        p = p * (1.0 - 0.8 * existing)
    return np.clip(p, 0.0, 0.99).astype(np.float32)
