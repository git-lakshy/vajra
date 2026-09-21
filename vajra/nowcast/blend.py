"""Tier-3: lead-time-dependent blend.

Skill-crossover: extrapolation dominates early; CI + NWP take over after
~90-120 min. Weights are functions of lead time (verified offline).
"""

from __future__ import annotations

import numpy as np


def blend_weights(lead_minutes: int) -> tuple[float, float, float]:
    """Return (extrapolation, ci, nwp) weights that sum to 1."""
    if lead_minutes <= 60:
        return (0.9, 0.1, 0.0)
    if lead_minutes <= 120:
        t = (lead_minutes - 60) / 60.0
        w_ext = 0.9 - 0.45 * t
        w_ci = 0.1 + 0.25 * t
        w_nwp = 0.2 * t
        s = w_ext + w_ci + w_nwp
        return (w_ext / s, w_ci / s, w_nwp / s)
    if lead_minutes <= 240:
        return (0.25, 0.35, 0.40)
    return (0.10, 0.30, 0.60)


def blend_prob(p_extrap: np.ndarray, p_ci: np.ndarray,
               p_nwp: np.ndarray | None, lead_minutes: int) -> np.ndarray:
    w_ext, w_ci, w_nwp = blend_weights(lead_minutes)
    if p_nwp is None:
        w_ext = w_ext + w_nwp
        w_nwp = 0.0
    return np.clip(w_ext * p_extrap + w_ci * p_ci + w_nwp * (p_nwp or 0.0),
                   0.0, 0.99).astype(np.float32)
