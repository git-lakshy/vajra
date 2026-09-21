"""Verification metrics: CSI/POD/FAR + FSS (fractions skill score)."""

from __future__ import annotations

import numpy as np


def contingency(forecast: np.ndarray, obs: np.ndarray, thr: float) -> dict:
    f = forecast >= thr
    o = obs >= thr
    hits = int(np.sum(f & o))
    misses = int(np.sum(~f & o))
    false_alarms = int(np.sum(f & ~o))
    correct_neg = int(np.sum(~f & ~o))
    pod = hits / (hits + misses) if (hits + misses) else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else 0.0
    # Wilks 2x2 table: a=hits, b=false_alarms, c=misses, d=correct_negatives
    denom = (hits + false_alarms) * (false_alarms + correct_neg) \
        + (hits + misses) * (misses + correct_neg)
    hss = (2 * (hits * correct_neg - misses * false_alarms) / denom) if denom else 0.0
    return {"hits": hits, "misses": misses, "fa": false_alarms,
            "csi": round(csi, 3), "pod": round(pod, 3),
            "far": round(far, 3), "hss": round(hss, 3)}


def _fraction_field(field: np.ndarray, thr: float, window: int) -> np.ndarray:
    from scipy import ndimage
    return ndimage.uniform_filter((field >= thr).astype(np.float32), window)


def fss(forecast: np.ndarray, obs: np.ndarray, thr: float, window_km: int) -> float:
    """Fractions Skill Score with a square neighbourhood window."""
    w = max(1, window_km)
    ff = _fraction_field(forecast, thr, w)
    of = _fraction_field(obs, thr, w)
    num = np.mean((ff - of) ** 2)
    den = np.mean(ff ** 2) + np.mean(of ** 2)
    return float(1.0 - num / den) if den > 0 else 1.0


def bt_threshold_baseline(bt: np.ndarray, thr_K: float = 240.0) -> np.ndarray:
    """Satellite/threshold baseline (§5.13): cold cloud tops (<240 K) as the
    convection proxy, scored like any other forecast field."""
    return (bt <= thr_K).astype(np.float32)
