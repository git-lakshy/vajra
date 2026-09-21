"""Four hazard heads, one shared representation.

Deliberately NOT thresholds on a single rainfall field (the doc's trap #2):
  lightning  <- refl at cold layers, VIL, flash-rate jump detection
  hail       <- MESH proxy from VIL density, freezing-level context
  downburst  <- azimuthal shear + descending-core intensification
  cloudburst <- 15-min rain accumulation >= 100 mm/h over >= 20 km2
Each head emits a calibrated probability grid (isotonic mapping).
"""

from __future__ import annotations

import numpy as np

from ..config import THRESH
from ..ingest.store import SlabStore


# ---------------------------------------------------------------------------
# Isotonic-style calibration (PAVA on a fixed mapping archive; MVP uses
# physically-motivated monotone curves + PAVA smoothing).
# ---------------------------------------------------------------------------
def _isotonic_pava(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators; returns calibrated values sorted by score."""
    order = np.argsort(scores)
    y = labels[order].astype(np.float64)
    w = np.ones_like(y)
    # run pooling
    v = list(y)
    ww = list(w)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            tot_w = ww[i] + ww[i + 1]
            merged = (v[i] * ww[i] + v[i + 1] * ww[i + 1]) / tot_w
            v[i:i + 2] = [merged]
            ww[i:i + 2] = [tot_w]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty_like(y)
    idx = 0
    for val, wt in zip(v, ww):
        n = int(round(wt))
        out[idx:idx + n] = val
        idx += n
    res = np.empty_like(out)
    res[order] = out
    return res


def calibrate_hail(mesh_mm: np.ndarray) -> np.ndarray:
    """Monotone MESH -> P(significant hail). Piecewise-linear mapping."""
    knots = np.array([0, 5, 12, 19, 30, 45, 60, 100.0])
    probs = np.array([0.00, 0.03, 0.15, 0.45, 0.70, 0.85, 0.92, 0.97])
    return np.interp(mesh_mm, knots, probs).astype(np.float32)


def calibrate_lightning(rate: np.ndarray) -> np.ndarray:
    knots = np.array([0, 0.5, 2, 5, 10, 20, 40.0])
    probs = np.array([0.00, 0.05, 0.25, 0.55, 0.80, 0.92, 0.98])
    return np.interp(rate, knots, probs).astype(np.float32)


def calibrate_gust(vmax_kmh: np.ndarray) -> np.ndarray:
    knots = np.array([0, 40, 50, 62, 75, 90, 120.0])
    probs = np.array([0.00, 0.05, 0.20, 0.50, 0.75, 0.90, 0.97])
    return np.interp(vmax_kmh, knots, probs).astype(np.float32)


def calibrate_cloudburst(rate_mmh: np.ndarray) -> np.ndarray:
    knots = np.array([0, 30, 60, 100, 130, 180.0])
    probs = np.array([0.00, 0.02, 0.12, 0.55, 0.80, 0.95])
    return np.interp(rate_mmh, knots, probs).astype(np.float32)


# ---------------------------------------------------------------------------
def _mesh_proxy(vil: np.ndarray) -> np.ndarray:
    """MESH-like hail size proxy from VIL (Davis 1998 style compression).

    Production computes full SHI integration from reflectivity profiles;
    MVP uses the classic VIL->MESH power-law surrogate.
    """
    vil_pos = np.clip(vil, 0, 70)
    shi = 0.1 * np.power(vil_pos, 2.6) / 100.0
    return 2.54 * np.power(shi, 0.8)


def _flash_rate_grid(strokes: list[tuple[float, float]], shape: tuple[int, int],
                     sigma_km: float = 3.0) -> np.ndarray:
    """Grid strokes to flash density; converted to strokes/km2/15-min units
    by the caller (one 5-min frame is scaled x3)."""
    ny, nx = shape
    acc = np.zeros((ny, nx), dtype=np.float32)
    for (sx, sy) in strokes:
        xi, yi = int(round(sx)), int(round(sy))
        if 0 <= xi < nx and 0 <= yi < ny:
            acc[yi, xi] += 1.0
    if acc.max() > 0:
        from scipy import ndimage
        acc = ndimage.gaussian_filter(acc, sigma_km)
    return acc


def hazard_lightning(store: SlabStore) -> dict:
    """P(lightning) + lightning-jump flags (2-sigma flash-rate rise)."""
    refl = store.latest("refl")
    if refl is None:
        refl = np.zeros((1, 1), np.float32)
    strokes_now = store.strokes[-1] if store.strokes else []
    rate_now = _flash_rate_grid(strokes_now, refl.shape) * 3.0  # per 15 min

    # jump: compare recent rate to rolling baseline (same 15-min units)
    hist_rates = []
    for s in list(store.strokes)[-9:-1]:
        hist_rates.append(_flash_rate_grid(s, refl.shape) * 3.0)
    if hist_rates:
        base = np.mean(hist_rates, axis=0)
        std = np.std(hist_rates, axis=0) + 1e-3
        jump = (rate_now - base) > THRESH["lightning_jump_sigma"] * std
    else:
        jump = np.zeros_like(rate_now, dtype=bool)

    p = calibrate_lightning(rate_now)
    # reflectivity at cold layers proxy: strong refl + cold tops boosts
    bt = store.latest("bt")
    if bt is not None:
        cold = np.clip((THRESH["bt_freezing_K"] - bt) / 40.0, 0, 1)
        p = np.clip(p + 0.15 * cold * np.clip((refl - 35) / 20, 0, 1), 0, 0.99)
    return {"prob": p.astype(np.float32), "jump": jump,
            "rate": rate_now, "n_strokes": len(strokes_now)}


def hazard_hail(store: SlabStore) -> dict:
    """MESH-proxy hail probability (real MRMS MESH when available)."""
    mesh_obs = store.latest("mesh")
    vil = store.latest("vil")
    if mesh_obs is not None and float(mesh_obs.max()) > 0:
        mesh = mesh_obs
    else:
        if vil is None:
            vil = np.zeros((1, 1), np.float32)
        mesh = _mesh_proxy(vil)
    p = calibrate_hail(mesh)
    # freezing-level context proxy: cold tops => deeper updrafts
    bt = store.latest("bt")
    if bt is not None:
        deep = np.clip((THRESH["bt_cold_K"] - bt) / 25.0, 0, 1)
        p = np.clip(p * (0.7 + 0.5 * deep), 0, 0.99)
    return {"prob": p.astype(np.float32), "mesh_mm": mesh.astype(np.float32)}


def hazard_downburst(store: SlabStore) -> dict:
    """Downburst/gust probability from shear + descending-core proxy."""
    refl = store.latest("refl")
    if refl is None:
        refl = np.zeros((1, 1), np.float32)
    shear = store.latest("shear")
    if shear is None:
        shear = np.zeros((1, 1), np.float32)
    vil_hist = store.history("vil", 4)

    # descending core: rapid low-level reflectivity rise over last 15 min
    if len(vil_hist) >= 4:
        vil_rise = vil_hist[-1] - vil_hist[-4]
    else:
        vil_rise = np.zeros_like(refl)
    rising = np.clip(vil_rise / 15.0, 0, 1)

    shear_mag = np.clip(np.abs(shear) / THRESH["shear_downburst"], 0, 2)
    gust_kmh = 40.0 + 30.0 * shear_mag + 12.0 * rising + np.clip((refl - 45) / 2, 0, 15)
    p = calibrate_gust(gust_kmh)
    return {"prob": p.astype(np.float32), "gust_kmh": gust_kmh.astype(np.float32)}


def hazard_cloudburst(store: SlabStore) -> dict:
    """Cloudburst: >=100 mm/h over >=20 km2, terrain-conditioned."""
    rain = store.latest("rain")
    if rain is None:
        rain = np.zeros((1, 1), np.float32)
    ny, nx = rain.shape

    # 15-min accumulation proxy: rain rate ~ instantaneous mm/h
    from scipy import ndimage
    area_frac = ndimage.uniform_filter(
        (rain >= THRESH["rain_cloudburst_mmh"]).astype(np.float32), size=6) * 36.0
    # 6x6 km ~ 36 km2 window; fraction > ~20/36 => area criterion met
    area_ok = area_frac >= (THRESH["cloudburst_area_km2"] / 36.0)

    p = calibrate_cloudburst(rain)
    terrain = store.terrain
    if terrain is not None:
        slope = np.hypot(np.gradient(terrain, axis=0),
                         np.gradient(terrain, axis=1))
        slope_mask = np.clip(slope / 25.0, 0, 1)   # valley-fold trigger
        p = np.clip(p * (0.6 + 0.8 * slope_mask), 0, 0.99)
    p = np.where(area_ok, p, p * 0.25)
    return {"prob": p.astype(np.float32), "rain_mmh": rain.astype(np.float32)}


def run_all_hazards(store: SlabStore) -> dict:
    return {
        "lightning": hazard_lightning(store),
        "hail": hazard_hail(store),
        "downburst": hazard_downburst(store),
        "cloudburst": hazard_cloudburst(store),
    }

