"""Motion estimation (block cross-correlation, pySTEPS-style) and
semi-Lagrangian extrapolation.

MVP implementation:
- block-based normalized cross-correlation (PIV-style) for dense advection:
  robust on smooth storm fields where per-pixel Lucas-Kanade diverges
- semi-Lagrangian back-advection via scipy.ndimage.map_coordinates
- trend-based growth/decay with lead-time damping (avoids the blur blow-up
  of naive persistence-of-trend)
- stochastic ensemble by perturbing motion + growth (NowcastNet-lite
  surrogate; generative decoder is the production upgrade path)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import FRAME_MINUTES, N_ENSEMBLE


# ---------------------------------------------------------------------------
# Optical flow: pySTEPS-style sparse Lucas-Kanade + IDW interpolation.
# Reference: pysteps.motion.lucaskanade.dense_lucaskanade (OpenCV pyramidal
# LK on Shi-Tomasi corners, 3-sigma outlier removal, inverse-distance
# weighted interpolation to a dense grid).
#
# Convention (matches pysteps.extrapolation.semilagrangian):
#   u = displacement along columns (east +), v = displacement along rows
#   (south +), both in grid px/frame; future(x) = field(x - v*t).
# ---------------------------------------------------------------------------
def dense_optical_flow(prev: np.ndarray, cur: np.ndarray,
                       max_corners: int = 400, quality: float = 0.05,
                       min_distance: int = 8, nr_std_outlier: float = 3.0,
                       smooth_sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Dense motion field from two consecutive frames.

    Returns (u, v) arrays shaped like the inputs, or None when no reliable
    features are found (caller falls back to the steering-wind prior).
    """
    try:
        import cv2
    except ImportError:
        return None

    # OpenCV (5.x) LK requires CV_8U images; scale both frames with one range
    scale = max(prev.max(), cur.max(), 1e-6)
    p = np.ascontiguousarray(np.clip(prev / scale, 0, 1) * 255, dtype=np.uint8)
    c = np.ascontiguousarray(np.clip(cur / scale, 0, 1) * 255, dtype=np.uint8)

    pts = cv2.goodFeaturesToTrack(
        p, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance, blockSize=16)
    if pts is None or len(pts) < 1:
        return None
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        p, c, pts, None, winSize=(25, 25), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))
    good = st.ravel() == 1
    xy = pts[good][:, 0, :].astype(np.float64)     # (N, 2): x=col, y=row
    uv = (nxt[good][:, 0, :] - pts[good][:, 0, :]).astype(np.float64)

    # outlier rejection: 3-sigma on displacement magnitude (pySTEPS default)
    mag = np.hypot(uv[:, 0], uv[:, 1])
    med = np.median(mag)
    mad = np.median(np.abs(mag - med)) * 1.4826 + 1e-6
    keep = np.abs(mag - med) <= nr_std_outlier * mad
    xy, uv = xy[keep], uv[keep]

    ny, nx = prev.shape
    if xy.shape[0] < 4:
        # too few features for interpolation: uniform flow from the mean
        if xy.shape[0] < 1:
            return None
        mu, mv = float(np.mean(uv[:, 0])), float(np.mean(uv[:, 1]))
        return (np.full((ny, nx), mu), np.full((ny, nx), mv))

    # IDW interpolation of sparse vectors onto the dense grid
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float64),
                         np.arange(nx, dtype=np.float64), indexing="ij")
    k = min(10, xy.shape[0])
    dist, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], axis=1), k=k)
    dist = np.maximum(dist, 1e-3)
    wts = (1.0 / dist ** 2).reshape(-1, k)
    wts /= wts.sum(axis=1, keepdims=True)
    u = (wts * uv[idx.ravel(), 0].reshape(-1, k)).sum(axis=1).reshape(ny, nx)
    v = (wts * uv[idx.ravel(), 1].reshape(-1, k)).sum(axis=1).reshape(ny, nx)
    u = ndimage.gaussian_filter(u, smooth_sigma)
    v = ndimage.gaussian_filter(v, smooth_sigma)
    return np.clip(u, -30, 30), np.clip(v, -30, 30)


# ---------------------------------------------------------------------------
# Semi-Lagrangian extrapolation
# ---------------------------------------------------------------------------
def semi_lagrangian_advection(field: np.ndarray, u: np.ndarray, v: np.ndarray,
                              steps: int, damping: float = 1.0) -> np.ndarray:
    """Back-advect `field` by `steps` frames using flow (u, v).

    Future(x) = Field(x - v*steps): v is displacement per frame in
    (row, col) px units (u = col displacement, v = row displacement).
    """
    ny, nx = field.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    x_src = xx - u * steps * damping
    y_src = yy - v * steps * damping
    out = ndimage.map_coordinates(field, [y_src, x_src], order=1,
                                  mode="nearest")
    return out.astype(np.float32)


def _apply_trend(base: np.ndarray, prev: np.ndarray, steps: int,
                 trend_gain: float = 0.5, max_delta: float = 12.0) -> np.ndarray:
    """Damped per-frame growth/decay applied to the advected field.

    `base - prev` is a one-frame change; it must NOT be multiplied by the
    number of steps. Damping decays the trend with lead time, mirroring
    extrapolation skill decay.
    """
    delta = np.clip(base - prev, -max_delta, max_delta)
    damping = max(0.0, 1.0 - 0.06 * steps)
    return base + trend_gain * damping * delta


def extrapolate(field: np.ndarray, prev: np.ndarray | None,
                u: np.ndarray, v: np.ndarray,
                lead_minutes: int) -> np.ndarray:
    """Deterministic extrapolation to a lead time (minutes)."""
    steps = max(1, round(lead_minutes / FRAME_MINUTES))
    out = field
    for s in range(1, steps + 1):
        out = semi_lagrangian_advection(field, u, v, s)
        if prev is not None and s <= 6:
            adv_prev = semi_lagrangian_advection(prev, u, v, s)
            out = _apply_trend(out, adv_prev, s)
        out = np.clip(out, 0, 75)
    return out


def ensemble_forecast(field: np.ndarray, prev: np.ndarray | None,
                      u: np.ndarray, v: np.ndarray,
                      leads_minutes: list[int],
                      n_members: int = N_ENSEMBLE,
                      rng: np.random.Generator | None = None) -> list[list[np.ndarray]]:
    """Stochastic ensemble: [lead][member] arrays of extrapolated fields."""
    rng = rng or np.random.default_rng()
    members_motion: list[tuple[np.ndarray, np.ndarray]] = []
    for m in range(n_members):
        scale = rng.uniform(0.85, 1.15)
        ang = np.deg2rad(rng.uniform(-4, 4))
        um = scale * (u * np.cos(ang) - v * np.sin(ang))
        vm = scale * (u * np.sin(ang) + v * np.cos(ang))
        members_motion.append((um, vm))

    out: list[list[np.ndarray]] = []
    for lead in leads_minutes:
        steps = max(1, round(lead / FRAME_MINUTES))
        lead_frames: list[np.ndarray] = []
        for (um, vm) in members_motion:
            f = semi_lagrangian_advection(field, um, vm, steps)
            if prev is not None:
                adv_prev = semi_lagrangian_advection(prev, um, vm, steps)
                f = _apply_trend(f, adv_prev, steps,
                                 trend_gain=rng.uniform(0.2, 0.6))
            f = np.clip(f + rng.normal(0, 1.2, size=f.shape), 0, 75)
            lead_frames.append(f.astype(np.float32))
        out.append(lead_frames)
    return out


def mean_flow_from_prior(nx: int, ny: int, u_ms: float, v_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform steering-wind flow in px/frame (fallback when flow fails).

    u_ms/v_ms in m/s (u eastward, v northward). Northward wind => row
    index decreases => row displacement v_row = -v_ms.
    """
    px_per_frame = FRAME_MINUTES * 3.6 / 1000.0 * 1000.0 / 1000.0  # m/s -> km per 5 min = *0.3
    px_per_frame = FRAME_MINUTES / 60.0 * 3.6  # km per frame per m/s
    u = np.full((ny, nx), u_ms * px_per_frame, dtype=np.float64)     # east -> +col
    v = np.full((ny, nx), -v_ms * px_per_frame, dtype=np.float64)    # north -> -row
    return u, v


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Non-overlapping f x f block mean (downsample by integer factor)."""
    ny, nx = a.shape
    ny2, nx2 = ny - ny % f, nx - nx % f
    return a[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))


def upsample_to(a: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Bilinear upsample back to (ny, nx)."""
    from scipy.ndimage import zoom
    return zoom(a, (ny / a.shape[0], nx / a.shape[1]), order=1)
