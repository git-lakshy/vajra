"""ML hazard heads: SEVIR-trained LightGBM models served on the live grid.

Features are the same 12 per-pixel predictors used in training
(scripts/train_sevir.py), computed on the 128x128 working grid (4 km cells,
24 km neighbourhood = 13-px window, matching the 24-px window at 1 km).
Probabilities are upsampled back to the 512x512 analysis grid.

Graceful degradation: if models/ are missing or lightgbm isn't installed,
`available` is False and the engine falls back to the physics heads.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

MODELS = "models"
FEATURE_NAMES = (
    "vil", "vil_max", "vil_p95", "vil_mean", "vil_trend", "vil_past_max",
    "bt_min", "bt_p5", "bt", "bt_trend", "bt_past_min", "vil_std",
)
WIN = 13           # 13 px x 4 km ~ 24-25 km radius neighbourhood
STRIDE = 4         # 512 -> 128 working grid


class MLHazardHeads:
    def __init__(self, models_dir: str = MODELS) -> None:
        self.available = False
        self.lightning = None
        self.severe = None
        self.iso: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            import lightgbm as lgb
            lp = os.path.join(models_dir, "lgbm_lightning.txt")
            sp = os.path.join(models_dir, "lgbm_severe.txt")
            if os.path.exists(lp) and os.path.exists(sp):
                self.lightning = lgb.Booster(model_file=lp)
                self.severe = lgb.Booster(model_file=sp)
                for name in ("lightning", "severe"):
                    pz = os.path.join(models_dir, f"isotonic_{name}.npz")
                    if os.path.exists(pz):
                        z = np.load(pz)
                        self.iso[name] = (z["X_thresholds_"],
                                          z["y_thresholds_"])
                self.available = True
        except Exception:
            self.available = False

    # ------------------------------------------------------------------
    def _isotonic(self, name: str, scores: np.ndarray) -> np.ndarray:
        if name not in self.iso:
            return scores
        xt, yt = self.iso[name]
        return np.interp(scores, xt, yt)

    def _features(self, store) -> np.ndarray | None:
        """Build the (N, 12) feature matrix on the working grid."""
        vil = store.latest("vil")
        bt = store.latest("bt")
        if vil is None:
            return None
        v = vil[::STRIDE, ::STRIDE].astype(np.float32)
        b = (bt[::STRIDE, ::STRIDE].astype(np.float32) if bt is not None
             else np.full_like(v, 273.0))
        # vil trend + 30-min past max from history (6 frames = 30 min)
        hist = [h[::STRIDE, ::STRIDE] for h in store.history("vil", 7)]
        if len(hist) >= 7:
            v_past0 = hist[0]
            v_past_max = np.max(np.stack(hist[:-1]), axis=0)
            b_hist = [h[::STRIDE, ::STRIDE] for h in store.history("bt", 7)]
            b_past0 = b_hist[0] if len(b_hist) >= 7 else b
            b_past_min = np.min(np.stack(b_hist[:-1]), axis=0)
        else:
            v_past0 = v
            v_past_max = v
            b_past0 = b
            b_past_min = b
        ny, nx = v.shape
        f = np.empty((ny, nx, 12), dtype=np.float32)
        mean1 = ndimage.uniform_filter(v, WIN)
        mean2 = ndimage.uniform_filter(v * v, WIN)
        f[..., 0] = v
        f[..., 1] = ndimage.maximum_filter(v, WIN)
        f[..., 2] = ndimage.percentile_filter(v, 95, WIN)
        f[..., 3] = mean1
        f[..., 4] = v - v_past0
        f[..., 5] = v_past_max
        f[..., 6] = ndimage.minimum_filter(b, WIN)
        f[..., 7] = ndimage.percentile_filter(b, 5, WIN)
        f[..., 8] = b
        f[..., 9] = b - b_past0
        f[..., 10] = b_past_min
        f[..., 11] = np.sqrt(np.maximum(mean2 - mean1 * mean1, 0.0))
        return f.reshape(-1, 12)

    def _to_grid(self, vals: np.ndarray) -> np.ndarray:
        ny = nx = int(np.sqrt(vals.size))
        g = vals.reshape(ny, nx).astype(np.float32)
        from scipy.ndimage import zoom
        return np.clip(zoom(g, 512 / ny, order=1), 0.0, 0.99)

    # ------------------------------------------------------------------
    def lightning_prob(self, store) -> np.ndarray | None:
        """P(lightning within ~24 km / 30 min) from the SEVIR model."""
        if not self.available:
            return None
        X = self._features(store)
        if X is None:
            return None
        raw = self.lightning.predict(X)
        return self._to_grid(self._isotonic("lightning", raw))

    def severe_prob(self, store) -> np.ndarray | None:
        """P(VIL >= 35 kg/m2 within ~24 km / 30 min) - hail proxy."""
        if not self.available:
            return None
        X = self._features(store)
        if X is None:
            return None
        raw = self.severe.predict(X)
        return self._to_grid(self._isotonic("severe", raw))
