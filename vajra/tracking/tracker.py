"""Object-based storm tracking: detection -> association -> Kalman -> ETA.

Turns gridded prediction into the live countdown clock:
  "Hail core arrives near <POI> in 27 +/- 6 minutes."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

from ..config import GRID, THRESH
from ..grid import km_to_deg_factors, latlon_to_grid_xy


@dataclass
class Detection:
    cx: float                 # grid column (fractional)
    cy: float                 # grid row (fractional)
    area_px: int
    max_refl: float
    max_vil: float
    max_mesh: float


@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    vx: float = 0.0           # km/frame eastward
    vy: float = 0.0           # km/frame northward
    age: int = 0
    misses: int = 0
    cov: np.ndarray = field(default_factory=lambda: np.diag([400.0, 400.0]))
    history: list[tuple[float, float]] = field(default_factory=list)
    max_refl: float = 0.0
    max_vil: float = 0.0
    max_mesh: float = 0.0

    def speed_kmh(self) -> float:
        # velocities are km per 5-min frame; x12 frames/h -> km/h
        return float(np.hypot(self.vx, self.vy) * 12.0)


class StormTracker:
    """Kalman (constant-velocity) tracker over thresholded storm objects."""

    def __init__(self, gate_km: float = 25.0) -> None:
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.gate_px = gate_km / GRID["dx"]  # in grid px (1 km cells)

    # ------------------------------------------------------------------
    def detect(self, refl: np.ndarray, vil: np.ndarray, mesh: np.ndarray) -> list[Detection]:
        mask = (refl >= THRESH["refl_severe_dbz"]).astype(np.uint8)
        mask = ndimage.binary_opening(mask, np.ones((3, 3), np.uint8))
        labels, n = ndimage.label(mask)
        dets: list[Detection] = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if len(ys) < 8:  # <8 km2: ignore specks
                continue
            dets.append(Detection(
                cx=float(xs.mean()), cy=float(ys.mean()),
                area_px=int(len(ys)),
                max_refl=float(refl[ys, xs].max()),
                max_vil=float(vil[ys, xs].max()) if vil is not None else 0.0,
                max_mesh=float(mesh[ys, xs].max()) if mesh is not None else 0.0,
            ))
        return dets

    # ------------------------------------------------------------------
    def update(self, refl: np.ndarray, vil: np.ndarray, mesh: np.ndarray) -> list[Track]:
        dets = self.detect(refl, vil, mesh)
        # predict existing tracks forward
        preds = {tid: (t.cx + t.vx, t.cy + t.vy) for tid, t in self.tracks.items()}
        pairs: list[tuple[int, int, float]] = []
        for tid, (px, py) in preds.items():
            for di, d in enumerate(dets):
                dist = np.hypot(d.cx - px, d.cy - py)
                if dist <= self.gate_px:
                    pairs.append((tid, di, dist))
        pairs.sort(key=lambda x: x[2])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for tid, di, _ in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            t = self.tracks[tid]
            d = dets[di]
            # Kalman-style blend: prior position from motion, measurement = det
            meas = np.array([d.cx, d.cy])
            prior = np.array([t.cx + t.vx, t.cy + t.vy])
            k = 0.6
            new_pos = prior + k * (meas - prior)
            t.vx = 0.7 * t.vx + 0.3 * (new_pos[0] - t.cx)
            t.vy = 0.7 * t.vy + 0.3 * (new_pos[1] - t.cy)
            t.cx, t.cy = new_pos
            t.age += 1
            t.misses = 0
            t.max_refl, t.max_vil, t.max_mesh = d.max_refl, d.max_vil, d.max_mesh
            t.history.append((t.cx, t.cy))
        # unmatched tracks
        for tid in list(self.tracks):
            if tid not in used_tracks:
                t = self.tracks[tid]
                t.cx += t.vx
                t.cy += t.vy
                t.misses += 1
                if t.misses > 4:
                    del self.tracks[tid]
        # new tracks: splits (area >> parent's expected) get the parent's
        # velocity; genuinely new cells get the steering prior
        for di, d in enumerate(dets):
            if di in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            parent = None
            for t in self.tracks.values():
                dist = np.hypot(d.cx - t.cx, d.cy - t.cy)
                expected = max(8.0, np.sqrt(max(t.max_vil, 1.0) * 12.0) * 2.0)
                if dist <= self.gate_px and d.area_px > 3.0 * expected:
                    parent = t
                    break
            u, v = ((parent.vx, parent.vy) if parent
                    else self._prior_motion_from_flow())
            t = Track(tid=tid, cx=d.cx, cy=d.cy, max_refl=d.max_refl,
                      max_vil=d.max_vil, max_mesh=d.max_mesh)
            t.vx, t.vy = u, v
            t.history.append((d.cx, d.cy))
            self.tracks[tid] = t
        return list(self.tracks.values())

    def _prior_motion_from_flow(self) -> tuple[float, float]:
        from ..config import STEERING_WIND_MS
        km_frame = 5.0 / 60.0 * 3.6
        return STEERING_WIND_MS["u"] * km_frame, STEERING_WIND_MS["v"] * km_frame

    # ------------------------------------------------------------------
    def eta_to_poi(self, track: Track, poi_lat: float, poi_lon: float) -> Optional[dict]:
        """Arrival countdown: distance along motion / speed, +/- sigma."""
        x_poi, y_poi = latlon_to_grid_xy(poi_lat, poi_lon)
        dx, dy = x_poi - track.cx, y_poi - track.cy
        v = np.array([track.vx, track.vy])
        vnorm = np.hypot(*v)
        if vnorm < 0.3 or track.age < 2:
            return None
        # time for the cell edge (use radius ~ sqrt(area)) to reach POI
        along = (dx * v[0] + dy * v[1]) / vnorm
        if along <= 0:
            return None
        radius_px = max(6.0, np.sqrt(max(track.max_vil, 1.0) * 12.0))
        dist = max(along - radius_px, 0.0)
        speed_kmh = track.speed_kmh()
        if speed_kmh < 2:
            return None
        eta_min = dist / speed_kmh * 60.0
        # uncertainty calibrated from real-case verification (Dec-2021):
        # median error ~+7 min, MAE ~9.5 min -> sigma ~ 45% of ETA
        sigma = max(5.0, 0.45 * eta_min)
        return {
            "eta_min": round(float(eta_min), 1),
            "sigma_min": round(float(sigma), 1),
            "speed_kmh": round(float(speed_kmh), 1),
            # compass bearing: vx eastward (+), vy is ROW units (south +),
            # so the northward component is -vy
            "bearing_deg": round(float(np.degrees(np.arctan2(v[0], -v[1])) % 360), 0),
        }

    # ------------------------------------------------------------------
    def cell_polygons(self) -> list[dict]:
        """Rough circular footprint polygons per track (for GeoJSON)."""
        dlat, dlon = km_to_deg_factors()
        out = []
        for t in self.tracks.values():
            radius_km = max(6.0, np.sqrt(max(t.max_vil, 1.0) * 12.0))
            n_pts = 16
            ring = []
            for k in range(n_pts + 1):
                ang = 2 * np.pi * k / n_pts
                dy = radius_km * np.cos(ang)   # northward
                dx = radius_km * np.sin(ang)   # eastward
                lat = GRID["center_lat"] + (GRID["ny"] / 2 - t.cy + dy) * dlat
                lon = GRID["center_lon"] + (t.cx - GRID["nx"] / 2 + dx) * dlon
                ring.append([round(lon, 4), round(lat, 4)])
            out.append({"tid": t.tid, "ring": ring, "radius_km": radius_km})
        return out
