"""MRMS replay adapter: real NOAA/NSSL radar mosaics as the ingestion source.

Reads the staged Dec 10-11 2021 case (scripts/download_mrms_case.ps1):
  MESH_00.50, MergedReflectivityQCComposite_00.50, PrecipRate_00.00,
  VIL_00.50, MergedAzShear_0-2kmAGL_00.50, RotationTrack30min_00.50

MRMS CONUS grid (verified via ecCodes): regular_ll, 7000x3500, 0.01 deg,
lat 54.995 (NW) -> 20.005, lon -129.995 -> -60.005. Files are gzipped
single-message GRIB2 (PNG-packed template); sentinels: refl -999, mesh -3.
"""

from __future__ import annotations

import gzip
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from scipy.ndimage import zoom

from ..grid import GRID

NY = NX = 512
PRODUCTS = {
    "refl": "MergedReflectivityQCComposite_00.50",
    "vil": "VIL_00.50",
    "rain": "PrecipRate_00.00",
    "mesh": "MESH_00.50",
    "shear": "MergedAzShear_0-2kmAGL_00.50",
    "rot": "RotationTrack30min_00.50",
}
# MRMS grid constants
LAT0, LON0, DL = 54.995, 230.005 - 360.0, 0.01
NROWS, NCOLS = 3500, 7000


def _subdomain_slices() -> tuple[slice, slice]:
    """Row/col slices of the MRMS grid covering a ~5.1 deg box around the
    configured domain centre (slightly padded for the 512x512 resample)."""
    clat, clon = GRID["center_lat"], GRID["center_lon"]
    half = 2.55
    i0 = int(round((LAT0 - (clat + half)) / DL))
    i1 = int(round((LAT0 - (clat - half)) / DL))
    j0 = int(round(((clon - half) - LON0) / DL))
    j1 = int(round(((clon + half) - LON0) / DL))
    i0 = max(0, min(i0, NROWS - 1)); i1 = max(i0 + 1, min(i1, NROWS))
    j0 = max(0, min(j0, NCOLS - 1)); j1 = max(j0 + 1, min(j1, NCOLS))
    return slice(i0, i1), slice(j0, j1)


def _resample512(sub: np.ndarray) -> np.ndarray:
    """Resample the cropped subdomain to the 512x512 analysis grid."""
    if sub.shape == (NY, NX):
        return sub.astype(np.float32)
    return zoom(sub, (NY / sub.shape[0], NX / sub.shape[1]),
                order=1).astype(np.float32)


@lru_cache(maxsize=4)
def _decode_grib(path: str) -> tuple[np.ndarray, float, float, float, float]:
    """Decode one gzipped MRMS GRIB2 file.

    Returns (values2d, lat_first, lon_first, dlat, dlon). MRMS products mix
    two grids: 0.01 deg (7000x3500) and 0.005 deg (14000x7000) — read the
    spec from the message instead of assuming. Rows run north -> south.
    """
    import eccodes as ec
    with gzip.open(path, "rb") as f:
        msg = f.read()
    gid = ec.codes_new_from_message(msg)
    try:
        ni = ec.codes_get(gid, "Ni")
        nj = ec.codes_get(gid, "Nj")
        lat0 = ec.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lon0 = ec.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        dlat = ec.codes_get(gid, "jDirectionIncrementInDegrees")
        dlon = ec.codes_get(gid, "iDirectionIncrementInDegrees")
        ec.codes_set(gid, "missingValue", -9999.0)
        vals = ec.codes_get_double_array(gid, "values").astype(np.float32)
    finally:
        ec.codes_release(gid)
    if vals.size != ni * nj:  # extra sentinel/flag points — trim defensively
        vals = vals[: ni * nj]
    grid = vals.reshape(nj, ni)
    return grid, lat0, lon0 - 360.0, dlat, dlon


def _clean(product: str, sub: np.ndarray) -> np.ndarray:
    sub = sub.copy()
    if product == "refl":
        sub[sub < -900] = 0.0        # -999 = no coverage
        sub = np.clip(sub, 0, 75)
    elif product == "mesh":
        sub[sub < 0] = 0.0           # -3 = no coverage
        sub = np.clip(sub, 0, 120)
    elif product == "rain":
        sub[sub < 0] = 0.0
        sub = np.clip(sub, 0, 250)
    elif product in ("shear", "rot"):
        # MRMS AzShear/RotationTrack: 0.005-deg grid, units 1e-3 s^-1
        sub[np.abs(sub) > 50] = 0.0  # sentinels
        sub = sub / 1000.0           # -> s^-1
    else:
        sub[sub < 0] = 0.0
    return sub


def _parse_ts(name: str) -> datetime:
    # MRMS_MESH_00.50_20211210-180040.grib2.gz
    stamp = name.split("_")[-1].replace(".grib2.gz", "")
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)


class MRMSCaseAdapter:
    """Serves engine frames (5-min cadence) from the staged MRMS case.

    Prefers the pre-decoded cache (data/cache/mrms2021/*.npy memmaps, built
    by scripts/build_cache.py, ~0.1 s/frame); falls back to on-the-fly GRIB
    decode when the cache is absent (~8 s/frame offline).
    """

    CACHE = "data/cache/mrms2021"
    CACHED_PRODUCTS = ("refl", "vil", "rain", "mesh", "shear", "rot", "bt")

    def __init__(self, case_dir: str,
                 start: datetime, end: datetime) -> None:
        self.start = start
        self.end = end
        self.slices = _subdomain_slices()
        self.mmaps: dict[str, np.ndarray] = {}
        cache_dir = os.path.join(os.path.dirname(case_dir), "..", "cache",
                                 "mrms2021")
        cache_dir = os.path.normpath(cache_dir)
        self.cache_dir = cache_dir
        manifest = os.path.join(cache_dir, "manifest.json")
        ok = os.path.exists(manifest)
        if ok:
            try:
                for k in self.CACHED_PRODUCTS:
                    p = os.path.join(cache_dir, f"{k}.npy")
                    if not os.path.exists(p):
                        ok = False
                        break
                    self.mmaps[k] = np.lib.format.open_memmap(p, mode="r")
            except Exception:
                ok = False
                self.mmaps = {}
        self.use_cache = ok
        if ok:
            self.n_frames = min(self.mmaps[k].shape[0]
                                for k in self.CACHED_PRODUCTS)
        self.files: dict[str, list[tuple[datetime, str]]] = {}
        for key, prod in PRODUCTS.items():
            d = os.path.join(case_dir, prod)
            if not os.path.isdir(d):
                self.files[key] = []
                continue
            entries = []
            for fn in os.listdir(d):
                if not fn.endswith(".grib2.gz"):
                    continue
                ts = _parse_ts(fn)
                if start - timedelta(minutes=3) <= ts <= end + timedelta(minutes=3):
                    entries.append((ts, os.path.join(d, fn)))
            self.files[key] = sorted(entries)
        self.n_frames = int((end - start).total_seconds() // 300)
        self._cache: dict[str, np.ndarray] = {}

    def _nearest(self, key: str, ts: datetime) -> str | None:
        best, dt = None, None
        for t, p in self.files.get(key, []):
            d = abs((t - ts).total_seconds())
            if dt is None or d < dt:
                best, dt = p, d
        return best

    def _product(self, key: str, ts: datetime) -> np.ndarray | None:
        path = self._nearest(key, ts)
        if path is None:
            return None
        grid, lat0, lon0, dlat, dlon = _decode_grib(path)
        clat, clon = GRID["center_lat"], GRID["center_lon"]
        half = 2.55
        i0 = int(round((lat0 - (clat + half)) / dlat))
        i1 = int(round((lat0 - (clat - half)) / dlat))
        j0 = int(round(((clon - half) - lon0) / dlon))
        j1 = int(round(((clon + half) - lon0) / dlon))
        nj, ni = grid.shape
        i0 = max(0, min(i0, nj - 1)); i1 = max(i0 + 1, min(i1, nj))
        j0 = max(0, min(j0, ni - 1)); j1 = max(j0 + 1, min(j1, ni))
        return _clean(key, _resample512(grid[i0:i1, j0:j1]))

    def frame(self, index: int) -> dict[str, np.ndarray | list]:
        """Observation fields for replay frame `index` (5-min cadence)."""
        if self.use_cache:
            out: dict[str, np.ndarray | list] = {}
            for key in ("refl", "vil", "rain", "mesh", "shear", "rot"):
                out[key] = np.asarray(self.mmaps[key][index])
            out["bt_cached"] = np.asarray(self.mmaps["bt"][index])
            out.setdefault("strokes", [])
            return out
        ts = self.start + timedelta(minutes=5 * index)
        out = {}
        for key in ("refl", "vil", "rain", "mesh", "shear", "rot"):
            arr = self._product(key, ts)
            if arr is not None:
                out[key] = arr
        out.setdefault("refl", np.zeros((NY, NX), np.float32))
        out.setdefault("vil", np.zeros((NY, NX), np.float32))
        out.setdefault("rain", np.zeros((NY, NX), np.float32))
        out.setdefault("mesh", np.zeros((NY, NX), np.float32))
        out.setdefault("shear", np.zeros((NY, NX), np.float32))
        out.setdefault("strokes", [])
        return out

    def cached_strokes(self, index: int) -> list[tuple[float, float]] | None:
        """Precomputed GLM strokes for frame `index` (cache path only)."""
        if not self.use_cache:
            return None
        if not hasattr(self, "_strokes"):
            import json
            try:
                with open(os.path.join(self.cache_dir, "strokes.json")) as f:
                    self._strokes = json.load(f)
            except Exception:
                self._strokes = {}
        pts = self._strokes.get(str(index), [])
        return [(float(x), float(y)) for x, y in pts]
