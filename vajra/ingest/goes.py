"""GOES-16 adapter: ABI C13 brightness temperature + GLM lightning flashes.

Uses the staged Dec-2021 case files (scripts/download_goes_case.ps1):
  abi/OR_ABI-L1b-RadC-M6C13_G16_*.nc  (5-min CONUS, 1500x2500)
  glm/OR_GLM-L2-LCFA_G16_*.nc         (20-s flashes; :00-second subset kept)

BT from scaled radiance via in-file Planck variables:
  BT = (fk2 / ln(fk1/Rad + 1) - bc1) / bc2
GOES geos-projection (lon_0=-75.2) is transformed to lat/lon with the
rasterio/PROJ stack and resampled onto the analysis grid (local affine
approximation over the ~5 deg box; 2 km pixels make this exact enough).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates

from .mrms import NY, NX


def _parse_goes_ts(name: str, which: str = "s") -> datetime:
    m = re.search(rf"_{which}(\d{{14}})_", name)
    if not m:
        raise ValueError(name)
    s = m.group(1)  # YYYYDDDHHMMSSs (13 = YYYYDDDHHMMSS, 14th = tenth of s)
    return datetime.strptime(s[:13], "%Y%j%H%M%S").replace(
        microsecond=int(s[13]) * 100000, tzinfo=timezone.utc)


@lru_cache(maxsize=8)
def _abi_bt(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (BT 2D, lat 2D, lon 2D) for one ABI L1b file."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True)
    rad = ds["Rad"].values.astype(np.float64)
    fk1 = float(ds["planck_fk1"].values)
    fk2 = float(ds["planck_fk2"].values)
    bc1 = float(ds["planck_bc1"].values)
    bc2 = float(ds["planck_bc2"].values)
    bt = (fk2 / np.log(fk1 / np.clip(rad, 1.0, None) + 1.0) - bc1) / bc2
    proj = ds["goes_imager_projection"].attrs
    lon0 = float(proj["longitude_of_projection_origin"])
    h = float(proj["perspective_point_height"])
    x_rad = ds["x"].values.astype(np.float64)   # scan angles, radians
    y_rad = ds["y"].values.astype(np.float64)
    ds.close()
    bt = bt.astype(np.float32)
    lat, lon = _geos_to_latlon(x_rad * h, y_rad * h, lon0, h)
    return bt, lat, lon


def _geos_to_latlon(x: np.ndarray, y: np.ndarray, lon0: float, h: float):
    from pyproj import Transformer, CRS
    src = CRS.from_proj4(
        f"+proj=geos +h={h} +lon_0={lon0} +sweep=x +a=6378137 "
        "+b=6356752.31414 +units=m +no_defs")
    tr = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
    xx, yy = np.meshgrid(x, y)
    lon, lat = tr.transform(xx.ravel(), yy.ravel())
    ny, nx = len(y), len(x)
    return (np.asarray(lat).reshape(ny, nx), np.asarray(lon).reshape(ny, nx))


class GOESCaseAdapter:
    """Serves satellite BT + lightning strokes for engine frames."""

    def __init__(self, goes_dir: str) -> None:
        self.abi: list[tuple[datetime, str]] = []
        self.glm: list[tuple[datetime, str]] = []
        ad = os.path.join(goes_dir, "abi")
        gd = os.path.join(goes_dir, "glm")
        if os.path.isdir(ad):
            self.abi = sorted((_parse_goes_ts(f), os.path.join(ad, f))
                              for f in os.listdir(ad) if f.endswith(".nc"))
        if os.path.isdir(gd):
            self.glm = sorted((_parse_goes_ts(f), os.path.join(gd, f))
                              for f in os.listdir(gd) if f.endswith(".nc"))

    def bt(self, ts: datetime) -> np.ndarray | None:
        """Brightness temperature resampled to the 512x512 analysis grid."""
        best, dt = None, None
        for t, p in self.abi:
            d = abs((t - ts).total_seconds())
            if dt is None or d < dt:
                best, dt = p, d
        if best is None or dt > 300:
            return None
        return _resample_bt(*_abi_bt(best))

    def strokes(self, ts: datetime, win_min: float = 2.5) -> list[tuple[float, float]]:
        """GLM flashes in [ts-win, ts+win] as grid (x, y) fractional coords."""
        from ..grid import latlon_to_grid_xy
        from ..config import GRID
        out: list[tuple[float, float]] = []
        nx, ny = GRID["nx"], GRID["ny"]
        for t, p in self.glm:
            if abs((t - ts).total_seconds()) > win_min * 60:
                continue
            for (la, lo) in _glm_flashes(p):
                x, y = latlon_to_grid_xy(la, lo)
                if 0 <= x < nx and 0 <= y < ny:
                    out.append((x, y))
        return out


def _resample_bt(bt: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Sample the ABI field onto the analysis grid via local index mapping."""
    from ..grid import make_mesh
    lat2d, lon2d = make_mesh()
    lat_lo, lat_hi = float(lat2d.min()), float(lat2d.max())
    lon_lo, lon_hi = float(lon2d.min()), float(lon2d.max())
    # rows: ABI CONUS y increases northward; lat array is then increasing by row
    if lat[0, 0] > lat[-1, 0]:
        lat = lat[::-1]; bt = bt[::-1]
    if lon[0, 0] > lon[0, -1]:
        lon = lon[:, ::-1]; bt = bt[:, ::-1]
    latv = lat[:, 0]; lonv = lon[0, :]
    rows = np.where((latv >= lat_lo - 2) & (latv <= lat_hi + 2))[0]
    cols = np.where((lonv >= lon_lo - 2) & (lonv <= lon_hi + 2))[0]
    if len(rows) < 2 or len(cols) < 2:
        return np.full((NY, NX), 273.0, np.float32)
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    sub_lat = lat[r0:r1 + 1, c0:c1 + 1]
    sub_lon = lon[r0:r1 + 1, c0:c1 + 1]
    sub_bt = bt[r0:r1 + 1, c0:c1 + 1]
    # local affine approx: fit row = f(lat), col = f(lon) linearly
    fr = np.polyfit(sub_lat[:, 0], np.arange(len(rows)), 1)
    fc = np.polyfit(sub_lon[0, :], np.arange(len(cols)), 1)
    src_rows = np.polyval(fr, lat2d)
    src_cols = np.polyval(fc, lon2d)
    out = map_coordinates(sub_bt, [src_rows, src_cols], order=1,
                          mode="nearest")
    return np.clip(out, 180, 330).astype(np.float32)


@lru_cache(maxsize=64)
def _glm_flashes(path: str) -> list[tuple[float, float]]:
    import xarray as xr
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
        if "flash_lat" in ds:
            lat = ds["flash_lat"].values
            lon = ds["flash_lon"].values
        ds.close()
    except Exception:
        return []
    return [(float(a), float(b)) for a, b in zip(lat, lon)]
