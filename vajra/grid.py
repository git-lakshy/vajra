"""Grid helpers: local 1 km analysis grid <-> lat/lon.

MVP uses a simple local plate-carree-style mapping over the small demo
domain (valid for a ~500 km box); production swaps in a Lambert Conformal
projection via pyproj without changing call sites.
"""

from __future__ import annotations

import numpy as np

from .config import GRID

_KM_PER_DEG_LAT = 110.574


def _km_per_deg_lon(lat: float) -> float:
    return 111.320 * np.cos(np.deg2rad(lat))


def make_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) meshes of shape (ny, nx) for the analysis grid."""
    ny, nx, dx = GRID["ny"], GRID["nx"], GRID["dx"]
    clat, clon = GRID["center_lat"], GRID["center_lon"]
    x = (np.arange(nx) - nx // 2) * dx          # km, west->east
    y = (ny // 2 - np.arange(ny)) * dx          # km, south->north
    klat = 1.0 / _KM_PER_DEG_LAT
    lat = clat + y * klat
    klon = 1.0 / _km_per_deg_lon(clat)
    lon = clon + x * klon
    return np.meshgrid(lat, lon, indexing="ij")


def km_to_deg_factors() -> tuple[float, float]:
    """(dlat_per_km, dlon_per_km) for converting grid km offsets to degrees."""
    return 1.0 / _KM_PER_DEG_LAT, 1.0 / _km_per_deg_lon(GRID["center_lat"])


def grid_xy_to_latlon(xi: float, yi: float) -> tuple[float, float]:
    """Convert grid column/row indices to (lat, lon)."""
    lat2d, lon2d = make_mesh()
    ny, nx = lat2d.shape
    xi = int(np.clip(xi, 0, nx - 1))
    yi = int(np.clip(yi, 0, ny - 1))
    return float(lat2d[yi, xi]), float(lon2d[yi, xi])


def latlon_to_grid_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert (lat, lon) to fractional grid (x_col, y_row)."""
    clat, clon = GRID["center_lat"], GRID["center_lon"]
    dlat, dlon = km_to_deg_factors()
    y_km = (lat - clat) / dlat
    x_km = (lon - clon) / dlon
    return GRID["nx"] / 2 + x_km / GRID["dx"], GRID["ny"] / 2 - y_km / GRID["dy"]


def latlon_mesh_flatten() -> tuple[np.ndarray, np.ndarray]:
    """Flat lat/lon arrays aligned with ravel((ny,nx)) grid arrays."""
    lat2d, lon2d = make_mesh()
    return lat2d.ravel(), lon2d.ravel()
