"""Domain configuration for replay cases.

The default domain is the Nagpur synthetic case. Real replay cases
(e.g. MRMS Dec-2021) call `set_domain()` BEFORE the engine is constructed;
GRID/POIS/STEERING are mutated in place so every module sees the same
objects.
"""

from __future__ import annotations

import copy

from ..config import GRID, POIS, STEERING_WIND_MS

_DEFAULT = {
    "GRID": copy.deepcopy(GRID),
    "POIS": copy.deepcopy(POIS),
    "STEERING": dict(STEERING_WIND_MS),
}


def set_domain(center_lat: float, center_lon: float,
               pois: dict[str, tuple[float, float]],
               steering_ms: tuple[float, float]) -> None:
    """Re-point the analysis grid, POIs and steering-wind prior in place."""
    GRID["center_lat"] = center_lat
    GRID["center_lon"] = center_lon
    dlat = 2.3  # 512 km / (110.574 km/deg) ~ 2.31 deg half-span
    GRID["lat_top"] = round(center_lat + dlat, 3)
    GRID["lat_bottom"] = round(center_lat - dlat, 3)
    dlon = 2.3  # ~1 km cells at this latitude (cos-adjusted in grid.py)
    GRID["lon_left"] = round(center_lon - dlon, 3)
    GRID["lon_right"] = round(center_lon + dlon, 3)
    POIS.clear()
    POIS.update(pois)
    STEERING_WIND_MS["u"] = steering_ms[0]
    STEERING_WIND_MS["v"] = steering_ms[1]


def reset_domain() -> None:
    set_domain(_DEFAULT["GRID"]["center_lat"], _DEFAULT["GRID"]["center_lon"],
               _DEFAULT["POIS"], (_DEFAULT["STEERING"]["u"],
                                  _DEFAULT["STEERING"]["v"]))
