"""Domain and regional scenario configurations for Indian nowcasting cases.

Provides preset meteorological scenarios across critical convective zones in India:
1. Uttarakhand Cloudburst (Himalayan orographic flash flood & extreme localized rain)
2. Delhi-NCR Severe Squall (Squall line, high MESH hail, >80 km/h downburst wind)
3. Kolkata Kalbaishakhi / Nor'wester (Explosive pre-monsoon convective initiation & lightning)
4. Central India (Vidarbha / Nagpur severe thunderstorm cluster)
"""

from __future__ import annotations

import copy
from typing import Any

from ..config import GRID, POIS, STEERING_WIND_MS

SCENARIOS: dict[str, dict[str, Any]] = {
    "uttarakhand_cloudburst": {
        "id": "uttarakhand_cloudburst",
        "name": "Uttarakhand Himalayan Cloudburst",
        "region": "Garhwal & Kumaon Himalayas",
        "description": "High-altitude orographic triggering with extreme localized precipitation (>100 mm/h) and valley flash-flood threat.",
        "hazard_focus": "Cloudburst & Flash Flood",
        "center_lat": 30.32,
        "center_lon": 78.03,
        "pois": {
            "Dehradun": (30.3165, 78.0322),
            "Rishikesh": (30.0869, 78.2676),
            "Haridwar": (29.9457, 78.1642),
            "Tehri": (30.3920, 78.4800),
            "Mussoorie": (30.4598, 78.0644),
        },
        "steering_ms": (8.0, 4.0),
        "elevation_base_m": 1200.0,
        "terrain_ridge_m": 2600.0,
        "hazard_type": "cloudburst",
    },
    "delhi_squall": {
        "id": "delhi_squall",
        "name": "Delhi-NCR Severe Squall & Downburst",
        "region": "Northern Plains & National Capital Region",
        "description": "Fast-moving linear convective squall line producing severe downburst winds (>80 km/h) and significant hail.",
        "hazard_focus": "Downburst Winds & Severe Hail",
        "center_lat": 28.61,
        "center_lon": 77.20,
        "pois": {
            "New Delhi": (28.6139, 77.2090),
            "Gurugram": (28.4595, 77.0266),
            "Noida": (28.5355, 77.3910),
            "Ghaziabad": (28.6692, 77.4538),
            "Faridabad": (28.4089, 77.3178),
        },
        "steering_ms": (16.0, -8.0),
        "elevation_base_m": 215.0,
        "terrain_ridge_m": 120.0,
        "hazard_type": "squall",
    },
    "kolkata_kalbaishakhi": {
        "id": "kolkata_kalbaishakhi",
        "name": "Kolkata Kalbaishakhi (Nor'wester)",
        "region": "Lower Gangetic Basin & Bengal Coast",
        "description": "Explosive pre-monsoon convective cell initiation moving from Chota Nagpur with rapid lightning jumps and severe microbursts.",
        "hazard_focus": "Lightning Jump & Severe Microburst",
        "center_lat": 22.57,
        "center_lon": 88.36,
        "pois": {
            "Kolkata": (22.5726, 88.3639),
            "Howrah": (22.5958, 88.2636),
            "Barasat": (22.7224, 88.4819),
            "Diamond Harbour": (22.1915, 88.1905),
            "Haldia": (22.0620, 88.0820),
        },
        "steering_ms": (14.0, -10.0),
        "elevation_base_m": 12.0,
        "terrain_ridge_m": 60.0,
        "hazard_type": "kalbaishakhi",
    },
    "nagpur_vidarbha": {
        "id": "nagpur_vidarbha",
        "name": "Central India Convective Cluster",
        "region": "Vidarbha / Maharashtra",
        "description": "Mesoscale convective system with multi-cell hail cores and high Vertically Integrated Liquid (VIL).",
        "hazard_focus": "Hail & Multicell Convection",
        "center_lat": 21.15,
        "center_lon": 79.09,
        "pois": {
            "Nagpur": (21.15, 79.09),
            "Kamptee": (21.22, 79.19),
            "Katol": (21.27, 78.58),
            "Saoner": (21.39, 78.92),
            "Butibori": (20.94, 79.02),
        },
        "steering_ms": (12.0, -6.0),
        "elevation_base_m": 310.0,
        "terrain_ridge_m": 450.0,
        "hazard_type": "multicell",
    },
}

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
    dlon = 2.3  # ~1 km cells at this latitude
    GRID["lon_left"] = round(center_lon - dlon, 3)
    GRID["lon_right"] = round(center_lon + dlon, 3)
    POIS.clear()
    POIS.update(pois)
    STEERING_WIND_MS["u"] = steering_ms[0]
    STEERING_WIND_MS["v"] = steering_ms[1]


def apply_scenario(scenario_id: str) -> dict[str, Any]:
    """Apply a predefined scenario's domain and parameters in place."""
    if scenario_id not in SCENARIOS:
        scenario_id = "uttarakhand_cloudburst"
    sc = SCENARIOS[scenario_id]
    set_domain(sc["center_lat"], sc["center_lon"], sc["pois"], sc["steering_ms"])
    return sc


def reset_domain() -> None:
    set_domain(_DEFAULT["GRID"]["center_lat"], _DEFAULT["GRID"]["center_lon"],
               _DEFAULT["POIS"], (_DEFAULT["STEERING"]["u"],
                                  _DEFAULT["STEERING"]["v"]))
