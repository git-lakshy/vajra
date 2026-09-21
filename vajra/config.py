"""Global configuration: analysis grid, hazard thresholds, latency budgets."""

# ---------------------------------------------------------------------------
# Analysis grid: 1 km Lambert-Conformal-style local grid over Vidarbha,
# NE India analogue region for the demo (Nagpur / Chhota Nagpur corridor,
# the classic Nor'wester path from the referenced MAUSAM study).
# ---------------------------------------------------------------------------
GRID = {
    "nx": 512,            # grid columns (1 km each) -> 512 km wide
    "ny": 512,            # grid rows  (1 km each) -> 512 km tall
    "dx": 1.0,            # km per cell
    "dy": 1.0,
    "center_lat": 21.15,  # Nagpur
    "center_lon": 79.09,
    "lat_top": 23.45,
    "lat_bottom": 18.85,
    "lon_left": 76.82,
    "lon_right": 81.36,
}

FRAME_MINUTES = 5         # analysis frame cadence (matching DWR volume scan)
NOWCAST_LEADS_MIN = [5, 10, 15, 30, 45, 60, 90, 120]   # Tier-2 output leads
CI_LEADS_MIN = [30, 60, 90]                            # Tier-1 CI leads
BLEND_LEADS_MIN = [150, 180, 240, 300, 360]            # Tier-3 NWP blend leads
N_ENSEMBLE = 12

# Latency budget (seconds) - the "<90 s ingest->alert" pitch claim
LATENCY_BUDGET_S = 90.0

# ---------------------------------------------------------------------------
# Physical thresholds (from the research notes; region/season tunable)
# ---------------------------------------------------------------------------
THRESH = {
    "refl_severe_dbz": 45.0,       # composite reflectivity storm threshold
    "refl_core_dbz": 50.0,         # 50 dBZ echo => severe per MAUSAM labels
    "vil_storm": 10.0,             # kg/m2, storm-scale threshold
    "vil_hail": 30.0,              # kg/m2, hail-favourable
    "mesh_mm": 19.0,               # MESH >= ~19 mm => significant hail proxy
    "rain_cloudburst_mmh": 100.0,  # IMD cloudburst threshold
    "cloudburst_area_km2": 20.0,
    "bt_cold_K": 240.0,            # TB10.8 below => deep convection
    "bt_freezing_K": 273.15,
    "cooling_rate_K_per_15min": -4.0,   # cloud-top cooling CI trigger
    "shear_downburst": 0.004,      # s^-1 azimuthal shear for downburst proxy
    "lightning_jump_sigma": 2.0,   # 2-sigma flash-rate jump
}

# Steering winds (m/s) fallback prior: pre-monsoon Nor'wester, NW -> SE.
# The MAUSAM study: storms move with the 3.6-5.0 km "driving winds".
STEERING_WIND_MS = {"u": 12.0, "v": -6.0}

# Hazard-ML fusion: weight on the learned severe-growth probability inside
# the hail head (max of MESH-map and weight*P_severe). Heuristic pending a
# fit on Indian label archives (TODO: fit HAIL_ML_WEIGHT by CSI sweep).
HAIL_ML_WEIGHT = 0.7

# Model version stamped into state + audit log for reproducibility
MODEL_VERSION = "0.2.0-advanced"

# Hazard demo POIs (villages/towns on the grid) for the countdown clock
POIS = {
    "Nagpur": (21.15, 79.09),
    "Kamptee": (21.22, 79.19),
    "Katol": (21.27, 78.58),
    "Saoner": (21.39, 78.92),
    "Butibori": (20.94, 79.02),
}

ROLE_VIEWS = ["district", "aviation", "agriculture"]
