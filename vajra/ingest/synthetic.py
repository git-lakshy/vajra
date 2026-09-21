"""Synthetic multi-source storm replay generator for Indian meteorological scenarios.

Produces physically-plausible convective cases on the 1 km grid at 5-minute cadence:
DWR composite reflectivity, VIL, rain rate, INSAT-style IR brightness temperature,
azimuthal shear, orography, and lightning strokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import GRID, FRAME_MINUTES, STEERING_WIND_MS
from ..grid import make_mesh
from .domains import SCENARIOS


@dataclass
class StormCell:
    x: float
    y: float
    radius: float          # km
    intensity: float       # 0..1 lifecycle intensity
    growth: float          # d(intensity)/frame
    vx: float              # km per frame (x)
    vy: float              # km per frame (y)
    age: int = 0
    max_age: int = 36
    peak_dbz: float = 62.0
    rotates: float = 1.0   # +1 supercell-ish, -1 anticyclonic
    cell_type: str = "standard"

    def step(self, rng: np.random.Generator) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        mid = self.max_age * 0.45
        env = 1.0 / (1.0 + np.exp(-(self.age - mid) / (self.max_age * 0.22)))
        self.intensity = float(np.clip(0.08 + 0.92 * env, 0.0, 1.0))
        self.radius *= (1.0 + 0.004 * self.intensity)
        self.vx += rng.normal(0, 0.02)
        self.vy += rng.normal(0, 0.02)


@dataclass
class SyntheticCase:
    """A seeded, reproducible severe-weather case for Indian domains."""

    scenario_id: str = "uttarakhand_cloudburst"
    n_cells_init: int = 3
    cells: list[StormCell] = field(default_factory=list)
    frame_index: int = 0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42))
    terrain: np.ndarray | None = None
    _mesh: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._mesh = make_mesh()
        ny, nx = GRID["ny"], GRID["nx"]
        yy, xx = np.mgrid[0:ny, 0:nx]

        sc = SCENARIOS.get(self.scenario_id, SCENARIOS["uttarakhand_cloudburst"])
        base_elev = sc.get("elevation_base_m", 500.0)
        ridge_elev = sc.get("terrain_ridge_m", 800.0)

        if self.scenario_id == "uttarakhand_cloudburst":
            # Steep Himalayan orographic ridge NW-to-SE with deep river valley corridors
            ridge = ridge_elev * np.exp(-((yy - (ny * 0.65 - xx * 0.25)) ** 2) / (2 * 50.0 ** 2))
            secondary_ridge = (ridge_elev * 0.7) * np.exp(-((yy - (ny * 0.85 - xx * 0.15)) ** 2) / (2 * 40.0 ** 2))
            valleys = 400.0 * np.sin(xx / 18.0) * np.cos(yy / 22.0)
            self.terrain = (base_elev + ridge + secondary_ridge + valleys).astype(np.float32)
        elif self.scenario_id == "delhi_squall":
            # Semi-arid Northern Plains with low Aravalli ridge fringe
            ridge = ridge_elev * np.exp(-((xx - ny * 0.3) ** 2) / (2 * 80.0 ** 2))
            self.terrain = (base_elev + ridge).astype(np.float32)
        elif self.scenario_id == "kolkata_kalbaishakhi":
            # Gangetic Delta flat terrain with gentle western slope
            slope = (nx - xx) / nx * ridge_elev
            self.terrain = (base_elev + slope).astype(np.float32)
        else:
            ridge = ridge_elev * np.exp(-((yy - (ny * 0.78 - xx * 0.35)) ** 2) / (2 * 70.0 ** 2))
            folds = 150.0 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
            self.terrain = (ridge + folds + base_elev).astype(np.float32)

        for _ in range(self.n_cells_init):
            self._spawn_cell()

    def _spawn_cell(self, strong: bool = False) -> None:
        ny, nx = GRID["ny"], GRID["nx"]
        # Influx from upwind boundaries
        u, v = STEERING_WIND_MS["u"], STEERING_WIND_MS["v"]
        km_per_frame = 5.0 / 60.0 * 3.6  # m/s -> km per 5-min frame

        if u >= 0:
            x = self.rng.uniform(nx * 0.05, nx * 0.35)
        else:
            x = self.rng.uniform(nx * 0.65, nx * 0.95)

        if v >= 0:
            y = self.rng.uniform(ny * 0.05, ny * 0.35)
        else:
            y = self.rng.uniform(ny * 0.65, ny * 0.95)

        peak_dbz = float(self.rng.uniform(54, 68) if strong else self.rng.uniform(48, 58))
        if self.scenario_id == "delhi_squall":
            peak_dbz = float(self.rng.uniform(58, 70))
        elif self.scenario_id == "uttarakhand_cloudburst":
            peak_dbz = float(self.rng.uniform(52, 66))

        self.cells.append(StormCell(
            x=x, y=y,
            radius=self.rng.uniform(10, 20),
            intensity=0.15,
            growth=0.0,
            vx=u * km_per_frame,
            vy=v * km_per_frame,
            max_age=int(self.rng.integers(28, 48)),
            peak_dbz=peak_dbz,
            rotates=float(self.rng.choice([-1.0, 1.0], p=[0.25, 0.75])),
            cell_type=self.scenario_id,
        ))

    def step(self) -> None:
        """Advance the scenario by one 5-minute frame."""
        self.frame_index += 1
        for c in self.cells:
            c.step(self.rng)
        # Periodic convective initiation
        if self.frame_index % 6 == 3:
            self._spawn_cell(strong=self.frame_index % 12 == 3)
        # Clean up cells exiting boundaries
        self.cells = [c for c in self.cells
                      if c.age < c.max_age and -20 <= c.x < GRID["nx"] + 40
                      and -20 <= c.y < GRID["ny"] + 40]

    def fields(self) -> dict[str, np.ndarray]:
        """Render observation fields on the 512x512 grid."""
        ny, nx = GRID["ny"], GRID["nx"]
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

        refl = np.full((ny, nx), 6.0)
        vil = np.zeros((ny, nx))
        rain = np.zeros((ny, nx))
        shear = np.zeros((ny, nx))
        # Clear-sky brightness temp with lapse rate
        bt = 280.0 - 0.006 * self.terrain

        for c in self.cells:
            r = max(c.radius, 2.0)
            d2 = (xx - c.x) ** 2 + (yy - c.y) ** 2
            core = np.exp(-d2 / (2 * (0.55 * r) ** 2))
            mid = np.exp(-d2 / (2 * (1.0 * r) ** 2))
            inten = c.intensity

            # Reflectivity
            refl = np.maximum(refl, c.peak_dbz * inten * core + 25.0 * mid)

            # Scenario-specific physical characteristics
            if self.scenario_id == "uttarakhand_cloudburst":
                # Orographic rain enhancement: extreme localized precipitation
                orographic_factor = 1.0 + np.clip(self.terrain / 1800.0, 0.0, 1.5)
                vil += 45.0 * inten * (core ** 1.3)
                rain += 135.0 * (inten ** 1.5) * core * orographic_factor
                shear += c.rotates * 0.008 * inten * mid
                bt -= 68.0 * inten * (core ** 0.8)
            elif self.scenario_id == "delhi_squall":
                # Severe downburst shear & large hail VIL
                vil += 65.0 * inten * (core ** 1.2)
                rain += 85.0 * (inten ** 1.4) * core
                ang = np.arctan2(yy - c.y, xx - c.x)
                shear += c.rotates * 0.020 * inten * mid * np.sin(2 * ang)
                bt -= 64.0 * inten * (core ** 0.8)
            elif self.scenario_id == "kolkata_kalbaishakhi":
                # Explosive cold cloud top & lightning
                vil += 50.0 * inten * (core ** 1.4)
                rain += 95.0 * (inten ** 1.5) * core
                ang = np.arctan2(yy - c.y, xx - c.x)
                shear += c.rotates * 0.015 * inten * mid * np.sin(2 * ang)
                bt -= 78.0 * inten * (core ** 0.85)  # Very cold anvil <-60C
            else:
                vil += 55.0 * inten * (core ** 1.4)
                rain += 110.0 * (inten ** 1.6) * core
                ang = np.arctan2(yy - c.y, xx - c.x)
                shear += c.rotates * 0.012 * inten * mid * np.sin(2 * ang)
                bt -= 62.0 * inten * (core ** 0.8) - 14.0 * mid

        noise = self.rng.normal(0, 1.4, size=(ny, nx))
        refl = np.clip(refl + noise, 0, 75)
        vil = np.clip(vil + self.rng.normal(0, 1.0, size=(ny, nx)), 0, 75)
        rain = np.clip(rain + self.rng.normal(0, 1.0, size=(ny, nx)), 0, 200)
        bt = np.clip(bt + self.rng.normal(0, 0.8, size=(ny, nx)), 180, 305)

        # Lightning stroke generation
        strokes: list[tuple[float, float]] = []
        for c in self.cells:
            # Kolkata & Delhi have higher lightning flash density
            mult = 22.0 if self.scenario_id == "kolkata_kalbaishakhi" else 15.0
            lam = mult * (c.intensity ** 2.2)
            n = self.rng.poisson(lam)
            for _ in range(n):
                sx = c.x + self.rng.normal(0, c.radius * 0.75)
                sy = c.y + self.rng.normal(0, c.radius * 0.75)
                if 0 <= sx < nx and 0 <= sy < ny:
                    strokes.append((float(sx), float(sy)))

        return {
            "refl": refl.astype(np.float32),
            "vil": vil.astype(np.float32),
            "rain": rain.astype(np.float32),
            "bt": bt.astype(np.float32),
            "shear": shear.astype(np.float32),
            "terrain": self.terrain,
            "strokes": strokes,
        }
