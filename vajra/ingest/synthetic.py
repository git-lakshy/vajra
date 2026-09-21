"""Synthetic multi-source storm replay generator.

Produces a physically-plausible pre-monsoon convective case on the 1 km
grid at 5-minute cadence: DWR composite reflectivity + VIL + rain rate,
INSAT-style IR brightness temperature, azimuthal shear, orography and
lightning strokes. Used as the replay-mode data source when live feeds
are unavailable (the doc's "never depend on live severe weather" rule).
Real adapters (MRMS grib2, SEVIR h5, MOSDAC) implement the same frame API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import GRID, FRAME_MINUTES, STEERING_WIND_MS
from ..grid import make_mesh


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

    def step(self, rng: np.random.Generator) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        # logistic lifecycle: growth -> maturity -> decay
        mid = self.max_age * 0.45
        env = 1.0 / (1.0 + np.exp(-(self.age - mid) / (self.max_age * 0.22)))
        self.intensity = float(np.clip(0.08 + 0.92 * env, 0.0, 1.0))
        self.radius *= (1.0 + 0.004 * self.intensity)
        self.vx += rng.normal(0, 0.02)
        self.vy += rng.normal(0, 0.02)


@dataclass
class SyntheticCase:
    """A seeded, reproducible severe-weather afternoon (12:00-18:00 LT)."""

    n_cells_init: int = 3
    cells: list[StormCell] = field(default_factory=list)
    frame_index: int = 0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42))
    terrain: np.ndarray | None = None
    _mesh: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._mesh = make_mesh()
        lat2d, _ = self._mesh
        # synthetic orography: ridge SW->NE + valley folds (cloudburst zone)
        ny, nx = GRID["ny"], GRID["nx"]
        yy, xx = np.mgrid[0:ny, 0:nx]
        ridge = 900.0 * np.exp(-((yy - (ny * 0.78 - xx * 0.35)) ** 2) / (2 * 70.0 ** 2))
        folds = 220.0 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
        self.terrain = (ridge + folds + 350.0).astype(np.float32)
        for _ in range(self.n_cells_init):
            self._spawn_cell()

    def _spawn_cell(self, strong: bool = False) -> None:
        ny, nx = GRID["ny"], GRID["nx"]
        # cells initiate over the Chhota Nagpur-ish upwind (NW) quadrant
        x = self.rng.uniform(nx * 0.05, nx * 0.45)
        y = self.rng.uniform(ny * 0.55, ny * 0.95)
        u, v = STEERING_WIND_MS["u"], STEERING_WIND_MS["v"]
        km_per_frame = 5.0 / 60.0 * 3.6  # m/s -> km per 5-min frame
        self.cells.append(StormCell(
            x=x, y=y,
            radius=self.rng.uniform(8, 16),
            intensity=0.08,
            growth=0.0,
            vx=u * km_per_frame, vy=v * km_per_frame,
            max_age=int(self.rng.integers(24, 40)),
            peak_dbz=float(self.rng.uniform(52, 68) if strong else self.rng.uniform(44, 56)),
            rotates=float(self.rng.choice([-1.0, 1.0], p=[0.3, 0.7])),
        ))

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance the case by one 5-minute frame."""
        self.frame_index += 1
        for c in self.cells:
            c.step(self.rng)
        # new convective initiation every ~8 frames (the CI schedule)
        if self.frame_index % 8 == 4:
            self._spawn_cell(strong=self.frame_index % 24 == 4)
        # remove dead cells
        self.cells = [c for c in self.cells
                      if c.age < c.max_age and 0 <= c.x < GRID["nx"] + 40
                      and -40 < c.y < GRID["ny"]]

    # ------------------------------------------------------------------
    def fields(self) -> dict[str, np.ndarray]:
        """Render current-frame observation fields on the shared grid."""
        lat2d, lon2d = self._mesh
        ny, nx = GRID["ny"], GRID["nx"]
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

        refl = np.full((ny, nx), 8.0)
        vil = np.zeros((ny, nx))
        rain = np.zeros((ny, nx))
        shear = np.zeros((ny, nx))
        bt = 278.0 - 0.006 * self.terrain  # clear-sky BT warm w/ terrain

        for c in self.cells:
            r = max(c.radius, 2.0)
            d2 = (xx - c.x) ** 2 + (yy - c.y) ** 2
            core = np.exp(-d2 / (2 * (0.55 * r) ** 2))
            mid = np.exp(-d2 / (2 * (1.0 * r) ** 2))
            inten = c.intensity
            refl = np.maximum(refl, c.peak_dbz * inten * core + 24.0 * mid)
            vil += 55.0 * inten * core ** 1.4
            rain += 120.0 * (inten ** 1.6) * core
            # azimuthal shear dipole (rotation proxy)
            ang = np.arctan2(yy - c.y, xx - c.x)
            shear += c.rotates * 0.012 * inten * mid * np.sin(2 * ang)
            # anvil & cold U-shaped top in IR
            bt -= 62.0 * inten * (core ** 0.8) - 14.0 * mid

        noise = self.rng.normal(0, 1.6, size=(ny, nx))
        refl = np.clip(refl + noise, 0, 75)
        vil = np.clip(vil + self.rng.normal(0, 1.2, size=(ny, nx)), 0, 70)
        rain = np.clip(rain + self.rng.normal(0, 1.0, size=(ny, nx)), 0, 180)
        bt = np.clip(bt + self.rng.normal(0, 0.8, size=(ny, nx)), 185, 300)

        # lightning strokes: Poisson near intense cores
        strokes: list[tuple[float, float]] = []
        for c in self.cells:
            lam = 14.0 * (c.intensity ** 2.5)
            n = self.rng.poisson(lam)
            for _ in range(n):
                sx = c.x + self.rng.normal(0, c.radius * 0.8)
                sy = c.y + self.rng.normal(0, c.radius * 0.8)
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
