"""FastAPI server: state, rasters, ETA, CAP alerts, verification, replay.

Serves the dashboard and pushes every 5-minute cycle of the replay engine
as queryable JSON + PNG rasters (Leaflet ImageOverlay).
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

from ..config import FRAME_MINUTES, LATENCY_BUDGET_S, POIS
from ..engine import VajraEngine
from ..grid import latlon_to_grid_xy, make_mesh

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False

DASH = Path(__file__).resolve().parents[2] / "dashboard" / "index.html"

# reflectivity colormap (dBZ): light rain -> hail core
DBZ_COLORS = [
    (0, (230, 235, 240)), (10, (170, 210, 240)), (20, (70, 150, 220)),
    (30, (40, 200, 120)), (35, (30, 160, 60)), (40, (250, 240, 90)),
    (45, (245, 170, 40)), (50, (235, 70, 40)), (55, (190, 20, 40)),
    (60, (160, 20, 120)), (65, (90, 20, 160)), (75, (250, 250, 250)),
]


def _dbz_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = np.array([c[0] for c in DBZ_COLORS])
    for ch in range(3):
        ys = np.array([c[1][ch] for c in DBZ_COLORS], dtype=float)
        lut[:, ch] = np.clip(np.interp(np.arange(256), xs, ys), 0, 255)
    return lut


_LUT = _dbz_lut()


def grid_to_png(field: np.ndarray, vmin: float = 0.0, vmax: float = 70.0) -> bytes:
    """Render a gridded field to PNG bytes (transparent where ~no signal)."""
    if not HAS_PIL:
        raise HTTPException(503, "Pillow not installed; pip install pillow")
    idx = np.clip((field - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    rgba = np.zeros((*idx.shape, 4), dtype=np.uint8)
    rgba[..., :3] = _LUT[idx]
    alpha = np.clip((field - 8) / 10.0, 0, 1) * 255
    rgba[..., 3] = alpha.astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class LiveState:
    """Holds the engine + its latest forecast fields for the API."""

    def __init__(self, source: str, interval_s: float) -> None:
        self.engine = VajraEngine(source=source)
        self.interval_s = interval_s
        self.paused = False
        self.latest_leads: dict[int, np.ndarray] = {}
        self.latest_ci: dict[int, np.ndarray] = {}
        self.latest_ens_spread: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def step_once(self) -> None:
        eng = self.engine
        state = eng.run_cycle()
        with self._lock:
            self.latest_leads = dict(eng.latest_leads)
            self.latest_ens_spread = eng.latest_spread30
            self.latest_ci = dict(eng.latest_ci)

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.paused:
                try:
                    self.step_once()
                except Exception as e:  # keep server alive during replays
                    print(f"[engine] cycle error: {e}")
            self._stop.wait(self.interval_s)

    def start_background(self) -> None:
        threading.Thread(target=self.run_forever, daemon=True).start()


def create_app(source: str = "synthetic", interval_s: float = 2.0) -> FastAPI:
    app = FastAPI(title="VAJRA Nowcast API", version="0.1.0")
    live = LiveState(source, interval_s)
    app.state.live = live

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        live.step_once()
        live.start_background()
        yield

    app.router.lifespan_context = lifespan

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DASH)

    @app.get("/api/state")
    def state() -> dict:
        st = live.engine.last_state
        return {
            "frame": st.frame, "ts": st.ts,
            "latency_s": st.latency_s,
            "latency_budget_s": LATENCY_BUDGET_S,
            "paused": live.paused,
            "grid": {"nx": 512, "ny": 512, "dx_km": 1.0,
                     "bounds": [[18.85, 76.82], [23.45, 81.36]]},
            "cells": st.cells,
            "hazards_summary": st.hazards_summary,
            "etas": st.etas,
            "alerts": [{"cell": a["cell"], "hazard": a["hazard"]} for a in st.alerts],
            "verify": st.verify,
        }

    @app.get("/api/alerts/cap")
    def cap_list() -> Response:
        st = live.engine.last_state
        if not st.alerts:
            return JSONResponse([])
        return JSONResponse([a["cap_xml"] for a in st.alerts])

    @app.get("/api/eta")
    def eta(lat: float = Query(...), lon: float = Query(...)) -> dict:
        plat, plon = lat, lon
        best = None
        for t in live.engine.tracker.tracks.values():
            e = live.engine.tracker.eta_to_poi(t, plat, plon)
            if e and (best is None or e["eta_min"] < best["eta_min"]):
                best = {"cell_id": t.tid, **e}
        return {"poi": {"lat": lat, "lon": lon}, "arrival": best or None}

    @app.get("/api/raster/{kind}/{value}")
    def raster(kind: str, value: int) -> Response:
        """kind in {obs, fcst, ci}; value = lead minutes (fcst/ci) or ignored."""
        with live._lock:
            if kind == "obs":
                f = live.engine.store.latest("refl")
                if f is None:
                    raise HTTPException(503, "no obs yet")
                return Response(grid_to_png(f), media_type="image/png")
            if kind == "fcst":
                f = live.latest_leads.get(value)
                if f is None:
                    raise HTTPException(404, f"no {value}-min forecast")
                return Response(grid_to_png(f), media_type="image/png")
            if kind == "ci":
                f = live.latest_ci.get(value)
                if f is None:
                    raise HTTPException(404, f"no CI at {value}")
                return Response(grid_to_png(f * 70.0), media_type="image/png")
            if kind == "spread":
                f = live.latest_ens_spread
                if f is None:
                    raise HTTPException(404, "no spread yet")
                return Response(grid_to_png(f * 6.0), media_type="image/png")
        raise HTTPException(400, "unknown kind")

    @app.post("/api/pause/{flag}")
    def pause(flag: str) -> dict:
        live.paused = flag == "on"
        return {"paused": live.paused}

    @app.post("/api/step")
    def step() -> dict:
        live.step_once()
        return {"frame": live.engine.last_state.frame}

    @app.get("/api/verify")
    def verify() -> dict:
        h = live.engine.verify_history
        if not h:
            return {"n": 0}
        agg = {"model": {}, "persistence": {}}
        for key in ("csi", "pod", "far", "hss"):
            agg["model"][key] = round(float(np.mean([x["model"][key] for x in h])), 3)
            agg["persistence"][key] = round(float(np.mean([x["persistence"][key] for x in h])), 3)
        agg["fss8_model"] = round(float(np.mean([x["fss8_model"] for x in h])), 3)
        agg["fss8_persistence"] = round(float(np.mean([x["fss8_persistence"] for x in h])), 3)
        agg["n"] = len(h)
        return agg

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="VAJRA nowcast server")
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic"], help="ingestion source")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds per replayed 5-min frame")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.source, args.interval),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
