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

from ..config import FRAME_MINUTES, GRID, LATENCY_BUDGET_S, POIS
from ..engine import VajraEngine
from ..grid import latlon_to_grid_xy

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
        from ..alerting.dispatcher import AlertDispatcher
        from ..config import POIS
        self.engine = VajraEngine(source=source)
        self.dispatcher = AlertDispatcher(pois=list(POIS))
        self.interval_s = interval_s
        self.paused = False
        self.auto_dispatch = False   # when True, approved drafts send at once
        self.latest_leads: dict[int, np.ndarray] = {}
        self.latest_ci: dict[int, np.ndarray] = {}
        self.latest_ens_spread: np.ndarray | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.history_records: list[dict] = []
        self.step_once()

    def step_once(self) -> None:
        eng = self.engine
        state = eng.run_cycle()
        with self._lock:
            self.latest_leads = dict(eng.latest_leads)
            self.latest_ens_spread = eng.latest_spread30
            self.latest_ci = dict(eng.latest_ci)
            self.dispatcher.ingest(state)
            if self.auto_dispatch:
                self.dispatcher.dispatch_approved()
            max_refl = max([c["max_refl_dbz"] for c in state.cells], default=38.0)
            max_mesh = max([c["mesh_mm"] for c in state.cells], default=8.0)
            max_wind = max([c["speed_kmh"] * 1.35 for c in state.cells], default=42.0)
            strokes_count = len(eng.store.strokes[-1]) if eng.store.strokes else 0
            ts_label = state.ts[11:16] if state.ts and len(state.ts) >= 16 else f"F{state.frame}"
            self.history_records.append({
                "frame": state.frame,
                "ts": ts_label,
                "max_refl": round(float(max_refl), 1),
                "max_mesh": round(float(max_mesh), 1),
                "max_wind": round(float(max_wind), 1),
                "strokes": int(strokes_count),
            })
            if len(self.history_records) > 20:
                self.history_records = self.history_records[-20:]

    def switch_scenario(self, scenario_id: str) -> dict:
        from ..alerting.dispatcher import AlertDispatcher
        from ..config import POIS
        with self._lock:
            meta = self.engine.switch_scenario(scenario_id)
            if meta:
                self.dispatcher = AlertDispatcher(pois=list(POIS))
                self.latest_leads.clear()
                self.latest_ci.clear()
                self.latest_ens_spread = None
        if meta:
            self.step_once()
        return meta

    def get_strokes_latlon(self) -> list[list[float]]:
        out = []
        if self.engine.store.strokes and self.engine.store.strokes[-1]:
            from ..grid import grid_xy_to_latlon
            for sx, sy in self.engine.store.strokes[-1][:80]:
                lat, lon = grid_xy_to_latlon(sx, sy)
                out.append([round(float(lat), 4), round(float(lon), 4)])
        return out

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
    from ..envfile import load_dotenv
    load_dotenv()
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
            "model_version": st.model_version,
            "qc": st.qc,
            "scenario_id": getattr(st, "scenario_id", "uttarakhand_cloudburst"),
            "scenario_name": getattr(st, "scenario_name", "Uttarakhand Himalayan Cloudburst"),
            "pois": {k: [v[0], v[1]] for k, v in POIS.items()},
            "grid": {"nx": GRID["nx"], "ny": GRID["ny"], "dx_km": GRID["dx"],
                     "center": [GRID["center_lat"], GRID["center_lon"]],
                     "bounds": [[GRID["lat_bottom"], GRID["lon_left"]],
                                [GRID["lat_top"], GRID["lon_right"]]]},
            "cells": st.cells,
            "hazards_summary": st.hazards_summary,
            "etas": st.etas,
            "alerts": [{"cell": a["cell"], "hazard": a["hazard"]} for a in st.alerts],
            "verify": st.verify,
            "strokes": live.get_strokes_latlon(),
            "history": list(live.history_records),
        }

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        from ..ingest.domains import SCENARIOS
        return list(SCENARIOS.values())

    @app.post("/api/scenario/{scenario_id}")
    def set_scenario(scenario_id: str) -> dict:
        meta = live.switch_scenario(scenario_id)
        if not meta:
            raise HTTPException(400, f"unknown scenario: {scenario_id}")
        return {"ok": True, "scenario": meta}

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
        """kind in {obs, fcst, ci}; value = lead minutes (fcst/ci) or ignored.
        fcst/180 is the Tier-3 blended severe probability (0-1), rendered on
        the same scale for display."""
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
                if value == 180:
                    f = f * 70.0  # probability -> display scale
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
            if kind == "wind":
                f = live.engine.store.latest("shear")
                if f is None:
                    f = np.zeros((512, 512), dtype=np.float32)
                return Response(grid_to_png(np.abs(f) * 2200.0), media_type="image/png")
            if kind == "bt":
                f = live.engine.store.latest("bt")
                if f is None:
                    f = np.full((512, 512), 273.0, dtype=np.float32)
                return Response(grid_to_png(np.clip((270.0 - f) * 1.1, 0, 75.0)), media_type="image/png")
            if kind == "mesh":
                f = live.engine.store.latest("vil")
                if f is None:
                    f = np.zeros((512, 512), dtype=np.float32)
                return Response(grid_to_png(f), media_type="image/png")
        raise HTTPException(400, "unknown kind")

    @app.post("/api/pause/{flag}")
    def pause(flag: str) -> dict:
        live.paused = flag == "on"
        return {"paused": live.paused}

    @app.post("/api/step")
    def step() -> dict:
        live.step_once()
        return {"frame": live.engine.last_state.frame}

    # ---------------- alert system ----------------
    @app.get("/api/alerts/outbox")
    def outbox() -> dict:
        with live._lock:
            return {"alerts": live.dispatcher.outbox()}

    @app.post("/api/alerts/approve/{alert_id}")
    def approve(alert_id: str) -> dict:
        with live._lock:
            return {"ok": live.dispatcher.approve(alert_id)}

    @app.post("/api/alerts/approve-all")
    def approve_all() -> dict:
        with live._lock:
            return {"approved": live.dispatcher.approve_all()}

    @app.post("/api/alerts/dispatch/{alert_id}")
    def dispatch(alert_id: str) -> dict:
        with live._lock:
            return {"receipts": live.dispatcher.dispatch(alert_id)}

    @app.post("/api/alerts/dispatch-approved")
    def dispatch_approved() -> dict:
        with live._lock:
            return {"dispatched": live.dispatcher.dispatch_approved()}

    @app.post("/api/alerts/auto/{flag}")
    def auto(flag: str) -> dict:
        live.auto_dispatch = flag == "on"
        return {"auto_dispatch": live.auto_dispatch}

    @app.get("/api/alerts/channels")
    def channels() -> dict:
        from ..alerting.channels import CHANNELS
        return {k: {"available": ok, "reason": reason}
                for k, ch in CHANNELS.items()
                for ok, reason in [ch.available()]}

    @app.get("/api/subscribers")
    def subscribers() -> dict:
        with live._lock:
            return {"subscribers": [
                {"id": s.id, "role": s.role, "poi": s.poi,
                 "languages": list(s.languages), "channels": s.channels}
                for s in live.dispatcher.subscribers]}

    @app.post("/api/subscribers/{sub_id}")
    def subscribe(sub_id: str, payload: dict) -> dict:
        """Register/update a subscriber address book, e.g.
        {"telegram": "<chat_id>", "sms": "+91...", "languages": ["hi"]}."""
        with live._lock:
            for s in live.dispatcher.subscribers:
                if s.id == sub_id:
                    for k, v in payload.items():
                        if k == "languages":
                            s.languages = tuple(v)
                        else:
                            s.channels[k] = v
                    return {"ok": True, "id": sub_id}
        return {"ok": False, "error": "unknown subscriber"}

    @app.get("/api/capfeed")
    def capfeed() -> Response:
        """SACHET-style ATOM feed of approved/sent CAP alerts."""
        from xml.sax.saxutils import escape
        with live._lock:
            items = []
            for r in live.dispatcher.alerts.values():
                if r.status not in ("approved", "sent") or not r.cap_xml:
                    continue
                items.append(
                    f"<entry><id>{escape(r.id)}</id>"
                    f"<title>{escape(r.hazard)} {escape(r.poi)}</title>"
                    f"<updated>{escape(r.ts)}</updated>"
                    f"<content type='application/cap+xml'>"
                    f"{escape(r.cap_xml)}</content></entry>")
        feed = (f'<?xml version="1.0" encoding="UTF-8"?>'
                f'<feed xmlns="http://www.w3.org/2005/Atom">'
                f"<title>VAJRA CAP feed (demo)</title>"
                f"{''.join(items)}</feed>")
        return Response(feed, media_type="application/atom+xml")

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
