"""VAJRA engine: the full per-cycle pipeline.

ingest -> regrid/store -> nowcast ensemble -> CI -> hazards -> tracking
-> ETA -> CAP alerts -> state snapshot (for API/dashboard/verification).
Every cycle times itself against the 90-second latency budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import (CI_LEADS_MIN, FRAME_MINUTES, GRID, LATENCY_BUDGET_S,
                     NOWCAST_LEADS_MIN, POIS, STEERING_WIND_MS, THRESH)
from .alerting.cap import cap_alert
from .hazards.heads import run_all_hazards
from .hazards.ml_heads import MLHazardHeads
from .ingest.store import SlabStore
from .ingest.synthetic import SyntheticCase
from .ingest.domains import set_domain
from .nowcast.advection import (dense_optical_flow, ensemble_forecast,
                                mean_flow_from_prior, upsample_to)
from .nowcast.ci import ci_probability
from .nowcast.blend import blend_prob
from .tracking.tracker import StormTracker
from .verify.metrics import contingency, fss

# ---------------------------------------------------------------------------
# Real replay case: Dec 10-11 2021 (Mayfield KY EF4 outbreak; hail, mesos
# and heavy rain all present in the staged MRMS window).
# Supercells tracked NE at ~50 kt from the SW -> steering prior (24, 17) m/s.
# ---------------------------------------------------------------------------
MRMS2021_CASE = {
    "case_dir": "data/mrms/case_20211210",
    "goes_dir": "data/goes/dec2021",
    "start": "2021-12-10T18:00:00+00:00",
    "end": "2021-12-11T02:00:00+00:00",
    "center": (36.75, -88.65),
    "pois": {
        "Mayfield": (36.74, -88.64),
        "Paducah": (37.09, -88.60),
        "Murray": (36.61, -88.31),
        "Benton": (36.85, -88.35),
        "Princeton": (37.11, -87.97),
    },
    "steering": (24.0, 17.0),
}


@dataclass
class EngineState:
    frame: int = 0
    ts: str = ""
    latency_s: float = 0.0
    obs_png_ready: bool = False
    cells: list[dict] = field(default_factory=list)
    hazards_summary: dict = field(default_factory=dict)
    etas: dict = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    verify: dict = field(default_factory=dict)


class VajraEngine:
    def __init__(self, source: str = "synthetic") -> None:
        self.store = SlabStore()
        self.tracker = StormTracker()
        self.source = source
        self.case = SyntheticCase() if source == "synthetic" else None
        self.mrms: object | None = None
        self.goes: object | None = None
        self.ts0 = datetime(2026, 5, 10, 7, 30, tzinfo=timezone.utc)  # 13:00 IST
        if source == "mrms2021":
            from datetime import datetime as _dt
            from .ingest.mrms import MRMSCaseAdapter
            from .ingest.goes import GOESCaseAdapter
            c = MRMS2021_CASE
            start = _dt.fromisoformat(c["start"])
            end = _dt.fromisoformat(c["end"])
            set_domain(c["center"][0], c["center"][1], c["pois"], c["steering"])
            self.mrms = MRMSCaseAdapter(c["case_dir"], start, end)
            self.goes = GOESCaseAdapter(c["goes_dir"])
            self.ts0 = start
            self.n_frames = self.mrms.n_frames
        else:
            self.n_frames = 10_000
        self._u = None
        self._v = None
        self._flow_confidence = 0.0
        self.latest_leads: dict[int, np.ndarray] = {}
        self.latest_spread30: np.ndarray | None = None
        self.latest_ci: dict[int, np.ndarray] = {}
        self.latest_hazards: dict = {}
        self.rng = np.random.default_rng(7)
        # SEVIR-trained ML heads (graceful fallback to physics heads)
        try:
            self.ml_heads = MLHazardHeads()
        except Exception:
            self.ml_heads = None
        # pending predictions for rolling verification
        self._pending: list[dict] = []
        self.verify_history: list[dict] = []
        self.last_state = EngineState()

    # ------------------------------------------------------------------
    def ingest_frame(self, frame: int = 1) -> dict:
        """One ingestion cycle from the active source adapter."""
        if self.source == "synthetic":
            f = self.case.fields()
            self.case.step()
            return f
        if self.source == "mrms2021":
            from datetime import timedelta
            f = self.mrms.frame(frame - 1)
            ts = self.ts0 + timedelta(minutes=FRAME_MINUTES * frame)
            if "bt_cached" in f:
                # cache fast-path: pre-decoded GOES BT (NaN => no coverage)
                bt_c = np.asarray(f.pop("bt_cached"))
                if np.isfinite(bt_c).any():
                    f["bt"] = np.where(np.isfinite(bt_c), bt_c, 273.0
                                       ).astype(np.float32)
                f["strokes"] = self.mrms.cached_strokes(frame - 1) or []
            elif self.goes is not None:
                bt = self.goes.bt(ts)
                if bt is not None:
                    f["bt"] = bt
                f["strokes"] = self.goes.strokes(ts)
            f.setdefault("bt", None)
            return f
        raise NotImplementedError(
            f"source '{self.source}' adapter not enabled yet")

    # ------------------------------------------------------------------
    def run_cycle(self) -> EngineState:
        t_start = time.perf_counter()
        frame = self.store.latest_frame() or 0
        frame += 1
        fields = self.ingest_frame(frame)
        self.store.add(frame, fields)

        refl = fields["refl"]
        vil = fields["vil"]

        # ---- motion estimation (Tier-2 core, half-res for speed) ------
        from .nowcast.advection import block_mean
        DS = 2
        hist = self.store.history("refl", 2)
        u = v = None
        if len(hist) >= 2 and hist[-1].max() > 30:
            flow = dense_optical_flow(block_mean(hist[-2], DS),
                                      block_mean(hist[-1], DS))
            if flow is not None:
                u, v = flow
                self._flow_confidence = float(
                    min(1.0, (hist[-1].max() - 30.0) / 30.0))
        if u is None:
            u, v = mean_flow_from_prior(GRID["nx"] // DS, GRID["ny"] // DS,
                                        STEERING_WIND_MS["u"],
                                        STEERING_WIND_MS["v"])
            self._flow_confidence = 0.0
        self._u, self._v = u, v

        # ---- nowcast ensemble (Tier-2, 0-2 h, computed half-res) -----
        leads = [m for m in NOWCAST_LEADS_MIN if m <= 120]
        refl_s = block_mean(refl, DS)
        prev_s = block_mean(hist[-2], DS) if len(hist) >= 2 else None
        ens = ensemble_forecast(refl_s, prev_s, u, v, leads,
                                n_members=6, rng=self.rng)
        # upsample means back to the 1 km analysis grid
        ens = [[upsample_to(m, GRID["ny"], GRID["nx"]) for m in lead]
               for lead in ens]
        self.latest_leads = {L: np.mean(ens[i], axis=0)
                             for i, L in enumerate(leads)}
        self.latest_spread30 = (np.std(ens[leads.index(30)], axis=0)
                                if 30 in leads else None)

        # ---- CI probabilities (Tier-1) -------------------------------
        ci = {L: ci_probability(self.store, L) for L in CI_LEADS_MIN}
        self.latest_ci = ci

        # ---- hazards (4 heads) ---------------------------------------
        hazards = run_all_hazards(self.store)
        # ML heads override: lightning from the SEVIR model when available;
        # hail takes the max of the MESH-based map and the learned
        # severe-growth probability (both calibrated)
        if getattr(self, "ml_heads", None) is not None and self.ml_heads.available:
            p_light_ml = self.ml_heads.lightning_prob(self.store)
            if p_light_ml is not None:
                hazards["lightning"]["prob"] = np.maximum(
                    hazards["lightning"]["prob"], p_light_ml)
                hazards["lightning"]["ml_active"] = True
            p_sev_ml = self.ml_heads.severe_prob(self.store)
            if p_sev_ml is not None:
                hazards["hail"]["prob"] = np.maximum(
                    hazards["hail"]["prob"], 0.7 * p_sev_ml)
                hazards["hail"]["ml_active"] = True
        self.latest_hazards = hazards

        # ---- tracking + ETA (real MESH when available) ---------------
        mesh = fields.get("mesh")
        if mesh is None or not mesh.any():
            mesh = hazards["hail"]["mesh_mm"]
        tracks = self.tracker.update(refl, vil, mesh)
        etas: dict[str, dict] = {}
        for name, (plat, plon) in POIS.items():
            best = None
            for t in tracks:
                e = self.tracker.eta_to_poi(t, plat, plon)
                if e and (best is None or e["eta_min"] < best["eta_min"]):
                    best = {"cell_id": t.tid, **e,
                            "max_mesh_mm": round(t.max_mesh, 1),
                            "max_refl_dbz": round(t.max_refl, 1)}
            etas[name] = best or {}

        # ---- verification (rolling, model vs persistence) ------------
        self._update_verification(frame, refl, ens, leads)

        # ---- CAP alerts ----------------------------------------------
        alerts = self._emit_alerts(tracks, hazards, etas)

        latency = time.perf_counter() - t_start
        state = EngineState(
            frame=frame,
            ts=(self.ts0 + timedelta(minutes=frame * FRAME_MINUTES)).isoformat(),
            latency_s=round(latency, 4),
            cells=self._cells_payload(tracks, etas),
            hazards_summary=self._hazards_summary(hazards),
            etas=etas,
            alerts=alerts,
            verify=self.verify_history[-1] if self.verify_history else {},
        )
        self.last_state = state
        return state

    # ------------------------------------------------------------------
    def _cells_payload(self, tracks, etas) -> list[dict]:
        from .grid import grid_xy_to_latlon
        out = []
        for t in tracks:
            lat, lon = grid_xy_to_latlon(t.cx, t.cy)
            etas_for_cell = [{"poi": name, **e} for name, e in etas.items()
                             if e and e.get("cell_id") == t.tid]
            out.append({
                "id": t.tid, "lat": round(lat, 4), "lon": round(lon, 4),
                "age_frames": t.age, "speed_kmh": t.speed_kmh(),
                "max_refl_dbz": round(t.max_refl, 1),
                "max_vil": round(t.max_vil, 1),
                "mesh_mm": round(t.max_mesh, 1),
                "severity": ("severe" if t.max_mesh >= THRESH["mesh_mm"]
                             or t.max_refl >= THRESH["refl_core_dbz"] else "moderate"),
                "etas": etas_for_cell,
            })
        return out

    def _hazards_summary(self, hazards) -> dict:
        out = {}
        for h, d in hazards.items():
            p = d["prob"]
            out[h] = {
                "max_prob": round(float(p.max()), 3),
                "mean_prob": round(float(p.mean()), 3),
                "area_km2_above_0.5": int((p >= 0.5).sum()),
            }
        if "lightning" in hazards:
            out["lightning"]["strokes_this_frame"] = hazards["lightning"]["n_strokes"]
            out["lightning"]["jump_cells"] = int(hazards["lightning"]["jump"].sum())
        return out

    def _emit_alerts(self, tracks, hazards, etas) -> list[dict]:
        alerts = []
        hail_p = hazards["hail"]["prob"]
        light_p = hazards["lightning"]["prob"]
        for t in tracks:
            if t.max_mesh >= THRESH["mesh_mm"] and t.age >= 2:
                ring = next((c["ring"] for c in self.tracker.cell_polygons()
                             if c["tid"] == t.tid), [])
                eta0 = next((e["eta_min"] for name, e in etas.items()
                             if e and e.get("cell_id") == t.tid), None)
                xml = cap_alert(t.tid, "Significant Hail", "Severe", "Likely",
                                f"Hail core tracked: MESH {t.max_mesh:.0f} mm",
                                "Take shelter; protect livestock and vehicles.",
                                ring, eta0)
                alerts.append({"cell": t.tid, "hazard": "hail",
                               "cap_xml": xml})
            if float(light_p.max()) >= 0.75 and t.age >= 2 and \
                    float(light_p[int(t.cy), int(t.cx)]) >= 0.6:
                ring = next((c["ring"] for c in self.tracker.cell_polygons()
                             if c["tid"] == t.tid), [])
                eta0 = next((e["eta_min"] for name, e in etas.items()
                             if e and e.get("cell_id") == t.tid), None)
                xml = cap_alert(t.tid, "Lightning", "Severe", "Likely",
                                "Electrically active storm nearby",
                                "Move indoors; avoid open fields and trees.",
                                ring, eta0)
                alerts.append({"cell": t.tid, "hazard": "lightning",
                               "cap_xml": xml})
        return alerts[:12]

    # ------------------------------------------------------------------
    def _update_verification(self, frame: int, refl_now: np.ndarray,
                             ens, leads) -> None:
        # score predictions issued `lead` minutes ago against current obs
        for pred in list(self._pending):
            if frame >= pred["due_frame"]:
                thr = 40.0
                scores_model = contingency(pred["fcst"], refl_now, thr)
                scores_persist = contingency(pred["persistence"], refl_now, thr)
                self.verify_history.append({
                    "frame": frame, "lead_min": pred["lead_min"],
                    "model": scores_model, "persistence": scores_persist,
                    "fss8_model": round(fss(pred["fcst"], refl_now, thr, 8), 3),
                    "fss8_persistence": round(fss(pred["persistence"], refl_now, thr, 8), 3),
                    "fss16_model": round(fss(pred["fcst"], refl_now, thr, 16), 3),
                    "fss16_persistence": round(fss(pred["persistence"], refl_now, thr, 16), 3),
                    "fss32_model": round(fss(pred["fcst"], refl_now, thr, 32), 3),
                    "fss32_persistence": round(fss(pred["persistence"], refl_now, thr, 32), 3),
                })
                self._pending.remove(pred)
        # register new predictions at all nowcast leads
        for i, lead in enumerate(leads):
            fcst = np.mean(ens[i], axis=0)
            self._pending.append({
                "due_frame": frame + round(lead / FRAME_MINUTES),
                "lead_min": lead, "fcst": fcst,
                "persistence": refl_now,
            })
        # keep bounded
        if len(self.verify_history) > 2000:
            self.verify_history = self.verify_history[-2000:]
        if len(self._pending) > 200:
            self._pending = self._pending[-200:]

