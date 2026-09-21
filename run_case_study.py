"""End-to-end case study: Dec 10-11 2021 Mayfield EF4 outbreak.

Runs all 96 frames through the live pipeline and produces:
  - alert timeline (CAP alerts with timestamps, cells, ETAs)
  - per-role decisions (district / aviation / agriculture thresholds)
  - hazard maxima over time + timeline plot
  - lead-time gain: first-alert to POI-arrival minutes per event
  - verification summary (engine rolling scores)
  - docs/case_study_dec2021.md + timeline plot

Usage: python run_case_study.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

from vajra.engine import VajraEngine
from vajra.config import POIS

OUT = "docs"

# Role-specific decision thresholds (documented policy, not tuned):
# district acts on high-confidence hail/cloudburst; aviation on wind/hail at
# lower threshold (cost of miss >> cost of false alarm); farms on lightning.
ROLE_POLICY = {
    "district": [("hail", 0.50), ("cloudburst", 0.50)],
    "aviation": [("downburst", 0.60), ("hail", 0.40)],
    "agriculture": [("lightning", 0.40)],
}


def main() -> None:
    eng = VajraEngine(source="mrms2021")
    n = eng.n_frames
    print(f"running full past case: {n} frames (8 h, Dec-2021 EF4 outbreak)")

    frames = []          # per-frame records
    hazard_series = {k: [] for k in ("lightning", "hail", "downburst",
                                     "cloudburst")}
    mesh_series, refl_series, stroke_series = [], [], []
    role_first: dict[str, str | None] = {r: None for r in ROLE_POLICY}
    role_counts = {r: 0 for r in ROLE_POLICY}

    for i in range(n):
        st = eng.run_cycle()
        hs = st.hazards_summary
        for k in hazard_series:
            hazard_series[k].append(hs.get(k, {}).get("max_prob", 0.0))
        mesh_series.append(float(eng.store.latest("mesh").max()))
        refl_series.append(float(eng.store.latest("refl").max()))
        stroke_series.append(len(eng.store.strokes[-1])
                             if eng.store.strokes else 0)
        # role decisions this frame (gated on tracked cells: no storm object
        # = no alert, otherwise background shear/noise would fire all night)
        if st.cells:
            fired = {r: [h for h, t in pol
                         if hs.get(h, {}).get("max_prob", 0) >= t]
                     for r, pol in ROLE_POLICY.items()}
        else:
            fired = {r: [] for r in ROLE_POLICY}
        for r, hazards in fired.items():
            if hazards:
                role_counts[r] += 1
                if role_first[r] is None:
                    role_first[r] = st.ts[11:16] + "Z"
        frames.append({
            "frame": st.frame, "ts": st.ts,
            "cells": [(c["id"], c["max_refl_dbz"], c["mesh_mm"])
                      for c in st.cells],
            "etas": {p: (e["cell_id"], e["eta_min"], e["sigma_min"])
                     for p, e in st.etas.items() if e},
            "alerts": [(a["cell"], a["hazard"]) for a in st.alerts],
            "roles": fired,
            "latency": st.latency_s,
        })
        if (i + 1) % 12 == 0:
            print(f"  frame {st.frame}/{n} {st.ts[11:16]}Z "
                  f"refl={refl_series[-1]:.0f} mesh={mesh_series[-1]:.0f} "
                  f"cells={len(st.cells)} alerts={len(st.alerts)}")

    # ---- lead-time gain: first CAP alert -> POI arrival ------------------
    # lead time ~= ETA stated at first alert for each (cell, POI) pair
    first_alert_frame: dict[tuple[int, str], int] = {}
    for f in frames:
        for cell, haz in f["alerts"]:
            for poi in POIS:
                key = (cell, poi)
                if key not in first_alert_frame:
                    first_alert_frame[key] = f["frame"]
    lead_times: list[dict] = []
    for f in frames:
        for cell, haz in f["alerts"]:
            for poi, (cid, eta, sig) in f["etas"].items():
                if cid == cell:
                    lead_times.append({"ts": f["ts"][11:16] + "Z",
                                       "cell": cell, "hazard": haz,
                                       "poi": poi, "eta_min": eta,
                                       "sigma": sig})

    # ---- verification summary -------------------------------------------
    by_lead: dict[int, list] = {}
    for v in eng.verify_history:
        by_lead.setdefault(v["lead_min"], []).append(v)

    # ---- plots ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(n) * 5 / 60 + 18  # hours UTC from 18Z
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7), sharex=True)
    for ax, series, title in zip(
            axes,
            (hazard_series, {"MESH mm": mesh_series,
                             "refl dBZ/2": [r / 2 for r in refl_series]},
             {"strokes": stroke_series}),
            ("max hazard probability", "storm intensity (observed)",
             "GLM strokes / frame")):
        for k, v in series.items():
            ax.plot(t, v, label=k)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("UTC hour, Dec 10-11 2021")
    fig.suptitle("VAJRA live replay: Dec-2021 EF4 case (real MRMS+GOES+GLM)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/case_timeline.png", dpi=140)

    # ---- markdown report ---------------------------------------------------
    L = ["# Case study: Dec 10-11 2021 (Mayfield KY EF4 outbreak)", ""]
    L.append(f"96 frames at 5-min cadence through the live pipeline "
             f"(cache fast-path, ML heads, Tier-3). Mean cycle latency: "
             f"{np.mean([f['latency'] for f in frames]):.2f} s.")
    L.append("")
    L.append("## Alert timeline (CAP alerts actually issued)")
    any_alert = False
    for f in frames:
        if f["alerts"]:
            any_alert = True
            eta_txt = "; ".join(
                f"{p}: cell #{c} in {e:.0f}+/-{s:.0f}m"
                for p, (c, e, s) in f["etas"].items())
            L.append(f"- **{f['ts'][11:16]}Z** " +
                     ", ".join(f"cell #{c} {h}" for c, h in f["alerts"]) +
                     (f" | {eta_txt}" if eta_txt else ""))
    if not any_alert:
        L.append("No CAP-threshold cells (MESH>=19) in window.")
    L.append("")
    L.append("## Per-role decisions (documented thresholds)")
    for r, pol in ROLE_POLICY.items():
        L.append(f"- **{r}** ({', '.join(f'{h}>={t}' for h, t in pol)}): "
                 f"first fired {role_first[r] or 'never'}, "
                 f"active in {role_counts[r]} frames")
    L.append("")
    L.append("## Warning lead times (first CAP alert -> stated ETA)")
    seen = set()
    for lt in lead_times:
        key = (lt["cell"], lt["poi"])
        if key in seen:
            continue
        seen.add(key)
        L.append(f"- {lt['ts']}: cell #{lt['cell']} {lt['hazard']} -> "
                 f"{lt['poi']}: **{lt['eta_min']:.0f} +/- {lt['sigma']:.0f} min**")
    if not seen:
        L.append("No cell-POI pairs with stated ETAs in window.")
    L.append("")
    L.append("## Verification on this case (rolling, 40 dBZ)")
    L.append("| lead | CSI | POD | FAR | HSS | CSI persist |")
    L.append("|---|---|---|---|---|---|")
    for Ld in sorted(by_lead):
        vs = by_lead[Ld]
        m = {k: float(np.mean([x["model"][k] for x in vs]))
             for k in ("csi", "pod", "far", "hss")}
        b = float(np.mean([x["persistence"]["csi"] for x in vs]))
        L.append(f"| {Ld} min | {m['csi']:.3f} | {m['pod']:.3f} | "
                 f"{m['far']:.3f} | {m['hss']:.3f} | {b:.3f} |")
    L.append("")
    L.append("Plot: case_timeline.png (hazard probabilities, observed "
             "intensity, lightning through the event).")
    with open(f"{OUT}/case_study_dec2021.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nwrote {OUT}/case_study_dec2021.md + {OUT}/case_timeline.png")
    print("role first-fire:", role_first)


if __name__ == "__main__":
    main()
