"""Full verification report on the real MRMS Dec-2021 case.

Produces (docs/verification/):
  csi_vs_lead.png, fss_vs_lead.png, cost_loss.png, eta_scatter.png,
  reliability_hail.png, reliability_lightning.png, verification_report.md

Usage: python run_verification.py
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from vajra.engine import VajraEngine
from vajra.config import POIS

OUT = "docs/verification"
DOWNSAMPLE = 4  # 512 -> 128 grids for the analysis
DS = slice(None, None, DOWNSAMPLE)
PAIR_FRAMES = 6  # 30 min = 6 frames


def _storm_mask(refl: np.ndarray) -> np.ndarray:
    """Dilated convective mask: reliability is scored only where convection
    is plausible (avoids the 'no-storm model gets 99% accuracy' trap)."""
    from scipy import ndimage
    return ndimage.binary_dilation(refl >= 25.0, iterations=12)[DS, DS]


def reliability_curve(probs: np.ndarray, events: np.ndarray, n_bins: int = 8):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1])
        if m.sum() < 10:
            continue
        xs.append(float(probs[m].mean()))
        ys.append(float(events[m].mean()))
        ns.append(int(m.sum()))
    return xs, ys, ns


def brier(probs: np.ndarray, events: np.ndarray) -> float:
    return float(np.mean((probs - events) ** 2)) if probs.size else float("nan")


def cost_loss_value(c: dict, ratios: np.ndarray) -> list[float]:
    """Richardson economic value: V(r) = [a - r(a+b)] / [(a+c)(1-r)]."""
    a, b, cm = c["hits"], c["fa"], c["misses"]
    events = a + cm
    out = []
    for r in ratios:
        out.append(0.0 if r >= 1.0 or events == 0
                   else float((a - r * (a + b)) / (events * (1 - r))))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eng = VajraEngine(source="mrms2021")
    n = eng.n_frames
    print(f"replaying {n} frames of the real case for verification ...")

    # delayed pairing queues: prob grids at t vs observations at t+30 min
    hail_q: list[np.ndarray] = []
    light_q: list[np.ndarray] = []
    mask_q: list[np.ndarray] = []

    hail_p: list[float] = []
    hail_e: list[int] = []
    light_p: list[float] = []
    light_e: list[int] = []
    eta_records: list[dict] = []
    cells_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    states = []

    for i in range(n):
        st = eng.run_cycle()
        states.append(st)
        hz = eng.latest_hazards
        refl = eng.store.latest("refl")
        mask = _storm_mask(refl)

        hp = hz["hail"]["prob"][DS, DS].astype(np.float32)
        lp = hz["lightning"]["prob"][DS, DS].astype(np.float32)
        mesh = eng.store.latest("mesh")[DS, DS].astype(np.float32)
        strokes = eng.store.strokes[-1] if eng.store.strokes else []

        hail_q.append(hp); mask_q.append(mask)
        light_q.append(lp)
        if len(hail_q) > PAIR_FRAMES:
            # lagged pairing: probability issued 30 min ago vs observations NOW
            mp = mask_q[0]
            if mp.any():
                hail_p.extend(hail_q[0][mp].tolist())
                hail_e.extend((mesh >= 19.0)[mp].astype(int).tolist())
                # lightning event: any stroke within ~3 cells, now
                lm = np.zeros((128, 128), np.float32)
                for (sx, sy) in strokes:
                    xi, yi = int(sx // DOWNSAMPLE), int(sy // DOWNSAMPLE)
                    if 0 <= xi < 128 and 0 <= yi < 128:
                        lm[yi, xi] = 1.0
                from scipy import ndimage
                lme = ndimage.maximum_filter(lm, size=7) > 0
                light_p.extend(light_q[0][mp].tolist())
                light_e.extend(lme[mp].astype(int).tolist())
            hail_q.pop(0); mask_q.pop(0); light_q.pop(0)

        for c in st.cells:
            cells_track[c["id"]].append((st.frame, c["lat"], c["lon"]))
        for poi, e in st.etas.items():
            if e:
                eta_records.append({"frame": st.frame, "cell": e["cell_id"],
                                    "poi": poi, "eta_min": e["eta_min"],
                                    "sigma_min": e["sigma_min"]})
        if (i + 1) % 24 == 0:
            print(f"  frame {st.frame}/{n}  paired={len(hail_p)}")

    # ------------------------------- plots --------------------------------
    by_lead = defaultdict(list)
    for v in eng.verify_history:
        by_lead[v["lead_min"]].append(v)
    leads = sorted(by_lead)

    # 1) CSI/POD/FAR vs lead
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, style, label in (("model", "o-", "VAJRA (LK extrapolation)"),
                              ("persistence", "s--", "Persistence")):
        csi = [np.mean([x[key]["csi"] for x in by_lead[L]]) for L in leads]
        ax.plot(leads, csi, style, label=f"CSI {label}")
        pod = [np.mean([x[key]["pod"] for x in by_lead[L]]) for L in leads]
        ax.plot(leads, pod, style, alpha=0.45, label=f"POD {label}")
        far = [np.mean([x[key]["far"] for x in by_lead[L]]) for L in leads]
        ax.plot(leads, far, style, alpha=0.25, label=f"FAR {label}")
    ax.set_xlabel("lead time (min)")
    ax.set_ylabel("score @ 40 dBZ")
    ax.set_title("Real case (Dec 10-11 2021): skill vs lead time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/csi_vs_lead.png", dpi=140)

    # 2) FSS vs lead
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for w, style in (("fss8_model", "o-"), ("fss16_model", "s-"),
                     ("fss32_model", "^-")):
        vals = [np.mean([x[w] for x in by_lead[L]]) for L in leads]
        ax.plot(leads, vals, style, label=w.replace("_model", "").upper())
    vals = [np.mean([x["fss32_persistence"] for x in by_lead[L]]) for L in leads]
    ax.plot(leads, vals, "k--", label="FSS32 persistence")
    ax.set_xlabel("lead time (min)"); ax.set_ylabel("FSS @ 40 dBZ")
    ax.set_title("Neighbourhood skill (FSS) vs lead time")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fss_vs_lead.png", dpi=140)

    # 3) cost-loss at 30-min
    agg = {}
    for who in ("model", "persistence"):
        agg[who] = {k: float(np.mean([x[who][k] for x in by_lead[30]]))
                    for k in ("hits", "fa", "misses")}
    ratios = np.linspace(0.02, 0.98, 40)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ratios, cost_loss_value(agg["model"], ratios), label="VAJRA")
    ax.plot(ratios, cost_loss_value(agg["persistence"], ratios), "--",
            label="persistence")
    ax.set_xlabel("cost/loss ratio"); ax.set_ylabel("economic value")
    ax.set_title("Cost-loss value, +30 min, 40 dBZ (real case)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/cost_loss.png", dpi=140)

    # 4) reliability diagrams
    for name, probs, events, fname in (
            ("Hail head vs observed MESH >= 19 mm (+30 min)",
             np.array(hail_p), np.array(hail_e, float), "reliability_hail"),
            ("Lightning head vs observed strokes (+30 min)",
             np.array(light_p), np.array(light_e, float), "reliability_lightning")):
        if probs.size < 50:
            print(f"  {name}: insufficient samples ({probs.size}), skipped")
            continue
        xs, ys, ns = reliability_curve(probs, events)
        base = float(events.mean())
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(xs, ys, "o-", color="#c62828")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
        ax.set_title(f"{name}\nBrier={brier(probs, events):.4f}, "
                     f"base rate={base:.3f}, n={probs.size}")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{OUT}/{fname}.png", dpi=140)

    # 5) ETA scatter
    errs = []
    for rec in eta_records:
        track = cells_track.get(rec["cell"], [])
        if not track:
            continue
        plat, plon = POIS[rec["poi"]]
        arr_frame = next(
            (fr for fr, la, lo in track
             if (abs(la - plat) * 111.0) ** 2 + (abs(lo - plon) * 88.0) ** 2 < 144.0),
            None)
        if arr_frame is None or arr_frame <= rec["frame"]:
            continue
        actual_min = (arr_frame - rec["frame"]) * 5.0
        errs.append({"eta_pred": rec["eta_pred"] if "eta_pred" in rec else rec["eta_min"],
                     "actual": actual_min, "err": actual_min - rec["eta_min"],
                     "sigma": rec["sigma_min"]})
    within = None
    if errs:
        errs_arr = np.array([e["err"] for e in errs])
        within = float(np.mean(
            np.abs(errs_arr) <= np.array([e["sigma"] for e in errs])))
        fig, ax = plt.subplots(figsize=(6.0, 4.6))
        ax.scatter([e["eta_pred"] for e in errs], [e["actual"] for e in errs],
                   s=18, color="#1565c0")
        lim = max(120, max(e["actual"] for e in errs))
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("predicted ETA (min)")
        ax.set_ylabel("actual arrival (min)")
        ax.set_title("Countdown-clock verification (real case)")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{OUT}/eta_scatter.png", dpi=140)

    # ------------------------------- report -------------------------------
    lines = ["# Verification report - real case (MRMS Dec 10-11 2021)", ""]
    lines.append(f"Frames: {n} (8 h at 5-min cadence). Threshold 40 dBZ.")
    lines.append("")
    lines.append("## Skill vs lead time")
    lines.append("| lead | CSI | POD | FAR | HSS | CSI persist | FSS8 | FSS32 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for L in leads:
        vs = by_lead[L]
        m = {k: np.mean([x["model"][k] for x in vs])
             for k in ("csi", "pod", "far", "hss")}
        b = np.mean([x["persistence"]["csi"] for x in vs])
        f8 = np.mean([x["fss8_model"] for x in vs])
        f32 = np.mean([x["fss32_model"] for x in vs])
        lines.append(f"| {L} min | {m['csi']:.3f} | {m['pod']:.3f} | "
                     f"{m['far']:.3f} | {m['hss']:.3f} | {b:.3f} | "
                     f"{f8:.3f} | {f32:.3f} |")
    if hail_p:
        lines.append("")
        lines.append(f"## Reliability (hail, n={len(hail_p)}, "
                     f"base rate {np.mean(hail_e):.4f}): "
                     f"Brier={brier(np.array(hail_p), np.array(hail_e, float)):.4f}")
    if light_p:
        lines.append(f"## Reliability (lightning, n={len(light_p)}, "
                     f"base rate {np.mean(light_e):.4f}): "
                     f"Brier={brier(np.array(light_p), np.array(light_e, float)):.4f}")
    if errs:
        lines.append("")
        lines.append(f"## ETA countdown: n={len(errs)}, "
                     f"median err {np.median(errs_arr):+.1f} min, "
                     f"MAE {np.mean(np.abs(errs_arr)):.1f} min, "
                     f"{within * 100:.0f}% within stated sigma")
    lines.append("")
    lines.append("Plots: csi_vs_lead.png, fss_vs_lead.png, cost_loss.png, "
                 "reliability_hail.png, reliability_lightning.png"
                 + (", eta_scatter.png" if errs else ""))
    with open(f"{OUT}/verification_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nreport written to {OUT}/verification_report.md")
    print(f"reliability samples: hail={len(hail_p)} lightning={len(light_p)}"
          + (f", eta={len(errs)}" if errs else ", eta=0"))


if __name__ == "__main__":
    main()
