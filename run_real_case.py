"""Full replay of the real MRMS Dec-2021 case with verification summary.

Usage: python run_real_case.py [--cycles 96]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from vajra.engine import VajraEngine
from vajra.config import LATENCY_BUDGET_S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=96)
    args = ap.parse_args()

    eng = VajraEngine(source="mrms2021")
    n = min(args.cycles, eng.n_frames)
    latencies = []
    states = []
    for i in range(n):
        st = eng.run_cycle()
        states.append(st)
        latencies.append(st.latency_s)
        if (i + 1) % 12 == 0:
            mesh = eng.store.latest("mesh")
            strokes = len(eng.store.strokes[-1]) if eng.store.strokes else 0
            print(f"  frame {st.frame:3d} {st.ts[11:16]}Z "
                  f"refl_max={float(eng.store.latest('refl').max()):5.1f} dBZ "
                  f"mesh_max={float(mesh.max()):5.1f} mm "
                  f"strokes={strokes:5d} cells={len(st.cells):2d} "
                  f"lat={st.latency_s:5.1f}s")

    print(f"\n[latency] mean {np.mean(latencies):.1f} s / frame "
          f"(budget {LATENCY_BUDGET_S:.0f}s; GRIB decode dominates offline)")

    by_lead: dict[int, list[dict]] = defaultdict(list)
    for v in eng.verify_history:
        by_lead[v["lead_min"]].append(v)
    print("\n[verification on REAL data] 40 dBZ, model vs persistence")
    print(f"  {'lead':>5} {'CSI':>6} {'POD':>6} {'FAR':>6} {'HSS':>6} | "
          f"{'CSI(base)':>9} {'FSS8':>6} {'FSS32':>6}")
    for lead in sorted(by_lead):
        vs = by_lead[lead]
        m = {k: float(np.mean([x['model'][k] for x in vs]))
             for k in ('csi', 'pod', 'far', 'hss')}
        b = float(np.mean([x['persistence']['csi'] for x in vs]))
        f8 = float(np.mean([x['fss8_model'] for x in vs]))
        f32 = float(np.mean([x['fss32_model'] for x in vs]))
        print(f"  {lead:>4}m {m['csi']:>6.3f} {m['pod']:>6.3f} {m['far']:>6.3f} "
              f"{m['hss']:>6.3f} | {b:>9.3f} {f8:>6.3f} {f32:>6.3f}")

    # hazard maxima across the whole case
    print("\n[hazard maxima over case]")
    hmax: dict[str, float] = defaultdict(float)
    alerts_total = 0
    for i in range(n):
        st = states[i]
        for k, v in st.hazards_summary.items():
            hmax[k] = max(hmax[k], v["max_prob"])
        alerts_total += len(st.alerts)
    for k, v in sorted(hmax.items()):
        print(f"  {k:12s} max_prob={v:.2f}")
    print(f"  CAP alerts issued during replay: {alerts_total}")

    # ETA demo from the last frame
    print("\n[ETA countdown demo - final frame]")
    for name, e in eng.last_state.etas.items():
        if e:
            print(f"  {name}: cell #{e['cell_id']} in {e['eta_min']} +/- "
                  f"{e['sigma_min']} min ({e['speed_kmh']} km/h, "
                  f"MESH {e['max_mesh_mm']} mm)")

    if eng.last_state.alerts:
        print("\n[CAP sample]")
        print(eng.last_state.alerts[0]["cap_xml"][:400], "...")


if __name__ == "__main__":
    main()
