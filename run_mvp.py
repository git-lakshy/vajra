"""VAJRA MVP headless run: replay, verification table, ETA demo, CAP sample.

Usage:
    python run_mvp.py --cycles 60            # ~5 simulated hours
    python run_mvp.py --cycles 60 --serve    # + live dashboard on :8000
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from vajra.api.server import create_app
from vajra.config import LATENCY_BUDGET_S
from vajra.engine import VajraEngine
from vajra.verify.metrics import contingency, fss


def headless(cycles: int) -> VajraEngine:
    eng = VajraEngine(source="synthetic")
    latencies = []
    for i in range(cycles):
        st = eng.run_cycle()
        latencies.append(st.latency_s)
        if (i + 1) % 12 == 0:
            print(f"  frame {st.frame:3d}  ts={st.ts[11:16]}Z  "
                  f"cells={len(st.cells):2d}  latency={st.latency_s*1000:6.1f} ms")
    print(f"\n[latency] ingest->alert: mean {np.mean(latencies)*1000:.1f} ms, "
          f"p95 {np.percentile(latencies, 95)*1000:.1f} ms "
          f"(budget {LATENCY_BUDGET_S:.0f} s)  "
          + ("[OK]" if np.mean(latencies) < LATENCY_BUDGET_S else "[OVER BUDGET]"))

    # ---- verification summary by lead ---------------------------------
    by_lead: dict[int, list[dict]] = defaultdict(list)
    for v in eng.verify_history:
        by_lead[v["lead_min"]].append(v)
    print("\n[verification] 40 dBZ threshold, model vs persistence baseline")
    print(f"  {'lead':>5} {'CSI':>6} {'POD':>6} {'FAR':>6} {'HSS':>6} | "
          f"{'CSI(base)':>9} {'FSS8':>6} {'FSS16':>6} {'FSS32':>6} {'FSS32(base)':>11}")
    for lead in sorted(by_lead):
        vs = by_lead[lead]
        m = {k: float(np.mean([x['model'][k] for x in vs])) for k in ('csi', 'pod', 'far', 'hss')}
        b = {k: float(np.mean([x['persistence'][k] for x in vs])) for k in ('csi', 'pod', 'far')}
        f8 = float(np.mean([x['fss8_model'] for x in vs]))
        f16 = float(np.mean([x['fss16_model'] for x in vs]))
        f32 = float(np.mean([x['fss32_model'] for x in vs]))
        f32b = float(np.mean([x['fss32_persistence'] for x in vs]))
        print(f"  {lead:>4}m {m['csi']:>6.3f} {m['pod']:>6.3f} {m['far']:>6.3f} "
              f"{m['hss']:>6.3f} | {b['csi']:>9.3f} {f8:>6.3f} {f16:>6.3f} {f32:>6.3f} {f32b:>11.3f}")

    # ---- ETA demo ------------------------------------------------------
    print("\n[ETA countdown demo]")
    for name, e in eng.last_state.etas.items():
        if e:
            print(f"  {name}: cell #{e['cell_id']} arrives in "
                  f"{e['eta_min']} +/- {e['sigma_min']} min "
                  f"({e['speed_kmh']} km/h, MESH {e['max_mesh_mm']} mm)")

    # ---- CAP sample ----------------------------------------------------
    if eng.last_state.alerts:
        print("\n[CAP 1.2 alert sample]")
        print(eng.last_state.alerts[0]["cap_xml"])
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=48)
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    eng = headless(args.cycles)
    if args.serve:
        print("\nStarting dashboard on http://127.0.0.1:8000 ...")
        import uvicorn
        app = create_app("synthetic", interval_s=2.0)
        app.state.live.engine = eng          # continue from the headless run
        app.state.live.step_once()           # refresh rasters/ci from that engine
        uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
