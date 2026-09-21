"""Live alert drill: real Dec-2021 case -> policies -> approval -> dispatch.

Simulates the operations desk: runs the replay, collects drafts, approves,
dispatches to all channels, and prints receipts. Provider channels without
credentials degrade to file-stub receipts (honest status per channel).

Usage: python run_alert_drill.py [--cycles 96]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vajra.alerting.dispatcher import AlertDispatcher
from vajra.config import POIS
from vajra.engine import VajraEngine

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi/Marathi samples on cp1252
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=96)
    args = ap.parse_args()

    eng = VajraEngine(source="mrms2021")
    disp = AlertDispatcher(pois=list(POIS))
    n = min(args.cycles, eng.n_frames)
    print(f"alert drill: {n} frames of the real case")

    for i in range(n):
        st = eng.run_cycle()
        new_ids = disp.ingest(st)
        # operations desk: approve district/aviation drafts as they appear
        for aid in new_ids:
            r = disp.alerts[aid]
            if r.role in ("district", "aviation"):
                disp.approve(aid)
        disp.dispatch_approved()
        if (i + 1) % 24 == 0:
            print(f"  frame {st.frame}/{n} {st.ts[11:16]}Z "
                  f"alerts={len(disp.alerts)}")

    # ---- drill report ------------------------------------------------------
    by_status: dict[str, int] = {}
    by_channel: dict[str, dict[str, int]] = {}
    for r in disp.alerts.values():
        by_status[r.status] = by_status.get(r.status, 0) + 1
        for rc in r.receipts:
            ch = by_channel.setdefault(rc["channel"], {})
            ch[rc["status"]] = ch.get(rc["status"], 0) + 1
    print(f"\nalerts: {len(disp.alerts)} total {by_status}")
    print("receipts by channel:")
    for ch, stats in sorted(by_channel.items()):
        print(f"  {ch:10s} {stats}")
    print("\nsample drafts:")
    for r in list(disp.alerts.values())[:6]:
        print(f"  {r.id} {r.ts[11:16]} {r.role}/{r.poi} {r.hazard} "
              f"P={r.prob:.2f} ETA={r.eta_min} [{r.status}]")
    # multilingual sample
    from vajra.alerting.directory import render
    print("\nmessage samples (cell #18, Mayfield, P=0.72, ETA 27 min):")
    for lang in ("en", "hi", "mr"):
        print(f"  [{lang}] {render('hail', lang, 'Mayfield', 0.72, 27.0)}")


if __name__ == "__main__":
    main()
