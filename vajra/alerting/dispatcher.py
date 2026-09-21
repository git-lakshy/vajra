"""Alert dispatcher: drafts -> approval -> multi-channel delivery.

Flow per engine cycle:
  1. PolicyEngine.evaluate(state) -> drafts
  2. drafts become AlertRecords (status: draft, or approved if the role is
     auto-approve, e.g. agriculture advisories)
  3. approve(alert_id) by a human (dashboard / API)
  4. dispatch(alert_id): resolve subscribers for (role, poi) -> render each
     language -> send on each channel -> receipts appended -> audit log

CAP feed: approved hail/downburst drafts also mint CAP 1.2 XML (existing
cap_alert composer) served at /api/capfeed for SACHET-style consumers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .cap import cap_alert
from .channels import CHANNELS
from .directory import Subscriber, render, seed_directory
from .policy import PolicyEngine


@dataclass
class AlertRecord:
    id: str
    role: str
    poi: str
    hazard: str
    prob: float
    eta_min: float | None
    ts: str
    frame: int
    channels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    status: str = "draft"          # draft -> approved -> sent | expired
    receipts: list[dict] = field(default_factory=list)
    cap_xml: str = ""


class AlertDispatcher:
    def __init__(self, pois: list[str] | None = None) -> None:
        self.policy = PolicyEngine()
        self.subscribers: list[Subscriber] = seed_directory(pois or [])
        self.alerts: dict[str, AlertRecord] = {}
        self._seq = itertools.count(1)

    # ------------------------------------------------------------------
    def ingest(self, state) -> list[str]:
        """Evaluate policies on an engine state; return new draft ids."""
        ids = []
        for d in self.policy.evaluate(state):
            aid = f"A{next(self._seq):04d}"
            auto = not d["require_approval"]
            rec = AlertRecord(
                id=aid, role=d["role"], poi=d["poi"], hazard=d["hazard"],
                prob=d["prob"], eta_min=d["eta_min"], ts=d["ts"],
                frame=d["frame"], channels=d["channels"],
                languages=d["languages"],
                status="approved" if auto else "draft")
            self.alerts[aid] = rec
            ids.append(aid)
        return ids

    def approve(self, alert_id: str) -> bool:
        rec = self.alerts.get(alert_id)
        if rec is None or rec.status != "draft":
            return False
        rec.status = "approved"
        return True

    def approve_all(self) -> int:
        n = 0
        for rec in self.alerts.values():
            if rec.status == "draft":
                rec.status = "approved"
                n += 1
        return n

    # ------------------------------------------------------------------
    def dispatch(self, alert_id: str) -> list[dict]:
        rec = self.alerts.get(alert_id)
        if rec is None or rec.status != "approved":
            return []
        subs = [s for s in self.subscribers
                if s.role == rec.role and s.poi in (rec.poi, "__domain__")]
        receipts: list[dict] = []
        for sub in subs:
            for lang in sub.languages or rec.languages:
                text = render(rec.hazard, lang, rec.poi, rec.prob,
                              rec.eta_min)
                meta = {"ts": rec.ts, "role": rec.role,
                        "hazard": rec.hazard, "poi": rec.poi,
                        "alert_id": rec.id, "lang": lang}
                for ch in rec.channels:
                    if ch == "cap":
                        continue  # CAP goes to the feed, not a subscriber
                    chan = CHANNELS.get(ch)
                    if chan is None:
                        continue
                    to = sub.channels.get(ch, "")
                    receipts.append(chan.send(to, text, meta))
        # CAP payload for the feed (hail/downburst carry polygons downstream;
        # polygon ring attached when the engine provides one)
        if "cap" in rec.channels and not rec.cap_xml:
            rec.cap_xml = cap_alert(
                0, rec.hazard.title(), "Severe", "Likely",
                f"VAJRA {rec.hazard} alert {rec.poi} P={rec.prob:.0%}",
                render(rec.hazard, "en", rec.poi, rec.prob, rec.eta_min),
                [[0.0, 0.0]], rec.eta_min)
        rec.receipts.extend(receipts)
        rec.status = "sent"
        self._audit(rec)
        return receipts

    def dispatch_approved(self) -> dict[str, list[dict]]:
        return {aid: self.dispatch(aid) for aid, r in self.alerts.items()
                if r.status == "approved"}

    def outbox(self) -> list[dict]:
        return [{"id": r.id, "role": r.role, "poi": r.poi,
                 "hazard": r.hazard, "prob": r.prob, "eta_min": r.eta_min,
                 "ts": r.ts, "status": r.status,
                 "channels": r.channels, "receipts": r.receipts}
                for r in self.alerts.values()]

    # ------------------------------------------------------------------
    def _audit(self, rec: AlertRecord) -> None:
        try:
            import json
            import os
            os.makedirs("logs", exist_ok=True)
            with open("logs/alerts.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "alert_id": rec.id, "ts": rec.ts, "role": rec.role,
                    "poi": rec.poi, "hazard": rec.hazard, "prob": rec.prob,
                    "eta_min": rec.eta_min, "status": rec.status,
                    "receipts": rec.receipts}) + "\n")
        except Exception:
            pass
