"""Alert policy engine: thresholds -> persistence -> cooldown -> approval.

A policy decides, per role and hazard, whether the current engine state
warrants an alert DRAFT. Drafts become sendable only after human approval
(projectreq human-in-the-loop), except roles explicitly configured auto.
Cooldown per (POI, hazard) prevents alert fatigue from continuous firing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RolePolicy:
    role: str
    thresholds: dict[str, float]          # hazard -> min probability
    persist_frames: int = 2               # consecutive frames above thr
    cooldown_min: int = 30                # quiet period per (poi, hazard)
    require_approval: bool = True
    channels: tuple[str, ...] = ("cap", "telegram", "file")
    languages: tuple[str, ...] = ("en",)


# Documented default policies (tunable per deployment; thresholds mirror the
# verified case-study policy). Aviation misses cost more than false alarms,
# so its thresholds are lower but cooldown is longer.
DEFAULT_POLICIES: dict[str, RolePolicy] = {
    "district": RolePolicy(
        role="district",
        thresholds={"hail": 0.50, "cloudburst": 0.50},
        persist_frames=2, cooldown_min=30, require_approval=True,
        channels=("cap", "telegram", "sms", "file"),
        languages=("en", "hi"),
    ),
    "aviation": RolePolicy(
        role="aviation",
        thresholds={"downburst": 0.60, "hail": 0.40},
        persist_frames=2, cooldown_min=45, require_approval=True,
        channels=("cap", "telegram", "email", "file"),
        languages=("en",),
    ),
    "agriculture": RolePolicy(
        role="agriculture",
        thresholds={"lightning": 0.40},
        persist_frames=1, cooldown_min=30, require_approval=False,
        channels=("telegram", "whatsapp", "sms", "ivr", "file"),
        languages=("en", "hi", "mr"),
    ),
}


@dataclass
class PolicyEngine:
    """Stateful evaluator: call `evaluate(state)` each engine cycle."""

    policies: dict[str, RolePolicy] = field(
        default_factory=lambda: dict(DEFAULT_POLICIES))
    _above: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _last_fired: dict[tuple[str, str, str], float] = field(default_factory=dict)

    def _minutes(self, ts_iso: str) -> float:
        from datetime import datetime
        return datetime.fromisoformat(ts_iso).timestamp() / 60.0

    def evaluate(self, state) -> list[dict]:
        """Return alert drafts: [{role, poi, hazard, prob, eta_min, ...}]."""
        drafts = []
        now_min = self._minutes(state.ts)
        hs = state.hazards_summary or {}
        for role, pol in self.policies.items():
            for hazard, thr in pol.thresholds.items():
                prob = hs.get(hazard, {}).get("max_prob", 0.0)
                # POI-scoped evaluation: which POIs have an ETA from a cell.
                # District/aviation require a located ETA (a domain-wide alert
                # with no location is the over-warning we criticize); only
                # agriculture advisories may fire domain-wide.
                pois = [p for p, e in (state.etas or {}).items() if e]
                if not pois:
                    if role != "agriculture":
                        continue
                    pois = ["__domain__"]
                for poi in pois:
                    key = (role, poi, hazard)
                    if prob >= thr and state.cells:
                        self._above[key] = self._above.get(key, 0) + 1
                    else:
                        self._above[key] = 0
                    if self._above[key] < pol.persist_frames:
                        continue
                    last = self._last_fired.get(key, -1e9)
                    if now_min - last < pol.cooldown_min:
                        continue
                    self._last_fired[key] = now_min
                    self._above[key] = 0
                    eta = next((e["eta_min"] for p, e in
                                (state.etas or {}).items()
                                if e and p == poi), None)
                    drafts.append({
                        "role": role, "poi": poi, "hazard": hazard,
                        "prob": round(float(prob), 3),
                        "eta_min": eta,
                        "ts": state.ts, "frame": state.frame,
                        "require_approval": pol.require_approval,
                        "channels": list(pol.channels),
                        "languages": list(pol.languages),
                    })
        return drafts
