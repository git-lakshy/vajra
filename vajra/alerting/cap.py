"""CAP 1.2 (Common Alerting Protocol) XML emission.

SACHET/NDMA-compatible alert path: one CAP alert per severe tracked cell.
"""

from __future__ import annotations

from datetime import datetime, timezone

from xml.sax.saxutils import escape


def cap_alert(track_id: int, hazard: str, severity: str, certainty: str,
              headline: str, description: str, polygon_ring: list[list[float]],
              eta_min: float | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    poly = " ".join(f"{lon},{lat}" for lon, lat in polygon_ring)
    eta_line = (f" Expected arrival in about {eta_min:.0f} minutes."
                if eta_min is not None else "")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>VAJRA-{track_id}-{hazard}-{now}</identifier>
  <sender>vajra@nowcast.local</sender>
  <sent>{now}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-IN</language>
    <category>Met</category>
    <event>{escape(hazard)}</event>
    <responseType>Prepare</responseType>
    <urgency>Immediate</urgency>
    <severity>{escape(severity)}</severity>
    <certainty>{escape(certainty)}</certainty>
    <headline>{escape(headline)}</headline>
    <description>{escape(description + eta_line)}</description>
    <area>
      <areaDesc>VAJRA tracked storm cell #{track_id}</areaDesc>
      <polygon>{poly}</polygon>
    </area>
  </info>
</alert>"""
