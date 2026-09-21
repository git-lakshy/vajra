# VAJRA — Hyperlocal Severe-Weather Nowcasting

**Velocity-And-Jump Radar-satellite Analytics** · Smart India Hackathon PS46

> *"A district is roughly 4,000 km². A hailstorm is roughly 20 km². We warn the 20."*

VAJRA is an operational nowcasting system — not a rain-prediction notebook. It fuses
radar, satellite, lightning, terrain, and NWP data on a shared 1 km grid, produces
**calibrated hazard probabilities** (lightning · hail · downburst · cloudburst) for
**0–6 hour** leads, tracks storms as objects with a **live arrival countdown**, and
emits **CAP 1.2 alerts** to district, aviation, and agriculture views.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

| Capability | Detail |
|---|---|
| Multi-source fusion | Radar mosaic + satellite IR + lightning strokes + terrain + NWP on one 1 km grid |
| Probabilistic nowcast | 8-member semi-Lagrangian ensemble (pySTEPS-class motion), 5–120 min leads |
| Convective-initiation head | Satellite cloud-top cooling → P(CI in 30/60/90 min), works without radar |
| 4 hazard heads | Lightning (2σ jump + trained LightGBM), hail (MESH + learned severe-growth), downburst (shear + rotation), cloudburst (IMD 100 mm/h + area + terrain) |
| Countdown clock | Kalman-tracked cells → *"Nagpur: cell #7 in 28 ± 6 min (47 km/h)"* |
| CAP 1.2 alerts | Per-cell XML polygons, SACHET/NDMA-compatible |
| Verification built-in | CSI/POD/FAR/HSS, FSS(8/16/32), reliability, Brier, cost–loss, ETA errors — model vs baselines on every run |
| Replayable | Seeded synthetic case + real Dec-2021 EF4 case at 5-min cadence; demo never needs live weather |

## 🚀 Quickstart

```powershell
# one command: venv -> deps -> tests -> server -> browser
.\start_mvp.bat
```

Manual:

```powershell
pip install -r requirements.txt
python -m vajra.api.server --port 8000   # dashboard at http://127.0.0.1:8000
```

Headless runs:

```powershell
python run_mvp.py --cycles 48            # synthetic replay + verification + ETA + CAP
python run_real_case.py --cycles 96      # real MRMS Dec-2021 case end-to-end
python run_verification.py               # full verification suite -> docs/verification/
python scripts\train_sevir.py           # (re)train SEVIR LightGBM heads -> models/
```

## 🏗️ Architecture

```
sources ──► 1 km grid + ring buffer ──► motion ──► ensemble nowcast (0–2 h)
   │              (SlabStore)              │              │
   │                                       ▼              ▼
   │── satellite ──► CI head ──► Tier-3 blend ──► 4 hazard heads ──► tracks/ETA ──► CAP/API/UI
```

Each cycle (`VajraEngine.run_cycle()`): ingest → motion → ensemble → CI →
hazards (+trained LightGBM overrides) → Kalman tracking → ETA → CAP → state,
timed against the 90 s ingest-to-alert budget.

```
vajra/                 pipeline package
  ingest/              synthetic replay · MRMS GRIB2/cache · GOES ABI+GLM · store
  nowcast/             LK optical flow + ensemble · CI head · Tier-3 blend weights
  hazards/             physics heads + SEVIR-trained LightGBM (isotonic-calibrated)
  tracking/            storm objects · Kalman tracks · ETA countdown
  alerting/            CAP 1.2 composer
  verify/              contingency, FSS, Brier, cost–loss
  api/                 FastAPI: state · PNG rasters · ETA · CAP · verification
dashboard/             Leaflet UI: 3 role views · layers · countdowns · Why panel
models/                trained LightGBM + isotonic calibrations (auto-fallback if absent)
```

## 📊 Verified performance

| Metric | VAJRA | Persistence baseline |
|---|---|---|
| CSI @40 dBZ, +30 min | **0.24–0.30** | 0.00–0.01 |
| FSS-8 km | **0.42–0.50** | 0.20–0.27 |
| Lead-time curve | beats persistence at **every** lead (5–120 min) | — |
| Tier-3 crossover | extrapolation → 0 by 120 min; **GFS leg CSI 0.09–0.13 at 3–6 h** | — |
| ETA countdown | median error **+6.6 min**, MAE 9.5 min (n=24, real case) | — |
| Lightning head (SEVIR) | **AUC 0.779**, Brier 0.029 | — |
| Latency | **~1.1 s** ingest→alert per cycle (budget 90 s) | — |

Full reports: `docs/verification/verification_report.md`. Every number above is
produced by the system, not asserted — see `run_verification.py`.

## 🔌 API

| Endpoint | Description |
|---|---|
| `GET /api/state` | Frame, cells, hazard summary, ETAs, alerts, latency, QC |
| `GET /api/raster/{obs\|fcst\|ci\|spread}/{value}` | PNG overlays for the map |
| `GET /api/eta?lat=&lon=` | Arrival countdown for any point |
| `GET /api/alerts/cap` | CAP 1.2 alert XML |
| `GET /api/verify` | Rolling model-vs-baseline scores |
| `POST /api/step` · `POST /api/pause/{on\|off}` | Replay control |

## 🗺️ Roadmap

Active work tracks in [`advanced-vajra`](https://github.com/git-lakshy/vajra/tree/advanced-vajra):

- [x] Tier-3 live blend — 180-min blended lead (extrapolation × CI) + dashboard layer
- [x] Rotation-track wiring into the downburst head
- [x] Dead-code removal · ML transparency (`ml_active` in API + UI badge)
- [x] Safety banner · per-cycle JSONL audit trail · QC/freshness flags
- [x] Satellite-threshold (BT<240K) baseline in the metrics module
- [ ] Calibrated hail blend — `HAIL_ML_WEIGHT` named in config; fit on labels
- [ ] Decision layer (role cost–loss thresholds) · second real case
- [ ] CI-as-LightGBM · evidence graphs · modality-dropout training
- [ ] Indian sensor adapters (MOSDAC/IMD, registration-gated)

## ⚠️ Status

Research prototype — outputs are **decision support, not official warnings**.
Validated on one real US severe case + SEVIR transfer path; Indian fine-tuning
pending data access. Limitations are documented honestly in `PROJECT.md §6`.

## 📚 References

- pySTEPS (Pulkkinen et al.) — motion + semi-Lagrangian reference implementation
- SEVIR (MIT, NeurIPS 2020) — pretraining data
- NOAA MRMS / GOES-16 / GFS open data — real-case replay
- IMD MAUSAM (Pradhan et al.) — Indian operational nowcasting baseline
- DGMR (DeepMind, Nature 2021) · NowcastNet (Nature 2023) — generative-nowcast direction

## 📄 License

MIT — see `LICENSE` (to be added) for terms. Data sources retain their own
licenses (NOAA/NASA open data, MIT SEVIR).
