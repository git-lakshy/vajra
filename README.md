# VAJRA — PS46 Severe Weather Nowcasting MVP

**Velocity-And-Jump Radar-satellite Analytics** — an operational-systems answer to PS46:
multi-source fusion (radar + satellite + lightning + terrain), 0–6 h hazard-specific
probabilities on a shared 1 km grid, object-based storm tracking with a live countdown
clock, and CAP 1.2 alerting.

> *"A district is roughly 4,000 km². A hailstorm is roughly 20 km². We warn the 20."*

## Quick start

```powershell
pip install -r requirements.txt
python run_mvp.py --cycles 48          # headless: replay + verification table + ETA demo
python -m vajra.api.server --port 8000 # live dashboard at http://127.0.0.1:8000
```

The MVP runs on a **synthetic replay case** by default (reproducible seeded
pre-monsoon storm sequence over the Nagpur/Chhota-Nagpur corridor) so the demo never
depends on live weather. The same pipeline consumes real feeds through the ingest
adapters when data is available.

## Architecture (per the research notes' tier design)

```
vajra/
  config.py                 1 km grid (512x512, Nagpur-centred), thresholds, POIs
  grid.py                   grid <-> lat/lon mapping
  ingest/
    synthetic.py            replay-case generator (refl, VIL, rain, IR BT, shear, strokes)
    store.py                rolling 6 h ring buffer of 5-min slabs (Zarr in production)
  nowcast/
    advection.py            pySTEPS-style OpenCV LK flow + semi-Lagrangian extrapolation
                            + damped trend + stochastic ensemble
    ci.py                   Tier-1 convective-initiation head (cloud-top cooling)
    blend.py                Tier-3 lead-dependent skill-crossover blending
  hazards/heads.py          4 calibrated heads: lightning (w/ 2-sigma jump), hail (MESH
                            proxy), downburst (shear+core), cloudburst (IMD 100 mm/h)
  tracking/tracker.py       storm objects -> Kalman tracks -> ETA countdown
  alerting/cap.py           CAP 1.2 XML per severe cell
  verify/metrics.py         CSI/POD/FAR/HSS + FSS (neighbourhood)
  engine.py                 per-cycle pipeline with latency accounting
  api/server.py             FastAPI: state, PNG rasters, ETA, CAP, verification
dashboard/index.html        Leaflet UI: 3 role views, layers, countdowns, Why panel
run_mvp.py                  headless demo + verification report
```

## Verified behaviour (synthetic case, 48 frames)

```
[latency] ingest->alert mean ~1.1 s per full pipeline cycle (budget 90 s)

[verification @40 dBZ, +30 min]      VAJRA    persistence
  CSI   0.299  HSS 0.414             vs  CSI   0.009
  POD   0.411  FAR 0.306             FSS8 0.496 vs 0.271
  FSS (8/16/32 km) 0.50/0.50/0.51    (neighbourhood scores address double penalty)

[ETA]  "Nagpur: cell #7 arrives in 28.4 +/- 5.7 min (47.4 km/h, MESH 15.7 mm)"
[CAP]  CAP 1.2 XML emitted per severe tracked cell
```

Motion estimation and advection are unit-tested against ground truth (blob
translation recovered at u=+4.00, v=-4.00 px/frame; round-trip correlation 1.000).

## Real data adapters (datasets researched & download scripts ready)

| Source | Access | Status |
|---|---|---|
| **SEVIR** (MIT, GOES-16 + NEXRAD VIL + GLM lightning) | `s3://sevir` — anonymous, verified | `scripts/download_sevir.ps1` (catalog + VIL/IR107/lightning storm-events, ~7 GB) |
| **MRMS** (NOAA mosaic, **MESH hail**, VIL, rotation tracks; 2-min, 2020-10→now) | `s3://noaa-mrms-pds` — anonymous, verified | `scripts/download_mrms_case.ps1` (Dec 10–11 2021 Mayfield EF4 case, 6 products) |
| **GFS** (NWP env fields) | `s3://noaa-gfs-bdp-pds` — anonymous | verified reachable (20211210 12z f000 = 502 MB) |
| **MOSDAC INSAT-3D/3DR** | free account + approval (`mosdac.gov.in/signup`, mdapi, quota 5k files/day) | register day one; parallax-corrected L1B |
| **IMERG-Early** (gap-filler rain) | NASA Earthdata login, `GPM_3IMERGHHE.07` | ~8 MB half-hourly files |
| **IMD** | `imdlib` (PyPI) daily 0.25° rain grids; `api.imd.gov.in` nowcast/AWS/lightning | labels + ops baseline |
| **IMDAA reanalysis** (12 km hourly) | `rds.ncmrwf.gov.in` free registration | training climatology |

Transfer-learning plan (per notes §8.2): pretrain Tier-2 backbone on SEVIR/MRMS, fine-tune
on Indian DWR cases; modality-dropout keeps the radar-off mode honest.

## Routes to production (beyond MVP)

- Tier-2 generative decoder (NowcastNet-style) trained on SEVIR → Indian fine-tune
- Zarr object store + Kafka replay engine; MRMS grib2 decode via cfgrib/eccodes
- Isotonic calibration per hazard from the verification archive (PAVA implemented)
- NWP (GFS/NCUM-R) blending weights fitted per lead — `blend.py` skeleton ready
- SACHET/NDMA delivery path via CAP 1.2 output
