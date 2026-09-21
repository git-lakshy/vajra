# VAJRA: Physics-Guided AI Engine for Severe Weather Nowcasting

**Velocity-And-Jump Radar-satellite Analytics (VAJRA)**  
*An open-source, operational nowcasting framework for convective-scale weather hazards (0–6 Hours) at 1×1 km resolution.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/git-lakshy/vajra/pulls)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📌 Overview

Severe convective atmospheric phenomena—such as mountainous cloudbursts, severe hailstorms, microburst downbursts, and rapid lightning outbreaks—develop and dissipate within **15 to 45 minutes** over localized spatial scales (**5 to 20 km²**). 

Traditional Numerical Weather Prediction (NWP) models (e.g., GFS, standard WRF) update on 6-to-12-hour cycles with grid spacings of 3 to 12 km, creating a critical latency and resolution gap during rapid atmospheric intensification.

**VAJRA** bridges this gap. It is a real-time, physics-guided AI nowcasting pipeline that operates at a **5-minute cadence** on a native **1×1 km analytical grid** (262,144 cells per $512 \times 512\text{ km}$ domain). By fusing Doppler Weather Radar (DWR), geostationary satellite thermal infrared (INSAT-3D/GOES), ground lightning strike networks, and digital elevation models, VAJRA produces calibrated hazard probabilities, tracks convective storm cells via Kalman filtering, and outputs actionable minute-by-minute arrival countdowns and OASIS Common Alerting Protocol (CAP v1.2) warnings.

---

## ✨ Key Capabilities

| Capability | Technical Approach | Operational Benefit |
| :--- | :--- | :--- |
| **Multi-Sensor Fusion** | Unified 1 km analytical grid combining radar reflectivity, VIL, satellite IR ($T_b$), lightning strokes, and SRTM terrain elevation. | Ingests heterogeneous sensor feeds into a normalized spatial slab memory store. |
| **Atmospheric Motion Extrapolation** | Multi-scale Pyramidal Lucas-Kanade optical flow with 3-sigma MAD outlier rejection and $k$-d tree IDW dense interpolation. | Generates dense advection velocity fields $(u, v)$ without artificial blur. |
| **Stochastic Ensemble Nowcasting** | Semi-Lagrangian backward-trajectory advection ($\mathbf{x}_{\text{src}} = \mathbf{x} - \mathbf{v}\Delta t$) with lead-time damped growth/decay and multi-member perturbation. | Delivers probabilistic risk envelopes rather than a single deterministic guess across 5–180 min leads. |
| **Calibrated Machine Learning** | SEVIR-trained LightGBM gradient-boosted decision trees with Pool Adjacent Violators Algorithm (PAVA) Isotonic Calibration. | Verified probability calibration: when VAJRA predicts 80% risk, empirical verification confirms an 8-in-10 occurrence rate. |
| **Satellite Convective Initiation (CI)** | Tier-1 satellite-first infrared time-differencing ($dT_b/dt < -4\,\text{K}/15\text{ min}$, $T_b < 240\,\text{K}$). | Overcomes radar terrain blockage in deep mountain valleys, alerting **30–90 min before the first radar echo**. |
| **Multi-Hazard Dedicated Heads** | 4 physically-grounded diagnostic heads for **Cloudburst** ($\ge 100\text{ mm/h}$ + slope gradient), **Hail** (MESH power-law proxy), **Downburst** (azimuthal shear + core collapse), and **Lightning Jump** (2σ flash surge). | Eliminates false alarms from naive single-variable rainfall thresholding. |
| **Cell Tracking & ETA Countdown** | TITAN-style morphological storm segmentation coupled with constant-velocity Kalman filter tracking. | Computes live arrival countdowns for critical infrastructure: *'Arriving at Station X in $24 \pm 5\text{ min}$ at $45\text{ km/h}$'*. |
| **Standardized CAP Alerting** | Automated composer for OASIS Common Alerting Protocol (CAP v1.2) XML with geofenced alert polygons. | Direct interoperability with national disaster warning gateways (such as NDMA SACHET). |

---

## 🏗️ System Architecture

```
                      ┌──────────────────────────────────────────────┐
                      │          Heterogeneous Ingestion             │
                      │  Doppler Radar · INSAT/GOES IR · Lightning   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    1 km Analytical SlabStore (262k cells)    │
                      │  Terrain Digital Elevation Model (DEM) Grid  │
                      └──────────────┬───────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│  Tier-1: Convective Initiation  │   │  Tier-2: Atmospheric Motion Engine  │
│  Satellite IR Cloud-Top Cooling │   │  Pyramidal Lucas-Kanade Flow + IDW  │
│  dTb/dt < -4 K / 15 min         │   │  Semi-Lagrangian Back-Advection     │
└────────────────┬────────────────┘   └──────────────────┬──────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                      ┌──────────────────────────────────────────────┐
                      │  Hazard Diagnostic & ML Inference Heads      │
                      │  • Lightning Jump (2σ flash-rate surge)      │
                      │  • Hail MESH (VIL density integration)       │
                      │  • Downburst (Azimuthal shear + core drop)   │
                      │  • Cloudburst (>=100 mm/h over >=20 km²)     │
                      │  • LightGBM GBDT Ensembles (SEVIR-Trained)   │
                      │  • PAVA Isotonic Probability Calibration     │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │     Kalman Object Tracking & POI ETA         │
                      │     Discrete Storm Cells · Vector Paths      │
                      └──────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
        ┌────────────────────────────────┐      ┌────────────────────────────────┐
        │       FastAPI Endpoints        │      │    CAP v1.2 XML Dispatcher     │
        │ State · PNG Rasters · Sparklines│     │ NDMA SACHET · Multi-Channel    │
        └────────────────────────────────┘      └────────────────────────────────┘
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/git-lakshy/vajra.git
cd vajra

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Launching the Dashboard Server

```bash
# Start the FastAPI nowcast server (default port 8000)
python -m vajra.api.server --port 8000
```
Once started, navigate to `http://127.0.0.1:8000` in your web browser to open the interactive Operations Console.

### Docker Deployment

```bash
# Build container image
docker build -t vajra-nowcast .

# Run container
docker run -d -p 8000:8000 --name vajra vajra-nowcast
```

---

## 💻 CLI & Headless Execution

VAJRA includes automated drivers for offline evaluation, historical storm replay, and model training:

```bash
# Run headless synthetic scenario cycle replay
python run_mvp.py --cycles 48

# Run historical severe convective case replay
python run_case_study.py --cycles 96

# Execute end-to-end verification benchmark suite
python run_verification.py

# Train LightGBM hazard heads on SEVIR storm catalog
python scripts/train_sevir.py --max-events 300
```

---

## 📊 Empirical Verification & Benchmarks

VAJRA is verified against historical convective case studies and standard held-out validation sets:

| Evaluation Metric | VAJRA System | Persistence Baseline | Reference / Dataset |
| :--- | :--- | :--- | :--- |
| **CSI @ 40 dBZ (+30 min lead)** | **0.24 – 0.30** | 0.00 – 0.01 | Radar volume scan archive |
| **Fractions Skill Score (FSS-8 km)** | **0.42 – 0.50** | 0.20 – 0.27 | Spatial neighborhood verification |
| **Lead-Time Skill Curve** | Superior at **every lead** (5–120 min) | Rapid decay to zero | Comparative verification suite |
| **ETA Countdown Error** | Median error **+6.6 min** (MAE 9.5 min) | N/A | Kalman cell tracking vs ground stations |
| **Lightning Hazard Model** | **AUC = 0.779**, Brier = 0.029 | Uncalibrated baseline | SEVIR held-out test events ($n=3,300$) |
| **Severe Convection / Hail Model** | **AUC = 0.993**, CSI = 0.974 | Climatological base rate | SEVIR held-out test events ($n=3,300$) |
| **Inference Processing Latency** | **~250 – 350 ms** per 5-min cycle | N/A | Standard multi-core x86 CPU |

*Detailed verification metrics, ROC curves, and reliability diagrams are available in `docs/verification/`.*

---

## 🔌 API Reference

The server exposes a RESTful API for downstream integration:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/state` | Returns current frame cycle, active storm cells, hazard summaries, and POI ETAs. |
| `GET` | `/api/raster/{type}/{lead}` | Returns georeferenced PNG overlays (`type`: `refl`, `wind`, `mesh`, `ci`, `bt`, `spread`). |
| `GET` | `/api/alerts/outbox` | Retrieves active emergency alert bulletins and dispatch status. |
| `GET` | `/api/alerts/cap` | Generates compliant OASIS CAP v1.2 XML for the current highest-severity cell. |
| `GET` | `/api/scenarios` | Lists configured regional domain scenarios. |
| `POST` | `/api/scenario/{id}` | Switches active meteorological domain scenario (`uttarakhand_cloudburst`, `delhi_squall`, `kolkata_kalbaishakhi`, `nagpur_vidarbha`). |
| `POST` | `/api/step` | Advances the simulation or radar replay by one 5-minute scan cycle. |
| `POST` | `/api/pause/{on\|off}` | Toggles real-time automated clock progression. |

---

## 📁 Repository Structure

```
vajra/
├── alerting/          # OASIS CAP v1.2 alert composer, dispatchers, and policy engine
├── api/               # FastAPI application, state serializers, and raster encoders
├── hazards/           # Physical hazard heads (hail, lightning, downburst, cloudburst) & ML models
├── ingest/            # Multi-source adapters (radar composites, satellite IR, lightning, DEM terrain)
├── nowcast/           # Pyramidal Lucas-Kanade optical flow, semi-Lagrangian advection, and CI models
├── tracking/          # Morphological storm cell segmentation, Kalman filter tracker, and POI ETA clock
└── verify/            # Meteorological verification metrics (CSI, POD, FAR, HSS, FSS, Brier score)

dashboard/             # Leaflet-based real-time operations console (multi-sensor overlays & alerts)
models/                # Pre-trained LightGBM booster weights and isotonic calibration artifacts
docs/                  # Verification reports, case studies, and architecture documentation
scripts/               # Dataset processing and SEVIR machine learning training pipelines
tests/                 # Automated unit and integration test suite
```

---

## 📚 References & Scientific Foundations

1. **pySTEPS:** Pulkkinen et al., *pySTEPS: an open-source Python library for probabilistic precipitation nowcasting*, Geoscientific Model Development (2019).
2. **SEVIR Dataset:** Veillette et al., *SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology*, NeurIPS (2020).
3. **Severe Convection in India:** Pradhan et al., *Doppler Weather Radar applications for nowcasting severe convective events*, MAUSAM (Indian Meteorological Department).
4. **Deep Learning Nowcasting:** Ravuri et al., *Skilful precipitation nowcasting using deep generative models of radar*, Nature (2021); Zhang et al., *NowcastNet*, Nature (2023).
5. **Common Alerting Protocol (CAP):** OASIS Standard CAP v1.2 / ITU-T Recommendation X.1303.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Associated open datasets (NOAA MRMS, GOES, NASA SRTM, MIT SEVIR) remain under their respective open-access public licenses.
