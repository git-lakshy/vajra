# VAJRA: Complete Live Pitch Script & Technical ELI5 Guide (SIH 2024)
**Theme:** *Physics-Guided AI for Hyperlocal Nowcasting of Extreme Convective Weather Hazards (0–6 Hours)*  
**Target Pitch Time:** 3 to 4 Minutes

---

# PART 1: The Live 3-Minute Presentation Script

### [0:00 – 0:35] The Hook & The Problem Statement
*(Speaker stands confident, addressing the panel)*

> "Good morning/afternoon, Respected Judges.
>
> In India, extreme convective weather events—such as Himalayan cloudbursts, destructive downburst winds, killer lightning strikes, and severe hailstorms—are our deadliest and costliest meteorological disasters.
>
> Traditional Numerical Weather Prediction models (like GFS or WRF) update every 6 to 12 hours on a coarse 10 to 12 km grid. But an extreme weather event doesn't strike an entire 5,000 km² district—**it strikes a hyperlocal corridor of 5 to 20 km² and explodes in under 30 minutes**, slipping completely through traditional models.
>
> To solve this critical gap, we built **VAJRA**: an operational, physics-guided AI nowcasting engine that ingests multi-sensor Doppler radar, INSAT-3D satellite, and lightning data every 5 minutes to deliver **hyperlocal 1×1 km early warnings with sub-second processing latency**."

---

### [0:35 – 1:20] Demonstrating the Live Console & 1×1 km Hyperlocal Grid
*(Action: Open `http://127.0.0.1:8000` on the projector. Click the **Scenario dropdown** at the top and select **"Uttarakhand Himalayan Cloudburst"**)*

> "Here is our live Operations Console.
>
> At the top, you can see our **Regional Scenario Selector**. We have modeled India's most critical convective archetypes:
> 1. **Uttarakhand:** Orographic cloudbursts in steep river valleys like Dehradun and Rishikesh.
> 2. **Delhi-NCR:** Fast-moving summer squall lines producing severe 80+ km/h downburst winds.
> 3. **Kolkata:** Severe pre-monsoon Nor'westers (*Kalbaishakhi*) with explosive lightning.
>
> *(Action: Point to the **'⊞ 1×1 km Grid'** button on the map tabs and zoom in slightly onto the storm)*
> Notice this: The entire surveillance domain is divided into **262,144 independent 1×1 km analytical cells**. Unlike conventional broad district alerts, every single square kilometer computes its own physical thresholds and machine learning probabilities in real time."

---

### [1:20 – 2:10] How We Predict Storm Movement (The Technical Engine)
*(Action: Click the **+15m**, **+30m**, and **+60m** timeline buttons)*

> "Notice these timeline buttons. How do we predict where the storm moves without hallucinating?
> 
> 1. **Pyramidal Lucas-Kanade Optical Flow:** We extract persistent Shi-Tomasi corners across consecutive 5-minute radar sweeps to detect instantaneous atmospheric motion vectors $(u, v)$.
> 2. **3-Sigma Outlier Rejection & IDW:** We apply Median Absolute Deviation (MAD) filtering to discard noisy radar artifacts, and interpolate vectors onto the dense 1 km grid using $k$-d tree Inverse-Distance Weighting.
> 3. **Semi-Lagrangian Back-Advection:** We extrapolate the storm field along back-trajectories with lead-time damped growth/decay, preserving sharp storm boundaries where deep learning models fail due to blurring.
> 4. **Object Tracking & Kalman Filtering (TITAN):** Convective cores are segmented into discrete storm objects. Our Kalman filter tracks their speed and direction, giving us an exact arrival vector."

---

### [2:10 – 2:50] Machine Learning & Multi-Sensor Modalities
*(Action: Switch through the Sensor Tabs: **Radar Composite ➔ Wind & Downburst ➔ Hail (MESH) ➔ Convective Initiation ➔ Lightning**)*

> "Under the hood, VAJRA doesn't just threshold rainfall—it uses a dual engine of **calibrated Machine Learning + Atmospheric Physics**:
> 
> - **Trained LightGBM Boosters:** In our production stack, we deployed gradient-boosted decision trees trained on thousands of storm events from the MIT/NOAA SEVIR dataset. Our severe hail model achieves an **AUC of 0.993**, and our lightning model achieves an **AUC of 0.779**.
> - **Isotonic Probability Calibration (PAVA):** To eliminate false alarms, raw model scores are calibrated using Isotonic Regression. When VAJRA outputs 80% risk, verified empirical records prove it occurs 8 out of 10 times.
> - **Satellite-First Convective Initiation (The Radar Blindspot Solution):** Radar cannot see inside deep Himalayan valleys due to ridge blockage. Our Tier-1 Convective Initiation model tracks INSAT-3D infrared cloud-top cooling rates ($dT_b/dt < -4\,\text{K}/15\text{ min}$) to detect storm birth **30 to 90 minutes before the first radar echo appears**."

---

### [2:50 – 3:30] Actionable Early Warnings & Real-World Impact
*(Action: Click on **Rishikesh Station** in the countdown list to zoom the map directly to it. Then click the **'CAP 1.2 XML'** link)*

> "Now, look at our Early Warning Center:
> 
> 1. **Station Arrival Countdowns:** When I click on Rishikesh, the map instantly zooms in. Disaster managers see an exact countdown: *'Hail core arriving in 28 ± 6 minutes at 47 km/h bearing 78°'*.
> 2. **NDMA SACHET-Compliant CAP 1.2 Alerts:** *(Action: Show the opened XML)* We generate OASIS Common Alerting Protocol (CAP v1.2) XML with geofenced polygons—ready for immediate, automated broadcast via SMS, radio, and sirens without human delay.
> 3. **Sector-Specific Action Views:** *(Action: Click 'Aviation' then 'Agriculture' tabs)* Depending on the authority—District Magistrate, Airport Runway Operations, or Agromet farmer advisories—the emergency actions dynamically reconfigure.
>
> **In conclusion:** VAJRA turns complex atmospheric physics and multi-sensor data into actionable, life-saving early warnings with a processing latency of **under 350 milliseconds**.
> 
> Thank you! We are now open for your questions."

---

# PART 2: Comprehensive ELI5 Glossary of All Project Terms

### 1. What is "Nowcasting" (0–6 Hours)?
* **Traditional Forecasting:** Predicts tomorrow's or next week's general weather over large regions using global supercomputers running for hours.
* **Nowcasting (ELI5):** Tracking an active storm right now with radar and calculating exactly where it will hit in the next 0 to 6 hours. It is essentially **real-time GPS turn-by-turn navigation for severe thunderstorms**.

### 2. Hyperlocal $1 \times 1\text{ km}$ Resolution
* **What it means:** Discretizing the $512 \times 512\text{ km}$ surveillance domain into **262,144 independent grid cells**, each measuring exactly $1.0\text{ km} \times 1.0\text{ km}$.
* **ELI5:** A district is 4,000 km², but a cloudburst or hailstorm is only 5 to 20 km². Instead of sounding an alarm for the whole district, VAJRA isolates the specific 1 km grid squares that are directly in harm's way.

### 3. How We Predict Storm Movement (Optical Flow + Semi-Lagrangian)
* **What it means:** 
  1. **Optical Flow:** Finding movement vectors $(u, v)$ between consecutive 5-minute radar scans using Pyramidal Lucas-Kanade corner tracking.
  2. **Semi-Lagrangian Advection:** Tracing the air parcel backwards in time ($\mathbf{x}_{\text{src}} = \mathbf{x} - \mathbf{v} \cdot \Delta t$) to compute future reflectivity.
* **ELI5:** Just like a camera tracking a fast football in a broadcast, optical flow calculates the storm's velocity vector and pushes the pixels forward for $+15$, $+30$, and $+60$ minutes without blurring out the edges.

### 4. Kalman Filter Cell Tracking (TITAN Algorithm)
* **What it means:** Segmenting contiguous severe storm cores ($\ge 45\text{ dBZ}$, area $\ge 8\text{ km}^2$), associating them across scans, and updating their velocity and trajectory covariance via a Kalman filter.
* **ELI5:** Giving each storm cell its own "Flight ID" (e.g., Storm #12) and projecting its flight path towards cities, airports, or pilgrim camps with an arrival uncertainty clock ($\text{ETA} \pm \sigma$).

### 5. SEVIR Machine Learning Models (LightGBM Boosters)
* **What it means:** Gradient-boosted decision tree models trained on the MIT Lincoln Lab / NOAA **SEVIR** (Storm EVent ImagegeRy) dataset:
  - **Lightning Head (`lgbm_lightning.txt`):** Evaluates $P(\text{lightning within 24 km in next 30 min})$ — **AUC = 0.779**.
  - **Severe Convection Head (`lgbm_severe.txt`):** Evaluates severe hail proxy ($VIL \ge 35\text{ kg/m}^2$) — **AUC = 0.993, CSI = 0.974**.
* **ELI5:** Instead of hardcoded guesses, we trained machine learning models on thousands of real severe storm events using 12 meteorological features like storm growth rate and cloud-top freezing trends.

### 6. Isotonic Probability Calibration (PAVA)
* **What it means:** Applying the **Pool Adjacent Violators Algorithm (PAVA)** to calibrate raw model scores into true frequentist probabilities.
* **ELI5:** Standard AI models often say "90% chance" when the real chance is only 40%. Our calibrated models guarantee that when VAJRA says 80%, it actually happened 8 out of 10 times in verified historical records.

### 7. dBZ (Decibels of Reflectivity)
* **What it means:** The logarithmic measure of radar energy backscattered by hydrometeors (water droplets and ice crystals).
* **ELI5 Scale:**
  - **20 dBZ:** Light drizzle.
  - **35–45 dBZ:** Normal moderate-to-heavy rain.
  - **50–55 dBZ:** Intense thunderstorm with strong updrafts.
  - **60–65+ dBZ:** Giant hail stones or extreme cloudburst downpours.

### 8. Cloudburst ($\ge 100\text{ mm/h}$) & Orographic Terrain Coupling
* **IMD Criteria:** Rainfall rate $\ge 100\text{ mm/h}$ over an area of $\approx 20\text{ to }30\text{ km}^2$.
* **ELI5:** It feels like an entire lake fell from the sky in 15 minutes. VAJRA combines radar rain rates with Digital Elevation Model (DEM) terrain slopes to catch mountain valley flash floods.

### 9. MESH (Maximum Estimated Size of Hail)
* **What it means:** A radar microphysics algorithm combining VIL density and the height of the atmospheric freezing levels ($0^\circ\text{C}$ and $-20^\circ\text{C}$) to calculate the maximum diameter of hail stones.
* **ELI5:** If MESH says 19 mm, hail is the size of a 1-rupee coin. If it exceeds 40 mm, it is golf-ball-sized hail capable of shattering vehicle windshields and destroying standing wheat or horticulture crops.

### 10. Downburst / Microburst Wind Shear
* **What it means:** Cold, dense downdrafts plunging from a thunderstorm core and violently spreading out horizontally upon hitting the surface.
* **ELI5:** An invisible hammer of air hitting the ground at 80 to 120 km/h, uprooting trees, collapsing power poles, and causing lethal low-altitude windshear for landing airplanes.

### 11. Lightning Jump (2-Sigma Trigger)
* **What it means:** A rapid surge where the total lightning flash rate exceeds its 45-minute rolling baseline by more than $2\sigma$ ($2$ standard deviations).
* **ELI5:** Lightning is the storm's heartbeat. When an updraft suddenly accelerates, the flash rate explodes 15 to 20 minutes before damaging hail or downbursts hit the ground.

### 12. Convective Initiation (CI) & INSAT Satellite IR
* **What it means:** Pre-radar storm detection using geostationary satellite Thermal Infrared (10.8 µm) cloud-top cooling rates ($dT_b/dt < -4\,\text{K}/15\text{ min}$) and freezing isotherm thresholds ($T_b < 240\,\text{K}$).
* **ELI5:** Radar is blind until rain drops form. INSAT satellite infrared watches cumulus clouds freezing and growing upwards, alerting us **30 to 90 minutes before the storm even appears on radar**.

### 13. VIL (Vertically Integrated Liquid)
* **What it means:** The total mass of liquid water suspended in a vertical column of the atmosphere (measured in $\text{kg/m}^2$).
* **ELI5:** The "water weight" of the cloud. When VIL exceeds $30\text{--}40\text{ kg/m}^2$, the cloud is dangerously overloaded and its core is about to collapse to the ground.

### 14. CAP 1.2 XML (Common Alerting Protocol)
* **What it means:** The OASIS international digital standard for disaster alerting.
* **ELI5:** The universal digital language for emergencies. It encodes the GPS polygon, urgency, severity, and safety instructions so that India's **NDMA SACHET** portal can instantly dispatch targeted SMS and sirens without waiting for manual paperwork.

### 15. Verification Metrics (CSI, POD, FAR, FSS)
* **CSI (Critical Success Index):** Joint measure of hits, misses, and false alarms (higher is better).
* **POD (Probability of Detection):** The percentage of actual severe events that were correctly detected (e.g., $0.85 = 85\%$).
* **FAR (False Alarm Ratio):** How often an alert was issued but nothing happened (lower is better, e.g., $<0.25$).
* **FSS (Fractions Skill Score):** Spatial accuracy score measuring neighborhood skill within an 8 km window.

---

# PART 3: Quick Winning Answers to Tough Judge Questions

**Q: "How do you predict storm motion? Is it just simple linear projection?"**
> *"No, sir. Simple linear projection fails because atmospheric storms rotate and deform. We use **Pyramidal Lucas-Kanade optical flow** on Shi-Tomasi features with **3-sigma MAD outlier rejection** to derive a dense motion vector field. We then perform **Semi-Lagrangian back-advection** with lead-time damped growth/decay, coupled with a **constant-velocity Kalman filter** for discrete storm cell objects."*

**Q: "Why didn't you use Deep Learning (UNet / ConvLSTM / Diffusion) for nowcasting?"**
> *"Deep learning models suffer from the well-documented **'blurring effect'**—after 30 minutes, they smooth out peak rainfall gradients, completely erasing the localized 1×1 km cloudburst peaks that cause casualties. Furthermore, they require power-hungry GPUs and take 5–10 seconds to infer. Our semi-Lagrangian optical flow preserves sharp storm edges, while our calibrated LightGBM models execute in **under 350 milliseconds on standard CPU hardware**."*

**Q: "What if Doppler Radar is unavailable or blocked by mountains?"**
> *"That is why VAJRA is built with a **Tier-1 Satellite-First Convective Initiation engine**. In mountainous zones like Uttarakhand or radar-poor areas in the Northeast, radar beams suffer from severe terrain blockage. Our CI head uses INSAT-3D/3DR geostationary infrared thermal channels to detect rapid cloud-top cooling ($dT_b/dt < -4\,\text{K}/15\text{ min}$), forecasting storm development up to 90 minutes in advance without needing ground radar."*

**Q: "Is your 1×1 km resolution real or just interpolated?"**
> *"It is our native analytical grid. The $512 \times 512\text{ km}$ surveillance domain is discretized into 262,144 independent $1 \times 1\text{ km}$ cells. Every cell evaluates physical parameters (Z-R rain accumulation, VIL density, MESH hail proxy, and terrain slope gradients) as well as 12-dimensional LightGBM feature vectors."*
