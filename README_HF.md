---
title: Grid Outage Forecaster
emoji: ⚡
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: 24h grid outage forecast + appliance load-shed planner for Rwanda SMEs
---

# ⚡ Grid Outage Forecaster + Appliance Prioritizer

**AIMS KTT Hackathon T2.3 | Nyingi Joseph**

Probabilistic 24-hour grid outage forecaster and appliance load-shed planner for small businesses (salons, cold rooms, tailor shops) in Rwanda. Maximises revenue during outages by ranking appliances critical → comfort → luxury.

## What it does

- **Forecasts** P(outage) for each of the next 24 hours with an 80% confidence band
- **Plans** which appliances to keep on, hour by hour, for your business type
- **Neighbor signal**: if 2+ nearby businesses report live outages, forces all hours to HIGH RISK
- **Live weather tab**: fetches real Kigali weather from OpenMeteo and runs inference

## Model

- LightGBM classifier (P(outage)) + LightGBM quantile regressor (E[duration], q=0.50)
- 5-fold OOF calibration → LogisticRegression meta-learner
- 48 features: cyclical hour encoding, lagged load, rolling stats, DGP interaction terms
- Brier Score: **0.0766** (naive baseline: 0.0787) | ROC-AUC: **0.68** | Lead time: **1.6h**

## GitHub

[Josephnyingi/Grid-Outage-Forecaster-Appliance-Prioritizer](https://github.com/Josephnyingi/Grid-Outage-Forecaster-Appliance-Prioritizer)
