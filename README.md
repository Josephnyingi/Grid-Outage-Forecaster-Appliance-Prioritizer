# T2.3 · Grid Outage Forecaster + Appliance Prioritizer

## AIMS KTT Hackathon — Tier 2 | Nyingi Joseph

---

## Problem

Small businesses in Rwanda — salons, cold rooms, tailor shops — lose significant revenue every time the grid goes down unexpectedly. A salon with a hair dryer running during an outage loses the appliance startup surge and the customer. A cold room that doesn't pre-shed non-critical load risks food spoilage.

The challenge: **predict grid outages 24 hours ahead and tell the business owner exactly which appliances to keep on, hour by hour, before the outage hits** — using only a feature phone (SMS) and an offline dashboard.

---

## Approach

The solution has three layers:

### 1. Probabilistic Forecaster (`forecaster.py`)

**Why LightGBM?** It handles tabular time-series well, trains in under 30 seconds on a laptop CPU, and produces calibrated probabilities without a GPU. Prophet was considered and rejected — it requires pandas <2.0 on Colab and doesn't natively produce calibrated P(outage).

**Features (48 total):**

- Lagged load: 1, 2, 3, 6, 12, 24, 48h — load at t-1 is the strongest single predictor (matches the DGP formula)
- Rolling stats: 3, 6, 12, 24h mean and std of load, rain, humidity
- Cyclical encoding: `sin(2π·hour/24)`, `cos(2π·hour/24)` — raw hour integers don't wrap correctly at midnight
- DGP interaction terms: `rain × load_lag1`, `rain × hour_sin` — explicitly encode the sigmoid formula structure

**Calibration — why OOF instead of a fixed holdout?**
A fixed 70/15/15 split caused distribution shift: the calibration window had an 11% outage rate while the eval window had 8.6%. This made the isotonic calibrator systematically overestimate risk. The fix: 5-fold TimeSeriesSplit out-of-fold (OOF) predictions fed into a LogisticRegression meta-learner. OOF spans the full training period, so the meta-learner sees the same distribution it will predict on.

**Duration:** LightGBM quantile regressor at q=0.50, which is MAE-optimal for log-normal distributions — better than predicting the mean (which is pulled high by long tails).

### 2. Appliance Prioritizer (`prioritizer.py`)

Sorts appliances into three tiers: **critical** (revenue-generating machinery) → **comfort** (lighting, fans) → **luxury** (TV, sound). Within each tier, highest revenue-per-watt stays on.

Per-hour decision rule:

- P < 10% → all ON
- 10% ≤ P < 25% → luxury OFF
- P ≥ 25% → critical only

**Neighbor signal:** If 2+ nearby businesses report live outages via SMS in the last 30 minutes, all hours are forced to HIGH RISK regardless of the model's output. The ML model is slow to react to fault propagation (load features need time to spike). Two confirmed neighbor reports are a faster signal.

```bash
python3 prioritizer.py --forecast outputs/forecast_2024-12-31.csv \
    --business salon --neighbor-alerts 2
```

### 3. Delivery (`lite_ui.html` + SMS digest)

- **lite_ui.html** — 13 KB static page, no server, works offline. Embeds the forecast as JSON directly. Open with `open lite_ui.html` in terminal.
- **SMS digest** — 3 messages at 06:30 AM, ≤160 characters each, no smartphone or data plan required.
- **LED relay board** — for non-literate users: green/amber/red LED physically cuts power to luxury-tier circuits.

---

## How to Run

### Full pipeline (2 commands)

```bash
pip install -r requirements.txt
python run_all.py
```

This runs in sequence:

1. `generate_data.py` — generates 365 days of hourly grid data in `data/`
2. `forecaster.py --train` — trains LightGBM classifier + regressor, saves to `models/`
3. `forecaster.py --forecast --eval` — produces `outputs/forecast_YYYY-MM-DD.csv` and `outputs/eval_metrics.json`
4. `prioritizer.py` — produces `outputs/plan_salon.json`, `plan_cold_room.json`, `plan_tailor.json`

Then open the dashboard:

```bash
open lite_ui.html
```

### Run individual components

```bash
# Regenerate data only
python generate_data.py

# Train only
python forecaster.py --train

# Forecast + evaluate only
python forecaster.py --forecast --eval

# Plan for one business
python prioritizer.py --forecast outputs/forecast_2024-12-31.csv --business salon

# Plan with neighbor override (2 nearby outage reports)
python prioritizer.py --forecast outputs/forecast_2024-12-31.csv --business salon --neighbor-alerts 2

# Open evaluation notebook
jupyter notebook eval.ipynb
```

---

## Project Structure

```
├── generate_data.py        # Synthetic data: 365 days hourly, spec-compliant DGP
├── forecaster.py           # LightGBM forecaster: train / forecast / eval
├── prioritizer.py          # Appliance load-shed planner + neighbor override
├── run_all.py              # One-shot pipeline runner
├── eval.ipynb              # Evaluation notebook: calibration, AUC, feature importance
├── lite_ui.html            # Offline dashboard (13 KB, open in any browser)
├── digest_spec.md          # SMS format, staleness budget, LED relay, revenue calc
├── process_log.md          # Hour-by-hour timeline + LLM usage declaration
├── SIGNED.md               # Honor code
├── requirements.txt
├── data/
│   ├── grid_history.csv    # 365 days × 24h = 8,760 rows (generated)
│   ├── appliances.json     # 10 appliances with category + watts + revenue
│   └── businesses.json     # 3 archetypes: salon, cold_room, tailor
├── models/                 # Saved after training
│   ├── outage_classifier.pkl
│   ├── duration_regressor.pkl
│   ├── isotonic_calibrator.pkl
│   ├── meta_scaler.pkl
│   └── meta_key_idx.pkl
└── outputs/
    ├── forecast_YYYY-MM-DD.csv   # 24h probabilistic forecast
    ├── plan_salon.json           # Per-hour appliance plan
    ├── plan_cold_room.json
    ├── plan_tailor.json
    └── eval_metrics.json         # Brier, MAE, lead time
```

**Workflow:** `generate_data` → `forecaster (train)` → `forecaster (forecast+eval)` → `prioritizer` → `lite_ui.html`

---

## Results & Interpretation

All metrics are from a 30-day held-out rolling evaluation (`eval.ipynb`).

| Metric | Value | Interpretation |
| ------ | ----- | -------------- |
| Brier Score | **0.0766** | Lower is better. Baseline (always predicting the mean rate) = 0.0787 |
| Brier Skill Score | **+0.027** | Positive = beats naive. The signal is real, not noise. |
| ROC-AUC | **0.68** | Random = 0.50. 0.68 means the model correctly ranks 68% of outage vs. non-outage pairs — meaningful discrimination on a 4% base-rate event. |
| Duration MAE | **56 min** | Mean absolute error on predicted outage duration. Typical outages last ~90 min; being within 56 min is enough to plan a useful contingency window. |
| Avg Lead Time | **1.6 h** | On average, the model flags an elevated-risk hour 1.6 hours before the outage. This is enough time for an owner to act on an SMS alert. |
| Inference time | **< 60 ms** | Well under the 300 ms spec constraint. |
| Training time | **< 30 s** | Well under the 10 min spec constraint. |

### Why the Brier improvement is modest — and why that's expected

The synthetic DGP has a maximum feature-outage correlation of ~0.08 (load_lag1 vs outage). The ground-truth formula `P = σ(−3.5 + 0.25·load_lag1 + 0.08·rain + 0.04·hour)` produces outages that are genuinely hard to predict from load and weather alone — most variance is pure noise. The OOF + meta-learner approach improved BSS from 0.0013 (near zero) to 0.027 (20× improvement) by fixing calibration distribution shift. Further gains would require additional signals (e.g., neighbour reports, infrastructure age), not more ML.

### Revenue impact (Salon, typical outage week)

| Scenario | Weekly revenue lost |
| -------- | ------------------ |
| Naïve (all on, no plan) | ~53,200 RWF lost to outages |
| With load-shed plan | ~47,880 RWF lost — saves **~5,320 RWF/week** |
| + startup spike savings | Additional **~2,100 RWF/week** |
| **Net weekly benefit** | **~7,420 RWF/week (~30,000 RWF/month)** |

The saving comes from two sources: critical appliances restart first (protecting the highest-revenue work), and avoiding startup current spikes on non-critical devices during uncertain hours.

---

## Technical Constraints Met

| Constraint | Required | Actual |
| ---------- | -------- | ------ |
| CPU-only | yes | ✅ |
| Re-training time | < 10 min | < 30 s |
| Forecast inference | < 300 ms | < 60 ms |
| `lite_ui.html` size | < 50 KB | 13 KB |
| Reproducibility | ≤ 2 commands | ✅ |

---

## Data

Generated by `generate_data.py`, following the challenge spec exactly:

- **Load (MW):** dual Gaussian peaks at 08h and 19h + weekly seasonality + rainy-season noise (March–May, October–November)
- **Outage probability:** `P = σ(−3.5 + 0.25·load_lag1 + 0.08·rain_mm + 0.04·hour)` — base rate ≈ 4%/h
- **Duration:** `LogNormal(μ=log(90)−0.18, σ=0.6)` — mean ≈ 90 min, clipped 5–480 min
- **Scale:** 365 days × 24 hours = 8,760 rows

The generator script reproduces the full dataset in under 2 minutes on a laptop.

---

Demo Link: https://www.youtube.com/watch?v=XUIn_kkvae0 

## License

MIT — see [LICENSE](LICENSE)
