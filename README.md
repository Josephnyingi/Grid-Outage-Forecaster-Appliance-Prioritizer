# T2.3 · Grid Outage Forecaster + Appliance Prioritizer
**AIMS KTT Hackathon — Tier 2**

> Probabilistic 24-hour grid outage forecaster and appliance load-shed planner for small businesses (salons, cold rooms, tailors). Maximises revenue during outages by ranking appliances critical → comfort → luxury. Built for low-bandwidth, offline-first environments in Rwanda.

---

## Quickstart (≤ 2 commands on free Colab CPU)

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` will:
1. Generate 180 days of synthetic grid data (`data/`)
2. Train the LightGBM outage classifier + duration regressor (`models/`)
3. Run 24-hour forecast + rolling 30-day evaluation (`outputs/`)
4. Generate load-shed plans for all 3 business archetypes (`outputs/`)

Then open `lite_ui.html` in any browser to see the live dashboard.

---

## Project Structure

```
├── generate_data.py        # Synthetic data generator (spec-compliant)
├── forecaster.py           # LightGBM probabilistic forecaster
├── prioritizer.py          # Appliance load-shed planner
├── run_all.py              # One-shot runner (train + forecast + plan)
├── eval.ipynb              # Rolling 30-day evaluation notebook
├── lite_ui.html            # Static 50KB dashboard (open in browser)
├── digest_spec.md          # Product & Business adaptation artifact
├── process_log.md          # Hour-by-hour timeline + LLM usage log
├── SIGNED.md               # Honor code
├── requirements.txt
├── data/
│   ├── grid_history.csv    # 180 days hourly (generated)
│   ├── appliances.json     # 10 appliances with categories
│   └── businesses.json     # 3 archetypes: salon, cold_room, tailor
├── models/
│   ├── outage_classifier.pkl
│   ├── duration_regressor.pkl
│   └── isotonic_calibrator.pkl
└── outputs/
    ├── forecast_YYYY-MM-DD.csv
    ├── plan_salon.json
    ├── plan_cold_room.json
    ├── plan_tailor.json
    └── eval_metrics.json
```

---

## Key Results (30-Day Held-Out Evaluation)

| Metric | Value |
|--------|-------|
| Brier Score | **0.0700** |
| Brier Score (naïve baseline) | 0.0703 |
| ROC-AUC | ~0.65 |
| Duration MAE | 38.4 min |
| Avg Lead Time | 1.0 h |
| Inference time | **< 60 ms** (CPU) |
| Training time | **< 30 s** (CPU) |

---

## Architecture

### Forecaster (`forecaster.py`)
- **Model**: LightGBM classifier (P(outage)) + LightGBM regressor (E[duration|outage])
- **Calibration**: Isotonic regression fitted on validation set
- **Features**: 36 engineered features — lagged load, rolling stats, weather, hour-of-day interactions
- **Output**: `p_outage`, `e_duration`, `lower_80`, `upper_80` per hour

### Prioritizer (`prioritizer.py`)
- **Rule**: Drop luxury before critical (enforced by category sort)
- **Thresholds**: Low < 10% ≤ Medium < 25% ≤ High
- **core function**: `plan(forecast, appliances, business_type)` — see live demo in video

### Lite UI (`lite_ui.html`)
- Static HTML + Chart.js (CDN) — **13 KB**, no server required
- Forecast uncertainty band, appliance heatmap, hourly table, SMS digest
- Offline: embed data as JSON, works without internet after first load

---

## Technical Constraints Met

- ✅ CPU-only (no GPU required)
- ✅ Re-training < 10 min (actual: < 30 s)
- ✅ Forecast API < 300 ms (actual: < 60 ms)
- ✅ lite_ui.html < 50 KB (actual: 13 KB)
- ✅ Reproducible in ≤ 2 commands on Colab free CPU

---

## Data

Synthetic data generated via `generate_data.py` following the spec exactly:
- **Load**: Dual-peak (08h, 19h) + weekly seasonality + rainy-season noise
- **Outage**: `P = σ(a0 + a1·load_lag1 + a2·rain + a3·hour)`, base rate 4%/h
- **Duration**: `LogNormal(mean≈90 min, σ=0.6)`

For large datasets (> 100 MB), host on Hugging Face Datasets and link here.
This repo's generator script reproduces the dataset in < 2 minutes on a laptop.

---

## License

MIT
