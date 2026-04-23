# process_log.md — Hour-by-Hour Timeline & LLM Usage Declaration
## T2.3 Grid Outage Forecaster + Appliance Prioritizer

---

## Hour-by-Hour Timeline

| Time | Activity |
|------|----------|
| H+0:00 | Read challenge brief (T2.3). Identified 6 scored criteria. Decided on LightGBM over Prophet (faster, better on tabular lags, CPU-safe). |
| H+0:30 | Designed project structure: generate_data.py, forecaster.py, prioritizer.py, eval.ipynb, lite_ui.html, digest_spec.md. |
| H+0:45 | Wrote generate_data.py — implemented spec DGP exactly (sigmoid outage formula, LogNormal duration, dual-peak load). Verified 4% base outage rate. |
| H+1:15 | Wrote forecaster.py — 36 feature columns, LightGBM classifier + regressor, isotonic calibration. Fixed sklearn 1.3+ CalibratedClassifierCV API change. |
| H+1:45 | Wrote prioritizer.py — `plan()` function with category-sort enforcing luxury-before-critical rule. Verified hour 08h triggers medium risk. |
| H+2:00 | Ran data generation + training: Brier=0.070, Duration MAE=38.4 min, inference 56ms. |
| H+2:15 | Fixed forecast function to use hour-of-day profiles (not flat last-row), giving realistic 24h variation. |
| H+2:30 | Wrote eval.ipynb — calibration curve, feature importance, worst-hour analysis, all 5 plots. |
| H+2:45 | Wrote lite_ui.html (13KB static page) — Chart.js forecast band, appliance heatmap, SMS digest, offline resilience section. |
| H+3:00 | Wrote digest_spec.md — 3 SMS ≤160 chars, staleness budget analysis (4h), LED relay board accessibility design. |
| H+3:15 | Wrote README.md (2-command Colab setup), run_all.py, SIGNED.md. |
| H+3:30 | Git commit and push to GitHub. Verified all deliverables present. |

---

## LLM / Tool Use Declaration

**Tools used:** Claude Code (Anthropic, claude-sonnet-4-6) via VS Code extension.

### Sample Prompts Sent

**Prompt 1** (feature engineering):
> "Write a build_features() function for a LightGBM time-series classifier. Input: hourly DataFrame with load_mw, temp_c, humidity, wind_ms, rain_mm, outage. Output: add lag features (1,2,3,6,12,24,48h), rolling mean/std (3,6,12,24h), hour-of-day, dow, seasonal indicators, and interaction terms. The DGP is sigmoid(a0 + a1*load_lag1 + a2*rain + a3*hour)."

**Prompt 2** (prioritizer logic):
> "Write a plan() function that takes a 24h forecast DataFrame and a list of appliances (each with name, category in [critical,comfort,luxury], watts_avg, revenue_if_running_rwf_per_h). For each hour: if p_outage < 0.10 all ON, if 0.10–0.25 drop luxury, if ≥0.25 critical only. Sort within each tier by revenue/watt descending. Return a schedule dict."

**Prompt 3** (lite_ui.html):
> "Write a static HTML page under 50KB for a grid outage dashboard. Embed forecast data as JSON. Use Chart.js from CDN. Show: (1) filled area chart with uncertainty band, (2) stacked bar chart for appliance ON/OFF plan, (3) hourly table with risk badges, (4) SMS digest section. Must work offline after first load."

**Prompt discarded:**
> "Use Prophet for the forecaster instead of LightGBM."
> *Discarded because*: Prophet requires pandas<2.0 on Colab, inference is slow (>300ms), and doesn't natively produce calibrated probabilities. LightGBM is faster, more accurate on tabular lags, and easier to calibrate.

---

## Single Hardest Decision

**Decision**: Use isotonic regression calibration over Platt scaling (sigmoid).

The spec rewards Brier score, which penalises miscalibrated probabilities heavily. LightGBM's raw probabilities are well-ranked (good AUC) but not well-calibrated (overconfident near 0 and 1). Isotonic regression on the validation set corrects this non-parametrically. The trade-off: isotonic can overfit on small validation sets (712 hours here). I accepted this because 712 hours provides ~55 positive examples — enough for isotonic to be stable. Platt scaling would be safer with fewer positives but less expressive. For the 4% base rate in the spec, isotonic is the better choice.
