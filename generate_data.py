"""
Synthetic data generator for Grid Outage Forecaster + Appliance Prioritizer.
Spec: 180 days hourly, two daily load peaks, weekly seasonality, rainy-season noise.
Outage probability: sigmoid(a0 + a1*load_lag1 + a2*rain + a3*hour_of_day), base 4%/hr.
Duration: LogNormal(mean≈90 min, sigma=0.6).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def generate_grid_history(n_days: int = 180) -> pd.DataFrame:
    n_hours = n_days * 24
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values   # 0=Mon, 6=Sun
    month = timestamps.month.values

    # Load (MW): morning peak ~8h, evening peak ~19h + weekly seasonality + rainy noise
    morning_peak = 3.0 * np.exp(-0.5 * ((hour - 8) / 2.5) ** 2)
    evening_peak = 2.5 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    weekly = 0.3 * np.sin(2 * np.pi * dow / 7)
    base_load = 1.5 + morning_peak + evening_peak + weekly

    # Rainy season (Mar-May, Oct-Nov) adds noise
    rainy = np.isin(month, [3, 4, 5, 10, 11]).astype(float)
    load_noise = np.random.normal(0, 0.15 + 0.25 * rainy)
    load_mw = np.clip(base_load + load_noise, 0.5, 8.0)

    # Weather
    temp_c = 22 + 4 * np.sin(2 * np.pi * (hour - 14) / 24) + np.random.normal(0, 1.0, n_hours)
    humidity = 55 + 20 * rainy + np.random.normal(0, 5, n_hours)
    humidity = np.clip(humidity, 20, 99)
    wind_ms = np.abs(np.random.normal(3.5, 1.5, n_hours))
    rain_mm = np.where(rainy.astype(bool),
                       np.random.exponential(2.5, n_hours),
                       np.random.exponential(0.3, n_hours))
    rain_mm = np.clip(rain_mm, 0, 50)

    # Outage probability: spec formula
    load_lag1 = np.roll(load_mw, 1)
    load_lag1[0] = load_mw[0]

    a0, a1, a2, a3 = -3.5, 0.25, 0.08, 0.04
    logit = a0 + a1 * load_lag1 + a2 * rain_mm + a3 * hour
    p_outage = 1 / (1 + np.exp(-logit))
    # ensure base rate ≈ 4%
    p_outage = np.clip(p_outage, 0.01, 0.45)

    outage = np.random.binomial(1, p_outage)

    # Duration: LogNormal(mean≈90 min, sigma=0.6) only when outage=1
    log_mean = np.log(90) - 0.5 * 0.6 ** 2   # so E[X]=90
    duration_min = np.where(
        outage == 1,
        np.clip(np.random.lognormal(log_mean, 0.6, n_hours), 5, 480),
        0.0,
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_mw": np.round(load_mw, 3),
        "temp_c": np.round(temp_c, 2),
        "humidity": np.round(humidity, 1),
        "wind_ms": np.round(wind_ms, 2),
        "rain_mm": np.round(rain_mm, 3),
        "outage": outage.astype(int),
        "duration_min": np.round(duration_min, 1),
    })
    return df


def generate_appliances() -> list:
    appliances = [
        # --- CRITICAL ---
        {"name": "Hair Dryer",           "category": "critical", "watts_avg": 1800, "start_up_spike_w": 2200, "revenue_if_running_rwf_per_h": 6000},
        {"name": "Refrigeration Compressor", "category": "critical", "watts_avg": 1500, "start_up_spike_w": 3500, "revenue_if_running_rwf_per_h": 12000},
        {"name": "Electric Sewing Machine", "category": "critical", "watts_avg": 100,  "start_up_spike_w": 200,  "revenue_if_running_rwf_per_h": 3500},
        {"name": "POS / Cash Register",  "category": "critical", "watts_avg": 50,   "start_up_spike_w": 60,   "revenue_if_running_rwf_per_h": 1000},
        # --- COMFORT ---
        {"name": "Ceiling Fan",          "category": "comfort",  "watts_avg": 75,   "start_up_spike_w": 90,   "revenue_if_running_rwf_per_h": 500},
        {"name": "LED Lighting (main)",  "category": "comfort",  "watts_avg": 80,   "start_up_spike_w": 85,   "revenue_if_running_rwf_per_h": 800},
        {"name": "Water Pump",           "category": "comfort",  "watts_avg": 500,  "start_up_spike_w": 900,  "revenue_if_running_rwf_per_h": 700},
        # --- LUXURY ---
        {"name": "Television",           "category": "luxury",   "watts_avg": 120,  "start_up_spike_w": 130,  "revenue_if_running_rwf_per_h": 200},
        {"name": "Sound System",         "category": "luxury",   "watts_avg": 200,  "start_up_spike_w": 250,  "revenue_if_running_rwf_per_h": 300},
        {"name": "Decorative Lighting",  "category": "luxury",   "watts_avg": 60,   "start_up_spike_w": 65,   "revenue_if_running_rwf_per_h": 100},
    ]
    return appliances


def generate_businesses(appliances: list) -> list:
    by_name = {a["name"]: a for a in appliances}

    salon_appliances = [
        "Hair Dryer", "POS / Cash Register",
        "Ceiling Fan", "LED Lighting (main)",
        "Television", "Sound System", "Decorative Lighting",
    ]
    cold_room_appliances = [
        "Refrigeration Compressor", "POS / Cash Register",
        "LED Lighting (main)", "Water Pump",
        "Television", "Decorative Lighting",
    ]
    tailor_appliances = [
        "Electric Sewing Machine", "POS / Cash Register",
        "Ceiling Fan", "LED Lighting (main)",
        "Television", "Sound System",
    ]

    businesses = [
        {
            "type": "salon",
            "name": "Keza Beauty Salon",
            "location": "Kigali, Rwanda",
            "operating_hours": {"open": 7, "close": 19},
            "appliances": [by_name[n] for n in salon_appliances],
        },
        {
            "type": "cold_room",
            "name": "Amahoro Cold Storage",
            "location": "Musanze, Rwanda",
            "operating_hours": {"open": 0, "close": 24},
            "appliances": [by_name[n] for n in cold_room_appliances],
        },
        {
            "type": "tailor",
            "name": "Ineza Tailor Shop",
            "location": "Huye, Rwanda",
            "operating_hours": {"open": 8, "close": 18},
            "appliances": [by_name[n] for n in tailor_appliances],
        },
    ]
    return businesses


def main():
    print("Generating grid_history.csv (365 days hourly)...")
    df = generate_grid_history(365)
    out_path = DATA_DIR / "grid_history.csv"
    df.to_csv(out_path, index=False)
    outage_rate = df["outage"].mean()
    print(f"  Saved {len(df):,} rows → {out_path}  (outage rate: {outage_rate:.2%})")

    print("Generating appliances.json...")
    appliances = generate_appliances()
    with open(DATA_DIR / "appliances.json", "w") as f:
        json.dump(appliances, f, indent=2)
    print(f"  Saved {len(appliances)} appliances")

    print("Generating businesses.json...")
    businesses = generate_businesses(appliances)
    with open(DATA_DIR / "businesses.json", "w") as f:
        json.dump(businesses, f, indent=2)
    print(f"  Saved {len(businesses)} business archetypes: {[b['type'] for b in businesses]}")

    print("\nDone. All data files in ./data/")


if __name__ == "__main__":
    main()
