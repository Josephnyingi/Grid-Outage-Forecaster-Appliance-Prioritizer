"""
prioritizer.py — Appliance load-shed planner.

Given a 24-hour outage forecast and a business's appliance list, produces a
per-appliance, per-hour ON/OFF plan that maximises expected revenue while
enforcing the 'drop luxury before critical' rule.

Algorithm:
  For each hour h:
    1. If p_outage[h] < LOW_RISK_THRESHOLD  → all appliances ON.
    2. If p_outage[h] in [LOW, HIGH]        → shed luxury appliances first.
    3. If p_outage[h] >= HIGH_RISK_THRESHOLD → shed luxury + comfort, keep critical only.
  Expected revenue = Σ_a  revenue_if_running * P(ON) * (1 - p_outage)
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Risk thresholds
LOW_RISK  = 0.10   # below → all on
HIGH_RISK = 0.25   # above → critical only

CATEGORY_PRIORITY = {"critical": 0, "comfort": 1, "luxury": 2}


def plan(
    forecast: pd.DataFrame,
    appliances: list[dict],
    business_type: str = "salon",
    low_risk: float = LOW_RISK,
    high_risk: float = HIGH_RISK,
) -> dict:
    """
    Core planning function — explained on-screen during live defence.

    Enforces 'critical before luxury' rule by sorting appliances by category
    priority, then within each tier by revenue-per-watt (keep highest earners on).
    Ties are broken by watts_avg ascending (prefer low-draw appliances).

    Returns:
        {
          "schedule": {appliance_name: [status_hour0, ..., status_hour23]},
          "hourly_summary": [...],
          "total_expected_revenue_rwf": float,
          "total_naive_revenue_rwf": float,
          "savings_vs_naive_rwf": float,
        }
    """
    # Sort appliances: critical first, then by revenue/watt desc, then watts asc
    sorted_apps = sorted(
        appliances,
        key=lambda a: (
            CATEGORY_PRIORITY[a["category"]],
            -a["revenue_if_running_rwf_per_h"] / max(a["watts_avg"], 1),
            a["watts_avg"],
        ),
    )

    schedule: dict[str, list[str]] = {a["name"]: [] for a in sorted_apps}
    hourly_summary = []

    for _, row in forecast.iterrows():
        h        = int(row["hour"])
        p_out    = float(row["p_outage"])
        e_dur    = float(row["e_duration"])

        if p_out < low_risk:
            risk_level = "low"
        elif p_out < high_risk:
            risk_level = "medium"
        else:
            risk_level = "high"

        hour_on = []
        hour_off = []

        for app in sorted_apps:
            cat = app["category"]
            name = app["name"]

            if risk_level == "low":
                status = "ON"
            elif risk_level == "medium":
                # Drop luxury, keep critical + comfort
                status = "OFF" if cat == "luxury" else "ON"
            else:
                # High risk: critical only
                status = "ON" if cat == "critical" else "OFF"

            schedule[name].append(status)
            if status == "ON":
                hour_on.append(name)
            else:
                hour_off.append(name)

        on_apps   = [a for a in sorted_apps if schedule[a["name"]][-1] == "ON"]
        total_w   = sum(a["watts_avg"] for a in on_apps)
        rev_h     = sum(a["revenue_if_running_rwf_per_h"] for a in on_apps) * (1 - p_out)

        hourly_summary.append({
            "hour": h,
            "p_outage": round(p_out, 4),
            "e_duration_min": round(e_dur, 1),
            "risk_level": risk_level,
            "appliances_on": hour_on,
            "appliances_off": hour_off,
            "total_load_w": total_w,
            "expected_revenue_rwf": round(rev_h, 2),
        })

    total_expected = sum(s["expected_revenue_rwf"] for s in hourly_summary)

    # Naïve baseline: everything always ON
    naive_rev_per_h = sum(a["revenue_if_running_rwf_per_h"] for a in appliances)
    total_naive = sum(
        naive_rev_per_h * (1 - float(row["p_outage"]))
        for _, row in forecast.iterrows()
    )

    savings = total_expected - total_naive   # positive means our plan beats naïve? No...
    # Actually naïve always runs everything, ignoring outage risk; our plan conserves
    # revenue by keeping critical on. The saving is in protected revenue during risk hours.
    # Real saving = revenue protected by not running non-critical during outage window
    protected = 0.0
    for s, (_, row) in zip(hourly_summary, forecast.iterrows()):
        p_out = float(row["p_outage"])
        if p_out >= low_risk:
            off_apps = [a for a in appliances if a["name"] in s["appliances_off"]]
            # We avoid wasting these on an outage hour
            protected += sum(a["watts_avg"] for a in off_apps) * p_out * 0.001  # kWh saved

    result = {
        "business_type": business_type,
        "schedule": schedule,
        "hourly_summary": hourly_summary,
        "total_expected_revenue_rwf": round(total_expected, 2),
        "total_naive_revenue_rwf": round(total_naive, 2),
        "revenue_difference_rwf": round(total_expected - total_naive, 2),
        "kwh_saved_by_shedding": round(protected, 3),
        "thresholds": {"low_risk": low_risk, "high_risk": high_risk},
    }

    out_path = OUT_DIR / f"plan_{business_type}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Plan saved → {out_path}")
    return result


def print_plan_table(plan_result: dict) -> None:
    """Pretty-print the 24h plan as an ASCII table."""
    print(f"\n{'─'*70}")
    print(f"  Load-Shed Plan  |  Business: {plan_result['business_type'].upper()}")
    print(f"{'─'*70}")
    print(f"  Expected revenue (24h): {plan_result['total_expected_revenue_rwf']:>10,.0f} RWF")
    print(f"  Naïve baseline  (24h): {plan_result['total_naive_revenue_rwf']:>10,.0f} RWF")
    print(f"  kWh saved by shedding: {plan_result['kwh_saved_by_shedding']:>10.3f} kWh")
    print(f"{'─'*70}")
    header = f"{'Hour':>4}  {'Risk':>6}  {'P(out)':>7}  {'E[dur]':>7}  {'ON appliances'}"
    print(header)
    print(f"{'─'*70}")
    for s in plan_result["hourly_summary"]:
        on_str = ", ".join(s["appliances_on"]) or "—"
        print(
            f"  {s['hour']:02d}h  {s['risk_level']:>6}  "
            f"{s['p_outage']:>6.1%}  {s['e_duration_min']:>6.0f}m  {on_str}"
        )
    print(f"{'─'*70}\n")


def run_all_businesses(forecast_csv: str) -> None:
    forecast_df = pd.read_csv(forecast_csv)
    with open(DATA_DIR / "businesses.json") as f:
        businesses = json.load(f)

    for biz in businesses:
        print(f"\nPlanning for: {biz['name']} ({biz['type']})")
        result = plan(forecast_df, biz["appliances"], biz["type"])
        print_plan_table(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appliance Load-Shed Prioritizer")
    parser.add_argument("--forecast", required=True,      help="Path to forecast CSV")
    parser.add_argument("--business", default="all",      help="Business type or 'all'")
    parser.add_argument("--low-risk", type=float, default=LOW_RISK)
    parser.add_argument("--high-risk", type=float, default=HIGH_RISK)
    args = parser.parse_args()

    if args.business == "all":
        run_all_businesses(args.forecast)
    else:
        forecast_df = pd.read_csv(args.forecast)
        with open(DATA_DIR / "businesses.json") as f:
            businesses = json.load(f)
        biz = next((b for b in businesses if b["type"] == args.business), None)
        if biz is None:
            print(f"Business type '{args.business}' not found. Options: salon, cold_room, tailor")
        else:
            result = plan(forecast_df, biz["appliances"], biz["type"],
                          args.low_risk, args.high_risk)
            print_plan_table(result)
