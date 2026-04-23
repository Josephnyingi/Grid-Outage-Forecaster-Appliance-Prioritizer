"""
run_all.py — One-shot runner: generate data, train, forecast, evaluate, plan.
Reproducible in ≤ 2 commands on a free Colab CPU.
"""

import subprocess, sys, json
from pathlib import Path

def run(cmd):
    print(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    py = sys.executable

    run([py, "generate_data.py"])
    run([py, "forecaster.py", "--train"])
    run([py, "forecaster.py", "--forecast", "--eval"])
    run([py, "prioritizer.py", "--forecast", "outputs/forecast_2024-06-29.csv", "--business", "all"])

    # Print summary
    with open("outputs/eval_metrics.json") as f:
        m = json.load(f)

    print("\n" + "="*60)
    print("  ALL DONE — Summary")
    print("="*60)
    print(f"  Brier Score     : {m['brier_score']}")
    print(f"  Duration MAE    : {m['duration_mae_min']} min")
    print(f"  Avg Lead Time   : {m['avg_lead_time_hours']} h")
    print(f"\nOpen lite_ui.html in your browser to view the dashboard.")
