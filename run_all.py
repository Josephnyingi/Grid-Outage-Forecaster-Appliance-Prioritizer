"""
run_all.py — One-shot runner: generate data, train, forecast, evaluate, plan.
Reproducible in ≤ 2 commands on a free Colab CPU.
"""

import subprocess, sys, json, glob
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

    # Discover the forecast file dynamically — robust across any run date
    forecast_files = sorted(glob.glob("outputs/forecast_*.csv"))
    if not forecast_files:
        print("ERROR: no forecast file found in outputs/")
        sys.exit(1)
    forecast_path = forecast_files[-1]
    print(f"\nUsing forecast: {forecast_path}")

    run([py, "prioritizer.py", "--forecast", forecast_path, "--business", "all"])

    # Execute eval notebook (nbconvert if available, else skip gracefully)
    try:
        run([py, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", "eval.ipynb"])
        print("eval.ipynb executed successfully.")
    except SystemExit:
        print("Note: jupyter not found — open eval.ipynb manually to run evaluation plots.")

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
