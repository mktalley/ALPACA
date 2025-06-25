#!/usr/bin/env bash
# Kick off 12-month 0-DTE grid back-test (~6k variants)
# Adds cache downloads as needed.  Rerunnable / idempotent.

set -euo pipefail

START=$(date -d "12 months ago" +%Y-%m-%d)
END=$(date +%Y-%m-%d)
OUT=results_phase2.csv
GRID_FILE=apps/zero_dte/phase2.yaml

python - << 'PY'
import yaml, json, os, subprocess, pathlib, sys
from datetime import date, timedelta
from apps.zero_dte.dl_missing import ensure_data

# 1. Build day list (weekdays only)
start = date.fromisoformat(os.environ.get("START_DATE", "${START}"))
end   = date.fromisoformat(os.environ.get("END_DATE", "${END}"))
days=[start+timedelta(d) for d in range((end-start).days+1) if (start+timedelta(d)).weekday()<5]
print(f"Need data for {len(days)} trading days …")

ensure_data("SPY", days)
print("Data cache ready ✔")

# 2. Load grid YAML and run back-test
p = pathlib.Path("apps/zero_dte/phase2.yaml")
params = yaml.safe_load(p.read_text())

grid_str = json.dumps(params)
cmd = [sys.executable, "-m", "apps.zero_dte.backtest_cli", "--symbol", "SPY",
       "--start", start.isoformat(), "--end", end.isoformat(),
       "--grid", grid_str, "--outfile", "${OUT}"]
print("Executing", " ".join(cmd))
subprocess.run(cmd, check=True)
PY
