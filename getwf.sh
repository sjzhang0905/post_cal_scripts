#!/usr/bin/env bash
set -euo pipefail

# ---- checks ----
command -v vaspkit >/dev/null 2>&1 || { echo "Error: vaspkit not found in PATH" >&2; exit 1; }
[[ -f LOCPOT ]] || { echo "Error: LOCPOT not found in current directory" >&2; exit 1; }
[[ -f DOSCAR ]] || { echo "Error: DOSCAR not found in current directory" >&2; exit 1; }

log="vaspkit_426.log"

# ---- run vaspkit (42->426, direction z=3) ----
# Capture both stdout and stderr for robustness.
printf "426\n3\n" | vaspkit >"$log" 2>&1

# ---- extract the Work Function line ----
wf_line="$(grep -E "Work Function[[:space:]]*\(eV\)" "$log" | tail -n 1 || true)"

if [[ -z "$wf_line" ]]; then
  echo "Error: Work Function line not found. See $log for full output." >&2
  exit 2
fi

echo "$wf_line"

