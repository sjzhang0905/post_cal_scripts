#!/bin/bash
set -euo pipefail

# ============================================================
# Charge-density difference + selectable-direction planar average + SVG plot
#
# Usage:
#   ./getCDD_planar_plot_direction_svg_v20260920.sh        # default: z
#   ./getCDD_planar_plot_direction_svg_v20260920.sh z      # z direction
#   ./getCDD_planar_plot_direction_svg_v20260920.sh y      # y direction
#   ./getCDD_planar_plot_direction_svg_v20260920.sh x      # x direction
#
# Also accepts 1/2/3 as aliases for x/y/z.
#
# Directory layout expected:
#   ./CHGCAR          : full adsorption/interface system
#   ./ads/CHGCAR      : adsorbate (same cell/grid/geometry basis)
#   ./surf/CHGCAR     : substrate/slab (same cell/grid/geometry basis)
#
# Outputs (for DIR=x/y/z):
#   CHGDIFF.vasp
#   PLANAR_AVERAGE.dat
#   PLANAR_AVERAGE_${DIR}.svg
# ============================================================

VASPKIT_BIN="${VASPKIT_BIN:-vaspkit}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---------- planar-average direction ----------
# Default is z. Accept x/y/z (case-insensitive) or 1/2/3.
DIR_RAW="${1:-z}"
DIR_RAW="$(printf '%s' "$DIR_RAW" | tr '[:upper:]' '[:lower:]')"

case "$DIR_RAW" in
    x|1)
        DIR="x"
        VASPKIT_DIR=1
        ;;
    y|2)
        DIR="y"
        VASPKIT_DIR=2
        ;;
    z|3)
        DIR="z"
        VASPKIT_DIR=3
        ;;
    *)
        echo "ERROR: invalid planar-average direction: '${1:-}'" >&2
        echo "Usage: $0 [x|y|z]" >&2
        echo "       $0 [1|2|3]    # 1=x, 2=y, 3=z" >&2
        exit 1
        ;;
esac

# ---------- basic checks ----------
command -v "$VASPKIT_BIN" >/dev/null 2>&1 || {
    echo "ERROR: cannot find VASPKIT executable: $VASPKIT_BIN" >&2
    exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "ERROR: cannot find Python executable: $PYTHON_BIN" >&2
    exit 1
}

for f in ./CHGCAR ./ads/CHGCAR ./surf/CHGCAR; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $f" >&2
        exit 1
    fi
done

# ---------- Step 1: charge-density difference ----------
echo "[1/3] Generating CHGDIFF.vasp ..."
"$VASPKIT_BIN" <<'EOF_VASPKIT_314'
314
./CHGCAR ./ads/CHGCAR ./surf/CHGCAR
EOF_VASPKIT_314

if [[ ! -s CHGDIFF.vasp ]]; then
    echo "ERROR: VASPKIT did not generate a non-empty CHGDIFF.vasp" >&2
    exit 1
fi

# ---------- Step 2: planar average ----------
echo "[2/3] Generating ${DIR}-direction PLANAR_AVERAGE.dat ..."
"$VASPKIT_BIN" <<EOF_VASPKIT_316
316
3
CHGDIFF.vasp
${VASPKIT_DIR}
EOF_VASPKIT_316

if [[ ! -s PLANAR_AVERAGE.dat ]]; then
    echo "ERROR: VASPKIT did not generate a non-empty PLANAR_AVERAGE.dat" >&2
    exit 1
fi

# ---------- Step 3: plot ----------
echo "[3/3] Plotting PLANAR_AVERAGE.dat (${DIR} direction) to SVG ..."
PLANAR_DIR="$DIR" "$PYTHON_BIN" <<'EOF_PYTHON'
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(
        "ERROR: plotting requires numpy and matplotlib. "
        "Install them in the current Python environment first.\n"
        f"Original error: {exc}"
    )

direction = os.environ.get("PLANAR_DIR", "z").lower()
if direction not in {"x", "y", "z"}:
    sys.exit(f"ERROR: invalid PLANAR_DIR={direction!r}")

infile = Path("PLANAR_AVERAGE.dat")
rows = []
for raw in infile.read_text(errors="replace").splitlines():
    line = raw.strip()
    if not line or line.startswith(("#", "!", ";")):
        continue

    parts = line.replace("D", "E").replace("d", "e").split()
    if len(parts) < 2:
        continue

    try:
        coord = float(parts[0])
        rho = float(parts[1])
    except ValueError:
        continue

    rows.append((coord, rho))

if not rows:
    sys.exit("ERROR: no two-column numeric data found in PLANAR_AVERAGE.dat")

data = np.asarray(rows, dtype=float)
coord = data[:, 0]
rho = data[:, 1]
order = np.argsort(coord)
coord = coord[order]
rho = rho[order]

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.plot(coord, rho, linewidth=1.6)
ax.axhline(0.0, linewidth=0.9, linestyle="--")
ax.set_xlabel(rf"${direction}$ ($\mathrm{{\AA}}$)")
ax.set_ylabel(rf"Planar-averaged $\Delta\rho({direction})$")
ax.set_xlim(coord.min(), coord.max())
ax.margins(x=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out")
fig.tight_layout()
svg = f"PLANAR_AVERAGE_{direction}.svg"
fig.savefig(svg, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"Read {len(coord)} points from {infile}")
print(f"Direction: {direction}")
print(f"Written: {svg}")
EOF_PYTHON

echo
echo "Done. Planar-average direction: ${DIR}"
echo "Generated files:"
echo "  CHGDIFF.vasp"
echo "  PLANAR_AVERAGE.dat"
echo "  PLANAR_AVERAGE_${DIR}.svg"
