#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pymatgen.io.lobster import Cohpcar
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# ================== Parameters: edit as needed ==================

# Gaussian smoothing width, in number of data points
sigma = 0.5

# Display range for E - Ef, in eV
emin, emax = -10, 10

# Figure size: height is about twice the width
fig_width = 5
fig_height = 10

# Font sizes
label_fs = 18   # axis labels
tick_fs  = 14   # ticks
legend_fs = 14  # legend

# Colors
color_cohp = "tab:red"    # bottom x-axis and -COHP
color_icohp = "tab:blue"  # top x-axis and -ICOHP

# Output directory
output_dir = Path("cohp_plots")

# ================== Function: Gaussian smoothing ==================
def gaussian_smooth(y, sigma=2):
    """Apply Gaussian smoothing to a 1D array y. sigma is in data points."""
    if sigma <= 0:
        return y
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def safe_filename(label):
    """Convert a COHP label into a safe filename."""
    text = str(label)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text if text else "bond"


# ================== Read COHP data ==================
cohp = Cohpcar(filename="COHPCAR.lobster")
energies = cohp.energies

print("All labels in cohp_data:")
for k in cohp.cohp_data.keys():
    print("  ", k)

# Exclude "average"; plot only real atom pairs
bonds_all = [b for b in cohp.cohp_data.keys() if str(b).lower() != "average"]

print("\nBonds used for plotting:")
for b in bonds_all:
    print("  ", b)

if not bonds_all:
    raise RuntimeError("No non-average bonds found. Please check COHPCAR.lobster / lobsterin.")

# Automatically determine all available spin channels.
# For spin-polarized calculations, UP and DOWN are summed before plotting.
# For non-spin-polarized calculations, the single available channel is used directly.
first_bond = bonds_all[0]
spin_keys = list(cohp.cohp_data[first_bond]["COHP"].keys())

if not spin_keys:
    raise RuntimeError("No spin channels were found in COHPCAR.lobster.")

print("\nAvailable spin channels:")
for key in spin_keys:
    print("  ", key)

if len(spin_keys) == 1:
    print("Using the single available spin channel.")
else:
    print(f"Summing {len(spin_keys)} spin channels for COHP and ICOHP.")

# Create output directory
output_dir.mkdir(exist_ok=True)

# ================== Plot one figure for each bond ==================
for bond in bonds_all:
    fig, ax_cohp = plt.subplots(figsize=(fig_width, fig_height))

    # --- Bottom x-axis: -COHP ---
    raw_cohp = -np.sum(
        [cohp.cohp_data[bond]["COHP"][key] for key in spin_keys],
        axis=0
    )
    sm_cohp = gaussian_smooth(raw_cohp, sigma=sigma)

    ax_cohp.plot(
        sm_cohp, energies,
        label=f"{bond}  -COHP (spin-summed)",
        color=color_cohp,
        linestyle="-"
    )

    # Grey dashed lines: E - Ef = 0 and -COHP = 0
    ax_cohp.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax_cohp.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)

    # y-axis range: E - Ef
    ax_cohp.set_ylim(emin, emax)

    ax_cohp.set_ylabel("E - E$_F$ (eV)", fontsize=label_fs)
    ax_cohp.set_xlabel("-COHP", fontsize=label_fs, color=color_cohp)

    ax_cohp.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax_cohp.tick_params(axis="x", colors=color_cohp)
    ax_cohp.spines["bottom"].set_color(color_cohp)

    # --- Top x-axis: -ICOHP ---
    ax_icohp = ax_cohp.twiny()

    raw_icohp = -np.sum(
        [cohp.cohp_data[bond]["ICOHP"][key] for key in spin_keys],
        axis=0
    )
    sm_icohp = gaussian_smooth(raw_icohp, sigma=sigma)

    ax_icohp.plot(
        sm_icohp, energies,
        label=f"{bond}  -ICOHP (spin-summed)",
        color=color_icohp,
        linestyle="-"
    )

    ax_icohp.set_xlabel("-ICOHP", fontsize=label_fs, color=color_icohp)
    ax_icohp.tick_params(axis="x", which="major", labelsize=tick_fs, colors=color_icohp)
    ax_icohp.spines["top"].set_color(color_icohp)

    # --- Merge legends ---
    handles1, labels1 = ax_cohp.get_legend_handles_labels()
    handles2, labels2 = ax_icohp.get_legend_handles_labels()

    # ax_cohp.legend(
    #     handles1 + handles2,
    #     labels1 + labels2,
    #     loc="best",
    #     fontsize=legend_fs
    # )

    plt.tight_layout()

    outfile = output_dir / f"cohp_icohp_smooth_{safe_filename(bond)}.svg"
    plt.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"Saved: {outfile}")
