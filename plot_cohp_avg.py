#!/usr/bin/env python3

from pymatgen.io.lobster import Cohpcar
import matplotlib.pyplot as plt
import numpy as np

# ================== User settings ==================

# Gaussian smoothing width in data-point units
sigma = 0.5

# Display range for E - Ef in eV
emin, emax = -10, 6

# Figure size
fig_width = 5
fig_height = 10

# Font sizes
label_fs = 18
tick_fs = 14
legend_fs = 14

# Colors
color_cohp = "tab:red"
color_icohp = "tab:blue"

# Output file name
outfile = "cohp_icohp_avg_smooth.svg"

# ================== Gaussian smoothing ==================

def gaussian_smooth(y, sigma=0.5):
    """Apply one-dimensional Gaussian smoothing.

    Parameters
    ----------
    y : array-like
        Input data.
    sigma : float
        Gaussian width in data-point units.
    """
    y = np.asarray(y, dtype=float)

    if sigma <= 0:
        return y.copy()

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    return np.convolve(y, kernel, mode="same")


def sum_spin_channels(spin_data, quantity_name):
    """Return the total quantity summed over all available spin channels.

    For a non-spin-polarized calculation, only one channel is present and is
    returned unchanged. For a spin-polarized calculation, all available spin
    channels are summed.
    """
    if not spin_data:
        raise RuntimeError(f"No spin channels found for {quantity_name}.")

    spin_keys = list(spin_data.keys())

    print(f"{quantity_name} spin channels:")
    for key in spin_keys:
        print(f"  {key}")

    arrays = [np.asarray(spin_data[key], dtype=float) for key in spin_keys]

    reference_shape = arrays[0].shape
    for key, arr in zip(spin_keys[1:], arrays[1:]):
        if arr.shape != reference_shape:
            raise RuntimeError(
                f"Inconsistent array shape for {quantity_name}: "
                f"{spin_keys[0]} has {reference_shape}, "
                f"but {key} has {arr.shape}."
            )

    if len(arrays) == 1:
        print(f"{quantity_name}: single spin channel detected.")
        return arrays[0].copy()

    print(
        f"{quantity_name}: {len(arrays)} spin channels detected; "
        "summing all channels."
    )
    return np.sum(np.stack(arrays, axis=0), axis=0)


# ================== Read COHP data ==================

cohp = Cohpcar(filename="COHPCAR.lobster")
energies = np.asarray(cohp.energies, dtype=float)

print("Labels in cohp_data:")
for key in cohp.cohp_data.keys():
    print(f"  {key}")

if "average" not in cohp.cohp_data:
    raise RuntimeError(
        "The 'average' entry is missing from cohp_data. "
        "Check COHPCAR.lobster and lobsterin."
    )

avg_data = cohp.cohp_data["average"]

if "COHP" not in avg_data or "ICOHP" not in avg_data:
    raise RuntimeError(
        "The 'average' entry does not contain both COHP and ICOHP data."
    )

cohp_avg_total = sum_spin_channels(avg_data["COHP"], "COHP")
icohp_avg_total = sum_spin_channels(avg_data["ICOHP"], "ICOHP")

if cohp_avg_total.shape != energies.shape:
    raise RuntimeError(
        f"COHP length mismatch: energies has shape {energies.shape}, "
        f"COHP has shape {cohp_avg_total.shape}."
    )

if icohp_avg_total.shape != energies.shape:
    raise RuntimeError(
        f"ICOHP length mismatch: energies has shape {energies.shape}, "
        f"ICOHP has shape {icohp_avg_total.shape}."
    )

# Convert to the commonly plotted -COHP and -ICOHP convention
minus_cohp = -cohp_avg_total
minus_icohp = -icohp_avg_total

# Apply smoothing
minus_cohp_smooth = gaussian_smooth(minus_cohp, sigma=sigma)
minus_icohp_smooth = gaussian_smooth(minus_icohp, sigma=sigma)

# ================== Plot ==================

fig, ax_cohp = plt.subplots(figsize=(fig_width, fig_height))

ax_cohp.plot(
    minus_cohp_smooth,
    energies,
    label="-COHP (average, spin-summed)",
    color=color_cohp,
    linestyle="-",
    linewidth=1.5,
)

ax_cohp.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
ax_cohp.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)

ax_cohp.set_ylim(emin, emax)

ax_cohp.set_ylabel("E - E$_F$ (eV)", fontsize=label_fs)
ax_cohp.set_xlabel(
    "-COHP (average, spin-summed)",
    fontsize=label_fs,
    color=color_cohp,
)

ax_cohp.tick_params(axis="both", which="major", labelsize=tick_fs)
ax_cohp.tick_params(axis="x", colors=color_cohp)
ax_cohp.spines["bottom"].set_color(color_cohp)

ax_icohp = ax_cohp.twiny()

ax_icohp.plot(
    minus_icohp_smooth,
    energies,
    label="-ICOHP (average, spin-summed)",
    color=color_icohp,
    linestyle="--",
    linewidth=1.5,
)

ax_icohp.set_xlabel(
    "-ICOHP (average, spin-summed)",
    fontsize=label_fs,
    color=color_icohp,
)
ax_icohp.tick_params(
    axis="x",
    which="major",
    labelsize=tick_fs,
    colors=color_icohp,
)
ax_icohp.spines["top"].set_color(color_icohp)

handles1, labels1 = ax_cohp.get_legend_handles_labels()
handles2, labels2 = ax_icohp.get_legend_handles_labels()
ax_cohp.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="best",
    fontsize=legend_fs,
)

plt.tight_layout()
plt.savefig(outfile, dpi=300)
# plt.show()

print(f"Saved figure to: {outfile}")
