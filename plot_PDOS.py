#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------- User settings ----------------
TDOS_FILE = "TDOS.dat"
PDOS_NONSPIN_PATTERN = "PDOS_*.dat"
PDOS_SPIN_UP_PATTERN = "PDOS_*_UP.dat"
PDOS_SPIN_DW_PATTERN = "PDOS_*_DW.dat"

GAUSS_SIGMA_POINTS = 1.5
LINEWIDTH = 1.2
MIRROR_SPIN = True
UP_LINESTYLE = "-"
DW_LINESTYLE = "-"

XRANGE = (-8, 4)
YMAX_FIXED = 20
# ------------------------------------------------


# Periodic-table block classification used for orbital selection.
# H is handled explicitly as s.
S_BLOCK_ELEMENTS = {"H"}


D_BLOCK_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
}

F_BLOCK_ELEMENTS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}

P_ORBITALS = ("py", "pz", "px")
D_ORBITALS = ("dxy", "dyz", "dz2", "dxz", "dx2-y2")

_NUMERIC_LINE_RE = re.compile(
    r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
)


def gaussian_smooth_points(y, sigma_pts):
    """Apply Gaussian smoothing with sigma expressed in grid points."""
    if sigma_pts is None or sigma_pts <= 0:
        return y

    half_w = int(np.ceil(6 * sigma_pts))
    xx = np.arange(-half_w, half_w + 1, dtype=float)
    kernel = np.exp(-0.5 * (xx / sigma_pts) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def load_tdos_dat_maybe_spin(fname, skiprows=1):
    """
    Read TDOS data.

    Two columns are interpreted as:
        Energy, Total

    Three or more columns are interpreted as:
        Energy, Up, Down
    """
    data = np.loadtxt(fname, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncol = data.shape[1]
    x = data[:, 0]

    if ncol >= 3:
        return "spin", x, {"up": data[:, 1], "dw": data[:, 2]}

    if ncol >= 2:
        return "nonspin", x, {"total": data[:, 1]}

    raise ValueError(f"{fname} does not contain enough columns.")


def load_xy_two_cols(fname, x_col=0, y_col=1, skiprows=1):
    data = np.loadtxt(fname, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, x_col]
    y = data[:, y_col]
    return x, y


def load_tdos_any():
    """
    Load TDOS from TDOS_FILE first, then fall back to TDOS_UP.dat and
    TDOS_DW.dat.

    Returns:
        mode, x, data

    mode is one of:
        "nonspin", "spin", "none"
    """
    if os.path.exists(TDOS_FILE):
        try:
            return load_tdos_dat_maybe_spin(TDOS_FILE, skiprows=1)
        except Exception:
            pass

    up = "TDOS_UP.dat"
    dw = "TDOS_DW.dat"

    if os.path.exists(up) and os.path.exists(dw):
        x_up, y_up = load_xy_two_cols(up, x_col=0, y_col=1, skiprows=1)
        x_dw, y_dw = load_xy_two_cols(dw, x_col=0, y_col=1, skiprows=1)

        if len(x_up) != len(x_dw) or np.max(np.abs(x_up - x_dw)) > 1e-12:
            y_dw = np.interp(x_up, x_dw, y_dw)

        return "spin", x_up, {"up": y_up, "dw": y_dw}

    if os.path.exists(TDOS_FILE):
        x, y = load_xy_two_cols(TDOS_FILE, x_col=0, y_col=1, skiprows=1)
        return "nonspin", x, {"total": y}

    return "none", None, {}


def extract_elem_nonspin(fname):
    match = re.match(r".*PDOS_([A-Za-z][a-z]?)\.dat$", fname)
    return match.group(1) if match else None


def extract_elem_spin(fname):
    match = re.match(r".*PDOS_([A-Za-z][a-z]?)_(UP|DW)\.dat$", fname)
    if match:
        return match.group(1), match.group(2)

    return None, None


def ensure_on_grid(x_ref, x_src, y_src):
    """Interpolate y_src onto x_ref when the energy grids differ."""
    if len(x_ref) != len(x_src) or np.max(np.abs(x_ref - x_src)) > 1e-12:
        return np.interp(x_ref, x_src, y_src)

    return y_src


def standardize_elem(elem):
    if not elem:
        return None

    elem = elem.strip()

    if len(elem) == 1:
        return elem.upper()

    return elem[0].upper() + elem[1:].lower()


def read_pdos_header_and_skip(fname):
    """
    Return:
        columns, skiprows

    The header is taken from a line containing "energy", case-insensitively.
    The first numeric line determines where np.loadtxt should start.
    """
    columns = None
    first_numeric_idx = None

    with open(fname, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        if columns is None and "energy" in stripped.lower():
            tokens = stripped.replace("#", "").split()
            columns = [token.strip() for token in tokens]

        if first_numeric_idx is None and _NUMERIC_LINE_RE.match(stripped):
            first_numeric_idx = i
            break

    if first_numeric_idx is None:
        raise ValueError(f"{fname}: no numeric data line was found.")

    if columns is None:
        ncol = len(lines[first_numeric_idx].split())
        columns = ["Energy"] + [f"col{i}" for i in range(1, ncol)]

    return columns, first_numeric_idx


def pick_orbital_indices(elem, columns):
    """
    Select orbital columns from the periodic-table block of the element.

    Rules:
        H                             -> s
        d-block elements            -> d
        lanthanides/actinides       -> f
        all remaining elements      -> p
    """
    elem = standardize_elem(elem)

    name_to_idx = {}
    for i, name in enumerate(columns):
        key = name.strip().replace("#", "").lower()
        name_to_idx[key] = i

    s_indices = [
        i for name, i in name_to_idx.items()
        if name == "s" or name.startswith("s_") or name.startswith("s-")
    ]

    p_indices = [name_to_idx[name] for name in P_ORBITALS if name in name_to_idx]
    if not p_indices:
        p_indices = [
            i for name, i in name_to_idx.items()
            if name.startswith("p") and name not in {"pdos"}
        ]

    d_indices = [name_to_idx[name] for name in D_ORBITALS if name in name_to_idx]
    if not d_indices:
        d_indices = [
            i for name, i in name_to_idx.items()
            if name.startswith("d")
        ]

    f_indices = [
        i for name, i in name_to_idx.items()
        if name.startswith("f")
    ]

    if elem in S_BLOCK_ELEMENTS:
        return sorted(set(s_indices)) if s_indices else None

    if elem in F_BLOCK_ELEMENTS:
        return sorted(set(f_indices)) if f_indices else None

    if elem in D_BLOCK_ELEMENTS:
        return sorted(set(d_indices)) if d_indices else None

    return sorted(set(p_indices)) if p_indices else None


def load_pdos_sum_by_rule(fname, elem):
    """Read a PDOS file and sum the orbitals selected for the element."""
    elem = standardize_elem(elem)
    columns, skiprows = read_pdos_header_and_skip(fname)

    data = np.loadtxt(fname, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0]
    indices = pick_orbital_indices(elem, columns)

    if not indices:
        y = data[:, -1]
        print(
            f"[WARN] {fname}: no matching orbital columns for {elem}; "
            "using the last column instead."
        )
        return x, y

    indices = [
        i for i in indices
        if i != 0 and i < data.shape[1]
    ]

    if not indices:
        y = data[:, -1]
        print(
            f"[WARN] {fname}: selected orbital indices are invalid for {elem}; "
            "using the last column instead."
        )
        return x, y

    y = np.sum(data[:, indices], axis=1)
    return x, y


def collect_pdos(x_grid):
    """
    Collect all PDOS data.

    Returned structure:
        {
            "spin": bool,
            "elems": {
                "Fe": {"up": array, "dw": array},
                "O": {"total": array},
            },
        }
    """
    result = {"spin": False, "elems": {}}

    up_files = sorted(glob.glob(PDOS_SPIN_UP_PATTERN))
    dw_files = sorted(glob.glob(PDOS_SPIN_DW_PATTERN))

    up_map = {}
    for fname in up_files:
        elem, tag = extract_elem_spin(fname)
        if elem and tag == "UP":
            up_map[standardize_elem(elem)] = fname

    dw_map = {}
    for fname in dw_files:
        elem, tag = extract_elem_spin(fname)
        if elem and tag == "DW":
            dw_map[standardize_elem(elem)] = fname

    spin_elems = sorted(set(up_map) & set(dw_map))

    if spin_elems:
        result["spin"] = True

    for elem in spin_elems:
        x_up, y_up = load_pdos_sum_by_rule(up_map[elem], elem)
        x_dw, y_dw = load_pdos_sum_by_rule(dw_map[elem], elem)

        y_up = ensure_on_grid(x_grid, x_up, y_up)
        y_dw = ensure_on_grid(x_grid, x_dw, y_dw)

        y_up = gaussian_smooth_points(y_up, GAUSS_SIGMA_POINTS)
        y_dw = gaussian_smooth_points(y_dw, GAUSS_SIGMA_POINTS)

        result["elems"][elem] = {"up": y_up, "dw": y_dw}

    for fname in sorted(glob.glob(PDOS_NONSPIN_PATTERN)):
        if re.search(r"_(UP|DW)\.dat$", fname):
            continue

        elem = standardize_elem(extract_elem_nonspin(fname))

        if not elem or elem in result["elems"]:
            continue

        x_p, y_p = load_pdos_sum_by_rule(fname, elem)
        y_p = ensure_on_grid(x_grid, x_p, y_p)
        y_p = gaussian_smooth_points(y_p, GAUSS_SIGMA_POINTS)

        result["elems"][elem] = {"total": y_p}

    return result


def assign_colors(elems_order):
    """Assign one color per element, shared by its spin channels."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {}

    for i, elem in enumerate(elems_order):
        colors[elem] = color_cycle[i % len(color_cycle)] if color_cycle else None

    return colors


def mirror_spin_down_if_needed(y):
    """
    Keep spin-down DOS on the negative side when MIRROR_SPIN is enabled.

    If the input is already mostly negative, it is left unchanged.
    """
    if not MIRROR_SPIN:
        return y

    try:
        median = float(np.nanmedian(y))
    except Exception:
        return y

    return -y if median > 0 else y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TDOS and element-resolved PDOS."
    )
    parser.add_argument(
        "ymax",
        nargs="?",
        type=float,
        help=(
            "Optional positive y-axis maximum. "
            "Overrides YMAX_FIXED from the script."
        ),
    )
    return parser.parse_args()


def main():
    global YMAX_FIXED

    args = parse_args()

    if args.ymax is not None:
        if args.ymax <= 0:
            raise ValueError("ymax must be a positive number.")
        YMAX_FIXED = args.ymax

    tdos_mode, x_t, tdos_data = load_tdos_any()

    if tdos_mode == "none":
        candidates = (
            sorted(glob.glob(PDOS_SPIN_UP_PATTERN))
            + sorted(glob.glob(PDOS_SPIN_DW_PATTERN))
            + sorted(glob.glob(PDOS_NONSPIN_PATTERN))
        )

        if not candidates:
            raise FileNotFoundError("No TDOS or PDOS data files were found.")

        match = re.search(
            r"PDOS_([A-Za-z][a-z]?)",
            os.path.basename(candidates[0]),
        )
        elem0 = standardize_elem(match.group(1)) if match else None
        x_t, _ = load_pdos_sum_by_rule(candidates[0], elem0)

    else:
        if tdos_mode == "nonspin":
            tdos_data["total"] = gaussian_smooth_points(
                tdos_data["total"],
                GAUSS_SIGMA_POINTS,
            )
        else:
            tdos_data["up"] = gaussian_smooth_points(
                tdos_data["up"],
                GAUSS_SIGMA_POINTS,
            )
            tdos_data["dw"] = gaussian_smooth_points(
                tdos_data["dw"],
                GAUSS_SIGMA_POINTS,
            )

    pdos = collect_pdos(x_t)
    spin_present = (tdos_mode == "spin") or pdos["spin"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=800)

    elems_order = sorted(pdos["elems"].keys())
    elem_colors = assign_colors(elems_order)

    if tdos_mode == "nonspin" and "total" in tdos_data:
        ax.plot(
            x_t,
            tdos_data["total"],
            color="black",
            lw=LINEWIDTH,
            label="Total",
        )

    elif tdos_mode == "spin":
        y_up = tdos_data["up"]
        y_dw = mirror_spin_down_if_needed(tdos_data["dw"])

        ax.plot(
            x_t,
            y_up,
            color="black",
            lw=LINEWIDTH,
            linestyle=UP_LINESTYLE,
            label="Total",
        )
        ax.plot(
            x_t,
            y_dw,
            color="black",
            lw=LINEWIDTH,
            linestyle=DW_LINESTYLE,
            label="_nolegend_",
        )

    for elem in elems_order:
        color = elem_colors.get(elem)
        data = pdos["elems"][elem]

        if "total" in data:
            ax.plot(
                x_t,
                data["total"],
                lw=LINEWIDTH,
                label=elem,
                color=color,
            )
        else:
            y_up = data["up"]
            y_dw = mirror_spin_down_if_needed(data["dw"])

            ax.plot(
                x_t,
                y_up,
                lw=LINEWIDTH,
                label=elem,
                color=color,
                linestyle=UP_LINESTYLE,
            )
            ax.plot(
                x_t,
                y_dw,
                lw=LINEWIDTH,
                label="_nolegend_",
                color=color,
                linestyle=DW_LINESTYLE,
            )

    ax.set_xlim(*XRANGE)

    if YMAX_FIXED is not None:
        if spin_present and MIRROR_SPIN:
            ax.set_ylim(-YMAX_FIXED, YMAX_FIXED)
        else:
            ax.set_ylim(0, YMAX_FIXED)

    else:
        ys = [line.get_ydata() for line in ax.get_lines()]

        if not ys:
            ax.set_ylim(0.0, 1.0)

        else:
            y_max = max(float(np.nanmax(y)) for y in ys)
            y_min = min(float(np.nanmin(y)) for y in ys)

            if spin_present and MIRROR_SPIN:
                bound = max(abs(y_max), abs(y_min))
                bound = 1.05 * bound if bound > 0 else 1.0
                ax.set_ylim(-bound, bound)

            else:
                lower = 1.05 * y_min if y_min < 0 else 0.0
                upper = 1.05 * y_max if y_max > 0 else 1.0
                ax.set_ylim(lower, upper)

    plt.xticks(fontsize=22, fontweight="bold")
    plt.yticks(fontsize=22, fontweight="bold")
    plt.xlabel(r"${E}$-$E_{f}$ (eV)", fontsize=28, fontweight="bold")
    plt.ylabel("DOS (states / eV)", fontsize=28, fontweight="bold")

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    plt.legend(
        prop={"size": 24, "weight": "bold"},
        labelcolor="linecolor",
        frameon=False,
        labelspacing=0.2,
        borderaxespad=0.2,
        handlelength=2.0,
        handletextpad=0.4,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig("DOS.svg")


if __name__ == "__main__":
    main()
