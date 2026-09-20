#!/usr/bin/env python3
"""
Plot a VASP/VASPKIT planar-averaged electrostatic potential and annotate the work function.

Typical usage
-------------
1) Automatic mode (reads PLANAR_AVERAGE.dat and tries to read E-fermi from OUTCAR):
   python plot_work_function.py

2) Provide E-fermi manually:
   python plot_work_function.py --efermi 3.72

3) Manually specify the vacuum plateau region (recommended when auto detection is not ideal):
   python plot_work_function.py --vac-range 25 30

4) Full example:
   python plot_work_function.py \
       --input PLANAR_AVERAGE.dat \
       --outcar OUTCAR \
       --output fig.png \
       --vac-range 24 30 \
       --dpi 300
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot work-function style figure from VASPKIT PLANAR_AVERAGE.dat"
    )
    parser.add_argument(
        "--input",
        default="PLANAR_AVERAGE.dat",
        help="Input data file from VASPKIT (default: PLANAR_AVERAGE.dat)",
    )
    parser.add_argument(
        "--outcar",
        default="OUTCAR",
        help="OUTCAR used to extract E-fermi automatically (default: OUTCAR)",
    )
    parser.add_argument(
        "--efermi",
        type=float,
        default=None,
        help="Fermi energy in eV. If not given, the script tries to read it from OUTCAR.",
    )
    parser.add_argument(
        "--vac-range",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Manual vacuum plateau range in angstrom, e.g. --vac-range 25 30",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Moving-average window for vacuum detection / optional plotting (default: 21)",
    )
    parser.add_argument(
        "--plot-smoothed",
        action="store_true",
        help="Plot the smoothed curve instead of the raw curve",
    )
    parser.add_argument(
        "--output",
        default="wf.svg",
        help="Output figure file (default: wf.svg)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits, e.g. --ylim -15 8",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=13,
        help="Base font size (default: 13)",
    )
    return parser.parse_args()


def read_planar_average(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find input file: {path}")

    xs: List[float] = []
    ys: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError(f"No valid numeric data found in {path}")

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    return x_arr, y_arr


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    if window >= len(y):
        window = max(3, len(y) // 2 * 2 - 1)
    if window <= 1:
        return y.copy()

    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_pad, kernel, mode="valid")


def read_efermi_from_outcar(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find OUTCAR: {path}. Please pass --efermi manually or provide the correct OUTCAR path."
        )

    pattern = re.compile(r"E-fermi\s*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)")
    last_match: Optional[float] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last_match = float(m.group(1))

    if last_match is None:
        raise ValueError(
            f"Failed to find 'E-fermi' in {path}. Please pass --efermi manually."
        )

    return last_match


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def detect_vacuum_level(
    x: np.ndarray,
    y_smooth: np.ndarray,
    vac_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, Tuple[int, int], str]:
    if vac_range is not None:
        xmin, xmax = vac_range
        if xmin >= xmax:
            raise ValueError("For --vac-range, XMIN must be smaller than XMAX.")
        mask = (x >= xmin) & (x <= xmax)
        if not np.any(mask):
            raise ValueError("The chosen --vac-range does not overlap the data range.")
        idx = np.where(mask)[0]
        evac = float(np.mean(y_smooth[idx]))
        return evac, (int(idx[0]), int(idx[-1])), "manual range"

    # Auto detection: look for a flat, high-potential region.
    grad = np.abs(np.gradient(y_smooth, x))
    high_mask = y_smooth >= np.percentile(y_smooth, 85)
    flat_mask = grad <= np.percentile(grad, 35)
    candidate = high_mask & flat_mask

    regions = contiguous_regions(candidate)
    if regions:
        # Prefer long regions; if tied, prefer the rightmost one.
        regions_sorted = sorted(regions, key=lambda r: ((r[1] - r[0] + 1), r[1]))
        i0, i1 = regions_sorted[-1]
        evac = float(np.mean(y_smooth[i0 : i1 + 1]))
        return evac, (i0, i1), ""

    # Fallback: use the highest 5% points.
    threshold = np.percentile(y_smooth, 95)
    mask = y_smooth >= threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        imax = int(np.argmax(y_smooth))
        return float(y_smooth[imax]), (imax, imax), "auto max"

    evac = float(np.mean(y_smooth[idx]))
    return evac, (int(idx[0]), int(idx[-1])), "auto top 5%"


def make_plot(
    x: np.ndarray,
    y: np.ndarray,
    y_smooth: np.ndarray,
    evac: float,
    efermi: Optional[float],
    vac_idx_range: Tuple[int, int],
    vac_method: str,
    output: Path,
    dpi: int,
    plot_smoothed: bool,
    title: Optional[str],
    ylim: Optional[Tuple[float, float]],
    fontsize: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    y_plot = y_smooth if plot_smoothed else y
    ax.plot(x, y_plot, lw=1.8, label="Planar average potential")

    # Vacuum level line
    ax.axhline(evac, ls="--", lw=1.1)

    xspan = x.max() - x.min()
    yspan = max(y_plot.max(), evac) - min(y_plot.min(), evac)
    yspan = yspan if yspan > 1e-8 else 1.0
    x_text = x.min() + 0.68 * xspan

    ax.text(
        x_text,
        evac + 0.03 * yspan,
        f"E_vac = {evac:.2f} eV",
        va="bottom",
        ha="left",
    )

    i0, i1 = vac_idx_range
    xmid = 0.5 * (x[i0] + x[i1])
    ax.text(
        xmid,
        evac - 0.06 * yspan,
        f"",
        va="top",
        ha="center",
        fontsize=fontsize * 0.85,
    )

    if efermi is not None:
        phi = evac - efermi
        ax.axhline(efermi, ls=":", lw=1.1)
        ax.text(
            x_text,
            efermi - 0.03 * yspan,
            f"E_F = {efermi:.2f} eV",
            va="top",
            ha="left",
        )

        arrow_x = x.max() - 0.08 * xspan
        ax.annotate(
            "",
            xy=(arrow_x, evac),
            xytext=(arrow_x, efermi),
            arrowprops=dict(arrowstyle="<->", lw=1.2),
        )
        ax.text(
            arrow_x + 0.02 * xspan,
            0.5 * (evac + efermi),
            f"$\\Phi$ = {phi:.2f} eV",
            va="center",
            ha="left",
        )

    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Electrostatic potential (eV)")

    if title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin = min(y_plot.min(), efermi if efermi is not None else y_plot.min(), evac)
        ymax = max(y_plot.max(), efermi if efermi is not None else y_plot.max(), evac)
        margin = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xlim(x.min(), x.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    outcar_path = Path(args.outcar)
    output_path = Path(args.output)

    x, y = read_planar_average(input_path)
    y_smooth = moving_average(y, args.smooth_window)

    efermi = args.efermi
    if efermi is None:
        try:
            efermi = read_efermi_from_outcar(outcar_path)
        except Exception as exc:
            print(f"[Warning] {exc}", file=sys.stderr)
            print("[Warning] The figure will be plotted without E_F / work-function annotation.", file=sys.stderr)
            efermi = None

    vac_range = tuple(args.vac_range) if args.vac_range is not None else None
    evac, vac_idx_range, vac_method = detect_vacuum_level(x, y_smooth, vac_range)

    make_plot(
        x=x,
        y=y,
        y_smooth=y_smooth,
        evac=evac,
        efermi=efermi,
        vac_idx_range=vac_idx_range,
        vac_method=vac_method,
        output=output_path,
        dpi=args.dpi,
        plot_smoothed=args.plot_smoothed,
        title=args.title,
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        fontsize=args.fontsize,
    )

    phi_msg = ""
    if efermi is not None:
        phi_msg = f", Phi = {evac - efermi:.2f} eV"
    print(f"Saved figure to: {output_path} | E_vac = {evac:.2f} eV{phi_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
