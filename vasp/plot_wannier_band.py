#!/usr/bin/env python3
"""
Compare a VASP Line-mode band structure with Wannier90 interpolated bands.

Default interactive use:
    python plot_wannier_band_v20260920_strict.py

At startup, the script asks for the ordinary VASP band directory.
Defaults:
  Wannier directory : current directory
  Fermi-level OUTCAR: ../OUTCAR relative to the Wannier directory

Outputs written in the current directory:
  1) <prefix>_vasp.svg
  2) <prefix>_wannier.svg
  3) <prefix>_compare.svg
  4) <prefix>_diagnostics.txt

The VASP band directory must contain:
  POSCAR, KPOINTS, EIGENVAL

The current Wannier directory must contain:
  <seed>_band.dat
and, for strict path validation:
  <seed>_band.gnu

Non-interactive use:
    python plot_wannier_band_v20260920_strict.py \
        --vasp-dir ../band \
        --emin -6 --emax 5

Dependencies:
  numpy, matplotlib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_poscar_lattice(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Invalid POSCAR: {path}")

    scale_tokens = [float(x) for x in lines[1].split()]
    raw = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )

    if len(scale_tokens) == 1:
        scale = scale_tokens[0]
        if scale > 0:
            lattice = raw * scale
        elif scale < 0:
            target_volume = abs(scale)
            raw_volume = abs(np.linalg.det(raw))
            if raw_volume <= 0:
                raise ValueError("POSCAR lattice has zero volume")
            lattice = raw * (target_volume / raw_volume) ** (1.0 / 3.0)
        else:
            raise ValueError("POSCAR scale factor cannot be zero")
    elif len(scale_tokens) == 3:
        scales = np.array(scale_tokens, dtype=float)
        if np.any(scales <= 0.0):
            raise ValueError("Three POSCAR scale factors must all be positive")
        lattice = raw * scales[None, :]
    else:
        raise ValueError("Unsupported POSCAR scaling line")

    return lattice


def reciprocal_lattice(lattice: np.ndarray) -> np.ndarray:
    # Direct vectors and reciprocal vectors are stored row-wise.
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def read_kpoints_line_mode(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Invalid KPOINTS: {path}")

    try:
        points_per_segment = int(lines[1].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("KPOINTS line 2 must contain points per segment") from exc

    if "line" not in lines[2].lower():
        raise ValueError("KPOINTS is not in line-mode")

    coordinate_mode = lines[3].strip().lower()
    if coordinate_mode.startswith("c"):
        raise ValueError(
            "Cartesian line-mode KPOINTS is not supported. Use Fractional, "
            "Reciprocal, or Direct coordinates."
        )
    if not coordinate_mode.startswith(("r", "d", "f")):
        raise ValueError(
            "Unsupported KPOINTS coordinate mode. Expected Fractional, "
            "Reciprocal, or Direct coordinates."
        )

    endpoints = []
    for line in lines[4:]:
        clean = line.strip()
        if not clean:
            continue
        left, *comment = re.split(r"[!#]", clean, maxsplit=1)
        fields = left.split()
        if len(fields) < 3:
            continue
        try:
            coord = np.array([float(x) for x in fields[:3]], dtype=float)
        except ValueError:
            continue
        label = comment[0].strip() if comment else ""
        if not label and len(fields) >= 4:
            label = fields[3]
        endpoints.append((coord, label))

    if len(endpoints) < 2 or len(endpoints) % 2 != 0:
        raise ValueError(
            "Could not parse KPOINTS segment endpoints; expected endpoint pairs"
        )

    segments = [
        (endpoints[i], endpoints[i + 1])
        for i in range(0, len(endpoints), 2)
    ]
    return points_per_segment, segments


def read_eigenval(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Invalid EIGENVAL: {path}")

    try:
        header = lines[5].split()
        if len(header) < 3:
            raise ValueError
        nkpts = int(header[1])
        nbands = int(header[2])
    except (ValueError, IndexError) as exc:
        raise ValueError("Could not parse NKPTS/NBANDS from EIGENVAL") from exc

    kpoints = []
    blocks = []
    cursor = 6

    for _ in range(nkpts):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ValueError("Unexpected end of EIGENVAL before k-point block")

        kp_fields = lines[cursor].split()
        if len(kp_fields) < 4:
            raise ValueError(f"Malformed EIGENVAL k-point line: {lines[cursor]}")
        kpoints.append([float(x) for x in kp_fields[:3]])
        cursor += 1

        band_rows = []
        for _band in range(nbands):
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor >= len(lines):
                raise ValueError(
                    "Unexpected end of EIGENVAL inside a band block"
                )
            fields = lines[cursor].split()
            if len(fields) < 3:
                raise ValueError(f"Malformed EIGENVAL band line: {lines[cursor]}")
            band_rows.append([float(x) for x in fields[1:]])
            cursor += 1
        blocks.append(band_rows)

    array = np.array(blocks, dtype=float)
    # ISPIN=1: band, energy, occupation -> two values after band index.
    # ISPIN=2: band, E_up, E_down, occ_up, occ_down -> four values.
    if array.shape[2] >= 4:
        energies = np.stack([array[:, :, 0], array[:, :, 1]], axis=0)
    else:
        energies = array[:, :, 0][None, :, :]

    return np.array(kpoints, dtype=float), energies


def merge_tick(ticks: list[tuple[float, str]], x: float, label: str) -> None:
    label = label.strip()
    if ticks and abs(ticks[-1][0] - x) < 1.0e-10:
        old = ticks[-1][1]
        if label and not old:
            ticks[-1] = (x, label)
        elif label and label != old and label not in old.split("|"):
            ticks[-1] = (x, f"{old}|{label}")
    else:
        ticks.append((x, label))


def normalize_k_label(label: str) -> tuple[str, ...]:
    cleaned = label.strip().upper()
    cleaned = cleaned.replace("Γ", "GAMMA")
    cleaned = cleaned.replace("\\GAMMA", "GAMMA")
    cleaned = cleaned.replace("$", "").replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("/", "|").replace(",", "|")
    parts = []
    for part in cleaned.split("|"):
        if part in {"G", "GAMMA"}:
            part = "GAMMA"
        if part:
            parts.append(part)
    return tuple(sorted(set(parts)))


def validate_kpath_ticks(
    source_ticks: list[tuple[float, str]],
    target_ticks: list[tuple[float, str]],
    allow_global_rescale: bool,
) -> None:
    if len(source_ticks) != len(target_ticks) or len(source_ticks) < 2:
        if allow_global_rescale:
            return
        raise ValueError(
            "Wannier and VASP k paths do not have the same number of path "
            "boundaries. Ensure <seed>_band.gnu exists and both calculations "
            "use the same k path. Use --allow-global-rescale only if a global "
            "x-axis rescaling is intentionally desired."
        )

    mismatches = []
    for index, ((_, source_label), (_, target_label)) in enumerate(
        zip(source_ticks, target_ticks)
    ):
        if not source_label.strip() or not target_label.strip():
            continue
        source_norm = normalize_k_label(source_label)
        target_norm = normalize_k_label(target_label)
        if source_norm != target_norm:
            mismatches.append(
                f"boundary {index}: Wannier={source_label!r}, VASP={target_label!r}"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Wannier and VASP k-path labels are inconsistent: " + details
        )


def build_vasp_path(
    kpoints_frac: np.ndarray,
    reciprocal: np.ndarray,
    points_per_segment: int,
    segments,
):
    nsegments = len(segments)
    expected = points_per_segment * nsegments
    if len(kpoints_frac) != expected:
        raise ValueError(
            f"EIGENVAL contains {len(kpoints_frac)} k points, but line-mode "
            f"KPOINTS implies {expected} ({points_per_segment} points per "
            f"segment x {nsegments} segments). Refusing to guess a path."
        )

    x = np.zeros(len(kpoints_frac), dtype=float)
    ticks: list[tuple[float, str]] = []
    base = 0.0

    for iseg, ((_, start_label), (_, end_label)) in enumerate(segments):
        lo = iseg * points_per_segment
        hi = lo + points_per_segment
        kcart = kpoints_frac[lo:hi] @ reciprocal
        local = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(kcart, axis=0), axis=1))]
        )
        x[lo:hi] = base + local
        merge_tick(ticks, base, start_label)
        base += local[-1]
        merge_tick(ticks, base, end_label)

    return x, ticks


def read_wannier_band_dat(path: Path):
    bands = []
    current = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = line.strip()
        if not clean:
            if current:
                bands.append(np.array(current, dtype=float))
                current = []
            continue
        if clean.startswith(("#", "!")):
            continue
        fields = clean.split()
        if len(fields) < 2:
            continue
        try:
            current.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue

    if current:
        bands.append(np.array(current, dtype=float))

    if not bands:
        raise ValueError(f"No band data found in {path}")
    return bands


def read_wannier_ticks_from_gnu(path: Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"set\s+xtics\s*\((.*?)\)", text, flags=re.I | re.S)
    if not match:
        return []

    ticks = []
    pattern = re.compile(
        r"""["']([^"']+)["']\s+([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)"""
    )
    for label, value in pattern.findall(match.group(1)):
        merge_tick(ticks, float(value), label.strip())
    return ticks


def piecewise_map(
    values: np.ndarray,
    source_ticks: list[tuple[float, str]],
    target_ticks: list[tuple[float, str]],
    allow_global_rescale: bool = False,
) -> np.ndarray:
    if len(source_ticks) != len(target_ticks) or len(source_ticks) < 2:
        if not allow_global_rescale:
            raise ValueError(
                "Cannot map Wannier x coordinates onto the VASP path because "
                "the path-boundary counts differ."
            )
        source_min = float(np.min(values))
        source_max = float(np.max(values))
        target_min = target_ticks[0][0] if target_ticks else 0.0
        target_max = target_ticks[-1][0] if target_ticks else source_max
        if abs(source_max - source_min) < 1.0e-14:
            return values.copy()
        return target_min + (values - source_min) * (
            (target_max - target_min) / (source_max - source_min)
        )

    sx = np.array([x for x, _ in source_ticks], dtype=float)
    tx = np.array([x for x, _ in target_ticks], dtype=float)
    mapped = np.empty_like(values, dtype=float)

    for i, value in enumerate(values):
        if value <= sx[0]:
            seg = 0
        elif value >= sx[-1]:
            seg = len(sx) - 2
        else:
            seg = int(np.searchsorted(sx, value, side="right") - 1)
            seg = min(max(seg, 0), len(sx) - 2)

        ds = sx[seg + 1] - sx[seg]
        if abs(ds) < 1.0e-14:
            mapped[i] = tx[seg]
        else:
            fraction = (value - sx[seg]) / ds
            mapped[i] = tx[seg] + fraction * (tx[seg + 1] - tx[seg])

    return mapped



def split_curve_at_tick_boundaries(
    curve: np.ndarray,
    ticks: list[tuple[float, str]],
) -> list[np.ndarray]:
    """Split one Wannier band at every k-path boundary.

    Wannier90 may repeat a high-symmetry point at the end of one segment and
    the start of the next. If independently sorted eigenvalues exchange band
    indices there, directly connecting the repeated points creates a false
    spike. The left segment therefore ends at the first occurrence of a
    boundary, while the right segment starts at the last occurrence.
    """
    if len(curve) < 2:
        return [curve]

    x = curve[:, 0]
    scale = max(float(np.max(x) - np.min(x)), 1.0)
    tolerance = max(1.0e-10, scale * 1.0e-8)

    if len(ticks) < 2:
        # Fallback: split wherever x does not increase.
        segments: list[np.ndarray] = []
        start = 0
        for index, delta in enumerate(np.diff(x)):
            if delta <= tolerance:
                part = curve[start : index + 1]
                if len(part) >= 2:
                    segments.append(part)
                start = index + 1
        part = curve[start:]
        if len(part) >= 2:
            segments.append(part)
        return segments or [curve]

    segments: list[np.ndarray] = []
    start = 0

    for boundary_x, _label in ticks[1:-1]:
        indices = np.where(np.abs(x - boundary_x) <= tolerance)[0]
        indices = indices[indices >= start]
        if len(indices) == 0:
            # Use the closest remaining point when the printed precision of
            # band.dat and band.gnu differs slightly.
            remaining = np.arange(start, len(x))
            if len(remaining) == 0:
                break
            closest = remaining[np.argmin(np.abs(x[remaining] - boundary_x))]
            first = last = int(closest)
        else:
            first = int(indices[0])
            last = int(indices[-1])

        left = curve[start : first + 1]
        if len(left) >= 2:
            segments.append(left)

        # Reuse the high-symmetry point in the next segment. When the point is
        # repeated, start from its last occurrence to avoid connecting two
        # different band-order assignments at identical x.
        start = last

    final = curve[start:]
    if len(final) >= 2:
        segments.append(final)

    return segments or [curve]


def wannier_boundary_diagnostics(
    bands: list[np.ndarray],
    ticks: list[tuple[float, str]],
) -> list[tuple[str, float, int]]:
    """Return maximum energy mismatch between repeated boundary points."""
    if len(ticks) < 3:
        return []

    x_all = np.concatenate([band[:, 0] for band in bands if len(band)])
    scale = max(float(np.max(x_all) - np.min(x_all)), 1.0)
    tolerance = max(1.0e-10, scale * 1.0e-8)
    diagnostics: list[tuple[str, float, int]] = []

    for boundary_x, label in ticks[1:-1]:
        max_jump = 0.0
        repeated_bands = 0
        for band in bands:
            indices = np.where(np.abs(band[:, 0] - boundary_x) <= tolerance)[0]
            if len(indices) >= 2:
                repeated_bands += 1
                jump = abs(
                    float(band[indices[-1], 1]) - float(band[indices[0], 1])
                )
                max_jump = max(max_jump, jump)
        diagnostics.append((label, max_jump, repeated_bands))

    return diagnostics


def vasp_segment_slices(
    points_per_segment: int,
    nsegments: int,
) -> list[slice]:
    return [
        slice(index * points_per_segment, (index + 1) * points_per_segment)
        for index in range(nsegments)
    ]


def minimum_cost_energy_pairs(
    vasp_values: np.ndarray,
    wannier_values: np.ndarray,
) -> list[tuple[float, float]]:
    vasp_sorted = np.sort(np.asarray(vasp_values, dtype=float))
    wannier_sorted = np.sort(np.asarray(wannier_values, dtype=float))
    if len(vasp_sorted) == 0 or len(wannier_sorted) == 0:
        return []

    if len(vasp_sorted) <= len(wannier_sorted):
        small = vasp_sorted
        large = wannier_sorted
        small_is_vasp = True
    else:
        small = wannier_sorted
        large = vasp_sorted
        small_is_vasp = False

    m = len(small)
    n = len(large)
    inf = float("inf")
    dp = np.full((m + 1, n + 1), inf, dtype=float)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    dp[0, :] = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            skip_cost = dp[i, j - 1]
            match_cost = dp[i - 1, j - 1] + abs(small[i - 1] - large[j - 1])
            if match_cost <= skip_cost:
                dp[i, j] = match_cost
                take[i, j] = True
            else:
                dp[i, j] = skip_cost

    selected: list[tuple[float, float]] = []
    i = m
    j = n
    while i > 0 and j > 0:
        if take[i, j]:
            a = float(small[i - 1])
            b = float(large[j - 1])
            if small_is_vasp:
                selected.append((a, b))
            else:
                selected.append((b, a))
            i -= 1
            j -= 1
        else:
            j -= 1

    selected.reverse()
    if len(selected) != m:
        raise RuntimeError("Internal energy-matching failure")
    return selected


def compute_spectral_metrics(
    x_vasp: np.ndarray,
    vasp_energy: np.ndarray,
    vasp_slices: list[slice],
    segmented_w90_bands: list[list[np.ndarray]],
    emin: float,
    emax: float,
):
    nsegments = len(vasp_slices)
    if any(len(parts) != nsegments for parts in segmented_w90_bands):
        return None

    deltas = []
    unmatched_vasp = 0
    unmatched_wannier = 0
    sampled_kpoints = 0
    eligible_kpoints = 0
    fully_matched_kpoints = 0
    total_vasp_states = 0
    total_wannier_states = 0
    max_record = None

    for iseg, segment_slice in enumerate(vasp_slices):
        xv = np.asarray(x_vasp[segment_slice], dtype=float)
        if len(xv) == 0:
            continue

        interpolated = []
        for band_parts in segmented_w90_bands:
            part = np.asarray(band_parts[iseg], dtype=float)
            if len(part) < 2:
                return None
            order = np.argsort(part[:, 0], kind="stable")
            xp = part[order, 0]
            yp = part[order, 1]
            xp_unique, unique_indices = np.unique(xp, return_index=True)
            yp_unique = yp[unique_indices]
            if len(xp_unique) < 2:
                return None
            interpolated.append(np.interp(xv, xp_unique, yp_unique))

        w_interp = np.asarray(interpolated, dtype=float)
        v_segment = np.asarray(vasp_energy[segment_slice, :], dtype=float)

        for ik_local, x_value in enumerate(xv):
            v_values = v_segment[ik_local, :]
            w_values = w_interp[:, ik_local]
            v_window = v_values[(v_values >= emin) & (v_values <= emax)]
            w_window = w_values[(w_values >= emin) & (w_values <= emax)]
            pairs = minimum_cost_energy_pairs(v_window, w_window)
            total_vasp_states += len(v_window)
            total_wannier_states += len(w_window)
            unmatched_vasp += len(v_window) - len(pairs)
            unmatched_wannier += len(w_window) - len(pairs)
            if len(v_window) or len(w_window):
                eligible_kpoints += 1
                if len(v_window) == len(w_window) == len(pairs):
                    fully_matched_kpoints += 1
            if not pairs:
                continue
            sampled_kpoints += 1

            for v_value, w_value in pairs:
                delta = w_value - v_value
                deltas.append(delta)
                abs_delta = abs(delta)
                if max_record is None or abs_delta > max_record[0]:
                    max_record = (
                        abs_delta,
                        float(x_value),
                        float(v_value),
                        float(w_value),
                        iseg + 1,
                    )

    delta_array = np.asarray(deltas, dtype=float)
    return {
        "matched_pairs": int(len(delta_array)),
        "total_kpoints": int(sum(len(x_vasp[sl]) for sl in vasp_slices)),
        "eligible_kpoints": eligible_kpoints,
        "fully_matched_kpoints": fully_matched_kpoints,
        "total_vasp_states": total_vasp_states,
        "total_wannier_states": total_wannier_states,
        "vasp_coverage": len(deltas)/total_vasp_states if total_vasp_states else float("nan"),
        "wannier_coverage": len(deltas)/total_wannier_states if total_wannier_states else float("nan"),
        "sampled_kpoints": int(sampled_kpoints),
        "unmatched_vasp": int(unmatched_vasp),
        "unmatched_wannier": int(unmatched_wannier),
        "mae": float(np.mean(np.abs(delta_array))) if len(deltas) else float("nan"),
        "rms": float(np.sqrt(np.mean(delta_array**2))) if len(deltas) else float("nan"),
        "max_abs": float(np.max(np.abs(delta_array))) if len(deltas) else float("nan"),
        "mean_signed": float(np.mean(delta_array)) if len(deltas) else float("nan"),
        "max_record": max_record,
    }


def read_fermi_from_outcar(path: Path) -> float:
    pattern = re.compile(r"E-fermi\s*:\s*([+\-0-9.Ee]+)")
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        raise ValueError(f"No E-fermi found in {path}")
    return values[-1]


def prompt_vasp_directory() -> Path:
    """Interactively request the ordinary VASP band directory."""
    while True:
        raw = input(
            "Enter the VASP band directory containing POSCAR, KPOINTS, and EIGENVAL:\n> "
        ).strip()
        if not raw:
            print("The directory cannot be empty. Please try again.")
            continue

        candidate = Path(raw).expanduser().resolve()
        if not candidate.is_dir():
            print(f"Directory does not exist: {candidate}")
            continue

        missing = [
            name
            for name in ("POSCAR", "KPOINTS", "EIGENVAL")
            if not (candidate / name).is_file()
        ]
        if missing:
            print(
                "The directory is missing required files: "
                + ", ".join(missing)
                + ". Please try again."
            )
            continue
        return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vasp-dir",
        default=None,
        help=(
            "Ordinary VASP band directory. When omitted, the script asks "
            "interactively."
        ),
    )
    parser.add_argument(
        "--wannier-dir",
        default=".",
        help="Wannier band directory (default: current directory)",
    )
    parser.add_argument("--seed", default="wannier90")
    parser.add_argument(
        "--spin",
        type=int,
        default=1,
        choices=(1, 2),
        help="VASP spin channel to plot when EIGENVAL is spin-polarized",
    )
    parser.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Common Fermi energy in eV; preferred for a strict comparison",
    )
    parser.add_argument(
        "--fermi-outcar",
        default=None,
        help=(
            "OUTCAR containing the common Fermi level. When omitted, use "
            "../OUTCAR relative to --wannier-dir."
        ),
    )
    parser.add_argument("--emin", type=float, default=-6.0)
    parser.add_argument("--emax", type=float, default=6.0)
    parser.add_argument(
        "--metric-emin",
        type=float,
        default=None,
        help="Lower energy bound for quantitative comparison; default: --emin.",
    )
    parser.add_argument(
        "--metric-emax",
        type=float,
        default=None,
        help="Upper energy bound for quantitative comparison; default: --emax.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable quantitative VASP-Wannier spectral agreement metrics.",
    )
    parser.add_argument(
        "--wannier-spin",
        type=int,
        default=None,
        choices=(1, 2),
        help=(
            "Declare which collinear spin channel the Wannier band file "
            "represents. Required when EIGENVAL contains two spin channels."
        ),
    )
    parser.add_argument(
        "--allow-global-rescale",
        action="store_true",
        help=(
            "Allow a global min-max rescaling of the Wannier x axis when "
            "path-boundary counts are unavailable or inconsistent. This is "
            "less strict and should be used only intentionally."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="band_compare_v20260920_strict",
        help="Prefix for all output SVG and diagnostic files.",
    )
    args = parser.parse_args()

    vasp_dir = (
        Path(args.vasp_dir).expanduser().resolve()
        if args.vasp_dir
        else prompt_vasp_directory()
    )
    wannier_dir = Path(args.wannier_dir).expanduser().resolve()

    lattice = read_poscar_lattice(vasp_dir / "POSCAR")
    reciprocal = reciprocal_lattice(lattice)
    points_per_segment, segments = read_kpoints_line_mode(vasp_dir / "KPOINTS")
    kpoints, energies = read_eigenval(vasp_dir / "EIGENVAL")

    if args.spin > energies.shape[0]:
        raise ValueError(
            f"EIGENVAL contains {energies.shape[0]} spin channel(s), "
            f"but --spin {args.spin} was requested"
        )

    if energies.shape[0] == 2:
        if args.wannier_spin is None:
            raise ValueError(
                "Spin-polarized EIGENVAL detected. Specify --wannier-spin 1 "
                "or --wannier-spin 2 to declare which spin channel the "
                "Wannier band file represents."
            )
        if args.wannier_spin != args.spin:
            raise ValueError(
                f"Spin mismatch: --spin {args.spin} selects the VASP channel, "
                f"but --wannier-spin {args.wannier_spin} declares a different "
                "Wannier channel."
            )
    elif args.wannier_spin not in (None, 1):
        raise ValueError(
            "A non-spin-polarized EIGENVAL cannot be compared with "
            "--wannier-spin 2."
        )

    x_vasp, vasp_ticks = build_vasp_path(
        kpoints, reciprocal, points_per_segment, segments
    )

    w90_dat = wannier_dir / f"{args.seed}_band.dat"
    w90_gnu = wannier_dir / f"{args.seed}_band.gnu"
    w90_bands = read_wannier_band_dat(w90_dat)
    w90_ticks = read_wannier_ticks_from_gnu(w90_gnu)

    if args.fermi is not None:
        fermi = args.fermi
        fermi_source = "manual --fermi value"
    else:
        if args.fermi_outcar is None:
            fermi_outcar = (wannier_dir / "../OUTCAR").resolve()
        else:
            fermi_outcar = Path(args.fermi_outcar).expanduser().resolve()
        if not fermi_outcar.is_file():
            raise FileNotFoundError(
                "Fermi-level OUTCAR was not found: "
                f"{fermi_outcar}. The default is ../OUTCAR relative to "
                "--wannier-dir. Use --fermi or --fermi-outcar to override it."
            )
        fermi = read_fermi_from_outcar(fermi_outcar)
        fermi_source = str(fermi_outcar)

    # Common tick/axis helpers
    if vasp_ticks:
        tick_positions = [x for x, _ in vasp_ticks]
        tick_labels = [label for _, label in vasp_ticks]
    else:
        tick_positions = []
        tick_labels = []

    x_vasp_min = float(np.min(x_vasp))
    x_vasp_max = float(np.max(x_vasp))

    validate_kpath_ticks(
        w90_ticks,
        vasp_ticks,
        allow_global_rescale=args.allow_global_rescale,
    )
    if args.allow_global_rescale and (
        len(w90_ticks) != len(vasp_ticks) or len(w90_ticks) < 2
    ):
        print(
            "WARNING: using global Wannier x-axis rescaling because strict "
            "path-boundary mapping is unavailable."
        )

    # Map Wannier band-path x coordinates onto the VASP path.
    mapped_w90_bands = []
    for band in w90_bands:
        x_w = piecewise_map(
            band[:, 0],
            w90_ticks,
            vasp_ticks,
            allow_global_rescale=args.allow_global_rescale,
        )
        mapped_w90_bands.append(np.column_stack((x_w, band[:, 1] - fermi)))

    vasp_energy = energies[args.spin - 1] - fermi
    vasp_slices = vasp_segment_slices(points_per_segment, len(segments))
    # mapped_w90_bands are already expressed in the VASP path-length
    # coordinate system. Therefore they must be split using vasp_ticks.
    # Using the original w90_ticks here would leave path boundaries connected
    # and create false spikes at repeated high-symmetry points.
    segmented_w90_bands = [
        split_curve_at_tick_boundaries(band, vasp_ticks)
        for band in mapped_w90_bands
    ]
    boundary_diagnostics = wannier_boundary_diagnostics(
        w90_bands,
        w90_ticks,
    )

    # Verify that every mapped VASP boundary is represented in every Wannier
    # curve to within numerical tolerance. Missing boundaries are reported,
    # because otherwise a plotting script could silently connect two segments.
    mapped_boundary_warnings: list[str] = []
    if len(vasp_ticks) >= 3:
        all_mapped_x = np.concatenate(
            [band[:, 0] for band in mapped_w90_bands if len(band)]
        )
        mapped_scale = max(
            float(np.max(all_mapped_x) - np.min(all_mapped_x)),
            1.0,
        )
        mapped_tolerance = max(1.0e-8, mapped_scale * 1.0e-7)

        for boundary_x, boundary_label in vasp_ticks[1:-1]:
            per_band_nearest = [
                float(np.min(np.abs(band[:, 0] - boundary_x)))
                for band in mapped_w90_bands
                if len(band)
            ]
            worst_nearest = max(per_band_nearest)
            if worst_nearest > mapped_tolerance:
                mapped_boundary_warnings.append(
                    f"{boundary_label}: worst mapped Wannier boundary mismatch "
                    f"is {worst_nearest:.6e}, tolerance {mapped_tolerance:.6e}"
                )

    metric_emin = args.emin if args.metric_emin is None else args.metric_emin
    metric_emax = args.emax if args.metric_emax is None else args.metric_emax
    if metric_emin >= metric_emax:
        raise ValueError("The metric energy window requires metric-emin < metric-emax")

    global_rescale_active = args.allow_global_rescale and (
        len(w90_ticks) != len(vasp_ticks) or len(w90_ticks) < 2
    )
    spectral_metrics = None
    metrics_status = "disabled by --no-metrics"
    if not args.no_metrics:
        if global_rescale_active:
            metrics_status = "not computed because global x-axis rescaling is active"
        else:
            spectral_metrics = compute_spectral_metrics(
                x_vasp,
                vasp_energy,
                vasp_slices,
                segmented_w90_bands,
                metric_emin,
                metric_emax,
            )
            metrics_status = (
                "computed"
                if spectral_metrics is not None
                else "not computed because segment-wise interpolation was unavailable"
            )

    def decorate_axis(ax):
        if tick_positions:
            ax.set_xticks(tick_positions, tick_labels)
            for x in tick_positions:
                ax.axvline(x, linewidth=0.5, alpha=0.35)
        ax.axhline(0.0, linewidth=0.7, alpha=0.55)
        ax.set_xlim(x_vasp_min, x_vasp_max)
        ax.set_ylim(args.emin, args.emax)
        ax.set_xlabel("k path")
        ax.set_ylabel(r"$E-E_F$ (eV)")

    output_vasp = Path.cwd() / f"{args.output_prefix}_vasp.svg"
    output_wannier = Path.cwd() / f"{args.output_prefix}_wannier.svg"
    output_combined = Path.cwd() / f"{args.output_prefix}_compare.svg"

    # 1) VASP-only figure
    fig_v, ax_v = plt.subplots(figsize=(7.2, 5.4))
    label_used = False
    for segment_slice in vasp_slices:
        for iband in range(vasp_energy.shape[1]):
            ax_v.plot(
                x_vasp[segment_slice],
                vasp_energy[segment_slice, iband],
                color="black",
                linewidth=0.8,
                alpha=0.8,
                label="VASP" if not label_used else None,
            )
            label_used = True
    decorate_axis(ax_v)
    ax_v.set_title("VASP bands")
    ax_v.legend()
    fig_v.tight_layout()
    fig_v.savefig(output_vasp, format="svg", bbox_inches="tight")
    plt.close(fig_v)

    # 2) Wannier-only figure
    fig_w, ax_w = plt.subplots(figsize=(7.2, 5.4))
    label_used = False
    for band_segments in segmented_w90_bands:
        for segment in band_segments:
            ax_w.plot(
                segment[:, 0],
                segment[:, 1],
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
                label="Wannier90" if not label_used else None,
            )
            label_used = True
    decorate_axis(ax_w)
    ax_w.set_title("Wannier interpolated bands")
    ax_w.legend()
    fig_w.tight_layout()
    fig_w.savefig(output_wannier, format="svg", bbox_inches="tight")
    plt.close(fig_w)

    # 3) Combined figure
    fig_c, ax_c = plt.subplots(figsize=(7.2, 5.4))
    label_used = False
    for segment_slice in vasp_slices:
        for iband in range(vasp_energy.shape[1]):
            ax_c.plot(
                x_vasp[segment_slice],
                vasp_energy[segment_slice, iband],
                color="black",
                linewidth=0.7,
                alpha=0.35,
                label="VASP" if not label_used else None,
                zorder=1,
            )
            label_used = True

    label_used = False
    for band_segments in segmented_w90_bands:
        for segment in band_segments:
            ax_c.plot(
                segment[:, 0],
                segment[:, 1],
                color="tab:red",
                linestyle="--",
                linewidth=1.5,
                alpha=1.0,
                label="Wannier90" if not label_used else None,
                zorder=3,
            )
            label_used = True
    decorate_axis(ax_c)
    ax_c.set_title("VASP vs Wannier bands")
    ax_c.legend()
    fig_c.tight_layout()
    fig_c.savefig(output_combined, format="svg", bbox_inches="tight")
    plt.close(fig_c)

    print(f"VASP band directory : {vasp_dir}")
    print(f"Wannier directory   : {wannier_dir}")
    print(f"Fermi energy used   : {fermi:.8f} eV")
    print(f"Fermi source        : {fermi_source}")
    print(f"Wrote SVG           : {output_vasp}")
    print(f"Wrote SVG           : {output_wannier}")
    print(f"Wrote SVG           : {output_combined}")
    print(f"Metrics status      : {metrics_status}")
    if spectral_metrics is not None:
        print(f"Matched-state coverage: VASP={spectral_metrics['vasp_coverage']:.2%}, "
              f"Wannier={spectral_metrics['wannier_coverage']:.2%}")
        if spectral_metrics['unmatched_vasp'] or spectral_metrics['unmatched_wannier']:
            print("WARNING: incomplete spectral coverage; matched-pair errors alone cannot assess agreement.")
        if not spectral_metrics['matched_pairs']:
            print("WARNING: no matched states; error metrics are undefined (nan).")
        print(
            "Spectral metrics    : "
            f"MAE={spectral_metrics['mae']:.6f} eV, "
            f"RMS={spectral_metrics['rms']:.6f} eV, "
            f"MAX={spectral_metrics['max_abs']:.6f} eV"
        )

    report_path = Path.cwd() / f"{args.output_prefix}_diagnostics.txt"
    report_lines = [
        "Wannier repeated-boundary diagnostics",
        "",
        "For each high-symmetry boundary, the table reports the largest",
        "energy difference between repeated points at the same band-path x.",
        "A large value indicates a plotting/band-order connection artifact",
        "when adjacent path segments are drawn as one continuous curve.",
        "",
        "label  repeated_bands  max_energy_jump_eV",
    ]
    for label, max_jump, repeated_bands in boundary_diagnostics:
        report_lines.append(
            f"{label:12s} {repeated_bands:14d} {max_jump:19.10f}"
        )

    report_lines.extend(
        [
            "",
            "Quantitative spectral agreement:",
            f"Status: {metrics_status}",
            f"Energy window: [{metric_emin:.6f}, {metric_emax:.6f}] eV relative to E_F",
            (
                "Method: Wannier energies are interpolated segment-wise onto VASP "
                "k points; energies inside the window are matched one-to-one by "
                "minimum absolute energy cost. This is a spectral agreement metric, "
                "not a band-character identity metric."
            ),
        ]
    )
    if spectral_metrics is not None:
        report_lines.extend(
            [
                "Errors describe matched states only; missing states are not zero-error matches.",
                ("WARNING: incomplete spectral coverage; inspect unmatched states before judging agreement."
                 if spectral_metrics['unmatched_vasp'] or spectral_metrics['unmatched_wannier']
                 else "No unmatched states within the selected window."),
                ("WARNING: no matched states; error metrics are undefined."
                 if not spectral_metrics['matched_pairs'] else "Error metrics apply to matched pairs only."),
                f"Total k points: {spectral_metrics['total_kpoints']}",
                f"K points with states in window: {spectral_metrics['eligible_kpoints']}",
                f"K points with all window states matched: {spectral_metrics['fully_matched_kpoints']}",
                f"VASP matched-state coverage: {spectral_metrics['vasp_coverage']:.8%}",
                f"Wannier matched-state coverage: {spectral_metrics['wannier_coverage']:.8%}",
                f"Total VASP states in window: {spectral_metrics['total_vasp_states']}",
                f"Total Wannier states in window: {spectral_metrics['total_wannier_states']}",
                f"Matched pairs: {spectral_metrics['matched_pairs']}",
                f"Sampled k points: {spectral_metrics['sampled_kpoints']}",
                f"Unmatched VASP states in window: {spectral_metrics['unmatched_vasp']}",
                f"Unmatched Wannier states in window: {spectral_metrics['unmatched_wannier']}",
                f"Mean signed difference (Wannier - VASP): {spectral_metrics['mean_signed']:.10f} eV",
                f"MAE: {spectral_metrics['mae']:.10f} eV",
                f"RMS: {spectral_metrics['rms']:.10f} eV",
                f"Maximum absolute deviation: {spectral_metrics['max_abs']:.10f} eV",
            ]
        )
        max_record = spectral_metrics["max_record"]
        if max_record is not None:
            report_lines.append(
                "Maximum-deviation location: "
                f"segment={max_record[4]}, x={max_record[1]:.10f}, "
                f"VASP={max_record[2]:.10f} eV, Wannier={max_record[3]:.10f} eV"
            )

    report_lines.extend(
        [
            "",
            "Mapped-boundary coordinate check:",
        ]
    )
    if mapped_boundary_warnings:
        for warning in mapped_boundary_warnings:
            report_lines.append(f"WARNING: {warning}")
    else:
        report_lines.append(
            "All mapped high-symmetry boundaries were found within tolerance."
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Boundary report     : {report_path}")


if __name__ == "__main__":
    main()
