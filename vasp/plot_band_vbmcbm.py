#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot VASPKIT band structures with robust VBM/CBM annotation.

Expected files in the current directory:
  - BAND.dat
  - BAND_GAP
  - KLABELS
  - KPOINTS (recommended for fractional-k to path-x projection)

Key behavior:
  - BAND_GAP is the primary source of VBM/CBM energies, positions, gap,
    and direct/indirect character.
  - BAND.dat is used for plotting and for locating the closest sampled
    band/spin state to each BAND_GAP edge.
  - Spin-polarized BAND.dat with k, spin-up, spin-down columns is supported.
  - Discontinuous line-mode paths such as X|U are handled segment by segment.
  - Missing or inconsistent gap evidence omits annotations while retaining the plot.
  - Optional EF fallback is disabled by default and rejects sampled crossings.
  - Output is SVG by default.
"""

import re
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# -------------------- User-tweakable params --------------------
FIGSIZE = (8, 12)
LINEWIDTH = 1.25
FERMI_LINE = True
FS_LABEL = 14
FS_TICK = 12
FS_ANN = 13
X_MERGE_TOL = 1e-4
Y_INIT = (-10, 10)
COLORMAP = "tab20"
SPARSE_XLABELS = False
MIN_LABEL_SPACING_FRAC = 0.06
OUTFILE = "band_plot.svg"

SPIN_UP_LINESTYLE = "-"
SPIN_DOWN_LINESTYLE = "--"
SHOW_SPIN_LEGEND = True

ALLOW_EF_FALLBACK = False
EF_FALLBACK_EXCLUSION = 1e-5
EDGE_ENERGY_MATCH_TOL = 2e-3
K_ENDPOINT_TOL = 1e-7
# ---------------------------------------------------------------


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _norm_label(lbl: str) -> str:
    s = lbl.strip().upper()
    return {
        "G": "Γ",
        "GAMMA": "Γ",
        "GAMMA_1": "Γ",
        "Γ": "Γ",
        "SIGMA": "Σ",
        "LAMBDA": "Λ",
        "DELTA": "Δ",
    }.get(s, s)


def _norm_label_for_display(lbl: str) -> str:
    parts = [p.strip() for p in lbl.split("|")]
    parts = [_norm_label(p) for p in parts]
    return "–".join(parts)


def _mod1_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    return d - np.round(d)


def _is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_band_dat(path: Path):
    """Return k, E, spin labels, and detected input layout.

    E has shape (nk, nb, nspin).

    Supported layouts:
      1. VASPKIT block layout, one block per band:
           k  energy
         or
           k  spin-up  spin-down
      2. Matrix/reformatted layout:
           k  band1  band2 ...
         or interleaved spin columns when a spin header is present.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    header_lower = "\n".join(line.lower() for line in lines if line.lstrip().startswith("#"))
    spin_header = (
        ("spin-up" in header_lower or "spin up" in header_lower)
        and ("spin-down" in header_lower or "spin down" in header_lower or "spin-dw" in header_lower)
    )
    explicit_band_headers = sum(
        1 for line in lines if re.search(r"band\s*[-_ ]?index", line, flags=re.I)
    )

    blocks = []
    cur = []

    def flush_current():
        nonlocal cur
        if cur:
            blocks.append(np.asarray(cur, dtype=float))
            cur = []

    for line in lines:
        s = line.strip()
        if not s:
            flush_current()
            continue
        if s.startswith("#"):
            if re.search(r"band\s*[-_ ]?index", s, flags=re.I):
                flush_current()
            continue

        toks = s.split()
        if len(toks) < 2 or not _is_float_token(toks[0]):
            continue

        row = []
        for tok in toks:
            if not _is_float_token(tok):
                break
            row.append(float(tok))
        if len(row) >= 2:
            cur.append(row)

    flush_current()

    if not blocks:
        raise RuntimeError("Failed to parse BAND.dat: no numeric band data found.")

    use_block_layout = explicit_band_headers > 0 or len(blocks) > 1

    if use_block_layout:
        npts = blocks[0].shape[0]
        if any(block.shape[0] != npts for block in blocks):
            raise RuntimeError("BAND.dat band blocks have inconsistent numbers of k points.")

        k = blocks[0][:, 0]
        for ib, block in enumerate(blocks[1:], start=2):
            if not np.allclose(block[:, 0], k, atol=1e-9, rtol=0.0):
                if np.allclose(block[::-1, 0], k, atol=1e-9, rtol=0.0):
                    blocks[ib-1] = block[::-1].copy()
                else:
                    raise RuntimeError(f"BAND.dat k-path differs in band block {ib}.")

        min_cols = min(block.shape[1] for block in blocks)
        max_cols = max(block.shape[1] for block in blocks)
        if min_cols != max_cols:
            raise RuntimeError("BAND.dat band blocks have inconsistent column counts.")

        if min_cols == 2:
            E = np.stack([block[:, 1] for block in blocks], axis=1)[:, :, None]
            spin_labels = ["single"]
            layout = "block-nonspin"
        elif min_cols >= 3 and spin_header:
            E_up = np.stack([block[:, 1] for block in blocks], axis=1)
            E_dn = np.stack([block[:, 2] for block in blocks], axis=1)
            E = np.stack([E_up, E_dn], axis=2)
            spin_labels = ["up", "down"]
            layout = "block-spin"
        elif min_cols == 3:
            print("[Warning] BAND.dat has three numeric columns per band block but no explicit spin header.")
            print("[Warning] Interpreting columns 2 and 3 as spin-up and spin-down.")
            E_up = np.stack([block[:, 1] for block in blocks], axis=1)
            E_dn = np.stack([block[:, 2] for block in blocks], axis=1)
            E = np.stack([E_up, E_dn], axis=2)
            spin_labels = ["up", "down"]
            layout = "block-spin-assumed"
        else:
            raise RuntimeError(
                "Unsupported BAND.dat block layout: expected 2 columns for non-spin or 3 for spin."
            )

        return k, E, spin_labels, layout

    data = blocks[0]
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError("Unsupported BAND.dat matrix layout.")

    k = data[:, 0]
    vals = data[:, 1:]

    if spin_header:
        if vals.shape[1] % 2 != 0:
            raise RuntimeError(
                "Spin-polarized matrix BAND.dat must contain an even number of energy columns."
            )
        nb = vals.shape[1] // 2
        E = vals.reshape(vals.shape[0], nb, 2)
        spin_labels = ["up", "down"]
        layout = "matrix-spin"
    else:
        E = vals[:, :, None]
        spin_labels = ["single"]
        layout = "matrix-nonspin"

    return k, E, spin_labels, layout


def parse_klabels(path: Path):
    if not path.exists():
        return None, None, None

    labs_match = []
    labs_disp = []
    xs = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("*") or "K-Label" in s:
            continue

        toks = s.split()
        if len(toks) < 2:
            continue

        try:
            x = float(toks[-1])
        except ValueError:
            print(f"[Warning] Skipping malformed KLABELS line: {s}")
            continue

        lab_raw = toks[0]
        labs_match.append(lab_raw)
        labs_disp.append(_norm_label_for_display(lab_raw))
        xs.append(x)

    if not labs_match:
        return None, None, None

    return labs_match, labs_disp, np.asarray(xs, dtype=float)


def _extract_kpoint_label(line: str) -> str:
    if "!" in line:
        label = line.split("!", 1)[1].strip()
        return _norm_label(label) if label else "?"

    toks = line.split()
    if len(toks) >= 4 and not _is_float_token(toks[3]):
        return _norm_label(toks[3])
    return "?"


def parse_kpoints_segments(path: Path):
    """Parse line-mode KPOINTS into physical path segments.

    Each returned item is (start_frac, end_frac, start_label, end_label).
    Discontinuous boundaries are preserved because segments are paired from
    the raw line-mode endpoints rather than deduplicated into one polyline.
    """
    if not path.exists():
        return None, None

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 5:
        return None, None

    mode = lines[3].strip().lower()
    if mode.startswith("c") or mode.startswith("k"):
        print("[Warning] KPOINTS appears to use Cartesian line-mode coordinates.")
        print("[Warning] BAND_GAP fractional coordinates will not be projected through KPOINTS.")
        return None, "cartesian"

    entries = []
    for line in lines[4:]:
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) < 3:
            continue
        try:
            coord = np.asarray([float(toks[0]), float(toks[1]), float(toks[2])], dtype=float)
        except ValueError:
            continue
        entries.append((coord, _extract_kpoint_label(s)))

    if len(entries) < 2:
        return None, "reciprocal"

    if len(entries) % 2 != 0:
        print("[Warning] KPOINTS line-mode endpoint count is odd; ignoring the last endpoint.")
        entries = entries[:-1]

    segments = []
    for i in range(0, len(entries), 2):
        a, la = entries[i]
        b, lb = entries[i + 1]
        segments.append((a, b, la, lb))

    return segments, "reciprocal"


def build_path_segments(kpoint_segments, kl_x):
    if kpoint_segments is None or kl_x is None:
        return None

    nseg = len(kpoint_segments)
    if len(kl_x) != nseg + 1:
        print(
            f"[Warning] KPOINTS contains {nseg} path segments but KLABELS contains "
            f"{len(kl_x)} x positions; expected {nseg + 1}."
        )
        print("[Warning] Fractional-k to path-x projection is disabled.")
        return None

    path_segments = []
    for i, (a, b, la, lb) in enumerate(kpoint_segments):
        path_segments.append(
            {
                "a": np.asarray(a, dtype=float),
                "b": np.asarray(b, dtype=float),
                "la": la,
                "lb": lb,
                "x0": float(kl_x[i]),
                "x1": float(kl_x[i + 1]),
            }
        )
    return path_segments


def parse_band_gap(path: Path):
    result = {
        "vfrac": None,
        "cfrac": None,
        "Evbm": None,
        "Ecbm": None,
        "Eg": None,
        "character": None,
        "homo_band": None,
        "lumo_band": None,
    }

    if not path.exists():
        return result

    txt = path.read_text(encoding="utf-8", errors="ignore")

    def search_float(pattern):
        m = re.search(pattern, txt, flags=re.I)
        return float(m.group(1)) if m else None

    def search_location(which):
        patterns = [
            rf"Location\s+of\s+{which}\s*(?:\(\s*frac\.?\s*\))?\s*:\s*({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})",
            rf"Location\s+of\s+{which}\s*(?:\(\s*frac\.?\s*\))?\s*=\s*({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})",
        ]
        for pattern in patterns:
            m = re.search(pattern, txt, flags=re.I)
            if m:
                return tuple(float(m.group(i)) for i in (1, 2, 3))
        return None

    m_char = re.search(r"Band\s+Character\s*:\s*([^\r\n]+)", txt, flags=re.I)
    if m_char:
        raw = m_char.group(1).strip().split()[0]
        result["character"] = raw.strip(";,.")

    result["Eg"] = search_float(rf"Band\s+Gap\s*\(\s*eV\s*\)\s*:\s*({FLOAT_RE})")
    if result["Eg"] is None:
        result["Eg"] = search_float(rf"Band\s+Gap\s*[:=]\s*({FLOAT_RE})\s*eV")

    result["Evbm"] = search_float(
        rf"Eigenvalue\s+of\s+VBM\s*\(\s*eV\s*\)\s*:\s*({FLOAT_RE})"
    )
    if result["Evbm"] is None:
        result["Evbm"] = search_float(rf"\bVBM\b\s*[:=]\s*({FLOAT_RE})\s*eV")

    result["Ecbm"] = search_float(
        rf"Eigenvalue\s+of\s+CBM\s*\(\s*eV\s*\)\s*:\s*({FLOAT_RE})"
    )
    if result["Ecbm"] is None:
        result["Ecbm"] = search_float(rf"\bCBM\b\s*[:=]\s*({FLOAT_RE})\s*eV")

    result["vfrac"] = search_location("VBM")
    result["cfrac"] = search_location("CBM")

    m_bands = re.search(
        r"HOMO\s*&\s*LUMO\s+Bands\s*:\s*(\d+)\s+(\d+)", txt, flags=re.I
    )
    if m_bands:
        result["homo_band"] = int(m_bands.group(1))
        result["lumo_band"] = int(m_bands.group(2))

    if result["Eg"] is None and result["Evbm"] is not None and result["Ecbm"] is not None:
        result["Eg"] = result["Ecbm"] - result["Evbm"]

    return result


def project_frac_to_path(kstar, path_segments):
    """Project reciprocal-equivalent points onto the actual line, not a wrapped line."""
    if kstar is None or not path_segments:
        return None, "?", None
    point = np.asarray(kstar, dtype=float)
    best = None
    tolerance = 2e-6
    for i, seg in enumerate(path_segments):
        a, b = seg["a"], seg["b"]
        d = b-a
        dd = float(np.dot(d,d))
        if dd < 1e-14:
            continue
        lo = np.ceil(np.minimum(a,b)-point-tolerance).astype(int)
        hi = np.floor(np.maximum(a,b)-point+tolerance).astype(int)
        for shift in itertools.product(*(range(l,h+1) for l,h in zip(lo,hi))):
            image = point+shift
            t = float(np.dot(image-a,d)/dd)
            residual = float(np.linalg.norm(image-(a+t*d)))
            if not -tolerance <= t <= 1+tolerance or residual > tolerance:
                continue
            t = float(np.clip(t,0,1))
            score = (residual,i,t)
            if best is None or score < best[0]:
                best = (score,i,t)
    if best is None:
        return None, "?", None
    _,i,t = best
    seg = path_segments[i]
    la,lb = _norm_label_for_display(seg["la"]),_norm_label_for_display(seg["lb"])
    label = la if t <= K_ENDPOINT_TOL else lb if t >= 1-K_ENDPOINT_TOL else f"{la}–{lb}"
    return float(seg["x0"]+t*(seg["x1"]-seg["x0"])),label,i


def locate_state_near_edge(k, E, target_energy, target_x=None, preferred_band=None):
    """Find the sampled state closest to a target BAND_GAP edge."""
    nk, nb, nspin = E.shape

    if target_x is not None:
        xdist = np.abs(k - target_x)
        min_xdist = float(np.min(xdist))
        k_candidates = np.where(np.isclose(xdist, min_xdist, atol=1e-12, rtol=0.0))[0]
    else:
        k_candidates = np.arange(nk, dtype=int)

    if preferred_band is not None and 0 <= preferred_band < nb:
        band_candidates = [preferred_band]
    else:
        band_candidates = range(nb)

    best = None
    for ik in k_candidates:
        for ib in band_candidates:
            for ispin in range(nspin):
                de = abs(float(E[ik, ib, ispin]) - float(target_energy))
                dx = abs(float(k[ik]) - float(target_x)) if target_x is not None else 0.0
                score = (dx, de, ib, ispin)
                if best is None or score < best[0]:
                    best = (score, ik, ib, ispin, float(E[ik, ib, ispin]))

    if best is None:
        return None

    _, ik, ib, ispin, energy = best

    if preferred_band is not None and abs(energy - float(target_energy)) > EDGE_ENERGY_MATCH_TOL:
        return locate_state_near_edge(k, E, target_energy, target_x=target_x, preferred_band=None)

    return {
        "ik": int(ik),
        "ib": int(ib),
        "ispin": int(ispin),
        "energy": energy,
        "energy_error": abs(energy - float(target_energy)),
    }


def fallback_edges_from_fermi(k, E):
    """Conservative fallback when BAND_GAP cannot provide both edge energies."""
    print("[Warning] BAND_GAP edge energies are unavailable or incomplete.")

    if not ALLOW_EF_FALLBACK:
        raise RuntimeError("BAND_GAP is required because ALLOW_EF_FALLBACK is False.")

    print("[Warning] Using optional EF=0 fallback; only sampled k points are assessed.")
    if np.any((np.min(E, axis=0) <= EF_FALLBACK_EXCLUSION) &
              (np.max(E, axis=0) >= -EF_FALLBACK_EXCLUSION)):
        raise RuntimeError("A sampled band touches or crosses EF; no insulating-gap annotation.")
    below = np.where(E <= -EF_FALLBACK_EXCLUSION, E, -np.inf)
    above = np.where(E >= +EF_FALLBACK_EXCLUSION, E, +np.inf)

    finite_below = np.isfinite(below)
    finite_above = np.isfinite(above)
    if not np.any(finite_below) or not np.any(finite_above):
        raise RuntimeError("Cannot determine band edges from the EF=0 fallback.")

    Ev = float(np.max(below[finite_below]))
    Ec = float(np.min(above[finite_above]))

    v_pos = np.argwhere(np.isclose(E, Ev, atol=EDGE_ENERGY_MATCH_TOL, rtol=0.0))
    c_pos = np.argwhere(np.isclose(E, Ec, atol=EDGE_ENERGY_MATCH_TOL, rtol=0.0))

    iv = int(v_pos[0, 0])
    ic = int(c_pos[0, 0])
    vb = int(v_pos[0, 1])
    cb = int(c_pos[0, 1])
    vs = int(v_pos[0, 2])
    cs = int(c_pos[0, 2])

    return {
        "Ev": Ev,
        "Ec": Ec,
        "Eg": Ec - Ev,
        "xv": float(k[iv]),
        "xc": float(k[ic]),
        "lbl_v": "sampled k",
        "lbl_c": "sampled k",
        "is_direct": iv == ic,
        "v_state": {"ik": iv, "ib": vb, "ispin": vs, "energy": Ev, "energy_error": 0.0},
        "c_state": {"ik": ic, "ib": cb, "ispin": cs, "energy": Ec, "energy_error": 0.0},
        "source": "EF fallback",
    }


def determine_edges(k, E, gap_info, path_segments):
    Ev = gap_info["Evbm"]
    Ec = gap_info["Ecbm"]

    if Ev is None or Ec is None:
        return fallback_edges_from_fermi(k, E)

    Eg = gap_info["Eg"] if gap_info["Eg"] is not None else Ec - Ev
    if Eg <= EF_FALLBACK_EXCLUSION or Ec <= Ev:
        raise RuntimeError("BAND_GAP reports no positive insulating gap.")
    if abs(Eg-(Ec-Ev)) > EDGE_ENERGY_MATCH_TOL:
        raise RuntimeError("BAND_GAP gap and edge energies are inconsistent.")

    xv, lbl_v, _ = project_frac_to_path(gap_info["vfrac"], path_segments)
    xc, lbl_c, _ = project_frac_to_path(gap_info["cfrac"], path_segments)

    if path_segments:
        if gap_info["vfrac"] is not None and xv is None:
            raise RuntimeError("VBM is not on the plotted k path.")
        if gap_info["cfrac"] is not None and xc is None:
            raise RuntimeError("CBM is not on the plotted k path.")

    homo_band = gap_info["homo_band"] - 1 if gap_info["homo_band"] is not None else None
    lumo_band = gap_info["lumo_band"] - 1 if gap_info["lumo_band"] is not None else None

    v_state = locate_state_near_edge(k, E, Ev, target_x=xv, preferred_band=homo_band)
    c_state = locate_state_near_edge(k, E, Ec, target_x=xc, preferred_band=lumo_band)

    if xv is None:
        if v_state is None:
            raise RuntimeError("Cannot place the VBM on the plotted k path.")
        xv = float(k[v_state["ik"]])
        lbl_v = "sampled k"

    if xc is None:
        if c_state is None:
            raise RuntimeError("Cannot place the CBM on the plotted k path.")
        xc = float(k[c_state["ik"]])
        lbl_c = "sampled k"

    if v_state is None:
        v_state = locate_state_near_edge(k, E, Ev, target_x=xv, preferred_band=homo_band)
    if c_state is None:
        c_state = locate_state_near_edge(k, E, Ec, target_x=xc, preferred_band=lumo_band)

    for name, state in (("VBM", v_state), ("CBM", c_state)):
        if state is None or state["energy_error"] > EDGE_ENERGY_MATCH_TOL:
            raise RuntimeError(f"{name} does not match the plotted energies; check energy reference and inputs.")

    character = (gap_info["character"] or "").strip().lower()
    if character.startswith("direct"):
        is_direct = True
    elif character.startswith("indirect"):
        is_direct = False
    elif gap_info["vfrac"] is not None and gap_info["cfrac"] is not None:
        is_direct = float(np.linalg.norm(_mod1_delta(gap_info["vfrac"], gap_info["cfrac"]))) < 1e-8
    else:
        is_direct = abs(float(xv) - float(xc)) < X_MERGE_TOL

    return {
        "Ev": float(Ev),
        "Ec": float(Ec),
        "Eg": float(Eg),
        "xv": float(xv),
        "xc": float(xc),
        "lbl_v": lbl_v,
        "lbl_c": lbl_c,
        "is_direct": bool(is_direct),
        "v_state": v_state,
        "c_state": c_state,
        "source": "BAND_GAP",
    }


def sparsify_ticks(xs, labels, min_frac):
    xmin = float(np.min(xs))
    xmax = float(np.max(xs))
    span = xmax - xmin if xmax > xmin else 1.0

    keep_x = []
    keep_l = []
    last_kept = None

    for x, lab in zip(xs, labels):
        if last_kept is None or (x - last_kept) >= (min_frac * span) or x >= xmax - 1e-12:
            keep_x.append(x)
            keep_l.append(lab)
            last_kept = x

    if keep_x and keep_x[-1] < xmax - 1e-12:
        keep_x.append(xmax)
        keep_l.append(labels[-1])

    return np.asarray(keep_x, dtype=float), keep_l


def _spin_name(spin_labels, ispin):
    if ispin is None or ispin >= len(spin_labels):
        return "?"
    return spin_labels[ispin]


def print_edge_state(name, state, k, spin_labels, target_energy):
    if state is None:
        print(f"[Summary] {name} sampled-state match: unavailable")
        return

    msg = (
        f"[Summary] {name} sampled state: k-index={state['ik']}, "
        f"x={float(k[state['ik']]):.8f}, band={state['ib'] + 1}, "
        f"spin={_spin_name(spin_labels, state['ispin'])}, "
        f"E={state['energy']:.8f} eV, |dE|={state['energy_error']:.3e} eV"
    )
    print(msg)
    if state["energy_error"] > EDGE_ENERGY_MATCH_TOL:
        print(
            f"[Warning] {name} BAND.dat match differs from BAND_GAP by more than "
            f"{EDGE_ENERGY_MATCH_TOL:g} eV."
        )
        print(f"[Warning] {name} BAND_GAP target energy is {target_energy:.8f} eV.")


def main():
    root = Path(".")

    k, E, spin_labels, band_layout = load_band_dat(root / "BAND.dat")
    nk, nb, nspin = E.shape

    kl_labs_match, kl_labs_disp, kl_x = parse_klabels(root / "KLABELS")
    kpoint_segments, kpoint_mode = parse_kpoints_segments(root / "KPOINTS")
    path_segments = build_path_segments(kpoint_segments, kl_x)
    gap_info = parse_band_gap(root / "BAND_GAP")

    try:
        edges = determine_edges(k, E, gap_info, path_segments)
    except RuntimeError as exc:
        print(f"[Warning] Gap annotation omitted: {exc}")
        edges = None
    if edges is not None:
        Ev = edges["Ev"]
        Ec = edges["Ec"]
        Eg = edges["Eg"]
        xv = edges["xv"]
        xc = edges["xc"]
        lbl_v = edges["lbl_v"]
        lbl_c = edges["lbl_c"]
        is_direct = edges["is_direct"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    cmap = plt.get_cmap(COLORMAP)
    colors = [cmap(i % cmap.N) for i in range(nb)]

    cuts = np.r_[0, np.where(np.abs(np.diff(k)) < 1e-10)[0]+1, len(k)]
    for ib in range(nb):
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            ax.plot(k[lo:hi], E[lo:hi, ib, 0], lw=LINEWIDTH, color=colors[ib], ls=SPIN_UP_LINESTYLE)
            if nspin == 2:
                ax.plot(k[lo:hi], E[lo:hi, ib, 1], lw=LINEWIDTH, color=colors[ib], ls=SPIN_DOWN_LINESTYLE)

    if kl_labs_disp is not None and kl_x is not None:
        for x in kl_x:
            ax.axvline(x, lw=0.8, c="0.85", zorder=0)

        if SPARSE_XLABELS:
            xs_show, labs_show = sparsify_ticks(kl_x, kl_labs_disp, MIN_LABEL_SPACING_FRAC)
            ax.set_xticks(xs_show)
            ax.set_xticklabels(labs_show, fontsize=FS_TICK)
        else:
            ax.set_xticks(kl_x)
            ax.set_xticklabels(kl_labs_disp, fontsize=FS_TICK)

    if FERMI_LINE:
        ax.axhline(0.0, lw=0.8, c="0.4", ls="--", zorder=0)

    if edges is not None:
        if abs(xv - xc) < X_MERGE_TOL:
            x_mid = 0.5 * (xv + xc)
            ax.axvline(x_mid, linestyle="--", linewidth=1.0, c="0.2")
        else:
            ax.axvline(xv, linestyle="--", linewidth=1.0, c="0.2")
            ax.axvline(xc, linestyle="--", linewidth=1.0, c="0.2")

        ax.scatter([xv], [Ev], marker="o", s=38, zorder=6, edgecolors="k", linewidths=0.6)
        ax.scatter([xc], [Ec], marker="s", s=40, zorder=6, edgecolors="k", linewidths=0.6)

        ax.annotate(
            "VBM",
            xy=(xv, Ev),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FS_ANN,
        )
        ax.annotate(
            "CBM",
            xy=(xc, Ec),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FS_ANN,
        )

        ax.annotate(
            "",
            xy=(xc, Ec),
            xytext=(xv, Ev),
            arrowprops=dict(arrowstyle="->", linestyle="--", linewidth=1.1, color="0.2"),
        )

        caption = (
            f"E$_g$={Eg:.3f} eV ({'direct' if is_direct else 'indirect'}), "
            f"VBM@{lbl_v} → CBM@{lbl_c}"
        )

        if is_direct:
            ax.text(
                xv,
                0.5 * (Ev + Ec),
                caption,
                ha="left",
                va="center",
                rotation=90,
                fontsize=FS_ANN,
            )
        else:
            ax.text(
                0.5 * (xv + xc),
                0.5 * (Ev + Ec),
                caption,
                ha="center",
                va="bottom",
                fontsize=FS_ANN,
            )

    if nspin == 2 and SHOW_SPIN_LEGEND:
        handles = [
            Line2D([0], [0], color="0.2", lw=LINEWIDTH, ls=SPIN_UP_LINESTYLE, label="Spin up"),
            Line2D([0], [0], color="0.2", lw=LINEWIDTH, ls=SPIN_DOWN_LINESTYLE, label="Spin down"),
        ]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=FS_TICK)

    ax.set_xlabel("k-path", fontsize=FS_LABEL)
    ax.set_ylabel(r"$E - E_F$ (eV)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(float(np.min(k)), float(np.max(k)))
    ax.set_ylim(Y_INIT[0], Y_INIT[1])

    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=300)
    plt.close(fig)

    print(f"Saved: {OUTFILE}")
    print(f"[Summary] BAND.dat layout: {band_layout}; nk={nk}; nb={nb}; nspin={nspin}")
    if edges is not None:
        print(f"[Summary] Edge source: {edges['source']}")
        if kpoint_mode is not None:
            print(f"[Summary] KPOINTS coordinate mode: {kpoint_mode}")
        print(f"[Summary] VBM: E={Ev:.8f} eV, x={xv:.8f}, label={lbl_v}")
        print(f"[Summary] CBM: E={Ec:.8f} eV, x={xc:.8f}, label={lbl_c}")
        print(f"[Summary] Eg={Eg:.8f} eV; {'direct' if is_direct else 'indirect'}")

        print_edge_state("VBM", edges["v_state"], k, spin_labels, Ev)
        print_edge_state("CBM", edges["c_state"], k, spin_labels, Ec)


if __name__ == "__main__":
    main()
