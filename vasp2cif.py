#!/usr/bin/env python3
"""
Convert VASP POSCAR/CONTCAR files to CIF.

Default behavior:
1. Scan the current directory for all regular files matching CONTCAR* and POSCAR*.
2. Ignore files ending in .cif so generated outputs are not re-read.
3. If exactly one CONTCAR* file and exactly one POSCAR* file are found,
   process only the CONTCAR* file.
4. Otherwise, process all matching files.
5. OUTCAR files are not processed.

The parser supports:
- VASP 5/6 element lines.
- VASP 4 style files when a readable POTCAR is available in the same directory.
- Direct coordinates.
- Cartesian coordinates, including C/c and K/k coordinate-mode prefixes.
- Selective Dynamics lines.
- One positive scale factor.
- One negative scale factor interpreted as target cell volume.
- Three positive Cartesian scale factors.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


Vector = List[float]
Matrix = List[Vector]


class VaspParseError(Exception):
    pass


def determinant3(m: Matrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def inverse3(m: Matrix) -> Matrix:
    det = determinant3(m)
    if abs(det) < 1.0e-15:
        raise VaspParseError("Lattice matrix is singular or nearly singular.")

    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]


def row_vector_times_matrix(v: Sequence[float], m: Matrix) -> Vector:
    return [
        v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]
        for j in range(3)
    ]


def vector_length(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def angle_degrees(u: Sequence[float], v: Sequence[float]) -> float:
    lu = vector_length(u)
    lv = vector_length(v)
    if lu == 0.0 or lv == 0.0:
        raise VaspParseError("Zero-length lattice vector found.")
    cosine = sum(a * b for a, b in zip(u, v)) / (lu * lv)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def parse_float_triplet(line: str, what: str) -> Vector:
    fields = line.split()
    if len(fields) < 3:
        raise VaspParseError(f"{what} requires at least three numeric values.")
    try:
        return [float(fields[0]), float(fields[1]), float(fields[2])]
    except ValueError as exc:
        raise VaspParseError(f"Invalid numeric values in {what}: {line!r}") from exc


def all_integer_tokens(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    try:
        for token in tokens:
            int(token)
        return True
    except ValueError:
        return False


def normalize_element_label(label: str) -> str:
    match = re.match(r"^([A-Z][a-z]?)", label.strip())
    if match:
        return match.group(1)
    return label.strip()


def read_elements_from_potcar(directory: Path, expected_species: int) -> List[str]:
    potcar = directory / "POTCAR"
    if not potcar.is_file():
        raise VaspParseError(
            "VASP 4 style file detected, but no POTCAR was found in the same directory."
        )

    elements: List[str] = []
    try:
        with potcar.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "TITEL" not in line:
                    continue
                parts = line.split()
                if len(parts) >= 4 and parts[0] == "TITEL":
                    elements.append(normalize_element_label(parts[3].split("_")[0]))
    except OSError as exc:
        raise VaspParseError(f"Could not read POTCAR: {exc}") from exc

    if len(elements) != expected_species:
        raise VaspParseError(
            f"POTCAR contains {len(elements)} TITEL entries, but "
            f"{expected_species} species are required."
        )

    return elements


def parse_scale_and_lattice(lines: Sequence[str]) -> Tuple[Matrix, Vector]:
    if len(lines) < 5:
        raise VaspParseError("File is too short to contain a valid POSCAR/CONTCAR.")

    scale_tokens = lines[1].split()

    try:
        scale_values = [float(x) for x in scale_tokens]
    except ValueError as exc:
        raise VaspParseError("Invalid scaling factor line.") from exc

    if len(scale_values) == 1:
        raw_scale = scale_values[0]
        if raw_scale == 0.0:
            raise VaspParseError("The POSCAR scale factor must not be zero.")
    elif len(scale_values) == 3:
        if any(x <= 0.0 for x in scale_values):
            raise VaspParseError(
                "Three-component POSCAR scale factors must all be positive."
            )
    else:
        raise VaspParseError(
            "The POSCAR scale line must contain either one or three numbers."
        )

    raw_lattice = [
        parse_float_triplet(lines[2], "first lattice vector"),
        parse_float_triplet(lines[3], "second lattice vector"),
        parse_float_triplet(lines[4], "third lattice vector"),
    ]

    raw_det = determinant3(raw_lattice)
    if abs(raw_det) < 1.0e-15:
        raise VaspParseError("Lattice vectors have zero or near-zero volume.")

    if len(scale_values) == 1:
        raw_scale = scale_values[0]
        if raw_scale > 0.0:
            cartesian_scale = [raw_scale, raw_scale, raw_scale]
        else:
            target_volume = abs(raw_scale)
            uniform_scale = (target_volume / abs(raw_det)) ** (1.0 / 3.0)
            cartesian_scale = [uniform_scale, uniform_scale, uniform_scale]
    else:
        cartesian_scale = scale_values

    lattice = [
        [
            raw_lattice[i][j] * cartesian_scale[j]
            for j in range(3)
        ]
        for i in range(3)
    ]

    return lattice, cartesian_scale


def parse_vasp_file(path: Path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise VaspParseError(f"Could not read file: {exc}") from exc

    if len(lines) < 8:
        raise VaspParseError("File is too short to be a valid POSCAR/CONTCAR.")

    lattice, cartesian_scale = parse_scale_and_lattice(lines)

    line5_tokens = lines[5].split()
    if not line5_tokens:
        raise VaspParseError("Missing element/count information.")

    if all_integer_tokens(line5_tokens):
        counts = [int(x) for x in line5_tokens]
        elements = read_elements_from_potcar(path.parent, len(counts))
        cursor = 6
    else:
        elements = [normalize_element_label(x) for x in line5_tokens]
        if len(lines) <= 6:
            raise VaspParseError("Missing atom-count line.")
        count_tokens = lines[6].split()
        if not all_integer_tokens(count_tokens):
            raise VaspParseError("Atom-count line does not contain valid integers.")
        counts = [int(x) for x in count_tokens]
        cursor = 7

    if len(elements) != len(counts):
        raise VaspParseError(
            f"Found {len(elements)} element labels but {len(counts)} atom counts."
        )

    if any(n < 0 for n in counts):
        raise VaspParseError("Atom counts must not be negative.")

    if cursor >= len(lines):
        raise VaspParseError("Missing coordinate-mode line.")

    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
        if cursor >= len(lines):
            raise VaspParseError("Missing coordinate-mode line after Selective Dynamics.")

    mode_line = lines[cursor].strip()
    if not mode_line:
        raise VaspParseError("Empty coordinate-mode line.")

    mode_char = mode_line[0].lower()
    cartesian = mode_char in {"c", "k"}
    cursor += 1

    total_atoms = sum(counts)
    if len(lines) - cursor < total_atoms:
        raise VaspParseError(
            f"Expected {total_atoms} coordinate lines, but only "
            f"{max(0, len(lines) - cursor)} are available."
        )

    atom_labels: List[str] = []
    for element, count in zip(elements, counts):
        atom_labels.extend([element] * count)

    inverse_lattice = inverse3(lattice)
    fractional_positions: List[Vector] = []

    for i in range(total_atoms):
        coords = parse_float_triplet(lines[cursor + i], f"atomic coordinate line {i + 1}")

        if cartesian:
            physical_cartesian = [
                coords[j] * cartesian_scale[j]
                for j in range(3)
            ]
            frac = row_vector_times_matrix(physical_cartesian, inverse_lattice)
        else:
            frac = coords

        fractional_positions.append(frac)

    return {
        "comment": lines[0].strip(),
        "elements": elements,
        "counts": counts,
        "labels": atom_labels,
        "lattice": lattice,
        "fractional_positions": fractional_positions,
    }


def cif_safe_data_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    safe = safe.strip("_.-")
    return safe or "vasp_structure"


def format_cif(data, source_name: str) -> str:
    a_vec, b_vec, c_vec = data["lattice"]

    a = vector_length(a_vec)
    b = vector_length(b_vec)
    c = vector_length(c_vec)

    alpha = angle_degrees(b_vec, c_vec)
    beta = angle_degrees(a_vec, c_vec)
    gamma = angle_degrees(a_vec, b_vec)

    output: List[str] = []
    output.append(f"data_{cif_safe_data_name(source_name)}")
    output.append("_audit_creation_method 'Generated by vasp2cif'")
    output.append(f"_cell_length_a {a:.12f}")
    output.append(f"_cell_length_b {b:.12f}")
    output.append(f"_cell_length_c {c:.12f}")
    output.append(f"_cell_angle_alpha {alpha:.10f}")
    output.append(f"_cell_angle_beta {beta:.10f}")
    output.append(f"_cell_angle_gamma {gamma:.10f}")
    output.append("_symmetry_space_group_name_H-M 'P 1'")
    output.append("_symmetry_Int_Tables_number 1")
    output.append("")
    output.append("loop_")
    output.append("_atom_site_label")
    output.append("_atom_site_type_symbol")
    output.append("_atom_site_fract_x")
    output.append("_atom_site_fract_y")
    output.append("_atom_site_fract_z")
    output.append("_atom_site_occupancy")

    per_element_index = {}
    for element, frac in zip(data["labels"], data["fractional_positions"]):
        per_element_index[element] = per_element_index.get(element, 0) + 1
        atom_label = f"{element}{per_element_index[element]}"
        output.append(
            f"{atom_label} {element} "
            f"{frac[0]:.15f} {frac[1]:.15f} {frac[2]:.15f} 1.0"
        )

    output.append("")
    return "\n".join(output)


def discover_default_inputs(directory: Path) -> List[Path]:
    contcars = sorted(
        p for p in directory.glob("CONTCAR*")
        if p.is_file() and not p.name.lower().endswith(".cif")
    )
    poscars = sorted(
        p for p in directory.glob("POSCAR*")
        if p.is_file() and not p.name.lower().endswith(".cif")
    )

    if len(contcars) == 1 and len(poscars) == 1:
        return contcars

    return contcars + poscars


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(input_path.name + ".cif")


def convert_one(path: Path) -> Path:
    data = parse_vasp_file(path)
    cif_text = format_cif(data, path.name)
    output_path = output_path_for(path)
    output_path.write_text(cif_text, encoding="utf-8", newline="\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert VASP POSCAR/CONTCAR files to CIF."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Optional explicit input files. If omitted, scan the current directory "
            "for CONTCAR* and POSCAR*."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.files:
        inputs = [Path(name) for name in args.files]
    else:
        inputs = discover_default_inputs(Path.cwd())

    if not inputs:
        print("No CONTCAR* or POSCAR* files found.", file=sys.stderr)
        return 1

    failures = 0

    for input_path in inputs:
        if not input_path.is_file():
            print(f"ERROR: Not a file: {input_path}", file=sys.stderr)
            failures += 1
            continue

        try:
            output_path = convert_one(input_path)
            print(f"Converted: {input_path} -> {output_path}")
        except (VaspParseError, OSError) as exc:
            print(f"ERROR: {input_path}: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
