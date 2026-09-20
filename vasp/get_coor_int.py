#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
ENERGY_RE = re.compile(r"energy\(sigma->0\)\s*=\s*(" + FLOAT_RE + r")", re.IGNORECASE)


class ParseError(RuntimeError):
    pass


def to_float(token):
    return float(token.replace("D", "E").replace("d", "e"))


def determinant3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def parse_poscar(path):
    lines = Path(path).read_text(errors="replace").splitlines()
    if len(lines) < 8:
        raise ParseError(f"{path} is too short to be a valid POSCAR.")

    comment = lines[0].strip() or "Extracted structure"
    scale_tokens = lines[1].split()
    try:
        scales = [to_float(x) for x in scale_tokens]
    except ValueError as exc:
        raise ParseError(f"Invalid POSCAR scale line: {lines[1]}") from exc

    if len(scales) not in (1, 3):
        raise ParseError("POSCAR scale line must contain one or three numbers.")

    try:
        raw_lattice = [
            [to_float(x) for x in lines[i].split()[:3]]
            for i in range(2, 5)
        ]
    except (ValueError, IndexError) as exc:
        raise ParseError("Failed to parse POSCAR lattice vectors.") from exc

    if any(len(v) != 3 for v in raw_lattice):
        raise ParseError("Each POSCAR lattice vector must contain three numbers.")

    if len(scales) == 1:
        scale = scales[0]
        if scale == 0.0:
            raise ParseError("POSCAR scale factor cannot be zero.")
        if scale > 0.0:
            lattice = [[scale * x for x in v] for v in raw_lattice]
        else:
            raw_volume = abs(determinant3(raw_lattice))
            if raw_volume <= 0.0:
                raise ParseError("POSCAR lattice has zero volume.")
            factor = (abs(scale) / raw_volume) ** (1.0 / 3.0)
            lattice = [[factor * x for x in v] for v in raw_lattice]
    else:
        if any(s <= 0.0 for s in scales):
            raise ParseError("Three POSCAR scale factors must all be positive.")
        lattice = [
            [raw_lattice[i][j] * scales[j] for j in range(3)]
            for i in range(3)
        ]

    line5 = lines[5].split()
    if not line5:
        raise ParseError("Missing POSCAR species/count line.")

    vasp4_style = all(re.fullmatch(r"[+-]?\d+", x) for x in line5)
    if vasp4_style:
        symbols = None
        counts = [int(x) for x in line5]
        counts_index = 5
    else:
        symbols = line5
        if len(lines) <= 6:
            raise ParseError("Missing POSCAR atom counts.")
        try:
            counts = [int(x) for x in lines[6].split()]
        except ValueError as exc:
            raise ParseError("Invalid POSCAR atom counts.") from exc
        counts_index = 6
        if len(symbols) != len(counts):
            raise ParseError("POSCAR species and count columns do not match.")

    if any(n < 0 for n in counts):
        raise ParseError("POSCAR atom counts cannot be negative.")

    natoms = sum(counts)
    idx = counts_index + 1
    selective = False

    if idx < len(lines) and lines[idx].strip().lower().startswith("s"):
        selective = True
        idx += 1

    if idx >= len(lines):
        raise ParseError("Missing POSCAR coordinate mode.")

    mode = lines[idx].strip()
    if not mode:
        raise ParseError("Missing POSCAR coordinate mode.")
    idx += 1

    if idx + natoms > len(lines):
        raise ParseError("POSCAR does not contain enough coordinate lines.")

    flags = None
    if selective:
        flags = []
        for atom_index in range(natoms):
            tokens = lines[idx + atom_index].split()
            if len(tokens) < 6:
                raise ParseError(
                    f"Selective dynamics flags are missing for atom {atom_index + 1}."
                )
            atom_flags = [x.upper() for x in tokens[3:6]]
            if any(x not in ("T", "F") for x in atom_flags):
                raise ParseError(
                    f"Invalid selective dynamics flags for atom {atom_index + 1}."
                )
            flags.append(atom_flags)

    return {
        "comment": comment,
        "symbols": symbols,
        "counts": counts,
        "natoms": natoms,
        "selective": selective,
        "flags": flags,
        "lattice": lattice,
    }


def parse_outcar(path, natoms):
    lines = Path(path).read_text(errors="replace").splitlines()
    positions = []
    lattices = []
    energies = []

    i = 0
    nlines = len(lines)

    while i < nlines:
        line = lines[i]

        if "direct lattice vectors" in line.lower():
            if i + 3 < nlines:
                block = []
                ok = True
                for j in range(1, 4):
                    tokens = lines[i + j].split()
                    if len(tokens) < 3:
                        ok = False
                        break
                    try:
                        block.append([to_float(tokens[k]) for k in range(3)])
                    except ValueError:
                        ok = False
                        break
                if ok:
                    lattices.append((i, block))
                    i += 4
                    continue

        match = ENERGY_RE.search(line)
        if match:
            energies.append((i, to_float(match.group(1))))

        if "POSITION" in line and "TOTAL-FORCE" in line:
            data_start = i + 1
            while data_start < nlines and (
                not lines[data_start].strip()
                or set(lines[data_start].strip()) <= {"-"}
            ):
                data_start += 1

            coords = []
            forces = []
            complete = True

            for atom_offset in range(natoms):
                line_index = data_start + atom_offset
                if line_index >= nlines:
                    complete = False
                    break
                tokens = lines[line_index].split()
                if len(tokens) < 6:
                    complete = False
                    break
                try:
                    values = [to_float(tokens[k]) for k in range(6)]
                except ValueError:
                    complete = False
                    break
                coords.append(values[:3])
                forces.append(values[3:6])

            if complete and len(coords) == natoms:
                positions.append(
                    {
                        "line": i,
                        "coords": coords,
                        "forces": forces,
                    }
                )
                i = data_start + natoms
                continue

        i += 1

    if not positions:
        raise ParseError("No complete POSITION/TOTAL-FORCE blocks were found in OUTCAR.")

    assign_energies(positions, energies)
    assign_lattices(positions, lattices)

    return positions


def nearest_event(line_number, events):
    if not events:
        return None
    return min(events, key=lambda item: abs(item[0] - line_number))


def assign_energies(steps, energies):
    if len(energies) == len(steps):
        for step, (_, energy) in zip(steps, energies):
            step["energy"] = energy
        return

    for step in steps:
        event = nearest_event(step["line"], energies)
        step["energy"] = None if event is None else event[1]


def assign_lattices(steps, lattices):
    if not lattices:
        for step in steps:
            step["lattice"] = None
        return

    if len(lattices) == 1:
        lattice = lattices[0][1]
        for step in steps:
            step["lattice"] = lattice
        return

    if len(lattices) == len(steps):
        for step, (_, lattice) in zip(steps, lattices):
            step["lattice"] = lattice
        return

    for step in steps:
        before = [event for event in lattices if event[0] <= step["line"]]
        if before:
            step["lattice"] = before[-1][1]
        else:
            step["lattice"] = nearest_event(step["line"], lattices)[1]


def write_summary(path, steps):
    with open(path, "w", encoding="utf-8") as handle:
        for index, step in enumerate(steps, start=1):
            handle.write(f"Step: {index}\n")
            handle.write("      x(A)          y(A)          z(A)        |F|(eV/A)\n")
            for coord, force in zip(step["coords"], step["forces"]):
                force_norm = math.sqrt(sum(x * x for x in force))
                handle.write(
                    f"{coord[0]:13.6f} {coord[1]:13.6f} {coord[2]:13.6f} "
                    f"{force_norm:14.7f}\n"
                )

            energy = step.get("energy")
            if energy is None:
                handle.write("energy(sigma->0) = NOT FOUND\n")
            else:
                handle.write(f"energy(sigma->0) = {energy:.12f} eV\n")
            handle.write("\n")


def write_poscar(path, poscar_info, step, step_number):
    lattice = step.get("lattice")
    if lattice is None:
        lattice = poscar_info["lattice"]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{poscar_info['comment']} | OUTCAR step {step_number}\n")
        handle.write("1.0\n")

        for vector in lattice:
            handle.write(
                f"{vector[0]:20.12f} {vector[1]:20.12f} {vector[2]:20.12f}\n"
            )

        if poscar_info["symbols"] is not None:
            handle.write(" ".join(poscar_info["symbols"]) + "\n")
        handle.write(" ".join(str(x) for x in poscar_info["counts"]) + "\n")

        if poscar_info["selective"]:
            handle.write("Selective dynamics\n")

        handle.write("Cartesian\n")

        flags = poscar_info["flags"]
        for atom_index, coord in enumerate(step["coords"]):
            line = f"{coord[0]:20.12f} {coord[1]:20.12f} {coord[2]:20.12f}"
            if flags is not None:
                line += "   " + " ".join(flags[atom_index])
            handle.write(line + "\n")


def parse_requested_steps(arguments):
    requested = []
    for item in arguments:
        try:
            step = int(item)
        except ValueError as exc:
            raise ParseError(f"Invalid step number: {item}") from exc
        if step <= 0:
            raise ParseError(f"Step numbers must be positive: {item}")
        requested.append(step)
    return sorted(set(requested))


def main():
    try:
        requested_steps = parse_requested_steps(sys.argv[1:])
        poscar_info = parse_poscar("POSCAR")
        steps = parse_outcar("OUTCAR", poscar_info["natoms"])

        write_summary("OUTCAR.pos", steps)

        for step_number in requested_steps:
            if step_number > len(steps):
                raise ParseError(
                    f"Requested step {step_number}, but OUTCAR contains only "
                    f"{len(steps)} complete ionic steps."
                )
            write_poscar(
                f"POSCAR{step_number}",
                poscar_info,
                steps[step_number - 1],
                step_number,
            )

        print(f"Parsed {len(steps)} complete ionic steps.")
        print("Wrote OUTCAR.pos.")
        if requested_steps:
            names = ", ".join(f"POSCAR{x}" for x in requested_steps)
            print(f"Wrote {names}.")

        missing_energy = sum(step.get("energy") is None for step in steps)
        if missing_energy:
            print(
                f"Warning: energy(sigma->0) was not found for "
                f"{missing_energy} step(s).",
                file=sys.stderr,
            )

        missing_lattice = sum(step.get("lattice") is None for step in steps)
        if missing_lattice:
            print(
                "Warning: no OUTCAR lattice was found for some steps; "
                "the scaled POSCAR lattice was used instead.",
                file=sys.stderr,
            )

    except (OSError, ParseError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
