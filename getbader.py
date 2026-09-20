#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import re
import subprocess
import time
from typing import List, Optional, Tuple

POSCAR = "POSCAR"
POTCAR = "POTCAR"
ACF = "ACF.dat"
OUTCSV = "bader_charge_summary.csv"
REQUIRED_INPUTS = (POSCAR, POTCAR, "CHGCAR", "AECCAR0", "AECCAR2")


# ----------- File and command checks -----------

def require_files(paths: Tuple[str, ...]) -> None:
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise SystemExit("Missing required file(s): " + ", ".join(missing))


def run_cmd(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit(
            f"Command not found: {' '.join(cmd)}. "
            "Make sure it is installed and available in PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Command failed: {' '.join(cmd)}\nReturn code: {exc.returncode}"
        )


# ----------- POSCAR parsing -----------

def parse_poscar(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if len(lines) < 7:
        raise SystemExit("POSCAR has too few lines to determine species and counts.")

    line6 = lines[5].split()
    line7 = lines[6].split()

    if not line6:
        raise SystemExit("POSCAR line 6 is empty.")

    def is_all_int(tokens: List[str]) -> bool:
        if not tokens:
            return False
        try:
            [int(token) for token in tokens]
            return True
        except ValueError:
            return False

    if not is_all_int(line6) and is_all_int(line7):
        elements = line6
        counts = [int(value) for value in line7]
        if len(elements) != len(counts):
            raise SystemExit(
                "POSCAR species/count mismatch: "
                f"{len(elements)} species labels but {len(counts)} counts."
            )
    elif is_all_int(line6):
        elements = []
        counts = [int(value) for value in line6]
    else:
        raise SystemExit(
            "Cannot determine whether POSCAR uses VASP4 or VASP5 format. "
            "Check lines 6 and 7."
        )

    if any(count < 0 for count in counts):
        raise SystemExit("POSCAR contains a negative atom count.")
    if sum(counts) <= 0:
        raise SystemExit("POSCAR contains no atoms.")

    return elements, counts


# ----------- POTCAR parsing -----------

def normalize_element_label(label: str) -> str:
    match = re.match(r"^([A-Z][a-z]?)", label.strip())
    if not match:
        raise SystemExit(f"Cannot extract an element symbol from label: {label!r}")
    return match.group(1)


def parse_potcar(path: str) -> Tuple[List[str], List[Optional[float]]]:
    species_labels: List[str] = []
    zvals: List[Optional[float]] = []

    current_label: Optional[str] = None
    current_zval: Optional[float] = None
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

    def finish_current_block() -> None:
        if current_label is not None:
            species_labels.append(current_label)
            zvals.append(current_zval)

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.lstrip()

            if stripped.startswith("TITEL"):
                finish_current_block()
                current_zval = None

                match = re.search(r"^\s*TITEL\s*=\s*(.*)$", line)
                title = match.group(1).strip() if match else stripped
                tokens = title.split()
                if len(tokens) < 2:
                    raise SystemExit(f"Cannot parse POTCAR TITEL line: {line.rstrip()}")
                current_label = tokens[1]
                continue

            if current_label is not None and "ZVAL" in line:
                match = re.search(rf"ZVAL\s*=\s*({number_pattern})", line)
                if match:
                    current_zval = float(match.group(1))

    finish_current_block()
    return species_labels, zvals


# ----------- ACF parsing -----------

def expand_by_counts(items: List, counts: List[int]) -> List:
    if len(items) != len(counts):
        raise SystemExit(
            f"Internal species/count mismatch: {len(items)} items and {len(counts)} counts."
        )

    expanded = []
    for item, count in zip(items, counts):
        expanded.extend([item] * count)
    return expanded


def read_acf_bader_electrons(path: str) -> List[float]:
    values: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or set(stripped) <= set("- "):
                continue

            parts = stripped.split()
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    values.append(float(parts[4]))
                except ValueError:
                    raise SystemExit(
                        f"Invalid CHARGE value in ACF.dat at line {line_number}: {parts[4]!r}"
                    )

    if not values:
        raise SystemExit(
            "No atomic CHARGE values were found in ACF.dat. "
            "Check the file format and Bader output."
        )

    return values


# ----------- Cross-file validation -----------

def validate_species(
    poscar_elements: List[str],
    poscar_counts: List[int],
    potcar_labels: List[str],
    potcar_zvals: List[Optional[float]],
) -> Tuple[List[str], List[float]]:
    if not potcar_labels:
        raise SystemExit(
            "No species were parsed from POTCAR. "
            "Make sure the file is complete and contains TITEL and ZVAL entries."
        )

    if len(poscar_counts) != len(potcar_labels):
        raise SystemExit(
            "Species-count mismatch between POSCAR and POTCAR: "
            f"POSCAR has {len(poscar_counts)} species, "
            f"POTCAR has {len(potcar_labels)} species."
        )

    if len(potcar_labels) != len(potcar_zvals):
        raise SystemExit("Internal POTCAR parsing error: species/ZVAL list lengths differ.")

    potcar_elements = [normalize_element_label(label) for label in potcar_labels]

    if poscar_elements:
        poscar_normalized = [normalize_element_label(label) for label in poscar_elements]
        if poscar_normalized != potcar_elements:
            pairs = ", ".join(
                f"{p}/{q}" for p, q in zip(poscar_normalized, potcar_elements)
            )
            raise SystemExit(
                "Species order mismatch between POSCAR and POTCAR. "
                f"POSCAR/POTCAR pairs: {pairs}"
            )
        species_names = poscar_normalized
    else:
        species_names = potcar_elements

    exact_zvals: List[float] = []
    for index, (label, zval) in enumerate(zip(potcar_labels, potcar_zvals), start=1):
        if zval is None:
            raise SystemExit(
                f"POTCAR species {index} ({label}) has no readable ZVAL entry. "
                "The script will not estimate ZVAL from VRHFIN."
            )
        exact_zvals.append(float(zval))

    return species_names, exact_zvals


# ----------- Main workflow -----------

def main() -> None:
    require_files(REQUIRED_INPUTS)

    print("Running: chgsum.pl AECCAR0 AECCAR2 ...")
    run_cmd(["chgsum.pl", "AECCAR0", "AECCAR2"])

    if not os.path.isfile("CHGCAR_sum"):
        raise SystemExit("chgsum.pl completed but CHGCAR_sum was not created.")

    time.sleep(3)

    print("Running: bader CHGCAR -ref CHGCAR_sum ...")
    run_cmd(["bader", "CHGCAR", "-ref", "CHGCAR_sum"])

    if not os.path.isfile(ACF):
        raise SystemExit("Bader completed but ACF.dat was not created.")

    print("Parsing and validating POSCAR/POTCAR ...")
    poscar_elements, poscar_counts = parse_poscar(POSCAR)
    potcar_labels, potcar_zvals = parse_potcar(POTCAR)
    species_names, zvals = validate_species(
        poscar_elements,
        poscar_counts,
        potcar_labels,
        potcar_zvals,
    )

    print("Detected species and exact POTCAR ZVAL values:")
    for name, label, zval in zip(species_names, potcar_labels, zvals):
        print(f"  {name}: POTCAR={label}, ZVAL={zval:g}")

    per_atom_elements = expand_by_counts(species_names, poscar_counts)
    per_atom_valence = expand_by_counts(zvals, poscar_counts)

    print("Reading ACF.dat ...")
    bader_electrons = read_acf_bader_electrons(ACF)

    natoms_poscar = len(per_atom_elements)
    natoms_acf = len(bader_electrons)
    if natoms_poscar != natoms_acf:
        raise SystemExit(
            "Atom-count mismatch between POSCAR and ACF.dat: "
            f"POSCAR has {natoms_poscar} atoms, "
            f"ACF.dat has {natoms_acf} atomic entries. "
            "No CSV was written."
        )

    total_valence = math.fsum(float(value) for value in per_atom_valence)
    total_bader = math.fsum(float(value) for value in bader_electrons)
    total_net_charge = total_valence - total_bader

    print("Charge sanity check:")
    print(f"  Total nominal valence electrons : {total_valence:.8f}")
    print(f"  Total Bader electrons           : {total_bader:.8f}")
    print(f"  Net Bader charge                : {total_net_charge:+.8f} e")
    print("  Compare the net value with the expected total charge of the calculation.")

    print(f"Writing {OUTCSV} ...")
    with open(OUTCSV, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Atom index",
                "Element",
                "POTCAR valence electrons (ZVAL)",
                "Bader electrons",
                "Bader net charge",
            ]
        )

        for index, (element, valence, electrons) in enumerate(
            zip(per_atom_elements, per_atom_valence, bader_electrons),
            start=1,
        ):
            valence_float = float(valence)
            electrons_float = float(electrons)
            bader_charge = valence_float - electrons_float
            writer.writerow(
                [
                    index,
                    element,
                    valence_float,
                    electrons_float,
                    bader_charge,
                ]
            )

    print("Done.")


if __name__ == "__main__":
    main()
