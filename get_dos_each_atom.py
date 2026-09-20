#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VASPKIT PDOS extraction for selected atom indices."
    )
    parser.add_argument(
        "atoms",
        nargs="+",
        type=int,
        help="Atom indices to process, for example: 153 154 155",
    )
    parser.add_argument(
        "--vaspkit",
        default="vaspkit",
        help="VASPKIT executable name or full path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing atom-resolved DOS files",
    )
    return parser.parse_args()


def check_executable(executable: str) -> None:
    if Path(executable).is_file():
        return
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"VASPKIT executable not found: {executable}"
        )


def remove_temporary_outputs() -> None:
    for filename in ("PDOS_SUM_UP.dat", "PDOS_SUM_DW.dat", "PDOS_SUM.dat"):
        path = Path(filename)
        if path.exists():
            path.unlink()


def run_vaspkit(executable: str, atom_index: int) -> None:
    input_text = f"114\n1\n{atom_index}\n"

    result = subprocess.run(
        [executable],
        input=input_text,
        text=True,
        stdout=None,
        stderr=None,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"VASPKIT failed for atom {atom_index} "
            f"with return code {result.returncode}"
        )


def rename_outputs(atom_index: int, overwrite: bool) -> list[str]:
    spin_output_map = {
        Path("PDOS_SUM_UP.dat"): Path(f"{atom_index}_UP.dat"),
        Path("PDOS_SUM_DW.dat"): Path(f"{atom_index}_DW.dat"),
    }
    nonspin_output_map = {
        Path("PDOS_SUM.dat"): Path(f"{atom_index}.dat"),
    }

    spin_sources = list(spin_output_map.keys())
    if all(source.is_file() for source in spin_sources):
        output_map = spin_output_map
    elif Path("PDOS_SUM.dat").is_file():
        output_map = nonspin_output_map
    else:
        existing_sources = [
            source.name
            for source in (*spin_sources, Path("PDOS_SUM.dat"))
            if source.is_file()
        ]
        details = (
            f" Found only: {', '.join(existing_sources)}."
            if existing_sources
            else ""
        )
        raise FileNotFoundError(
            f"VASPKIT did not generate expected PDOS output for atom "
            f"{atom_index}. Expected either PDOS_SUM_UP.dat and "
            f"PDOS_SUM_DW.dat, or PDOS_SUM.dat.{details}"
        )

    existing = [
        str(target)
        for target in output_map.values()
        if target.exists() and not overwrite
    ]
    if existing:
        raise FileExistsError(
            "Output file already exists. Use --overwrite to replace it: "
            + ", ".join(existing)
        )

    saved = []
    for source, target in output_map.items():
        if target.exists():
            target.unlink()
        source.rename(target)
        saved.append(str(target))

    return saved


def main() -> int:
    args = parse_args()

    try:
        check_executable(args.vaspkit)

        total = len(args.atoms)
        for position, atom_index in enumerate(args.atoms, start=1):
            if atom_index <= 0:
                raise ValueError(
                    f"Atom index must be a positive integer: {atom_index}"
                )

            print(
                f"[{position}/{total}] Processing atom {atom_index}",
                flush=True,
            )

            remove_temporary_outputs()
            run_vaspkit(args.vaspkit, atom_index)
            saved_files = rename_outputs(atom_index, args.overwrite)

            print(
                f"Saved: {', '.join(saved_files)}",
                flush=True,
            )

        print("All requested atoms have been processed.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
