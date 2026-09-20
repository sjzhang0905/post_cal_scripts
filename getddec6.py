#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

INFILE = Path("DDEC6_even_tempered_net_atomic_charges.xyz")
OUTFILE = Path("DDEC6_charges_simpleblock.csv")

def main():
    if not INFILE.exists():
        raise SystemExit(f"ERROR: cannot find {INFILE}")

    with INFILE.open("r", encoding="utf-8", errors="replace") as f:
        # Line 1: number of atoms
        first = f.readline()
        if not first:
            raise SystemExit("ERROR: empty file")

        try:
            n_atoms = int(first.strip().split()[0])
        except Exception:
            raise SystemExit(f"ERROR: cannot parse atom count from first line: {first!r}")

        # Line 2: comment (ignore)
        _ = f.readline()

        rows = []
        for i in range(1, n_atoms + 1):
            line = f.readline()
            if not line:
                raise SystemExit(f"ERROR: file ended early while reading atom {i}/{n_atoms}")

            parts = line.strip().split()
            if len(parts) < 5:
                raise SystemExit(f"ERROR: line {i+2} has <5 columns: {line!r}")

            elem = parts[0]          # 第1列：元素
            charge = float(parts[4]) # 第5列：DDEC6 电荷
            rows.append((i, elem, charge))

    # Write CSV
    with OUTFILE.open("w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["index", "element", "ddec6_charge"])
        w.writerows(rows)

    print(f"[OK] Wrote {OUTFILE} with {n_atoms} atoms (from the first block).")

if __name__ == "__main__":
    main()
