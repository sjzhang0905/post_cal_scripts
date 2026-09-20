#!/usr/bin/env python3
"""Estimate a sampled band gap from EIGENVAL occupations, including both spins."""
import argparse
from pathlib import Path
import re
import numpy as np


def read_eigenval(path, with_metadata=False):
    lines = Path(path).read_text().splitlines()
    header = lines[5].split()
    nelect, nk, nb = float(header[0]), int(header[1]), int(header[2])
    blocks = []
    kpoints, weights = [], []
    cursor = 6
    for _ in range(nk):
        while not lines[cursor].strip():
            cursor += 1
        point = [float(x) for x in lines[cursor].split()]
        kpoints.append(point[:3])
        weights.append(point[3])
        cursor += 1
        rows = []
        for _ in range(nb):
            row = [float(x) for x in lines[cursor].split()[1:]]
            if len(row) not in (2, 4):
                raise ValueError('Unsupported EIGENVAL band columns.')
            rows.append(row)
            cursor += 1
        blocks.append(rows)
    data = np.asarray(blocks)
    if data.shape[2] == 4:
        result = (data[:, :, :2], data[:, :, 2:], 1.0)
    else:
        result = (data[:, :, :1], data[:, :, 1:], None)
    if with_metadata:
        return (*result, {"nelect": nelect, "weights": np.asarray(weights),
                          "kpoints": np.asarray(kpoints)})
    return result


def infer_capacity(occupation, metadata, outcar_text="", override=None):
    """Distinguish stored occupation normalization from physical spin degeneracy.

    Scalar occupations may be stored on a 0..1 or 0..2 scale. For a
    weighted SCF mesh, NELECT = degeneracy * weighted_sum / capacity.
    Small Methfessel-Paxton overshoots do not change this normalization.
    """
    nspin = occupation.shape[2]
    if nspin == 2:
        if override not in (None, 1.):
            raise ValueError('Two-spin EIGENVAL requires full occupation 1 per spin.')
        return 1., 'two explicit spin channels'
    if override is not None:
        return override, 'explicit stored-occupation normalization'
    spinor = bool(re.search(r'(?:LSORBIT|LNONCOLLINEAR)\s*=\s*\.?T', outcar_text, re.I))
    scalar = bool(re.search(r'LNONCOLLINEAR\s*=\s*\.?F', outcar_text, re.I)) and not spinor
    weights = metadata['weights']
    nelect = metadata['nelect']
    if np.any(weights < 0) or nelect <= 0:
        raise ValueError('Invalid weights or electron count.')
    # Only a unit-normalized mesh (zero-weight path points are allowed)
    # can supply this electron-count cross-check.
    if abs(float(weights.sum())-1.) < 1e-4:
        total = float(np.dot(weights, occupation.sum(axis=(1,2)))/weights.sum())
        candidates = [(1.,1.)] if spinor else [(1.,2.),(2.,2.)]
        if not scalar and not spinor:
            candidates.append((1.,1.))
        matches = [(cap,deg) for cap,deg in candidates
                   if abs(deg*total/cap-nelect) <= max(1e-3,nelect*1e-5)]
        # Exclude a unit-normalized interpretation only for substantial
        # values above 1; MP overshoots near a band edge are not decisive.
        if np.max(occupation) > 1.2:
            matches = [(cap,deg) for cap,deg in matches if cap == 2.]
        if len(matches) == 1:
            cap,deg = matches[0]
            return cap, f'weighted occupations + NELECT (spin degeneracy {deg:g})'
    if spinor:
        return 1., 'spinor calculation'
    # On line paths, deep filled bands establish a plateau; this is not
    # inferred from a single maximum that may carry a smearing overshoot.
    plateau = np.median(occupation,axis=0).ravel()
    if np.any(np.abs(plateau-2.) < 1e-3):
        return 2., 'full-band occupation plateau near 2'
    if np.any(np.abs(plateau-1.) < 1e-3):
        # Without electron-count evidence, scalar half filling is ambiguous.
        if not scalar:
            raise ValueError('Occupation normalization is ambiguous. Supply matching OUTCAR '
                             'or --max-occupation 1/2 for the values stored in EIGENVAL.')
        # A full band near 1 with known scalar spin degeneracy is not enough
        # to distinguish a normalized band from an exactly half-filled band.
        raise ValueError('Cannot verify occupation normalization on this k sampling. '
                         'Use --max-occupation 1 or 2 for the stored occupation scale.')
    raise ValueError('Cannot infer full occupation from this data. Use --max-occupation 1 or 2.')


def analyze(energy, occupation, capacity, occ_tol=0.001, energy_tol=1e-5):
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(occupation)):
        raise ValueError('Non-finite energies or occupations.')
    occupied = occupation > capacity / 2
    # A fixed band/spin changing occupation along k is a sampled crossing.
    crossing = np.any(np.any(occupied, axis=0) & ~np.all(occupied, axis=0))
    if crossing:
        return 'metallic', 0.0, None, None
    partial = (occupation > capacity * occ_tol) & (occupation < capacity * (1-occ_tol))
    if np.any(partial):
        return 'indeterminate_partial_occupations', None, None, None
    if not np.any(occupied) or np.all(occupied):
        raise ValueError('Both occupied and empty states are required; check NBANDS and occupations.')
    vbm = float(np.max(energy[occupied]))
    cbm = float(np.min(energy[~occupied]))
    gap = cbm-vbm
    if gap <= energy_tol:
        return 'metallic_or_zero_gap', 0.0, vbm, cbm
    return 'gapped_on_sampled_kpoints', gap, vbm, cbm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='EIGENVAL')
    parser.add_argument('--outcar', default='OUTCAR')
    parser.add_argument('--max-occupation', type=float, choices=(1., 2.),
                        help='Full occupation as stored in EIGENVAL, not spin degeneracy.')
    parser.add_argument('--occupation-tol', type=float, default=0.001,
                        help='Fraction of full occupation; partial states make the result indeterminate.')
    args = parser.parse_args()
    if not 0 < args.occupation_tol < 0.5:
        parser.error('--occupation-tol must be between 0 and 0.5.')
    energy, occupation, _, metadata = read_eigenval(args.input, with_metadata=True)
    text = Path(args.outcar).read_text(errors='replace') if Path(args.outcar).is_file() else ''
    capacity, source = infer_capacity(occupation, metadata, text, args.max_occupation)
    print(f'Stored full occupation: {capacity:g}; source: {source}')
    status, gap, vbm, cbm = analyze(energy, occupation, capacity, args.occupation_tol)
    print('Status:', status)
    if gap is None:
        print('Eg = undetermined: partial occupations may indicate a metal or excessive smearing.')
    else:
        print(f'Eg = {gap:.8f} eV')
    if vbm is not None:
        print(f'VBM = {vbm:.8f} eV; CBM = {cbm:.8f} eV (EIGENVAL reference)')
    if gap is not None and gap > 0:
        occupied = occupation > capacity/2
        vi = np.unravel_index(np.argmax(np.where(occupied, energy, -np.inf)), energy.shape)
        ci = np.unravel_index(np.argmin(np.where(~occupied, energy, np.inf)), energy.shape)
        for label, index in (('VBM',vi),('CBM',ci)):
            coords = ' '.join(f'{v:.7f}' for v in metadata['kpoints'][index[0]])
            print(f'{label}: band {index[1]+1}, spin {index[2]+1}, k(frac) = {coords}')
        delta = metadata['kpoints'][vi[0]]-metadata['kpoints'][ci[0]]
        same_k = np.linalg.norm(delta-np.round(delta)) < 1e-6
        # Equivalent extrema may be degenerate: test the entire sampled set.
        valence = np.max(np.where(occupied,energy,-np.inf),axis=(1,2))
        conduction = np.min(np.where(~occupied,energy,np.inf),axis=(1,2))
        vertical_gap = float(np.min(conduction-valence))
        direct = abs(vertical_gap-gap) <= 1e-5 or same_k
        print('Gap character on sampled mesh:', 'direct' if direct else 'indirect')
        print(f'Minimum same-k gap = {vertical_gap:.8f} eV (not an optical selection-rule test)')
    print('Scope: supplied k points only; a line path does not establish a full-zone gap.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, IndexError) as exc:
        raise SystemExit(f'ERROR: {exc}')
