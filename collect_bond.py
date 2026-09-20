#!/usr/bin/env python3
"""Count undirected periodic bonds per cell and report their mean length."""
import itertools
import sys
import numpy as np

# Edit element pairs and inclusive distance windows in angstrom here.
BOND_LIST = [('Mn', 'O', 1.65, 2.20)]


def read_poscar(path):
    with open(path) as handle:
        lines = handle.read().splitlines()
    scales = np.array([float(x) for x in lines[1].split()])
    raw = np.array([[float(x) for x in line.split()[:3]] for line in lines[2:5]])
    if len(scales) == 1:
        scale = scales[0]
        if scale == 0:
            raise ValueError('Scale cannot be zero.')
        factor = scale if scale > 0 else (abs(scale)/abs(np.linalg.det(raw)))**(1/3)
        scales = np.full(3, factor)
    elif len(scales) != 3 or np.any(scales <= 0):
        raise ValueError('Expected one nonzero or three positive scale factors.')
    lat = raw * scales
    symbols = lines[5].split()
    try:
        [int(x) for x in symbols]
    except ValueError:
        pass
    else:
        raise ValueError('Element labels are required; convert VASP4 input to VASP5 first.')
    counts = [int(x) for x in lines[6].split()]
    if len(counts) != len(symbols) or any(n < 0 for n in counts):
        raise ValueError('Invalid species/count list.')
    cursor = 7
    if lines[cursor].strip().lower().startswith('s'):
        cursor += 1
    mode = lines[cursor].strip().lower()[0]
    cursor += 1
    xyz = np.array([[float(x) for x in line.split()[:3]]
                    for line in lines[cursor:cursor+sum(counts)]])
    if xyz.shape != (sum(counts), 3):
        raise ValueError('Incomplete coordinates.')
    if mode in ('c', 'k'):
        frac = (xyz * scales) @ np.linalg.inv(lat)
    elif mode == 'd':
        frac = xyz
    else:
        raise ValueError('Unknown coordinate mode.')
    return {'comment': lines[0], 'lat': lat, 'frac': frac,
            'elements': [s for s, n in zip(symbols, counts) for _ in range(n)]}


def bond_stats(data, elem1, elem2, rmin, rmax):
    lat = np.asarray(data['lat'], float)
    frac = np.asarray(data['frac'], float) % 1.0
    ids1 = [i for i, e in enumerate(data['elements']) if e == elem1]
    ids2 = [i for i, e in enumerate(data['elements']) if e == elem2]
    if not ids1 or not ids2:
        raise ValueError(f'Missing element: {elem1} or {elem2}')
    rmin, rmax = sorted((rmin, rmax))
    if rmin < 0 or rmax <= 0:
        raise ValueError('Distance window must be nonnegative with positive upper bound.')
    # If |cart| <= rmax, each fractional component is bounded by the
    # corresponding column norm of the inverse lattice (Cauchy-Schwarz).
    reach = rmax * np.linalg.norm(np.linalg.inv(lat), axis=0)
    distances = []
    for i in ids1:
        for j in ids2:
            if elem1 == elem2 and j < i:
                continue
            df = frac[j]-frac[i]
            lower = np.ceil(-reach-df-1e-12).astype(int)
            upper = np.floor(reach-df+1e-12).astype(int)
            for shift in itertools.product(*(range(a,b+1) for a,b in zip(lower,upper))):
                if i == j and shift <= (0,0,0):
                    continue
                distance = float(np.linalg.norm((df+shift) @ lat))
                if rmin-1e-10 <= distance <= rmax+1e-10:
                    distances.append(distance)
    return len(distances), float(np.mean(distances)) if distances else float('nan')


def main():
    path = sys.argv[1] if len(sys.argv)>1 else 'CONTCAR'
    data = read_poscar(path)
    print('File:', path)
    print('Counts are undirected bonds per cell, including periodic images.')
    for e1,e2,rmin,rmax in BOND_LIST:
        count,mean = bond_stats(data,e1,e2,rmin,rmax)
        print(f'{e1}-{e2} [{rmin:g}, {rmax:g}] A: count={count}, mean={mean:.8f} A')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, IndexError, np.linalg.LinAlgError) as exc:
        raise SystemExit(f'ERROR: {exc}')
