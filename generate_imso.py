#!/usr/bin/env python3
"""
generate_imso.py
================

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Incremental Multiple Superimposed Oscillators (IMSO) signal of the paper, saved in
datasets/IMSO.csv: 12 regimes of 1000 steps, the regime i (i = 1..12) is MSO_i, the sum of
the first i sinusoids sin(f_j t) with frequencies MSO_FREQS, t = 0, ..., 11999 (continuous
time across the regimes).

Generated, written and read exactly as in the code of the paper: the CSV is written by
pandas (columns date, value) and read by pandas (load), whose parser rounds some values in
the last digit; the signal read back is therefore identical bit by bit to the one of the
experiments of the paper (not to the formula, from which it differs by at most 2e-15).

Usage:
    python generate_imso.py                      # datasets/IMSO.csv
    python generate_imso.py --csv other.csv      # another file (the tests read datasets/IMSO.csv)

----

Luca Pedrelli
luca.pedrelli@santannapisa.it
lucapedrelli@gmail.com

The Telecommunications, Computer Engineering, and Photonics Institute (TeCIP)
Scuola Superiore Sant'Anna in Pisa
ICT-COISP (ICT for Complex Industrial Systems and Processes)
https://www.santannapisa.it/it/istituto/tecip

----
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "datasets", "IMSO.csv")

SEGMENT = 1000                                   # steps per regime
MSO_FREQS = [0.2, 0.311, 0.42, 0.51, 0.63, 0.74, 0.85, 0.97, 1.08, 1.19, 1.27, 1.32]


def generate():
    """IMSO signal, array (12000,): regime i = sum of the first i sinusoids."""
    t = np.arange(SEGMENT * len(MSO_FREQS), dtype=np.float64)
    u = np.zeros_like(t)
    for i in range(len(MSO_FREQS)):
        sl = slice(i * SEGMENT, (i + 1) * SEGMENT)
        u[sl] = sum(np.sin(f * t[sl]) for f in MSO_FREQS[:i + 1])
    return u


def save(u, path=CSV):
    """CSV as in the code of the paper (pandas, columns date, value)."""
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"date": np.arange(len(u)), "value": u}).to_csv(path, index=False)


def load(path=CSV):
    """signal of the CSV, array (T, 1), read as in the code of the paper (pandas)."""
    import pandas as pd
    return pd.read_csv(path).iloc[:, 1:].values.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV)
    a = ap.parse_args()
    u = generate()
    save(u, a.csv)
    print(f"{a.csv}: {len(u)} steps, {len(MSO_FREQS)} regimes of {SEGMENT}; max difference from "
          f"the formula after reading: {np.max(np.abs(load(a.csv)[:, 0] - u)):.1e}")


if __name__ == "__main__":
    main()
