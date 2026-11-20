#!/usr/bin/env python3
"""
mc_test.py
==========

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Test of RAM-ESN on the Memory Capacity task (MC) of the paper. For each network (seeds
0, 1, ... of the paper): the model is driven by the input of training + validation +
test (from the zero state); the readout (NE) is trained on training + validation (after
the washout) to reconstruct the input delayed by k = 1..200 steps; on the test segment,
MC = sum over k of the squared correlation between the delayed input and its
reconstruction.

The model is a DeepESN (DeepESN Python Library, https://github.com/lucapedrelli/DeepESN,
version 1.2.0) with one RAMESNLayer of 100 units (ramesn_layer.py), or the ESN
baseline. The hyperparameters are
the ones selected in the paper (Bayesian optimization on the validation segment), read
from hyperparameters.csv: only the test is run (the test MC of the paper is printed next
to each result). Test MC of each network: results/mc_scores.csv.

Usage:
    python mc_test.py                      # main experiment: mc_ram1_NE_t0.0001_mopt, RAM-ESN (IN-v),
                                           # tau = 1e-4, m selected
    python mc_test.py --experiment mc_esn_NE                  # one experiment (--list for the names)
    python mc_test.py --experiment mc_esn_NE,mc_ram1_NE_t0.1_m100   # more experiments
    python mc_test.py --all                # all the 26 MC experiments of hyperparameters.csv
    python mc_test.py --paper              # the 17 experiments of Table II of the paper, and the
                                           # table (terminal and results/tables/table_mc.pdf)
    python mc_test.py --list               # names of the MC experiments
Options for all the commands:
    --csv FILE        hyperparameters (default hyperparameters.csv)
    --results DIR     folder of the results (default results/)

Data: datasets/MC.csv (generate_mc.py; generated if missing): i.i.d. input uniform in
(-0.8, 0.8), 5000 / 1000 / 1000 steps.

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
import os

import common                              # first: one BLAS thread

import generate_mc


# ------------------------------------------------------------------ model
# DeepESN Python Library (https://github.com/lucapedrelli/DeepESN). Please cite:
#   C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical
#   Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99.
from deepesn import DeepESN, Readout
from deepesn.tasks import MemoryCapacity


def build_model(e, seed, n_inputs=1):
    """DeepESN with one recurrent layer, RAM-ESN (ramesn_layer.RAMESNLayer) or the ESN
    baseline (common.layer), and the readout of the experiment e (row of hyperparameters.csv)."""
    return DeepESN(n_inputs, layers=[common.layer(e)],
                   readout=Readout(e["readout"], regularization=float(e["lambda2"])), seed=seed)


def load_mc(path=generate_mc.CSV):
    """(U_train, U_val, U_test), arrays (T, 1)."""
    return generate_mc.load(path) if os.path.exists(path) else generate_mc.generate()


def make_task(washout):
    return MemoryCapacity(*load_mc(), washout=washout, max_delay=generate_mc.MAX_DELAY)


# ------------------------------------------------------------------ Table II of the paper
TAUS = [("1.0", "1"), ("0.1", "0.1"), ("0.001", "0.001"), ("0.0001", "1e-4")]
MS = [200, 150, 100, 50]
PAPER = ["mc_esn_NE"] + [f"mc_ram1_NE_t{t}_m{m}" for t, _ in TAUS for m in MS]


def table(done, results):
    """Table II: test MC (mean +- std over the networks), ESN and RAM-ESN for each tau and m."""
    S = {e["experiment"]: common.summary(v) for e, v, _ in done}
    header = ["Model"] + [f"m={m}" for m in MS]
    rows = [["ESN"] + [common.pm(*S["mc_esn_NE"], 2)] + [""] * (len(MS) - 1)]
    cells = []
    for i, (t, tl) in enumerate(TAUS, start=1):
        row = [f"RAM-ESN, tau={tl}"]
        for j, m in enumerate(MS, start=1):
            d = S[f"mc_ram1_NE_t{t}_m{m}"]
            row.append(common.pm(*d, 2))
            cells.append(((i, j), d))
        rows.append(row)
    bold = common.best(cells, 2, maximize=True)
    return [common.report_table(os.path.join(results, "tables", "table_mc.pdf"),
                                "Table II. Memory Capacity: test MC (mean \u00b1 std over 5 networks), NE readout",
                                header, rows, bold,
                                "Best configuration in bold (on ties, the smallest std); ESN: no memory window.")]


if __name__ == "__main__":
    common.main("MC", "mc_ram1_NE_t0.0001_mopt", make_task, "mc", __doc__, build_model, paper=(PAPER, table),
                paper_help="the 17 experiments of Table II of the paper and the table")
