"""
common.py
=========

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Parts shared by the test scripts (mc_test.py, imso_test.py, real_test.py): hyperparameters of the experiments of
the paper (hyperparameters.csv), model (DeepESN with
one RAMESNLayer or the ESN baseline) and test of an experiment on the networks of the
paper (seeds 0, 1, ...), with the bit-by-bit comparison of the test scores.

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
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")         # one BLAS thread: results identical bit by bit

import argparse
import csv
import time

import numpy as np

# DeepESN Python Library (https://github.com/lucapedrelli/DeepESN). Please cite:
#   C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical
#   Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99.
from deepesn.layers import ESNLayer, LiESNLayer

from ramesn_layer import RAMESNLayer

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "hyperparameters.csv")
RESULTS = os.path.join(HERE, "results")

UNITS = 100                  # reservoir units (one layer)
MAX_MEMORY = 500             # attention logits drawn per neuron


def read_experiments(dataset, path=CSV):
    """rows of hyperparameters.csv of the dataset, by experiment name."""
    with open(path, newline="") as f:
        return {r["experiment"]: r for r in csv.DictReader(f) if r["dataset"] == dataset}


def layer(e):
    """recurrent layer of UNITS units for the experiment e (row of the CSV):
    RAMESNLayer (ram 1..6: IN-v, IN-c, OUT-v, OUT-c, LI-v, LI-c) or the ESN baseline
    (LiESNLayer, or ESNLayer if there is no leaking rate)."""
    num = lambda k: float(e[k])
    common = dict(spectral_radius=num("rho"), input_scaling=num("sigma_in"))
    ram = int(e["ram"])
    if ram == 0:
        return LiESNLayer(UNITS, leaking_rate=num("leaky"), **common) if e["leaky"] \
            else ESNLayer(UNITS, **common)
    return RAMESNLayer(UNITS, memory=int(e["m"]), temperature=num("tau"), variant=ram,
                       max_memory=MAX_MEMORY, **common)


def split_standardize(data, splits):
    """(U_train, U_val, U_test) of the chronological splits (rows) of data (T, n_inputs),
    standardized with the statistics of the training stream (training clipped to +-10)."""
    ntr, nval, nte = splits
    data = data[:ntr + nval + nte]
    mu = data[:ntr].mean(axis=0, keepdims=True)
    sd = data[:ntr].std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    norm = lambda x: (x - mu) / sd
    return (np.clip(norm(data[:ntr]), -10, 10), norm(data[ntr:ntr + nval]),
            norm(data[ntr + nval:]))


def describe(e):
    hp = ", ".join(f"{k} = {float(e[k]):.6g}" for k in ("rho", "sigma_in", "lambda2", "leaky") if e[k])
    name = "ESN" if e["ram"] == "0" else f"RAM-ESN {e['model']}"
    mem = f", m = {e['m']}, tau = {e['tau']}" if e["ram"] != "0" else ""
    return f"{name}, readout {e['readout']}{mem}, {hp}"


def run(e, task, key, build_model, n_inputs=1):
    """test of the experiment e on the networks of the paper (seeds 0, 1, ...).
    Returns the test score (result[key]: 'mse' or 'mc') of each network and the results
    of the task of each network (with the predictions)."""
    ref = [float(x) for x in e["test_scores"].split(";")]
    print(f"\n{e['experiment']}: {describe(e)}", flush=True)
    vals, results = [], []
    for i, s in enumerate(int(x) for x in e["seeds"].split(";")):
        t0 = time.time()
        r = task.evaluate(build_model(e, s, n_inputs), "test")
        vals.append(r[key])
        results.append(r)
        print(f"   seed {s}: test {e['metric']} {r[key]:.6g}   (paper {ref[i]:.6g})   "
              f"{time.time() - t0:.1f} s", flush=True)
    vals = np.array(vals)
    print(f"   test {e['metric']} {vals.mean():.6g} +- {vals.std():.6g}   "
          f"(paper {float(e['test_mean']):.6g} +- {float(e['test_std']):.6g})", flush=True)
    return vals, results


def save_scores(path, done):
    """CSV of the test scores: one row per (experiment, seed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "model", "readout", "tau", "m", "seed", "metric", "test"])
        for e, vals, _ in done:
            for s, v in zip(e["seeds"].split(";"), vals):
                w.writerow([e["experiment"], e["model"], e["readout"], e["tau"], e["m"], s, e["metric"],
                            repr(float(v))])


# ------------------------------------------------------------------ tables of the paper
DIVERGES = 100.0             # test MSE above this value: the rollout diverges for some network


def summary(vals):
    """(mean, std) of the test scores of the networks."""
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std())


def pm(mean, std, dec):
    return f"{mean:.{dec}f} \u00b1 {std:.{dec}f}"


def best(cells, dec, maximize=False):
    """indexes of the best cells (mean, std) at the printed precision; on ties, the
    smallest std."""
    if not cells:
        return set()
    key = lambda c: round(c[1][0], dec) * (-1 if maximize else 1)
    top = min(key(c) for c in cells)
    ties = [c for c in cells if key(c) == top]
    s_min = min(round(c[1][1], dec) for c in ties)
    return {c[0] for c in ties if round(c[1][1], dec) == s_min}


def print_table(title, header, rows, bold=(), note=None):
    """the table in the terminal (bold cells marked with *)."""
    cells = [[(f"*{v}" if (i, j) in bold else v) for j, v in enumerate(r)] for i, r in enumerate(rows)]
    w = [max(len(str(x)) for x in col) for col in zip(header, *cells)]
    line = "-" * (sum(w) + 2 * (len(w) - 1))
    print(f"\n{title}\n{line}")
    print("  ".join(f"{h:<{w[j]}}" if j == 0 else f"{h:>{w[j]}}" for j, h in enumerate(header)))
    print(line)
    for r in cells:
        print("  ".join(f"{v:<{w[j]}}" if j == 0 else f"{v:>{w[j]}}" for j, v in enumerate(r)))
    print(line + ("" if not note else "\n" + note))


def table_pdf(path, title, header, rows, bold=(), note=None):
    """the table in a PDF (bold cells in bold)."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(header)), 0.32 * (len(rows) + 1) + 1.0))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.auto_set_column_width(list(range(len(header))))     # columns as wide as their text
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0 or (r - 1, c) in bold:
            cell.get_text().set_fontweight("bold")
        if c == 0:
            cell.get_text().set_horizontalalignment("left")
    ax.set_title(title, fontsize=10)
    if note:
        fig.text(0.02, 0.0, textwrap.fill(note, 130), fontsize=7, va="top")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def report_table(path, title, header, rows, bold=(), note=None):
    """the table in the terminal and in a PDF; returns the PDF."""
    print_table(title, header, rows, bold, note)
    return table_pdf(path, title, header, rows, bold, note)


# ------------------------------------------------------------------ command line
def run_experiments(E, names, make_task, key, build_model, n_inputs=1):
    """test of the experiments names (rows of E): list of (row, scores, results)."""
    for n in names:
        if n not in E:
            raise SystemExit(f"{n} not in {CSV} (--list)")
    return [(E[n],) + run(E[n], make_task(int(E[n]["washout"])), key, build_model, n_inputs)
            for n in names]


def report_files(title, files):
    print(f"{title}:")
    for f in files:
        print("   " + f)


def arguments(dataset, default, doc, paper_help, extra=None):
    """options of the test scripts: --experiment (one or more models), --all, --paper, --list."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if extra is not None:
        extra(ap)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--experiment", default=None,
                   help=f"experiment(s) of the CSV, comma separated (default {default})")
    g.add_argument("--all", action="store_true", help="all the experiments of the CSV")
    g.add_argument("--paper", action="store_true", help=paper_help)
    g.add_argument("--list", action="store_true", help="list the experiments of the CSV")
    ap.add_argument("--csv", default=CSV, help="hyperparameters of the experiments (default hyperparameters.csv)")
    ap.add_argument("--results", default=RESULTS, help="folder of the results (default results/)")
    return ap.parse_args()


def list_experiments(E):
    for k, e in E.items():
        print(f"{k:34s} {e['model']:6s} {e['readout']:4s} tau={e['tau'] or '-':7s} m={e['m'] or '-':4s} "
              f"test {e['metric']} (paper) {float(e['test_mean']):.5g}")


def main(dataset, default, make_task, key, doc, build_model, figures=None, n_inputs=1, paper=None,
         paper_help="experiments of the paper: table (and figures)"):
    """command line of the test scripts of one dataset:
        (no option)          the default experiment
        --experiment a,b     the experiments a, b (one or more models)
        --all                all the experiments of the dataset
        --paper              the experiments of the paper: paper = (names, report), with
                             report(done, results folder) -> files written (table, figures)
        --list               list the experiments
    build_model(e, seed, n_inputs): the DeepESN model of the experiment e (row of the CSV).
    n_inputs: number of input channels of the dataset.
    figures(done, folder): figures of the experiments done (list of (row, scores, results)),
    saved in results/figures; returns the files written (not with --paper)."""
    a = arguments(dataset, default, doc, paper_help)
    E = read_experiments(dataset, a.csv)
    if a.list:
        list_experiments(E)
        return
    if a.paper and paper is None:
        raise SystemExit(f"no --paper for {dataset}")
    names = paper[0] if a.paper else list(E) if a.all else (a.experiment or default).split(",")
    done = run_experiments(E, names, make_task, key, build_model, n_inputs)

    scores = os.path.join(a.results, f"{dataset.lower()}_scores.csv")
    save_scores(scores, done)
    print(f"\n{'=' * 80}\ntest {done[0][0]['metric']} of each network: {scores}")
    if a.paper:
        report_files("PAPER: table and figures", paper[1](done, a.results))
    elif figures is not None:
        report_files("FIGURES: look at them in " + os.path.join(a.results, "figures"),
                     figures(done, os.path.join(a.results, "figures")))
    print("=" * 80)
