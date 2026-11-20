#!/usr/bin/env python3
"""
imso_test.py
============

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Test of RAM-ESN on the incremental Multiple Superimposed Oscillators task (IMSO) with
the online generative protocol of the paper. For each network (seeds 0, 1, ... of the
paper): the readout is trained on training + validation (one-step-ahead targets, after
the washout); then, at every step of the test stream, a 96-step generative rollout (the
model runs freely on its own predictions) followed by the update of the readout with the
true value (iNE or RLS; NE: offline readout, not updated). Score: MSE of the rollouts.

The model is a DeepESN (DeepESN Python Library, https://github.com/lucapedrelli/DeepESN,
version 1.2.0) with one RAMESNLayer of 100 units (ramesn_layer.py, defined on top of
deepesn.layers.Layer), or the ESN baseline. The hyperparameters are the ones selected in the paper (Bayesian
optimization on the validation stream), read from hyperparameters.csv: only the test is
run (the test MSE of the paper is printed next to each result).

Usage:
    python imso_test.py                    # main experiment: imso_ram1_iNE_t0.0001_mopt, RAM-ESN (IN-v),
                                           # iNE readout, m selected
    python imso_test.py --experiment imso_esn_iNE             # one experiment (--list for the names)
    python imso_test.py --experiment imso_esn_NE,imso_ram1_RLS_t0.0001_mopt   # more experiments
    python imso_test.py --all              # all the 33 IMSO experiments of hyperparameters.csv
    python imso_test.py --paper            # the 6 experiments of Table IV and of the figures of the
                                           # paper: the table (terminal and results/tables/table_imso.pdf)
                                           # and the figures (results/figures)
    python imso_test.py --list             # names of the IMSO experiments
Options for all the commands:
    --csv FILE        hyperparameters (default hyperparameters.csv)
    --results DIR     folder of the results (default results/)

Results (folder results/):
    imso_scores.csv                       test MSE of each network
    figures/IMSO.pdf                      the IMSO signal
    figures/<experiment>_rollout_first.pdf, _last.pdf
                                          mean rollout of the networks vs. ground truth on the
                                          first and on the last window of the test stream
                                          (not with --paper)
    figures/imso_rollouts_first.pdf, imso_rollouts_last.pdf
                                          figures of the paper (ESN and RAM-ESN, NE and iNE), with
                                          --paper or --all
    tables/table_imso.pdf                 Table IV of the paper, with --paper

Data: datasets/IMSO.csv (generate_imso.py), read as in the code of the paper (pandas).
Splits 8000 / 2000 /
2000 steps, standardized with the statistics of the training stream.

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
import numpy as np
from online_generative_forecasting import OnlineGenerativeForecasting

import generate_imso

# ------------------------------------------------------------------ model
# DeepESN Python Library (https://github.com/lucapedrelli/DeepESN). Please cite:
#   C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical
#   Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99.
from deepesn import DeepESN, Readout


def build_model(e, seed, n_inputs=1):
    """DeepESN with one recurrent layer, RAM-ESN (ramesn_layer.RAMESNLayer) or the ESN
    baseline (common.layer), and the readout of the experiment e (row of hyperparameters.csv)."""
    return DeepESN(n_inputs, layers=[common.layer(e)],
                   readout=Readout(e["readout"], regularization=float(e["lambda2"])), seed=seed)


HORIZON = 96                 # steps of each generative rollout
SPLITS = (8000, 2000, 2000)  # training, validation, test


def load_imso(path=generate_imso.CSV):
    """(U_train, U_val, U_test) of shape (T, 1), standardized on the training stream."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: python generate_imso.py")
    return common.split_standardize(generate_imso.load(path), SPLITS)


def make_task(washout):
    return OnlineGenerativeForecasting(*load_imso(), washout=washout, horizon=HORIZON, stride=1)


# ------------------------------------------------------------------ figures
T0 = SPLITS[0] + SPLITS[1]   # first step of the test stream


def label(e):
    return ("ESN" if e["ram"] == "0" else "RAM-ESN" if e["model"] == "IN-v" else f"RAM-ESN {e['model']}") \
        + f"-{e['readout']}"


def window(results, which):
    """mean rollout over the networks, true values and global steps of the first or last
    test window (window w predicts the steps T0 + w + 1, ..., T0 + w + HORIZON)."""
    k = 0 if which == "first" else -1
    pred = np.mean([np.asarray(r["pred"])[k][:, 0] for r in results], axis=0)
    true = np.asarray(results[0]["true"])[k][:, 0]
    w = 0 if which == "first" else len(results[0]["pred"]) - 1
    return pred, true, T0 + w + 1 + np.arange(len(true))


def plot_windows(panels, which, out):
    """one panel per experiment: ground truth and mean rollout on the first or last window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.3 * len(panels)), sharex=True)
    for ax, (e, results) in zip(np.atleast_1d(axes), panels):
        pred, true, steps = window(results, which)
        mse = float(np.mean((pred - true) ** 2))
        ax.plot(steps, true, color="steelblue", lw=1.8, alpha=0.85, label="ground truth")
        ax.plot(steps, pred, color="tomato", lw=1.5, alpha=0.9, label=f"{label(e)} (window MSE = {mse:.4f})")
        ax.set_ylabel(r"$u(t)$", fontsize=13)
        ax.set_xlim(steps[0], steps[-1])
        ax.set_xticks(np.round(np.linspace(steps[0], steps[-1], 5)).astype(int))
        ax.tick_params(labelbottom=True, labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                  borderaxespad=0.1)
    np.atleast_1d(axes)[-1].set_xlabel("time step", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_signal(out):
    """IMSO signal: whole signal with the splits and three segments of 1000 steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = generate_imso.load()[:, 0]
    T, B, t = len(u), generate_imso.SEGMENT, np.arange(len(u))
    tr, vl, ts = SPLITS
    zooms = [(0, r"(b) MSO$_1$, steps 0–1000 (training)"),
             (6000, r"(c) MSO$_7$, steps 6000–7000 (training)"),
             (T - B, r"(d) MSO$_{12}$, steps 11000–12000 (test)")]
    fig = plt.figure(figsize=(8, 7.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 0.62, 0.62, 0.62], hspace=0.8)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, u, color="steelblue", lw=0.6)
    for x0, _ in zooms:
        ax.axvspan(x0, x0 + B, color="gray", alpha=0.18, lw=0)
    ax.axvline(tr, color="black", ls="--", lw=1.2)
    ax.axvline(tr + vl, color="black", ls="--", lw=1.2)
    ytop = np.abs(u).max() * 1.12
    ax.set_ylim(-ytop, ytop * 1.25)
    for n in range(1, len(generate_imso.MSO_FREQS) + 1):
        ax.text((n - 0.5) * B, ytop * 1.2, rf"$\mathrm{{MSO}}_{{{n}}}$", ha="center", va="top",
                fontsize=10, color="gray")
    for x, s_ in ((tr / 2, "train"), (tr + vl / 2, "val"), (tr + vl + ts / 2, "test")):
        ax.text(x, -ytop * 0.97, s_, ha="center", va="bottom", fontsize=14)
    ax.set_xlim(0, T)
    ax.set_ylabel(r"$u(t)$", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_title("(a) IMSO signal", fontsize=13)
    for k, (x0, title) in enumerate(zooms):
        a = fig.add_subplot(gs[k + 1])
        sl = slice(x0, x0 + B)
        a.plot(t[sl], u[sl], color="steelblue", lw=0.7)
        a.set_xlim(x0, x0 + B)
        a.set_ylim(-ytop, ytop)
        a.set_yticks([-10, 0, 10])
        a.set_ylabel(r"$u(t)$", fontsize=12)
        a.tick_params(labelsize=10)
        a.grid(True, alpha=0.25)
        a.set_title(title, fontsize=12)
        if k == len(zooms) - 1:
            a.set_xlabel("time step", fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# experiments of the figures of the rollouts of the paper (first and last test window)
PAPER_FIGURES = {
    "imso_rollouts_first.pdf": ("first", ["imso_esn_NE", "imso_esn_iNE", "imso_ram1_NE_t0.0001_mopt",
                                          "imso_ram1_iNE_t0.0001_mopt"]),
    "imso_rollouts_last.pdf": ("last", ["imso_esn_iNE", "imso_ram1_NE_t0.0001_mopt",
                                        "imso_ram1_iNE_t0.0001_mopt"]),
}


def paper_figures(done, folder):
    """IMSO.pdf (signal) and imso_rollouts_first.pdf, imso_rollouts_last.pdf (figures of the
    paper) if their experiments were run."""
    os.makedirs(folder, exist_ok=True)
    files = [os.path.join(folder, "IMSO.pdf")]
    plot_signal(files[0])
    by_name = {e["experiment"]: (e, results) for e, _, results in done}
    for name, (which, names) in PAPER_FIGURES.items():
        if all(n in by_name for n in names):
            files.append(os.path.join(folder, name))
            plot_windows([by_name[n] for n in names], which, files[-1])
    return files


def figures(done, folder):
    """the figures of the paper (paper_figures) and, for each experiment,
    <experiment>_rollout_first.pdf and _last.pdf (first and last test window)."""
    files = paper_figures(done, folder)
    for e, _, results in done:
        for which in ("first", "last"):
            files.append(os.path.join(folder, f"{e['experiment']}_rollout_{which}.pdf"))
            plot_windows([(e, results)], which, files[-1])
    return files


# ------------------------------------------------------------------ Table IV of the paper
TABLE = [("ESN-NE", "imso_esn_NE"), ("ESN-iNE", "imso_esn_iNE"), ("ESN-RLS", "imso_esn_RLS"),
         ("RAM-ESN-NE", "imso_ram1_NE_t0.0001_mopt"), ("RAM-ESN-iNE", "imso_ram1_iNE_t0.0001_mopt"),
         ("RAM-ESN-RLS", "imso_ram1_RLS_t0.0001_mopt")]
PAPER = [n for _, n in TABLE]                    # also the experiments of the figures


def paper(done, results):
    """Table IV (test MSE and MAE, mean +- std over the networks) and the figures of the paper."""
    by_name = {e["experiment"]: (e, v, r) for e, v, r in done}
    rows, mse, mae = [], [], []
    for i, (label_, n) in enumerate(TABLE):
        e, v, r = by_name[n]
        d = common.summary(v)
        a = common.summary([x["mae"] for x in r]) if all("mae" in x for x in r) else None
        if d[0] > common.DIVERGES:
            rows.append([label_, e["m"] or "-", "diverges", "diverges"])
            continue
        rows.append([label_, e["m"] or "-", common.pm(*d, 3), "-" if a is None else common.pm(*a, 3)])
        mse.append(((i, 2), d))
        if a is not None:
            mae.append(((i, 3), a))
    bold = common.best(mse, 3) | common.best(mae, 3)
    files = [common.report_table(os.path.join(results, "tables", "table_imso.pdf"),
                                 "Table IV. IMSO: test errors of the generative rollout (H = 96), "
                                 "mean \u00b1 std over 5 networks",
                                 ["Model", "m", "MSE", "MAE"], rows, bold,
                                 "RAM-ESN: tau = 1e-4, m selected by Bayesian optimization. 'diverges': the "
                                 "generative rollout diverges for at least one of the 5 networks. Best in bold.")]
    return files + paper_figures(done, os.path.join(results, "figures"))


if __name__ == "__main__":
    common.main("IMSO", "imso_ram1_iNE_t0.0001_mopt", make_task, "mse", __doc__, build_model, figures,
                paper=(PAPER, paper), paper_help="the 6 experiments of Table IV and of the figures of the "
                                                 "paper: the table and the figures")
