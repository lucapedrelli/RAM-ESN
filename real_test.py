#!/usr/bin/env python3
"""
real_test.py
============

Reference paper of RAM-ESN (please cite):
L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative
Reservoir Computing", 2026.

Test of RAM-ESN on the real-world forecasting benchmarks of the paper (ETTm1, ETTm2,
Weather) with the online generative protocol. For each network (seeds 0, 1, ... of the
paper): the readout is trained on training + validation (one-step-ahead targets, after
the washout); then, at every step of the test stream, a 96-step generative rollout of all
the channels (the model runs freely on its own predictions) followed by the update of the
readout with the true values (iNE or RLS; NE: offline readout, not updated). Score: MSE of
the rollouts.

The model is a DeepESN (DeepESN Python Library, https://github.com/lucapedrelli/DeepESN,
version 1.2.0) with one RAMESNLayer of 100 units (ramesn_layer.py), or the ESN
baseline. The hyperparameters are
the ones selected in the paper (Bayesian optimization on the validation stream), read
from hyperparameters.csv: only the test is run (the test MSE of the paper is printed next
to each result).

Usage:
    python real_test.py                    # main experiment on ETTm1: ettm1_ram1_iNE_t0.0001_mopt,
                                           # RAM-ESN (IN-v), iNE readout, m selected
    python real_test.py --dataset ETTm2    # main experiment on ETTm2 (datasets: ETTm1, ETTm2, weather)
    python real_test.py --dataset weather --experiment weather_esn_iNE   # one experiment (--list)
    python real_test.py --dataset ETTm1 --experiment ettm1_esn_NE,ettm1_ram1_RLS_t0.0001_mopt
                                           # more experiments
    python real_test.py --dataset ETTm1 --all   # all the 33 experiments of the dataset
    python real_test.py --paper            # the 18 experiments of Table VI of the paper (6 for each
                                           # dataset, all the three datasets), and the table
                                           # (terminal and results/tables/table_real.pdf)
    python real_test.py --dataset ETTm1 --list  # names of the experiments of the dataset
Options for all the commands:
    --dataset NAME    ETTm1 (default), ETTm2 or weather (not used by --paper)
    --csv FILE        hyperparameters (default hyperparameters.csv)
    --results DIR     folder of the results (default results/)

Results: results/<dataset>_scores.csv, test MSE of each network.

Data (not included, from the original sources), in datasets/:
    ETTm1.csv, ETTm2.csv   https://github.com/zhouhaoyi/ETDataset (folder ETT-small)
    weather.csv            https://github.com/thuml/Time-Series-Library (Weather dataset)
All the channels except the date column; standard chronological splits (Informer / TSLib),
standardized with the statistics of the training stream.

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
DATA = os.path.join(common.HERE, "datasets")

# dataset -> (file, (training, validation, test) rows)
DATASETS = {
    "ETTm1":   ("ETTm1.csv",   (34560, 11520, 11520)),
    "ETTm2":   ("ETTm2.csv",   (34560, 11520, 11520)),
    "weather": ("weather.csv", (36887, 5270, 10539)),
}


def load_real(dataset):
    """(U_train, U_val, U_test) of shape (T, n_channels), standardized on the training stream."""
    from pandas import read_csv
    file, splits = DATASETS[dataset]
    path = os.path.join(DATA, file)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: download {file} from its original source (see the README)")
    data = read_csv(path).iloc[:, 1:].values.astype(np.float64)     # drop the date column
    return common.split_standardize(data, splits)


# ------------------------------------------------------------------ Table VI of the paper
RC = [("RAM-ESN-iNE", "ram1_iNE_t0.0001_mopt"), ("RAM-ESN-RLS", "ram1_RLS_t0.0001_mopt"),
      ("RAM-ESN-NE", "ram1_NE_t0.0001_mopt"), ("ESN-iNE", "esn_iNE"), ("ESN-RLS", "esn_RLS"),
      ("ESN-NE", "esn_NE")]
# published test MSE of the offline deep learning models (input length 96, predict 96), from
# S. Wang et al., TimeMixer++, ICLR 2025: ETTm1, ETTm2, Weather
DEEP = [("TimeMixer++", 0.310, 0.170, 0.155), ("TimeMixer", 0.320, 0.175, 0.163),
        ("iTransformer", 0.334, 0.180, 0.174), ("PatchTST", 0.352, 0.183, 0.186),
        ("TimesNet", 0.338, 0.187, 0.172), ("DLinear", 0.346, 0.193, 0.195),
        ("Crossformer", 0.404, 0.287, 0.195), ("TiDE", 0.364, 0.207, 0.202),
        ("FEDformer", 0.379, 0.203, 0.217), ("Autoformer", 0.505, 0.255, 0.266)]


def table(S, results):
    """Table VI: test MSE (mean +- std over the networks) of the online generative RC models
    on the three datasets (S[dataset][experiment] = (mean, std)) and the published results of
    the deep learning models. Bold: best RC model per dataset and, when better, the best deep
    learning model."""
    rows, bold = [], set()
    for i, (label, n) in enumerate(RC):
        row = [label]
        for ds in DATASETS:
            d = S[ds][f"{ds.lower()}_{n}"]
            row.append("diverges" if d[0] > common.DIVERGES else common.pm(*d, 3))
        rows.append(row)
    for j, ds in enumerate(DATASETS, start=1):
        rc = [((i, j), S[ds][f"{ds.lower()}_{n}"]) for i, (_, n) in enumerate(RC)
              if S[ds][f"{ds.lower()}_{n}"][0] <= common.DIVERGES]
        b = common.best(rc, 3)
        bold |= b
        rc_best = min(round(d[0], 3) for _, d in rc)
        deep = [((len(RC) + k, j), (r[j], 0.0)) for k, r in enumerate(DEEP)]
        bd = common.best(deep, 3)
        if min(round(d[0], 3) for c, d in deep if c in bd) < rc_best:
            bold |= bd
    rows += [[name] + [f"{v:.3f}" for v in vals] for name, *vals in DEEP]
    return [common.report_table(os.path.join(results, "tables", "table_real.pdf"),
                                "Table VI. Online generative forecasting (H = 96): test MSE",
                                ["Model"] + ["weather".capitalize() if d == "weather" else d for d in DATASETS],
                                rows, bold,
                                "Upper block (RC, ours): mean \u00b1 std over 5 networks, RAM-ESN with tau = 1e-4 "
                                "and m selected by Bayesian optimization. Lower block: published results of offline "
                                "deep learning models (TimeMixer++, ICLR 2025; input length 96). 'diverges': the "
                                "generative rollout diverges for at least one of the 5 networks. Bold: best RC model "
                                "and, when better, best deep learning model. The time column of the paper is not "
                                "computed here.")]


def make_task(U):
    return lambda washout: OnlineGenerativeForecasting(*U, washout=washout, horizon=HORIZON, stride=1)


if __name__ == "__main__":
    a = common.arguments("real", "<dataset>_ram1_iNE_t0.0001_mopt", __doc__,
                         "the experiments of Table VI of the paper (all the datasets) and the table",
                         extra=lambda ap: ap.add_argument("--dataset", default="ETTm1",
                                                          help="ETTm1 (default), ETTm2 or weather"))
    names = {k.lower(): k for k in DATASETS}
    if a.dataset.lower() not in names:
        raise SystemExit(f"unknown dataset {a.dataset}: one of {', '.join(DATASETS)}")
    datasets = list(DATASETS) if a.paper else [names[a.dataset.lower()]]
    if a.list:
        common.list_experiments(common.read_experiments(datasets[0], a.csv))
        raise SystemExit
    S = {}
    for ds in datasets:
        E = common.read_experiments(ds, a.csv)
        exps = ([f"{ds.lower()}_{n}" for _, n in RC] if a.paper else list(E) if a.all
                else (a.experiment or f"{ds.lower()}_ram1_iNE_t0.0001_mopt").split(","))
        U = load_real(ds)
        done = common.run_experiments(E, exps, make_task(U), "mse", build_model, U[0].shape[1])
        scores = os.path.join(a.results, f"{ds.lower()}_scores.csv")
        common.save_scores(scores, done)
        print(f"\n{'=' * 80}\ntest MSE of each network: {scores}\n{'=' * 80}")
        S[ds] = {e["experiment"]: common.summary(v) for e, v, _ in done}
    if a.paper:
        common.report_files("PAPER: table", table(S, a.results))
