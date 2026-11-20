# RAM-ESN

Random Attention Memory Echo State Networks (RAM-ESN) for Online Generative Reservoir Computing.

> L. Pedrelli, S. Dettori, V. Colla, *Random Attention Memory for Online Generative Reservoir Computing*.

RAM-ESN is implemented as a layer of the [DeepESN Python Library](https://github.com/lucapedrelli/DeepESN)
(version 1.2.0): `ramesn_layer.RAMESNLayer` is a subclass of `deepesn.layers.Layer` and is
used in a `DeepESN` model like the layers of the DeepESN Python Library; the online generative protocol
(`online_generative_forecasting.OnlineGenerativeForecasting`) is a subclass of `deepesn.tasks.Task`.

```python
from deepesn import DeepESN, Readout
from online_generative_forecasting import OnlineGenerativeForecasting
from ramesn_layer import RAMESNLayer

layer = RAMESNLayer(100, spectral_radius=0.9, input_scaling=0.1, memory=150, temperature=1e-4)
model = DeepESN(1, layers=[layer], readout=Readout('iNE', regularization=1e-6), seed=0)

task = OnlineGenerativeForecasting(U_train, U_val, U_test, washout=500, horizon=96)
result = task.evaluate(model, 'test')        # result['mse']
```

## Installation

```bash
pip install -r requirements.txt
```

## Files

| file | content |
|---|---|
| `ramesn_layer.py` | the RAM-ESN layer: variants IN-v (RAM-ESN), IN-c, OUT-v, OUT-c, LI-v, LI-c, sparse attention, effective spectral radius |
| `online_generative_forecasting.py` | the online generative protocol (task) |
| `generate_imso.py`, `generate_mc.py` | datasets of the paper: `datasets/IMSO.csv`, `datasets/MC.csv` |
| `imso_test.py` | test on IMSO (online generative protocol) |
| `real_test.py` | test on the real-world benchmarks ETTm1, ETTm2, Weather (online generative protocol) |
| `mc_test.py` | test on the memory capacity |
| `hyperparameters.csv` | hyperparameters selected in the paper and test results, one row per experiment |
| `common.py` | model, data splits and test shared by `mc_test.py`, `imso_test.py` and `real_test.py` |

## Datasets

```bash
python generate_imso.py     # datasets/IMSO.csv: 12 regimes MSO_1..MSO_12 of 1000 steps
python generate_mc.py       # datasets/MC.csv: i.i.d. input in (-0.8, 0.8), 5000/1000/1000 steps
```

Both are generated, written and read as in the code of the paper. The CSV files in `datasets/`
are the inputs of the experiments of the paper and are the ones read by the tests: use them as
they are. With the reference environment (see [Reproducibility](#reproducibility)) the two
scripts regenerate them identical bit by bit; with other versions of the libraries or on other
platforms they may differ slightly, so do not overwrite the files of the repository.

The real-world datasets are not included: download them from their original sources into
`datasets/`:

| file | source |
|---|---|
| `ETTm1.csv`, `ETTm2.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset), folder `ETT-small` |
| `weather.csv` | Weather dataset of [Time-Series-Library](https://github.com/thuml/Time-Series-Library) |

All the channels are used (the date column is dropped), with the standard chronological splits
(ETTm1, ETTm2: 34560/11520/11520 steps; Weather: 36887/5270/10539 steps), standardized with the
statistics of the training stream.

## Reproducibility

The experiments of the paper were run on an Apple MacBook Pro with M5 Max chip, with the
versions of the libraries listed in `requirements.txt`, and with one BLAS thread (set by
`common.py`). In this environment the scripts reproduce the test scores of the paper exactly,
and `generate_imso.py` and `generate_mc.py` regenerate `datasets/IMSO.csv` and
`datasets/MC.csv` identical bit by bit.

With other versions of the libraries (in particular NumPy) or on other platforms, the
floating-point operations may differ slightly:

- the regenerated datasets (`IMSO.csv` and `MC.csv`) may differ slightly from the ones of the
  repository: always use `datasets/IMSO.csv` and `datasets/MC.csv` as provided, since a slight
  difference in the input is propagated by the recurrent dynamics and by the online readouts
  over thousands of steps, and can produce a larger difference in the results;
- the test scores may differ slightly from the ones of the paper, within the variability
  of the experiments over the 5 network instances (seeds 0..4).

## Test with the hyperparameters of the paper

Only the test is run, with the hyperparameters selected in the paper (`hyperparameters.csv`), on
the same networks (seeds 0..4); the test score of the paper is printed next to each result.

Each script runs one experiment (by default the main one of the paper), the experiments
chosen with `--experiment` (names comma separated, `--list` lists them), all the experiments
of the dataset (`--all`), or the experiments of the paper (`--paper`), whose table is printed
in the terminal and saved as a PDF in `results/tables/`.

```bash
python mc_test.py                    # RAM-ESN, tau = 1e-4, m selected
python mc_test.py --paper            # Table II (ESN and RAM-ESN for each tau and m): 17 experiments
python imso_test.py                  # RAM-ESN, iNE readout, m selected
python imso_test.py --paper          # Table IV and the figures of the IMSO signal and of the rollouts: 6 experiments
python real_test.py --dataset ETTm1  # RAM-ESN, iNE readout, m selected; also ETTm2, weather
python real_test.py --paper          # Table VI (ETTm1, ETTm2, Weather): 18 experiments
python imso_test.py --experiment imso_esn_iNE        # one experiment, by name
python real_test.py --dataset weather --all          # all the Weather experiments
python mc_test.py --list                             # list the MC experiments
```

IMSO, ETTm1, ETTm2, Weather: readout trained on training + validation, then online generative
protocol on the test stream (96-step rollout and readout update at every step). MC: readout trained on training +
validation, MC computed on the test segment.

Results in `results/`: `mc_scores.csv`, `imso_scores.csv`, `ettm1_scores.csv`, `ettm2_scores.csv`,
`weather_scores.csv` (test score of each network) and,
for IMSO, the figures in `results/figures/`: the IMSO signal (`IMSO.pdf`), the mean rollout on the
first and last test window of each experiment, and the figures of the paper
(`imso_rollouts_first.pdf`, `imso_rollouts_last.pdf`, with `--paper` or `--all`). With `--paper`,
the tables of the paper in `results/tables/` (`table_mc.pdf`, `table_imso.pdf`, `table_real.pdf`).

## Citation

If you use this code, please cite the RAM-ESN paper:

> L. Pedrelli, S. Dettori, V. Colla, "Random Attention Memory for Online Generative Reservoir Computing", 2026.

and the DeepESN Python Library on which it is built:

> C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis",
> Neurocomputing, 2017, vol. 268, pp. 87-99.

## License

BSD 3-Clause.
