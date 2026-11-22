# Mid-Price Forecasting

Limit order books are noisy, high-frequency market microstructure data. This notebook studies whether short-horizon mid-price movement can be predicted from order-book features, then compares simple baselines with deeper sequence models and interpretability probes.

**Start with the [chronological baseline](#chronological-baseline).** It
provides a repeatable, CPU-only check of whether a fitted linear model
beats an always-stationary rule on the same unseen data. The
[exploratory notebook](baseline.ipynb) is retained for the model and
interpretability history, not as evidence of a time-held-out result.

## Chronological baseline

Use the NoAuction / DecPre version of FI-2010 and its anchored fold files.
Put the four files below anywhere under `Data/`, with only one copy of
each filename. The script does not pool overlapping training folds or
different normalizations.

| Partition | Files | Period |
| --- | --- | --- |
| Fit the scaler and model | `Train_Dst_NoAuction_DecPre_CF_7.txt` | Trading days 1–7 |
| Report validation | `Test_Dst_NoAuction_DecPre_CF_7.txt` | Trading day 8 |
| Report final test | `Test_Dst_NoAuction_DecPre_CF_8.txt`, `Test_Dst_NoAuction_DecPre_CF_9.txt` | Trading days 9–10 |

Using Python 3.12, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python chronological_baseline.py --data-dir Data
```

The report includes accuracy, macro-F1, and a confusion matrix for
logistic regression and the always-stationary rule, plus observation
counts and SHA-256 hashes of the inputs. Confusion matrices use the
original class-ID order `1, 2, 3`; class `2` is stationary. All 144
feature rows are used, with the first label row as the target (a
10-event horizon). Scaling is fitted on the training file only. No
model settings are selected from the validation or test scores.

To verify the pipeline without downloading the market data:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic FI-2010-shaped files. They demonstrate that the
command runs and follows the fixed file split; they do not measure
forecasting performance. No new FI-2010 metrics are claimed here.

## Exploratory notebook

[`baseline.ipynb`](baseline.ipynb) explores logistic regression, an
MLP, a DeepLOB-style sequence model, and attribution methods. Its
archived results use a different workflow: preprocessing combined
input files before splitting, and the sequence model randomly split
overlapping windows. The saved scores are useful for development
history but cannot establish how the models compare on later trading
days. The script above does not reproduce the notebook's deep models.

## Dataset

This project uses the FI-2010 benchmark by Ntakaris et al.:

- [Dataset mirror on Kaggle](https://www.kaggle.com/datasets/praanj/limit-orderbook-data?resource=download)
- [Benchmark paper](https://arxiv.org/abs/1705.03233)
- [DeepLOB paper](https://arxiv.org/abs/1808.03668) for the anchored-fold and chronological evaluation context

The files contain already-preprocessed features and forward labels from
the benchmark. The chronological script keeps its own fitting separate
from evaluation; it does not reconstruct feature or label generation
from the raw market feed. The data is not committed. This is a research
exercise, not a deployed or tested trading strategy.
