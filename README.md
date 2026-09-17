# WorkloadCal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananya484/workloadcal/blob/main/WorkloadCal_MASTER.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Primary data: CC BY 4.0](https://img.shields.io/badge/primary%20data-CC%20BY%204.0-green.svg)](https://doi.org/10.6084/m9.figshare.29279702)

**Calibrated Blood-Lactate Estimation from Wearable Sensor Arrays via Workload-Conditioned Conformal Calibration**

Companion code for the manuscript under review at *IEEE Sensors Letters* (revised submission, August 2026).
This repository contains the exact notebook and outputs that were submitted as supplementary material with the revised manuscript.

## What it does

A calibration layer for wearable physiological sensor arrays (ECG / HRV, respiratory, gait IMU) that

1. conditions a ridge regressor on workload descriptors computed from the sensor stream itself,
2. wraps the point estimate in a **subject-stratified split-conformal** predictor, and
3. is evaluated under strict **leave-one-subject-out (LOSO)** cross-validation, with all subject-relative features and outlier caps computed **inside each fold on training subjects only** (no look-ahead leakage).

## Reproduce (one click)

1. Open `WorkloadCal_MASTER.ipynb` in Google Colab (badge above).
2. Runtime → Run all (~7–10 min on the free CPU runtime).

The notebook downloads both datasets itself and regenerates every table and figure in the manuscript into `results/` and `figures/`.

Cell 11 additionally builds the manuscript `.docx`; it needs the IEEE Sensors Letters Word template placed at `/content/`, which is IEEE-copyrighted and not included here. All scientific outputs are produced by Cells 1–10 and do not need it.

Local run:

```bash
git clone https://github.com/ananya484/workloadcal.git
cd workloadcal
pip install -r requirements.txt
jupyter notebook WorkloadCal_MASTER.ipynb
```

## Data

| Cohort | Source | Licence | In this repo? |
|---|---|---|---|
| Primary — 19 subjects, incremental treadmill, ECG + metabolic cart + IMU | Figshare [10.6084/m9.figshare.29279702](https://doi.org/10.6084/m9.figshare.29279702) | CC BY 4.0 | No — auto-downloaded to `data/` |
| External — Jamnick et al. 2018, 16 cyclists, 27 GXT recordings, 268 stages | OSF [293ns](https://osf.io/293ns/) (PLOS ONE, doi:10.1371/journal.pone.0199794) | No licence tag on OSF | No — auto-downloaded to `ext_data/`; **not redistributed** |

Because the Jamnick data carry no reuse licence, this repository ships only *model predictions* for that cohort (`results/external_jamnick_predictions.csv`: subject, stage, predicted lactate). Running the notebook regenerates the full per-stage file locally.

## Results shipped in `results/`

All figures below are read directly from the CSVs in this repository.

**Headline LOSO accuracy** (`headline_metrics.csv`, ridge regression):

| Configuration | Features | MAE (mmol/L) | RMSE | R² |
|---|---|---|---|---|
| P — HR only | 1 | 1.446 | 2.277 | 0.304 |
| WEAR — wearable-grade only | 22 | 1.286 | 1.864 | 0.534 |
| LAB+WEAR — + metabolic cart | 43 | **1.143** | 1.780 | 0.575 |

Bootstrap 95 % CIs on LOSO MAE (`bootstrap_ci.csv`): P [1.09, 1.69], WEAR [0.95, 1.51], LAB+WEAR [0.88, 1.41].

**Pooled cross-cohort LOSO, N = 46 subjects** (`pooled_cross_cohort.csv`, common feature subset across treadmill + cycling):

| Configuration | MAE | R² | Wilcoxon |
|---|---|---|---|
| P — HR only | 1.365 | 0.324 | — |
| Workload-conditioned | **0.993** | 0.645 | W = 1024, p < 0.0001 |

**External transfer, Jamnick 2018** (`external_jamnick_summary.csv`): MAE 1.062 mmol/L, RMSE 1.75, 95 % conformal coverage 0.828.

**Split-conformal calibration under LOSO** (`table_II_conformal.csv`):

| Regime | Nominal | Empirical | Mean width (mmol/L) |
|---|---|---|---|
| LAB+WEAR | 0.95 | 0.861 | 7.24 |
| LAB+WEAR | 0.90 | 0.830 | 5.47 |
| LAB+WEAR | 0.80 | 0.703 | 3.69 |
| WEAR | 0.95 | 0.876 | 7.65 |
| WEAR | 0.90 | 0.852 | 5.99 |
| WEAR | 0.80 | 0.768 | 3.92 |

Under-coverage relative to nominal is expected and reported as such: LOSO breaks exchangeability between calibration and test subjects, so the finite-sample split-conformal guarantee does not formally hold under subject shift. The numbers are the honest empirical figure.

Per-subject LOSO MAE is in `per_subject_mae.csv`; per-configuration Wilcoxon tests in `wilcoxon.csv`.

## Layout

```
workloadcal/
├── WorkloadCal_MASTER.ipynb   # the single reproducible pipeline (submitted as supplementary)
├── requirements.txt
├── results/                   # CSV outputs of the notebook (as submitted)
├── figures/                   # 6 manuscript figures + graphical abstract, 200 dpi
├── data/                      # primary dataset lands here (git-ignored)
├── ext_data/                  # external dataset lands here (git-ignored)
├── CITATION.cff
└── LICENSE                    # MIT
```

## History

The first submission (May 2026) and its notebook are preserved in this repository's git history (commits up to 2026-06-08). The revised pipeline in this tree differs from it in three ways, all described in the response to reviewers: split-conformal is implemented exactly as §II.D describes (earlier code ran conformalised quantile regression); subject-relative features and HRV winsorisation are computed per fold on training subjects only; and the single-subject Kaggle recording used for external validation was replaced by the 16-subject Jamnick cohort.

## Citation

See `CITATION.cff`. Please also cite the two source datasets.
