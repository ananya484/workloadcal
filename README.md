# WorkloadCal

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Primary data: CC BY 4.0](https://img.shields.io/badge/primary%20data-CC%20BY%204.0-green.svg)](https://doi.org/10.6084/m9.figshare.29279702)

**Exercise stage explains most of the accuracy of blood-lactate estimation in graded exercise tests**

Analysis code for a paper submitted to *Physiological Measurement* (September 2026).

## The finding

Wearable blood-lactate estimators are usually validated on graded exercise tests and compared
against a heart-rate-only baseline. In a graded test, lactate rises with the prescribed stage, so
the protocol itself predicts the target. Across 35 participants from two public cohorts, with the
exercise modality known:

| Predictor set | Sensor data | Ridge MAE | Ridge AUC | Forest MAE | Forest AUC |
|---|---|---|---|---|---|
| Heart rate | yes | 1.366 | 0.769 | 1.384 | 0.757 |
| Stage | **none** | 0.966 | 0.945 | 0.823 | 0.954 |
| Protocol | **none** | 0.977 | 0.945 | 0.879 | 0.954 |
| Protocol + heart rate | yes | 0.906 | 0.953 | 0.796 | 0.966 |

MAE in mmol/L, leave-one-subject-out grouped by participant; AUC for lactate >= 4 mmol/L.
Relative to the stage-only baseline, adding heart rate improves MAE by 3-6 %, and the confidence
interval of that improvement includes zero under both models; relative to the heart-rate-only
baseline, the same model improves MAE by 34-43 %. The baseline, not the sensing, accounts for
most of the reported gain.

## Reproduce

```bash
git clone https://github.com/ananya484/workloadcal.git
cd workloadcal
pip install -r requirements.txt
python analysis/final_numbers.py     # treadmill-cohort tables, conformal intervals, threshold
python analysis/final_v2.py          # combined-cohort tables, model comparison, transfer
python analysis/make_figures.py      # coverage and threshold figures
python analysis/make_figures_v2.py   # all four manuscript figures -> figures_v2/
```

`analysis/robustness.py` holds the supporting checks: unit-free workload, leave-one-cohort-out
transfer, and ridge vs random forest.

## Data

| Cohort | Source | Licence | In this repo? |
|---|---|---|---|
| Treadmill — 19 participants, ECG + metabolic cart + IMU | Figshare [10.6084/m9.figshare.29279702](https://doi.org/10.6084/m9.figshare.29279702) | CC BY 4.0 | No — place at `data/data.csv` |
| Cycling — Jamnick *et al.* 2018, 16 cyclists, 27 recordings | OSF [293ns](https://osf.io/293ns/) · [PLOS ONE](https://doi.org/10.1371/journal.pone.0199794) | No licence tag on OSF | No — place at `ext_data/jamnick_DataSet.xlsx`; not redistributed |

## Corrections relative to the earlier version

An earlier version of this work, framed as a sensor-calibration method, contained two defects:

1. **Pooled cohort size.** Cross-validation folds were keyed on cycling *recordings*, not people,
   so N was reported as 46 and the same cyclist appeared on both sides of different folds. The
   correct N is 35; folds are now grouped by participant.
2. **Missing protocol baseline.** Only a heart-rate-only baseline was reported. Adding the
   protocol-only baseline showed that most of the reported improvement came from the protocol,
   which became this paper's finding.

The earlier version remains in git history. `results/` retains its CSVs (`headline_metrics.csv`,
`table_II_conformal.csv`, `pooled_cross_cohort.csv`) for comparison; `v2_*`, `final_*` and
`robustness.csv` are the current manuscript's numbers.

## Citation

See `CITATION.cff`. Please also cite both source datasets.
