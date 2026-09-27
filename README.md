# WorkloadCal

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Primary data: CC BY 4.0](https://img.shields.io/badge/primary%20data-CC%20BY%204.0-green.svg)](https://doi.org/10.6084/m9.figshare.29279702)

**Physiological features outperform prescribed workload for cross-modality blood-lactate estimation**

Analysis code for a Letter submitted to *Physiological Measurement* (September 2026).

> **Note on history.** An earlier version of this work, framed as a wearable sensor-calibration
> method, was submitted to *IEEE Sensors Letters* and rejected in September 2026. Reviewer
> criticism identified two defects that are corrected here, and the corrections changed the
> paper's conclusion for the better. Both are documented below. The earlier version remains in
> this repository's git history.

## The question

Wearable blood-lactate estimators are conventionally validated inside a single graded exercise
protocol and benchmarked against a heart-rate-only baseline. But in a graded protocol the
participant is *told* to move at a speed that rises on a fixed schedule, and lactate rises along
with it — so the stage index is a proxy for the target that needs no sensor at all.

Two questions follow:

1. Does physiological measurement add anything beyond the prescribed protocol?
2. Does either source of information survive a change of exercise modality?

## Headline results

**Within one protocol, the protocol is a strong baseline.** Exercise stage alone — one integer,
zero sensors — beats heart rate alone.

| Predictor group | Features | LOSO MAE (mmol/L) | R² | AUC @ 4 mM |
|---|---|---|---|---|
| Stage only | 1 | 1.363 | 0.437 | 0.871 |
| Workload only | 5 | 1.407 | 0.395 | 0.880 |
| Heart rate only | 1 | 1.446 | 0.304 | 0.856 |
| Workload + physiological | 22 | 1.059 | 0.632 | 0.922 |
| Workload + physiological + cart | 43 | **1.004** | 0.663 | **0.944** |

Improvement over the *stage-only* baseline is 26.3 % (Wilcoxon W = 154, p = 0.008) — not the
30.6 % that the same model shows against a heart-rate-only baseline.

**Across modalities, the relationship inverts.** Pooling the treadmill cohort with an independent
cycling cohort, LOSO grouped by unique participant:

| Predictor group | MAE (mmol/L) | R² | AUC | vs stage-only |
|---|---|---|---|---|
| Stage only | 1.512 | 0.276 | 0.812 | — |
| Workload only | 1.439 | 0.298 | 0.879 | +4.8 % |
| Heart rate only | 1.365 | 0.324 | 0.769 | +9.7 % |
| Heart rate + workload | **0.953** | 0.659 | **0.926** | **+36.9 %, p < 0.0001** |

Protocol-only predictors *lose* accuracy when modality is varied (stage-only 1.363 → 1.512).
Heart-rate-based predictors do not. That is the paper's finding.

## Corrections relative to the IEEE submission

**1. Pooled cohort size was wrong.** LOSO folds were keyed on Jamnick *recordings*
(`11MP_GXT3`, `11MP_GXT4`), so the same cyclist appeared on both sides of different folds and N
was reported as 46. There are 16 unique cyclists, so the correct pooled N is **35**, not 46.
Fixed in `analysis/pooled_corrected.py`, which prints both groupings side by side. The effect on
the result is small (MAE 0.945 → 0.953) and significance is unchanged (p < 0.0001), but the
reported N and the fold structure were incorrect as published.

**2. No protocol-only baseline was reported.** The published baseline was heart-rate-only, which
is weaker than the exercise stage index. Reported improvements were therefore optimistic. Fixed
in `analysis/baseline_check.py`.

Also changed: the word "calibrated" is dropped from the title. Split conformal attains nominal
coverage here (0.985 at nominal 0.95) but with intervals ~11.6 mmol/L wide, comparable to the
physiological range of the measurement; a workload-band Mondrian variant is 38 % narrower at
0.924 coverage. Both are reported rather than only the more favourable one.

## Reproducing

```bash
git clone https://github.com/ananya484/workloadcal.git
cd workloadcal
pip install -r requirements.txt

python analysis/baseline_check.py     # protocol-only baselines, primary cohort
python analysis/pooled_corrected.py   # pooled LOSO, both groupings compared
python analysis/final_numbers.py      # every number cited in the manuscript
python analysis/make_figures.py       # the three manuscript figures
```

The primary dataset downloads automatically. The external dataset must be placed at
`ext_data/jamnick_DataSet.xlsx` — see **Data** below.

## Data

| Cohort | Source | Licence | In this repo? |
|---|---|---|---|
| Primary — 19 participants, incremental treadmill, ECG + metabolic cart + IMU | Figshare [10.6084/m9.figshare.29279702](https://doi.org/10.6084/m9.figshare.29279702) | CC BY 4.0 | No — auto-downloaded to `data/` |
| External — Jamnick *et al.* 2018, 16 cyclists, 27 GXT recordings, 268 stages | OSF [293ns](https://osf.io/293ns/) · [PLOS ONE](https://doi.org/10.1371/journal.pone.0199794) | No licence tag on OSF | No — **not redistributed** |

The Jamnick data carry no reuse licence, so this repository ships no raw values from that cohort —
only model predictions (`results/external_jamnick_predictions.csv`: participant, stage, predicted
lactate). Download the workbook from OSF yourself to reproduce the external and pooled analyses.

## Layout

```
workloadcal/
├── analysis/                  # scripts generating every number in the manuscript
├── results/                   # CSV/JSON outputs (final_*.csv are the current manuscript)
├── figures_letter/            # the three manuscript figures, 300 dpi
├── WorkloadCal_MASTER.ipynb   # notebook from the earlier IEEE submission (superseded)
├── data/ · ext_data/          # datasets land here (git-ignored)
├── CITATION.cff · LICENSE     # MIT
```

`results/` also retains the CSVs from the earlier IEEE submission (`headline_metrics.csv`,
`table_II_conformal.csv`, `pooled_cross_cohort.csv`) so the corrections above can be checked
against what was originally reported. The `final_*` files are the current manuscript's numbers.

## Citation

See `CITATION.cff`. Please also cite both source datasets.
