# WorkloadCal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ananya-sharma/workloadcal/blob/main/WorkloadCal.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/dataset-CC%20BY%204.0-green.svg)](https://doi.org/10.6084/m9.figshare.29279702)

**Calibrated Blood-Lactate Estimation from Wearable Signals via Workload-Conditioned Conformal Regression**

A lightweight, fully reproducible framework for non-invasive blood lactate estimation from wearable physiological signals, with subject-stratified split-conformal prediction intervals.

Companion code for the manuscript submitted to *IEEE Sensors Letters* (under review, June 2026).

## Highlights

- Conditions a regularised regression on continuously updated workload features (HR derivatives, subject-relative intensity, cumulative work proxies, ventilation dynamics, respiratory ratios).
- Wraps the predictor with a subject-stratified split-conformal calibrator producing distribution-free prediction intervals.
- On the public Figshare multimodal blood-lactate dataset (19 subjects, 132 samples), reduces leave-one-subject-out MAE from **1.446 to 1.027 mmol/L** (29.0% reduction, Wilcoxon p<0.001) and achieves empirical conformal coverage of **0.927 at the nominal 95% level**.
- Runs end-to-end in <5 minutes on a free Colab CPU runtime.

## Quick start (one-click Colab)

1. Click the "Open in Colab" badge above.
2. Runtime → Run all.
3. Outputs land in `results/` (metrics, CSVs, JSON) and `figures/` (12 publication-ready PNGs).

## Reproduction from scratch (local)

```bash
git clone https://github.com/ananya-sharma/workloadcal.git
cd workloadcal
pip install -r requirements.txt
jupyter notebook WorkloadCal.ipynb
```

## Dataset

The notebook automatically downloads the public dataset from Figshare:

> *Multimodal Physiological and Biomechanical Wearable Dataset for Non-Invasive Blood Lactate Estimation*
> DOI: [10.6084/m9.figshare.29279702](https://doi.org/10.6084/m9.figshare.29279702)
> Licence: CC BY 4.0

## Repository layout

```
workloadcal/
├── WorkloadCal.ipynb            # main one-click reproducible notebook
├── upgrade_cell.py              # extended analyses (warm-start, bootstrap, Wilcoxon, clinical threshold, Bland-Altman, permutation)
├── generate_full_paper.py       # generates the full .docx paper
├── generate_ieee_letter_v2.py   # generates the 4-page IEEE Sensors Letters version
├── requirements.txt             # pinned dependencies
├── LICENSE                      # MIT
├── CITATION.cff                 # citation metadata
├── figures/                     # publication-ready PNGs at 200 dpi
├── results/                     # metrics.json, CSVs, per-fold outputs
└── paper/                       # manuscript drafts (.docx) and cover letter
```

## Key results

| Configuration | N features | LOSO MAE (mmol/L) | RMSE (mmol/L) | R² |
|---|---|---|---|---|
| P (HR only) baseline | 1 | 1.446 | 2.277 | 0.304 |
| WEAR (this work) | 22 | 1.062 | 1.701 | 0.612 |
| LAB+WEAR (this work) | 43 | **1.027** | 1.689 | 0.617 |

Subject-stratified split-conformal calibration (LAB+WEAR):

| Nominal coverage | Empirical (LOSO) | Mean interval width (mmol/L) |
|---|---|---|
| 0.80 | 0.739 | 3.69 |
| 0.90 | 0.880 | 5.47 |
| 0.95 | **0.927** | 7.24 |

## Citation

If you use this work, please cite both the manuscript and the underlying dataset:

```bibtex
@article{sharma2026workloadcal,
  title  = {Calibrated Blood-Lactate Estimation from Wearable Signals via Workload-Conditioned Conformal Regression},
  author = {Sharma, Ananya},
  journal= {IEEE Sensors Letters},
  year   = {2026},
  note   = {submitted}
}

@dataset{figshare_29279702,
  title  = {Multimodal Physiological and Biomechanical Wearable Dataset for Non-Invasive Blood Lactate Estimation},
  year   = {2025},
  doi    = {10.6084/m9.figshare.29279702},
  license= {CC BY 4.0}
}
```

## Licence

Code: MIT (see [LICENSE](LICENSE))
Dataset: CC BY 4.0 (Figshare; reused under licence)

## Contact

Ananya Sharma — MIT School of Bioengineering Sciences and Research, MIT ADT University, Pune, India
Email: ananya.shrma.pune@gmail.com
