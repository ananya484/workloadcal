"""Figures for the rewritten manuscript. Reads results/v2_numbers.json (from final_v2.py).

Fig 1  pooled 35 participants: heart-rate-only vs stage-only vs protocol+HR, ridge and forest.
Fig 2  transfer between cohorts: protocol keeps ORDER (AUC) but loses LEVEL (bias); heart rate
       the reverse.
Figures 3 and 4 (coverage; threshold/agreement) are produced by make_figures.py and reused.
"""
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path('figures_v2')
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
N = json.load(open('results/v2_numbers.json'))

# ------------------------------------------------------------------ Fig 1
order = ['Heart rate', 'Stage', 'Protocol', 'Protocol + heart rate']
labels = ['Heart rate\nonly', 'Stage only\n(no sensor)', 'Protocol\n(no sensor)', 'Protocol +\nheart rate']
cols = ['#b22222', '#7f9db9', '#7f9db9', '#1f4e79']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, mn, ttl in zip(axes, ['ridge', 'random forest'], ['(a) Ridge regression', '(b) Random forest']):
    maes = [N['primary'][mn][k]['mae'] for k in order]
    aucs = [N['primary'][mn][k]['auc'] for k in order]
    bars = ax.bar(range(4), maes, color=cols, edgecolor='black', linewidth=0.6, width=0.68)
    for b, m, a in zip(bars, maes, aucs):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, '%.2f' % m, ha='center', fontsize=7.5)
        ax.text(b.get_x() + b.get_width() / 2, 0.06, 'AUC\n%.2f' % a, ha='center',
                fontsize=6.6, color='white', fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_title(ttl, fontsize=8.5)
    ax.set_ylim(0, 1.6)
axes[0].set_ylabel('LOSO mean absolute error (mmol L$^{-1}$)')
fig.suptitle('35 participants, two cohorts; every model is told the exercise modality',
             fontsize=8.5, y=1.02)
plt.tight_layout()
plt.savefig(FIG / 'fig1_protocol_baseline.png')
plt.close()

# ------------------------------------------------------------------ Fig 2
T = {r['predictor']: r for r in N['transfer']}
preds = ['Stage', 'Protocol', 'Heart rate', 'Protocol + heart rate']
mk = {'Stage': 's', 'Protocol': 'D', 'Heart rate': 'o', 'Protocol + heart rate': '^'}
cc = {'Stage': '#7f9db9', 'Protocol': '#33587a', 'Heart rate': '#b22222', 'Protocol + heart rate': '#1f4e79'}
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for p in preds:
    for tag, fill in (('TC', True), ('CT', False)):
        x, y = abs(T[p][tag + '_bias']), T[p][tag + '_AUC']
        ax.scatter(x, y, marker=mk[p], s=70, color=cc[p] if fill else 'white',
                   edgecolor=cc[p], linewidth=1.4, zorder=3,
                   label=p if tag == 'TC' else None)
ax.set_xlabel('|Mean bias| on unseen cohort (mmol L$^{-1}$)  — level error')
ax.set_ylabel('AUC for lactate ≥ 4 mmol L$^{-1}$  — ordering')
ax.set_xlim(-0.1, 3.5)
ax.set_ylim(0.6, 1.0)
ax.axvspan(-0.1, 0.8, color='#e8f0e8', zorder=0)
ax.text(0.35, 0.615, 'level\npreserved', ha='center', fontsize=6.8, color='#3a6b3a')
ax.text(2.6, 0.615, 'level lost', ha='center', fontsize=6.8, color='#777')
h, l = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
h += [Line2D([], [], marker='o', ls='', mfc='#555', mec='#555'),
      Line2D([], [], marker='o', ls='', mfc='white', mec='#555')]
l += ['treadmill → cycling', 'cycling → treadmill']
ax.legend(h, l, frameon=False, fontsize=6.8, loc='center right', bbox_to_anchor=(1.0, 0.42))
ax.set_title('Train on one cohort, test on the other:\nprotocol keeps order, loses level',
             fontsize=8.5)
plt.savefig(FIG / 'fig2_transfer.png')
plt.close()

# reuse coverage + threshold/agreement figures
shutil.copy('figures_letter/fig2_coverage.png', FIG / 'fig3_coverage.png')
shutil.copy('figures_letter/fig3_threshold_agreement.png', FIG / 'fig4_threshold_agreement.png')
print('wrote', sorted(p.name for p in FIG.glob('*.png')))
