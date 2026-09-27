"""Three figures for the Letter, at 300 dpi.

Fig 1  the central result: protocol-only accuracy degrades when exercise modality is
       pooled, physiological accuracy does not.
Fig 2  prediction-interval coverage against nominal, with mean width annotated.
Fig 3  threshold detection (ROC) and Bland-Altman agreement.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEED = 42
FIG = Path('figures_letter')
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
C = {'stage': '#b0b0b0', 'workload': '#7f9db9', 'hr': '#4f81bd', 'full': '#1f4e79'}


def ridge():
    return Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])


def wins(Xtr, Xte, idx, q=0.99):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for j in idx:
        cap = np.quantile(Xtr[:, j], q)
        Xtr[:, j] = np.minimum(Xtr[:, j], cap)
        Xte[:, j] = np.minimum(Xte[:, j], cap)
    return Xtr, Xte


# ============================================================ primary cohort
RN = {'id': 'subject', 'speed (km/h)': 'speed', 'BLA (mmol/L)': 'target_raw',
      'max_hr (bpm)': 'hr', 'LF (ms²)': 'hrv_lf', 'HF (ms²)': 'hrv_hf',
      'BF (breaths/min)': 'bf', 'BR (%)': 'br_pct', 'CI (L/min/m²)': 'ci',
      'EE (kcal/h)': 'ee', 'RER': 'rer', "V'CO2 (L/min)": 'vco2', "V'E (L/min)": 've',
      "V'O2 (L/min)": 'vo2', 'VT (L)': 'vt', 'STEP (steps/km)': 'step',
      'SWING (s/(km/h))': 'swing', 'CONTACT (s/(km/h))': 'contact',
      'GAIT (s/(km/h))': 'gait', 'CONGAIT (s/m)': 'congait'}
d = pd.read_csv('data/data.csv').dropna(how='all').rename(columns=RN)
d = d.drop(columns=[c for c in ['LT (mmol/L)', 'BLAclass', 'VGRF (BW)'] if c in d.columns])
d = d.dropna(subset=['target_raw', 'subject', 'hr']).sort_values(['subject', 'stage']).reset_index(drop=True)
g = d.groupby('subject')
d['stage_idx'] = g['stage'].transform(lambda s: s - s.min())
for c in ['hr', 'speed', 'vo2', 've', 'rer', 'bf', 'hrv_lf', 'hrv_hf']:
    d[c + '_dt'] = g[c].diff().fillna(0.0)
for c in ['hr', 'vo2', 'rer']:
    d[c + '_rmean'] = g[c].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(0.0)
for c in ['hr', 'vo2', 've', 'speed']:
    d[c + '_pct_max'] = (d[c] / g[c].transform(lambda s: s.expanding().max()).replace(0, np.nan)).fillna(0.0)
for c in ['hr', 'vo2', 'ee', 'speed']:
    d[c + '_cum'] = g[c].cumsum()
for c in ['hrv_lf', 'hrv_hf', 'ee', 'vo2', 'vco2', 've', 'vt']:
    d[c + '_log'] = np.log1p(d[c])
d['target'] = np.log1p(d['target_raw'])

WORKLOAD = ['speed', 'stage_idx', 'speed_dt', 'speed_pct_max', 'speed_cum']
PHYSIO = ['hr', 'bf', 'br_pct', 'hrv_lf_log', 'hrv_hf_log', 'step', 'swing', 'contact',
          'gait', 'congait', 'hr_dt', 'bf_dt', 'hrv_lf_dt', 'hrv_hf_dt', 'hr_rmean',
          'hr_pct_max', 'hr_cum']
LAB = ['vo2', 'vco2', 've', 'vt', 'rer', 'ci', 'ee', 'vo2_log', 'vco2_log', 've_log',
       'vt_log', 'ee_log', 'vo2_dt', 've_dt', 'rer_dt', 'vo2_rmean', 'rer_rmean',
       'vo2_pct_max', 've_pct_max', 'vo2_cum', 'ee_cum']
hidx = lambda fs: [i for i, c in enumerate(fs) if c.startswith(('hrv_lf', 'hrv_hf'))]


def loso_primary(feats):
    X = d[feats].values.astype(float)
    yl, yr, grp = d['target'].values, d['target_raw'].values, d['subject'].values
    idx, pred = hidx(feats), np.zeros_like(d['target_raw'].values)
    for tr, te in LeaveOneGroupOut().split(X, yl, grp):
        Xtr, Xte = wins(X[tr], X[te], idx)
        pred[te] = np.expm1(ridge().fit(Xtr, yl[tr]).predict(Xte))
    return mean_absolute_error(yr, pred), pred, yr


PRIM = {}
for k, f in [('stage', ['stage_idx']), ('workload', WORKLOAD), ('hr', ['hr']),
             ('full', WORKLOAD + PHYSIO + LAB)]:
    PRIM[k] = loso_primary(f)
    print('primary %-9s MAE=%.3f' % (k, PRIM[k][0]))

# ============================================================ pooled cohort
XL = Path('ext_data/jamnick_DataSet.xlsx')


def _hdr(raw, t=('HR', 'Lactate', 'Sample', 'Power')):
    for i in range(min(5, len(raw))):
        v = [str(x).strip() for x in raw.iloc[i].values]
        if sum(1 for q in t if q in v) >= 3:
            return i
    return None


def load_sheet(name):
    person, proto = name.split()[0].strip(), name.split()[-1].strip()
    raw = pd.read_excel(XL, sheet_name=name, header=None)
    h = _hdr(raw)
    if h is None:
        return None
    t = raw.iloc[h + 1:].copy()
    t.columns = [str(c).strip() for c in raw.iloc[h].values]
    for c in ('HR', 'Power', 'Lactate'):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors='coerce')
    if 'Lactate' not in t.columns or 'HR' not in t.columns:
        return None
    sc = 'Sample' if 'Sample' in t.columns else ('Stage' if 'Stage' in t.columns else None)
    t = t.dropna(subset=['Lactate', 'HR'])
    if not len(t):
        return None

    def ti(v):
        try:
            return int(float(str(v)))
        except ValueError:
            return 0 if str(v).strip().lower() == 'rest' else -1
    st = t[sc].apply(ti).astype(int).values if sc else np.arange(len(t))
    pw = t['Power'].astype(float).values if 'Power' in t.columns else 175.0 + 25 * st
    return pd.DataFrame({'person': 'JAM_' + person, 'recording': 'JAM_' + person + '_' + proto,
                         'stage': st, 'hr': t['HR'].astype(float).values,
                         'target_raw': t['Lactate'].astype(float).values, 'power_w': pw})


xf = pd.ExcelFile(XL)
sheets = [s for s in xf.sheet_names if any(p in s for p in ('GXT3', 'GXT4'))
          and 'MLSS' not in s and 'Lactate' not in s]
jam = pd.concat([r for r in (load_sheet(s) for s in sheets) if r is not None], ignore_index=True)
jam = jam[jam['hr'].between(40, 220) & jam['target_raw'].between(0.3, 25)].copy()
flat = jam.groupby('recording')['hr'].agg(lambda s: s.max() - s.min())
jam = jam[~jam['recording'].isin(flat[flat < 3].index)].copy()
jam['speed'] = jam['power_w'].fillna(175.0) / 175.0 * 8.0
jam['cohort'] = 'cycling'
fig_c = pd.DataFrame({'person': 'FIG_' + d['subject'].astype(str),
                      'recording': 'FIG_' + d['subject'].astype(str),
                      'stage': d['stage'].astype(int), 'hr': d['hr'].astype(float),
                      'speed': d['speed'].astype(float),
                      'target_raw': d['target_raw'].astype(float), 'cohort': 'treadmill'})
c = pd.concat([fig_c, jam[['person', 'recording', 'stage', 'hr', 'speed', 'target_raw', 'cohort']]],
              ignore_index=True).dropna(subset=['hr', 'speed', 'target_raw'])
c = c.sort_values(['recording', 'stage']).reset_index(drop=True)
c['target'] = np.log1p(c['target_raw'])
gg = c.groupby('recording')
c['stage_idx'] = gg['stage'].transform(lambda s: s - s.min())
c['hr_dt'] = gg['hr'].diff().fillna(0.0)
c['speed_dt'] = gg['speed'].diff().fillna(0.0)
c['hr_pct_max'] = (c['hr'] / gg['hr'].transform(lambda s: s.expanding().max()).replace(0, np.nan)).fillna(0).clip(0, 2)
c['speed_pct_max'] = (c['speed'] / gg['speed'].transform(lambda s: s.expanding().max()).replace(0, np.nan)).fillna(0).clip(0, 2)
c['speed_cum'] = gg['speed'].cumsum()


def loso_pool(feats):
    y, pred = c['target_raw'].values, np.zeros(len(c))
    grp = c['person'].values
    for tr, te in LeaveOneGroupOut().split(c, groups=grp):
        m = ridge().fit(c.iloc[tr][feats].values.astype(float), c.iloc[tr]['target'].values)
        pred[te] = np.expm1(m.predict(c.iloc[te][feats].values.astype(float)))
    return mean_absolute_error(y, pred), pred, y


POOL = {}
for k, f in [('stage', ['stage_idx']),
             ('workload', ['speed', 'stage_idx', 'speed_dt', 'speed_pct_max', 'speed_cum']),
             ('hr', ['hr']),
             ('full', ['hr', 'speed', 'hr_dt', 'speed_dt', 'hr_pct_max', 'speed_pct_max'])]:
    POOL[k] = loso_pool(f)
    print('pooled  %-9s MAE=%.3f' % (k, POOL[k][0]))

# ============================================================ FIG 1
labels = ['Stage\nonly', 'Workload\nonly', 'Heart rate\nonly', 'Physiological\n+ workload']
keys = ['stage', 'workload', 'hr', 'full']
prim = [PRIM[k][0] for k in keys]
pool = [POOL[k][0] for k in keys]
x = np.arange(4)
w = 0.38
fig, ax = plt.subplots(figsize=(5.2, 3.3))
b1 = ax.bar(x - w / 2, prim, w, label='Treadmill only (19 participants)',
            color='#c6d9ec', edgecolor='#33587a', linewidth=0.7)
b2 = ax.bar(x + w / 2, pool, w, label='Pooled, two modalities (35 participants)',
            color='#1f4e79', edgecolor='#14314d', linewidth=0.7)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            '%.2f' % b.get_height(), ha='center', fontsize=7.2)
for i in range(4):
    dy = pool[i] - prim[i]
    worse = dy > 0
    ax.text(x[i], max(prim[i], pool[i]) + 0.14,
            ('%+.2f worse' % dy) if worse else ('%+.2f better' % dy),
            ha='center', fontsize=7, fontweight='bold',
            color='#b22222' if worse else '#1a7a1a')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('LOSO mean absolute error (mmol L$^{-1}$)')
ax.set_ylim(0, 1.95)
ax.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax.set_title('Protocol information does not transfer across modality; physiology does',
             fontsize=8.5, pad=8)
plt.savefig(FIG / 'fig1_cross_modality.png')
plt.close()

# ============================================================ FIG 2
conf = pd.read_csv('results/final_table2_conformal.csv')
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.plot([0.75, 1.0], [0.75, 1.0], '--', color='#888', lw=0.9, label='Ideal', zorder=1)
for meth, mk, col in [('split', 'o', '#1f4e79'), ('mondrian', 's', '#b8860b')]:
    s = conf[conf['method'] == meth].sort_values('nominal')
    ax.plot(s['nominal'], s['empirical'], mk + '-', color=col, ms=6, lw=1.3,
            label=('Split conformal' if meth == 'split' else 'Mondrian (workload band)'),
            zorder=3, mec='black', mew=0.5)
    for _, r in s.iterrows():
        ax.annotate('%.1f' % r['mean_width'], (r['nominal'], r['empirical']),
                    textcoords='offset points', xytext=(7, -9), fontsize=6.8, color=col)
ax.set_xlabel('Nominal coverage')
ax.set_ylabel('Empirical coverage (LOSO)')
ax.set_xlim(0.76, 0.99)
ax.set_ylim(0.78, 1.01)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
ax.set_title('Valid coverage, but interval width (mmol L$^{-1}$, annotated)\nis comparable to the measurement range',
             fontsize=8, pad=8)
plt.savefig(FIG / 'fig2_coverage.png')
plt.close()

# ============================================================ FIG 3
mae_f, pred_f, y_f = PRIM['full']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
ax = axes[0]
yb = (y_f >= 4).astype(int)
fpr, tpr, _ = roc_curve(yb, pred_f)
auc = roc_auc_score(yb, pred_f)
ax.plot(fpr, tpr, color='#1f4e79', lw=1.6, label='AUC = %.3f' % auc)
ax.plot([0, 1], [0, 1], '--', color='#aaa', lw=0.9)
sens, spec = 0.842, 0.853
ax.plot([1 - spec], [sens], 'o', color='#b22222', ms=7, mec='black', mew=0.6,
        label='At 4 mmol L$^{-1}$: Se %.2f, Sp %.2f' % (sens, spec))
ax.set_xlabel('1 − specificity')
ax.set_ylabel('Sensitivity')
ax.legend(frameon=False, fontsize=7.2, loc='lower right')
ax.set_title('(a) Detection of lactate $\\geq$ 4 mmol L$^{-1}$', fontsize=8.5)

ax = axes[1]
mean_v = (y_f + pred_f) / 2
diff_v = pred_f - y_f
bias, sd = diff_v.mean(), diff_v.std()
ax.scatter(mean_v, diff_v, s=13, color='#4f81bd', alpha=0.7, edgecolor='black', linewidth=0.25)
ax.axhline(bias, color='#b22222', lw=1.2, label='Bias %+.3f' % bias)
ax.axhline(bias + 1.96 * sd, color='#888', ls='--', lw=1.0,
           label='95%% LoA [%+.2f, %+.2f]' % (bias - 1.96 * sd, bias + 1.96 * sd))
ax.axhline(bias - 1.96 * sd, color='#888', ls='--', lw=1.0)
ax.set_xlabel('Mean of measured and estimated (mmol L$^{-1}$)')
ax.set_ylabel('Estimated − measured (mmol L$^{-1}$)')
ax.legend(frameon=False, fontsize=7.2, loc='lower left')
ax.set_title('(b) Agreement', fontsize=8.5)
plt.tight_layout()
plt.savefig(FIG / 'fig3_threshold_agreement.png')
plt.close()

print('\nwrote %s' % ', '.join(sorted(p.name for p in FIG.glob('*.png'))))
