"""Every number cited in the Letter, generated in one pass.

Primary cohort  : Figshare treadmill, 19 subjects, full feature sets.
Pooled cohort   : Figshare + Jamnick cycling, LOSO grouped by PERSON (35), common features.
Conformal       : absolute-residual split conformal with the finite-sample
                  ceil((n+1)(1-alpha))/n quantile, plus a workload-band (Mondrian)
                  variant. Coverage pooled at sample level, not averaged over folds.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

SEED = 42
RES = Path('results')
RES.mkdir(exist_ok=True)
OUT = {}


def ridge():
    return Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])


# ================================================================ PRIMARY COHORT
RN = {'id': 'subject', 'speed (km/h)': 'speed', 'BLA (mmol/L)': 'target_raw',
      'max_hr (bpm)': 'hr', 'LF (ms²)': 'hrv_lf', 'HF (ms²)': 'hrv_hf',
      'BF (breaths/min)': 'bf', 'BR (%)': 'br_pct', 'CI (L/min/m²)': 'ci',
      'EE (kcal/h)': 'ee', 'RER': 'rer', "V'CO2 (L/min)": 'vco2', "V'E (L/min)": 've',
      "V'O2 (L/min)": 'vo2', 'VT (L)': 'vt', 'STEP (steps/km)': 'step',
      'SWING (s/(km/h))': 'swing', 'CONTACT (s/(km/h))': 'contact',
      'GAIT (s/(km/h))': 'gait', 'CONGAIT (s/m)': 'congait'}

d = pd.read_csv('data/data.csv').dropna(how='all').rename(columns=RN)
d = d.drop(columns=[c for c in ['LT (mmol/L)', 'BLAclass', 'VGRF (BW)'] if c in d.columns])
d = d.dropna(subset=['target_raw', 'subject', 'hr'])
d = d.sort_values(['subject', 'stage']).reset_index(drop=True)
g = d.groupby('subject')
d['stage_idx'] = g['stage'].transform(lambda s: s - s.min())
for c in ['hr', 'speed', 'vo2', 've', 'rer', 'bf', 'hrv_lf', 'hrv_hf']:
    d[c + '_dt'] = g[c].diff().fillna(0.0)
for c in ['hr', 'vo2', 'rer']:
    d[c + '_rmean'] = g[c].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(0.0)
for c in ['hr', 'vo2', 've', 'speed']:                      # causal: expanding max
    d[c + '_pct_max'] = (d[c] / g[c].transform(lambda s: s.expanding().max())
                         .replace(0, np.nan)).fillna(0.0)
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
HRV_IDX_OF = lambda fs: [i for i, c in enumerate(fs) if c.startswith(('hrv_lf', 'hrv_hf'))]

SETS_P = {
    'stage only': ['stage_idx'],
    'workload only': WORKLOAD,
    'HR only': ['hr'],
    'workload+physio (WEAR)': WORKLOAD + PHYSIO,
    'workload+physio+cart': WORKLOAD + PHYSIO + LAB,
}


def wins(Xtr, Xte, idx, q=0.99):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for j in idx:
        cap = np.quantile(Xtr[:, j], q)
        Xtr[:, j] = np.minimum(Xtr[:, j], cap)
        Xte[:, j] = np.minimum(Xte[:, j], cap)
    return Xtr, Xte


def loso_primary(feats):
    X = d[feats].values.astype(float)
    yl, yr = d['target'].values, d['target_raw'].values
    grp = d['subject'].values
    idx = HRV_IDX_OF(feats)
    pred = np.zeros_like(yr)
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, yl, grp):
        Xtr, Xte = wins(X[tr], X[te], idx)
        p = np.expm1(ridge().fit(Xtr, yl[tr]).predict(Xte))
        pred[te] = p
        rows.append({'group': str(grp[te[0]]), 'mae': mean_absolute_error(yr[te], p)})
    return (mean_absolute_error(yr, pred), float(np.sqrt(mean_squared_error(yr, pred))),
            r2_score(yr, pred), pd.DataFrame(rows), pred, yr)


print('=' * 72)
print('TABLE 1  primary cohort: 19 subjects, %d stage-aggregated samples' % len(d))
print('=' * 72)
print('%-24s %5s %7s %7s %7s %8s' % ('configuration', 'n', 'MAE', 'RMSE', 'R2', 'AUC'))
P = {}
for name, f in SETS_P.items():
    mae, rmse, r2, fd, pred, yr = loso_primary(f)
    auc = roc_auc_score((yr >= 4).astype(int), pred)
    P[name] = dict(mae=mae, rmse=rmse, r2=r2, fd=fd, pred=pred, n=len(f), auc=auc)
    print('%-24s %5d %7.3f %7.3f %7.3f %8.3f' % (name, len(f), mae, rmse, r2, auc))
pd.DataFrame([{'config': k, 'n_feat': v['n'], 'MAE': round(v['mae'], 3),
               'RMSE': round(v['rmse'], 3), 'R2': round(v['r2'], 3),
               'AUC_4mM': round(v['auc'], 3)} for k, v in P.items()]) \
    .to_csv(RES / 'final_table1_primary.csv', index=False)

print('\npaired Wilcoxon, one-sided (primary cohort, 19 folds)')
PAIRS = [('stage only', 'workload+physio (WEAR)'), ('stage only', 'workload+physio+cart'),
         ('workload only', 'workload+physio (WEAR)'), ('workload only', 'workload+physio+cart'),
         ('HR only', 'workload+physio+cart')]
wrows = []
for a, b in PAIRS:
    m = P[a]['fd'].merge(P[b]['fd'], on='group', suffixes=('_a', '_b'))
    W, p = wilcoxon(m['mae_a'], m['mae_b'], alternative='greater')
    imp = 100 * (P[a]['mae'] - P[b]['mae']) / P[a]['mae']
    wrows.append({'baseline': a, 'model': b, 'improvement_pct': round(imp, 1),
                  'W': int(W), 'p_one_sided': round(float(p), 4)})
    print('  %-16s -> %-24s %+5.1f%%  W=%3.0f  p=%.4f  %s'
          % (a, b, imp, W, p, 'SIG' if p < 0.05 else 'ns'))
pd.DataFrame(wrows).to_csv(RES / 'final_wilcoxon_primary.csv', index=False)

# ---------------------------------------------------------------- conformal
def conformal(feats, method, alphas=(0.05, 0.10, 0.20)):
    X = d[feats].values.astype(float)
    yl, yr = d['target'].values, d['target_raw'].values
    grp = d['subject'].values
    idx = HRV_IDX_OF(feats)
    band = pd.cut(d['stage_idx'], bins=[-1, 2, 5, 99], labels=[0, 1, 2]).astype(int).values
    acc = {a: {'cov': [], 'w': []} for a in alphas}
    for fi, (tr_all, te) in enumerate(LeaveOneGroupOut().split(X, yl, grp)):
        subj = np.unique(grp[tr_all])
        rs = np.random.RandomState(SEED + fi)
        rs.shuffle(subj)
        cal_s = set(subj[:max(4, len(subj) // 4)])
        pro = np.array([i for i in tr_all if grp[i] not in cal_s])
        cal = np.array([i for i in tr_all if grp[i] in cal_s])
        Xp, Xte = wins(X[pro], X[te], idx)
        _, Xc = wins(X[pro], X[cal], idx)
        m = ridge().fit(Xp, yl[pro])
        resid = np.abs(yr[cal] - np.expm1(m.predict(Xc)))
        pte = np.expm1(m.predict(Xte))
        for a in alphas:
            n = len(resid)
            lvl = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
            if method == 'split':
                q = np.quantile(resid, lvl, method='higher')
            else:
                qg = np.quantile(resid, lvl, method='higher')
                qb = {}
                for b in np.unique(band[cal]):
                    r = resid[band[cal] == b]
                    qb[b] = (np.quantile(r, min(1.0, np.ceil((len(r) + 1) * (1 - a)) / len(r)),
                                         method='higher') if len(r) >= 8 else qg)
                q = np.array([qb.get(b, qg) for b in band[te]])
            lo, hi = pte - q, pte + q
            acc[a]['cov'].extend(((yr[te] >= lo) & (yr[te] <= hi)).tolist())
            acc[a]['w'].extend((hi - lo).tolist())
    return pd.DataFrame([{'method': method, 'nominal': round(1 - a, 2),
                          'empirical': round(float(np.mean(acc[a]['cov'])), 3),
                          'mean_width': round(float(np.mean(acc[a]['w'])), 2)} for a in alphas])


print('\n' + '=' * 72)
print('TABLE 2  split-conformal intervals, primary cohort (workload+physio+cart)')
print('=' * 72)
conf = pd.concat([conformal(WORKLOAD + PHYSIO + LAB, m) for m in ('split', 'mondrian')],
                 ignore_index=True)
print(conf.to_string(index=False))
conf.to_csv(RES / 'final_table2_conformal.csv', index=False)

# ---------------------------------------------------------------- threshold detail
best = P['workload+physio+cart']
yb = (best['pred'] * 0 + (d['target_raw'].values >= 4)).astype(int)
pb = (best['pred'] >= 4).astype(int)
tn, fp, fn, tp = confusion_matrix(yb, pb).ravel()
thr = dict(threshold=4.0, n_above=int(yb.sum()), n_below=int((1 - yb).sum()),
           TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
           sensitivity=round(tp / (tp + fn), 3), specificity=round(tn / (tn + fp), 3),
           PPV=round(tp / (tp + fp), 3), NPV=round(tn / (tn + fn), 3),
           AUC=round(best['auc'], 3))
print('\n4 mmol/L threshold (primary, workload+physio+cart):')
print('  ' + '  '.join('%s=%s' % (k, v) for k, v in thr.items()))
pd.DataFrame([thr]).to_csv(RES / 'final_threshold.csv', index=False)

bias = float(np.mean(best['pred'] - d['target_raw'].values))
sd = float(np.std(best['pred'] - d['target_raw'].values))
print('  Bland-Altman bias=%+.3f  LoA=[%+.2f, %+.2f]' % (bias, bias - 1.96 * sd, bias + 1.96 * sd))

OUT['primary'] = {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                      for kk, vv in v.items() if kk in ('mae', 'rmse', 'r2', 'n', 'auc')}
                  for k, v in P.items()}
OUT['wilcoxon_primary'] = wrows
OUT['conformal'] = conf.to_dict('records')
OUT['threshold'] = thr
OUT['bland_altman'] = {'bias': round(bias, 3), 'loa_lo': round(bias - 1.96 * sd, 2),
                       'loa_hi': round(bias + 1.96 * sd, 2)}
OUT['n_primary_subjects'] = int(d['subject'].nunique())
OUT['n_primary_samples'] = int(len(d))
with open(RES / 'final_numbers.json', 'w') as f:
    json.dump(OUT, f, indent=2)
print('\nwrote results/final_table1_primary.csv, final_table2_conformal.csv,')
print('      final_wilcoxon_primary.csv, final_threshold.csv, final_numbers.json')
