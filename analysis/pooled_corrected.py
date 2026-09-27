"""Corrected pooled cross-cohort analysis.

Two defects in the submitted version, both raised by Reviewer 4:

R4.2  LOSO groups were Jamnick *recordings* ("11MP_GXT3", "11MP_GXT4"), so the same
      cyclist appeared in both the training and test side of different folds and N was
      inflated from ~35 to 46. Fixed here: LOSO groups are unique people. Per-stage
      derivatives are still computed within a recording, which is correct.

R4.4  The only baseline was HR-only. In a graded protocol the prescribed stage is itself
      a strong lactate predictor, so a stage-only / workload-only baseline is the honest
      comparator. Added.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

RES = Path('results')
RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- primary cohort
d = pd.read_csv('data/data.csv').dropna(how='all')
d = d.rename(columns={'id': 'subject', 'speed (km/h)': 'speed',
                      'BLA (mmol/L)': 'target_raw', 'max_hr (bpm)': 'hr'})
d = d.dropna(subset=['target_raw', 'subject', 'hr'])
d = d.sort_values(['subject', 'stage']).reset_index(drop=True)
fig = pd.DataFrame({
    'person': 'FIG_' + d['subject'].astype(str),
    'recording': 'FIG_' + d['subject'].astype(str),
    'stage': d['stage'].astype(int),
    'hr': d['hr'].astype(float),
    'speed': d['speed'].astype(float),
    'target_raw': d['target_raw'].astype(float),
    'cohort': 'treadmill',
})

# ---------------------------------------------------------------- Jamnick cohort
XL = Path('ext_data/jamnick_DataSet.xlsx')


def _hdr(raw, targets=('HR', 'Lactate', 'Sample', 'Power')):
    for i in range(min(5, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].values]
        if sum(1 for t in targets if t in vals) >= 3:
            return i
    return None


def load_sheet(name):
    person, proto = name.split()[0].strip(), name.split()[-1].strip()
    raw = pd.read_excel(XL, sheet_name=name, header=None)
    h = _hdr(raw)
    if h is None:
        return None
    cols = [str(cc).strip() for cc in raw.iloc[h].values]
    t = raw.iloc[h + 1:].copy()
    t.columns = cols
    for cc in ('HR', 'Power', 'Lactate'):
        if cc in t.columns:
            t[cc] = pd.to_numeric(t[cc], errors='coerce')
    if 'Lactate' not in t.columns or 'HR' not in t.columns:
        return None
    sc = 'Sample' if 'Sample' in t.columns else ('Stage' if 'Stage' in t.columns else None)
    t = t.dropna(subset=['Lactate', 'HR'])
    if not len(t):
        return None

    def to_int(v):
        try:
            return int(float(str(v)))
        except ValueError:
            return 0 if str(v).strip().lower() == 'rest' else -1

    st = t[sc].apply(to_int).astype(int).values if sc else np.arange(len(t))
    power = (t['Power'].astype(float).values if 'Power' in t.columns else 175.0 + 25 * st)
    return pd.DataFrame({
        'person': 'JAM_' + person,
        'recording': 'JAM_' + person + '_' + proto,
        'stage': st,
        'hr': t['HR'].astype(float).values,
        'target_raw': t['Lactate'].astype(float).values,
        'power_w': power,
    })


xf = pd.ExcelFile(XL)
sheets = [s for s in xf.sheet_names
          if any(p in s for p in ('GXT3', 'GXT4')) and 'MLSS' not in s and 'Lactate' not in s]
jam = pd.concat([r for r in (load_sheet(s) for s in sheets) if r is not None], ignore_index=True)
jam = jam[jam['hr'].between(40, 220) & jam['target_raw'].between(0.3, 25)].copy()
flat = jam.groupby('recording')['hr'].agg(lambda s: s.max() - s.min())
jam = jam[~jam['recording'].isin(flat[flat < 3].index)].copy()
jam['speed'] = jam['power_w'].fillna(175.0) / 175.0 * 8.0
jam['cohort'] = 'cycling'
jam = jam[['person', 'recording', 'stage', 'hr', 'speed', 'target_raw', 'cohort']]

print('Jamnick after QC: %d people, %d recordings, %d stages'
      % (jam['person'].nunique(), jam['recording'].nunique(), len(jam)))

# ---------------------------------------------------------------- pool
c = pd.concat([fig, jam], ignore_index=True).dropna(subset=['hr', 'speed', 'target_raw'])
c = c.sort_values(['recording', 'stage']).reset_index(drop=True)
c['target'] = np.log1p(c['target_raw'])
g = c.groupby('recording')                                  # within-recording dynamics
c['stage_idx'] = g['stage'].transform(lambda s: s - s.min())
c['hr_dt'] = g['hr'].diff().fillna(0.0)
c['speed_dt'] = g['speed'].diff().fillna(0.0)
c['hr_pct_max'] = (c['hr'] / g['hr'].transform(lambda s: s.expanding().max())
                   .replace(0, np.nan)).fillna(0).clip(0, 2)
c['speed_pct_max'] = (c['speed'] / g['speed'].transform(lambda s: s.expanding().max())
                      .replace(0, np.nan)).fillna(0).clip(0, 2)
c['speed_cum'] = g['speed'].cumsum()

print('POOLED: %d unique people, %d recordings, %d stages'
      % (c['person'].nunique(), c['recording'].nunique(), len(c)))
print('  (submitted manuscript reported N=%d -- it counted recordings as subjects)'
      % c['recording'].nunique())

SETS = {
    'stage only': ['stage_idx'],
    'workload only (no physio)': ['speed', 'stage_idx', 'speed_dt', 'speed_pct_max', 'speed_cum'],
    'HR only (old baseline)': ['hr'],
    'HR + workload': ['hr', 'speed', 'hr_dt', 'speed_dt', 'hr_pct_max', 'speed_pct_max'],
}


def pooled(feats, groups):
    y = c['target_raw'].values
    pred = np.zeros_like(y)
    rows = []
    for tr, te in LeaveOneGroupOut().split(c, groups=groups):
        m = Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])
        m.fit(c.iloc[tr][feats].values.astype(float), c.iloc[tr]['target'].values)
        p = np.expm1(m.predict(c.iloc[te][feats].values.astype(float)))
        pred[te] = p
        rows.append({'group': str(groups[te[0]]), 'cohort': c['cohort'].iloc[te[0]],
                     'mae': mean_absolute_error(y[te], p)})
    return mean_absolute_error(y, pred), r2_score(y, pred), pd.DataFrame(rows), pred


for label, groups in [('WRONG (by recording, as submitted)', c['recording'].values),
                      ('CORRECT (by person)', c['person'].values)]:
    print('\n===== pooled LOSO grouped %s  N=%d =====' % (label, pd.Series(groups).nunique()))
    print('%-28s %7s %7s %8s' % ('configuration', 'MAE', 'R2', 'AUC@4mM'))
    res = {}
    for name, f in SETS.items():
        mae, r2, fd, pred = pooled(f, groups)
        res[name] = (mae, r2, fd)
        auc = roc_auc_score((c['target_raw'].values >= 4).astype(int), pred)
        print('%-28s %7.3f %7.3f %8.3f' % (name, mae, r2, auc))
    for a, b in [('HR only (old baseline)', 'HR + workload'),
                 ('workload only (no physio)', 'HR + workload'),
                 ('stage only', 'HR + workload')]:
        m = res[a][2].merge(res[b][2], on='group', suffixes=('_a', '_b'))
        W, p = wilcoxon(m['mae_a'], m['mae_b'], alternative='greater')
        imp = 100 * (res[a][0] - res[b][0]) / res[a][0]
        print('  %-26s -> %-14s improve %+5.1f%%  W=%5.0f  p=%.4f  %s'
              % (a, b, imp, W, p, 'SIG' if p < 0.05 else 'ns'))
    if 'CORRECT' in label:
        rows = [{'config': n, 'n_people': pd.Series(groups).nunique(),
                 'MAE': round(mae, 3), 'R2': round(r2, 3)}
                for n, (mae, r2, _) in res.items()]
        m = res['stage only'][2].merge(res['HR + workload'][2], on='group', suffixes=('_a', '_b'))
        W, p = wilcoxon(m['mae_a'], m['mae_b'], alternative='greater')
        pd.DataFrame(rows).assign(W_vs_stage_only=int(W), p_vs_stage_only=float(p)) \
            .to_csv(RES / 'pooled_corrected.csv', index=False)
        for coh in ('treadmill', 'cycling'):
            s = res['HR + workload'][2]
            s = s[s['cohort'] == coh]
            print('  per-cohort %s: N=%d people, mean fold MAE=%.3f'
                  % (coh, len(s), s['mae'].mean()))

print('\nwrote results/pooled_corrected.csv')
