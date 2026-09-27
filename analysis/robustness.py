"""Robustness of the central claim: protocol information does not transfer across modality,
physiological information does.

Three attacks a reviewer can make, each tested here:

A1  Unit artefact. The cycling cohort's "speed" is power rescaled (W / 175 * 8). If workload
    fails to transfer only because of that arbitrary mapping, the claim is an artefact.
    Tests: (i) stage-only uses no unit conversion at all; (ii) workload z-scored within
    cohort removes any unit/scale mismatch entirely.

A2  Pooled LOSO is not a transfer test, because training folds contain both modalities.
    Test: leave-one-COHORT-out -- train on one modality only, predict the other.

A3  Linear-model artefact. Test: repeat the pooled comparison with a random forest.

Plus participant-level bootstrap 95% CIs on the key MAE difference.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

SEED = 42
RES = Path('results')
RES.mkdir(exist_ok=True)


def ridge():
    return Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))])


def forest():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)


# ------------------------------------------------------------------ load both cohorts
d = pd.read_csv('data/data.csv').dropna(how='all')
d = d.rename(columns={'id': 'subject', 'speed (km/h)': 'speed', 'BLA (mmol/L)': 'target_raw',
                      'max_hr (bpm)': 'hr'})
d = d.dropna(subset=['target_raw', 'subject', 'hr']).sort_values(['subject', 'stage'])
fig = pd.DataFrame({'person': 'FIG_' + d['subject'].astype(str),
                    'recording': 'FIG_' + d['subject'].astype(str),
                    'stage': d['stage'].astype(int).values, 'hr': d['hr'].astype(float).values,
                    'workload_native': d['speed'].astype(float).values,
                    'target_raw': d['target_raw'].astype(float).values, 'cohort': 'treadmill'})

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
                         'workload_native': pw, 'target_raw': t['Lactate'].astype(float).values})


xf = pd.ExcelFile(XL)
sheets = [s for s in xf.sheet_names if any(p in s for p in ('GXT3', 'GXT4'))
          and 'MLSS' not in s and 'Lactate' not in s]
jam = pd.concat([r for r in (load_sheet(s) for s in sheets) if r is not None], ignore_index=True)
jam = jam[jam['hr'].between(40, 220) & jam['target_raw'].between(0.3, 25)].copy()
flat = jam.groupby('recording')['hr'].agg(lambda s: s.max() - s.min())
jam = jam[~jam['recording'].isin(flat[flat < 3].index)].copy()
jam['workload_native'] = jam['workload_native'].fillna(175.0)
jam['cohort'] = 'cycling'

c = pd.concat([fig, jam], ignore_index=True).dropna(subset=['hr', 'workload_native', 'target_raw'])
c = c.sort_values(['recording', 'stage']).reset_index(drop=True)
c['target'] = np.log1p(c['target_raw'])
g = c.groupby('recording')
c['stage_idx'] = g['stage'].transform(lambda s: s - s.min())
c['hr_dt'] = g['hr'].diff().fillna(0.0)
c['hr_rmean'] = g['hr'].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(0.0)

# workload representations
c['wl_rescaled'] = np.where(c['cohort'] == 'cycling', c['workload_native'] / 175.0 * 8.0,
                            c['workload_native'])                       # as in the manuscript
c['wl_z'] = c.groupby('cohort')['workload_native'].transform(
    lambda s: (s - s.mean()) / s.std())                                  # unit-free, A1(ii)
c['wl_rescaled_dt'] = c.groupby('recording')['wl_rescaled'].diff().fillna(0.0)
c['wl_z_dt'] = c.groupby('recording')['wl_z'].diff().fillna(0.0)

print('combined: %d people, %d recordings, %d stages'
      % (c['person'].nunique(), c['recording'].nunique(), len(c)))

SETS = {
    'stage only':                 ['stage_idx'],
    'workload (rescaled units)':  ['wl_rescaled', 'wl_rescaled_dt', 'stage_idx'],
    'workload (z within cohort)': ['wl_z', 'wl_z_dt', 'stage_idx'],
    'heart rate only':            ['hr'],
    'heart-rate dynamics':        ['hr', 'hr_dt', 'hr_rmean'],
    'HR dyn + workload (z)':      ['hr', 'hr_dt', 'hr_rmean', 'wl_z', 'wl_z_dt', 'stage_idx'],
}


def pooled_loso(feats, make):
    y, pred, rows = c['target_raw'].values, np.zeros(len(c)), []
    grp = c['person'].values
    for tr, te in LeaveOneGroupOut().split(c, groups=grp):
        m = make().fit(c.iloc[tr][feats].values.astype(float), c.iloc[tr]['target'].values)
        p = np.expm1(m.predict(c.iloc[te][feats].values.astype(float)))
        pred[te] = p
        rows.append({'person': grp[te[0]], 'mae': mean_absolute_error(y[te], p)})
    return mean_absolute_error(y, pred), r2_score(y, pred), pd.DataFrame(rows)


def cross_cohort(feats, train_on, make):
    tr = c['cohort'] == train_on
    te = ~tr
    m = make().fit(c.loc[tr, feats].values.astype(float), c.loc[tr, 'target'].values)
    p = np.expm1(m.predict(c.loc[te, feats].values.astype(float)))
    y = c.loc[te, 'target_raw'].values
    return mean_absolute_error(y, p), r2_score(y, p)


def boot_ci(fd_a, fd_b, n=5000):
    m = fd_a.merge(fd_b, on='person', suffixes=('_a', '_b'))
    diff = (m['mae_a'] - m['mae_b']).values
    rs = np.random.RandomState(SEED)
    bs = [diff[rs.randint(0, len(diff), len(diff))].mean() for _ in range(n)]
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


out = []

# ---------------------------------------------------------------- A1 + A3: pooled LOSO, two models
for model_name, make in [('ridge', ridge), ('random forest', forest)]:
    print('\n===== pooled LOSO by person, %s =====' % model_name)
    print('%-28s %7s %7s' % ('predictor', 'MAE', 'R2'))
    res = {}
    for name, f in SETS.items():
        mae, r2, fd = pooled_loso(f, make)
        res[name] = (mae, r2, fd)
        print('%-28s %7.3f %7.3f' % (name, mae, r2))
        out.append({'analysis': 'pooled LOSO', 'model': model_name, 'predictor': name,
                    'MAE': round(mae, 3), 'R2': round(r2, 3)})
    for base in ['stage only', 'workload (z within cohort)']:
        a, b = res[base], res['HR dyn + workload (z)']
        m = a[2].merge(b[2], on='person', suffixes=('_a', '_b'))
        W, p = wilcoxon(m['mae_a'], m['mae_b'], alternative='greater')
        mu, lo, hi = boot_ci(a[2], b[2])
        print('  %-26s -> HR dyn + workload(z): dMAE %.3f [95%% CI %.3f, %.3f]  W=%.0f p=%.2g'
              % (base, mu, lo, hi, W, p))
        out.append({'analysis': 'pooled LOSO', 'model': model_name,
                    'predictor': 'HR dyn+wl(z) vs ' + base, 'MAE': round(mu, 3),
                    'R2': np.nan, 'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3),
                    'W': int(W), 'p': float(p)})

# ---------------------------------------------------------------- A2: train one modality, test other
for model_name, make in [('ridge', ridge), ('random forest', forest)]:
    print('\n===== leave-one-COHORT-out, %s =====' % model_name)
    print('%-28s %22s %22s' % ('predictor', 'treadmill->cycling', 'cycling->treadmill'))
    print('%-28s %11s %10s %11s %10s' % ('', 'MAE', 'R2', 'MAE', 'R2'))
    for name, f in SETS.items():
        m1, r1 = cross_cohort(f, 'treadmill', make)
        m2, r2 = cross_cohort(f, 'cycling', make)
        print('%-28s %11.3f %10.3f %11.3f %10.3f' % (name, m1, r1, m2, r2))
        out.append({'analysis': 'train treadmill -> test cycling', 'model': model_name,
                    'predictor': name, 'MAE': round(m1, 3), 'R2': round(r1, 3)})
        out.append({'analysis': 'train cycling -> test treadmill', 'model': model_name,
                    'predictor': name, 'MAE': round(m2, 3), 'R2': round(r2, 3)})

pd.DataFrame(out).to_csv(RES / 'robustness.csv', index=False)
print('\nwrote results/robustness.csv')
