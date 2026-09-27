"""Every number in the rewritten manuscript.

Primary   : pooled 35 participants (LOSO by person). Every model -- baselines included -- is given
            a modality indicator, since modality is always known at deployment. This is the
            fairest possible protocol baseline. Ridge and random forest both reported.
Secondary : treadmill cohort (19), full instrumentation, both models.
Transfer  : leave-one-cohort-out, ordering (AUC, Spearman) vs level (MAE, bias).
"""
import json
import io
import contextlib
import runpy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

SEED = 42
RES = Path('results')
OUT = {}

with contextlib.redirect_stdout(io.StringIO()):
    rb = runpy.run_path('analysis/robustness.py')          # builds pooled frame c, ridge(), forest()
    fn = runpy.run_path('analysis/final_numbers.py')       # builds treadmill frame d + helpers
c, ridge, forest = rb['c'], rb['ridge'], rb['forest']
c['modality'] = (c['cohort'] == 'cycling').astype(float)

MODELS = {'ridge': ridge, 'random forest': forest}


def boot(diff, n=5000):
    rs = np.random.RandomState(SEED)
    b = [diff[rs.randint(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# =============================================================== PRIMARY: pooled 35
P_SETS = {
    'Stage':                   ['stage_idx', 'modality'],
    'Protocol':                ['stage_idx', 'wl_z', 'wl_z_dt', 'modality'],
    'Heart rate':              ['hr', 'modality'],
    'Protocol + heart rate':   ['stage_idx', 'wl_z', 'wl_z_dt', 'hr', 'hr_dt', 'hr_rmean', 'modality'],
}


def pooled(feats, make):
    y, pred, rows = c['target_raw'].values, np.zeros(len(c)), []
    g = c['person'].values
    for tr, te in LeaveOneGroupOut().split(c, groups=g):
        m = make().fit(c.iloc[tr][feats].values.astype(float), c.iloc[tr]['target'].values)
        p = np.expm1(m.predict(c.iloc[te][feats].values.astype(float)))
        pred[te] = p
        rows.append({'person': g[te[0]], 'mae': mean_absolute_error(y[te], p)})
    auc = roc_auc_score((y >= 4).astype(int), pred)
    return dict(mae=mean_absolute_error(y, pred), r2=r2_score(y, pred), auc=auc,
                fd=pd.DataFrame(rows))


print('=' * 78)
print('PRIMARY  pooled LOSO by person: %d people, %d recordings, %d stages; modality flag in all'
      % (c['person'].nunique(), c['recording'].nunique(), len(c)))
print('=' * 78)
prim = {}
for mn, mk in MODELS.items():
    prim[mn] = {k: pooled(f, mk) for k, f in P_SETS.items()}
    print('\n%s' % mn)
    for k, r in prim[mn].items():
        print('  %-24s MAE %.3f  R2 %.3f  AUC %.3f' % (k, r['mae'], r['r2'], r['auc']))
    for base in ('Stage', 'Protocol', 'Heart rate'):
        m = prim[mn][base]['fd'].merge(prim[mn]['Protocol + heart rate']['fd'], on='person')
        diff = (m['mae_x'] - m['mae_y']).values
        W, p = wilcoxon(m['mae_x'], m['mae_y'], alternative='greater')
        lo, hi = boot(diff)
        pct = 100 * (prim[mn][base]['mae'] - prim[mn]['Protocol + heart rate']['mae']) / prim[mn][base]['mae']
        print('  %-11s -> P+HR: dMAE %.3f [%.3f, %.3f]  %+.1f%%  W=%.0f  p=%.2g'
              % (base, diff.mean(), lo, hi, pct, W, p))
        OUT.setdefault('primary_tests', []).append(
            dict(model=mn, baseline=base, dmae=round(float(diff.mean()), 3), ci=[round(lo, 3), round(hi, 3)],
                 pct=round(pct, 1), W=int(W), p=float(p)))
OUT['primary'] = {mn: {k: dict(mae=round(r['mae'], 3), r2=round(r['r2'], 3), auc=round(r['auc'], 3))
                       for k, r in v.items()} for mn, v in prim.items()}

# =============================================================== SECONDARY: treadmill 19
d, WORKLOAD, PHYSIO, LAB, wins, hidx = (fn['d'], fn['WORKLOAD'], fn['PHYSIO'], fn['LAB'],
                                        fn['wins'], fn['HRV_IDX_OF'])
S_SETS = {'Stage': ['stage_idx'], 'Protocol': WORKLOAD, 'Heart rate': ['hr'],
          'Protocol + physiological': WORKLOAD + PHYSIO,
          'Protocol + physiological + cart': WORKLOAD + PHYSIO + LAB}


def tread(f, make):
    X = d[f].values.astype(float)
    yl, yr, g = d['target'].values, d['target_raw'].values, d['subject'].values
    idx, pred, rows = hidx(f), np.zeros(len(d)), []
    for tr, te in LeaveOneGroupOut().split(X, yl, g):
        Xtr, Xte = wins(X[tr], X[te], idx)
        p = np.expm1(make().fit(Xtr, yl[tr]).predict(Xte))
        pred[te] = p
        rows.append({'s': g[te[0]], 'mae': mean_absolute_error(yr[te], p)})
    return dict(mae=mean_absolute_error(yr, pred), r2=r2_score(yr, pred),
                auc=roc_auc_score((yr >= 4).astype(int), pred), fd=pd.DataFrame(rows))


print('\n' + '=' * 78)
print('SECONDARY  treadmill cohort only, 19 participants, full instrumentation')
print('=' * 78)
sec = {}
for mn, mk in MODELS.items():
    sec[mn] = {k: tread(f, mk) for k, f in S_SETS.items()}
    print('\n%s' % mn)
    for k, r in sec[mn].items():
        print('  %-32s MAE %.3f  R2 %.3f  AUC %.3f' % (k, r['mae'], r['r2'], r['auc']))
    tests = [('Stage', 'Heart rate', 'two-sided'),
             ('Stage', 'Protocol + physiological + cart', 'greater'),
             ('Protocol', 'Protocol + physiological + cart', 'greater')]
    for a, b, alt in tests:
        m = sec[mn][a]['fd'].merge(sec[mn][b]['fd'], on='s')
        W, p = wilcoxon(m['mae_x'], m['mae_y'], alternative=alt)
        print('  %-9s vs %-32s dMAE %+.3f  W=%.0f  p=%.4f (%s)'
              % (a, b, sec[mn][a]['mae'] - sec[mn][b]['mae'], W, p, alt))
        OUT.setdefault('secondary_tests', []).append(
            dict(model=mn, a=a, b=b, dmae=round(sec[mn][a]['mae'] - sec[mn][b]['mae'], 3),
                 W=int(W), p=round(float(p), 4), alt=alt))
OUT['secondary'] = {mn: {k: dict(mae=round(r['mae'], 3), r2=round(r['r2'], 3), auc=round(r['auc'], 3))
                         for k, r in v.items()} for mn, v in sec.items()}

# =============================================================== TRANSFER
T_SETS = {'Stage': ['stage_idx'], 'Protocol': ['stage_idx', 'wl_z', 'wl_z_dt'],
          'Heart rate': ['hr'], 'Protocol + heart rate': ['stage_idx', 'wl_z', 'wl_z_dt', 'hr', 'hr_dt', 'hr_rmean']}
print('\n' + '=' * 78)
print('TRANSFER  train on one cohort, test on the other (ridge)')
print('=' * 78)
tr_rows = []
for k, f in T_SETS.items():
    r = {'predictor': k}
    for train, tag in (('treadmill', 'TC'), ('cycling', 'CT')):
        trm = c['cohort'] == train
        m = ridge().fit(c.loc[trm, f].values.astype(float), c.loc[trm, 'target'].values)
        p = np.expm1(m.predict(c.loc[~trm, f].values.astype(float)))
        y = c.loc[~trm, 'target_raw'].values
        r.update({tag + '_MAE': round(mean_absolute_error(y, p), 2),
                  tag + '_bias': round(float(np.mean(p - y)), 2),
                  tag + '_rho': round(float(spearmanr(y, p).correlation), 2),
                  tag + '_AUC': round(roc_auc_score((y >= 4).astype(int), p), 2)})
    tr_rows.append(r)
tr = pd.DataFrame(tr_rows)
print(tr.to_string(index=False))
tr.to_csv(RES / 'v2_transfer.csv', index=False)
OUT['transfer'] = tr.to_dict('records')
OUT['cohort_lactate'] = c.groupby('cohort')['target_raw'].agg(['mean', 'std']).round(2).to_dict()

with open(RES / 'v2_numbers.json', 'w') as f:
    json.dump(OUT, f, indent=2)
print('\nwrote results/v2_numbers.json, v2_transfer.csv')
