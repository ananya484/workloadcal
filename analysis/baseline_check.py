"""R4's point 4: is the physiological sensor array doing anything beyond the prescribed stage?

In a graded treadmill protocol, stage number and speed ARE the protocol. If lactate is
predictable from the protocol alone, the ECG/respiratory/IMU array adds nothing and the
paper has no sensor finding. Reviewer 4 asked for this baseline; it was never reported.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

d = pd.read_csv('data/data.csv').dropna(how='all')
d = d.rename(columns={'id':'subject','stage':'stage','speed (km/h)':'speed','BLA (mmol/L)':'target',
    'max_hr (bpm)':'hr','LF (ms²)':'hrv_lf','HF (ms²)':'hrv_hf','BF (breaths/min)':'bf','BR (%)':'br_pct',
    'CI (L/min/m²)':'ci','EE (kcal/h)':'ee','RER':'rer',"V'CO2 (L/min)":'vco2',"V'E (L/min)":'ve',
    "V'O2 (L/min)":'vo2','VT (L)':'vt','STEP (steps/km)':'step','SWING (s/(km/h))':'swing',
    'CONTACT (s/(km/h))':'contact','GAIT (s/(km/h))':'gait','CONGAIT (s/m)':'congait'})
d = d.drop(columns=[c for c in ['LT (mmol/L)','BLAclass','VGRF (BW)'] if c in d.columns])
d = d.dropna(subset=['target','subject','hr']).sort_values(['subject','stage']).reset_index(drop=True)
g = d.groupby('subject')
d['stage_idx'] = g['stage'].transform(lambda s: s - s.min())
for c in ['hr','speed','vo2','ve','rer','bf','hrv_lf','hrv_hf']:
    d[f'{c}_dt'] = g[c].diff().fillna(0.0)
for c in ['hr','vo2','rer']:
    d[f'{c}_rmean'] = g[c].transform(lambda s: s.shift(1).rolling(2,min_periods=1).mean()).fillna(0.0)
for c in ['hr','vo2','ve','speed']:
    d[f'{c}_pct_max'] = (d[c]/g[c].transform(lambda s: s.expanding().max()).replace(0,np.nan)).fillna(0.0)
for c in ['hr','vo2','ee','speed']:
    d[f'{c}_cum'] = g[c].cumsum()
for c in ['hrv_lf','hrv_hf','ee','vo2','vco2','ve','vt']:
    d[f'{c}_log'] = np.log1p(d[c])
d['target_raw'] = d['target']; d['target'] = np.log1p(d['target_raw'])

WORKLOAD = ['speed','stage_idx','speed_dt','speed_pct_max','speed_cum']
PHYSIO   = ['hr','bf','br_pct','hrv_lf_log','hrv_hf_log','step','swing','contact','gait','congait',
            'hr_dt','bf_dt','hrv_lf_dt','hrv_hf_dt','hr_rmean','hr_pct_max','hr_cum']
LAB      = ['vo2','vco2','ve','vt','rer','ci','ee','vo2_log','vco2_log','ve_log','vt_log','ee_log',
            'vo2_dt','ve_dt','rer_dt','vo2_rmean','rer_rmean','vo2_pct_max','ve_pct_max','vo2_cum','ee_cum']

SETS = {
 'A. stage_idx ONLY':          ['stage_idx'],
 'B. speed ONLY':              ['speed'],
 'C. WORKLOAD only (no physio)': WORKLOAD,
 'D. HR only (paper baseline)': ['hr'],
 'E. WEAR (workload+physio)':  WORKLOAD + PHYSIO,
 'F. LAB+WEAR (paper primary)': WORKLOAD + PHYSIO + LAB,
}

def loso(feats):
    X = d[feats].values.astype(float); yl = d['target'].values; yr = d['target_raw'].values
    sub = d['subject'].values; pred = np.zeros_like(yr); rows=[]
    for tr,te in LeaveOneGroupOut().split(X,yl,sub):
        m = Pipeline([('sc',StandardScaler()),('m',Ridge(alpha=1.0))]).fit(X[tr],yl[tr])
        p = np.expm1(m.predict(X[te])); pred[te]=p
        rows.append({'subject':str(sub[te][0]),'mae':mean_absolute_error(yr[te],p)})
    return mean_absolute_error(yr,pred), r2_score(yr,pred), pd.DataFrame(rows), pred, yr

print(f'{"configuration":32s} {"n":>3s} {"MAE":>7s} {"R2":>7s}')
print('-'*54)
out={}
for name,f in SETS.items():
    mae,r2,fd,pred,yr = loso(f); out[name]=(mae,r2,fd,pred,yr)
    print(f'{name:32s} {len(f):3d} {mae:7.3f} {r2:7.3f}')

print('\n--- does the physiological array beat the protocol alone? ---')
for a,b in [('C. WORKLOAD only (no physio)','E. WEAR (workload+physio)'),
            ('C. WORKLOAD only (no physio)','F. LAB+WEAR (paper primary)'),
            ('A. stage_idx ONLY','F. LAB+WEAR (paper primary)')]:
    m = out[a][2].merge(out[b][2],on='subject',suffixes=('_a','_b'))
    W,p = wilcoxon(m['mae_a'],m['mae_b'],alternative='greater')
    d_mae = out[a][0]-out[b][0]
    print(f'  {a.split(".")[0]} vs {b.split(".")[0]}: dMAE={d_mae:+.3f}  W={W:.0f}  p={p:.4f}  {"SIGNIFICANT" if p<0.05 else "not significant"}')

from sklearn.metrics import roc_auc_score
print('\n--- 4 mmol/L threshold AUC ---')
for name in ['A. stage_idx ONLY','C. WORKLOAD only (no physio)','F. LAB+WEAR (paper primary)']:
    _,_,_,pred,yr = out[name]
    print(f'  {name:32s} AUC={roc_auc_score((yr>=4).astype(int),pred):.3f}')
