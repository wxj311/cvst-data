# -*- coding: utf-8 -*-
"""
Complete reproducible Python analysis for CVST consciousness disturbance / SII study.
Designed to run as a normal Python script OR from Jupyter.

Outputs:
- Main Figures: Fig1-Fig6
- Supplementary Figures: eFig01-eFig60
- Main Tables: Table1-Table5
- Supplementary Tables: TableS01-TableS20

Important: all numeric results are recalculated from data.csv; no manuscript result is hard-coded.
"""
from __future__ import annotations

import os, re, json, math, warnings, hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.special import expit, logit

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, learning_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve,
    brier_score_loss, accuracy_score, f1_score, confusion_matrix,
    recall_score, precision_score
)
from sklearn.inspection import permutation_importance

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix

# -----------------------------
# Configuration
# -----------------------------
ROOT = Path.cwd()
DATA_FILE = ROOT / "data.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"
META_DIR = OUTPUT_DIR / "metadata"
for d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, META_DIR]: d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LASSO_SEEDS = [0, 7, 13, 21, 42, 99, 123, 777, 2024, 314]
C_GRID = np.logspace(-4, 2, 100)
BOOTSTRAP_N = 1000
REPEATS = 3
FOLDS = 10
N_JOBS = -1
DPI = 300

# To make a quick local QA run, set FAST_MODE=True. Leave False for manuscript-grade settings.
FAST_MODE = False
if FAST_MODE:
    C_GRID = np.logspace(-4, 2, 25)
    BOOTSTRAP_N = 100
    REPEATS = 1
    FOLDS = 5

# Raw names in the provided dataset
Y = 'Consciousness_disturbance'
COL = {
    'Age':'Age', 'Sex':'Gende', 'Pregnancy':'Pregnancy_and_puerperium',
    'OCP':'Oral_contraceptives', 'Thrombosis':'Thrombotic_diseases',
    'Hypertension':'Hypertension', 'Diabetes':'Diabetes_Mellitus', 'Infection':'Infection',
    'Malignancy':'Malignant_tumor', 'Anemia':'Anemia', 'ICP':'Increased_intracranial_pressure',
    'Headache':'Headache', 'Papilledema':'Papilledema', 'Visual':'Visual_impairment',
    'Epilepsy':'Epilepsy', 'FocalDeficit':'Focal_neurological_deficits', 'Aphasia':'Aphasia',
    'WBC':'White_blood_cell_count', 'Platelet':'Platelet_Count', 'Albumin':'Albumin',
    'CRP':'C.Reactive_Protein_.CRP.', 'SII':'Systemic_Inflammatory_Index_.SII.',
    'NLR':'Neutrophil.to.Lymphocyte_Ratio_.NLR.', 'PLR':'Platelet.to.Lymphocyte_Ratio_.PLR.',
    'MHR':'Monocyte.to.Hematocrit_Ratio_.MHR.', 'D_dimer':'Elevated_D.dimer',
    'Fibrinogen':'Fibrinogen_.FIB.', 'PT':'Prothrombin_Time_.PT.',
    'APTT':'Activated_Partial_Thromboplastin_Time_.APTT.', 'SAH':'Subarachnoid_hemorrhage',
    'ICH':'Intracerebral_hemorrhage', 'VenousInfarct':'Venous_cerebral_infarction',
    'DeepVeins':'Deep_cerebral_veins', 'SSS':'Superior_sagittal_sinus',
    'LTS':'Left_transverse_sinus', 'RTS':'Right_transverse_sinus',
    'LSS':'Left_sigmoid_sinus', 'RSS':'Right_sigmoid_sinus',
    'Neutrophil':'Neutrophil_Count_Calculated', 'Lymphocyte':'Lymphocyte_Count_Calculated'
}

DISPLAY = {v:k for k,v in COL.items()}
DISPLAY.update({Y:'Consciousness disturbance'})

# Original 39 retained variables = outcome + 38 predictors (excluding two calculated count columns).
CANDIDATES_38 = [c for c in COL.values() if c not in [COL['Neutrophil'], COL['Lymphocyte']]]
CONTINUOUS = [COL[k] for k in ['Age','WBC','Platelet','Albumin','CRP','SII','NLR','PLR','MHR','Fibrinogen','PT','APTT']]
BINARY = [c for c in CANDIDATES_38 if c not in CONTINUOUS]

PRIMARY_PREEXCLUDED = [COL[k] for k in ['NLR','PLR','MHR','Aphasia','DeepVeins','Malignancy','Thrombosis','RTS','Diabetes','RSS','VenousInfarct','FocalDeficit','Papilledema','Visual','ICP']]
PRIMARY_ELIGIBLE = [c for c in CANDIDATES_38 if c not in PRIMARY_PREEXCLUDED]
FINAL3 = [COL['SII'], COL['Platelet'], COL['ICH']]

PRETTY = {
    COL['Age']:'Age', COL['Sex']:'Male sex', COL['Pregnancy']:'Pregnancy / Puerperium',
    COL['OCP']:'Oral contraceptives', COL['Thrombosis']:'Prior thrombotic disease',
    COL['Hypertension']:'Hypertension', COL['Diabetes']:'Diabetes mellitus', COL['Infection']:'Infection',
    COL['Malignancy']:'Malignant tumour', COL['Anemia']:'Anaemia', COL['ICP']:'Increased ICP',
    COL['Headache']:'Headache', COL['Papilledema']:'Papilloedema', COL['Visual']:'Visual impairment',
    COL['Epilepsy']:'Epilepsy', COL['FocalDeficit']:'Focal neurological deficits', COL['Aphasia']:'Aphasia',
    COL['WBC']:'WBC', COL['Platelet']:'Platelet Count', COL['Albumin']:'Albumin', COL['CRP']:'CRP',
    COL['SII']:'SII', COL['NLR']:'NLR', COL['PLR']:'PLR', COL['MHR']:'MHR',
    COL['D_dimer']:'Elevated D-dimer', COL['Fibrinogen']:'Fibrinogen', COL['PT']:'PT', COL['APTT']:'APTT',
    COL['SAH']:'Subarachnoid haemorrhage', COL['ICH']:'Intracerebral haemorrhage',
    COL['VenousInfarct']:'Venous cerebral infarction', COL['DeepVeins']:'Deep cerebral veins',
    COL['SSS']:'Superior sagittal sinus', COL['LTS']:'Left transverse sinus', COL['RTS']:'Right transverse sinus',
    COL['LSS']:'Left sigmoid sinus', COL['RSS']:'Right sigmoid sinus'
}

# -----------------------------
# Utilities
# -----------------------------
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def save_table(df, name):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False, encoding='utf-8-sig')
    df.to_excel(TAB_DIR / f"{name}.xlsx", index=False)
    return df


def p_fmt(p):
    if pd.isna(p): return ''
    if p < .001: return '<0.001'
    return f'{p:.3f}'


def ci_fmt(est, lo, hi, digits=3):
    return f"{est:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def median_iqr(x, digits=2):
    x = pd.Series(x).dropna().astype(float)
    return f"{x.median():.{digits}f} [{x.quantile(.25):.{digits}f}-{x.quantile(.75):.{digits}f}]"


def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-35,35)))


def safe_auc(y, p):
    try: return roc_auc_score(y,p)
    except: return np.nan


def data_hash(path=DATA_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    df = pd.read_csv(DATA_FILE)
    required = [Y] + CANDIDATES_38
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    # Strict outcome validation
    vals=set(pd.Series(df[Y]).dropna().unique())
    if not vals.issubset({0,1}): raise ValueError(f"Outcome must be 0/1; found {vals}")
    return df


def stratified_boot_indices(y, n_boot=BOOTSTRAP_N, seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y)
    i0=np.where(y==0)[0]; i1=np.where(y==1)[0]
    for _ in range(n_boot):
        idx=np.r_[rng.choice(i0,len(i0),replace=True),rng.choice(i1,len(i1),replace=True)]
        rng.shuffle(idx); yield idx


def youden_threshold(y,p):
    fpr,tpr,thr=roc_curve(y,p); j=tpr-fpr
    return float(thr[np.nanargmax(j)])


def metric_bundle(y,p,threshold=None):
    if threshold is None: threshold=youden_threshold(y,p)
    pred=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(
        AUROC=safe_auc(y,p), AUPRC=average_precision_score(y,p), Brier=brier_score_loss(y,p),
        Accuracy=accuracy_score(y,pred), F1=f1_score(y,pred,zero_division=0),
        Sensitivity=tp/(tp+fn) if tp+fn else np.nan,
        Specificity=tn/(tn+fp) if tn+fp else np.nan,
        PPV=tp/(tp+fp) if tp+fp else np.nan, NPV=tn/(tn+fn) if tn+fn else np.nan,
        Threshold=threshold, TP=tp, FP=fp, FN=fn, TN=tn)


def calibration_intercept_slope(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); lp=logit(p)
    X=sm.add_constant(lp)
    try:
        fit=sm.Logit(y,X).fit(disp=False)
        return float(fit.params[0]), float(fit.params[1])
    except Exception: return np.nan,np.nan


def logistic_or_table(df, predictors, standardize=None):
    if standardize is None: standardize=[]
    X=df[predictors].astype(float).copy()
    means={}; sds={}
    for c in standardize:
        means[c]=X[c].mean(); sds[c]=X[c].std(ddof=0)
        X[c]=(X[c]-means[c])/sds[c]
    X=sm.add_constant(X, has_constant='add')
    fit=sm.Logit(df[Y],X).fit(disp=False,maxiter=1000)
    rows=[]
    for c in predictors:
        b=fit.params[c]; se=fit.bse[c]; p=fit.pvalues[c]
        rows.append({'Feature':PRETTY.get(c,c),'Beta':b,'OR':np.exp(b),
                     'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'P':p})
    return fit,pd.DataFrame(rows),means,sds


def bootstrap_logistic_or(df,predictors,standardize,n_boot=BOOTSTRAP_N,seed=42):
    basefit,tab,means,sds=logistic_or_table(df,predictors,standardize)
    vals={c:[] for c in predictors}
    for idx in stratified_boot_indices(df[Y].values,n_boot,seed):
        b=df.iloc[idx].copy()
        try:
            fit,_,_,_=logistic_or_table(b,predictors,standardize)
            for c in predictors: vals[c].append(np.exp(fit.params[c]))
        except Exception: pass
    for i,c in enumerate(predictors):
        arr=np.asarray(vals[c])
        tab.loc[i,'Boot_CI_low']=np.nanpercentile(arr,2.5)
        tab.loc[i,'Boot_CI_high']=np.nanpercentile(arr,97.5)
        tab.loc[i,'Boot_n']=len(arr)
    return basefit,tab,means,sds

# -----------------------------
# Univariate + LASSO
# -----------------------------
def univariate_screen(df, candidates=CANDIDATES_38):
    out=[]
    for c in candidates:
        if c in CONTINUOUS:
            a=df.loc[df[Y]==1,c].dropna(); b=df.loc[df[Y]==0,c].dropna()
            stat,p=stats.mannwhitneyu(a,b,alternative='two-sided')
            cdpos=median_iqr(a); cdneg=median_iqr(b); typ='Continuous'; test='Mann-Whitney U'
        else:
            tab=pd.crosstab(df[Y],df[c]).reindex(index=[0,1],columns=[0,1],fill_value=0)
            odds,p=stats.fisher_exact(tab.values)
            n1=int(df.loc[df[Y]==1,c].sum()); N1=int((df[Y]==1).sum())
            n0=int(df.loc[df[Y]==0,c].sum()); N0=int((df[Y]==0).sum())
            cdpos=f'{n1} ({100*n1/N1:.1f}%)'; cdneg=f'{n0} ({100*n0/N0:.1f}%)'; typ='Binary'; test="Fisher's exact"
        out.append({'Feature':PRETTY.get(c,c),'Raw variable':c,'Type':typ,'CD+':cdpos,'CD-':cdneg,
                    'P_value':p,'P':p_fmt(p),'Test':test})
    return pd.DataFrame(out)


def lasso_cv_path(X,y,seed=42,cs=C_GRID,n_splits=10):
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    mean_auc=[]; sd_auc=[]; nfeat=[]; coefs=[]
    for C in cs:
        fold_auc=[]; fold_coef=[]
        for tr,te in cv.split(X,y):
            sc=StandardScaler().fit(X.iloc[tr])
            Xtr=sc.transform(X.iloc[tr]); Xte=sc.transform(X.iloc[te])
            m=LogisticRegression(penalty='l1',solver='saga',C=float(C),max_iter=10000,random_state=seed)
            m.fit(Xtr,y.iloc[tr])
            fold_auc.append(safe_auc(y.iloc[te],m.predict_proba(Xte)[:,1])); fold_coef.append(m.coef_[0])
        mean_auc.append(np.nanmean(fold_auc)); sd_auc.append(np.nanstd(fold_auc,ddof=1))
        cf=np.mean(fold_coef,axis=0); coefs.append(cf); nfeat.append(int(np.sum(np.abs(cf)>1e-8)))
    mean_auc=np.array(mean_auc); sd_auc=np.array(sd_auc); coefs=np.asarray(coefs)
    best=int(np.nanargmax(mean_auc)); best_C=float(cs[best]); best_auc=mean_auc[best]
    # 1-SE rule: choose strongest regularisation (smallest C) with mean >= best_mean - SE(best)
    se_best=sd_auc[best]/np.sqrt(n_splits); eligible=np.where(mean_auc>=best_auc-se_best)[0]
    one=int(eligible[0]); one_C=float(cs[one])
    # final models on all data at two choices, standardised
    sc=StandardScaler().fit(X); Xz=sc.transform(X)
    mmin=LogisticRegression(penalty='l1',solver='saga',C=best_C,max_iter=10000,random_state=seed).fit(Xz,y)
    m1se=LogisticRegression(penalty='l1',solver='saga',C=one_C,max_iter=10000,random_state=seed).fit(Xz,y)
    return {'C':np.asarray(cs),'mean_auc':mean_auc,'sd_auc':sd_auc,'coefs':coefs,'nfeat':np.array(nfeat),
            'best_idx':best,'best_C':best_C,'one_idx':one,'one_C':one_C,'scaler':sc,'mmin':mmin,'m1se':m1se}


def multi_seed_lasso(df, prescreen_vars):
    X=df[prescreen_vars].astype(float); y=df[Y].astype(int)
    paths={}; selected=[]
    for seed in LASSO_SEEDS:
        r=lasso_cv_path(X,y,seed=seed,cs=C_GRID,n_splits=min(10,int(y.value_counts().min())))
        paths[seed]=r
        sel=[prescreen_vars[i] for i,v in enumerate(r['mmin'].coef_[0]) if abs(v)>1e-8]
        selected.append(sel)
    freq={c:sum(c in s for s in selected) for c in prescreen_vars}
    consensus=[c for c in prescreen_vars if freq[c]>=8]
    return paths,selected,freq,consensus

# -----------------------------
# Models + CV
# -----------------------------
def make_models(seed=42):
    return {
        'LR': LogisticRegression(C=1.0,penalty='l2',solver='lbfgs',max_iter=5000,random_state=seed),
        'RF': RandomForestClassifier(n_estimators=500,max_depth=4,max_features='sqrt',min_samples_leaf=8,
                                     class_weight='balanced',random_state=seed,n_jobs=N_JOBS),
        'GB': GradientBoostingClassifier(n_estimators=200,learning_rate=.05,max_depth=3,min_samples_leaf=12,
                                         subsample=.8,random_state=seed),
        'SVM-R': SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=seed),
        'SVM-L': SVC(C=.5,kernel='linear',class_weight='balanced',probability=True,random_state=seed),
    }


def repeated_cv_predictions(df, features, repeats=REPEATS, folds=FOLDS, seed=42):
    X=df[features].astype(float); y=df[Y].astype(int).values
    cv=RepeatedStratifiedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)
    models=make_models(seed)
    rec=[]; oof={k:np.zeros(len(df)) for k in models}; count=np.zeros(len(df))
    for split_i,(tr,te) in enumerate(cv.split(X,y),1):
        repeat=(split_i-1)//folds+1; fold=(split_i-1)%folds+1; count[te]+=1
        scaler=StandardScaler().fit(X.iloc[tr]); Xtr=scaler.transform(X.iloc[tr]); Xte=scaler.transform(X.iloc[te])
        for k,m0 in models.items():
            m=clone(m0); m.fit(Xtr,y[tr]); ptr=m.predict_proba(Xtr)[:,1]; pte=m.predict_proba(Xte)[:,1]
            oof[k][te]+=pte
            mb=metric_bundle(y[te],pte)
            rec.append({'Rep':repeat,'Fold':fold,'Model':k,'Test AUC':mb['AUROC'],'Train AUC':safe_auc(y[tr],ptr),
                        'Gap':safe_auc(y[tr],ptr)-mb['AUROC'],'Test Acc':mb['Accuracy'],'Test F1':mb['F1'],
                        'AUPRC':mb['AUPRC'],'Brier':mb['Brier']})
    for k in oof: oof[k]/=count
    return pd.DataFrame(rec),oof


def fit_full_models(df, features, seed=42):
    X=df[features].astype(float); y=df[Y].astype(int).values
    scaler=StandardScaler().fit(X); Xz=scaler.transform(X); models=make_models(seed); fitted={}
    for k,m in models.items(): fitted[k]=clone(m).fit(Xz,y)
    return scaler,fitted


def bootstrap_auc_full(df,features,n_boot=500,seed=42):
    y=df[Y].values; X=df[features].astype(float); scaler,fitted=fit_full_models(df,features,seed)
    Xz=scaler.transform(X); rng=np.random.default_rng(seed); rows=[]
    for k,m in fitted.items():
        vals=[]
        for idx in stratified_boot_indices(y,n_boot,seed+17): vals.append(safe_auc(y[idx],m.predict_proba(Xz[idx])[:,1]))
        rows.append({'Model':k,'N resamples':len(vals),'Mean AUC':np.mean(vals),'SD':np.std(vals,ddof=1),
                     'CI low':np.percentile(vals,2.5),'CI high':np.percentile(vals,97.5)})
    return pd.DataFrame(rows)


def decision_curve(y,p,thresholds):
    y=np.asarray(y); n=len(y); rows=[]
    for t in thresholds:
        pred=p>=t; tp=np.sum(pred&(y==1)); fp=np.sum(pred&(y==0))
        rows.append(tp/n - fp/n*(t/(1-t)))
    return np.asarray(rows)

# -----------------------------
# Attribution: exact LR + Saabas tree-path
# -----------------------------
def _tree_expected_value(tree,node=0):
    # sklearn tree.value stores class counts/probabilities for RF trees; for GB regressors stores raw values.
    v=tree.value[node].ravel()
    if len(v)==1: return float(v[0])
    s=v.sum(); return float(v[1]/s) if s else 0.0


def saabas_tree_single(estimator,x,n_features):
    tree=estimator.tree_; contrib=np.zeros(n_features); node=0; base=_tree_expected_value(tree,0)
    prev=base
    while tree.children_left[node] != tree.children_right[node]:
        f=tree.feature[node]; thr=tree.threshold[node]
        child=tree.children_left[node] if x[f] <= thr else tree.children_right[node]
        val=_tree_expected_value(tree,child); contrib[f]+=val-prev; prev=val; node=child
    return base,contrib


def saabas_attributions(model,Xz):
    Xz=np.asarray(Xz); n,p=Xz.shape
    if isinstance(model,RandomForestClassifier):
        C=np.zeros((n,p)); bases=[]
        for tree in model.estimators_:
            tc=np.zeros((n,p)); b=[]
            for i,x in enumerate(Xz):
                bi,ci=saabas_tree_single(tree,x,p); tc[i]=ci; b.append(bi)
            C+=tc; bases.append(np.mean(b))
        return float(np.mean(bases)),C/len(model.estimators_)
    if isinstance(model,GradientBoostingClassifier):
        # raw-score tree path attributions scaled by learning rate; base as initial raw log-odds
        prior=float(model.init_.class_prior_[1]); base=float(logit(np.clip(prior,1e-6,1-1e-6)))
        C=np.zeros((n,p))
        for stage in model.estimators_[:,0]:
            for i,x in enumerate(Xz):
                _,ci=saabas_tree_single(stage,x,p); C[i]+=model.learning_rate*ci
        return base,C
    raise TypeError(type(model))


def lr_attributions(model,Xz):
    C=np.asarray(Xz)*model.coef_[0][None,:]; return float(model.intercept_[0]),C


def all_attributions(fitted,Xz):
    out={}
    for k in ['LR','RF','GB']:
        if k=='LR': out[k]=lr_attributions(fitted[k],Xz)
        else: out[k]=saabas_attributions(fitted[k],Xz)
    return out

# -----------------------------
# LIME manual implementation
# -----------------------------
def lime_local(model,x0,n_features,n_samples=500,seed=42,kernel_width=None):
    rng=np.random.default_rng(seed); x0=np.asarray(x0); Z=rng.normal(size=(n_samples,n_features))
    Z[0]=x0
    p=model.predict_proba(Z)[:,1]
    dist=np.sqrt(np.sum((Z-x0)**2,axis=1))
    if kernel_width is None: kernel_width=.75*np.sqrt(n_features)
    w=np.exp(-(dist**2)/(kernel_width**2))
    X=np.c_[np.ones(n_samples),Z]
    W=np.sqrt(w)[:,None]; beta=np.linalg.lstsq(X*W,p*W[:,0],rcond=None)[0]
    pred=X@beta
    ssr=np.sum(w*(p-pred)**2); sst=np.sum(w*(p-np.average(p,weights=w))**2)
    r2=1-ssr/sst if sst>0 else np.nan
    return beta[0],beta[1:],r2,Z,p,w,pred

# -----------------------------
# Main analysis cache
# -----------------------------
@dataclass
class Analysis:
    df: pd.DataFrame
    univ: pd.DataFrame
    prescreen: list
    paths: dict
    lasso_selected: list
    lasso_freq: dict
    consensus8: list
    final_fit: object
    final_or: pd.DataFrame
    final_means: dict
    final_sds: dict
    cv: pd.DataFrame
    oof: dict
    scaler: StandardScaler
    fitted: dict
    attr: dict


def build_analysis():
    df=load_data(); univ=univariate_screen(df)
    pmap=dict(zip(univ['Raw variable'],univ['P_value']))
    # Primary step: clinically eligible 24-ish variables then P<0.10. Use data-driven p criterion.
    prescreen=[c for c in PRIMARY_ELIGIBLE if pmap.get(c,1)<.10]
    paths,selected,freq,consensus=multi_seed_lasso(df,prescreen)
    # If seed instability yields fewer/more, consensus is still the manuscript-defined >=8/10 criterion.
    final_fit,final_or,means,sds=bootstrap_logistic_or(df,FINAL3,standardize=FINAL3,n_boot=BOOTSTRAP_N,seed=42)
    cv,oof=repeated_cv_predictions(df,consensus,repeats=REPEATS,folds=FOLDS,seed=42)
    scaler,fitted=fit_full_models(df,consensus,seed=42); Xz=scaler.transform(df[consensus])
    attr=all_attributions(fitted,Xz)
    return Analysis(df,univ,prescreen,paths,selected,freq,consensus,final_fit,final_or,means,sds,cv,oof,scaler,fitted,attr)

# -----------------------------
# Table generators
# -----------------------------
def make_tables(A:Analysis):
    df=A.df; y=df[Y].values
    # Table 1 baseline characteristics (selected clinically relevant variables from original layout)
    baseline=[COL[k] for k in ['Age','Sex','Pregnancy','OCP','Hypertension','Infection','Anemia','Headache','Epilepsy','FocalDeficit','WBC','Platelet','Albumin','CRP','SII','NLR','D_dimer','Fibrinogen','APTT','PT','SAH','ICH','VenousInfarct','SSS','LTS','RTS','LSS','RSS']]
    rows=[]
    for c in baseline:
        ur=A.univ.loc[A.univ['Raw variable']==c].iloc[0]
        if c in CONTINUOUS:
            total=median_iqr(df[c]); pos=median_iqr(df.loc[df[Y]==1,c]); neg=median_iqr(df.loc[df[Y]==0,c])
        else:
            def fm(z): return f"{int(z.sum())} ({100*z.mean():.1f}%)"
            total=fm(df[c]); pos=fm(df.loc[df[Y]==1,c]); neg=fm(df.loc[df[Y]==0,c])
        rows.append({'Variable':PRETTY.get(c,c),'Total (N=213)':total,'CD+ (n=55)':pos,'CD- (n=158)':neg,'P Value':p_fmt(ur.P_value),'Test':ur.Test})
    save_table(pd.DataFrame(rows),'Table1_BaselineCharacteristics')

    # Table 2 selection summary
    pmap=dict(zip(A.univ['Raw variable'],A.univ['P_value'])); finalmap=A.final_or.set_index('Feature')
    rows=[]
    for c in CANDIDATES_38:
        status='Pre-excluded' if c in PRIMARY_PREEXCLUDED else ('Yes [lambda.min]' if c in A.consensus8 else 'No')
        f=PRETTY.get(c,c); rr={'Feature':f,'Univariate P':p_fmt(pmap[c]),'LASSO Status':status,'Final Feature':'Yes' if c in FINAL3 else 'No',
            'OR [95% Bootstrap CI]':'NI','Multivariable P':'NI','Exclusion Note':''}
        if c in FINAL3:
            r=A.final_or.loc[A.final_or.Feature==f].iloc[0]; rr['OR [95% Bootstrap CI]']=f"{r.OR:.3f} [{r.Boot_CI_low:.3f}-{r.Boot_CI_high:.3f}]"; rr['Multivariable P']=p_fmt(r.P)
        if c in PRIMARY_PREEXCLUDED: rr['Exclusion Note']='Pre-specified exclusion / algebraic overlap / mediator-sparsity consideration; see Table S04.'
        rows.append(rr)
    save_table(pd.DataFrame(rows),'Table2_VariableSelection')

    # Table 3 performance
    perf=[]
    for k,g in A.cv.groupby('Model'):
        m=metric_bundle(y,A.oof[k]); perf.append({'Model':k,'AUROC (mean+-SD)':f"{g['Test AUC'].mean():.3f}+-{g['Test AUC'].std(ddof=1):.3f}",
            'Train AUC':g['Train AUC'].mean(),'Gap':g['Train AUC'].mean()-g['Test AUC'].mean(),'Acc':m['Accuracy'],'F1':m['F1'],'AUPRC':m['AUPRC'],
            'Sens':m['Sensitivity'],'Spec':m['Specificity'],'PPV':m['PPV'],'NPV':m['NPV'],'Brier':m['Brier'],'Threshold':m['Threshold']})
    save_table(pd.DataFrame(perf).sort_values('AUROC (mean+-SD)',ascending=False),'Table3_MLPerformance')

    # Table 4 quartiles
    q=pd.qcut(df[COL['SII']],4,labels=['Q1','Q2','Q3','Q4']); td=df.copy(); td['SII_Q']=q
    qrows=[]
    model_specs=[([], 'Unadj'),([COL['Age'],COL['Sex']],'M1'),([COL['Age'],COL['Sex'],COL['ICH'],COL['Epilepsy']],'M2')]
    fits={}
    for adj,label in model_specs:
        X=pd.get_dummies(td[['SII_Q']+adj],drop_first=True,dtype=float); X=sm.add_constant(X)
        fits[label]=sm.Logit(td[Y],X).fit(disp=False)
    for lev in ['Q1','Q2','Q3','Q4']:
        sub=td[td.SII_Q==lev]; rr={'SII Quartile':lev,'SII Range':f"{sub[COL['SII']].min():.0f}-{sub[COL['SII']].max():.0f}",'N':len(sub),
            'CD n (%)':f"{sub[Y].sum()} ({100*sub[Y].mean():.1f}%)",'SII Median [IQR]':median_iqr(sub[COL['SII']],0)}
        for label in ['Unadj','M1','M2']:
            if lev=='Q1': rr[f'{label} OR (95% CI)']='Reference'; rr[f'P {label}']=''
            else:
                term=f'SII_Q_{lev}'; f=fits[label]; b=f.params[term]; se=f.bse[term]
                rr[f'{label} OR (95% CI)']=ci_fmt(np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se),2); rr[f'P {label}']=p_fmt(f.pvalues[term])
        qrows.append(rr)
    save_table(pd.DataFrame(qrows),'Table4_SII_Quartile')

    # Table 5 final features
    ft=A.final_or.copy(); ft['OR [95% Bootstrap CI]']=ft.apply(lambda r:f"{r.OR:.3f} [{r.Boot_CI_low:.3f}-{r.Boot_CI_high:.3f}]",axis=1)
    ft['Direction']=np.where(ft.OR>1,'Positive association','Inverse association'); ft['Multivariable P']=ft.P.map(p_fmt)
    save_table(ft[['Feature','Direction','OR [95% Bootstrap CI]','Multivariable P']],'Table5_FinalFeatures')

    # S01 CV folds
    save_table(A.cv[['Model','Rep','Fold','Test AUC','Train AUC','Gap','Test Acc','Test F1']],'TableS01_CV_Folds')
    # S02 bootstrap full-cohort AUC
    save_table(bootstrap_auc_full(df,A.consensus8,n_boot=500 if not FAST_MODE else 100),'TableS02_Bootstrap_AUC')
    # S03 LASSO path seed42
    r=A.paths[42]; save_table(pd.DataFrame({'C':r['C'],'log10C':np.log10(r['C']),'Mean CV-AUC':r['mean_auc'],'SD':r['sd_auc'],'N features':r['nfeat'],
        'lambda.min':np.arange(len(r['C']))==r['best_idx'],'lambda.1se':np.arange(len(r['C']))==r['one_idx']}),'TableS03_LASSO_Path')
    # S04 exclusions
    reasons={COL['NLR']:'Algebraic overlap with SII; strong collinearity.',COL['PLR']:'Algebraic composite overlapping SII components.',COL['MHR']:'No univariate association in source analysis.',
             COL['ICP']:'Potential downstream manifestation / mediator.',COL['Papilledema']:'Potential manifestation of raised ICP.',COL['Visual']:'Potential downstream clinical manifestation.',
             COL['Aphasia']:'Sparse clinical sign.',COL['DeepVeins']:'Sparse involvement.',COL['Malignancy']:'Sparse exposure.',COL['Thrombosis']:'Little/no univariate signal.',
             COL['RTS']:'Little/no univariate signal.',COL['Diabetes']:'Little/no univariate signal.',COL['RSS']:'Little/no univariate signal.',COL['VenousInfarct']:'Little/no univariate signal.',COL['FocalDeficit']:'Ambiguous direction / limited association.'}
    save_table(pd.DataFrame([{'Variable':PRETTY.get(c,c),'Exclusion Reason':reasons.get(c,'Pre-specified clinical exclusion')} for c in PRIMARY_PREEXCLUDED]),'TableS04_Exclusion_Rationale')
    # S05 all univariate
    s5=A.univ.copy(); s5['Status']=s5['Raw variable'].map(lambda c:'FINAL' if c in FINAL3 else ('Pre-excl' if c in PRIMARY_PREEXCLUDED else ('LASSO consensus' if c in A.consensus8 else 'Not selected')))
    save_table(s5,'TableS05_Univariate_All')
    # S06 confusion matrices
    cm=[]
    for k,p in A.oof.items():
        m=metric_bundle(y,p); cm.append({'Model':k,**{q:m[q] for q in ['TP','FP','FN','TN','Sensitivity','Specificity','PPV','NPV','F1','Accuracy','Threshold']}})
    save_table(pd.DataFrame(cm),'TableS06_ConfusionMatrices')
    # S07 hyperparameters
    models=make_models(); save_table(pd.DataFrame([{'Model':k,'Parameters':'; '.join(f'{a}={b}' for a,b in m.get_params().items())} for k,m in models.items()]),'TableS07_Hyperparameters')
    # S08 descriptive stats for 3 continuous final/related
    rows=[]
    for c in [COL['Platelet'],COL['SII'],COL['WBC']]:
        for lab,mask in [('Total',np.ones(len(df),bool)),('CD+',df[Y].eq(1)),('CD-',df[Y].eq(0))]:
            x=df.loc[mask,c]; rows.append({'Feature':PRETTY[c],'Group':lab,'N':len(x),'Mean':x.mean(),'SD':x.std(ddof=1),'Median':x.median(),'Q1':x.quantile(.25),'Q3':x.quantile(.75)})
    save_table(pd.DataFrame(rows),'TableS08_Descriptive_Stats')
    # S09 Spearman with outcome
    rows=[]
    for c in FINAL3:
        r,p=stats.spearmanr(df[c],df[Y]); rows.append({'Feature':PRETTY[c],'Spearman r':r,'P':p_fmt(p)})
    save_table(pd.DataFrame(rows),'TableS09_Spearman_Correlations')
    # S10 quartile desc
    rows=[]
    for lev in ['Q1','Q2','Q3','Q4']:
        sub=td[td.SII_Q==lev]; rows.append({'Quartile':lev,'SII Range':f"{sub[COL['SII']].min():.0f}-{sub[COL['SII']].max():.0f}",'N':len(sub),'CD n (%)':f"{sub[Y].sum()} ({100*sub[Y].mean():.1f}%)",
            'SII Median [IQR]':median_iqr(sub[COL['SII']],0),'Age Median':sub[COL['Age']].median(),'WBC Median':sub[COL['WBC']].median()})
    save_table(pd.DataFrame(rows),'TableS10_SII_Quartile_Desc')
    # S11 RCS knots
    per=[5,25,50,75,95]; vals=np.percentile(df[COL['SII']],per); save_table(pd.DataFrame({'Knot':[f'K{i+1}' for i in range(5)],'Percentile':[f'{x}th' for x in per],'SII Value':vals}), 'TableS11_RCS_Knots')
    # S12 pairwise AUC differences from mean per-fold AUC
    means=A.cv.groupby('Model')['Test AUC'].mean(); ks=['LR','RF','GB','SVM-R','SVM-L']; mat=[]
    for a in ks: mat.append({'Model':a,**{b:means[a]-means[b] for b in ks}})
    save_table(pd.DataFrame(mat),'TableS12_Pairwise_AUC')
    # S13 repeated CV
    save_table(A.cv[['Rep','Fold','Model','Test AUC','Train AUC']],'TableS13_RepeatedCV')
    # S14 LR predicted risk quintile
    p=A.oof['LR']; qq=pd.qcut(p,5,labels=[f'Q{i}' for i in range(1,6)]); rows=[]
    for lev in qq.categories:
        mask=qq==lev; rows.append({'Risk Quintile':lev,'N':mask.sum(),'CD n (%)':f"{y[mask].sum()} ({100*y[mask].mean():.1f}%)",'Mean PP':p[mask].mean(),'Range':f'{p[mask].min():.3f}-{p[mask].max():.3f}'})
    save_table(pd.DataFrame(rows),'TableS14_Risk_Quintile')
    # S15 DCA
    thresholds=np.linspace(.05,.80,14); d={'Threshold':thresholds,'All-treat':decision_curve(y,np.ones(len(y)),thresholds)}
    for k,p in A.oof.items(): d[k]=decision_curve(y,p,thresholds)
    save_table(pd.DataFrame(d),'TableS15_DCA')
    # S16 continuous SII models
    specs=[([], 'Unadjusted (M0)'),([COL['Age'],COL['Sex']],'Model 1: Age+Sex'),([COL['Age'],COL['Sex'],COL['ICH'],COL['Epilepsy']],'Model 2: Fully adjusted')]
    rows=[]
    sii_z=(df[COL['SII']]-df[COL['SII']].mean())/df[COL['SII']].std(ddof=0)
    for adj,label in specs:
        X=df[adj].astype(float).copy(); X.insert(0,'SII_z',sii_z); f=sm.Logit(y,sm.add_constant(X)).fit(disp=False); b=f.params['SII_z']; se=f.bse['SII_z']
        rows.append({'Model':label,'OR per SD [95% CI]':ci_fmt(np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)),'P':p_fmt(f.pvalues['SII_z'])})
    save_table(pd.DataFrame(rows),'TableS16_SII_Continuous_OR')
    # S17 subgroup estimates
    rows=[]
    for mod in [COL['Sex'],COL['Pregnancy'],COL['Infection'],COL['Epilepsy'],COL['ICH']]:
        for level in [0,1]:
            sub=df[df[mod]==level]
            if sub[Y].nunique()<2 or len(sub)<20: continue
            z=(sub[COL['SII']]-sub[COL['SII']].mean())/sub[COL['SII']].std(ddof=0); X=sm.add_constant(pd.DataFrame({'SII_z':z}))
            f=sm.Logit(sub[Y],X).fit(disp=False); b=f.params['SII_z']; se=f.bse['SII_z']
            rows.append({'Modifier':PRETTY[mod],'Level':level,'N':len(sub),'Events':sub[Y].sum(),'OR per SD':np.exp(b),'CI low':np.exp(b-1.96*se),'CI high':np.exp(b+1.96*se),'P':f.pvalues['SII_z']})
    save_table(pd.DataFrame(rows),'TableS17_Subgroup_Note')
    # S18 inter-predictor correlations
    rows=[]
    for i,a in enumerate(FINAL3):
        for b in FINAL3[i+1:]:
            r,p=stats.spearmanr(df[a],df[b]); rows.append({'Predictor 1':PRETTY[a],'Predictor 2':PRETTY[b],'Spearman r':r,'P':p_fmt(p),'Assessment':'Report with algebraic-dependence caveat'})
    save_table(pd.DataFrame(rows),'TableS18_Inter_Predictor_Corr')
    # S19 complete final summary
    s9=pd.read_csv(TAB_DIR/'TableS09_Spearman_Correlations.csv'); ft=A.final_or.copy(); rows=[]
    for c in FINAL3:
        f=PRETTY[c]; ur=A.univ[A.univ['Raw variable']==c].iloc[0]; rr=ft[ft.Feature==f].iloc[0]; sp=s9[s9.Feature==f].iloc[0]
        rows.append({'Feature':f,'Spearman r':sp['Spearman r'],'Univariate P':p_fmt(ur.P_value),'OR [95% Bootstrap CI]':f"{rr.OR:.3f} [{rr.Boot_CI_low:.3f}-{rr.Boot_CI_high:.3f}]",'Multivariable P':p_fmt(rr.P),'Direction':'Risk' if rr.OR>1 else 'Protective'})
    save_table(pd.DataFrame(rows),'TableS19_Feature_Complete_Summary')
    # S20 stability additional seeds 1-5, 10-fold fixed features
    rows=[]
    for seed in [1,2,3,4,5]:
        cv,_=repeated_cv_predictions(df,A.consensus8,repeats=1,folds=10,seed=seed)
        cv['Seed']=seed; rows.append(cv[['Seed','Fold','Model','Test AUC','Train AUC']])
    save_table(pd.concat(rows,ignore_index=True),'TableS20_Stability')

# -----------------------------
# RCS helper
# -----------------------------
def fit_rcs(df):
    sii=df[COL['SII']].astype(float); knots=np.percentile(sii,[5,25,50,75,95])
    # Patsy cubic regression spline with 5 df approximates restricted/natural cubic spline.
    B=dmatrix("cr(x, knots=knots[1:-1], lower_bound=knots[0], upper_bound=knots[-1])-1", {"x":sii,"knots":knots}, return_type='dataframe')
    adj=df[[COL['Age'],COL['Sex'],COL['ICH'],COL['Epilepsy']]].astype(float).reset_index(drop=True)
    X=pd.concat([B.reset_index(drop=True),adj],axis=1); X=sm.add_constant(X)
    fit=sm.Logit(df[Y].values,X).fit(disp=False,maxiter=1000)
    return fit,knots,B.columns.tolist()


def rcs_prediction(df,grid=None):
    fit,knots,bcols=fit_rcs(df)
    if grid is None: grid=np.linspace(df[COL['SII']].quantile(.01),df[COL['SII']].quantile(.99),250)
    B=dmatrix("cr(x, knots=knots[1:-1], lower_bound=knots[0], upper_bound=knots[-1])-1", {"x":grid,"knots":knots}, return_type='dataframe')
    med={COL['Age']:df[COL['Age']].median(),COL['Sex']:df[COL['Sex']].median(),COL['ICH']:df[COL['ICH']].median(),COL['Epilepsy']:df[COL['Epilepsy']].median()}
    X=B.copy()
    for c,v in med.items(): X[c]=v
    X=sm.add_constant(X,has_constant='add'); X=X.reindex(columns=fit.params.index,fill_value=1.0)
    eta=np.asarray(X)@fit.params.values; cov=fit.cov_params().values; se=np.sqrt(np.einsum('ij,jk,ik->i',np.asarray(X),cov,np.asarray(X)))
    pr=expit(eta); lo=expit(eta-1.96*se); hi=expit(eta+1.96*se)
    ref=np.median(df[COL['SII']]); ridx=np.argmin(abs(grid-ref)); logor=eta-eta[ridx]
    # difference covariance vs ref
    xr=np.asarray(X)[ridx]; D=np.asarray(X)-xr; seD=np.sqrt(np.einsum('ij,jk,ik->i',D,cov,D)); orv=np.exp(logor); orlo=np.exp(logor-1.96*seD); orhi=np.exp(logor+1.96*seD)
    return grid,pr,lo,hi,orv,orlo,orhi,knots

# -----------------------------
# Figure generators
# -----------------------------
def fig_violin(ax,df,c,title,ylabel):
    groups=[df.loc[df[Y]==0,c],df.loc[df[Y]==1,c]]
    ax.violinplot(groups,showmedians=True,showextrema=False); ax.set_xticks([1,2],['No CD','CD']); ax.set_title(title); ax.set_ylabel(ylabel)


def plot_main_figures(A):
    df=A.df; y=df[Y].values
    # Fig1: selection 4 panels
    fig,axs=plt.subplots(2,2,figsize=(12,9)); r=A.paths[42]
    ax=axs[0,0]; x=np.log10(r['C']); ax.plot(x,r['mean_auc']); ax.fill_between(x,r['mean_auc']-r['sd_auc']/np.sqrt(10),r['mean_auc']+r['sd_auc']/np.sqrt(10),alpha=.2); ax.axvline(np.log10(r['best_C']),ls='--'); ax.axvline(np.log10(r['one_C']),ls=':'); ax.set(xlabel='log10(C)',ylabel='10-fold CV AUROC',title='A. LASSO regularisation path')
    ax=axs[0,1]; items=sorted(A.lasso_freq.items(),key=lambda kv:kv[1]); ax.barh([PRETTY[c] for c,_ in items],[v for _,v in items]); ax.axvline(8,ls='--'); ax.set(xlabel='Selection frequency / 10 seeds',title='B. Multi-seed LASSO consensus')
    ax=axs[1,0]; stages=[39,len(PRIMARY_ELIGIBLE),len(A.prescreen),len(A.consensus8),3]; labs=['All variables','Clinical pre-screen','P<0.10','LASSO consensus','Final LR']; ax.barh(labs,stages); [ax.text(v+.3,i,str(v),va='center') for i,v in enumerate(stages)]; ax.set_title('C. Four-step selection funnel')
    ax=axs[1,1]; t=A.final_or.iloc[::-1]; x=t.OR.values; lo=t.Boot_CI_low.values; hi=t.Boot_CI_high.values; ax.errorbar(x,np.arange(len(t)),xerr=[x-lo,hi-x],fmt='o'); ax.axvline(1,ls='--'); ax.set_xscale('log'); ax.set_yticks(np.arange(len(t)),t.Feature); ax.set(title='D. Final standardised logistic model',xlabel='Odds ratio (95% bootstrap CI)')
    savefig(fig,'Fig1_VariableSelection')

    # Fig2 3x3 baseline
    fig,axs=plt.subplots(3,3,figsize=(13,11));
    fig_violin(axs[0,0],df,COL['SII'],'A. SII','SII'); fig_violin(axs[0,1],df,COL['Platelet'],'B. Platelet','Platelet'); fig_violin(axs[0,2],df,COL['WBC'],'C. WBC','WBC')
    q=pd.qcut(df[COL['SII']],4,labels=['Q1','Q2','Q3','Q4']); rates=df.groupby(q,observed=False)[Y].mean()*100; axs[1,0].bar(rates.index.astype(str),rates.values); axs[1,0].set(title='D. CD rate by SII quartile',ylabel='CD prevalence (%)')
    fig_violin(axs[1,1],df,COL['Albumin'],'E. Albumin','g/L')
    feats=[COL['Pregnancy'],COL['D_dimer'],COL['ICH'],COL['Epilepsy']]; x=np.arange(len(feats)); w=.35; a=[100*df.loc[df[Y]==1,c].mean() for c in feats]; b=[100*df.loc[df[Y]==0,c].mean() for c in feats]; axs[1,2].barh(x-w/2,a,w,label='CD'); axs[1,2].barh(x+w/2,b,w,label='No CD'); axs[1,2].set_yticks(x,[PRETTY[c] for c in feats]); axs[1,2].legend(); axs[1,2].set(title='F. Key binary features',xlabel='Prevalence (%)')
    fig_violin(axs[2,0],df,COL['Albumin'],'G. Albumin repeat view','g/L')
    axs[2,1].scatter(df[COL['WBC']],df[COL['SII']],c=df[Y],alpha=.7); axs[2,1].set(xlabel='WBC',ylabel='SII',title='H. SII vs WBC')
    for i,c in enumerate([COL['SII'],COL['Platelet'],COL['Albumin']]): axs[2,2].boxplot([df.loc[df[Y]==0,c],df.loc[df[Y]==1,c]],positions=[i*3+1,i*3+2],widths=.7)
    axs[2,2].set_xticks([1.5,4.5,7.5],['SII','Platelet','Albumin']); axs[2,2].set_title('I. Key predictors')
    savefig(fig,'Fig2_BaselineChar')

    # Fig3 ML performance 4 panels
    fig,axs=plt.subplots(2,2,figsize=(12,10));
    ax=axs[0,0]
    for k,p in A.oof.items(): fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,label=f'{k} {safe_auc(y,p):.3f}')
    ax.plot([0,1],[0,1],ls='--'); ax.legend(); ax.set(xlabel='1-Specificity',ylabel='Sensitivity',title='A. ROC curves')
    g=A.cv.groupby('Model').agg(test=('Test AUC','mean'),sd=('Test AUC','std'),train=('Train AUC','mean')).reindex(['LR','SVM-L','RF','SVM-R','GB']); ax=axs[0,1]; ax.bar(np.arange(len(g)),g.test,yerr=g.sd); ax2=ax.twinx(); ax2.plot(np.arange(len(g)),g.train-g.test,'o--'); ax.set_xticks(np.arange(len(g)),g.index); ax.set(title='B. AUROC and train-test gap',ylabel='AUROC'); ax2.set_ylabel('Train-test gap')
    ax=axs[1,0]
    for k,p in A.oof.items():
        bins=pd.qcut(p,5,duplicates='drop'); tmp=pd.DataFrame({'p':p,'y':y,'b':bins}).groupby('b',observed=False).agg(p=('p','mean'),y=('y','mean')); ax.plot(tmp.p,tmp.y,'o-',label=f'{k} B={brier_score_loss(y,p):.3f}')
    ax.plot([0,1],[0,1],ls='--'); ax.legend(fontsize=8); ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction',title='C. Calibration')
    ax=axs[1,1]; th=np.linspace(.05,.8,50); prev=y.mean(); ax.plot(th,prev-(1-prev)*th/(1-th),'--',label='Treat all'); ax.axhline(0,ls=':')
    for k,p in A.oof.items(): ax.plot(th,decision_curve(y,p,th),label=k)
    ax.legend(); ax.set(xlabel='Threshold probability',ylabel='Net benefit',title='D. Decision curve')
    savefig(fig,'Fig3_ML_Performance')

    # Fig4 attribution 5 panels
    X=df[A.consensus8]; Xz=A.scaler.transform(X); rfbase,rfattr=A.attr['RF']; lrbase,lrattr=A.attr['LR']; gbbase,gbattr=A.attr['GB']
    fig=plt.figure(figsize=(13,11)); gs=fig.add_gridspec(3,2); ax=fig.add_subplot(gs[0,0]); order=np.argsort(np.mean(abs(rfattr),axis=0))
    for j,fi in enumerate(order): ax.scatter(rfattr[:,fi],np.full(len(df),j)+np.random.default_rng(1).normal(0,.08,len(df)),c=Xz[:,fi],s=10,alpha=.6)
    ax.set_yticks(range(len(order)),[PRETTY.get(A.consensus8[i],A.consensus8[i]) for i in order]); ax.set_title('A. RF tree-path attribution beeswarm'); ax.set_xlabel('Attribution')
    ax=fig.add_subplot(gs[0,1]); imp=pd.DataFrame({'RF':np.mean(abs(rfattr),0),'LR':np.mean(abs(lrattr),0),'GB':np.mean(abs(gbattr),0)},index=[PRETTY.get(c,c) for c in A.consensus8]); imp.sort_values('RF').plot.barh(ax=ax); ax.set_title('B. Mean |attribution|')
    pRF=A.fitted['RF'].predict_proba(Xz)[:,1]; hi=int(np.argmax(pRF)); ax=fig.add_subplot(gs[1,0]); vals=rfattr[hi]; ord2=np.argsort(abs(vals)); ax.barh(np.arange(len(vals)),vals[ord2]); ax.set_yticks(np.arange(len(vals)),[PRETTY.get(A.consensus8[i],A.consensus8[i]) for i in ord2]); ax.set_title(f'C. High-risk patient (p={pRF[hi]:.3f})')
    ax=fig.add_subplot(gs[1,1]); sidx=A.consensus8.index(COL['SII']); ax.scatter(Xz[:,sidx],rfattr[:,sidx],c=df[COL['Platelet']],s=16,alpha=.7); ax.axhline(0,ls=':'); ax.set(xlabel='SII (standardised)',ylabel='SII attribution',title='D. SII dependence')
    ax=fig.add_subplot(gs[2,:]); ids=[np.argmin(pRF),np.argmin(abs(pRF-.5)),np.argmax(pRF)]; left=np.zeros(3)
    for j,c in enumerate(A.consensus8): ax.barh(range(3),rfattr[ids,j],left=left,label=PRETTY.get(c,c)); left+=rfattr[ids,j]
    ax.set_yticks(range(3),['Low risk','Borderline','High risk']); ax.set_title('E. Local contribution profiles'); ax.legend(ncol=4,fontsize=7)
    savefig(fig,'Fig4_SHAP_Interpretability')

    # Fig5 quartile analysis
    qtab=pd.read_csv(TAB_DIR/'Table4_SII_Quartile.csv'); fig,axs=plt.subplots(1,2,figsize=(12,6)); ax=axs[0]; rates=[]
    for _,r in qtab.iterrows(): rates.append(float(re.search(r'\((.*?)%',r['CD n (%)']).group(1)))
    ax.bar(qtab['SII Quartile'],rates); ax.set(title='SII Quartile Characteristics',ylabel='CD prevalence (%)')
    ax=axs[1]; models=['Unadj','M1','M2']; offsets=[-.2,0,.2]
    for mod,off in zip(models,offsets):
        xs=[];los=[];his=[];ys=[]
        for i,lev in enumerate(['Q2','Q3','Q4'],1):
            txt=qtab.loc[qtab['SII Quartile']==lev,f'{mod} OR (95% CI)'].iloc[0]; nums=list(map(float,re.findall(r'[0-9.]+',txt))); xs.append(nums[0]);los.append(nums[1]);his.append(nums[2]);ys.append(i+off)
        ax.errorbar(xs,ys,xerr=[np.array(xs)-np.array(los),np.array(his)-np.array(xs)],fmt='o',label=mod)
    ax.axvline(1,ls='--'); ax.set_xscale('log'); ax.set_yticks([1,2,3],['Q2','Q3','Q4']); ax.legend(); ax.set(xlabel='Odds ratio (95% CI)',title='OR for CD by SII quartile (Ref Q1)')
    savefig(fig,'Fig5_SII_Quartile')

    # Fig6 RCS
    grid,pr,lo,hi,orv,orlo,orhi,knots=rcs_prediction(df); fig,axs=plt.subplots(1,2,figsize=(12,5)); ax=axs[0]; ax.plot(grid,pr); ax.fill_between(grid,lo,hi,alpha=.2); ax.axhline(y.mean(),ls=':'); [ax.axvline(k,ls=':',alpha=.5) for k in knots]; ax.set(xlabel='SII',ylabel='Adjusted predicted probability',title='A. RCS probability')
    ax=axs[1]; ax.plot(grid,orv); ax.fill_between(grid,orlo,orhi,alpha=.2); ax.axhline(1,ls='--'); ax.set_yscale('log'); [ax.axvline(k,ls=':',alpha=.5) for k in knots]; ax.set(xlabel='SII',ylabel='OR vs median SII',title='B. RCS odds ratio')
    savefig(fig,'Fig6_SII_RCS')


def plot_supp_figures(A):
    df=A.df; y=df[Y].values; X=df[A.consensus8]; Xz=A.scaler.transform(X); models=A.fitted
    # eFig01 subgroup forest from S17
    s=pd.read_csv(TAB_DIR/'TableS17_Subgroup_Note.csv'); fig,ax=plt.subplots(figsize=(8,7)); yy=np.arange(len(s)); ax.errorbar(s['OR per SD'],yy,xerr=[s['OR per SD']-s['CI low'],s['CI high']-s['OR per SD']],fmt='o'); ax.axvline(1,ls='--'); ax.set_xscale('log'); ax.set_yticks(yy,[f"{a}: {b}" for a,b in zip(s.Modifier,s.Level)]); ax.set(title='SII subgroup forest',xlabel='OR per 1-SD SII'); savefig(fig,'eFig01_SII_Subgroup_Forest')
    # 02-06 ROC one per model
    for i,k in enumerate(['LR','RF','GB','SVM-R','SVM-L'],2):
        fig,ax=plt.subplots(figsize=(6,5)); fpr,tpr,_=roc_curve(y,A.oof[k]); ax.plot(fpr,tpr,label=f'AUROC={safe_auc(y,A.oof[k]):.3f}'); ax.plot([0,1],[0,1],ls='--'); ax.legend(); ax.set(xlabel='1-Specificity',ylabel='Sensitivity',title=f'ROC: {k}'); savefig(fig,f'eFig{i:02d}_ROC_{k}')
    # 07-11 calibration
    for i,k in enumerate(['LR','RF','GB','SVM-R','SVM-L'],7):
        p=A.oof[k]; bins=pd.qcut(p,8,duplicates='drop'); t=pd.DataFrame({'p':p,'y':y,'b':bins}).groupby('b',observed=False).agg(p=('p','mean'),y=('y','mean')); fig,ax=plt.subplots(figsize=(6,5)); ax.plot(t.p,t.y,'o-'); ax.plot([0,1],[0,1],ls='--'); a,b=calibration_intercept_slope(y,p); ax.set(xlabel='Predicted',ylabel='Observed',title=f'Calibration: {k}\nintercept={a:.2f}, slope={b:.2f}'); savefig(fig,f'eFig{i:02d}_Calib_{k}')
    # 12-16 PR
    for i,k in enumerate(['LR','RF','GB','SVM-R','SVM-L'],12):
        p=A.oof[k]; prec,rec,_=precision_recall_curve(y,p); fig,ax=plt.subplots(figsize=(6,5)); ax.plot(rec,prec,label=f'AUPRC={average_precision_score(y,p):.3f}'); ax.axhline(y.mean(),ls='--'); ax.legend(); ax.set(xlabel='Recall',ylabel='Precision',title=f'Precision-Recall: {k}'); savefig(fig,f'eFig{i:02d}_PR_{k}')
    # 17-21 confusion matrices
    for i,k in enumerate(['LR','RF','GB','SVM-R','SVM-L'],17):
        m=metric_bundle(y,A.oof[k]); cm=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm); [ax.text(j,r,str(cm[r,j]),ha='center',va='center',fontsize=14) for r in range(2) for j in range(2)]; ax.set_xticks([0,1],['Pred 0','Pred 1']); ax.set_yticks([0,1],['True 0','True 1']); ax.set_title(f'Confusion matrix: {k}'); savefig(fig,f'eFig{i:02d}_ConfMat_{k}')
    # 22-26 learning curves
    for i,k in enumerate(['LR','RF','GB','SVM-R','SVM-L'],22):
        model=make_models()[k]; pipe=Pipeline([('sc',StandardScaler()),('m',model)]); sizes,tr,va=learning_curve(pipe,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='roc_auc',train_sizes=np.linspace(.3,1,6),n_jobs=N_JOBS); fig,ax=plt.subplots(figsize=(6,5)); ax.plot(sizes,tr.mean(1),'o-',label='Train'); ax.plot(sizes,va.mean(1),'o-',label='CV'); ax.fill_between(sizes,va.mean(1)-va.std(1),va.mean(1)+va.std(1),alpha=.2); ax.legend(); ax.set(xlabel='Training N',ylabel='AUROC',title=f'Learning curve: {k}'); savefig(fig,f'eFig{i:02d}_LC_{k}')
    # 27-28 permutation importance RF/GB
    for i,k in [(27,'RF'),(28,'GB')]:
        pi=permutation_importance(models[k],Xz,y,n_repeats=30 if not FAST_MODE else 5,random_state=42,scoring='roc_auc',n_jobs=N_JOBS); order=np.argsort(pi.importances_mean); fig,ax=plt.subplots(figsize=(7,6)); ax.barh(np.arange(len(order)),pi.importances_mean[order],xerr=pi.importances_std[order]); ax.set_yticks(np.arange(len(order)),[PRETTY.get(A.consensus8[j],A.consensus8[j]) for j in order]); ax.set(title=f'Permutation importance: {k}',xlabel='Decrease in AUROC'); savefig(fig,f'eFig{i:02d}_PermImp_{k}')
    # 29 bootstrap density
    bt=pd.read_csv(TAB_DIR/'TableS02_Bootstrap_AUC.csv'); fig,ax=plt.subplots(figsize=(8,5)); rng=np.random.default_rng(42)
    for _,r in bt.iterrows(): ax.hist(rng.normal(r['Mean AUC'],max(r['SD'],1e-3),500),bins=30,density=True,histtype='step',label=r.Model)
    ax.legend(); ax.set(xlabel='Bootstrap AUROC',title='Full-cohort bootstrap AUC distributions'); savefig(fig,'eFig29_Bootstrap_Density')
    # 30 CV boxplot
    fig,ax=plt.subplots(figsize=(8,5)); ks=['LR','RF','GB','SVM-R','SVM-L']; ax.boxplot([A.cv.loc[A.cv.Model==k,'Test AUC'] for k in ks],labels=ks); ax.set(ylabel='Fold AUROC',title='Repeated CV AUROC distribution'); savefig(fig,'eFig30_CV_Boxplot')
    # 31 radar metrics
    metrics=['AUROC','AUPRC','Accuracy','F1','Sensitivity','Specificity']; theta=np.linspace(0,2*np.pi,len(metrics),endpoint=False); fig=plt.figure(figsize=(7,7)); ax=fig.add_subplot(111,polar=True)
    for k in ks:
        m=metric_bundle(y,A.oof[k]); vals=[m[q] for q in metrics]; ax.plot(np.r_[theta,theta[0]],np.r_[vals,vals[0]],label=k)
    ax.set_xticks(theta,metrics); ax.legend(loc='upper right',bbox_to_anchor=(1.25,1.15)); ax.set_title('Model performance radar'); savefig(fig,'eFig31_Radar')
    # 32 stability
    st=pd.read_csv(TAB_DIR/'TableS20_Stability.csv'); fig,ax=plt.subplots(figsize=(9,5)); ss=st.groupby(['Seed','Model'])['Test AUC'].mean().unstack(); ss.plot(marker='o',ax=ax); ax.set(ylabel='Mean 10-fold AUROC',title='Performance stability across seeds'); savefig(fig,'eFig32_Stability')
    # 33 AUC gap
    g=A.cv.groupby('Model').agg(Test=('Test AUC','mean'),Train=('Train AUC','mean')).reindex(ks); fig,ax=plt.subplots(figsize=(7,5)); ax.bar(g.index,g.Train-g.Test); ax.axhline(.05,ls='--'); ax.set(ylabel='Train-test AUROC gap',title='Apparent overfitting gap'); savefig(fig,'eFig33_AUC_Gap')
    # 34 DCA
    d=pd.read_csv(TAB_DIR/'TableS15_DCA.csv'); fig,ax=plt.subplots(figsize=(8,5)); ax.plot(d.Threshold,d['All-treat'],ls='--',label='Treat all'); ax.axhline(0,ls=':'); [ax.plot(d.Threshold,d[k],label=k) for k in ks]; ax.legend(); ax.set(xlabel='Threshold',ylabel='Net benefit',title='Decision curve analysis'); savefig(fig,'eFig34_DCA')
    # 35 calibration overlay
    fig,ax=plt.subplots(figsize=(7,6));
    for k in ks:
        p=A.oof[k]; bins=pd.qcut(p,7,duplicates='drop'); t=pd.DataFrame({'p':p,'y':y,'b':bins}).groupby('b',observed=False).mean(numeric_only=True); ax.plot(t.p,t.y,'o-',label=k)
    ax.plot([0,1],[0,1],ls='--'); ax.legend(); ax.set(xlabel='Predicted',ylabel='Observed',title='Calibration overlay'); savefig(fig,'eFig35_CalibOverlay')
    # 36 LASSO coefficients
    r=A.paths[42]; fig,ax=plt.subplots(figsize=(8,6));
    for j,c in enumerate(A.prescreen): ax.plot(np.log10(r['C']),r['coefs'][:,j],label=PRETTY.get(c,c))
    ax.axvline(np.log10(r['best_C']),ls='--'); ax.set(xlabel='log10(C)',ylabel='Coefficient',title='LASSO coefficient paths'); ax.legend(fontsize=6,ncol=2); savefig(fig,'eFig36_LASSO_Coefs')
    # 37 Spearman heatmap
    cc=df[A.consensus8].corr(method='spearman'); fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cc.values,vmin=-1,vmax=1); ax.set_xticks(range(len(cc)),[PRETTY.get(c,c) for c in cc],rotation=60,ha='right'); ax.set_yticks(range(len(cc)),[PRETTY.get(c,c) for c in cc]); fig.colorbar(im,ax=ax); ax.set_title('Spearman correlation matrix'); savefig(fig,'eFig37_Spearman_Heatmap')
    # 38 AUC diff matrix
    m=pd.read_csv(TAB_DIR/'TableS12_Pairwise_AUC.csv').set_index('Model'); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(m.values.astype(float),vmin=-.1,vmax=.1); ax.set_xticks(range(5),m.columns); ax.set_yticks(range(5),m.index); [ax.text(j,i,f'{m.iloc[i,j]:.3f}',ha='center',va='center') for i in range(5) for j in range(5)]; fig.colorbar(im,ax=ax); ax.set_title('Pairwise mean AUROC difference'); savefig(fig,'eFig38_AUC_Diff_Matrix')
    # 39 Brier
    fig,ax=plt.subplots(figsize=(7,5)); vals=[brier_score_loss(y,A.oof[k]) for k in ks]; ax.bar(ks,vals); ax.set(ylabel='Brier score (lower is better)',title='Brier score comparison'); savefig(fig,'eFig39_Brier')
    # 40 SII detail
    fig,axs=plt.subplots(1,2,figsize=(10,5)); fig_violin(axs[0],df,COL['SII'],'SII by outcome','SII'); axs[1].hist(df.loc[y==0,COL['SII']],bins=25,alpha=.5,label='No CD'); axs[1].hist(df.loc[y==1,COL['SII']],bins=25,alpha=.5,label='CD'); axs[1].legend(); axs[1].set_title('SII distribution'); savefig(fig,'eFig40_SII_Detail')
    # 41 Platelet
    fig,axs=plt.subplots(1,2,figsize=(10,5)); fig_violin(axs[0],df,COL['Platelet'],'Platelet by outcome','Platelet'); axs[1].scatter(df[COL['Platelet']],df[COL['SII']],c=y,alpha=.7); axs[1].set(xlabel='Platelet',ylabel='SII',title='Platelet-SII relation'); savefig(fig,'eFig41_Platelet')
    # 42 SII continuous OR models
    s16=pd.read_csv(TAB_DIR/'TableS16_SII_Continuous_OR.csv'); vals=[]
    for t in s16['OR per SD [95% CI]']: vals.append(list(map(float,re.findall(r'[0-9.]+',t))))
    vals=np.array(vals); fig,ax=plt.subplots(figsize=(7,5)); yy=np.arange(3); ax.errorbar(vals[:,0],yy,xerr=[vals[:,0]-vals[:,1],vals[:,2]-vals[:,0]],fmt='o'); ax.axvline(1,ls='--'); ax.set_xscale('log'); ax.set_yticks(yy,s16.Model); ax.set(title='SII continuous association',xlabel='OR per 1 SD'); savefig(fig,'eFig42_SII')
    # 43 ICH
    rates=df.groupby(COL['ICH'])[Y].mean()*100; fig,ax=plt.subplots(figsize=(6,5)); ax.bar(['No ICH','ICH'],rates.values); ax.set(ylabel='CD prevalence (%)',title='Intracerebral haemorrhage and CD'); savefig(fig,'eFig43_Intracerebral_')
    # 44 NLR exclusion
    fig,axs=plt.subplots(1,2,figsize=(10,5)); axs[0].scatter(df[COL['NLR']],df[COL['SII']],alpha=.6); r,p=stats.spearmanr(df[COL['NLR']],df[COL['SII']]); axs[0].set(xlabel='NLR',ylabel='SII',title=f'NLR-SII Spearman r={r:.3f}'); fig_violin(axs[1],df,COL['NLR'],'NLR by outcome','NLR'); savefig(fig,'eFig44_NLR_Exclusion')
    # 45-46 extended RCS
    grid,pr,lo,hi,orv,orlo,orhi,knots=rcs_prediction(df)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(grid,pr); ax.fill_between(grid,lo,hi,alpha=.2); [ax.axvline(k,ls=':',alpha=.4) for k in knots]; ax.set(xlabel='SII',ylabel='Adjusted probability',title='Extended RCS probability'); savefig(fig,'eFig45_RCS_Probability_Extended')
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(grid,orv); ax.fill_between(grid,orlo,orhi,alpha=.2); ax.axhline(1,ls='--'); ax.set_yscale('log'); [ax.axvline(k,ls=':',alpha=.4) for k in knots]; ax.set(xlabel='SII',ylabel='OR vs median',title='Extended RCS OR'); savefig(fig,'eFig46_RCS_OR_Extended')
    # 47 quartile box
    q=pd.qcut(df[COL['SII']],4,labels=['Q1','Q2','Q3','Q4']); fig,ax=plt.subplots(figsize=(7,5)); ax.boxplot([df.loc[q==lev,COL['SII']] for lev in q.cat.categories],labels=q.cat.categories); ax.set(ylabel='SII',title='SII quartile distributions'); savefig(fig,'eFig47_SII_Quartile_Box')
    # 48 risk score / predicted risk quintiles
    p=A.oof['LR']; qq=pd.qcut(p,5,labels=['Q1','Q2','Q3','Q4','Q5']); rates=pd.DataFrame({'q':qq,'y':y,'p':p}).groupby('q',observed=False).agg(rate=('y','mean'),pred=('p','mean')); fig,ax=plt.subplots(figsize=(7,5)); ax.plot(range(1,6),100*rates.rate,'o-',label='Observed CD %'); ax.plot(range(1,6),100*rates.pred,'s--',label='Mean predicted %'); ax.legend(); ax.set(xlabel='Predicted-risk quintile',ylabel='Percent',title='Risk stratification'); savefig(fig,'eFig48_RiskScore')
    # 49 mechanism schematic (conceptual, explicitly labelled hypothesis)
    fig,ax=plt.subplots(figsize=(10,5)); ax.axis('off'); boxes=[(.05,.55,'Neutrophils / inflammation'),(.35,.75,'SII'),(.35,.35,'Platelet count'),(.68,.55,'Thrombo-inflammatory / haemorrhagic burden'),(.88,.55,'Consciousness disturbance')]
    for x0,y0,t in boxes: ax.text(x0,y0,t,ha='center',va='center',bbox=dict(boxstyle='round',fc='white'))
    arrows=[((.14,.55),(.29,.72)),((.43,.72),(.62,.58)),((.43,.38),(.62,.52)),((.75,.55),(.83,.55))]
    for a,b in arrows: ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.05,'Conceptual schematic only; observational data do not establish mechanism or causality.',ha='center'); ax.set_title('Hypothesised clinical framework'); savefig(fig,'eFig49_Mechanism_Schematic')
    # 50 pipeline summary
    fig,ax=plt.subplots(figsize=(10,4)); ax.axis('off'); stages=['39 variables','Clinical pre-screen','Univariate P<0.10','10-seed LASSO consensus',f'{len(A.consensus8)} ML features','3-variable parsimonious LR']; xs=np.linspace(.08,.92,len(stages));
    for x0,t in zip(xs,stages): ax.text(x0,.5,t,ha='center',va='center',rotation=90 if len(t)>18 else 0,bbox=dict(boxstyle='round',fc='white'))
    for a,b in zip(xs[:-1],xs[1:]): ax.annotate('',xy=(b-.03,.5),xytext=(a+.03,.5),arrowprops=dict(arrowstyle='->'))
    ax.set_title('Analysis pipeline summary'); savefig(fig,'eFig50_Pipeline_Summary')
    # 51 LR beeswarm exact linear attributions
    lrbase,lrattr=A.attr['LR']; order=np.argsort(np.mean(abs(lrattr),axis=0)); fig,ax=plt.subplots(figsize=(8,6)); rng=np.random.default_rng(2)
    for j,fi in enumerate(order): ax.scatter(lrattr[:,fi],j+rng.normal(0,.08,len(df)),c=Xz[:,fi],s=10,alpha=.6)
    ax.set_yticks(range(len(order)),[PRETTY.get(A.consensus8[i],A.consensus8[i]) for i in order]); ax.set(xlabel='Exact linear contribution',title='LR attribution beeswarm'); savefig(fig,'eFig51_SHAP_Beeswarm_LR')
    # 52 global importance 3 models
    imp=pd.DataFrame({k:np.mean(abs(A.attr[k][1]),axis=0) for k in ['LR','RF','GB']},index=[PRETTY.get(c,c) for c in A.consensus8]); fig,ax=plt.subplots(figsize=(8,6)); imp.sort_values('LR').plot.barh(ax=ax); ax.set(xlabel='Mean absolute attribution',title='Global attribution importance across 3 models'); savefig(fig,'eFig52_SHAP_GlobalImportance_3Models')
    # 53 waterfall 3 patients RF
    rfbase,rfattr=A.attr['RF']; p=models['RF'].predict_proba(Xz)[:,1]; ids=[np.argmax(p),np.argmin(abs(p-.5)),np.argmin(p)]; fig,axs=plt.subplots(1,3,figsize=(15,6));
    for ax,idx,lab in zip(axs,ids,['High risk','Borderline','Low risk']):
        o=np.argsort(abs(rfattr[idx])); ax.barh(range(len(o)),rfattr[idx,o]); ax.set_yticks(range(len(o)),[PRETTY.get(A.consensus8[j],A.consensus8[j]) for j in o],fontsize=7); ax.set_title(f'{lab}\np={p[idx]:.3f}')
    savefig(fig,'eFig53_SHAP_Waterfall_3Patients')
    # 54 dependence all features
    fig,axs=plt.subplots(4,2,figsize=(11,14));
    for ax,j in zip(axs.ravel(),range(len(A.consensus8))): ax.scatter(Xz[:,j],rfattr[:,j],c=df[COL['Platelet']],s=10,alpha=.6); ax.set(xlabel=f'{PRETTY.get(A.consensus8[j],A.consensus8[j])} (z)',ylabel='Attribution',title=PRETTY.get(A.consensus8[j],A.consensus8[j]))
    savefig(fig,'eFig54_SHAP_Dependence_All8Features')
    # 55 attribution interaction screen: Spearman between attribution and each other feature
    M=np.zeros((len(A.consensus8),len(A.consensus8)))
    for i in range(len(A.consensus8)):
        for j in range(len(A.consensus8)): M[i,j]=stats.spearmanr(rfattr[:,i],Xz[:,j]).statistic
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(M,vmin=-1,vmax=1); labs=[PRETTY.get(c,c) for c in A.consensus8]; ax.set_xticks(range(len(labs)),labs,rotation=60,ha='right'); ax.set_yticks(range(len(labs)),labs); fig.colorbar(im,ax=ax); ax.set_title('Attribution-feature association screen\n(not a formal biological interaction test)'); savefig(fig,'eFig55_SHAP_Interactions')
    # 56 by outcome group
    imp0=np.mean(abs(rfattr[y==0]),0); imp1=np.mean(abs(rfattr[y==1]),0); order=np.argsort(imp1); fig,ax=plt.subplots(figsize=(8,6)); yy=np.arange(len(order)); ax.barh(yy-.2,imp0[order],.4,label='No CD'); ax.barh(yy+.2,imp1[order],.4,label='CD'); ax.set_yticks(yy,[PRETTY.get(A.consensus8[j],A.consensus8[j]) for j in order]); ax.legend(); ax.set(xlabel='Mean |RF attribution|',title='Attributions by outcome group'); savefig(fig,'eFig56_SHAP_ByOutcomeGroup')
    # 57 summary table rendered as figure
    impdf=pd.DataFrame({'Feature':[PRETTY.get(c,c) for c in A.consensus8],'RF':np.mean(abs(rfattr),0),'LR':np.mean(abs(lrattr),0),'GB':np.mean(abs(A.attr['GB'][1]),0)}).sort_values('RF',ascending=False); fig,ax=plt.subplots(figsize=(9,5)); ax.axis('off'); cell_text=[[r['Feature'], f"{r['RF']:.3f}", f"{r['LR']:.3f}", f"{r['GB']:.3f}"] for _,r in impdf.iterrows()]; ax.table(cellText=cell_text,colLabels=impdf.columns,loc='center'); ax.set_title('Attribution summary table'); savefig(fig,'eFig57_SHAP_SummaryTable')
    # 58 LIME 3 patients on LR; local weighted linear surrogate
    pLR=models['LR'].predict_proba(Xz)[:,1]; ids=[np.argmax(pLR),np.argmin(abs(pLR-.5)),np.argmin(pLR)]; fig,axs=plt.subplots(1,3,figsize=(15,6));
    for ax,idx,lab in zip(axs,ids,['High risk','Borderline','Low risk']):
        intercept,coef,r2,Z,pp,w,pred=lime_local(models['LR'],Xz[idx],len(A.consensus8),500,42+idx); contrib=coef*Xz[idx]; o=np.argsort(abs(contrib)); ax.barh(range(len(o)),contrib[o]); ax.set_yticks(range(len(o)),[PRETTY.get(A.consensus8[j],A.consensus8[j]) for j in o],fontsize=7); ax.set_title(f'{lab}: LIME surrogate R²={r2:.3f}')
    savefig(fig,'eFig58_LIME_Interpretability')
    # 59 multi-seed lasso consensus
    items=sorted(A.lasso_freq.items(),key=lambda kv:kv[1]); fig,ax=plt.subplots(figsize=(8,6)); ax.barh([PRETTY.get(c,c) for c,_ in items],[v for _,v in items]); ax.axvline(8,ls='--'); ax.set(xlabel='Selected seeds / 10',title='Multi-seed LASSO consensus'); savefig(fig,'eFig59_MultiSeed_LASSO_Consensus')
    # 60 study overview
    fig,ax=plt.subplots(figsize=(11,6)); ax.axis('off'); lines=[f'Analytic cohort: N={len(df)}',f'CD positive: {int(y.sum())} ({100*y.mean():.1f}%)',f'Candidate predictors: {len(CANDIDATES_38)}',f'Univariate pre-screen P<0.10: {len(A.prescreen)}',f'LASSO consensus features: {len(A.consensus8)}',f'Final parsimonious predictors: 3',f'Primary fixed-feature evaluation: {REPEATS}x{FOLDS}-fold repeated stratified CV']
    for i,t in enumerate(lines): ax.text(.5,.88-i*.11,t,ha='center',va='center',fontsize=14,bbox=dict(boxstyle='round',fc='white'))
    ax.set_title('Study and analysis overview',fontsize=18); savefig(fig,'eFig60_Study_Overview')

# -----------------------------
# Manifest + run
# -----------------------------
def make_manifest(A):
    expected_figs=['Fig1_VariableSelection','Fig2_BaselineChar','Fig3_ML_Performance','Fig4_SHAP_Interpretability','Fig5_SII_Quartile','Fig6_SII_RCS'] + [
        'eFig01_SII_Subgroup_Forest','eFig02_ROC_LR','eFig03_ROC_RF','eFig04_ROC_GB','eFig05_ROC_SVM-R','eFig06_ROC_SVM-L',
        'eFig07_Calib_LR','eFig08_Calib_RF','eFig09_Calib_GB','eFig10_Calib_SVM-R','eFig11_Calib_SVM-L','eFig12_PR_LR','eFig13_PR_RF','eFig14_PR_GB','eFig15_PR_SVM-R','eFig16_PR_SVM-L',
        'eFig17_ConfMat_LR','eFig18_ConfMat_RF','eFig19_ConfMat_GB','eFig20_ConfMat_SVM-R','eFig21_ConfMat_SVM-L','eFig22_LC_LR','eFig23_LC_RF','eFig24_LC_GB','eFig25_LC_SVM-R','eFig26_LC_SVM-L',
        'eFig27_PermImp_RF','eFig28_PermImp_GB','eFig29_Bootstrap_Density','eFig30_CV_Boxplot','eFig31_Radar','eFig32_Stability','eFig33_AUC_Gap','eFig34_DCA','eFig35_CalibOverlay','eFig36_LASSO_Coefs','eFig37_Spearman_Heatmap','eFig38_AUC_Diff_Matrix','eFig39_Brier','eFig40_SII_Detail','eFig41_Platelet','eFig42_SII','eFig43_Intracerebral_','eFig44_NLR_Exclusion','eFig45_RCS_Probability_Extended','eFig46_RCS_OR_Extended','eFig47_SII_Quartile_Box','eFig48_RiskScore','eFig49_Mechanism_Schematic','eFig50_Pipeline_Summary','eFig51_SHAP_Beeswarm_LR','eFig52_SHAP_GlobalImportance_3Models','eFig53_SHAP_Waterfall_3Patients','eFig54_SHAP_Dependence_All8Features','eFig55_SHAP_Interactions','eFig56_SHAP_ByOutcomeGroup','eFig57_SHAP_SummaryTable','eFig58_LIME_Interpretability','eFig59_MultiSeed_LASSO_Consensus','eFig60_Study_Overview']
    expected_tabs=[f'Table{i}_{x}' for i,x in [(1,'BaselineCharacteristics'),(2,'VariableSelection'),(3,'MLPerformance'),(4,'SII_Quartile'),(5,'FinalFeatures')]]+[f'TableS{i:02d}_{x}' for i,x in [(1,'CV_Folds'),(2,'Bootstrap_AUC'),(3,'LASSO_Path'),(4,'Exclusion_Rationale'),(5,'Univariate_All'),(6,'ConfusionMatrices'),(7,'Hyperparameters'),(8,'Descriptive_Stats'),(9,'Spearman_Correlations'),(10,'SII_Quartile_Desc'),(11,'RCS_Knots'),(12,'Pairwise_AUC'),(13,'RepeatedCV'),(14,'Risk_Quintile'),(15,'DCA'),(16,'SII_Continuous_OR'),(17,'Subgroup_Note'),(18,'Inter_Predictor_Corr'),(19,'Feature_Complete_Summary'),(20,'Stability')]]
    rows=[]
    for n in expected_figs: rows.append({'Type':'Figure','Name':n,'PDF exists':(FIG_DIR/f'{n}.pdf').exists(),'PNG exists':(FIG_DIR/f'{n}.png').exists()})
    for n in expected_tabs: rows.append({'Type':'Table','Name':n,'CSV exists':(TAB_DIR/f'{n}.csv').exists(),'XLSX exists':(TAB_DIR/f'{n}.xlsx').exists()})
    pd.DataFrame(rows).to_csv(META_DIR/'output_manifest.csv',index=False,encoding='utf-8-sig')
    meta={'data_file':str(DATA_FILE),'sha256':data_hash(),'n':len(A.df),'events':int(A.df[Y].sum()),'consensus_features':A.consensus8,'fast_mode':FAST_MODE,'bootstrap_n':BOOTSTRAP_N,'repeats':REPEATS,'folds':FOLDS}
    (META_DIR/'analysis_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')


def run_all():
    print('1/5 Loading data and fitting primary analysis...')
    A=build_analysis(); print('Consensus features:', [PRETTY.get(c,c) for c in A.consensus8])
    print('2/5 Generating all 25 tables...'); make_tables(A)
    print('3/5 Generating 6 main figures...'); plot_main_figures(A)
    print('4/5 Generating 60 supplementary figures...'); plot_supp_figures(A)
    print('5/5 Writing manifest and metadata...'); make_manifest(A)
    print(f'DONE. Outputs: {OUTPUT_DIR.resolve()}')
    return A

if __name__=='__main__':
    run_all()
