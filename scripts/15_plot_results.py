#!/usr/bin/env python3
"""Step 15: main results figure (AUROC, anchoring, unsupported numbers, SHAP contradictions).
Usage: 15_plot_results.py <evaluation_dir> <cases.jsonl> <runs_dir> <output_stem>
"""
import pandas as pd, numpy as np, json, ast, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse, glob, os
_ap = argparse.ArgumentParser(description="Main results figure (step 15)")
for _a in ("eval_dir", "cases", "runs", "out"):
    _ap.add_argument(_a)
_ap.add_argument("--providers", nargs="+", default=["zhipu", "siliconflow"], help="provider order (max 2 shown)")
_ap.add_argument("--scatter_provider", default=None, help="provider for panel (b); default: first one")
_A = _ap.parse_args()
E, CASES, RUNS, OUT = _A.eval_dir, _A.cases, _A.runs, _A.out
import os as _os; _os.makedirs(_os.path.dirname(_os.path.abspath(OUT)), exist_ok=True)
PROVS = _A.providers[:2]
RUNDIR = {pr: sorted(glob.glob(os.path.join(RUNS, pr + "__*")))[0] for pr in PROVS}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
p=pd.read_csv(E+'/prediction_metrics.csv'); f=pd.read_csv(E+'/faithfulness_metrics.csv')
_PAL = ['#2a78d6', '#eb6834']; _MK = ['o', 's']
_NAMES = {'zhipu': 'GLM-4.5V (Zhipu)', 'siliconflow': 'Qwen3-VL-32B-Instruct (SiliconFlow)', 'mock': 'mock model'}
C = {pr: _PAL[i] for i, pr in enumerate(PROVS)}; MK = {pr: _MK[i] for i, pr in enumerate(PROVS)}
NAME = {pr: _NAMES.get(pr, pr) for pr in PROVS}
SP = _A.scatter_provider or PROVS[0]
cfgs=['M1','M2','M3f','M3','M4','F3f']; lab=['M1','M2','M3f','M3','M4','B1+M3f\nfusion']
fig,ax=plt.subplots(2,2,figsize=(7.16,5.4))
a=ax[0,0]
b1=p[(p.config=='B1')&(p.split=='heldout_test')].iloc[0]; lo,hi=ast.literal_eval(b1.auroc_ci)
a.axhspan(lo,hi,color='#e8e7e3',zorder=0); a.axhline(b1.auroc,color='#52514e',lw=1.2,ls='--',zorder=1)
a.text(-0.45,hi+0.005,f'B1 reference: {b1.auroc:.3f} (shaded: 95% CI)',ha='left',va='bottom',fontsize=7,color='#52514e')
for j,pr in enumerate(PROVS):
    for i,c in enumerate(cfgs):
        r=p[(p.provider==pr)&(p.config==c)&(p.split=='heldout_test')].iloc[0]; l,h=ast.literal_eval(r.auroc_ci)
        x=i+((-0.13 if j==0 else 0.13) if len(PROVS)>1 else 0)
        a.plot([x,x],[l,h],color=C[pr],lw=1.5,zorder=2); a.plot(x,r.auroc,MK[pr],color=C[pr],ms=6,mec='#fcfcfb',mew=1,zorder=3)
a.set_xticks(range(len(cfgs))); a.set_xticklabels(lab); a.set_ylabel('Held-out AUROC (95% CI)'); a.set_ylim(0.5,0.86); a.set_xlim(-0.5,5.5)
a.set_title('(a) Discrimination, held-out test',loc='left',fontsize=8.5,fontweight='bold'); a.grid(axis='y',color='#e8e7e3',lw=0.6)
a=ax[0,1]
cases={json.loads(l)['pid']:json.loads(l) for l in open(CASES)}
for cfg,fill in [('M3f',False),('M3',True)]:
    d=pd.read_csv(os.path.join(RUNDIR[SP], f'{cfg}_heldout_test.csv')).groupby('pid').probability.mean().reset_index()
    d['b1']=d.pid.map(lambda q:cases[q]['xgb_probability'])
    a.plot(d.b1,d.probability,'o',ms=4,mfc=C[SP] if fill else 'none',mec=C[SP],mew=0.8,
           label='M3 (B1 probability shown)' if fill else 'M3f (features only)')
a.plot([0,1],[0,1],color='#52514e',lw=0.8,ls='--'); a.set_xlim(0,1); a.set_ylim(0,1)
a.set_xlabel('B1 predicted probability'); a.set_ylabel(NAME[SP].split(' (')[0] + ' predicted probability')
a.set_title('(b) Anchoring on B1 (' + NAME[SP].split(' (')[0] + ', test)',loc='left',fontsize=8.5,fontweight='bold'); a.legend(frameon=False,fontsize=7,loc='upper left')
for k,(col,title,ylab) in enumerate([('hallucination_rate','(c) Unsupported numbers (development)','% of numeric values'),
                                    ('contradiction_rate','(d) Direction vs. SHAP sign (development)','% of directional evidence items')]):
    a=ax[1,k]; cf=['M1','M2','M3f','M3','M4']
    for j,pr in enumerate(PROVS):
        r=f[(f.provider==pr)&(f.split=='development')].set_index('config').loc[cf]
        lo_=col.replace('rate','ci_low'); hi_=col.replace('rate','ci_high')
        x=np.arange(len(cf))+((-0.19 if j==0 else 0.19) if len(PROVS)>1 else 0)
        a.bar(x,100*r[col],width=0.36,color=C[pr],edgecolor='#fcfcfb',linewidth=1)
        a.errorbar(x,100*r[col],yerr=[100*(r[col]-r[lo_]),100*(r[hi_]-r[col])],fmt='none',ecolor='#0b0b0b',elinewidth=0.8,capsize=2)
        if k==0:
            for xi,v,h in zip(x,100*r[col],100*r[hi_]): a.text(xi,h+0.12,f'{v:.1f}',ha='center',va='bottom',fontsize=6.5,color='#52514e')
    a.set_xticks(range(len(cf))); a.set_xticklabels(cf); a.set_ylabel(ylab); a.grid(axis='y',color='#e8e7e3',lw=0.6); a.set_axisbelow(True)
    a.set_title(title,loc='left',fontsize=8.5,fontweight='bold')
ax[1,0].set_ylim(0,6.2); ax[1,1].set_ylim(0,30)
h=[Line2D([],[],marker=MK[pr],color=C[pr],ls='',ms=6,label=NAME[pr]) for pr in PROVS]
fig.legend(handles=h,loc='upper center',ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(OUT+'.pdf'); fig.savefig(OUT+'.png',dpi=160)
