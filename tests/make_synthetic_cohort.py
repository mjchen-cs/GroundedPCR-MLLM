"""
Synthetic DCE cohort generator for the smoke test (NOT real data).

usage: python tests/make_synthetic_cohort.py ROOT N_CASES SEED MIN_PHASES MAX_PHASES_EXCL N_HELDOUT
"""
import numpy as np, nibabel as nib, pandas as pd, sys, os
root=sys.argv[1]; n=int(sys.argv[2]); seed=int(sys.argv[3]); ntr=(int(sys.argv[4]),int(sys.argv[5])); ntest=int(sys.argv[6])
rng=np.random.default_rng(seed)
os.makedirs(f"{root}/data/dce",exist_ok=True); os.makedirs(f"{root}/masks",exist_ok=True)
rows=[]
for i in range(n):
    pid=f"{os.path.basename(root)}_{i:03d}"; d=f"{root}/data/dce/{pid}"; os.makedirs(d,exist_ok=True)
    hr=int(rng.random()<0.5); her2=int(rng.random()<0.3)
    logit=-0.8-1.0*hr+0.8*her2+rng.normal(0,0.8); y=int(rng.random()<1/(1+np.exp(-logit)))
    r=int(rng.integers(4,9)) - (1 if y else 0)
    zz,yy,xx=np.mgrid[:24,:24,:24]; m=((zz-12)**2+(yy-12)**2+(xx-12)**2)<=r*r
    nt=int(rng.integers(*ntr)); aff=np.diag([0.8,0.8,1.5,1])
    dist=np.sqrt((zz-12)**2+(yy-12)**2+(xx-12)**2)
    for t in range(nt):
        kin=(1+0.6*y+0.3*(dist/r))
        enh=0 if t==0 else kin*(1-np.exp(-t/1.5))*np.exp(-(0.05+0.08*y)*t)
        img=100+rng.normal(0,3,m.shape)+m*100*enh
        nib.save(nib.Nifti1Image(img.astype(np.float32),aff),f"{d}/{pid}_spy2_vis1_dce_aqc_{t}.nii.gz")
    nib.save(nib.Nifti1Image(m.astype(np.uint8),aff),f"{root}/masks/{pid}.nii.gz")
    rows.append(dict(pid=pid,pCR=y,test=1 if i>=n-ntest else 0,age=int(rng.integers(30,70)),hr=hr if rng.random()>0.03 else np.nan,her2=her2,n_times=nt,pre=0,post_early=1,post_late=nt-1))
pd.DataFrame(rows).to_csv(f"{root}/data/metadata.csv",index=False)
