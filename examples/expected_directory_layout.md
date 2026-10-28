# Expected directory layout

```
<full_root>/
├── metadata.csv
└── dce/
    ├── <pid>/
    │   ├── <pid>_spy2_vis1_dce_aqc_0.nii.gz   # pre-contrast
    │   ├── <pid>_spy2_vis1_dce_aqc_1.nii.gz   # first post-contrast
    │   └── ...
    └── ...
<mask_dir>/
    ├── <pid>.nii.gz            # or <pid>_mask.nii.gz / <pid>_spy2_vis1_mask.nii.gz
    └── ...
```

Timepoints are discovered from the `_aqc_<t>` suffix. Masks must match the
reference phase (`post_early`, else the second phase) in shape, spacing,
orientation and affine. Prepare one such tree per cohort (I-SPY 2, I-SPY 1, Duke).

`metadata.csv` columns: `pid`, `pCR` (0/1), `test` (I-SPY 2 only: 1/2 = held
out), `age`, `hr`, `her2` (TNBC is derived if no `triple_neg` column), optional
`pre` / `post_early` / `post_late` phase indices, optional acquisition times
(`timepoints_sec` as a list, or `tp<t>_sec` columns), optional `tr_ms`,
`flip_angle_deg`, `t10_ms` for SPGR conversion. See `examples/metadata_example_synthetic.csv`.
