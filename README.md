# Inclusive Ridge — replication code for Section 5

Scripts that reproduce Tables 1–7 of *Inclusive Ridge: Adaptive Shrinkage
with Applications to Causal Inference*, built on the reusable solver
`ir\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_solver.py`.

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `scikit-learn`,
`scikit-image`:

```bash
pip install numpy scipy pandas scikit-learn scikit-image
```

## Data

|Table|Data|Source|
|-|-|-|
|1, 2, 4–6|simulated|generated in the scripts|
|3, 4|Ames house prices, `train.csv` (1,460 × 81)|Kaggle, *House Prices: Advanced Regression Techniques*|

## Running

```bash
python table1\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_phillips.py                # expected risk (24), as in the paper  (example: C\\\\path\\\_to\\\_file\\\_directory> \*\*python Phillips.py\*\*)
python table2\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_tomography.py
python tables3\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_4\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_ames.py --data train.csv   (example: C:\\\\path\\\_to-files>\*\*python tables3\\\_4\\\_ames.py --data train.csv\*\*)
python table5\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_sdid\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_mc.py --reps 500 --workers 8 --prop99 california\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_prop99.csv
python table6\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_exterior.py --reps 300
```

Approximate run times on one core: Tables 1–4  under a minute each;
Table 6 about 3 minutes; Table 5 about 15 minutes for 4 × 500 replications on
two cores.

## Saved output

`results/` holds the output of the runs summarized below.

At the command prompt: C:\\path\_to-files>python tables3\_4\_ames.py --data train.csv

## Files

|File|Contents|
|-|-|
|`ir\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_solver.py`|`svd`, `ir`, `irmse`, `asymir`, `asymmse`, `asymcov`, noise estimation, Picard index, EGCV, grid tuners|
|`table1\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_phillips.py`|Phillips test problem (Hansen's Regularization Tools, ported) and oracle tuning|
|`table2\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_tomography.py`|Radon-transform operator (32 × 32 image, 24 angles)|
|`tables3\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_4\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_ames.py`|Ames preprocessing, noise estimates, tail F-test, Tables 3 and 4|
|`sdid\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_common.py`|donor-weight estimators, rolling CV, simplex tools, SDID estimator|
|`table5\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_sdid\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_mc.py`|SDID Monte Carlo (stylized and Prop 99-calibrated designs)|
|`table6\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_exterior.py`|extrapolation beyond the donor convex hull|

### Choices not specified in the paper

* **Ames features.** The paper's p = 288 and rank 255 are reproduced only when
the Kaggle `Id` column is kept as a numeric feature. Dropping it gives
p = 287 and rank 254.
* **Picard index.** Our rule returns ι̂ = 30 on the Ames design; the paper
reports 29. This is probably a 0- vs. 1-based indexing convention.
* **Table 6 target.** Post-treatment RMSE is measured against the noiseless
counterfactual Σᵢ wᵢ Y\_it. Only this definition reproduces the paper's
values; including the treated unit's own noise puts a floor of about 0.1
under every entry. SDID's σ̂ is the noise level. The first-difference σ̂ of
Arkhangelsky et al. is about 28 in this design and forces near-uniform
weights.
* **Rolling CV with short pre-periods.** When T\_pre < 30, five
non-overlapping 5-period windows don't fit, so validation windows overlap
(`sdid\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_common.rolling\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_folds`).

### 

