"""
Table 6 -- Extrapolation beyond the donor convex hull.

Rank-three factor model with T0 = 20 pre-treatment periods, J = 40 donors:
    Y_it = 100 + 10 (F Lambda^T)_it + eps_it,  eps ~ N(0, 0.1^2),
F and Lambda standard normal. The treated unit is Y_1t = sum_i w_i Y_it + eps_1t
for a fixed true weight vector w; ten post-treatment periods use the same
loadings and fresh factor draws; the treatment effect is zero, and the
post-treatment target is the noiseless counterfactual sum_i w_i Y_it, so all
post-treatment deviation is estimation error. 300 replications per design.

Designs: A interior (1/3, 1/3, 1/3), B interior (eight donors at 1/8),
         C (1.5, -0.5), D (0.9, 0.9, -0.4, -0.4), E (2.0, -1.0) exterior.
Estimators (fitted to the demeaned pre-period design, unpenalized intercept):
  IR+plug  exponential filter, kappa by the Picard plug-in risk (Sec. 4.3)
  IR+CV    exponential filter, kappa by five-fold CV
  EN       elastic net, l1 ratio and penalty by five-fold CV
  SDID     simplex weights with zeta = (N_tr T_post)^(1/4) sigma_hat
Reported: median post-treatment prediction RMSE and median weight-recovery
error ||w_hat - w||^2, with bootstrap standard errors of the medians.

Usage:  python table6_exterior.py [--reps 300] [--seed 0]
"""
import argparse
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from ir_solver import svd, asymir, tune_asymir, estimate_noise
from sdid_common import simplex_ls

warnings.filterwarnings("ignore", category=ConvergenceWarning)

T0, T1, J, RANK, SIGMA = 20, 10, 40, 3, 0.1
DESIGNS = {
    "A. interior (1/3,1/3,1/3)": np.r_[np.full(3, 1 / 3)],
    "B. interior, 8 donors": np.r_[np.full(8, 1 / 8)],
    "C. exterior (1.5,-0.5)": np.r_[1.5, -0.5],
    "D. exterior (.9,.9,-.4,-.4)": np.r_[0.9, 0.9, -0.4, -0.4],
    "E. exterior (2.0,-1.0)": np.r_[2.0, -1.0],
}
METHODS = ("IR+plug", "IR+CV", "EN", "SDID")
H_PICARD = 3          # Picard window; the demeaned design has rank <= T0 - 1


def simulate(w_head, rng):
    w = np.zeros(J)
    w[:w_head.size] = w_head
    Lam = rng.standard_normal((J, RANK))
    F_pre, F_post = rng.standard_normal((T0, RANK)), rng.standard_normal((T1, RANK))
    D_pre = 100 + 10 * F_pre @ Lam.T + SIGMA * rng.standard_normal((T0, J))
    D_post = 100 + 10 * F_post @ Lam.T + SIGMA * rng.standard_normal((T1, J))
    y_pre = D_pre @ w + SIGMA * rng.standard_normal(T0)
    y_post = D_post @ w           # counterfactual target (tau = 0, no own noise)
    return D_pre, D_post, y_pre, y_post, w


def kappa_cv(Xc, yc, kappas, k=5):
    err = np.zeros(len(kappas))
    for tr, va in KFold(k, shuffle=True, random_state=0).split(Xc):
        xm, ym = Xc[tr].mean(0), yc[tr].mean()
        u, s, v = svd(Xc[tr] - xm)
        z = u.T @ (yc[tr] - ym)
        for i, kap in enumerate(kappas):
            b = v @ (-np.expm1(-s**2 / kap) * z / s)
            err[i] += np.sum((ym + (Xc[va] - xm) @ b - yc[va]) ** 2)
    return kappas[int(np.argmin(err))]


def fit(method, D_pre, y_pre):
    xm, ym = D_pre.mean(0), y_pre.mean()
    Xc, yc = D_pre - xm, y_pre - ym
    if method == "IR+plug":
        w = tune_asymir(Xc, yc, signal="picard", h=H_PICARD)["coeff"]
    elif method == "IR+CV":
        s2 = svd(Xc)[1] ** 2
        kappas = np.logspace(np.log10(s2.min()) - 2, np.log10(s2.max()) + 2, 200)
        w = asymir(kappa_cv(Xc, yc, kappas), Xc, yc)
    elif method == "EN":
        en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5,
                          max_iter=20000).fit(D_pre, y_pre)
        return en.coef_, en.intercept_
    elif method == "SDID":
        # zeta = (N_tr T_post)^(1/4) sigma_hat. Here sigma_hat is the noise
        # level (Picard tail estimate). Arkhangelsky et al.'s first-difference
        # estimate is ~28 in this design because the factors are i.i.d. over
        # time, which forces near-uniform weights.
        sig = np.sqrt(estimate_noise(Xc, yc, method="picard", h=H_PICARD)[0])
        zeta = (1 * T1) ** 0.25 * sig
        w = simplex_ls(yc, Xc, pen=zeta**2 * T0, intercept=False)
    return w, ym - xm @ w


def boot_se_median(x, rng, B=500):
    return np.std([np.median(rng.choice(x, x.size)) for _ in range(B)])


def main(reps=300, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for name, wh in DESIGNS.items():
        post, werr = {m: [] for m in METHODS}, {m: [] for m in METHODS}
        for _ in range(reps):
            D_pre, D_post, y_pre, y_post, w = simulate(wh, rng)
            for m in METHODS:
                wh_, c = fit(m, D_pre, y_pre)
                post[m].append(np.sqrt(np.mean((c + D_post @ wh_ - y_post) ** 2)))
                werr[m].append(np.sum((wh_ - w) ** 2))
        res[name] = (post, werr)
    brng = np.random.default_rng(seed + 1)
    for title, idx in (("post-treatment prediction RMSE", 0),
                       ("weight recovery ||w_hat - w||^2", 1)):
        print(f"\nTable 6 ({title}): median of {reps} replications (bootstrap SE)")
        print(f"{'Design':<30}" + "".join(f"{m:>17}" for m in METHODS))
        for name in DESIGNS:
            d = res[name][idx]
            print(f"{name:<30}" + "".join(
                f"{np.median(d[m]):>9.3f} ({boot_se_median(np.array(d[m]), brng):.3f})"
                for m in METHODS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.reps, a.seed)
