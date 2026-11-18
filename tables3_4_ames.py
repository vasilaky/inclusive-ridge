"""
Tables 3 and 4 -- Ames house prices (Kaggle "House Prices: Advanced Regression
Techniques", train.csv, n = 1,460).

Preprocessing (reproduces the paper's p = 288 features of rank 255):
  * outcome log(SalePrice);
  * every numeric column (the Kaggle `Id` column included, as in the paper's
    design) median-imputed, and log1p-transformed if |skew| > 0.75;
  * one indicator per observed level of every categorical column (missing
    values get no indicator);
  * constant columns dropped, columns standardized, intercept unpenalized
    (y and X centered).

Table 3: five-fold held-out RMSE, all tuning inside training folds, for
  (a) plug-in risk with sigma^2 = V(iota_hat) and tail coefficients set to 0,
  (b) plug-in risk with the residual-space sigma^2 and tail set to 0,
  (c) inner five-fold cross-validation over (zeta, eta).
Table 4: semi-synthetic simulation on the fixed design with known beta.

Usage:  python tables3_4_ames.py --data train.csv [--table 3|4|all] [--draws 200]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from ir_solver import svd, picard_index

H, DELTA = 10, 0.05                      # Picard rule window and tolerance
ETAS = np.array([0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200])


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_ames(path):
    d = pd.read_csv(path)
    y = np.log(d["SalePrice"].to_numpy(float))
    X = d.drop(columns=["SalePrice"])
    num = X.select_dtypes("number").columns
    cat = X.columns.difference(num)
    Xn = X[num].astype(float)
    Xn = Xn.fillna(Xn.median())
    skew = Xn.skew().abs()
    cols = skew[skew > 0.75].index
    Xn[cols] = np.log1p(Xn[cols])
    M = pd.concat([Xn, pd.get_dummies(X[cat])], axis=1).astype(float)
    M = M.loc[:, M.std() > 0]
    return M.to_numpy(), y


def standardize(Xtr, Xte=None):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 0
    f = lambda A: (A[:, keep] - mu[keep]) / sd[keep]
    return (f(Xtr), None) if Xte is None else (f(Xtr), f(Xte))


# ---------------------------------------------------------------------------
# filter grids (all candidate (zeta, eta) pairs plus the exponential limit)
# ---------------------------------------------------------------------------

def filter_grid(s, n_scale=80, etas=ETAS):
    """Returns (F, labels): F[g, j] is the filter value of candidate g at
    direction j; labels[g] = (zeta, eta) with eta = inf for the limit
    (then the first entry is kappa)."""
    s2 = s**2
    scales = np.logspace(np.log10(s2.min()) - 3, np.log10(s2.max()) + 3, n_scale)
    F, lab = [], []
    for eta in etas:
        F.append(-np.expm1(-eta * np.log1p(s2[None, :] / scales[:, None])))
        lab += [(z, eta) for z in scales]
    F.append(-np.expm1(-s2[None, :] / scales[:, None]))
    lab += [(k, np.inf) for k in scales]
    return np.vstack(F), lab


def risk_grid(F, s, a2, sigma2):
    """Coefficient risk (5)/(11) for every row of F."""
    return (sigma2 / s**2 * F**2 + a2 * (1 - F) ** 2).sum(1)


def spectral(Z, yc):
    u, s, v = svd(Z)
    z = u.T @ yc
    n, r = Z.shape[0], s.size
    resid = yc - u @ z
    s2_res = resid @ resid / (n - r - 1)                 # intercept uses 1 df
    iota, V = picard_index(z, H, DELTA)
    return u, s, v, z, s2_res, iota, V


def tail_test(z, iota, s2_res, df_res):
    t = z[iota:]                                         # directions j > iota
    F = (t @ t / t.size) / s2_res
    return F, t.size, stats.f.sf(F, t.size, df_res)


def truncated_a2(z, s, iota):
    a2 = z**2 / s**2
    a2[iota - 1:] = 0.0                                  # j >= iota_hat
    return a2


def tested_a2(z, s, iota, s2_res, df_res, level=0.05):
    """Tail kept at zero unless the F test rejects; then the tail directions
    get a pooled isotropic signal estimate,
        a_j^2 = tau2_hat = max(sum_tail (z_j^2 - sigma2), 0) / sum_tail s_j^2.
    NOTE: the paper does not spell out its pooled tail estimate; this is our
    choice, and the dense-signal rows of Table 4 are sensitive to it."""
    a2 = truncated_a2(z, s, iota)
    _, _, p = tail_test(z, iota, s2_res, df_res)
    if p < level:
        t = slice(iota - 1, None)
        a2[t] = max(np.sum(z[t] ** 2 - s2_res), 0.0) / np.sum(s[t] ** 2)
    return a2, p < level


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------

def predict(Fg, s, v, z, Zte):
    """Predictions (centered) of every grid candidate on Zte."""
    B = (Fg * (z / s)) @ v.T                             # G x p coefficients
    return B @ Zte.T


def cv_select(Z, yc, F_rows_fn, k=5, seed=0):
    """Inner K-fold CV over a grid built from each inner-fold SVD."""
    errs = None
    for tr, va in KFold(k, shuffle=True, random_state=seed).split(Z):
        ztr, zva = Z[tr], Z[va]
        mu = yc[tr].mean()
        u, s, v = svd(ztr - ztr.mean(0))
        zz = u.T @ (yc[tr] - mu)
        F, lab = F_rows_fn(s)
        pred = mu + predict(F, s, v, zz, zva - ztr.mean(0))
        e = ((pred - yc[va]) ** 2).sum(1)
        errs = e if errs is None else errs + e
    return lab[int(np.argmin(errs))]


def fit_with(label, s, v, z):
    scale, eta = label
    s2 = s**2
    f = -np.expm1(-s2 / scale) if np.isinf(eta) else -np.expm1(-eta * np.log1p(s2 / scale))
    return v @ (f * z / s)


def table3(X, y, seed=0):
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    rules = ["Plug-in, sigma2 = V(iota), tail = 0",
             "Plug-in, sigma2 = residual, tail = 0",
             "Cross-validation over (zeta, eta)"]
    sse = {(r, m): 0.0 for r in rules for m in ("IR", "Ridge")}
    chosen = []
    for tr, te in folds:
        Ztr, Zte = standardize(X[tr], X[te])
        ybar = y[tr].mean()
        yc = y[tr] - ybar
        u, s, v, z, s2_res, iota, V = spectral(Ztr, yc)
        F, lab = filter_grid(s)
        ridge = np.array([e == 1 for _, e in lab])
        a2 = truncated_a2(z, s, iota)
        for rule, s2 in ((rules[0], V[iota - 1]), (rules[1], s2_res)):
            R = risk_grid(F, s, a2, s2)
            for m, mask in (("IR", np.ones_like(ridge)), ("Ridge", ridge)):
                g = np.flatnonzero(mask)[np.argmin(R[mask])]
                pred = ybar + Zte @ fit_with(lab[g], s, v, z)
                sse[(rule, m)] += np.sum((pred - y[te]) ** 2)
                if m == "IR" and rule == rules[1]:
                    chosen.append(float(lab[g][1]))
        for m, etas in (("IR", ETAS), ("Ridge", np.array([1.0]))):
            label = cv_select(Ztr, yc, lambda ss: filter_grid(ss, etas=etas)
                              if m == "IR" else _ridge_grid(ss))
            pred = ybar + Zte @ fit_with(label, s, v, z)
            sse[(rules[2], m)] += np.sum((pred - y[te]) ** 2)
    n = len(y)
    print("\nTable 3: five-fold held-out RMSE of log sale price")
    print(f"{'Tuning rule':<40}{'IR':>8}{'Ridge':>8}")
    for r in rules:
        print(f"{r:<40}{np.sqrt(sse[(r, 'IR')] / n):>8.4f}{np.sqrt(sse[(r, 'Ridge')] / n):>8.4f}")
    print(f"(plug-in IR with residual sigma^2 selected eta = {chosen} across folds)")


def _ridge_grid(s, n_scale=80):
    F, lab = filter_grid(s, n_scale, etas=np.array([1.0]))
    keep = [i for i, (_, e) in enumerate(lab) if e == 1]
    return F[keep], [lab[i] for i in keep]


# ---------------------------------------------------------------------------
# Table 4
# ---------------------------------------------------------------------------

def table4(X, y, draws=200, seed=0):
    Z, _ = standardize(X)
    yc = y - y.mean()
    n = Z.shape[0]
    u, s, v, z, s2_res, iota, V = spectral(Z, yc)
    r = s.size
    df_res = n - r - 1
    F_stat, df1, p = tail_test(z, iota, s2_res, df_res)
    print(f"\nRank r = {r}, residual df = {df_res}")
    print(f"sigma2_res = {s2_res:.4f},  V(iota) = {V[iota - 1]:.4f} (iota = {iota})")
    print(f"F_tail = {F_stat:.2f} on ({df1}, {df_res}) df, p = {p:.1e}")

    sigma2 = s2_res
    lead = slice(0, iota - 1)
    rng = np.random.default_rng(seed)

    # true spectral coefficients a_j = v_j^T beta for each scenario
    a_lead = np.zeros(r)
    a_lead[lead] = z[lead] / s[lead]
    tail = slice(iota - 1, None)
    tau2 = max(np.mean(z[tail] ** 2) - sigma2, 0.0) / np.mean(s[tail] ** 2)
    a_iso = a_lead.copy()
    a_iso[tail] = np.sqrt(tau2) * rng.standard_normal(r - iota + 1)
    s2 = s**2
    gcv = lambda f: np.sum(((1 - f) * z) ** 2) / (n - np.sum(f)) ** 2
    scales = np.logspace(np.log10(s2.min()) - 3, np.log10(s2.max()) + 3, 400)
    zr = scales[np.argmin([gcv(s2 / (s2 + c)) for c in scales])]
    ka = scales[np.argmin([gcv(-np.expm1(-s2 / c)) for c in scales])]
    a_ridge = s2 / (s2 + zr) * z / s
    a_exp = -np.expm1(-s2 / ka) * z / s
    lasso = LassoCV(cv=5, random_state=0, fit_intercept=False, max_iter=20000).fit(Z, yc)
    a_lasso = v.T @ lasso.coef_
    scenarios = [("Leading %d directions only" % (iota - 1), a_lead),
                 ("Leading %d plus isotropic tail" % (iota - 1), a_iso),
                 ("Ridge fit (dense)", a_ridge),
                 ("Lasso fit (sparse)", a_lasso),
                 ("Exponential-filter fit", a_exp)]

    F, lab = filter_grid(s)
    ridge = np.array([e == 1 for _, e in lab])
    print(f"\nTable 4: plug-in IR risk / oracle IR risk, {draws} draws, sigma2 = {sigma2:.4f}")
    print(f"{'True beta':<34}{'F rejects':>10}{'Tail=0':>8}{'Tested':>8}{'IR gain':>9}")
    for name, a in scenarios:
        R_true = risk_grid(F, s, a**2, sigma2)
        oracle = R_true.min()
        gain = 1 - oracle / R_true[ridge].min()
        rej, r0, rt = 0, [], []
        for _ in range(draws):
            zz = s * a + np.sqrt(sigma2) * rng.standard_normal(r)
            s2_hat = sigma2 * rng.chisquare(df_res) / df_res      # residual-space estimate
            io, _ = picard_index(zz, H, DELTA)
            a0 = truncated_a2(zz, s, io)
            at, rejected = tested_a2(zz, s, io, s2_hat, df_res)
            rej += rejected
            r0.append(R_true[np.argmin(risk_grid(F, s, a0, s2_hat))])
            rt.append(R_true[np.argmin(risk_grid(F, s, at, s2_hat))])
        print(f"{name:<34}{100 * rej / draws:>9.0f}%{np.mean(r0) / oracle:>8.2f}"
              f"{np.mean(rt) / oracle:>8.2f}{100 * gain:>8.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Kaggle train.csv")
    ap.add_argument("--table", choices=["3", "4", "all"], default="all")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X, y = load_ames(a.data)
    print(f"Design: n = {X.shape[0]}, p = {X.shape[1]}")
    if a.table in ("4", "all"):
        table4(X, y, a.draws, a.seed)
    if a.table in ("3", "all"):
        table3(X, y, a.seed)
