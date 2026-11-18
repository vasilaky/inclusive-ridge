"""
Shared SDID machinery for Tables 5-7: donor-weight estimators (ridge, IR,
simplex-projected ridge/IR, lasso, elastic net), expanding rolling-window
cross-validation on the treated unit's pre-treatment series, SDID time
weights and the SDID treatment-effect estimator.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from ir_solver import svd

ZETAS = np.logspace(-3, 6, 60)
ETAS_U = np.arange(1, 61)
ETAS_C = np.arange(1, 16)
ALPHAS = np.logspace(-4, 1, 30)
L1_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
METHODS = ("Ridge", "IR", "C-Ridge", "C-IR", "LASSO", "Elastic Net")


# ---------------------------------------------------------------------------
# simplex tools
# ---------------------------------------------------------------------------

def proj_simplex(V):
    """Euclidean projection of each row of V onto the probability simplex."""
    V = np.atleast_2d(V)
    m, n = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    cond = U - css / np.arange(1, n + 1) > 0
    rho = cond.sum(1)
    theta = css[np.arange(m), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def simplex_ls(y, X, pen=0.0, intercept=True, iters=3000):
    """min ||y - c - X w||^2 + pen ||w||^2 over w in the simplex (FISTA)."""
    if intercept:
        X = X - X.mean(0)
        y = y - y.mean()
    L = 2 * (np.linalg.norm(X, 2) ** 2 + pen)
    w = np.full(X.shape[1], 1.0 / X.shape[1])
    v, t = w.copy(), 1.0
    for _ in range(iters):
        g = 2 * (X.T @ (X @ v - y) + pen * v)
        w_new = proj_simplex(v - g / L)[0]
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = w_new + (t - 1) / t_new * (w_new - w)
        if np.max(np.abs(w_new - w)) < 1e-10:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def simplex_ridge(XtX, Xty, zeta, c, w_init, L=None, iters=5000, tol=1e-9):
    """min ||y - X w||^2 + zeta ||w - c||^2 over the simplex, given X'X and X'y
    (FISTA with warm start)."""
    L = 2 * (np.linalg.eigvalsh(XtX)[-1] + zeta) if L is None else L
    w, v, t = w_init.copy(), w_init.copy(), 1.0
    for _ in range(iters):
        g = 2 * (XtX @ v - Xty + zeta * (v - c))
        w_new = proj_simplex(v - g / L)[0]
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = w_new + (t - 1) / t_new * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w, t = w_new, t_new
    return w


def proximal_path(Xc, yc, w0, zeta, eta_max):
    """Simplex-constrained IR: eta proximal steps
    w_k = argmin_{w in simplex} ||y - X w||^2 + zeta ||w - w_{k-1}||^2,
    returning w_1, ..., w_{eta_max} (eta = 1 is simplex-constrained ridge)."""
    XtX, Xty = Xc.T @ Xc, Xc.T @ yc
    L = 2 * (np.linalg.eigvalsh(XtX)[-1] + zeta)
    w, path = proj_simplex(w0)[0], []
    for _ in range(eta_max):
        w = simplex_ridge(XtX, Xty, zeta, w, w, L)
        path.append(w)
    return np.array(path)


# ---------------------------------------------------------------------------
# rolling folds
# ---------------------------------------------------------------------------

def rolling_folds(T, k=5, window=5, min_train=5):
    """Expanding-window folds on periods 0..T-1. Validation windows of length
    `window` end at T, T-step, ..., with step = window when the series is long
    enough and smaller (overlapping windows) otherwise."""
    window = min(window, T - min_train - (k - 1))
    step = max(1, min(window, (T - min_train - window) // max(k - 1, 1)))
    folds = []
    for j in range(k):
        end = T - (k - 1 - j) * step
        folds.append((np.arange(0, end - window), np.arange(end - window, end)))
    return folds


# ---------------------------------------------------------------------------
# donor weights
# ---------------------------------------------------------------------------

def ir_grid_weights(Xtr, ytr, w0, zetas, etas, intercept=True):
    """Unconstrained IR weights (eq. 26) for every (zeta, eta): array of shape
    (len(zetas)*len(etas), N0), plus labels. With intercept=True the design
    and outcome are centered over time (unpenalized intercept)."""
    if intercept:
        xm, ym = Xtr.mean(0), ytr.mean()
    else:
        xm, ym = np.zeros(Xtr.shape[1]), 0.0
    Xc, yc = Xtr - xm, ytr - ym
    u, s, v = svd(Xc)
    z = u.T @ (yc - Xc @ w0)
    s2 = s**2
    F = -np.expm1(-etas[None, :, None] * np.log1p(s2[None, None, :] / zetas[:, None, None]))
    F = F.reshape(-1, s.size)
    W = w0 + (F * (z / s)) @ v.T
    labels = [(zt, et) for zt in zetas for et in etas]
    return W, labels, xm, ym


def _proximal_grid(Xtr, ytr, w0, zetas, etas, intercept):
    if intercept:
        xm, ym = Xtr.mean(0), ytr.mean()
    else:
        xm, ym = np.zeros(Xtr.shape[1]), 0.0
    eta_max = int(max(etas))
    W, labels = [], []
    for zt in zetas:
        P = proximal_path(Xtr - xm, ytr - ym, w0, zt, eta_max)
        for et in etas:
            W.append(P[int(et) - 1])
            labels.append((zt, et))
    return np.array(W), labels, xm, ym


def fit_ir_family(X, y, w0, zetas, etas, constrained, intercept=True, folds=None,
                  mode="projection"):
    """Select (zeta, eta) by rolling CV and refit on all pre-periods.
    constrained=True: mode="projection" projects the unconstrained IR weights
    onto the simplex (Sec. 5.4.1); mode="proximal" runs eta simplex-constrained
    proximal ridge steps instead (slower)."""
    T = len(y)
    folds = rolling_folds(T) if folds is None else folds
    if constrained and mode == "proximal":
        err = 0.0
        for tr, va in folds:
            W, labels, xm, ym = _proximal_grid(X[tr], y[tr], w0, zetas, etas, intercept)
            pred = ym + (X[va] - xm) @ W.T
            err = err + ((pred - y[va, None]) ** 2).sum(0)
        zt, et = labels[int(np.argmin(err))]
        W, _, _, _ = _proximal_grid(X, y, w0, np.array([zt]), np.array([et]), intercept)
        return W[0], zt, et
    err = 0.0
    for tr, va in folds:
        W, labels, xm, ym = ir_grid_weights(X[tr], y[tr], w0, zetas, etas, intercept)
        if constrained:
            W = proj_simplex(W)
        pred = ym + (X[va] - xm) @ W.T
        err = err + ((pred - y[va, None]) ** 2).sum(0)
    g = int(np.argmin(err))
    zt, et = labels[g]
    W, _, _, _ = ir_grid_weights(X, y, w0, np.array([zt]), np.array([et]), intercept)
    w = proj_simplex(W)[0] if constrained else W[0]
    return w, zt, et


def fit_enet(X, y, l1_ratios, folds=None):  # intercept always fitted
    """Lasso (l1_ratios=(1,)) or elastic net with rolling-CV tuning."""
    folds = rolling_folds(len(y)) if folds is None else folds
    best = (np.inf, None, None)
    for l1 in l1_ratios:
        err = 0.0
        for tr, va in folds:
            xm, ym = X[tr].mean(0), y[tr].mean()
            _, coefs, _ = enet_path(X[tr] - xm, y[tr] - ym, l1_ratio=l1,
                                    alphas=ALPHAS, max_iter=5000)
            pred = ym + (X[va] - xm) @ coefs
            err = err + ((pred - y[va, None]) ** 2).sum(0)
        i = int(np.argmin(err))
        if err[i] < best[0]:
            best = (err[i], l1, ALPHAS[i])
    _, l1, a = best
    xm, ym = X.mean(0), y.mean()
    _, coefs, _ = enet_path(X - xm, y - ym, l1_ratio=l1, alphas=[a], max_iter=5000)
    return coefs[:, 0]


def unit_weights(method, X, y, intercept=True, mode="projection", zetas_c=None):
    """Donor weights for one method. X: T_pre x N0, y: T_pre.
    Returns (w, zeta_hat, eta_hat). `mode` applies to C-Ridge / C-IR;
    `zetas_c` optionally replaces the zeta grid for the constrained methods."""
    N0 = X.shape[1]
    w0 = np.full(N0, 1.0 / N0)
    zc = ZETAS if zetas_c is None else zetas_c
    if method == "Ridge":
        return fit_ir_family(X, y, w0, ZETAS, np.array([1]), False, intercept)
    if method == "IR":
        return fit_ir_family(X, y, w0, ZETAS, ETAS_U, False, intercept)
    if method == "C-Ridge":
        return fit_ir_family(X, y, w0, zc, np.array([1]), True, intercept, mode=mode)
    if method == "C-IR":
        return fit_ir_family(X, y, w0, zc, ETAS_C, True, intercept, mode=mode)
    if method == "LASSO":
        return fit_enet(X, y, (1.0,)), None, None
    if method == "Elastic Net":
        return fit_enet(X, y, L1_RATIOS), None, None
    raise ValueError(method)


# ---------------------------------------------------------------------------
# SDID
# ---------------------------------------------------------------------------

def time_weights(Ydon_pre, Ydon_post):
    """SDID time weights: simplex + intercept regression of the donors'
    post-period means on their pre-period outcomes (rows = donors)."""
    return simplex_ls(Ydon_post.mean(1), Ydon_pre, pen=1e-6, intercept=True)


def sdid_tau(w, lam, y_pre, y_post, Ydon_pre, Ydon_post):
    """tau_hat = [ybar_post - lam'y_pre]_treated - w'[Ybar_post - Y_pre lam]_donors."""
    tr = y_post.mean() - lam @ y_pre
    don = Ydon_post.mean(1) - Ydon_pre @ lam
    return tr - w @ don
