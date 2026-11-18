"""
ir_solver.py -- Reusable solvers for Inclusive Ridge (IR).

Implements the estimators and coefficient-risk formulas of
"Inclusive Ridge: Adaptive Shrinkage with Applications to Causal Inference".

Model:  y = X beta + eps,  eps ~ N(0, sigma_eps^2 I_n),  X = U diag(s) V^T.

Spectral quantities (paper notation, s_j = sigma_j):
    z_j     = u_j^T (y - X beta0)
    r_j     = zeta / (zeta + s_j^2)
    a_j     = v_j^T (beta - beta0)

Core functions
--------------
    u, s, v = svd(X)                              thin SVD, X = u @ diag(s) @ v.T
    coeff   = ir(zeta, eta, X, y, initial)        IR / ridge estimator      eq. (3)
    mse     = irmse(zeta, eta, X, y, initial)     IR / ridge coefficient risk eq. (5)
    coeff   = asymir(kappa, X, y, initial)        exponential-limit estimator eq. (10)
    emse    = asymmse(kappa, X, y, initial)       exponential-limit risk    eq. (11)/(15)

eta = 1 gives ridge shrinking toward `initial`. Every function accepts a
precomputed `usv=svd(X)`, so evaluating a whole (zeta, eta) or kappa grid
costs O(p) per point after one SVD.

Risk functions
--------------
If the true `beta` and `sigma2` are passed, the exact (oracle) risk is
returned. Otherwise a feasible plug-in risk is returned:
    signal="unbiased"  a_j^2 = (z_j^2 - sigma2_hat) / s_j^2           (Sec. 4.1)
    signal="truncated" same, floored at zero
    signal="picard"    a_j^2 = z_j^2 / s_j^2 for j < iota_hat, else 0 (Sec. 4.3)
Noise variance: residual-space estimate ||(I-UU^T)y||^2/(n-r) when n > r,
otherwise the Picard tail energy V(iota_hat) (Levin-Meltzer rule, eq. 20-21).

Notes
-----
* Rank-deficient X (e.g. p > n) is allowed: sums run over the r nonzero
  singular values, and the component of `initial` in the null space of X is
  kept unchanged. Plug-in risks then omit the (unidentifiable) null-space bias.
* Requires only numpy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "svd", "ir", "irmse", "asymir", "asymmse", "asymcov",
    "estimate_noise", "picard_index", "egcv", "tune_ir", "tune_asymir",
]


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def svd(X, rtol=None):
    """Thin SVD of X with numerically zero singular values removed.

    Returns (u, s, v) with X ~= u @ np.diag(s) @ v.T, where u is n x r,
    s has length r (descending) and v is p x r. Note: v, not v transpose.
    """
    X = np.asarray(X, dtype=float)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    if rtol is None:
        rtol = max(X.shape) * np.finfo(float).eps
    keep = s > rtol * s[0] if s.size and s[0] > 0 else np.zeros_like(s, bool)
    return u[:, keep], s[keep], vt[keep].T


def _prepare(X, y, initial, usv):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = X.shape[1]
    beta0 = np.zeros(p) if initial is None else np.asarray(initial, float).ravel()
    if beta0.shape != (p,):
        raise ValueError(f"initial must have length p={p}")
    u, s, v = svd(X) if usv is None else usv
    z = u.T @ (y - X @ beta0)
    return X, y, beta0, u, s, v, z


def _survival(zeta, eta, s):
    """r_j^eta = (zeta/(zeta+s_j^2))^eta, computed stably."""
    if zeta <= 0 or eta <= 0:
        raise ValueError("zeta and eta must be positive")
    return np.exp(-eta * np.log1p(s**2 / zeta))


def _asym_survival(kappa, s):
    """exp(-s_j^2/kappa)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return np.exp(-(s**2) / kappa)


# ---------------------------------------------------------------------------
# Noise variance and Picard index
# ---------------------------------------------------------------------------

def picard_index(z, h=10, delta=0.05):
    """Empirical Picard index (Levin-Meltzer tail-energy rule, eq. 20-21).

    V(k) = mean(z_k^2, ..., z_r^2);  iota = min{k : |V(k+h)-V(k)|/V(k) < delta}.
    Returns (iota, V) with iota 1-based as in the paper. If no k satisfies
    the rule, iota = r (only the last coefficient is treated as noise).
    """
    z2 = np.asarray(z, float) ** 2
    r = z2.size
    tail = np.cumsum(z2[::-1])[::-1] / np.arange(r, 0, -1)    # V(1..r)
    for k in range(1, r - h + 1):                              # need V(k+h)
        vk, vkh = tail[k - 1], tail[k + h - 1]
        if vk > 0 and abs(vkh - vk) / vk < delta:
            return k, tail
    return r, tail


def estimate_noise(X, y, initial=None, usv=None, method="auto", h=10, delta=0.05):
    """Estimate sigma_eps^2.

    method="residual": ||(I - UU^T) y||^2 / (n - r)   (needs n > r)
    method="picard"  : V(iota_hat), Picard tail energy
    method="auto"    : residual if n > r, else picard
    Returns (sigma2_hat, iota_hat or None).
    """
    X, y, beta0, u, s, v, z = _prepare(X, y, initial, usv)
    n, r = X.shape[0], s.size
    if method == "auto":
        method = "residual" if n > r else "picard"
    if method == "residual":
        if n <= r:
            raise ValueError("residual-space estimate needs n > rank(X)")
        resid = (y - X @ beta0) - u @ z
        return float(resid @ resid / (n - r)), None
    if method == "picard":
        iota, tail = picard_index(z, h=h, delta=delta)
        return float(tail[iota - 1]), iota
    raise ValueError("method must be 'auto', 'residual' or 'picard'")


def _signal_energy(z, s, sigma2, signal, iota):
    """Plug-in estimates of a_j^2."""
    if signal == "unbiased":
        return (z**2 - sigma2) / s**2
    if signal == "truncated":
        return np.maximum(z**2 - sigma2, 0.0) / s**2
    if signal == "picard":
        a2 = z**2 / s**2
        a2[iota - 1:] = 0.0                       # j >= iota set to zero
        return a2
    raise ValueError("signal must be 'unbiased', 'truncated' or 'picard'")


def _risk_inputs(X, y, initial, usv, beta, sigma2, signal, noise, h, delta):
    """Return (s, a2, sigma2) for exact or plug-in risk."""
    X, y, beta0, u, s, v, z = _prepare(X, y, initial, usv)
    if beta is not None:                                       # oracle
        if sigma2 is None:
            raise ValueError("oracle risk needs sigma2 together with beta")
        a = v.T @ (np.asarray(beta, float).ravel() - beta0)
        return s, a**2, float(sigma2)
    iota = None
    if signal == "picard":
        s2_hat, iota = estimate_noise(X, y, initial, (u, s, v), "picard", h, delta)
        if sigma2 is None:
            sigma2 = s2_hat
    elif sigma2 is None:
        sigma2, _ = estimate_noise(X, y, initial, (u, s, v), noise, h, delta)
    return s, _signal_energy(z, s, sigma2, signal, iota), float(sigma2)


# ---------------------------------------------------------------------------
# Finite-order IR (eta = 1 is ridge)
# ---------------------------------------------------------------------------

def ir(zeta, eta, X, y, initial=None, usv=None):
    """IR estimator, eq. (3):
        beta_hat = beta0 + sum_j [1 - r_j^eta] (z_j / s_j) v_j.
    eta = 1 is ridge shrinking toward `initial`; real eta > 0 is allowed.
    """
    X, y, beta0, u, s, v, z = _prepare(X, y, initial, usv)
    w = -np.expm1(-eta * np.log1p(s**2 / zeta))                # 1 - r^eta
    return beta0 + v @ (w * z / s)


def irmse(zeta, eta, X, y, initial=None, usv=None, *, beta=None, sigma2=None,
          signal="unbiased", noise="auto", h=10, delta=0.05):
    """Coefficient risk E||beta_hat - beta||^2 of the IR estimator, eq. (5):
        R = sum_j [ sigma2/s_j^2 (1 - r_j^eta)^2 + a_j^2 r_j^(2 eta) ].
    Exact if `beta` and `sigma2` are given; otherwise a plug-in estimate
    (see module docstring for `signal` and `noise`).
    """
    s, a2, s2 = _risk_inputs(X, y, initial, usv, beta, sigma2, signal, noise, h, delta)
    rk = _survival(zeta, eta, s)
    return float(np.sum(s2 / s**2 * (1 - rk) ** 2 + a2 * rk**2))


# ---------------------------------------------------------------------------
# Exponential (asymptotic) limit: zeta = kappa * eta, eta -> infinity
# ---------------------------------------------------------------------------

def asymir(kappa, X, y, initial=None, usv=None):
    """Limiting exponential estimator, eq. (10):
        beta_hat = beta0 + sum_j (1 - exp(-s_j^2/kappa)) (z_j / s_j) v_j.
    """
    X, y, beta0, u, s, v, z = _prepare(X, y, initial, usv)
    w = -np.expm1(-(s**2) / kappa)
    return beta0 + v @ (w * z / s)


def asymmse(kappa, X, y, initial=None, usv=None, *, beta=None, sigma2=None,
            signal="unbiased", noise="auto", h=10, delta=0.05):
    """Coefficient risk of the exponential estimator, eq. (11):
        R = sum_j [ a_j^2 exp(-2 s_j^2/kappa) + sigma2/s_j^2 (1 - exp(-s_j^2/kappa))^2 ].
    Exact if `beta` and `sigma2` are given; otherwise the plug-in risk
    (15) (signal="unbiased") or the Picard version of Sec. 4.3 (signal="picard").
    """
    s, a2, s2 = _risk_inputs(X, y, initial, usv, beta, sigma2, signal, noise, h, delta)
    e = _asym_survival(kappa, s)
    return float(np.sum(a2 * e**2 + s2 / s**2 * (1 - e) ** 2))


def asymcov(kappa, X, y=None, initial=None, usv=None, *, sigma2=None,
            noise="auto", h=10, delta=0.05):
    """Covariance of the exponential estimator, eq. (14):
        Sigma_kappa = sigma2 * sum_j ((1 - exp(-s_j^2/kappa)) / s_j)^2 v_j v_j^T.
    If sigma2 is None it is estimated from (X, y).
    """
    u, s, v = svd(X) if usv is None else usv
    if sigma2 is None:
        if y is None:
            raise ValueError("pass sigma2 or y")
        sigma2, _ = estimate_noise(X, y, initial, (u, s, v), noise, h, delta)
    g = (-np.expm1(-(s**2) / kappa) / s) ** 2
    return sigma2 * (v * g) @ v.T


# ---------------------------------------------------------------------------
# Tuning helpers
# ---------------------------------------------------------------------------

def egcv(zeta, eta, X, y, initial=None, usv=None):
    """Extended GCV criterion, eq. (19)."""
    X, y, beta0, u, s, v, z = _prepare(X, y, initial, usv)
    n = X.shape[0]
    rk = _survival(zeta, eta, s)
    resid_perp = (y - X @ beta0) - u @ z
    num = n * (resid_perp @ resid_perp + np.sum(rk**2 * z**2))
    den = (n - s.size + np.sum(rk)) ** 2
    return float(num / den)


def tune_ir(X, y, initial=None, zetas=None, etas=None, criterion="mse", **risk_kw):
    """Grid search over (zeta, eta) minimizing plug-in risk ("mse") or EGCV.
    Returns dict(zeta, eta, value, coeff). One SVD is shared by all points.
    """
    usv = svd(X)
    zetas = np.logspace(-3, 6, 60) if zetas is None else np.asarray(zetas, float)
    etas = np.arange(1, 61) if etas is None else np.asarray(etas, float)
    best = (np.inf, None, None)
    for zt in zetas:
        for et in etas:
            val = (irmse(zt, et, X, y, initial, usv, **risk_kw) if criterion == "mse"
                   else egcv(zt, et, X, y, initial, usv))
            if val < best[0]:
                best = (val, zt, et)
    val, zt, et = best
    return dict(zeta=float(zt), eta=float(et), value=float(val),
                coeff=ir(zt, et, X, y, initial, usv))


def tune_asymir(X, y, initial=None, kappas=None, **risk_kw):
    """Grid search over kappa minimizing the plug-in risk of the exponential
    estimator. Default grid spans the squared singular values of X.
    Returns dict(kappa, value, coeff).
    """
    usv = svd(X)
    if kappas is None:
        s2 = usv[1] ** 2
        kappas = np.logspace(np.log10(s2.min()) - 2, np.log10(s2.max()) + 2, 200)
    vals = [asymmse(k, X, y, initial, usv, **risk_kw) for k in kappas]
    i = int(np.argmin(vals))
    k = float(kappas[i])
    return dict(kappa=k, value=float(vals[i]), coeff=asymir(k, X, y, initial, usv))


if __name__ == "__main__":
    # Minimal demo on an ill-conditioned design.
    rng = np.random.default_rng(0)
    n, p = 200, 30
    X = rng.standard_normal((n, p)) @ np.diag(np.logspace(0, -3, p))
    beta = rng.standard_normal(p)
    y = X @ beta + 0.05 * rng.standard_normal(n)
    usv = svd(X)

    print("ridge  zeta=1e-3     risk:", irmse(1e-3, 1, X, y, usv=usv, beta=beta, sigma2=0.05**2))
    print("IR     zeta=1e-2,e=5 risk:", irmse(1e-2, 5, X, y, usv=usv, beta=beta, sigma2=0.05**2))
    t = tune_asymir(X, y)
    print("asymptotic plug-in kappa:", t["kappa"],
          " true risk:", asymmse(t["kappa"], X, y, usv=usv, beta=beta, sigma2=0.05**2))
