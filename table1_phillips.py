"""
Table 1 -- Phillips Fredholm inverse problem (n = 400): oracle coefficient MSE
of optimally tuned ridge, finite-order IR and the limiting exponential filter.

Two oracle definitions are available:
  --mode expected : minimizes the expected risk (24),
                    R(f) = sum_j [sigma^2/s_j f_j^2 + alpha_j^2 (1-f_j)^2] / n,
                    with sigma = 5% of ||A x|| / sqrt(n).  Deterministic.
  --mode realized : draws one noise vector (seeded), and minimizes the realized
                    error ||x_hat - x||^2 / n for that draw.
The paper's Table 1 reports the expected risk (the default mode). The realized
mode is provided for comparison; single-draw gains vary widely across seeds.

Usage:  python table1_phillips.py [--mode expected|realized] [--seed 0] [--eta-max 200]
"""
import argparse
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import minimize, minimize_scalar


def phillips(n=400):
    """Phillips test problem (Hansen, Regularization Tools): A, b = A x, x."""
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    h, n4 = 12.0 / n, n // 4
    c = np.cos(np.arange(-1, n4 + 1) * 4 * np.pi / n)
    r1 = np.zeros(n)
    r1[:n4] = h + 9 / (h * np.pi**2) * (2 * c[1:n4 + 1] - c[0:n4] - c[2:n4 + 2])
    r1[n4] = h / 2 + 9 / (h * np.pi**2) * (np.cos(4 * np.pi / n) - 1)
    A = toeplitz(r1)
    cc, k = np.pi / 3, np.arange(1, n4 + 1)
    x = np.zeros(n)
    x[2 * n4:3 * n4] = (h + (np.sin(cc * k * h) - np.sin(cc * (k - 1) * h)) / cc) / np.sqrt(h)
    x[n4:2 * n4] = x[3 * n4 - 1:2 * n4 - 1:-1]
    return A, A @ x, x


def filters(s2, zeta=None, eta=None, kappa=None):
    if kappa is not None:
        return -np.expm1(-s2 / kappa)
    return -np.expm1(-eta * np.log1p(s2 / zeta))


def main(mode="expected", seed=0, n=400, rel_noise=0.05, eta_max=200.0):
    A, b, x = phillips(n)
    U, s, Vt = np.linalg.svd(A)
    s2, alpha = s**2, Vt @ x
    sig = rel_noise * np.linalg.norm(b) / np.sqrt(n)

    if mode == "expected":
        def risk(f):
            return np.sum(sig**2 / s2 * f**2 + alpha**2 * (1 - f) ** 2) / n
        print(f"Oracle = expected risk (24), sigma = {rel_noise:.1%} of ||Ax||/sqrt(n)")
    else:
        e = sig * np.random.default_rng(seed).standard_normal(n)
        print(f"Oracle = realized error, seed {seed}, "
              f"realized relative noise {np.linalg.norm(e) / np.linalg.norm(b):.3%}")
        z = U.T @ (b + e)

        def risk(f):
            return np.sum((f * z / s - alpha) ** 2) / n      # V orthogonal

    # ridge
    rr = minimize_scalar(lambda lz: risk(s2 / (s2 + np.exp(lz))), bounds=(-25, 10),
                         method="bounded", options={"xatol": 1e-12})
    z_r, R_r = np.exp(rr.x), rr.fun
    # limiting IR
    ra = minimize_scalar(lambda lk: risk(filters(s2, kappa=np.exp(lk))), bounds=(-25, 10),
                         method="bounded", options={"xatol": 1e-12})
    k_a, R_a = np.exp(ra.x), ra.fun
    # finite IR: multistart bounded search in (log zeta, log eta), eta <= eta_max.
    # When the exponential limit is optimal the finite optimum sits at eta_max.
    best = None
    for le in np.log([1, 2, 5, 20, 100, eta_max]):
        lz0 = np.log(k_a) + le                                  # start on zeta = kappa*eta
        res = minimize(lambda t: risk(filters(s2, np.exp(t[0]), np.exp(t[1]))),
                       [lz0, le], method="L-BFGS-B",
                       bounds=[(-30, 30), (np.log(1e-2), np.log(eta_max))],
                       options={"ftol": 1e-15, "gtol": 1e-14, "maxiter": 20000})
        if best is None or res.fun < best.fun:
            best = res
    z_f, e_f, R_f = np.exp(best.x[0]), np.exp(best.x[1]), best.fun

    print(f"\n{'Estimator':<12}{'zeta*':>14}{'eta* or kappa*':>22}{'MSE':>16}{'Gain (%)':>10}")
    print(f"{'Ridge':<12}{z_r:>14.8g}{'eta=1':>22}{R_r:>16.8e}{'--':>10}")
    print(f"{'Finite IR':<12}{z_f:>14.8g}{'eta = %.8g' % e_f:>22}{R_f:>16.8e}"
          f"{100 * (1 - R_f / R_r):>10.3f}")
    print(f"{'Limiting IR':<12}{'--':>14}{'kappa = %.8g' % k_a:>22}{R_a:>16.8e}"
          f"{100 * (1 - R_a / R_r):>10.3f}")
    print(f"\nzeta*/eta* = {z_f / e_f:.8g}   vs kappa* = {k_a:.8g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expected", "realized"], default="expected")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta-max", type=float, default=200.0)
    a = ap.parse_args()
    main(a.mode, a.seed, eta_max=a.eta_max)
