"""
Table 2 -- Tomographic reconstruction: relative error under oracle tuning,
averaged over 20 noise realizations.

Forward operator: parallel-beam Radon transform (scikit-image) of a 32 x 32
image at 24 angles in [0, 180), giving 46 x 24 = 1,104 measurements and 1,024
unknowns. Targets: a smooth two-bump image and the Shepp-Logan phantom.
Relative Gaussian noise of 1% and 5% (||e|| = level * ||A x||).
Estimators: oracle-tuned ridge (eta=1), finite IR with eta in {2, 5} (zeta
tuned), and the exponential limit (kappa tuned). Error is measured in the row
space of A:  ||V^T (x_hat - x)|| / ||V^T x||.

The smooth image's bump positions/widths are not given in the paper; the
choice below is ours, so the smooth rows are reproduced in pattern rather
than to three decimals.

Usage:  python table2_tomography.py [--reps 20] [--seed 0]
"""
import argparse
import numpy as np
from scipy.optimize import minimize_scalar
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

N, N_ANGLES = 32, 24


def radon_matrix(N=N, n_angles=N_ANGLES):
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    cols = []
    for k in range(N * N):
        e = np.zeros(N * N)
        e[k] = 1.0
        cols.append(radon(e.reshape(N, N), theta=theta, circle=False).ravel())
    return np.array(cols).T


def smooth_two_bump(N=N):
    g = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(g, g)
    bump = lambda cx, cy, w: np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w**2))
    return (bump(0.38, 0.42, 0.10) + 0.7 * bump(0.64, 0.60, 0.13)).ravel()


def shepp_logan(N=N):
    return resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).ravel()


def filt(s2, kind, p):
    if kind == "ridge":
        return s2 / (s2 + p)
    if kind == "inf":
        return -np.expm1(-s2 / p)
    return -np.expm1(-kind * np.log1p(s2 / p))          # finite IR, eta = kind


def oracle_error(kind, s, z, alpha):
    s2 = s**2
    norm = np.linalg.norm(alpha)
    f = lambda lp: np.linalg.norm(filt(s2, kind, np.exp(lp)) * z / s - alpha) / norm
    lo, hi = np.log(s2.min()) - 10, np.log(s2.max()) + 10
    grid = np.linspace(lo, hi, 400)                     # coarse grid, then refine
    i = int(np.argmin([f(g) for g in grid]))
    r = minimize_scalar(f, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 399)]),
                        method="bounded", options={"xatol": 1e-10})
    return r.fun


def main(reps=20, seed=0):
    A = radon_matrix()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    keep = s > s[0] * 1e-14
    U, s, Vt = U[:, keep], s[keep], Vt[keep]
    print(f"A: {A.shape}, numerical rank {s.size}, condition number {s[0] / s[-1]:.2e}")
    rng = np.random.default_rng(seed)
    kinds = [("Ridge", "ridge"), ("IR2", 2), ("IR5", 5), ("IRinf", "inf")]
    print(f"\n{'Signal':<13}{'Noise':>6}" + "".join(f"{k:>8}" for k, _ in kinds))
    for name, x in [("Smooth", smooth_two_bump()), ("Shepp-Logan", shepp_logan())]:
        b = A @ x
        alpha = Vt @ x
        for level in (0.01, 0.05):
            errs = np.zeros((reps, len(kinds)))
            for r in range(reps):
                e = rng.standard_normal(b.size)
                e *= level * np.linalg.norm(b) / np.linalg.norm(e)
                z = U.T @ (b + e)
                errs[r] = [oracle_error(k, s, z, alpha) for _, k in kinds]
            print(f"{name:<13}{level:>6.0%}" + "".join(f"{v:>8.3f}" for v in errs.mean(0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.reps, a.seed)
