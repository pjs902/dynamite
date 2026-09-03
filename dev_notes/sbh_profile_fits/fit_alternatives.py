"""Is there a form that fits BOTH LIMEPY and PhaseFlow better than a Zhao?

Candidates:
  - single Zhao (baseline, 5 par)
  - double Zhao, 3 slopes / 2 breaks (8 par), with heavy multistart
  - spherical MGE: sum of N Gaussians, masses by NNLS (N par, shape-free)

MGE is scored at N = 6, 10, 16 Gaussians. Both rho and M(<r) are analytic
for a Gaussian, so the MGE gives closed forms for everything DYNAMITE needs
-- and triaxpotent already implements them.
"""
import os
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import erf
from scipy.integrate import cumulative_trapezoid
from fit_joint import TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- analytic MGE ----------------------------------------
def mge_rho(r, M, sig):
    return np.sum(M[:, None] / ((2 * np.pi) ** 1.5 * sig[:, None] ** 3)
                  * np.exp(-r[None, :] ** 2 / (2 * sig[:, None] ** 2)), axis=0)


def mge_menc(r, M, sig):
    x = r[None, :] / sig[:, None]
    return np.sum(M[:, None] * (erf(x / np.sqrt(2))
                                - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2)), axis=0)


def fit_mge(r, rho, M, n):
    """NNLS for Gaussian masses, on log-spaced widths, relative-error weighted."""
    sig = np.geomspace(r[0] / 3, r[-1] * 3, n)
    # stack rho and M rows, each row scaled by 1/target => relative residuals
    A_rho = np.array([1.0 / ((2 * np.pi) ** 1.5 * s ** 3) * np.exp(-r ** 2 / (2 * s ** 2))
                      for s in sig]).T / rho[:, None]
    x = r[None, :] / sig[:, None]
    A_M = (erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x ** 2 / 2)).T / M[:, None]
    A = np.vstack([A_rho, A_M])
    b = np.ones(2 * len(r))
    w, _ = nnls(A, b)
    return w, sig


# ---------------- Zhao forms ------------------------------------------
def f_zhao(r, p):
    lrho0, la, al, b, g = p
    x = r / np.exp(la)
    return np.exp(lrho0) * x ** -g * (1 + x ** al) ** (-(b - g) / al)


def f_dzhao(r, p):
    lrho0, lr1, lr2, al1, al2, g1, g2, g3 = p
    x1, x2 = r / np.exp(lr1), r / np.exp(lr2)
    return (np.exp(lrho0) * x1 ** -g1 * (1 + x1 ** al1) ** ((g1 - g2) / al1)
            * (1 + x2 ** al2) ** ((g2 - g3) / al2))


def menc_num(f, p, r):
    rr = np.geomspace(r[0] * 1e-4, r[-1], 4000)
    M = cumulative_trapezoid(4 * np.pi * rr ** 2 * f(rr, p), rr, initial=0.0)
    return np.exp(np.interp(np.log(r), np.log(rr), np.log(np.maximum(M, 1e-300))))


def fit_form(r, rho, M, f, p0s, bnds):
    ly, lM = np.log(rho), np.log(M)
    def resid(p):
        return np.concatenate([np.log(np.maximum(f(r, p), 1e-300)) - ly,
                               np.log(menc_num(f, p, r)) - lM])
    best = None
    for p0 in p0s:
        try:
            s = least_squares(resid, p0, bounds=bnds, max_nfev=6000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    n = len(r)
    rr = best.fun[:n] / np.log(10); mm = best.fun[n:] / np.log(10)
    return np.sqrt(np.mean(rr ** 2)), np.sqrt(np.mean(mm ** 2)), best.x


rng = np.random.default_rng(0)
for tname in ["PhaseFlow BH, 1e-4-5 pc", "LIMEPY, full 1e-3-50 pc"]:
    r, rho, M = TARGETS[tname]
    print("\n=== %s" % tname)
    print("    %-26s %5s %9s %9s   %s" % ("form", "npar", "rho dex", "M dex", "max|dM| Msun"))

    # single Zhao
    p0s = [[np.log(rho[0]), s, 2.0, 4.0, 1.0] for s in (-6, -3, 0, 3)]
    b = ([-np.inf, -10, .2, 3.05, -.5], [np.inf, 10, 12, 12, 2.9])
    er, em, px = fit_form(r, rho, M, f_zhao, p0s, b)
    dM = np.max(np.abs(menc_num(f_zhao, px, r) - M))
    print("    %-26s %5d %9.3f %9.3f   %12.0f" % ("single Zhao", 5, er, em, dM))

    # double Zhao, 60 random restarts
    p0s = []
    for _ in range(60):
        p0s.append([np.log(rho[0]) + rng.normal(0, 2),
                    rng.uniform(np.log(r[0]), np.log(r[-1])),
                    rng.uniform(np.log(r[0]), np.log(r[-1])),
                    rng.uniform(.5, 5), rng.uniform(.5, 5),
                    rng.uniform(-.4, 2.5), rng.uniform(0, 3.5), rng.uniform(2, 8)])
    b = ([-np.inf, -10, -10, .2, .2, -.5, -.5, -.5],
         [np.inf, 10, 10, 12, 12, 2.9, 8, 14])
    er, em, px = fit_form(r, rho, M, f_dzhao, p0s, b)
    dM = np.max(np.abs(menc_num(f_dzhao, px, r) - M))
    print("    %-26s %5d %9.3f %9.3f   %12.0f" % ("double Zhao (3 slopes)", 8, er, em, dM))

    # MGE
    for n in (6, 10, 16):
        w, sig = fit_mge(r, rho, M, n)
        mr, mm = mge_rho(r, w, sig), mge_menc(r, w, sig)
        er = np.sqrt(np.mean(np.log10(mr / rho) ** 2))
        em = np.sqrt(np.mean(np.log10(mm / M) ** 2))
        print("    %-26s %5d %9.3f %9.3f   %12.0f"
              % ("MGE, %d Gaussians" % n, int((w > 0).sum()), er, em,
                 np.max(np.abs(mm - M))))
