"""Realistic chi2 landscapes for generator comparison.

Three surfaces whose structure mirrors what Schwarzschild orbit-superposition
chi2 surfaces actually look like (ranges calibrated to the NGC6278 runs):

1. qml_ridge (ml, q, p): flattening valley with mass-flattening tilt and
   banana curvature in (p, q) - the shape-parameter problem.
2. production (log10 MBH, ml, q, p): adds a tight, near-orthogonal BH bowl;
   the free set of NGC5139_config_production.yaml.
3. halo_banana (ml, c-dh, f-dh): the classic strong degeneracy - total mass
   inside the aperture pins a curved valley across (log f, ml) whose position
   bends with c; steep across, nearly flat along it.

Shared realism features: exponential tails (heavier than Gaussian), small
orbit-library-style ripple, anisotropic stiffness, known global minimum.
"""

import numpy as np


def _ripple(x, amp, seed):
    """Small multiplicative jitter, mimicking discrete orbit libraries."""
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(6, x.shape[-1]))
    ph = rng.uniform(0, 2 * np.pi, size=6)
    u = x - 0.5
    s = np.sin(u @ k.T + ph)
    return 1.0 + amp * np.mean(s, axis=-1)


class Landscape:
    def __init__(self, name, names, bounds, func, threshold, desc):
        self.name = name
        self.names = names  # parameter display names
        self.bounds = bounds  # list of (lo, hi) per axis
        self._func = func
        self.threshold = threshold  # 'converged' chi2 level
        self.desc = desc

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._func(X)

    def grid(self, n=120, axes=(0, 1), other=None):
        """Chi2 on a 2D grid over `axes`, other coords fixed at `other`."""
        other = other if other is not None else [0.5 * (lo + hi) for lo, hi in self.bounds]
        xs = np.linspace(self.bounds[axes[0]][0], self.bounds[axes[0]][1], n)
        ys = np.linspace(self.bounds[axes[1]][0], self.bounds[axes[1]][1], n)
        G, H = np.meshgrid(xs, ys)
        pts = np.tile(np.asarray(other, dtype=float), (G.size, 1))
        pts[:, axes[0]] = G.ravel()
        pts[:, axes[1]] = H.ravel()
        Z = self(pts).reshape(G.shape)
        return G, H, Z


# ---------------------------------------------------------------------------
# 1. shape ridge: ml, q, p
# ---------------------------------------------------------------------------


def _qml_ridge(X):
    ml, q, p = X[:, 0], X[:, 1], X[:, 2]
    # curved locus of best (p, q): banana in the triaxiality plane,
    # p_opt = p0 + curvature * (q - q0)^2 - flattening-mass tilt with ml
    q0 = 0.53 + 0.06 * (ml - 3.0)  # rounder -> slightly larger ml
    p0 = 0.94 + 0.35 * (q - 0.5) ** 2  # banana curvature
    d_q = (q - q0) / 0.075
    d_p = (p - p0) / 0.030
    chi_shape = 90.0 * np.sqrt(d_q**2 + d_p**2 + 0.15 * d_q * d_p) + 12.0 * d_p**4  # asymmetric stiff wall in p
    chi_ml = 260.0 * (ml - 3.55) ** 2 / 1.44
    base = 6420.0 + chi_shape + chi_ml
    return base * _ripple(X, 0.004, 11) - 420.0 * np.exp(-((d_q / 1.2) ** 2 + ((p - p0) / 0.05) ** 2))


QML_RIDGE = Landscape(
    "shape-ridge",
    ["ml", "q", "p"],
    [(1.0, 5.0), (0.30, 0.89), (0.90, 0.999)],
    _qml_ridge,
    threshold=6075.0,
    desc="flattening valley + mass-flattening tilt; banana in (p, q)",
)


# ---------------------------------------------------------------------------
# 2. production-like: log10 MBH, ml, q, p
# ---------------------------------------------------------------------------


def _production(X):
    mbh, ml, q, p = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    # BH: tight bowl in log-space, slight tilt with q (central kinematics)
    bh0 = 4.31 + 0.25 * (q - 0.52)
    chi_bh = 950.0 * (mbh - bh0) ** 2 / 0.0225
    # shapes: elongated tilted band (correlated q-p), weakly coupled to ml
    q0 = 0.50 + 0.04 * (ml - 3.5) / 2.0
    d_q = (q - q0) / 0.085
    d_p = (p - (0.965 + 0.25 * (q - 0.5))) / 0.02
    chi_sh = 240.0 * (d_q**2 + 0.8 * d_q * d_p + d_p**2) + 20.0 * d_p**3
    chi_ml = 300.0 * (ml - 3.45) ** 2 / 1.0
    base = 6380.0 + chi_bh + chi_sh + chi_ml
    return base * _ripple(X, 0.003, 23) - 380.0 * np.exp(-(d_q**2 + d_p**2))


PRODUCTION = Landscape(
    "production-4d",
    ["log10 MBH", "ml", "q", "p"],
    [(3.90, 4.78), (1.0, 5.0), (0.30, 0.89), (0.90, 0.999)],
    _production,
    threshold=6120.0,
    desc="tight orthogonal BH bowl over an elongated shape band",
)


# ---------------------------------------------------------------------------
# 3. halo banana: ml, c-dh, f-dh (log M200/M*)
# ---------------------------------------------------------------------------


def _halo_banana(X):
    ml, c, lf = X[:, 0], X[:, 1], X[:, 2]
    # constant-total-mass valley: ml * (1 + 10^lf * g(c)) ~ const, where the
    # halo weight g(c) shrinks with concentration -> valley bends in
    # (lf, ml); c only matters through that bending => flat direction.
    g = 10.0 ** (-0.08 * c)
    m_ap = ml * (1.0 + 10.0**lf * g)
    d_mass = (m_ap - 3.62) / 0.16  # steep across the valley
    d_c = (c - 6.0) / 6.5  # gentle along it
    chi = 520.0 * np.sqrt(d_mass**2 + 0.02 * d_c**2)
    # secondary basin: high-c, low-f branch end (a real nuisance in halos)
    sec = 130.0 * np.exp(-(((c - 14.0) / 3.0) ** 2 + ((lf + 7.6) / 0.35) ** 2))
    base = 6400.0 + chi - sec + 180.0 * d_c**2 * (c > 12.0)
    # valley floor sits ~480 below the rim, like the real runs' dynamic range
    return base * _ripple(X, 0.004, 37) - 480.0


HALO_BANANA = Landscape(
    "halo-banana",
    ["ml", "c-dh", "log f-dh"],
    [(1.0, 5.0), (0.0, 20.0), (-10.0, -5.0)],
    _halo_banana,
    threshold=6180.0,
    desc="curved constant-mass valley; flat in c; secondary high-c basin",
)
