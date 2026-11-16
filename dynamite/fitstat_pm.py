"""Per-star likelihood terms for veldist histogram sets (``statistic: laplace``).

Port of veldist ``src/veldist/fitstat.py`` (design: veldist
``docs/fit-statistic-design.md``), kept dependency-free. Every ``h`` is a
flat histogram of cell MASSES in DYNAMITE units (the aperture's model mass;
data normalised to 1 and times the MGE projected mass ``mass``).

The exact statistic, extended Poisson in mass units, is
``-2 log L(h) = 2 [-sum_s log(m_s . h) + (n_stars / mass) sum(h)]`` with
``m_s`` star s's design row. Its expansion about ``h0`` is

    Q(h) = (h - h0)' (F + delta I) (h - h0) + 2 g' (h - h0)

with ``F = M' diag(1/p0^2) M``, ``g = (n/mass) 1 - M' (1/p0)``, ``p0 = M h0``.
The quadratic part goes to the solver as rows ``Lt h`` against ``Lt h0``
(``Lt' Lt = F + delta I``, unit errors), the linear part as ``2 g' h`` into
the cross term directly. Folding ``g`` into the rows instead (veldist's
``laplace_rows``) puts a constant ``g' (F + delta I)^-1 g`` of ~1e8 per
aperture into the quadratic form: cells without stars have ``F ~ 0`` and
``g = n/mass``, i.e. a linear penalty carried by the tiny ridge. In the
float32 Gram path that constant would swamp the ~1e3 differences between
models. With ``sum(h0) = mass``, ``g' h0 = n - n = 0``, so ``Q`` is offset-free.
"""

import numpy as np

__all__ = ["laplace_terms", "laplace_lt", "default_delta", "stat_offset_free",
           "stat_exact"]


def default_delta(n_stars, mass):
    """Ridge ``1e-3 n / mass^2`` (F scales as n / mass^2). veldist C1b: a
    100x smaller ridge biased sigma by ~1 km/s; this one does not."""
    return 1e-3 * n_stars / mass ** 2


def laplace_terms(M, h0, n_stars, mass):
    """``F`` (K2, K2) and ``g`` (K2,) of ``-log L`` at ``h0``; ``M`` dense (n, K2)."""
    M = np.asarray(M, dtype=np.float64)
    p0 = M @ h0
    F = M.T @ ((1.0 / p0 ** 2)[:, None] * M)
    g = n_stars / mass - M.T @ (1.0 / p0)
    return F, g


def laplace_lt(F, delta):
    """``Lt`` with ``Lt' Lt = F + delta I`` (upper Cholesky factor)."""
    return np.linalg.cholesky(F + delta * np.eye(F.shape[0])).T


def stat_offset_free(h, Lt, g, h0):
    """``Q(h) - Q(h0)`` = ``|Lt (h - h0)|^2 + 2 g' (h - h0)``, ``-2 log L`` scale."""
    d = np.asarray(h, dtype=np.float64) - h0
    q = Lt @ d
    return float(q @ q + 2.0 * (g @ d))


def stat_exact(h, M, n_stars, mass):
    """``-2 log L`` (exact, extended Poisson in mass units)."""
    p = np.asarray(M, dtype=np.float64) @ h
    return float(2.0 * (-np.sum(np.log(np.maximum(p, 1e-300)))
                        + n_stars / mass * np.sum(h)))
