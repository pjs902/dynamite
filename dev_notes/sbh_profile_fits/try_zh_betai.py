"""Does the existing Fortran zh_betai already handle q < 0?

Faithful Python port of legacy_fortran/sub/specfunc_beta.f90, tested against
mpmath. If the NR continued fraction is accurate for b < 0, the Fortran
needs a single zh_betai call; otherwise use the downward recurrence.
"""
import numpy as np
import mpmath as mp
from math import lgamma, exp

mp.mp.dps = 50
MAXIT, EPS, FPMIN = 100, 3.0e-16, 1.0e-300


def zh_betacf(a, b, x):
    """Numerical Recipes betacf, transcribed from specfunc_beta.f90."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            return h
    return np.nan          # 'a or b too big, or MAXIT too small'


def zh_beta(z, w):
    return exp(lgamma(z) + lgamma(w) - lgamma(z + w))


def zh_betai(a, b, x):
    """Unregularized incomplete beta, as in specfunc_beta.f90."""
    if x < 0.0 or x > 1.0:
        return np.nan
    if x == 0.0:
        bt = 0.0
    elif x == 1.0:
        return zh_beta(a, b)
    else:
        bt = x ** a * (1.0 - x) ** b
    if x < (a + 1.0) / (a + b + 2.0) or b <= 0.0:
        return bt * zh_betacf(a, b, x) / a
    return zh_beta(a, b) - bt * zh_betacf(b, a, 1.0 - x) / b


def B_any_recurrence(x, p, q):
    if q > 0:
        return zh_betai(p, q, x)
    n = int(np.ceil(1.0 - q)) + 1
    val = zh_betai(p, q + n, x)
    for j in range(n, 0, -1):
        qq = q + j - 1
        val = ((p + qq) * val - x ** p * (1.0 - x) ** qq) / qq
    return val


def B_ref(x, p, q):
    f = lambda u: u ** (mp.mpf(p) - 1) * (1 - u) ** (mp.mpf(q) - 1)
    return mp.quad(f, [0, mp.mpf(x)])


cases = [(2.15, 12.0, 1.74), (3.91, 4.50, 2.24), (2.0, 4.0, 2.18),
         (0.31, 12.0, 0.51), (1.0, 3.5, 1.75), (6.46, 4.0, 2.24),
         (2.0, 3.5, 2.90), (4.0, 5.0, 2.50), (1.0, 4.0, 0.0)]

print("%-22s %7s | %-12s | %-12s" % ("(al, b, g)", "q", "direct call", "recurrence"))
wd = wr = 0.0
for al, b, g in cases:
    p, q = (b - 2) / al, (2 - g) / al
    ed, er = [], []
    for x in np.geomspace(1e-6, 1e4, 40):
        y = 1.0 / (1.0 + x ** al)          # = 1 - t
        if not (0.0 < y < 1.0):
            continue
        ref = B_ref(y, p, q)
        if ref == 0:
            continue
        d = zh_betai(p, q, y)
        r = B_any_recurrence(y, p, q)
        ed.append(abs(mp.mpf(d) / ref - 1) if np.isfinite(d) else mp.inf)
        er.append(abs(mp.mpf(r) / ref - 1) if np.isfinite(r) else mp.inf)
    md, mr = float(max(ed)), float(max(er))
    print("%-22s %7.3f | %12.2e | %12.2e"
          % ("(%.2f, %.2f, %.2f)" % (al, b, g), q, md, mr))
    wd, wr = max(wd, md), max(wr, mr)

print("\nworst direct zh_betai(q<0) : %.2e" % wd)
print("worst recurrence           : %.2e" % wr)
print("\n-> %s" % ("direct call suffices, no recurrence needed"
                   if wd < 1e-10 else
                   "direct call NOT reliable; use the recurrence"))
