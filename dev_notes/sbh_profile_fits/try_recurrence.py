"""Incomplete beta with q <= 0 via downward recurrence from positive q.

  B(x;p,q) = [ (p+q) B(x;p,q+1) - x^p (1-x)^q ] / q

Start at q+n > 0 (computable by any standard betai, e.g. the Fortran
zh_betai already in dmpotent.f90), step down n times. This is what the
Fortran would do; the reference is mpmath at 50 digits.

Target: the potential's outer term
  T(r) = (4 pi a^2 rho0 / alpha) * B(1-t; (beta-2)/alpha, (2-gamma)/alpha)
with t = x^alpha/(1+x^alpha), which needs q = (2-gamma)/alpha <= 0 whenever
gamma >= 2.
"""
import numpy as np
from scipy.special import betainc, beta as Bfunc
import mpmath as mp

mp.mp.dps = 50


def B_pos(x, p, q):
    """Unregularized incomplete beta, q > 0 (stands in for zh_betai)."""
    return betainc(p, q, x) * Bfunc(p, q)


def B_any(x, p, q):
    """Unregularized incomplete beta for any q != 0, 0 < x < 1."""
    if q > 0:
        return B_pos(x, p, q)
    n = int(np.ceil(1.0 - q)) + 1          # steps to reach q+n > 0
    val = B_pos(x, p, q + n)
    for j in range(n, 0, -1):              # q+n -> q+n-1 -> ... -> q
        qq = q + j - 1                     # target of this step
        val = ((p + qq) * val - x ** p * (1.0 - x) ** qq) / qq
    return val


def B_ref(x, p, q):
    f = lambda u: u ** (mp.mpf(p) - 1) * (1 - u) ** (mp.mpf(q) - 1)
    return mp.quad(f, [0, mp.mpf(x)])


cases = [(2.15, 12.0, 1.74), (3.91, 4.50, 2.24), (2.0, 4.0, 2.18),
         (0.31, 12.0, 0.51), (1.0, 3.5, 1.75), (6.46, 4.0, 2.24),
         (2.0, 3.5, 2.90), (4.0, 5.0, 2.50), (1.0, 4.0, 0.0)]

print("Downward recurrence vs mpmath, over r/a = 1e-6 .. 1e4")
print("%-22s %7s %7s %14s" % ("(al, b, g)", "q", "steps", "max rel err"))
worst = 0.0
for al, b, g in cases:
    p, q = (b - 2) / al, (2 - g) / al
    n = 0 if q > 0 else int(np.ceil(1.0 - q)) + 1
    errs = []
    for x in np.geomspace(1e-6, 1e4, 40):
        t = x ** al / (1 + x ** al)
        y = 1.0 - t
        if not (0 < y < 1):
            continue
        got = B_any(y, p, q)
        ref = B_ref(y, p, q)
        if ref != 0:
            errs.append(abs(mp.mpf(got) / ref - 1))
    e = float(max(errs))
    print("%-22s %7.3f %7d %14.2e"
          % ("(%.2f, %.2f, %.2f)" % (al, b, g), q, n, e))
    worst = max(worst, e)

print("\nworst relative error: %.2e" % worst)
assert worst < 1e-8, "recurrence not accurate enough: %.2e" % worst
print("OK: closed form valid for all gamma < 3 using only a positive-q betai"
      "\n    plus %d-step recurrence. No tabulation needed." % n)
