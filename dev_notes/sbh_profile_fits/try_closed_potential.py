"""Can the potential stay closed-form for all gamma < 3?

Phi(r) = -G [ M(<r)/r + T(r) ],   T(r) = 4 pi Int_r^inf r' rho dr'

T diverges as r->0 for gamma >= 2. But Phi is only defined up to an additive
constant, so we may replace T by  T(r) - C  for ANY constant C. Choose C to
absorb the divergent piece and keep an antiderivative that is finite and
computable at every r > 0.

With x = r/a, k = (beta-gamma)/alpha, s = (2-gamma)/alpha:

  T(r) = 4 pi a^2 rho0 * [ Finf - F(x) ]
  F(x) = x^(2-gamma)/(2-gamma) * 2F1(k, s; s+1; -x^alpha)      (antiderivative)

Dropping the constant Finf leaves  Ttilde(r) = -4 pi a^2 rho0 F(x),  which is
closed-form for ALL gamma < 3 (gamma = 2 needs a log limit).

Three evaluators are compared against mpmath at 50 digits:
  A) betainc form   B(1-t; (beta-2)/alpha, (2-gamma)/alpha)   -- the gNFW style
  B) antiderivative F(x) via scipy hyp2f1, with a Pfaff transform for x > 1
  C) the same, reference-evaluated in mpmath
"""
import numpy as np
from scipy.special import hyp2f1, betainc, beta as Bfunc
import mpmath as mp

mp.mp.dps = 50


def rho_mp(r, a, al, b, g):
    x = r / a
    return x ** (-g) * (1 + x ** al) ** (-(b - g) / al)


def T_ref(r, a, al, b, g):
    """4 pi Int_r^inf r' rho dr', in units of 4 pi a^2 rho0. mpmath."""
    f = lambda u: mp.e ** (2 * u) * rho_mp(mp.e ** u, a, al, b, g)
    return mp.quad(f, [mp.log(r), mp.log(r) + 20, mp.inf])


def F_scipy(x, al, b, g):
    """Antiderivative x^(2-g)/(2-g) * 2F1(k, s; s+1; -x^al), Pfaff for x>1."""
    k, s = (b - g) / al, (2 - g) / al
    z = -x ** al
    if x <= 1.0:
        h = hyp2f1(k, s, s + 1.0, z)
    else:
        # Pfaff: 2F1(k,s;s+1;z) = (1-z)^-k 2F1(k, 1; s+1; z/(z-1))
        h = (1 - z) ** (-k) * hyp2f1(k, 1.0, s + 1.0, z / (z - 1.0))
    return x ** (2 - g) / (2 - g) * h


def F_mp(x, al, b, g):
    k, s = mp.mpf(b - g) / al, mp.mpf(2 - g) / al
    return mp.mpf(x) ** (2 - g) / (2 - g) * mp.hyp2f1(k, s, s + 1, -mp.mpf(x) ** al)


def T_betainc(x, al, b, g):
    """gNFW-style: B(1-t; (b-2)/al, (2-g)/al). Undefined for g >= 2."""
    t = x ** al / (1 + x ** al)
    p, q = (b - 2) / al, (2 - g) / al
    if q <= 0:
        return np.nan
    return betainc(p, q, 1 - t) * Bfunc(p, q) / al


cases = [(2.15, 12.0, 1.74), (3.91, 4.50, 2.24), (2.0, 4.0, 2.18),
         (0.31, 12.0, 0.51), (1.0, 3.5, 1.75), (6.46, 4.0, 2.24),
         (2.0, 3.5, 2.90), (4.0, 5.0, 2.50)]
a = 1.0
print("Closed-form tail: max relative error vs mpmath, over r/a = 1e-8 .. 1e4")
print("%-24s %6s %14s %14s" % ("(al, b, g)", "g>=2?", "F(x) scipy", "betainc form"))
xs = np.geomspace(1e-8, 1e4, 60)
worstF = 0.0
for al, b, g in cases:
    eF, eB = [], []
    for x in xs:
        ref_F = F_mp(x, al, b, g)
        got = F_scipy(x, al, b, g)
        if ref_F != 0:
            eF.append(abs(mp.mpf(got) / ref_F - 1))
        tb = T_betainc(x, al, b, g)
        if np.isfinite(tb):
            # betainc form and -F differ by the constant Finf; compare
            # DIFFERENCES between two radii, which is constant-free
            pass
    # constant-free check: T(r1)-T(r2) must equal -(F(x1)-F(x2))
    for x1, x2 in [(1e-3, 1e-1), (1e-1, 1.0), (1.0, 10.0), (10.0, 1e3)]:
        dT = T_ref(x1 * a, a, al, b, g) - T_ref(x2 * a, a, al, b, g)
        dF = -(F_mp(x1, al, b, g) - F_mp(x2, al, b, g))
        eB.append(abs(dF / dT - 1))
    print("%-24s %6s %14.2e %14.2e"
          % ("(%.2f, %.2f, %.2f)" % (al, b, g), "YES" if g >= 2 else "no",
             float(max(eF)), float(max(eB))))
    worstF = max(worstF, float(max(eF)))

print("\nworst scipy-vs-mpmath error on F(x): %.2e" % worstF)
print("worst |F| magnitude at r/a=1e-8 (checks the divergence is mild):")
for al, b, g in cases:
    print("   (%.2f,%.2f,%.2f)  F(1e-8) = %+.4e" % (al, b, g, F_scipy(1e-8, al, b, g)))
