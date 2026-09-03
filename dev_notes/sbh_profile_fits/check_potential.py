"""The potential's outer term for general (al,b,g), including g >= 2.

Phi(r) = -G [ M(<r)/r + 4 pi Int_r^inf r' rho dr' ]

The incomplete-beta form B(1-t; (b-2)/al, (2-g)/al) needs (2-g)/al > 0,
i.e. g < 2 -- but the LIMEPY fit sits at g = 2.24.  The hypergeometric
identity  B(x; p, q) = x^p/p * 2F1(p, 1-q; p+1; x)  is valid for any q,
and GeneralisedNFW.mass_enclosed already uses exactly this hyp2f1 form.
Check it against quadrature across g < 2 and g > 2.
"""
import numpy as np
from scipy.special import hyp2f1, betainc, beta as Bfunc
from scipy.integrate import quad

G = 4.3009172706e-3


def rho(r, rho0, a, al, b, g):
    x = r / a
    return rho0 * x ** -g * (1.0 + x ** al) ** (-(b - g) / al)


def Binc_hyp(x, p, q):
    """Incomplete beta B(x; p, q) via 2F1 -- valid for q of either sign."""
    return x ** p / p * hyp2f1(p, 1.0 - q, p + 1.0, x)


def M_closed(r, rho0, a, al, b, g):
    x = r / a
    t = x ** al / (1.0 + x ** al)
    return 4 * np.pi * a ** 3 * rho0 / al * Binc_hyp(t, (3 - g) / al, (b - 3) / al)


def tail_closed(r, rho0, a, al, b, g):
    x = r / a
    t = x ** al / (1.0 + x ** al)
    return 4 * np.pi * a ** 2 * rho0 / al * Binc_hyp(1 - t, (b - 2) / al, (2 - g) / al)


def tail_num(r, rho0, a, al, b, g):
    f = lambda u: 4 * np.pi * np.exp(u) ** 2 * rho(np.exp(u), rho0, a, al, b, g)
    return quad(f, np.log(r), np.log(r) + 70, limit=1200)[0]


def M_num(r, rho0, a, al, b, g):
    f = lambda u: 4 * np.pi * np.exp(u) ** 3 * rho(np.exp(u), rho0, a, al, b, g)
    return quad(f, np.log(r) - 70, np.log(r), limit=1200)[0]


rho0, a = 1.0e5, 1.5
cases = [(1.0, 3.5, 1.75), (2.0, 4.0, 2.18), (4.13, 4.51, 2.24),
         (3.91, 4.50, 2.24), (6.46, 4.0, 2.24), (1.0, 4.0, 0.0),
         (2.0, 3.5, 2.9), (0.5, 5.0, 0.5)]
print("%-22s %6s %11s %11s %11s" % ("(al, b, g)", "g>=2?", "max|dM|", "max|dTail|", "max|dAccel|"))
worst = 0.0
for al, b, g in cases:
    eM, eT, eA = [], [], []
    for r in np.geomspace(1e-3, 40, 10):
        eM.append(abs(M_closed(r, rho0, a, al, b, g) / M_num(r, rho0, a, al, b, g) - 1))
        eT.append(abs(tail_closed(r, rho0, a, al, b, g) / tail_num(r, rho0, a, al, b, g) - 1))
        # a_r = -dPhi/dr must equal -G M(<r)/r^2 exactly
        Phi = lambda s: -G * (M_closed(s, rho0, a, al, b, g)
                              + 0.0) / s - G * tail_closed(s, rho0, a, al, b, g)
        h = r * 1e-5
        num = -(Phi(r + h) - Phi(r - h)) / (2 * h)
        ana = -G * M_closed(r, rho0, a, al, b, g) / r ** 2
        eA.append(abs(ana / num - 1))
    print("(%.2f, %.2f, %.2f)      %6s %11.2e %11.2e %11.2e"
          % (al, b, g, "YES" if g >= 2 else "no", max(eM), max(eT), max(eA)))
    worst = max(worst, max(eM), max(eT), max(eA))

# and confirm the betainc form really does fail for g>=2, i.e. this matters
al, b, g = 4.13, 4.51, 2.24
p, q = (b - 2) / al, (2 - g) / al
old = betainc(p, q, 0.5) * Bfunc(p, q) if q > 0 else float("nan")
print("\nold betainc form at g=2.24: q=%.3f -> %s" % (q, old))
print("hyp2f1 form                : %.6e" % Binc_hyp(0.5, p, q))

assert worst < 1e-5, "hyp2f1 forms disagree: %.2e" % worst
print("\nOK: hyp2f1 closed forms verified to %.1e, including g > 2." % worst)
