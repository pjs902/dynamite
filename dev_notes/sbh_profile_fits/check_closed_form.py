"""Verify closed forms for the general Zhao (alpha,beta,gamma) profile.

rho(r) = rho0 * x^-g * (1+x^al)^(-(b-g)/al),   x = r/a

Claim 1:  M(<r) = (4 pi a^3 rho0 / al) * B(t; (3-g)/al, (b-3)/al),  t = x^al/(1+x^al)
Claim 2:  M_tot = (4 pi a^3 rho0 / al) * B((3-g)/al, (b-3)/al)        [complete]
Claim 3:  4 pi Int_r^inf r' rho dr' = (4 pi a^2 rho0 / al) * B(1-t; (b-2)/al, (2-g)/al)
          so  Phi(r) = -G [ M(<r)/r + that ]

Both need b > 3 (mass) and b > 2 (potential tail); g < 3.
If these hold, the Fortran needs only an incomplete beta -- no quadrature,
no interpolation table -- in the orbit-integration inner loop.
"""
import numpy as np
from scipy.special import betainc, beta as Bfunc
from scipy.integrate import quad

G = 4.3009172706e-3  # pc (km/s)^2 / Msun


def rho(r, rho0, a, al, b, g):
    x = r / a
    return rho0 * x ** -g * (1.0 + x ** al) ** (-(b - g) / al)


def M_closed(r, rho0, a, al, b, g):
    x = r / a
    t = x ** al / (1.0 + x ** al)
    p, q = (3 - g) / al, (b - 3) / al
    return 4 * np.pi * a ** 3 * rho0 / al * betainc(p, q, t) * Bfunc(p, q)


def Mtot_closed(rho0, a, al, b, g):
    p, q = (3 - g) / al, (b - 3) / al
    return 4 * np.pi * a ** 3 * rho0 / al * Bfunc(p, q)


def tail_closed(r, rho0, a, al, b, g):
    x = r / a
    t = x ** al / (1.0 + x ** al)
    p, q = (b - 2) / al, (2 - g) / al
    return 4 * np.pi * a ** 2 * rho0 / al * betainc(p, q, 1 - t) * Bfunc(p, q)


def M_num(r, rho0, a, al, b, g):
    """Integrate in log r: s=e^u, ds=s du -- keeps the integrand tame."""
    f = lambda u: 4 * np.pi * np.exp(u) ** 3 * rho(np.exp(u), rho0, a, al, b, g)
    return quad(f, np.log(r) - 60, np.log(r), limit=800)[0]


def tail_num(r, rho0, a, al, b, g):
    f = lambda u: 4 * np.pi * np.exp(u) ** 2 * rho(np.exp(u), rho0, a, al, b, g)
    return quad(f, np.log(r), np.log(r) + 60, limit=800)[0]


def Mtot_num(rho0, a, al, b, g):
    f = lambda u: 4 * np.pi * np.exp(u) ** 3 * rho(np.exp(u), rho0, a, al, b, g)
    return quad(f, np.log(a) - 80, np.log(a) + 80, limit=1600)[0]


cases = [
    # (al, b, g)  -- spanning the fitted region and the two legacy specials
    (1.0, 3.5, 1.75), (2.0, 4.0, 2.18), (4.13, 4.51, 2.24),
    (2.81, 4.0, 1.74), (0.5, 5.0, 0.5), (6.46, 4.0, 2.24),
    (1.0, 4.0, 0.0), (3.0, 3.2, 2.5),
]
rho0, a = 1.0e5, 1.5
print("%-22s %11s %11s %11s" % ("(al, b, g)", "max|dM|", "max|dTail|", "M_tot check"))
worst = 0.0
for al, b, g in cases:
    errM, errT = [], []
    for r in np.geomspace(1e-3, 60, 12):
        mc, mn = M_closed(r, rho0, a, al, b, g), M_num(r, rho0, a, al, b, g)
        tc, tn = tail_closed(r, rho0, a, al, b, g), tail_num(r, rho0, a, al, b, g)
        errM.append(abs(mc / mn - 1)); errT.append(abs(tc / tn - 1))
    mt = Mtot_closed(rho0, a, al, b, g)
    mt_n = Mtot_num(rho0, a, al, b, g)
    print("(%.2f, %.2f, %.2f)      %11.2e %11.2e %11.2e"
          % (al, b, g, max(errM), max(errT), abs(mt / mt_n - 1)))
    worst = max(worst, max(errM), max(errT))

# acceleration is exact from M(<r): a_r = -G M(<r)/r^2, check against -dPhi/dr
print("\nradial acceleration vs numerical dPhi/dr:")
al, b, g = 4.13, 4.51, 2.24
Phi = lambda r: -G * (M_closed(r, rho0, a, al, b, g) / r
                      + tail_closed(r, rho0, a, al, b, g))
for r in [0.05, 0.5, 5.0]:
    h = r * 1e-5
    num = -(Phi(r + h) - Phi(r - h)) / (2 * h)
    ana = -G * M_closed(r, rho0, a, al, b, g) / r ** 2
    print("  r=%6.3f  analytic %+.6e  numeric %+.6e  rel %.2e"
          % (r, ana, num, abs(ana / num - 1)))

assert worst < 1e-6, "closed forms disagree with quadrature: %.2e" % worst
print("\nOK: closed forms verified to %.1e over all cases." % worst)
