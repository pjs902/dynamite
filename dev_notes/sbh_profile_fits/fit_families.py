"""Which Zhao/abg variant matches BOTH external sBH profiles?

Family:  rho(r) = rho0 * (r/a)^-g * (1 + (r/a)^al)^(-(b-g)/al)
Variants differ only in which shape exponents are FREE vs hardcoded.

Fit in log(rho) vs log(r) on a log-uniform grid => equal weight per decade.
Score = RMS of log10 residual (0.05 dex ~ 12% in density).
"""
import os
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))


def zhao(r, lrho0, la, al, b, g):
    x = r / np.exp(la)
    return lrho0 - g * np.log(x) - (b - g) / al * np.log1p(x ** al)


# name -> (free-parameter names, fixed values, n_free)
VARIANTS = {
    "gNFW (legacy code 5: al=1, b=3)":      dict(free=["g"], al=1.0, b=3.0),
    "Zhao al=2, b=4 fixed":                 dict(free=["g"], al=2.0, b=4.0),
    "Zhao al=4, b=3 fixed (notebook fit)":  dict(free=["g"], al=4.0, b=3.0),
    "Zhao b free (al=2)":                   dict(free=["g", "b"], al=2.0),
    "Zhao al free (b=4)":                   dict(free=["g", "al"], b=4.0),
    "Zhao al,b,g all free":                 dict(free=["g", "al", "b"]),
}
GUESS = {"g": 1.0, "al": 2.0, "b": 4.0}
BOUNDS = {"g": (-0.5, 2.9), "al": (0.2, 10.0), "b": (3.01, 12.0)}


def fit(r, rho, spec):
    lr, ly = np.log(r), np.log(rho)
    free = spec["free"]
    p0 = [np.max(ly), np.log(np.median(r))] + [GUESS[k] for k in free]
    lo = [-np.inf, np.log(r.min() / 100)] + [BOUNDS[k][0] for k in free]
    hi = [np.inf, np.log(r.max() * 100)] + [BOUNDS[k][1] for k in free]

    def resid(p):
        kw = dict(al=spec.get("al"), b=spec.get("b"), g=spec.get("g"))
        for k, v in zip(free, p[2:]):
            kw[k] = v
        return zhao(r, p[0], p[1], kw["al"], kw["b"], kw["g"]) - ly

    best = None
    for scale in (0.03, 0.3, 3.0, 30.0):          # multistart on scale radius
        q0 = list(p0); q0[1] = np.log(scale)
        if not (lo[1] < q0[1] < hi[1]):
            continue
        try:
            s = least_squares(resid, q0, bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    rms = np.sqrt(np.mean((best.fun / np.log(10)) ** 2))
    kw = dict(al=spec.get("al"), b=spec.get("b"), g=spec.get("g"))
    for k, v in zip(free, best.x[2:]):
        kw[k] = v
    return rms, np.exp(best.x[1]), kw, len(free) + 2


# ---- targets ----------------------------------------------------------
pf = np.load(os.path.join(HERE, "target_phaseflow.npz"))
gc = np.load(os.path.join(HERE, "target_gcfit.npz"))
gr, grho = gc["r"], gc["rho"][2]
ok = (gr > 0) & (grho > 0)
gr, grho = gr[ok], grho[ok]

# resample each onto a log-uniform grid over the range that matters
def regrid(r, rho, lo, hi, n=200):
    m = (r >= lo) & (r <= hi)
    g = np.geomspace(max(lo, r[m][0]), min(hi, r[m][-1]), n)
    return g, np.exp(np.interp(np.log(g), np.log(r[m]), np.log(rho[m])))

TARGETS = {
    "PhaseFlow fiducial BH (cusp), 1e-4-5 pc": regrid(pf["r"], pf["rho"], 1e-4, 5.0),
    "GCfit LIMEPY rho_BH (core), 1e-3-20 pc":  regrid(gr, grho, 1e-3, 20.0),
    "GCfit LIMEPY, observable 0.05-10 pc":     regrid(gr, grho, 0.05, 10.0),
}

for tname, (r, rho) in TARGETS.items():
    print("\n=== %s   (%d pts)" % (tname, len(r)))
    print("    %-38s %8s %10s  %s" % ("variant", "RMS dex", "a [pc]", "shape"))
    for vname, spec in VARIANTS.items():
        rms, a, kw, npar = fit(r, rho, spec)
        print("    %-38s %8.3f %10.4g  g=%.2f al=%.2f b=%.2f  (%d par)"
              % (vname, rms, a, kw["g"], kw["al"], kw["b"], npar))
