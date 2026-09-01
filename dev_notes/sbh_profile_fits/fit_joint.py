"""Score Zhao variants on BOTH rho(r) and the enclosed-mass profile M(<r).

Joint objective: concatenate log-residuals in rho and in M(<r), equal weight
per decade in r for each. M(<r) is what the orbits actually feel, and with an
M_sBH normalisation a bad M(<r) is a biased total mass.
"""
import os, re, glob, configparser
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
PF = "/Users/pesmith/research/phaseflow/omegacen"


def zhao_rho(r, rho0, a, al, b, g):
    x = r / a
    return rho0 * x ** -g * (1.0 + x ** al) ** (-(b - g) / al)


def zhao_menc(r, rho0, a, al, b, g):
    """M(<r) by quadrature on a fine log grid (no closed form for general al)."""
    rr = np.geomspace(r[0] * 1e-3, r[-1], 3000)
    integrand = 4 * np.pi * rr ** 2 * zhao_rho(rr, rho0, a, al, b, g)
    M = cumulative_trapezoid(integrand, rr, initial=0.0)
    return np.exp(np.interp(np.log(r), np.log(rr), np.log(np.maximum(M, 1e-300))))


VARIANTS = {
    "gNFW (legacy 5: al=1,b=3)":  dict(free=["g"], al=1.0, b=3.0),
    "al=4, b=3 (notebook)":       dict(free=["g"], al=4.0, b=3.0),
    "al=2, b=4":                  dict(free=["g"], al=2.0, b=4.0),
    "b free (al=2)":              dict(free=["g", "b"], al=2.0),
    "al free (b=4)":              dict(free=["g", "al"], b=4.0),
    "al,b,g all free":            dict(free=["g", "al", "b"]),
}
GUESS = {"g": 1.0, "al": 2.0, "b": 4.0}
BOUNDS = {"g": (-0.5, 2.9), "al": (0.2, 12.0), "b": (3.05, 12.0)}


def fit(r, rho, M, spec, w_mass=1.0):
    free = spec["free"]
    ly, lM = np.log(rho), np.log(M)

    def unpack(p):
        kw = dict(al=spec.get("al"), b=spec.get("b"), g=spec.get("g"))
        for k, v in zip(free, p[2:]):
            kw[k] = v
        return np.exp(p[0]), np.exp(p[1]), kw

    def resid(p):
        rho0, a, kw = unpack(p)
        rm = np.log(zhao_rho(r, rho0, a, kw["al"], kw["b"], kw["g"])) - ly
        mm = np.log(zhao_menc(r, rho0, a, kw["al"], kw["b"], kw["g"])) - lM
        return np.concatenate([rm, w_mass * mm]) / np.sqrt(len(r))

    lo = [-np.inf, np.log(r.min() / 100)] + [BOUNDS[k][0] for k in free]
    hi = [np.inf, np.log(r.max() * 100)] + [BOUNDS[k][1] for k in free]
    best = None
    for sc in (0.01, 0.1, 1.0, 10.0, 100.0):
        p0 = [np.log(rho[0]), np.log(sc)] + [GUESS[k] for k in free]
        if not (lo[1] < p0[1] < hi[1]):
            continue
        try:
            s = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    rho0, a, kw = unpack(best.x)
    n = len(r)
    rr = best.fun[:n] * np.sqrt(n) / np.log(10)
    mm = best.fun[n:] * np.sqrt(n) / np.log(10) / w_mass
    Mtot = zhao_menc(np.array([r[-1]]), rho0, a, kw["al"], kw["b"], kw["g"])[0]
    return dict(rms_rho=np.sqrt(np.mean(rr ** 2)), rms_M=np.sqrt(np.mean(mm ** 2)),
                a=a, Mtot=Mtot, **kw)


# ---------------- targets, now with M(<r) -------------------------------
cfg = configparser.ConfigParser(); cfg.read(os.path.join(PF, "omegacen_pdmf.ini"))
secs = [s for s in cfg.sections() if s != "PhaseFlow"]
suf2type = {chr(ord('a') + i): s.split()[0] for i, s in enumerate(secs)}
prefix = os.path.join(PF, "output_pdmf", "omegacen_pdmf")
rs, rhos, Ms = [], [], []
for suf in [s for s, t in suf2type.items() if t == "BH"]:
    files = [f for f in glob.glob(prefix + "*" + suf) if not f.endswith(".log")]
    ts = [int(m.group(1)) for f in files
          if (m := re.match(re.escape(prefix) + r"(\d+)" + suf + r"$", f))]
    d = np.loadtxt(prefix + str(max(ts)) + suf, skiprows=2)
    rs.append(d[:, 0]); rhos.append(d[:, 3]); Ms.append(d[:, 1])
rg = np.geomspace(max(r.min() for r in rs), min(r.max() for r in rs), 400)
lg = lambda r, y: np.exp(np.interp(np.log(rg), np.log(r), np.log(np.maximum(y, 1e-300))))
pf_rho = np.sum([lg(r, y) for r, y in zip(rs, rhos)], axis=0)
pf_M = np.sum([lg(r, y) for r, y in zip(rs, Ms)], axis=0)

gc = np.load(os.path.join(HERE, "target_gcfit.npz"))
g_r, g_rho, g_M = gc["r"], gc["rho"][2], gc["cum"][2]

def cut(r, rho, M, lo, hi, n=150):
    m = (r >= lo) & (r <= hi) & (rho > 0) & (M > 0)
    g = np.geomspace(r[m][0], r[m][-1], n)
    ip = lambda y: np.exp(np.interp(np.log(g), np.log(r[m]), np.log(y[m])))
    return g, ip(rho), ip(M)

TARGETS = {
    "PhaseFlow BH, 1e-4-5 pc":    cut(rg, pf_rho, pf_M, 1e-4, 5.0),
    "LIMEPY, 0.05-10 pc":         cut(g_r, g_rho, g_M, 0.05, 10.0),
    "LIMEPY, full 1e-3-50 pc":    cut(g_r, g_rho, g_M, 1e-3, 50.0),
}
print("GCfit total M_BH within fit ranges: %.4g (10pc)  %.4g (50pc)"
      % (np.interp(10, g_r, g_M), np.interp(50, g_r, g_M)))
for tname, (r, rho, M) in TARGETS.items():
    print("\n=== %s   M(<r_max)=%.4g" % (tname, M[-1]))
    print("    %-28s %9s %9s %10s %12s  shape" % ("variant", "rho dex", "M dex", "a [pc]", "M_tot fit"))
    for vname, spec in VARIANTS.items():
        f = fit(r, rho, M, spec)
        print("    %-28s %9.3f %9.3f %10.4g %12.4g  g=%.2f al=%.2f b=%.2f"
              % (vname, f["rms_rho"], f["rms_M"], f["a"], f["Mtot"],
                 f["g"], f["al"], f["b"]))
