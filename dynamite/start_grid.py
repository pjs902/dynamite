"""Orbit start grids placed in action space (AGAMA), as an alternative to the Fortran ``orbitstart``.

Enabled by ``initial_conditions: actions`` in ``orblib_settings``; without it (or with ``default``) DYNAMITE runs
``orbitstart`` exactly as before. Same (nE, nI2, nI3, dithering, logrmin, logrmax) as ``orbitstart``; what changes:
  tubes  every ray (E_i, theta_j) starts at its thin tube, t_in = argmin_t J_r along the ray (orbitstart starts inside
         it at "irregular" energies and so samples those orbits twice), and its nI3*dithering rows are spread evenly
         in J_r/J_tot from there to the zero-velocity curve (orbitstart spreads them evenly in radius, which leaves
         most of them near-circular). Rays are found in the axisymmetrised potential (what the Staeckel action finder
         needs) and stretched onto the triaxial zero-velocity curve; v_y is set from E_i in the triaxial potential.
  boxes  at rest on the triaxial equipotential on orbitstart's open (theta, phi) grid, as orbitstart.
  E_i    Phi_triaxial(r_c,i, 0, 0) on orbitstart's log-uniform r_c grid; tcirc_i from the thin tube of the last theta
         row (sets the integration time only).
The files are written in orbitstart's format; the Fortran integrator reads rows by position only. The last column
(do-not-regularise flag, read only by the legacy Fortran regularisation) is 0 for every row; orbitstart sets it for
some edge rows of triaxial models. A sidecar ``datfil/start_grid.json`` holds a fingerprint of the grid, checked by
``LegacyOrbitLibrary.ics_match_settings`` so that switching ``initial_conditions`` regenerates the files.

Needs ``agama`` (optional dependency, imported only when enabled). Not available for bar/disk systems.
The potential is rebuilt from the model's own ``infil/parameters_pot.in`` (stellar MGE, Plummer BH, NFW halo,
StellarBlackHoles type 6) and its stellar part is checked against DYNAMITE's Python twin of the Fortran force before
any row is written. ``mass_radmass`` (output only) keeps its nE uniform shells; the fitted intrinsic-mass constraints
(qgrid) do not depend on the start grid.
"""
import hashlib
import json
import logging
import os

import numpy as np

from dynamite import constants as const

VERSION = 1  # bump when the generated rows change for the same settings
SIDECAR = "start_grid.json"
FMT = "%5d%5d%5d" + "%30.10E" * 9 + "%4d"
KM = const.PARSEC_KM  # km per pc; also s per [pc/(km/s)]
G_PC = const.GRAV_CONST_KM / const.PARSEC_KM


OPTIONS = ("default", "actions")


def enabled(settings):
    """True for ``initial_conditions: actions``; missing or ``default`` means orbitstart."""
    return settings.get("initial_conditions", "default") == "actions"


def validate(settings):
    """Check ``settings['initial_conditions']`` and that agama is installed for ``actions``."""
    on = settings.get("initial_conditions", "default")
    if on not in OPTIONS:
        raise ValueError(f"orblib_settings: initial_conditions must be one of {OPTIONS}, not {on!r}")
    if on == "actions":
        try:
            import agama  # noqa: F401
        except ImportError as e:
            raise ImportError("orblib_settings: initial_conditions: actions needs the agama package "
                              "(pip install agama)") from e


def fingerprint(settings):
    """Identifies the start grid the settings ask for (None: orbitstart)."""
    if not enabled(settings):
        return None
    key = dict(version=VERSION, grid=[settings[k] for k in ("nE", "nI2", "nI3", "dithering", "logrmin", "logrmax")])
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def read_fingerprint(datfil):
    try:
        with open(os.path.join(datfil, SIDECAR)) as f:
            return json.load(f)["fingerprint"]
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- potential (from the model's parameters_pot.in)

def read_parameters_pot(path):
    """parameters_pot.in in the order iniparam_f.f90 reads it (lengths in arcsec, masses in Msun)."""
    with open(path) as f:
        lines = [ln.split() for ln in f.read().splitlines() if ln.strip()]
    n = int(lines[0][0])
    mge = np.array(lines[1:1 + n], float)
    it = iter(lines[1 + n:])
    p = dict(I=mge[:, 0], sigma=mge[:, 1], q=mge[:, 2], pa_twist=mge[:, 3])
    p["dist_mpc"] = float(next(it)[0])
    p["angles"] = tuple(float(v) for v in next(it)[:3])
    p["ml"] = float(next(it)[0])
    p["mbh"] = float(next(it)[0])
    p["soft_arcsec"] = float(next(it)[0])
    next(it)  # nE rlogmin rlogmax
    for _ in range(6):  # nI2 nI3 dithering quad_nr quad_nth quad_nph
        next(it)
    dm_type, ndm = (int(v) for v in next(it)[:2])
    p["dm_type"], p["dm_par"] = dm_type, [float(v) for v in next(it)[:ndm]]
    p["H"] = float(next(it)[0])
    rest = list(it)
    p["sbh"] = None
    if rest and int(rest[0][0]) == 6:
        p["sbh"] = dict(zip(("rho0_km", "a_km", "alpha", "beta", "gamma"), (float(v) for v in rest[1][:5])))
    elif rest and int(rest[0][0]) != 0:
        raise NotImplementedError(f"initial_conditions actions: sBH block type {rest[0][0]} is not supported")
    if dm_type not in (0, 1):
        raise NotImplementedError(f"initial_conditions actions: dm_profile_type {dm_type} is not supported (only none and NFW)")
    p["arcsec_pc"] = p["dist_mpc"] * 1e6 * np.tan(np.pi / 648e3)
    return p


def deproject(p):
    """Intrinsic Gaussians [pc, Msun] at the file's (theta, phi, psi), triaxpotent.f90:106-166."""
    sig_pc = p["sigma"] * p["arcsec_pc"]
    qo = p["q"]
    mass = 2 * np.pi * p["I"] * qo * sig_pc ** 2 * p["ml"]
    th, ph = np.radians(p["angles"][0]), np.radians(p["angles"][1])
    ps = np.radians(p["angles"][2] + p["pa_twist"])  # iniparam_f.f90: psi_obs += psi_view
    secth, cotph = 1 / np.cos(th), 1 / np.tan(ph)
    delp = 1 - qo ** 2
    nom1 = delp * (2 * np.cos(2 * ps) + np.sin(2 * ps) * (secth * cotph - np.cos(th) * np.tan(ph)))
    nomp = delp * (2 * np.cos(2 * ps) + np.sin(2 * ps) * (np.cos(th) * cotph - secth * np.tan(ph)))
    den = 2 * np.sin(th) ** 2 * (delp * np.cos(ps) * (np.cos(ps) + secth * cotph * np.sin(ps)) - 1)
    q2 = 1 - nom1 / den
    p2 = q2 + nomp / den
    if np.any(q2 < 0) or np.any(p2 < 0):
        raise ValueError("initial_conditions actions: p^2 or q^2 < 0, no deprojection for these angles")
    qi, pi_ = np.sqrt(q2), np.sqrt(p2)
    sig_in = sig_pc * np.sqrt(qo / np.sqrt((pi_ * np.cos(th)) ** 2 + (qi * np.sin(th)) ** 2
                                           * ((pi_ * np.cos(ph)) ** 2 + np.sin(ph) ** 2)))
    return mass, sig_in, pi_, qi


def potentials(p):
    """(triaxial, axisymmetrised, stars-only triaxial) AGAMA potentials in pc, km/s, G = 1."""
    import agama
    mass, s, pp, qq = deproject(p)
    dens = agama.Density(*[agama.Density(type="Spheroid", densityNorm=G_PC * m / ((2 * np.pi) ** 1.5 * a * b * c ** 3),
                                         alpha=1.0, beta=0.0, gamma=0.0, scaleRadius=c, outerCutoffRadius=np.sqrt(2) * c,
                                         cutoffStrength=2.0, axisRatioY=a, axisRatioZ=b)
                           for m, c, a, b in zip(mass, s, pp, qq)])
    extra = []
    if p["dm_type"] == 1:
        c, f = p["dm_par"]
        rho_crit = 3 * p["H"] ** 2 / (8 * np.pi * G_PC)
        rhoc = 200 / 3 * rho_crit * c ** 3 / (np.log(1 + c) - c / (1 + c))
        rc = (3 / (800 * np.pi * rho_crit * c ** 3) * f * mass.sum()) ** (1 / 3)
        extra.append(agama.Potential(type="NFW", mass=G_PC * 4 * np.pi * rhoc * rc ** 3, scaleRadius=rc))
    if p["mbh"] > 0:
        extra.append(agama.Potential(type="Plummer", mass=G_PC * p["mbh"], scaleRadius=p["soft_arcsec"] * p["arcsec_pc"]))
    if p["sbh"]:
        s_ = p["sbh"]
        extra.append(agama.Potential(type="Spheroid", densityNorm=G_PC * s_["rho0_km"] * KM ** 3, scaleRadius=s_["a_km"] / KM,
                                     alpha=s_["alpha"], beta=s_["beta"], gamma=s_["gamma"]))
    mp = lambda ax: agama.Potential(type="Multipole", density=dens, lmax=32, gridSizeR=60, mmax=0 if ax else 6,  # noqa: E731
                                    symmetry="axisymmetric" if ax else "triaxial")
    stars = mp(False)
    return agama.Potential(stars, *extra), agama.Potential(mp(True), *extra), stars


def check_stars(p, stars, r_pc, tol=1e-3):
    """AGAMA's stellar force vs DYNAMITE's Python twin of the Fortran one at random points over the energy range."""
    from dynamite.physical_system import TriaxialVisibleComponent as TVC
    rng = np.random.default_rng(1)
    u = rng.normal(size=(24, 3))
    xyz = u / np.linalg.norm(u, axis=1)[:, None] * np.exp(rng.uniform(np.log(r_pc[0]), np.log(r_pc[-1]), 24))[:, None]
    a_d = np.stack(TVC.acceleration(*xyz.T, p["angles"], p["ml"], p["I"], p["sigma"] * p["arcsec_pc"], p["q"],
                                    psi_off=p["pa_twist"], epsrel=1e-7), 1)
    rel = np.linalg.norm(stars.force(xyz) - a_d, axis=1) / np.linalg.norm(a_d, axis=1)
    if rel.max() > tol:
        raise RuntimeError(f"initial_conditions actions: AGAMA stellar force differs from DYNAMITE's by {rel.max():.1e} (> {tol})")
    return rel.max()


# ---------------------------------------------------------------- rows

def tube_xv(pax, E, th, t):
    pos = np.column_stack([t * np.sin(th), np.zeros_like(t), t * np.cos(th)])
    vy = np.sqrt(np.clip(2 * (E - pax.potential(pos)), 0, None))
    z = np.zeros_like(t)
    return np.column_stack([pos, z, vy, z])


def equipot_radius(pot, E, th, ph):
    """r n with Phi(r n) = E along direction (th, ph), log bisection over a wide bracket."""
    n = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    lo, hi = np.full(len(E), 1e-6), np.full(len(E), 1e7)
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        up = pot.potential(n * mid[:, None]) > E
        lo, hi = np.where(up, lo, mid), np.where(up, mid, hi)
    return np.sqrt(lo * hi)[:, None] * n


def find_req(pot, E, th, ph, rc, n=48):
    """r n with Phi(r n) = E by bisection in [0.01, 1.1] rc (orbitstart findReq)."""
    n3 = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    lo, hi = 0.01 * rc, 1.1 * rc
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        up = pot.potential(n3 * mid[:, None]) > E
        lo, hi = np.where(up, lo, mid), np.where(up, mid, hi)
    return 0.5 * (lo + hi)[:, None] * n3


def ray_bounds(pax, af, E, th, nscan=64):
    """t_in = argmin J_r along the ray (thin tube, parabola-refined), t_out = zero-velocity curve."""
    tout = np.linalg.norm(equipot_radius(pax, E, th, np.zeros_like(E)), axis=1) * (1 - 1e-6)
    s = np.linspace(0.02, 0.9999, nscan)
    t = tout[:, None] * s[None, :]
    J = af(tube_xv(pax, np.repeat(E, nscan), np.repeat(th, nscan), t.ravel())).reshape(len(E), nscan, 3)
    k = np.clip(np.argmin(J[..., 0], axis=1), 1, nscan - 2)
    f0, f1, f2 = (np.take_along_axis(J[..., 0], (k + o)[:, None], 1)[:, 0] for o in (-1, 0, 1))
    den = f0 - 2 * f1 + f2
    off = np.where(den > 0, 0.5 * (f0 - f2) / np.where(den > 0, den, 1), 0)
    return tout * (s[k] + np.clip(off, -1, 1) * (s[1] - s[0])), tout


def tube_rows(pax, af, E, th, tin, tout, n, nscan=96):
    """t of the n rows per ray between tin and tout, evenly in J_r/J_tot."""
    sgrid = 1 - (1 - np.linspace(0, 1, nscan)) ** 3  # dense near the outer edge, where J_r/J_tot climbs
    t = tin[:, None] + (tout - tin)[:, None] * sgrid[None, :]
    J = af(tube_xv(pax, np.repeat(E, nscan), np.repeat(th, nscan), t.ravel()))
    y = np.maximum.accumulate((J[:, 0] / np.abs(J).sum(1)).reshape(-1, nscan), axis=1)
    target = (np.arange(1, n + 1) - 0.5) / n
    rows = np.empty((len(E), n))
    for r in range(len(E)):
        yt = y[r, 0] + (y[r, -1] - y[r, 0]) * target
        rows[r] = np.interp(yt, y[r] + 1e-12 * np.arange(nscan), t[r])
    return rows


def make_rows(settings, p):
    """(tube rows, box rows, dithered shape) in orbitstart's 13-column file format."""
    import agama
    d = settings["dithering"]
    nEf, n2f, n3f = (settings[k] * d for k in ("nE", "nI2", "nI3"))
    r_as = 10 ** np.linspace(settings["logrmin"], settings["logrmax"], nEf)  # orbitstart_f.f90:67-72
    rc = r_as * p["arcsec_pc"]
    pot, pax, stars = potentials(p)
    err = check_stars(p, stars, rc)
    af = agama.ActionFinder(pax, interp=False)
    z = np.zeros_like(rc)
    Ea = pax.potential(np.column_stack([rc, z, z]))
    E = pot.potential(np.column_stack([rc, z, z]))
    thf = np.pi / 2 * (np.arange(1, n2f + 1) - 0.5) / n2f
    E2, T2 = (a.ravel() for a in np.meshgrid(Ea, thf, indexing="ij"))
    tin, tout = ray_bounds(pax, af, E2, T2)
    t = tube_rows(pax, af, E2, T2, tin, tout, n3f)
    xv = tube_xv(pax, np.repeat(E2, n3f), np.repeat(T2, n3f), t.ravel())
    # stretch each ray so its zero-velocity point moves from the axisymmetric to the triaxial equipotential
    rho = (np.linalg.norm(find_req(pot, np.repeat(E, n2f), T2, 0 * T2, np.repeat(rc, n2f)), axis=1)
           / np.linalg.norm(equipot_radius(pax, E2, T2, 0 * T2), axis=1))
    pos = (xv[:, :3].reshape(nEf * n2f, n3f, 3) * rho[:, None, None]).reshape(-1, 3)
    cols = lambda a: np.repeat(a, n2f * n3f)  # noqa: E731
    Ei = cols(E)
    vy2 = 2 * (Ei - pot.potential(pos))
    if not (vy2 > 0).all():
        raise RuntimeError(f"initial_conditions actions: {(vy2 <= 0).sum()} tube starts at/outside the triaxial zero-velocity curve")
    tin_last = tin.reshape(nEf, n2f)[:, -1]
    vc = np.sqrt(tin_last * np.abs(pot.force(np.column_stack([tin_last, z, z]))[:, 0]))
    tc = 2 * np.pi * tin_last / vc
    ijk = np.stack(np.meshgrid(*(np.arange(1, n + 1) for n in (nEf, n2f, n3f)), indexing="ij"), -1).reshape(-1, 3)
    head = np.column_stack([cols(rc * KM), cols(tc * KM), cols(vc)])
    zz = np.zeros(len(pos))
    tube = np.column_stack([ijk, pos * KM, zz, np.sqrt(vy2), zz, head, zz])
    th = np.pi / 2 * (ijk[:, 1] - 0.5) / n2f
    ph = np.pi / 2 * (ijk[:, 2] - 0.5) / n3f
    bpos = find_req(pot, Ei, th, ph, cols(rc))
    if not np.allclose(pot.potential(bpos), Ei, rtol=1e-9):
        raise RuntimeError("initial_conditions actions: box starts are not on the equipotential")
    box = np.column_stack([ijk, bpos * KM, np.zeros((len(bpos), 3)), head, zz])
    # fetched here: a logger made at import time is disabled by Configuration(reset_logging=True)
    logging.getLogger(__name__).info(f"initial_conditions actions: {nEf}x{n2f}x{n3f} dithered starts per family, "
                f"r_c {r_as[0]:.4g}-{r_as[-1]:.4g} arcsec; AGAMA vs DYNAMITE stellar force {err:.1e}; ray stretch "
                f"{rho.min():.5f}-{rho.max():.5f}")
    return tube, box, (nEf, n2f, n3f)


def write_start_files(settings, mod_dir):
    """Write datfil/begin.dat, datfil/beginbox.dat and the sidecar for the model in ``mod_dir``."""
    p = read_parameters_pot(os.path.join(mod_dir, "infil/parameters_pot.in"))
    tube, box, shape = make_rows(settings, p)
    datfil = os.path.join(mod_dir, "datfil")
    head = "".join(f"{n:12d}" for n in shape)
    for fn, rows in (("begin.dat", tube), ("beginbox.dat", box)):
        np.savetxt(os.path.join(datfil, fn + ".tmp"), rows, fmt=FMT, header=head, comments="")
        os.replace(os.path.join(datfil, fn + ".tmp"), os.path.join(datfil, fn))
    with open(os.path.join(datfil, SIDECAR), "w") as f:  # last: a crash before this leaves files that do not match
        json.dump(dict(fingerprint=fingerprint(settings), version=VERSION), f, indent=1)
