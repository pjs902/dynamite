#!/usr/bin/env python
"""How far DYNAMITE's orblib ``ml``-reuse trick departs from exactness.

An orbit library is reused across ``ml`` by scaling LOSVD velocities by
``sqrt(ml/ml_orblib)`` (orblib.py:2536).  That is exact only if EVERY mass in
the potential scales by ``ml``.  Stars do -- ``surf_km(:) = surf_km(:)*upsilon``
(iniparam_f.f90:197).  The central BH (``xmbh``) and the fitted
``StellarBlackHoles`` subcluster (``sbh_rho0``) do NOT: their masses are
written into parameters_pot.in raw and never multiplied by upsilon.

So the trick assumes

    M_tot(<r; ml) == (ml/ml_orb) * M_tot(<r; ml_orb)

and this script measures how badly that fails over the radial range the orbits
actually sample (``logrmin``..``logrmax``).  The answer decides whether you can
leave ``ml`` free while fitting the IMBH and sBH masses, or must fix it.

Defaults are the numbers in ``user_test_config_sbh.yaml``.  Override on the
command line for your own system.
"""

import argparse
import os
import sys

import numpy as np
from astropy.table import Table
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamite.physical_system import StellarBlackHoles  # noqa: E402

ARCSEC_RAD = np.pi / (180.0 * 3600.0)


def stellar_mass_enclosed(r_arcsec, mge_file, dist_mpc):
    """M(<r) of the potential MGE, per unit ml, in Msun.

    ponytail: Gaussians treated as spherical.  The diagnostic is a ratio of
    total mass profiles and the stellar term is the only one that scales with
    ml, so flattening shifts both numerator and denominator the same way.
    """
    t = Table.read(mge_file)
    surf, sigma, q = t["I"], t["sigma"], t["q"]
    pc_per_arcsec = dist_mpc * 1e6 * np.tan(ARCSEC_RAD)
    sigma_pc = np.asarray(sigma) * pc_per_arcsec
    m_tot = 2.0 * np.pi * np.asarray(surf) * sigma_pc**2 * np.asarray(q)
    x = r_arcsec[:, None] / np.asarray(sigma)[None, :]
    shape = special.erf(x / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * x * np.exp(
        -0.5 * x**2
    )
    return (m_tot[None, :] * shape).sum(axis=1)


def total_mass_enclosed(r, ml, m_star_unit_ml, m_bh, a_bh, sbh):
    """M(<r) in Msun: stars scale with ml, the two dark terms do not."""
    m = ml * m_star_unit_ml
    m = m + m_bh * r**3 / (r**2 + a_bh**2) ** 1.5  # Plummer point BH
    if sbh is not None:
        m_sbh, a, alpha, beta, gamma = sbh
        rho0 = StellarBlackHoles.rho0_from_mass(m_sbh, a, alpha, beta, gamma)
        m = m + StellarBlackHoles.mass_enclosed(
            r, 0.0, 0.0, (rho0, a, alpha, beta, gamma)
        )
    return m


def error_profile(r, ml, ml_orb, *args):
    """Relative error of the velocity-rescale assumption, per radius."""
    num = total_mass_enclosed(r, ml, *args)
    den = total_mass_enclosed(r, ml_orb, *args)
    return (num / den) / (ml / ml_orb) - 1.0


def worst_error(r, ml, ml_orb, *args):
    """Max |relative error| of the velocity-rescale assumption over r."""
    return np.max(np.abs(error_profile(r, ml, ml_orb, *args)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mge", default="NGC6278_input/mge_lum.ecsv",
                   help="potential MGE (the 'mge_pot' key's file)")
    p.add_argument("--distMPc", type=float, default=39.96)
    p.add_argument("--logrmin", type=float, default=-0.101275)
    p.add_argument("--logrmax", type=float, default=1.99123)
    p.add_argument("--ml-lo", type=float, default=4.0)
    p.add_argument("--ml-hi", type=float, default=6.0)
    p.add_argument("--m-bh", type=float, default=1e5, help="IMBH mass [Msun]")
    p.add_argument("--a-bh", type=float, default=1e-3, help="BH soft. [arcsec]")
    p.add_argument("--m-sbh", type=float, default=1e5, help="sBH mass [Msun]")
    p.add_argument("--a-sbh", type=float, default=352.0, help="[arcsec]")
    p.add_argument("--alpha", type=float, default=2.15)
    p.add_argument("--beta", type=float, default=4.5)
    p.add_argument("--gamma", type=float, default=1.75)
    args = p.parse_args()

    r = np.logspace(args.logrmin, args.logrmax, 400)
    m_star = stellar_mass_enclosed(r, args.mge, args.distMPc)
    sbh = (args.m_sbh, args.a_sbh, args.alpha, args.beta, args.gamma)
    pars = (m_star, args.m_bh, args.a_bh, sbh)

    print(f"radial range        : {r[0]:.3g} .. {r[-1]:.3g} arcsec")
    print(f"M_star(<rmax) * ml  : {args.ml_lo * m_star[-1]:.4g} Msun "
          f"(at ml={args.ml_lo:g})")
    print(f"M_IMBH              : {args.m_bh:.4g} Msun")
    print(f"M_sBH(<rmax)        : "
          f"{total_mass_enclosed(r, 0.0, np.zeros_like(r), 0.0, 1.0, sbh)[-1]:.4g}"
          " Msun")
    print()

    # self-check: with no non-scaling mass, the rescale is exact
    exact = worst_error(r, args.ml_hi, args.ml_lo, m_star, 0.0, 1.0, None)
    assert exact < 1e-12, f"stars-only case should be exact, got {exact:.3g}"
    print(f"self-check (stars only, no BH/sBH): {exact:.2e}  [must be ~0]")
    print()

    print("worst |error| in M(<r) from reusing an orblib built at ml_orb:")
    print(f"{'ml_orb':>8}{'ml':>8}{'max err':>12}{'M_dark factor':>16}")
    for ml_orb in (args.ml_lo, 0.5 * (args.ml_lo + args.ml_hi), args.ml_hi):
        for ml in (args.ml_lo, args.ml_hi):
            if ml == ml_orb:
                continue
            err = worst_error(r, ml, ml_orb, *pars)
            print(f"{ml_orb:8.2f}{ml:8.2f}{err:11.2%}{ml / ml_orb:16.3f}")
    ml_orb = 0.5 * (args.ml_lo + args.ml_hi)
    err = error_profile(r, args.ml_hi, ml_orb, *pars)
    print()
    print(f"error vs radius, orblib built at ml_orb={ml_orb:g}, reused at "
          f"ml={args.ml_hi:g}:")
    print(f"{'r [arcsec]':>12}{'M(<r) err':>12}")
    for i in np.linspace(0, len(r) - 1, 10).astype(int):
        print(f"{r[i]:12.3g}{err[i]:11.2%}")
    ok = np.where(np.abs(err) < 0.01)[0]
    print("\nerror falls below 1% beyond r = "
          f"{r[ok[0]]:.3g} arcsec" if len(ok) else
          "\nerror never falls below 1% over this range")
    print()
    print("'M_dark factor' = ml/ml_orb: the factor by which the IMBH and sBH")
    print("masses you fitted differ from the physical masses of that model.")
    print("ponytail: DM halo omitted -- its rc depends on totalmass^(1/3), so")
    print("it breaks self-similarity too; add it if your system has one.")


if __name__ == "__main__":
    main()
