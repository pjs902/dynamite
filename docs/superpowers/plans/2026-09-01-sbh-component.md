# sBH Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stellar-mass black hole (sBH) mass component to DYNAMITE — a spherical Zhao/alpha-beta-gamma profile that can coexist with a DM halo — plus an MGE variant for a fixed externally-supplied profile.

**Architecture:** A new `DarkComponent` subclass (`legacy_code = 6`) with closed-form density, enclosed mass and acceleration, mirroring the existing `Hernquist`/`GeneralisedNFW` pattern. Python converts the sampled total mass to a scale density and writes it into a *second* dark-matter block appended at the end of `parameters_pot.in`; `iniparam_f.f90` reads that block optionally under `iostat`, and `dmpotent.f90` adds the sBH contribution in a dedicated slot that never touches the existing halo cases. A second class, `StellarBlackHolesMGE`, needs no Fortran at all — its Gaussians concatenate into the potential MGE.

**Tech Stack:** Python 3 (numpy, scipy.special), Fortran 90 (gfortran/ifort via `legacy_fortran/Makefile`), standalone `dev_tests` scripts.

**Spec:** `docs/superpowers/specs/2026-09-01-sbh-component-design.md`

## Global Constraints

- Branch: `sBH`. Push target is `origin` (the pjs902 fork) only — **never** upstream.
- Python env: `/opt/miniconda3/envs/main/bin/python`. The base miniconda on `PATH` is not the right one.
- `legacy_code = 6`. Codes 0,1,2,3,5 are taken; 4 is commented out and must stay unused.
- Parameter constraints: `m > 0`, `a > 0`, `alpha > 0`, `beta > 3`, `gamma < 3`, and `gamma != 2` (exactly 2 gives a division by zero in the beta recurrence).
- Units at the Python/Fortran boundary follow the `Hernquist` precedent: the legacy file carries **`rhoc` in Msun/km^3** and **`a` in km**. The config carries `m` in Msun and `a` in **arcsec** (matching the `Plummer` black hole component). Python does every conversion.
- Exactly two dark slots: slot 1 is the halo (or `0 0`), slot 2 is always the sBH. Existing cases 1, 2, 3, 5 must not be modified.
- `zh_betai` must **never** be called with a second argument <= 0. It returns `inf` there (measured — see `dev_notes/sbh_profile_fits/try_zh_betai.py`).
- Do not "fix" the latent `dm_potent` case-5 gNFW divergence. Out of scope, deliberately.

**Note on an existing bug:** `GeneralisedNFW.acceleration` (`physical_system.py:1448`) calls `GeneralisedNFW.convert_parset`, which is **not defined anywhere in the codebase**. That method is dead code and would raise `AttributeError` if called. Mirror `GeneralisedNFW`'s *structure*, not this call. Do not fix it here — out of scope.

**Correction to the spec:** the spec's Potential section argues that the divergent additive constant must be dropped. That argument is sound but unnecessary in practice: `B(y;p,q) = Int_0^y u^(p-1) (1-u)^(q-1) du` is finite for `y < 1` even when `q < 0`, because the singularity sits at `u = 1`. The recurrence therefore returns the exact tail directly, with no constant games. `try_recurrence.py` verified exactly this.

---

### Task 1: Zhao profile maths in Python

**Files:**
- Modify: `dynamite/physical_system.py` (add a new class after `GeneralisedNFW`, before `class Chi2Ext`)
- Test: `dev_tests/test_sbh_profile.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `StellarBlackHoles.density(x, y, z, pars)` -> ndarray, `pars = (rho0, a, alpha, beta, gamma)`
  - `StellarBlackHoles.mass_enclosed(x, y, z, pars)` -> ndarray, same `pars`
  - `StellarBlackHoles.rho0_from_mass(m, a, alpha, beta, gamma)` -> float
  - `StellarBlackHoles.incomplete_beta(x, p, q)` -> float (handles `q <= 0`)

- [ ] **Step 1: Write the failing test**

Create `dev_tests/test_sbh_profile.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone checks for the sBH (Zhao alpha-beta-gamma) component.

Run directly:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py
Exits nonzero on the first failure.
"""
import sys
import numpy as np
from scipy.integrate import quad

import dynamite as dyn
from dynamite.physical_system import StellarBlackHoles as SBH

# (alpha, beta, gamma) spanning both external models, incl. gamma > 2
CASES = [
    (2.15, 12.0, 1.74),   # PhaseFlow fiducial fit
    (3.91, 4.50, 2.24),   # LIMEPY 0.05-10 pc fit  (gamma > 2)
    (2.00, 4.00, 2.18),   # gamma > 2
    (0.31, 12.0, 0.51),   # LIMEPY full-range fit
    (1.00, 3.50, 1.75),   # Bahcall-Wolf-like
    (4.00, 5.00, 2.50),   # gamma > 2, steep
]


def _rho_scalar(r, rho0, a, al, b, g):
    x = r / a
    return rho0 * x ** -g * (1.0 + x ** al) ** (-(b - g) / al)


def test_density_matches_formula():
    """density() must equal the Zhao formula on a spherical shell."""
    rho0, a = 1.0e5, 1.5
    for al, b, g in CASES:
        for r in np.geomspace(1e-4, 1e3, 12):
            got = SBH.density(r, 0.0, 0.0, (rho0, a, al, b, g))
            want = _rho_scalar(r, rho0, a, al, b, g)
            rel = abs(float(got) / want - 1.0)
            assert rel < 1e-12, \
                f'density (al,b,g)=({al},{b},{g}) r={r}: rel err {rel:.2e}'
    print('  density_matches_formula OK')


def test_mass_enclosed_matches_quadrature():
    """mass_enclosed() must equal a direct integral of 4 pi r^2 rho."""
    rho0, a = 1.0e5, 1.5
    for al, b, g in CASES:
        for r in np.geomspace(1e-3, 1e2, 8):
            got = float(SBH.mass_enclosed(r, 0.0, 0.0, (rho0, a, al, b, g)))
            # integrate in log r: the linear-space integrand is near-singular
            f = lambda u: (4 * np.pi * np.exp(u) ** 3
                           * _rho_scalar(np.exp(u), rho0, a, al, b, g))
            want, _ = quad(f, np.log(r) - 60, np.log(r), limit=800)
            rel = abs(got / want - 1.0)
            assert rel < 1e-8, \
                f'M(<r) (al,b,g)=({al},{b},{g}) r={r}: rel err {rel:.2e}'
    print('  mass_enclosed_matches_quadrature OK')


def test_rho0_from_mass_round_trips():
    """rho0_from_mass must make M(<inf) equal the requested total mass."""
    m_target, a = 1.79e5, 3.06
    for al, b, g in CASES:
        rho0 = SBH.rho0_from_mass(m_target, a, al, b, g)
        # M(<r) approaches M_tot as (1-t)**q with q = (beta-3)/alpha, which
        # is only 0.5 for e.g. (1.0, 3.5, 1.75) -- so even at r = a*1e12 the
        # truncated integral is ~1e-6 short. The point of this test is that
        # the normalisation constant is right, not that a truncated integral
        # converges; Task 7's Fortran agreement test is the tighter net.
        m_inf = float(SBH.mass_enclosed(a * 1e12, 0.0, 0.0,
                                        (rho0, a, al, b, g)))
        rel = abs(m_inf / m_target - 1.0)
        assert rel < 1e-3, \
            f'rho0 round trip (al,b,g)=({al},{b},{g}): rel err {rel:.2e}'
    print('  rho0_from_mass_round_trips OK')


def test_incomplete_beta_negative_q():
    """The q<=0 branch must match quadrature (this is where zh_betai fails)."""
    for p, q in [(0.64, -0.061), (1.0, -0.09), (0.31, -0.45), (0.75, -0.125)]:
        for x in [1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999]:
            got = SBH.incomplete_beta(x, p, q)
            want, _ = quad(lambda u: u ** (p - 1) * (1 - u) ** (q - 1),
                           0.0, x, limit=400)
            rel = abs(got / want - 1.0)
            assert rel < 1e-9, \
                f'incomplete_beta p={p} q={q} x={x}: rel err {rel:.2e}'
    print('  incomplete_beta_negative_q OK')


TESTS = [
    test_density_matches_formula,
    test_mass_enclosed_matches_quadrature,
    test_rho0_from_mass_round_trips,
    test_incomplete_beta_negative_q,
]

if __name__ == '__main__':
    print(f'dynamite {dyn.__version__ if hasattr(dyn, "__version__") else ""}')
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py`
Expected: FAIL with `ImportError: cannot import name 'StellarBlackHoles'`

- [ ] **Step 3: Write minimal implementation**

In `dynamite/physical_system.py`, insert immediately **after** the `GeneralisedNFW` class and **before** `class Chi2Ext`. Note `from scipy import special` is already imported at module top (used by `GeneralisedNFW`); confirm before adding an import.

```python
class StellarBlackHoles(DarkComponent):
    """A subcluster of stellar-mass black holes

    Spherical Zhao (1996) alpha-beta-gamma double power law::

        rho(r) = rho0 * (r/a)**-gamma * (1 + (r/a)**alpha)**(-(beta-gamma)/alpha)

    Config parameters: m [total sBH mass, Msun], a [scale radius, arcsec],
    alpha [transition sharpness], beta [outer log-slope], gamma [inner
    log-slope].

    The profile family was chosen by fitting both the PhaseFlow relaxed cusp
    and the GCfit/LIMEPY posterior jointly on rho(r) and M(<r); see
    ``dev_notes/sbh_profile_fits/`` and the design spec.

    Note ``beta > 3`` is required for the total mass to converge and
    ``gamma < 3`` for M(<r) to converge at the origin. ``gamma == 2``
    exactly is excluded because it makes the beta-function recurrence
    divide by zero; the physical content there is a logarithmic limit and
    the parameter is continuous.
    """
    # legacy sequence: rhoc replaces m, and a is in km not arcsec
    par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']
    # config/sampled parameter names
    par = ['m', 'a', 'alpha', 'beta', 'gamma']

    def __init__(self, **kwds):
        self.legacy_code = 6
        super().__init__(symmetry='spherical', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')

    def validate(self):
        super().validate(par=self.par)

    def validate_parset(self, par):
        """
        Validate the sBH parameter values.

        Parameters
        ----------
        par : dict
            { "p":val, ... } where "p" are the component's parameters and
            val are their respective values

        Returns
        -------
        bool
            True if the parameter set is valid, False otherwise

        """
        ok = (par['m'] > 0.
              and par['a'] > 0.
              and par['alpha'] > 0.
              and par['beta'] > 3.
              and par['gamma'] < 3.
              and abs(par['gamma'] - 2.) > 1e-6)
        if not ok:
            self.logger.debug(f'Invalid sBH parset {dict(par)}: needs m>0, '
                              'a>0, alpha>0, beta>3, gamma<3, gamma!=2.')
        return bool(ok)

    @staticmethod
    def incomplete_beta(x, p, q):
        """Unregularised incomplete beta ``B(x; p, q)``, valid for q <= 0.

        ``B(x;p,q) = int_0^x u**(p-1) * (1-u)**(q-1) du``.

        For ``q > 0`` this is ``betainc(p,q,x) * beta(p,q)``. For ``q <= 0``
        the complete beta is undefined, so we step down from a positive-q
        evaluation using the contiguous relation::

            B(x;p,q) = [ (p+q) * B(x;p,q+1) - x**p * (1-x)**q ] / q

        This is the same recurrence the Fortran uses, and is why
        ``zh_betai`` is never called with a non-positive second argument.

        Parameters
        ----------
        x : float
            upper limit, 0 < x < 1
        p : float
            first parameter, must be > 0
        q : float
            second parameter, may be <= 0 but must not be 0

        Returns
        -------
        float
            the incomplete beta function value

        """
        if q > 0.:
            return special.betainc(p, q, x) * special.beta(p, q)
        n = int(np.ceil(1. - q)) + 1
        val = special.betainc(p, q + n, x) * special.beta(p, q + n)
        for j in range(n, 0, -1):
            qq = q + j - 1.
            val = ((p + qq) * val - x ** p * (1. - x) ** qq) / qq
        return val

    @staticmethod
    def density(x, y, z, pars):
        '''
        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates, same length units as ``a``
        pars : tuple
            (rho0, a, alpha, beta, gamma)

        Returns
        -------
        rho : float or ndarray
            density, in mass units of rho0
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r / a
        return rho0 * xx**(-gamma) * (1. + xx**alpha)**(-(beta-gamma)/alpha)

    @staticmethod
    def mass_enclosed(x, y, z, pars):
        '''
        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates, same length units as ``a``
        pars : tuple
            (rho0, a, alpha, beta, gamma)

        Returns
        -------
        Menc : float or ndarray
            mass within r, = 4 pi a^3 rho0 / alpha * B(t; (3-g)/al, (b-3)/al)
            with t = (r/a)^alpha / (1 + (r/a)^alpha)
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r / a
        t = xx**alpha / (1. + xx**alpha)
        p = (3. - gamma) / alpha
        q = (beta - 3.) / alpha
        # both p and q are > 0 given gamma < 3 and beta > 3
        bi = special.betainc(p, q, t) * special.beta(p, q)
        return 4. * np.pi * a**3 * rho0 / alpha * bi

    @staticmethod
    def rho0_from_mass(m, a, alpha, beta, gamma):
        """Scale density giving a total mass ``m``.

        ``M_tot = 4 pi a^3 rho0 / alpha * B((3-gamma)/alpha, (beta-3)/alpha)``
        using the *complete* beta, which converges only for beta > 3.

        Parameters
        ----------
        m : float
            total sBH mass
        a : float
            scale radius
        alpha, beta, gamma : float
            shape exponents

        Returns
        -------
        float
            rho0, in mass units of m over length units of a cubed

        """
        b_complete = special.beta((3. - gamma) / alpha, (beta - 3.) / alpha)
        return m * alpha / (4. * np.pi * a**3 * b_complete)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py`
Expected: `4/4 passed`, exit 0

- [ ] **Step 5: Commit**

```bash
git add dynamite/physical_system.py dev_tests/test_sbh_profile.py
git commit -m "feat(sbh): Zhao alpha-beta-gamma profile maths for the sBH component

Closed-form density, enclosed mass and total-mass normalisation, plus an
incomplete beta valid for q<=0 via downward recurrence (needed because
gamma>=2 is physically reachable and the standard betainc is undefined
there).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 2: Acceleration, potential and the legacy strings

**Files:**
- Modify: `dynamite/physical_system.py` (add methods to `StellarBlackHoles`)
- Modify: `dev_tests/test_sbh_profile.py` (add tests to `TESTS`)

**Interfaces:**
- Consumes: `StellarBlackHoles.density`, `.mass_enclosed`, `.rho0_from_mass`, `.incomplete_beta` (Task 1).
- Produces:
  - `StellarBlackHoles.acceleration(x, y, z, par)` -> `(ax, ay, az)`, `par` a dict-like with keys `m`, `a_km`, `alpha`, `beta`, `gamma`
  - `StellarBlackHoles.potential(x, y, z, pars)` -> ndarray, `pars = (rho0, a, alpha, beta, gamma)`
  - `StellarBlackHoles.get_dh_legacy_strings(parset, system)` -> `(specs, par_vals)`

- [ ] **Step 1: Write the failing test**

Add to `dev_tests/test_sbh_profile.py`, above the `TESTS` list:

```python
def test_acceleration_equals_minus_grad_potential():
    """a_r must equal -dPhi/dr, which ties potential() and acceleration()."""
    rho0, a = 1.0e5, 1.5
    G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
    for al, b, g in CASES:
        pars = (rho0, a, al, b, g)
        for r in np.geomspace(1e-2, 1e2, 8):
            h = r * 1e-6
            phi_p = float(SBH.potential(r + h, 0.0, 0.0, pars))
            phi_m = float(SBH.potential(r - h, 0.0, 0.0, pars))
            num = -(phi_p - phi_m) / (2 * h)
            ana = -G * float(SBH.mass_enclosed(r, 0.0, 0.0, pars)) / r ** 2
            rel = abs(ana / num - 1.0)
            assert rel < 1e-5, \
                f'accel vs -dPhi/dr (al,b,g)=({al},{b},{g}) r={r}: {rel:.2e}'
    print('  acceleration_equals_minus_grad_potential OK')


def test_acceleration_points_inward():
    """Gravity must attract: the radial acceleration is negative."""
    par = {'m': 1.79e5, 'a_pc': 3.06, 'alpha': 3.91,
           'beta': 4.50, 'gamma': 2.24}
    for r in np.geomspace(1e-2, 1e2, 10):
        ax, ay, az = SBH.acceleration(r, 0.0, 0.0, par)
        assert float(ax) < 0.0, f'acceleration not inward at r={r}: {ax}'
    print('  acceleration_points_inward OK')
```

Register both in `TESTS`:

```python
TESTS = [
    test_density_matches_formula,
    test_mass_enclosed_matches_quadrature,
    test_rho0_from_mass_round_trips,
    test_incomplete_beta_negative_q,
    test_acceleration_equals_minus_grad_potential,
    test_acceleration_points_inward,
]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py`
Expected: FAIL with `AttributeError: type object 'StellarBlackHoles' has no attribute 'potential'`

- [ ] **Step 3: Write minimal implementation**

Append these methods to `StellarBlackHoles`:

```python
    @staticmethod
    def potential(x, y, z, pars):
        '''
        Gravitational potential Phi (negative, and -> 0 at large r for
        gamma < 2; for gamma >= 2 it diverges as r -> 0, which is physical).

        Phi(r) = -G [ M(<r)/r + 4 pi int_r^inf r' rho dr' ]

        The outer term is
        ``(4 pi a^2 rho0 / alpha) * B(1-t; (beta-2)/alpha, (2-gamma)/alpha)``,
        whose second beta parameter is <= 0 when gamma >= 2 -- hence
        ``incomplete_beta`` rather than ``scipy.special.betainc``.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc]
        pars : tuple
            (rho0, a, alpha, beta, gamma), rho0 in Msun/pc**3, a in pc

        Returns
        -------
        Phi : ndarray
            potential [(km/s)**2]
        '''
        rho0, a, alpha, beta, gamma = pars
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)
        xx = r / a
        t = xx**alpha / (1. + xx**alpha)
        p_out = (beta - 2.) / alpha
        q_out = (2. - gamma) / alpha
        tail = np.vectorize(StellarBlackHoles.incomplete_beta)(
            1. - t, p_out, q_out)
        tail = 4. * np.pi * a**2 * rho0 / alpha * tail
        m_enc = StellarBlackHoles.mass_enclosed(x, y, z, pars)
        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        return -G * (m_enc / r + tail)

    @staticmethod
    def acceleration(x, y, z, par):
        """
        Gravitational acceleration of the sBH subcluster.

        Exact for all gamma < 3: ``a_r = -G M(<r) / r**2``.

        Parameters
        ----------
        x, y, z : float or array-like
            Cartesian coordinates [pc]
        par : dict
            must contain m [Msun], a_pc [pc], alpha, beta, gamma

        Returns
        -------
        ax, ay, az : ndarray
            Acceleration components [(km/s)**2/pc]
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        r = np.sqrt(x**2 + y**2 + z**2)

        # 'a_pc' only -- deliberately NO a_km fallback. Mixing the two
        # silently scales the profile by ~3e13. The km-unit path is the
        # legacy file, and get_dh_legacy_strings converts there separately.
        a_pc = par['a_pc']
        rho0 = StellarBlackHoles.rho0_from_mass(
            par['m'], a_pc, par['alpha'], par['beta'], par['gamma'])
        pars = (rho0, a_pc, par['alpha'], par['beta'], par['gamma'])
        m_enc = StellarBlackHoles.mass_enclosed(x, y, z, pars)

        G = dyn.constants.GRAV_CONST_KM / dyn.constants.PARSEC_KM
        factor = -G * m_enc / r**3
        return factor * x, factor * y, factor * z

    def get_dh_legacy_strings(self, parset, system):
        """
        Generate the two strings the legacy Fortran needs.

        Overrides the parent because the sampled parameters (m in Msun,
        a in arcsec) differ from the legacy sequence (rhoc in Msun/km**3,
        a in km). This mirrors ``NFW_m200_c``, which likewise injects a
        derived quantity, and ``Hernquist``, which likewise passes a scale
        density rather than a mass.

        Parameters
        ----------
        parset : astropy table row
            Holds the parameter set.
        system : a ``dyn.physical_system.System`` object
            Needed for the distance, to convert arcsec to km.

        Returns
        -------
        specs : str
            legacy code and number of parameters, space separated
        par_vals : str
            parameter values in the sequence legacy Fortran expects

        """
        m = parset[f'm-{self.name}']
        a_arcsec = parset[f'a-{self.name}']
        alpha = parset[f'alpha-{self.name}']
        beta = parset[f'beta-{self.name}']
        gamma = parset[f'gamma-{self.name}']
        a_km = a_arcsec * dyn.constants.ARC_KM(system.distMPc)
        rhoc = self.rho0_from_mass(m, a_km, alpha, beta, gamma)
        specs = f'{self.legacy_code} {len(self.par_names)}'
        par_vals = f'{rhoc} {a_km} {alpha} {beta} {gamma}'
        self.logger.debug(f'sBH legacy strings: {specs} / {par_vals} '
                          f'(from m={m} Msun, a={a_arcsec} arcsec)')
        return specs, par_vals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py`
Expected: `6/6 passed`, exit 0

- [ ] **Step 5: Commit**

```bash
git add dynamite/physical_system.py dev_tests/test_sbh_profile.py
git commit -m "feat(sbh): acceleration, potential and legacy-file strings

a_r = -G M(<r)/r^2 is exact for all gamma<3. The potential's outer term
needs the q<=0 incomplete beta from Task 1, since gamma>=2 is reachable.
get_dh_legacy_strings converts the sampled (m, a_arcsec) into the legacy
(rhoc, a_km) sequence, following the NFW_m200_c precedent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 3: Config validation for two dark slots

**Files:**
- Modify: `dynamite/config_reader.py:1064-1077` (the dark-halo count and component-count checks)
- Modify: `dynamite/config_reader.py:1127-1134` (the allowed-DM-halo-types whitelist)
- Modify: `dynamite/physical_system.py` (add a `System` helper)
- Test: `dev_tests/test_sbh_config.py` (create)

**Interfaces:**
- Consumes: `StellarBlackHoles` (Task 1).
- Produces:
  - `System.get_sbh_component()` -> `StellarBlackHoles` instance or `None`
  - `System.get_halo_component()` -> halo instance or `None`

- [ ] **Step 1: Write the failing test**

Create `dev_tests/test_sbh_config.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config-level checks for the two-dark-slot rule.

Run:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py
"""
import sys
import dynamite as dyn
from dynamite import physical_system as physys


def _mk(cls, name):
    c = cls.__new__(cls)
    c.name = name
    return c


def test_helpers_split_halo_and_sbh():
    """get_halo_component / get_sbh_component must partition the dark comps."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh'),
                  _mk(physys.NFW, 'dh'),
                  _mk(physys.StellarBlackHoles, 'sbh')]
    halo = s.get_halo_component()
    sbh = s.get_sbh_component()
    assert isinstance(halo, physys.NFW), f'halo was {halo}'
    assert isinstance(sbh, physys.StellarBlackHoles), f'sbh was {sbh}'
    print('  helpers_split_halo_and_sbh OK')


def test_helpers_return_none_when_absent():
    """Both helpers must return None rather than raising when unused."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh')]
    assert s.get_halo_component() is None
    assert s.get_sbh_component() is None
    print('  helpers_return_none_when_absent OK')


def test_two_halos_still_rejected():
    """Two *halos* remain illegal; only halo+sBH is the new legal pairing."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh'),
                  _mk(physys.NFW, 'dh1'),
                  _mk(physys.Hernquist, 'dh2')]
    try:
        s.get_halo_component()
    except ValueError:
        print('  two_halos_still_rejected OK')
        return
    raise AssertionError('two halos were accepted')


TESTS = [test_helpers_split_halo_and_sbh,
         test_helpers_return_none_when_absent,
         test_two_halos_still_rejected]

if __name__ == '__main__':
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py`
Expected: FAIL with `AttributeError: 'System' object has no attribute 'get_halo_component'`

- [ ] **Step 3: Write minimal implementation**

In `dynamite/physical_system.py`, add to `class System`, immediately after `get_all_dark_non_plummer_components`:

```python
    def get_sbh_component(self):
        """Get the stellar-black-hole component, if any

        Returns
        -------
        StellarBlackHoles or StellarBlackHolesMGE or None

        Raises
        ------
        ValueError : if more than one sBH component is present

        """
        sbh = [c for c in self.cmp_list
               if isinstance(c, (StellarBlackHoles, StellarBlackHolesMGE))]
        if len(sbh) > 1:
            text = f'System can have at most one sBH component, not {len(sbh)}'
            self.logger.error(text)
            raise ValueError(text)
        return sbh[0] if sbh else None

    def get_halo_component(self):
        """Get the dark halo component, if any

        The halo is any dark, non-Plummer, non-sBH component.

        Returns
        -------
        Component or None

        Raises
        ------
        ValueError : if more than one halo is present

        """
        halo = [c for c in self.get_all_dark_non_plummer_components()
                if not isinstance(c, (StellarBlackHoles,
                                      StellarBlackHolesMGE))]
        if len(halo) > 1:
            text = f'System can have at most one DM halo, not {len(halo)}'
            self.logger.error(text)
            raise ValueError(text)
        return halo[0] if halo else None
```

In `dynamite/config_reader.py`, replace the block at lines 1064-1077:

```python
        if len(self.system.get_all_dark_non_plummer_components()) > 1:
            self.logger.error('System must have zero or one DM Halo object')
            raise ValueError('System must have zero or one DM Halo object')

        if self.system.get_unique_ext_chi2_component() is None:
            check = (2, 3)
        else:
            check = (3, 4)
        if len(self.system.cmp_list) not in check:
            txt = 'System needs to comprise exactly one Plummer, ' \
                  'one VisibleComponent, and zero or one DM Halo object(s)'
            self.logger.error(txt)
            raise ValueError(txt)
```

with:

```python
        # raises if there is more than one of either kind
        has_halo = self.system.get_halo_component() is not None
        has_sbh = self.system.get_sbh_component() is not None

        # base = one Plummer + one VisibleComponent, plus the optional
        # halo, sBH and Chi2Ext components
        n_expected = 2 + int(has_halo) + int(has_sbh)
        if self.system.get_unique_ext_chi2_component() is not None:
            n_expected += 1
        if len(self.system.cmp_list) != n_expected:
            txt = 'System needs to comprise exactly one Plummer, one ' \
                  'VisibleComponent, and at most one DM Halo, one sBH ' \
                  f'and one Chi2Ext object; expected {n_expected} ' \
                  f'components, got {len(self.system.cmp_list)}'
            self.logger.error(txt)
            raise ValueError(txt)
```

In `dynamite/config_reader.py`, extend the whitelist at lines 1130-1134:

```python
                if type(c) not in [physys.NFW, physys.NFW_m200_c,
                                   physys.Hernquist,
                                   physys.TriaxialCoredLogPotential,
                                   physys.GeneralisedNFW,
                                   physys.StellarBlackHoles,
                                   physys.StellarBlackHolesMGE]:
                    text = 'DM Halo needs to be of type NFW, NFW_m200_c, ' \
                           'Hernquist, TriaxialCoredLogPotential, ' \
                           'GeneralisedNFW, StellarBlackHoles, ' \
                           f'or StellarBlackHolesMGE, not {type(c)}'
```

`StellarBlackHolesMGE` does not exist until Task 8. To keep this task
self-contained and the test green, add a placeholder class now, directly
after `StellarBlackHoles` in `physical_system.py`; Task 8 fills it in:

```python
class StellarBlackHolesMGE(DarkComponent):
    """A fixed, externally-supplied sBH profile represented as an MGE.

    Filled in by Task 8. Carries an ``mge_pot`` whose Gaussians are
    concatenated into the potential MGE, so it needs no legacy code and no
    Fortran changes.
    """
    par_names = []

    def __init__(self, mge_pot=None, **kwds):
        self.mge_pot = mge_pot
        super().__init__(symmetry='spherical', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py`
Expected: `3/3 passed`, exit 0

Then confirm nothing regressed for existing configs:

Run: `/opt/miniconda3/envs/main/bin/python -c "import dynamite as dyn; c=dyn.config_reader.Configuration('dev_tests/user_test_config.yaml'); print('OK', len(c.system.cmp_list))"`
Expected: prints `OK 3` (or `OK 2` for a halo-less config) with no exception.

- [ ] **Step 5: Commit**

```bash
git add dynamite/physical_system.py dynamite/config_reader.py dev_tests/test_sbh_config.py
git commit -m "feat(sbh): allow a DM halo and an sBH component in the same system

Replaces the single-dark-halo check with separate halo and sBH slots, and
makes the component-count check derive its expectation instead of using
hardcoded tuples. Two halos remain illegal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 4: Write the second dm block in parameters_pot.in

**Files:**
- Modify: `dynamite/orblib.py:459-497` (the dark-halo string block and the header text)
- Test: `dev_tests/test_sbh_config.py` (add a test)

**Interfaces:**
- Consumes: `System.get_halo_component`, `System.get_sbh_component` (Task 3), `StellarBlackHoles.get_dh_legacy_strings` (Task 2).
- Produces: `parameters_pot.in` gains an optional trailing line pair `"<code> <npar>\n<values>"` after the `H` line.

- [ ] **Step 1: Write the failing test**

Add to `dev_tests/test_sbh_config.py`, above `TESTS`:

```python
def test_sbh_legacy_strings():
    """The legacy strings must carry code 6, 5 params, and derived rhoc."""
    import numpy as np
    from dynamite import constants

    c = _mk(physys.StellarBlackHoles, 'sbh')
    c.legacy_code = 6
    c.par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']
    import logging
    c.logger = logging.getLogger('test')

    class _Sys:
        distMPc = 0.00543          # omega Cen, 5.43 kpc

    parset = {'m-sbh': 1.79e5, 'a-sbh': 100.0, 'alpha-sbh': 3.91,
              'beta-sbh': 4.50, 'gamma-sbh': 2.24}
    specs, vals = c.get_dh_legacy_strings(parset, _Sys())
    assert specs == '6 5', f'specs was {specs!r}'
    parts = [float(v) for v in vals.split()]
    assert len(parts) == 5, f'expected 5 values, got {parts}'
    rhoc, a_km, al, b, g = parts
    assert (al, b, g) == (3.91, 4.50, 2.24), f'shape params wrong: {parts}'
    # a_km must be the arcsec value times the arcsec->km factor
    want_a = 100.0 * constants.ARC_KM(0.00543)
    assert abs(a_km / want_a - 1) < 1e-12, f'a_km {a_km} != {want_a}'
    # and rhoc must reproduce the requested total mass
    m_back = float(physys.StellarBlackHoles.mass_enclosed(
        a_km * 1e8, 0.0, 0.0, (rhoc, a_km, al, b, g)))
    assert abs(m_back / 1.79e5 - 1) < 1e-6, f'mass round trip: {m_back}'
    print('  sbh_legacy_strings OK')
```

Register it in `TESTS`.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py`
Expected: FAIL — either `AssertionError` or `TypeError` on the two-argument
`get_dh_legacy_strings` if Task 2 was skipped.

- [ ] **Step 3: Write minimal implementation**

In `dynamite/orblib.py`, replace the dark-halo block (currently lines 459-476):

```python
        # get dark halo
        dh = self.system.get_all_dark_non_plummer_components()
        if len(dh) > 1:
            txt = f"Zero or one non-plummer dark component should be  present, not {len(dh)}."
            self.logger.error(txt)
            raise ValueError(txt)
        if len(dh) > 0:
            dh = dh[0]  # extract the one and only dm component

            if isinstance(dh, physys.NFW_m200_c):
                # fix c via m200_c relation, for legacy Fortran it is still NFW
                dm_specs, dm_par_vals = dh.get_dh_legacy_strings(
                    self.parset, self.system
                )
            else:
                dm_specs, dm_par_vals = dh.get_dh_legacy_strings(self.parset)
        else:
            dm_specs = "0 0"
            dm_par_vals = ""
```

with:

```python
        # slot 1: the dark halo (raises if more than one is present)
        dh = self.system.get_halo_component()
        if dh is not None:
            if isinstance(dh, physys.NFW_m200_c):
                # fix c via m200_c relation, for legacy Fortran it is still NFW
                dm_specs, dm_par_vals = dh.get_dh_legacy_strings(
                    self.parset, self.system
                )
            else:
                dm_specs, dm_par_vals = dh.get_dh_legacy_strings(self.parset)
        else:
            dm_specs = "0 0"
            dm_par_vals = ""

        # slot 2: the sBH component, written at the END of the file so that
        # existing single-halo parameters_pot.in files parse unchanged (the
        # Fortran reads it under iostat and hits EOF when it is absent).
        # StellarBlackHolesMGE contributes Gaussians to mge_pot instead and
        # so has no legacy block.
        sbh = self.system.get_sbh_component()
        if isinstance(sbh, physys.StellarBlackHoles):
            sbh_specs, sbh_par_vals = sbh.get_dh_legacy_strings(
                self.parset, self.system
            )
            sbh_block = f"{sbh_specs}\n{sbh_par_vals}\n"
        else:
            sbh_block = ""
```

Then find where the potential file text is finalised — the `H` (Hubble
constant) line is the last thing written before the MGE table is appended
by `np.savetxt`. Locate it with:

```bash
grep -n "system.H\|savetxt" dynamite/orblib.py | head
```

Append `sbh_block` to the very end of the file, **after** the `np.savetxt`
call that writes the MGE table, using the same file handle or an append
open. If the file is written via `np.savetxt(fname, mge_pot.data,
header=header_string_pot, comments="")`, add immediately after it:

```python
        if sbh_block:
            with open(fname_pot, "a") as f_sbh:
                f_sbh.write(sbh_block)
```

where `fname_pot` is the path variable already used for
`parameters_pot.in`. Confirm the exact variable name from the surrounding
code before writing.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py`
Expected: `4/4 passed`, exit 0

Then verify the file format by hand on a config with no sBH, confirming it
is byte-identical to before the change:

```bash
git stash
/opt/miniconda3/envs/main/bin/python -c "
import dynamite as dyn
c = dyn.config_reader.Configuration('dev_tests/user_test_config.yaml')
" && cp <output_dir>/models/*/infil/parameters_pot.in /tmp/pot_before.in
git stash pop
```
Then regenerate and `diff /tmp/pot_before.in <new file>`.
Expected: no differences.

- [ ] **Step 5: Commit**

```bash
git add dynamite/orblib.py dev_tests/test_sbh_config.py
git commit -m "feat(sbh): write the optional second dm block to parameters_pot.in

The sBH block goes at the end of the file, after the MGE table, so that
existing single-halo files are byte-identical and the Fortran can read the
block under iostat.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 5: Read the second dm block in Fortran

**Files:**
- Modify: `legacy_fortran/iniparam_f.f90:57-59` (declarations)
- Modify: `legacy_fortran/iniparam_f.f90:149-152` (the `iniparam` read site)
- Modify: `legacy_fortran/iniparam_f.f90:284-288` (the bar-variant read site)

**Interfaces:**
- Consumes: the file format from Task 4.
- Produces, in module `initial_parameters`:
  - `integer(kind=i4b), public :: sbh_profile_type, n_sbhparam`
  - `real(kind=dp), dimension(:), allocatable, public :: sbhparam`
  - `sbh_profile_type` is 0 when no sBH block is present.

- [ ] **Step 1: Add the declarations**

In `legacy_fortran/iniparam_f.f90`, after line 57
(`integer(kind=i4b), public :: n_dmparam, dm_profile_type`) add:

```fortran
    ! Optional second dark component (stellar-mass black hole subcluster).
    ! Written at the END of parameters_pot.in so that files without it parse
    ! unchanged; sbh_profile_type stays 0 when the block is absent.
    integer(kind=i4b), public :: n_sbhparam, sbh_profile_type
    real(kind=dp), dimension(:), allocatable, public :: sbhparam
```

- [ ] **Step 2: Add the optional read at both sites**

In `iniparam` (after `read (unit=13, fmt=*) H`, before `close (unit=13)`):

```fortran
        ! Optional sBH block, appended after H. Absent in every pre-existing
        ! parameters_pot.in, so a nonzero iostat simply means "no sBH".
        sbh_profile_type = 0
        n_sbhparam = 0
        read (unit=13, fmt=*, iostat=sbh_iostat) sbh_profile_type, n_sbhparam
        if (sbh_iostat /= 0) then
            sbh_profile_type = 0
            n_sbhparam = 0
        else
            allocate (sbhparam(n_sbhparam))
            read (unit=13, fmt=*) sbhparam(1:n_sbhparam)
        end if
```

Declare `sbh_iostat` alongside the other locals in that subroutine
(`integer(kind=i4b) :: sbh_iostat`).

Repeat verbatim in the bar variant, which reads `Omega` then `H` — insert
after its `read (unit=13, fmt=*) H`, before its `close (unit=13)`.

- [ ] **Step 3: Rebuild**

Run:
```bash
cd legacy_fortran && make all 2>&1 | tail -20
```
Expected: compiles with no new errors.

- [ ] **Step 4: Verify existing models still run**

Run an existing single-halo model end to end and confirm it produces the
same chi2 as before the change:

```bash
cd dev_tests && /opt/miniconda3/envs/main/bin/python test_nnls.py 2>&1 | tail -20
```
Expected: same result as on `master`. If `test_nnls.py` cannot gate (it
imports the *installed* dynamite and its reference data predates several
merged PRs), instead confirm manually that `orbstart.log` reports the same
`dm_profile_type` and the run completes.

- [ ] **Step 5: Commit**

```bash
git add legacy_fortran/iniparam_f.f90
git commit -m "feat(sbh): optionally read a second dm block from parameters_pot.in

Read under iostat after H, at both read sites (iniparam and the bar
variant). sbh_profile_type stays 0 when the block is absent, so existing
files behave identically.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 6: sBH potential and acceleration in Fortran

**Files:**
- Modify: `legacy_fortran/dmpotent.f90` (module variables, `dm_setup`, `dm_potent`, `dm_accel`, plus one new private function)

**Interfaces:**
- Consumes: `sbh_profile_type`, `n_sbhparam`, `sbhparam` (Task 5); `zh_betai` from `sub/specfunc_beta.f90` (already linked, used by case 5).
- Produces: the sBH contribution added to `pot` and to `(vx,vy,vz)` for every evaluation.

- [ ] **Step 1: Add module state and the beta helper**

In `legacy_fortran/dmpotent.f90`, extend the private declaration
(currently `real(kind=dp), private :: rhoc, rc, dm_logslp`):

```fortran
    real(kind=dp), private :: rhoc, rc, dm_logslp
    ! sBH subcluster (Zhao alpha-beta-gamma), kept in its own slot so the
    ! existing halo cases 1,2,3,5 are untouched.
    logical, private :: sbh_present = .false.
    real(kind=dp), private :: sbh_rho0, sbh_a, sbh_al, sbh_be, sbh_ga
```

Add this function inside the `contains` section:

```fortran
    ! Unregularized incomplete beta B(x;p,q) = int_0^x u^(p-1) (1-u)^(q-1) du,
    ! valid for q <= 0.  zh_betai returns inf for a non-positive second
    ! argument (Numerical Recipes' betacf does not converge there), so step
    ! down from a positive-q evaluation using
    !     B(x;p,q) = [ (p+q) B(x;p,q+1) - x^p (1-x)^q ] / q
    ! Verified against mpmath to 3.3e-14 over r/a = 1e-6..1e4; see
    ! dev_notes/sbh_profile_fits/try_recurrence.py
    function sbh_betai(p, q, x) result(bval)
        real(kind=dp), intent(in) :: p, q, x
        real(kind=dp) :: bval, qq, zh_betai
        integer(kind=i4b) :: n, j

        if (q > 0.0_dp) then
            bval = zh_betai(p, q, x)
            return
        end if
        n = ceiling(1.0_dp - q) + 1
        bval = zh_betai(p, q + real(n, dp), x)
        do j = n, 1, -1
            qq = q + real(j, dp) - 1.0_dp
            bval = ((p + qq)*bval - x**p*(1.0_dp - x)**qq)/qq
        end do
    end function sbh_betai

    ! M(<r) for the sBH profile, in Msun. Both beta parameters are strictly
    ! positive given gamma < 3 and beta > 3.
    function sbh_menc(r) result(menc)
        real(kind=dp), intent(in) :: r
        real(kind=dp) :: menc, xx, t

        xx = r/sbh_a
        t = xx**sbh_al/(1.0_dp + xx**sbh_al)
        menc = 4.0_dp*pi_d*sbh_a**3*sbh_rho0/sbh_al &
               *sbh_betai((3.0_dp - sbh_ga)/sbh_al, &
                          (sbh_be - 3.0_dp)/sbh_al, t)
    end function sbh_menc
```

- [ ] **Step 2: Set up the slot in dm_setup**

At the **end** of `dm_setup`, after the existing `end select`:

```fortran
        ! Optional sBH subcluster, independent of the halo slot above.
        sbh_present = (sbh_profile_type == 6)
        if (sbh_present) then
            if (n_sbhparam /= 5) stop 'wrong number of sBH parameters'
            sbh_rho0 = sbhparam(1)   ! Msun/km^3
            sbh_a = sbhparam(2)      ! km
            sbh_al = sbhparam(3)
            sbh_be = sbhparam(4)
            sbh_ga = sbhparam(5)
            if (sbh_be <= 3.0_dp) stop 'sBH beta must be > 3 (finite mass)'
            if (sbh_ga >= 3.0_dp) stop 'sBH gamma must be < 3'
            if (abs(sbh_ga - 2.0_dp) < 1.0e-6_dp) &
                stop 'sBH gamma must not equal 2 (beta recurrence divides by zero)'
            if (sbh_al <= 0.0_dp) stop 'sBH alpha must be > 0'
            if (sbh_a <= 0.0_dp) stop 'sBH scale radius must be > 0'
            print *, '  * sBH subcluster: rho0, a, alpha, beta, gamma =', &
                sbh_rho0, sbh_a, sbh_al, sbh_be, sbh_ga
            print *, '    total sBH mass (Msun):', sbh_menc(sbh_a*1.0e8_dp)
        end if
```

- [ ] **Step 3: Add the potential contribution**

At the end of `dm_potent`, after its `end select`:

```fortran
        if (sbh_present) then
            d = sqrt(d2)
            xi = (d/sbh_a)**sbh_al/(1.0_dp + (d/sbh_a)**sbh_al)
            ! outer term: 4 pi Int_r^inf r' rho dr'. Its second beta
            ! parameter is <= 0 for gamma >= 2, hence sbh_betai.
            ibeta_v2 = 4.0_dp*pi_d*sbh_a*sbh_a*sbh_rho0/sbh_al &
                       *sbh_betai((sbh_be - 2.0_dp)/sbh_al, &
                                  (2.0_dp - sbh_ga)/sbh_al, 1.0_dp - xi)
            ! this module's `pot` is positive (psi = -Phi), matching the
            ! Plummer and NFW terms above
            pot = pot + grav_const_km*(sbh_menc(d)/d + ibeta_v2)
        end if
```

`d`, `xi` and `ibeta_v2` are already declared in `dm_potent`; confirm and
add any that are missing.

- [ ] **Step 4: Add the acceleration contribution**

At the end of `dm_accel`, after its `end select`:

```fortran
        if (sbh_present) then
            d = sqrt(d2)
            ! exact for all gamma < 3
            acceleration_r = -grav_const_km*sbh_menc(d)/d2
            vx = vx + x/d*acceleration_r
            vy = vy + y/d*acceleration_r
            vz = vz + z/d*acceleration_r
        end if
```

- [ ] **Step 5: Expose sBH-only wrappers for the agreement test**

Task 7's probe cannot call `dm_setup`/`dm_potent`/`dm_accel`, because those
call into `triaxpotent`, which needs a stellar MGE the probe does not have.
Add three thin public wrappers that run *only* the sBH block. They must call
the same `sbh_menc`/`sbh_betai` helpers as the real code paths, otherwise
the agreement test verifies nothing.

Add to the module's public list:

```fortran
    ! sBH-only entry points, for the standalone agreement probe. These
    ! duplicate the sBH blocks of dm_setup/dm_potent/dm_accel but MUST call
    ! the same sbh_menc/sbh_betai helpers, so they cannot drift.
    public:: dm_setup_sbh_only, dm_potent_sbh_only, dm_accel_sbh_only
```

and in `contains`:

```fortran
    subroutine dm_setup_sbh_only()
        sbh_present = (sbh_profile_type == 6)
        if (.not. sbh_present) stop 'dm_setup_sbh_only: no sBH block'
        if (n_sbhparam /= 5) stop 'wrong number of sBH parameters'
        sbh_rho0 = sbhparam(1)
        sbh_a = sbhparam(2)
        sbh_al = sbhparam(3)
        sbh_be = sbhparam(4)
        sbh_ga = sbhparam(5)
    end subroutine dm_setup_sbh_only

    subroutine dm_potent_sbh_only(x, y, z, pot)
        real(kind=dp), intent(in) :: x, y, z
        real(kind=dp), intent(out) :: pot
        real(kind=dp) :: d, xi, tail

        d = sqrt(x*x + y*y + z*z)
        xi = (d/sbh_a)**sbh_al/(1.0_dp + (d/sbh_a)**sbh_al)
        tail = 4.0_dp*pi_d*sbh_a*sbh_a*sbh_rho0/sbh_al &
               *sbh_betai((sbh_be - 2.0_dp)/sbh_al, &
                          (2.0_dp - sbh_ga)/sbh_al, 1.0_dp - xi)
        pot = grav_const_km*(sbh_menc(d)/d + tail)
    end subroutine dm_potent_sbh_only

    subroutine dm_accel_sbh_only(x, y, z, vx, vy, vz)
        real(kind=dp), intent(in) :: x, y, z
        real(kind=dp), intent(out) :: vx, vy, vz
        real(kind=dp) :: d, acceleration_r

        d = sqrt(x*x + y*y + z*z)
        acceleration_r = -grav_const_km*sbh_menc(d)/(d*d)
        vx = x/d*acceleration_r
        vy = y/d*acceleration_r
        vz = z/d*acceleration_r
    end subroutine dm_accel_sbh_only
```

- [ ] **Step 6: Rebuild and commit**

Run:
```bash
cd legacy_fortran && make all 2>&1 | tail -20
```
Expected: compiles clean.

```bash
git add legacy_fortran/dmpotent.f90
git commit -m "feat(sbh): Zhao alpha-beta-gamma potential and acceleration in Fortran

Adds an sBH slot alongside the halo slot, so cases 1,2,3,5 are untouched.
M(<r) uses zh_betai directly (both parameters positive); the potential's
outer term needs the q<=0 downward recurrence, since gamma>=2 is reachable
and zh_betai returns inf there.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 7: Fortran-versus-Python agreement test

**Files:**
- Create: `dev_tests/test_sbh_fortran.py`
- Create: `legacy_fortran/sbh_probe.f90`
- Modify: `legacy_fortran/Makefile` (add the probe target)

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 6.
- Produces: nothing consumed downstream.

This is the only check that guards the duplicated physics. Without it, the
Python and Fortran implementations can silently diverge.

- [ ] **Step 1: Write the probe**

Create `legacy_fortran/sbh_probe.f90` — a standalone program that sets the
sBH module state directly and prints potential and acceleration on a grid,
so the test does not need a full model run:

```fortran
! Standalone probe: prints the sBH potential and radial acceleration on a
! log grid, for cross-checking against the Python implementation.
! Usage:  ./sbh_probe <rho0> <a> <alpha> <beta> <gamma>
program sbh_probe
    use numeric_kinds
    use initial_parameters
    use dmpotent
    implicit none
    character(len=64) :: arg
    real(kind=dp) :: r, pot, vx, vy, vz
    integer(kind=i4b) :: i

    sbh_profile_type = 6
    n_sbhparam = 5
    allocate (sbhparam(5))
    do i = 1, 5
        call get_command_argument(i, arg)
        read (arg, *) sbhparam(i)
    end do
    ! no halo, no stellar MGE, no central black hole
    dm_profile_type = 0
    n_dmparam = 0
    xmbh = 0.0_dp
    softl_km = 0.0_dp
    call dm_setup_sbh_only()

    do i = -60, 40
        r = 10.0_dp**(real(i, dp)/10.0_dp)
        call dm_potent_sbh_only(r, 0.0_dp, 0.0_dp, pot)
        call dm_accel_sbh_only(r, 0.0_dp, 0.0_dp, vx, vy, vz)
        write (*, '(3ES30.18)') r, pot, vx
    end do
end program sbh_probe
```

**Note for the implementer:** the `*_sbh_only` wrappers this probe calls are
created in **Task 6, Step 5** — they already exist by the time you get here.
Do not add them to `dmpotent.f90` again.

- [ ] **Step 2: Add the Makefile target**

In `legacy_fortran/Makefile`, near the other program targets:

```make
sbh_probe : sbh_probe.f90 dmpotent.o iniparam_f.o numeric_kinds_f.o specfunc_beta.o triaxpotent.o
	$(fortran90) sbh_probe.f90
	$(link) -o sbh_probe sbh_probe.o dmpotent.o iniparam_f.o numeric_kinds_f.o specfunc_beta.o triaxpotent.o dqxgs.o ellipint.o
```

Match the exact `$(fortran90)` / `$(link)` macro names and the object list
used by the neighbouring targets — copy from the `orbitstart` target rather
than trusting the list above.

- [ ] **Step 3: Write the failing test**

Create `dev_tests/test_sbh_fortran.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fortran-vs-Python agreement for the sBH component.

The two implementations of the same physics must not drift. Requires
legacy_fortran/sbh_probe to be built:  cd legacy_fortran && make sbh_probe

Run:  /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_fortran.py
"""
import os
import subprocess
import sys

import numpy as np

from dynamite.physical_system import StellarBlackHoles as SBH

PROBE = os.path.join(os.path.dirname(__file__), '..',
                     'legacy_fortran', 'sbh_probe')

# (alpha, beta, gamma); the last three have gamma > 2, where the naive
# incomplete beta fails and the recurrence is required
CASES = [
    (2.15, 12.0, 1.74),
    (0.31, 12.0, 0.51),
    (3.91, 4.50, 2.24),
    (2.00, 4.00, 2.18),
    (4.00, 5.00, 2.50),
]
RHO0, A = 1.0e-8, 1.0e14      # Msun/km^3, km -- plausible legacy-file scale


def run_probe(rho0, a, al, b, g):
    out = subprocess.run(
        [PROBE, repr(rho0), repr(a), repr(al), repr(b), repr(g)],
        capture_output=True, text=True, check=True).stdout
    rows = [list(map(float, ln.split())) for ln in out.strip().splitlines()]
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]      # r, pot, a_x


def test_fortran_matches_python():
    if not os.path.exists(PROBE):
        print(f'SKIP: {PROBE} not built (cd legacy_fortran && make sbh_probe)')
        return
    G = 6.67428e-11 * 1.98892e30 / 1e9          # grav_const_km
    for al, b, g in CASES:
        r, pot_f, ax_f = run_probe(RHO0, A, al, b, g)
        pars = (RHO0, A, al, b, g)
        m_enc = SBH.mass_enclosed(r, 0.0, 0.0, pars)
        ax_p = -G * m_enc / r ** 2
        # the Fortran `pot` is positive (psi = -Phi)
        t = (r / A) ** al / (1.0 + (r / A) ** al)
        tail = np.array([SBH.incomplete_beta(1.0 - ti, (b - 2.0) / al,
                                             (2.0 - g) / al) for ti in t])
        tail *= 4.0 * np.pi * A ** 2 * RHO0 / al
        pot_p = G * (m_enc / r + tail)

        ea = np.max(np.abs(ax_f / ax_p - 1.0))
        ep = np.max(np.abs(pot_f / pot_p - 1.0))
        assert ea < 1e-10, f'accel (al,b,g)=({al},{b},{g}): rel err {ea:.2e}'
        assert ep < 1e-10, f'pot   (al,b,g)=({al},{b},{g}): rel err {ep:.2e}'
        print(f'  ({al}, {b}, {g}): accel {ea:.2e}, pot {ep:.2e}')
    print('  fortran_matches_python OK')


TESTS = [test_fortran_matches_python]

if __name__ == '__main__':
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f'FAIL {t.__name__}: {e}')
            failed += 1
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)
```

- [ ] **Step 4: Build and run**

Run:
```bash
cd legacy_fortran && make sbh_probe && cd .. \
  && /opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_fortran.py
```
Expected: `1/1 passed`, with per-case relative errors below 1e-10.

If a `gamma > 2` case fails while the `gamma < 2` cases pass, the Fortran
recurrence is wrong — compare against
`dev_notes/sbh_profile_fits/try_recurrence.py`, which is the validated
reference.

- [ ] **Step 5: Commit**

```bash
git add legacy_fortran/sbh_probe.f90 legacy_fortran/Makefile dev_tests/test_sbh_fortran.py
git commit -m "test(sbh): Fortran-vs-Python agreement on potential and acceleration

Standalone probe plus a comparison script. This is the only check guarding
the two independent implementations of the same physics, and it covers
gamma>2 where the beta recurrence is required.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 8: The MGE variant

**Files:**
- Modify: `dynamite/physical_system.py` (fill in `StellarBlackHolesMGE`)
- Modify: `dynamite/config_reader.py` (allow `mge_pot` in `keys_ok`)
- Modify: `dynamite/orblib.py` (concatenate the sBH Gaussians into `mge_pot`)
- Test: `dev_tests/test_sbh_config.py` (add a test)

**Interfaces:**
- Consumes: `System.get_sbh_component` (Task 3).
- Produces: `StellarBlackHolesMGE.mge_pot`, a `dyn.mges.MGE`.

- [ ] **Step 1: Write the failing test**

Add to `dev_tests/test_sbh_config.py`, above `TESTS`:

```python
def test_mge_sbh_is_not_a_legacy_block():
    """The MGE variant must contribute Gaussians, never a dm block."""
    s = physys.System()
    s.cmp_list = [_mk(physys.Plummer, 'bh'),
                  _mk(physys.StellarBlackHolesMGE, 'sbh')]
    sbh = s.get_sbh_component()
    assert isinstance(sbh, physys.StellarBlackHolesMGE)
    # it must NOT advertise a legacy code, or orblib would write a block
    assert not hasattr(sbh, 'legacy_code'), \
        'StellarBlackHolesMGE must not define legacy_code'
    print('  mge_sbh_is_not_a_legacy_block OK')
```

Register it in `TESTS`.

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py`
Expected: FAIL if the placeholder from Task 3 defines `legacy_code`; PASS
trivially otherwise — in which case proceed, the substantive work is in
Steps 3-4.

- [ ] **Step 3: Fill in the class**

Replace the Task 3 placeholder in `dynamite/physical_system.py`:

```python
class StellarBlackHolesMGE(DarkComponent):
    """A fixed, externally-supplied sBH profile, as an MGE

    Represents an sBH subcluster whose shape comes from an external model
    (LIMEPY, PhaseFlow, or a collaborator's fit) rather than being fitted.
    Its Gaussians are concatenated into the potential MGE, exactly as
    ``BarDiskComponent``'s ``disk_pot`` is, so this component needs no
    legacy code and no Fortran changes at all.

    Note the structural limit: a sum of Gaussians is flat at the origin, so
    an MGE cannot represent a central cusp below its smallest sigma. Use it
    for a cored profile, or for a cusp only over a bounded radial range.

    Parameters
    ----------
    mge_pot : a ``dyn.mges.MGE`` object
        the (projected) surface-mass density of the sBH subcluster

    """
    # deliberately no legacy_code: this component contributes Gaussians,
    # not a dm block, and orblib.py keys on isinstance(..., StellarBlackHoles)
    par_names = []

    def __init__(self, mge_pot=None, **kwds):
        self.mge_pot = mge_pot
        super().__init__(symmetry='spherical', **kwds)
        self.logger = logging.getLogger(f'{__name__}.{__class__.__name__}')

    def validate(self):
        if not isinstance(self.mge_pot, mge.MGE):
            text = f'{self.__class__.__name__}.mge_pot must be an mges.MGE ' \
                   'object'
            self.logger.error(text)
            raise ValueError(text)

    def validate_parset(self, par):
        # the profile is fixed; there is nothing to sample
        return True
```

**Note:** `Component.validate` requires a non-empty `self.parameters`. This
class has none, so it overrides `validate` entirely rather than calling
`super().validate()`. Confirm that no other code path assumes every
component has parameters — grep for `.parameters` in `parameter_space.py`
and add an exclusion if one is found.

- [ ] **Step 4: Allow mge_pot in the config and concatenate it**

In `dynamite/config_reader.py`, near the existing `keys_ok` block (around
line 1085), extend for the MGE variant:

```python
                    if isinstance(c, physys.StellarBlackHolesMGE):
                        keys_ok.append('mge_pot')
```

Place this alongside the existing `isinstance(c, physys.VisibleComponent)`
and `isinstance(c, physys.Chi2Ext)` branches, and make sure the MGE file is
read into an `mges.MGE` the same way `VisibleComponent.mge_pot` is —
follow the existing reader for `mge_pot`.

In `dynamite/orblib.py`, where `mge_pot` is assembled (line ~515,
`mge_pot = stars.mge_pot + stars.disk_pot` in the bar branch and
`mge_pot = stars.mge_pot` otherwise), append the sBH Gaussians:

```python
        sbh = self.system.get_sbh_component()
        if isinstance(sbh, physys.StellarBlackHolesMGE):
            mge_pot = mge_pot + sbh.mge_pot
            len_mge_pot = len(mge_pot.data)
            header_string_pot = str(len_mge_pot)
```

Place this immediately after `mge_pot` and `header_string_pot` are set, and
before `np.savetxt`. Confirm the header format for the non-bar branch — it
is `str(len_mge_pot)` — and recompute it rather than leaving it stale.

- [ ] **Step 5: Run tests and commit**

Run:
```bash
/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_config.py
/opt/miniconda3/envs/main/bin/python dev_tests/test_sbh_profile.py
```
Expected: all pass.

```bash
git add dynamite/physical_system.py dynamite/config_reader.py dynamite/orblib.py dev_tests/test_sbh_config.py
git commit -m "feat(sbh): MGE variant for a fixed externally-supplied sBH profile

Concatenates the sBH Gaussians into the potential MGE the way
BarDiskComponent already does with disk_pot, so no Fortran changes are
needed. Mutually exclusive with the fitted Zhao component.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

### Task 9: Reference config, regression check and dev note

**Files:**
- Create: `dev_tests/user_test_config_sbh.yaml`
- Create: `dev_notes/sbh_component.md`
- Modify: `CLAUDE.md` (document the component)

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable reference config.

- [ ] **Step 1: Write the reference config**

Copy `dev_tests/user_test_config.yaml` to
`dev_tests/user_test_config_sbh.yaml` and add an sBH component to
`system_components`. Reference values are the PhaseFlow fiducial fit; `a`
is in arcsec, so convert from the fitted 9.27 pc at 5.43 kpc
(1 arcsec = 0.02633 pc, so 9.27 pc = 352 arcsec):

```yaml
    sbh:
        type: "StellarBlackHoles"
        include: True
        contributes_to_potential: True
        parameters:
            m:
                par_generator_settings: {lo: 1.0e4, hi: 5.0e5, step: 5.0e4, minstep: 1.0e4}
                fixed: False
                value: 1.0e5
                logarithmic: False
            a:
                fixed: True
                value: 352.0
            alpha:
                fixed: True
                value: 2.15
            beta:
                fixed: True
                value: 4.5
            gamma:
                fixed: True
                value: 1.75
```

Match the exact `parameters` schema used by the halo component in the same
file — copy its structure rather than the sketch above, which may not match
this repo's current key names.

- [ ] **Step 2: Verify the config loads**

Run:
```bash
/opt/miniconda3/envs/main/bin/python -c "
import dynamite as dyn
c = dyn.config_reader.Configuration('dev_tests/user_test_config_sbh.yaml')
print('components:', [x.name for x in c.system.cmp_list])
print('sbh:', c.system.get_sbh_component())
print('halo:', c.system.get_halo_component())
"
```
Expected: lists the components, shows a `StellarBlackHoles` and the halo.

- [ ] **Step 3: Run the regression check on an existing config**

Confirm a single-halo model is unaffected by all of the above:

```bash
cd dev_tests && /opt/miniconda3/envs/main/bin/python test_nnls.py 2>&1 | tail -20
```
Expected: unchanged from `master`. Record the actual output in the commit
message — if it cannot gate (it imports the *installed* dynamite and its
reference data predates merged PRs #513/#515/#517), say so explicitly
rather than implying a pass.

- [ ] **Step 4: Write the dev note**

Create `dev_notes/sbh_component.md` recording, with the numbers:
the chosen family and why; the reference fit values table from the spec;
the ~12,000 Msun irreducible LIMEPY mass residual; the range-dependence of
gamma (0.51 vs 2.24 from the same data); the 17x total-mass disagreement
between PhaseFlow and LIMEPY; that density below ~0.01 pc is untrustworthy;
and that `zh_betai` returns `inf` for a non-positive second argument.
Link to `dev_notes/sbh_profile_fits/` and the spec.

- [ ] **Step 5: Document in CLAUDE.md and commit**

Add a section to `CLAUDE.md` after "Parameter Generator Variants":

```markdown
### sBH component (`sBH` branch)

`StellarBlackHoles` — spherical Zhao alpha-beta-gamma subcluster of
stellar-mass black holes, `legacy_code = 6`. Config parameters: `m`
[Msun], `a` [arcsec], `alpha`, `beta`, `gamma`. Requires `beta > 3`,
`gamma < 3`, `gamma != 2`. Coexists with a DM halo (two dark slots; the
sBH block is appended at the end of `parameters_pot.in`).

`StellarBlackHolesMGE` — a fixed externally-supplied profile whose
Gaussians concatenate into the potential MGE; no Fortran involved.

Design: `docs/superpowers/specs/2026-09-01-sbh-component-design.md`.
Fits and provenance: `dev_notes/sbh_profile_fits/`.
Tests: `dev_tests/test_sbh_profile.py`, `test_sbh_config.py`,
`test_sbh_fortran.py` (needs `make sbh_probe`).

**Trap:** never call `zh_betai` with a non-positive second argument — it
returns `inf`. Use the downward recurrence (`sbh_betai` in `dmpotent.f90`).
```

```bash
git add dev_tests/user_test_config_sbh.yaml dev_notes/sbh_component.md CLAUDE.md
git commit -m "docs(sbh): reference config, dev note and CLAUDE.md entry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VEMKXNwsfNfaM6jt51Na9j"
```

---

## Self-review notes

**Spec coverage.** Every spec section maps to a task: profile choice ->
Task 1; closed forms -> Tasks 1, 2, 6; potential/recurrence -> Tasks 2, 6;
two components -> Tasks 1, 8; file format -> Tasks 4, 5; changes-by-file
table -> Tasks 3-8; verification -> Tasks 1, 2, 7, 9; provenance -> Task 9.
Out-of-scope items (gNFW fix, >2 dark components, triaxial sBH, the 17x
mass gap) have no tasks, as intended.

**Known soft spots**, flagged rather than hidden:
- Task 4 Step 3 and Task 8 Step 4 require reading the surrounding
  `orblib.py` code to get exact variable names; the plan says so rather
  than inventing them.
- Task 7 s wrappers: RESOLVED in the pre-flight scan, now written out in
  Task 6 Step 5.

- Task 9 Step 1's YAML schema must be copied from the neighbouring halo
  component; the sketch may not match current key names.
