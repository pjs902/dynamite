# Stellar-mass black hole (sBH) component — design

Date: 2026-09-01
Branch: `sBH`
Status: approved design, not yet implemented

## Problem

DYNAMITE models of omega Cen currently represent all dark mass through
either the stellar `ml` or a DM halo. Neither is right for a retained
population of stellar-mass black holes, which is the dominant baryonic dark
mass in the cluster centre and the principal degeneracy against a putative
IMBH. We need a mass component that:

1. represents an sBH subcluster with a shape flexible enough to span a
   flat core and a Bahcall-Wolf cusp;
2. can coexist with a DM halo, so the "is there dark matter beyond the
   black holes" question is answerable;
3. carries a total mass that can be fitted by the kinematics rather than
   assumed.

## Choice of density profile — decided by fitting the external models

Two external models were fitted, both jointly on rho(r) and on the enclosed
mass profile M(<r), with equal weight per decade in r:

- **PhaseFlow** fiducial PDMF run (`~/research/phaseflow/omegacen`,
  `output_pdmf/omegacen_pdmf`), summed over its 6 BH mass bins at the final
  snapshot. Relaxed, contains a central IMBH (`Mbh=8000`). Trimmed to
  `3 x captureRadius < r < 5 pc` to avoid the absorbing-boundary and
  grid-boundary contamination zones.
- **GCfit / LIMEPY** `rho_BH` and `cum_M_BH` median from
  `~/research/omegaCen/BH_profiles/NGC5139_sampler.hdf` via
  `gcfit.analysis.NestedRun(...).get_CImodel()`. Cored, contains no IMBH.

### Result

The adopted family is the **Zhao (1996) alpha-beta-gamma double power law**,
spherical, with all three exponents free:

```
rho(r) = rho0 * (r/a)^-gamma * (1 + (r/a)^alpha)^(-(beta-gamma)/alpha)
```

RMS log10 residual, each variant scored against both targets:

| variant                     | free | PhaseFlow | LIMEPY (0.05-10 pc) | worst |
|-----------------------------|------|-----------|---------------------|-------|
| gNFW (existing legacy 5)    | 3    | 0.033     | 0.128               | 0.128 |
| alpha=4, beta=3 (notebook)  | 3    | 0.021     | 0.104               | 0.104 |
| alpha=2, beta=4             | 3    | 0.023     | 0.059               | 0.059 |
| beta free (alpha=2)         | 4    | 0.021     | 0.034               | 0.034 |
| alpha free (beta=4)         | 4    | 0.019     | 0.031               | 0.031 |
| **alpha, beta, gamma free** | 5    | 0.019     | 0.027               | 0.027 |

Reusing the existing `GeneralisedNFW` (legacy code 5) was rejected: its
fixed beta=3 cannot follow the LIMEPY steepening, and its residual is
systematic rather than noise.

The decisive argument for freeing all three exponents is not the dex score
but the **absolute** enclosed-mass error against the LIMEPY target, which is
what biases an IMBH measurement:

| variant                 | worst \|dM(<r)\| |
|-------------------------|------------------|
| gNFW                    | 50,000 Msun      |
| alpha=2, beta=4         | 34,000 Msun      |
| alpha, beta, gamma free | 13,000 Msun      |

### No better form exists

Checked directly, scored on both rho and M(<r):

| form                              | npar | PhaseFlow max\|dM\| | LIMEPY max\|dM\| |
|-----------------------------------|------|---------------------|------------------|
| single Zhao                       | 5    | 111 Msun            | 12,100 Msun      |
| double Zhao (3 slopes, 2 breaks)  | 8    | 62                  | 18,800 (worse)   |
| MGE, 16 Gaussians                 | 14   | 226                 | 12,600           |

Double-Zhao (60 random restarts) improves rho but degrades the mass fit on
LIMEPY. A 16-Gaussian MGE gets rho five times better yet lands at the same
mass error. The ~12,000 Msun LIMEPY residual is irreducible across all three
families: it sits at 0.1-10 pc where the mass is, and reflects LIMEPY's
core-plus-truncation not being a smooth power law. Adding parameters buys
nothing.

### Known systematics (record these; do not treat the fits as constraints)

- **The residual bounds expressiveness, not model error.** LIMEPY and
  PhaseFlow are shape *references*. `M_sBH, a, alpha, beta, gamma` are all
  free and chosen by the kinematics. The residual becomes a real bias only
  if parameters are ever *fixed* to the fitted values.
- **gamma is range-dependent.** The same LIMEPY data gives gamma = 0.51
  fitted over 1e-3 to 50 pc and gamma = 2.24 fitted over 0.05-10 pc. Any
  gamma reported from a DYNAMITE run is sensitive to which radii carry the
  constraining kinematics.
- **The two external models disagree on total sBH mass by 17x**: PhaseFlow
  retains 1.07e4 Msun, LIMEPY gives 1.79e5 Msun. This is far larger than any
  profile-shape effect measured here. Suspected cause is the `BHret`/`a3`
  sweep axes rather than anything about the profile; not investigated. It is
  the reason `M_sBH` must be freely fitted with wide bounds and no prior.
- **Density below ~0.01 pc is not trustworthy.** The gamma=2.24 fit
  overshoots LIMEPY's core by up to 2.7 dex there. Harmless dynamically
  (~5 Msun enclosed, and the Plummer IMBH dominates), but the component's
  density should not be quoted at those radii.

## Closed forms (verified numerically to 1e-14)

With `x = r/a` and `t = x^alpha / (1 + x^alpha)`:

```
M(<r) = (4 pi a^3 rho0 / alpha) * B(t; (3-gamma)/alpha, (beta-3)/alpha)
M_tot = (4 pi a^3 rho0 / alpha) * B((3-gamma)/alpha, (beta-3)/alpha)
a_r   = -G * M(<r) / r^2
```

using the incomplete beta `B(x; p, q) = x^p/p * 2F1(p, 1-q; p+1; x)`, the
same `hyp2f1` form `GeneralisedNFW.mass_enclosed` already uses.

**Both beta-function parameters are strictly positive** given gamma < 3 and
beta > 3, so the existing Fortran `zh_betai` covers M(<r) and hence the
acceleration with no new special functions.

Constraints, and where they come from:
- `beta > 3` — required for M_tot to converge (integrand ~ r^(2-beta)).
  Not binding in practice: the joint fits land at beta = 4-4.5 unforced and
  recover M_tot to ~2%.
- `gamma < 3` — required for M(<r) to converge at the origin.
- `gamma < 2` is **not** imposed. Phi(0) genuinely diverges for gamma >= 2,
  but see below: that radius is never evaluated.

## Potential

`Phi(r) = -G [ M(<r)/r + 4 pi Int_r^inf r' rho dr' ]`. The outer term's
integrand goes as r^(1-gamma) and so diverges as r -> 0 for gamma >= 2. This
is real physics, not a formula defect (the hypergeometric convergence
condition c-a-b>0 reduces exactly to gamma < 2).

It does not require restricting gamma, because of how DYNAMITE evaluates
potentials:

- The orbit integrators call `ip_accel`/`ip_potent` (`interpolpotent.f90`),
  never `dm_accel`/`dm_potent` directly.
- `ip_setup_grid` tabulates **accelerations only** on a 640 x 64 x 64
  log-r/theta/phi grid, calling `dm_accel` to fill it, and caches it to
  disk. Its inner radius is bounded by
  `rmin2 = min((10^rlogmin * 0.01)^2, (min(sigobs_km)/10)^2)`, so r=0 is
  never reached.
- `ip_potent` is **not** interpolated — it passes straight through to
  `dm_potent`, so `dm_potent` must still be correct and cheap.

**Decision:** tabulate Phi once in `dm_setup` by cumulative integration of
the analytic `G*M(<r)/r^2` over a log-r grid spanning the interpolation
grid's range, and interpolate in `dm_potent`. ~20 lines, contained, never
touches the acceleration path, and the arbitrary additive constant is
irrelevant to the dynamics. `ip_testaccuracy` already hard-stops on a
misbehaving grid, so failures are loud.

## Components

Two classes, mutually exclusive per model:

1. **`StellarBlackHoles(DarkComponent)`** — the fitted component.
   `legacy_code = 6`, `symmetry = 'spherical'`.

   The sampled/YAML parameters and the legacy-file parameters differ, so the
   class carries both:

   - `par = ['m', 'a', 'alpha', 'beta', 'gamma']` — what the user writes in
     the config and what the parameter generators sample. `m` is the total
     sBH mass in Msun; `a` is the scale radius.
   - `par_names = ['rhoc', 'a', 'alpha', 'beta', 'gamma']` — the legacy
     sequence written to `parameters_pot.in`.

   `get_dh_legacy_strings` is overridden to substitute
   `rho0 = m * alpha / (4 pi a^3 B((3-gamma)/alpha, (beta-3)/alpha))`
   for `m`. Precedent for a legacy sequence that differs from the sampled
   parameters is `NFW_m200_c`, which injects a derived `c200` the same way;
   precedent for passing a scale density rather than a mass is `Hernquist`.

   Doing the conversion in Python keeps the complete beta function in scipy
   where it is easy to test, at the cost of `parameters_pot.in` no longer
   showing the physically meaningful mass. The Fortran therefore never needs
   the *complete* beta, only the incomplete one it already has.

   `validate_parset` enforces `m > 0`, `a > 0`, `alpha > 0`, `beta > 3`,
   `gamma < 3`.
   Implements `density`, `mass_enclosed`, `acceleration` staticmethods,
   mirroring `GeneralisedNFW`.

2. **`StellarBlackHolesMGE(DarkComponent)`** — a fixed, externally supplied
   profile (from LIMEPY, PhaseFlow, or a collaborator), carrying an
   `mge_pot`. No legacy code and **no Fortran changes at all**: the
   Gaussians are concatenated into the potential MGE in `orblib.py`
   (precedent: `mge_pot = stars.mge_pot + stars.disk_pot`, `orblib.py:515`),
   and `iniparam_f.f90` reads the Gaussian count from the top of the file,
   so a longer list is transparent.
   Note the structural limit: a sum of Gaussians is flat at the origin, so
   an MGE cannot represent a central cusp below its smallest sigma. It is
   the right tool for a core, and for a cusp only over a bounded range.

## File format

`parameters_pot.in` is read sequentially by `iniparam_f.f90` in the order:
MGE Gaussian count and table, scalars, dm block, (`Omega` in the bar
variant,) `H`, then `close`. The file ends at `H`.

**The second dm block is appended after `H`, at the end of the file.**
Existing single-halo files then parse byte-identically and hit EOF where the
sBH block would be, read under `iostat`. Appending it adjacent to the first
dm block was rejected because it would shift everything after it.

Two read sites must be updated: `iniparam` (~line 149) and the bar variant
(~line 284).

## Changes by file

| File | Change |
|------|--------|
| `dynamite/physical_system.py` | Add `StellarBlackHoles` and `StellarBlackHolesMGE`. |
| `dynamite/config_reader.py` | Whitelist both classes (~line 1130); relax the single-dark-halo check to at most one halo plus at most one sBH; widen the component-count check; add `mge_pot` to `keys_ok` for the MGE variant; forbid both sBH variants at once. |
| `dynamite/orblib.py` | Write the optional second dm block at end of file; concatenate the sBH MGE into `mge_pot`. |
| `legacy_fortran/iniparam_f.f90` | New public `dm2_profile_type`, `n_dm2param`, `dm2param`; optional read at both sites. |
| `legacy_fortran/dmpotent.f90` | `case (6)` in `dm_setup`/`dm_potent`/`dm_accel`; second component slot; Phi tabulation. |
| `dev_tests/test_sbh_profile.py` | Standalone verification script (see below). |
| `dev_notes/sbh_component.md` | Fitted reference values, provenance, fitting scripts. |

Exactly two dark slots are supported: one halo, one sBH. `dmpotent` keeps
two named parameter sets rather than a general list, so the existing
cases 1, 2, 3 and 5 are not refactored and cannot regress.

## Out of scope

- **The latent gNFW divergence.** `dm_potent` case 5 calls
  `zh_betai(1.0, 2.0 - gamma_var, ...)`, whose second argument goes negative
  for gamma > 2. It is unreachable today because `GeneralisedNFW.validate_parset`
  requires `gam <= 1`. Left untouched deliberately: it is not an active bug,
  and editing a working upstream-shared path widens the merge surface for no
  current benefit. Recorded here so the next person to relax that bound
  knows.
- More than two dark components.
- Triaxial or flattened sBH (both external models are spherical codes).
- Resolving the 17x total-mass disagreement.

## Verification

`dev_tests/test_sbh_profile.py`, a standalone script matching the existing
`dev_tests` convention (these are run directly, not collected by pytest):

1. Python `density` integrates to `M_sBH` across a spread of
   (alpha, beta, gamma), including gamma > 2.
2. Python `acceleration` equals `-dPhi/dr` by finite difference.
3. `mass_enclosed` agrees with direct quadrature to < 1e-8.
4. Fortran vs Python agreement on potential and acceleration at a spread of
   radii, from a tiny model run. This is the only check that guards the
   duplicated physics.
5. Regression: an existing single-halo NFW model produces an unchanged
   potential, guarding the file-format change on the common path.

## Provenance

Fitting and verification scripts to be committed under `dev_notes/`:
`target_phaseflow.py`, `target_gcfit.py`, `fit_joint.py`,
`fit_alternatives.py`, `check_closed_form.py`, `check_potential.py`,
`plot_bestfit.py`.

Reference values from the joint (rho + M) fits:

| target | gamma | alpha | beta | a [pc] | rho dex | M dex |
|--------|-------|-------|------|--------|---------|-------|
| PhaseFlow fiducial   | 1.74 | 2.15 | 12.0 | 9.27 | 0.023 | 0.034 |
| LIMEPY, full range   | 0.51 | 0.31 | 12.0 | 79.4 | 0.336 | 0.247 |
| LIMEPY, 0.05-10 pc   | 2.24 | 3.91 | 4.50 | 3.06 | 0.040 | 0.038 |

These are starting values for a YAML, not priors. Note the beta=12 entries
are at the fit bound and are degenerate with large `a` — bound `alpha` and
`beta` in the config rather than letting a sampler wander there.
