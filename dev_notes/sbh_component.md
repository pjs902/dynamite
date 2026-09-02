# sBH component: physics, numerics, and traps

Branch `sBH`. Design doc:
`docs/superpowers/specs/2026-09-01-sbh-component-design.md`.
Fitting/provenance scripts: `dev_notes/sbh_profile_fits/`.

## What this is

A spherical mass component representing a retained subcluster of
stellar-mass black holes (sBH) at the centre of a globular cluster (omega
Cen is the motivating case), which needs to be distinguishable from a
putative IMBH and from a conventional DM halo. Two classes:

- `StellarBlackHoles` (`legacy_code = 6`) — the fitted Zhao alpha-beta-gamma
  profile, Fortran-backed, sampled by the parameter generators.
- `StellarBlackHolesMGE` — a fixed externally-supplied MGE profile, no
  Fortran involved, Gaussians concatenated straight into the potential MGE.

They are mutually exclusive per model. `StellarBlackHoles` can coexist with
a DM halo (two dark slots: halo + sBH), which is the point — it lets a run
answer "is there dark matter beyond the black holes?" rather than
conflating the two.

## Why Zhao alpha-beta-gamma, not gNFW

Two external references were fitted jointly on rho(r) and M(<r):
**PhaseFlow** (relaxed N-body PDMF run with a central IMBH) and
**GCfit/LIMEPY** (cored, no IMBH). Candidate families were scored on RMS
log10 residual and, more decisively, on worst absolute enclosed-mass error
against LIMEPY (the number that biases an IMBH measurement):

| variant                 | worst \|dM(<r)\| |
|--------------------------|------------------|
| gNFW (existing legacy 5) | 50,000 Msun      |
| alpha=2, beta=4 (fixed)  | 34,000 Msun      |
| alpha, beta, gamma free  | 13,000 Msun      |

The existing `GeneralisedNFW` (legacy code 5) was rejected because its
fixed beta=3 cannot follow the LIMEPY steepening and its residual is
systematic, not noise. All three Zhao exponents (alpha, beta, gamma) are
therefore free parameters, sampled from the kinematics like any other
component parameter — the fitted values below are starting points, not
priors.

Adding more shape freedom does not help further: a double-Zhao (8 params,
3 slopes/2 breaks) improves rho but *degrades* the LIMEPY mass fit
(18,800 Msun, worse than single Zhao), and a 16-Gaussian MGE gets rho five
times better yet lands at essentially the same mass error (12,600 Msun).

## The irreducible ~12,000 Msun LIMEPY mass residual

Across every family tried (single Zhao, double Zhao, 16-Gaussian MGE), the
LIMEPY enclosed-mass residual bottoms out around 12,000-13,000 Msun and
will not shrink with more parameters. It sits at 0.1-10 pc, where the mass
actually is, and reflects LIMEPY's core-plus-truncation shape not being a
smooth power law — a real shape mismatch, not an underfit. Treat this as a
floor on how well any smooth analytic profile can represent LIMEPY, not as
a bug to chase with more parameters.

## gamma is range-dependent — the same data, two very different numbers

Fitting the *same* LIMEPY data over different radial ranges gives:

- gamma = 0.51 over the full range (1e-3 to 50 pc)
- gamma = 2.24 over 0.05-10 pc

A `gamma` reported from any DYNAMITE run is therefore sensitive to which
radii carry the constraining kinematics, not a single well-defined number
intrinsic to the target population. Don't over-interpret a fitted gamma
without checking what radial range the data actually constrains.

## PhaseFlow vs LIMEPY disagree on total mass by 17x

PhaseFlow retains 1.07e4 Msun in stellar-mass black holes; LIMEPY gives
1.79e5 Msun for the same population — a 17x disagreement, far larger than
any profile-shape effect measured in this work. Suspected (not
investigated) cause is the `BHret`/`a3` sweep axes in the external codes
rather than anything about the density profile itself. This is why `m`
(total sBH mass) must always be freely fitted with wide bounds and no
prior — see the reference config's `par_generator_settings` for `m`
(1e4 to 5e5 Msun) — never fixed to either external value.

## Density below ~0.01 pc is not trustworthy

The gamma=2.24 fit overshoots LIMEPY's actual core by up to 2.7 dex at
these radii. This is dynamically harmless in practice (~5 Msun enclosed
there, and a Plummer IMBH dominates that regime anyway), but the
component's *density* should never be quoted or plotted at radii below
about 0.01 pc.

## Potential at gamma >= 2: real divergence, handled by dropping a constant

`T(r) = 4 pi Int_r^inf r' rho dr'` (the outer-tail part of the potential)
is expressed via an incomplete beta function with a parameter
`q = (2-gamma)/alpha`, which goes non-positive whenever gamma >= 2 — and
the LIMEPY 0.05-10 pc fit lands at gamma = 2.24, squarely in that regime.
This divergence is real physics (the hypergeometric convergence condition
reduces exactly to gamma < 2), not a formula defect. It is handled without
restricting gamma or tabulating anything:

1. Phi is only defined up to an additive constant, so the divergent
   constant is simply dropped; the constant-free difference
   `T(r1) - T(r2)` matches mpmath quadrature to 1e-16 .. 1e-50 across test
   cases including all gamma >= 2, and the divergence is mild in
   magnitude (the antiderivative reaches only ~ -350 at r/a = 1e-8 for
   gamma = 2.24).
2. The incomplete beta is extended to q <= 0 by a downward recurrence from
   a positive-q evaluation (start at q+n with n = ceil(1-q)+1, evaluate
   with the existing well-tested `zh_betai` there, step down n times).
   Verified against mpmath over r/a = 1e-6 .. 1e4 to worst relative error
   3.3e-14.

`gamma = 2` exactly is rejected in `validate_parset` (division by zero in
the recurrence, and `2-gamma = 0` in the antiderivative) — the physical
content there is a log limit, and the parameter is continuous with no
reason to land exactly on it.

## Trap: `zh_betai` returns `inf` for a non-positive second argument

Do not call `zh_betai` directly with a non-positive second argument. It
has a `b <= 0` branch that forces Numerical Recipes' continued-fraction
evaluation (`betacf`), and a faithful Python transcription of
`sub/specfunc_beta.f90`, tested against mpmath, returns `inf` for every
q <= 0 case tried — `betacf` simply does not converge for negative b. This
is measured, not assumed, and is the entire reason the downward recurrence
above exists (`sbh_betai` in `dmpotent.f90`). The acceleration path never
hits this: both beta-function parameters for `M(<r)` —
`(3-gamma)/alpha` and `(beta-3)/alpha` — are strictly positive given the
enforced constraints `gamma < 3` and `beta > 3`, so `zh_betai` is used
there unmodified.

This same measured behaviour is also recorded as a latent, deliberately
untouched issue in the pre-existing gNFW path (`dm_potent` case 5, legacy
code 5): it calls `zh_betai(1.0, 2.0 - gamma_var, ...)`, whose second
argument goes negative for gamma > 2. It is unreachable today because
`GeneralisedNFW.validate_parset` requires `gam <= 1`, and is left alone
deliberately (not an active bug; editing a working upstream-shared path
widens the merge surface for no current benefit). Anyone relaxing that
constraint would get `inf` potentials, not merely reduced accuracy; the
fix, if ever wanted, is the same downward recurrence used here.

## Reference fit values

Joint (rho + M) fits against the two external targets, used as the
starting point for the reference config (`dev_tests/user_test_config_sbh.yaml`,
which uses the PhaseFlow fiducial row):

| target                | gamma | alpha | beta | a [pc] | rho dex | M dex |
|-----------------------|-------|-------|------|--------|---------|-------|
| PhaseFlow fiducial     | 1.74  | 2.15  | 12.0 | 9.27   | 0.023   | 0.034 |
| LIMEPY, full range     | 0.51  | 0.31  | 12.0 | 79.4   | 0.336   | 0.247 |
| LIMEPY, 0.05-10 pc     | 2.24  | 3.91  | 4.50 | 3.06   | 0.040   | 0.038 |

Note the beta=12 rows sit at the fit bound and are degenerate with large
`a`; bound `alpha` and `beta` in any config that samples them rather than
letting a sampler wander toward that degeneracy. The reference config
instead fixes `alpha, beta, gamma` at values within the well-conditioned
range (2.15, 4.5, 1.75) and only frees `m`.

## Reference config

`dev_tests/user_test_config_sbh.yaml` — a copy of
`dev_tests/user_test_config.yaml` with an `sbh` component added alongside
the existing `bh` and `dh` components (Zhao alpha-beta-gamma, `a` converted
from the PhaseFlow fiducial 9.27 pc at 5.43 kpc to 352 arcsec using
1 arcsec = 0.02633 pc). Verified to load via `Configuration(...)`, showing
`bh`, `dh`, `sbh`, `stars` as components and `StellarBlackHoles` +
`NFW` returned from `get_sbh_component()`/`get_halo_component()`.

## Regression status

`cd dev_tests && python test_nnls.py` was run against the pre-existing,
untouched single-halo `user_test_config_ml.yaml` (the `sbh` component was
never invoked). It completed end to end with the local editable
`dynamite` — config load, orblib build, NNLS weight solve, kinchi2/chi2
plots, kinematic map — with no exceptions and no unexpected NaN (one row's
`nan` chi2 is expected `LegacyGridSearch` behaviour, outside the
3-model toy grid's evaluated range).

**This is not a pass/fail gate.** `test_nnls.py` has no `assert` or
reference-comparison statements at all (zero matches for
`assert|reference|expected`) — see [[dynamite_stale_test_references]].
It prints a table and makes plots with no stored baseline to diff
against, so running it cannot verify "unchanged from master"; it only
shows the single-halo path still runs to completion untouched by this
branch's changes.
