# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DYNAMITE (DYnamics, Age and Metallicity Indicators Tracing Evolution) is a Python package for Schwarzschild orbit-superposition and stellar-population modelling of stellar systems (galaxies, globular clusters). Python orchestrates compiled Fortran binaries for the heavy orbit integration and weight-solving computation.

## Installation & Build

The package requires two separate build steps before it can be used:

```bash
# 1. Compile Fortran orbit/weight-solving programs
cd legacy_fortran/galahad-2.3 && ./install_galahad
cd legacy_fortran && make all

# 2. Install the Python package
python setup.py install
# or for editable/development installs:
pip install -e .
```

Optional dependency for CVXOPT weight solver:
```bash
pip install cvxopt>=1.2.6
```

## Running Tests

Tests live in `dev_tests/` and are **standalone scripts**, not a pytest suite. Run them directly from `dev_tests/`:

```bash
cd dev_tests

# Single model run test (main integration test)
python test_nnls.py

# Data preparation test
python test_dataprep.py

# Orbit LOSVD test
python test_orbit_losvds.py

# Orbit decomposition test
python test_decomp.py

# Full grid of weight solvers × parameter generators (takes a long time)
bash test_all.sh
# With SLURM: bash test_all.sh SLURM
```

`test_all.sh` programmatically injects weight solver and parameter generator variants into config files via `awk`/`sed` — check that script when debugging test configuration issues.

## Architecture

### Entry Point Pattern

Every workflow starts with a `Configuration` object parsed from a YAML file, which is then passed to virtually all other constructors:

```python
import dynamite as dyn
c = dyn.config_reader.Configuration('my_config.yaml')
# c.system    → physical_system.System (galaxy components)
# c.settings  → config_reader.Settings (orblib/weight/IO/multiprocessing settings)
# c.parspace  → parameter_space.ParameterSpace (list of Parameters)
```

### Key Modules

| Module | Role |
|--------|------|
| `config_reader.py` | Parses YAML config into `Configuration`, `Settings` objects |
| `physical_system.py` | `System` + component hierarchy (`VisibleComponent`, `DarkComponent` subclasses like `NFW`, `Plummer`, `Hernquist`) |
| `parameter_space.py` | `ParameterSpace` (list of `Parameter`) + parameter generators: `LegacyGridSearch`, `GridWalk`, `FullGrid`, `SpecificModels` |
| `orblib.py` | `LegacyOrbitLibrary` — calls compiled Fortran binaries as subprocesses to integrate orbits and produce LOSVDs |
| `weight_solvers.py` | `LegacyWeightSolver` (calls Fortran NNLS), `NNLS` (Python/scipy/cvxopt) — finds orbital weights that best fit kinematics |
| `model.py` | `Model` (single model: directories + orblib + weights) and `AllModels` (astropy table tracking all runs) |
| `model_iterator.py` | `ModelIterator` — drives the parameter search loop: generate parset → run model → check stopping criteria → repeat |
| `analysis.py` | `Decomposition` — decomposes orbits into kinematic components post-modelling |
| `plotter.py` | Visualization of chi2 landscapes, kinematic maps, orbit weights |
| `kinematics.py` | Kinematic data containers and LOSVD handling |
| `populations.py` | Stellar population data handling |
| `data_prep/` | Preprocessing utilities to prepare kinematic input data |

### Python–Fortran Interface

The Fortran programs (`orblib`, `orbitstart`, `triaxnnls_*`, etc.) are compiled into `legacy_fortran/` and packaged as data files alongside the Python package. `LegacyOrbitLibrary` and `LegacyWeightSolver` write Fortran-format input files, call these binaries as subprocesses, then read output files back. When modifying these classes, the Fortran I/O format is the contract.

### Configuration YAML Structure

Config files have two top-level sections:
- `system_attributes` + `system_components` — defines the galaxy (black hole, dark halo, stars, kinematics, MGE data)
- Settings blocks: `orblib_settings`, `parameter_space_settings`, `weight_solver_settings`, `io_settings`, `multiprocessing_settings`, `legacy_settings`

See `dev_tests/user_test_config.yaml` or `dev_tests/user_test_config_ml.yaml` for working examples.

### Weight Solver Variants

- `LegacyWeightSolver` — wraps the compiled Fortran NNLS solver; uses `CRcut` flag for counter-rotating orbits (see Zhu+2018)
- `NNLS` — pure Python implementation using scipy or cvxopt; selected by `type:` in `weight_solver_settings`

### Parameter Generator Variants

Selected by `generator_type` in `parameter_space_settings`:
- `LegacyGridSearch` — fixed grid search (legacy behavior)
- `GridWalk` — adaptive grid walk
- `FullGrid` — exhaustive grid
- `SpecificModels` — run a user-specified list of parameter sets
- `BayesOptGenerator` — GP-driven Bayesian Optimization via BoTorch (bayesopt branch)

### Mass parameters are stored per orbit library

An orblib is reused across `ml` by rescaling LOSVD velocity axes by
`sqrt(ml/ml_orblib)`, which scales the *whole potential* by `ml/ml_orblib`.
So dark-component masses in `all_models` are **per orbit library**:

```
physical mass = stored value * ml/ml_orblib  ( = scale_factor**2 )
```

This applies to `m-bh`, the fitted `StellarBlackHoles` `m`, `Hernquist`
`rhoc` and `GeneralisedNFW` `Mvir`. `NFW`'s `c`/`f` and the shape exponents
are invariant (`f` is a fraction of `totalmass`, which is `ml`-scaled in
`iniparam_f.f90`), as is `StellarBlackHolesMGE`, whose Gaussians sit in the
`ml`-scaled potential MGE. `TriaxialCoredLogPotential`'s `Vc` is a velocity,
so it takes `scale_factor**1`.

Use `AllModels.get_physical_parameter_table()` for any readout of masses for
analysis, plotting or a warm start. Use the raw `self.table` only to
reconstruct what the Fortran actually computed — in particular
`get_model_from_parset()` matches on *stored* values. Corollary: reuse across
`ml` and an `ml`-independent absolute mass are mutually exclusive; the only
reuse-invariant way to write a mass is as a ratio, which is what `f` is.
Quantify the effect for a given system with
`dev_tests/check_ml_selfsimilarity.py`.

### sBH component (`sBH` branch)

`StellarBlackHoles` — spherical Zhao alpha-beta-gamma subcluster of
stellar-mass black holes, `legacy_code = 6`. Config parameters: `m`
[Msun], `a` [arcsec], `alpha`, `beta`, `gamma`. Requires `beta > 3`,
`gamma < 3`, `gamma != 2`. Coexists with a DM halo (two dark slots; the
sBH block is appended at the end of `parameters_pot.in`).

**Default shape: the GCfit/LIMEPY 0.05-10 pc fit** — `a = 3.06 pc`
(116.2 arcsec at 5.43 kpc), `alpha = 3.91`, `beta = 4.50`, `gamma = 2.24`,
all **fixed**, with only `m` free. This is the `'production'` case in
`test_sbh_fortran.py`. Do not use the PhaseFlow fiducial shape: its scale
radius falls outside its own fitting range. The shape choice shifts
M_sBH(<10 arcsec) by ~12x at fixed total mass and so biases `m-bh`
directly — treat it as a systematic, not a detail.

`StellarBlackHolesMGE` — a fixed externally-supplied profile whose
Gaussians concatenate into the potential MGE; no Fortran involved. It has
no sampled parameters, but the config reader still demands the key: its
YAML entry needs an explicit `parameters: {}`. On a bar-disk system its
Gaussians are folded in BEFORE the disk block, since the Fortran reads
`[bulge, disk]` in file order.

Design: `docs/superpowers/specs/2026-09-01-sbh-component-design.md`.
Fits and provenance: `dev_notes/sbh_profile_fits/`.
Tests: `dev_tests/test_sbh_profile.py`, `test_sbh_config.py`,
`test_sbh_fortran.py` (needs `make sbh_probe`).

**Numerics:** the Fortran sBH path is self-contained in `dmpotent.f90`
(`sbh_beta_series`, `sbh_betacf`, `sbh_binc`, `sbh_outer_tail_integral`)
and must NEVER call `zh_betai` — the shared `zh_betacf` in
`sub/specfunc_beta.f90` carries a single-precision `EPS = 3e-7`, which
would cap Fortran-vs-Python agreement at ~1e-7 instead of the achieved
1e-13. Separately, `zh_betai` returns `inf` for a non-positive second
argument; that is a latent trap in the PRE-EXISTING gNFW `dm_potent`
case 5 only, not something the sBH path touches. See
`dev_notes/sbh_component.md` "Numerical traps".

### BayesOptGenerator (`bayesopt` branch)

Located in `parameter_space.py:~1132`. Requires `botorch`, `gpytorch`, `torch`.

**Key `generator_settings` keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `n_initial_random` | 10 | Sobol warm-up models before GP (sobol mode) |
| `batch_size` | 8 | Models per iteration |
| `n_orblib_configs` | 4 | Distinct potential configs per Sobol batch |
| `n_ml_per_config` | 2 | ml values per orblib config (Sobol only) |
| `warmup_mode` | `sobol` | `sobol` or `initial_guess` (axial design) |
| `initial_guess` | `{}` | Physical values for axial center (initial_guess mode) |
| `initial_step_size` | 0.1 | Axial step in normalized [0,1] space |
| `discretize_non_ml_params` | `False` | Snap non-ml GP proposals to par grid for orblib reuse |

**Orblib reuse:** DYNAMITE already supports reusing orbit libraries when only `ml` changes (`model_iterator.py:is_new_orblib()`). With `discretize_non_ml_params: true`, GP proposals are snapped to the `par_generator_settings.step` grid, making repeated visits to the same potential-shape grid cell likely — each reuse saves one full orblib run (~18 h on cluster).

**Stopping criteria:** Exactly one of `min_delta_chi2_abs` or `min_delta_chi2_rel` must be present. Set to a large negative value (e.g. `-1e6`) when using `initial_guess` warm-up to prevent premature termination when axial probes explore unfavorable directions.

**Tests:** `dev_tests/test_bayesopt_generator.py` — run with `pytest dev_tests/test_bayesopt_generator.py` (38 tests). Reference YAMLs: `dev_tests/bayesopt_ml_modelinner.yaml`, `dev_tests/bayesopt_qml_modelinner.yaml`.

**Production comparison script:** `dev_tests/run_comparison_real.py` — runs BayesOpt, GridWalk, and LegacyGridSearch on NGC6278 with 3 free params (ml, c-dh, f-dh — both NFW halo params) and generates corner/convergence/chi2-surface plots. Stars shape (q) is fixed to avoid triaxiality validity conflicts at batch_size=1.

```bash
python dev_tests/run_comparison_real.py --ncpus 48 --nmodels 100 --nE 11 --nI2 7 --nI3 5 --dithering 3
# resume / regenerate plots only:
python dev_tests/run_comparison_real.py --output-dir comparison_YYYYMMDD_HHMMSS --skip-runs
```

**v2 (2026-08-22):** partial-free triaxial feasibility (q,p free with u fixed works), in-place GP warm-start from existing `all_models.ecsv` (switch `generator_type` to `BayesOptGenerator` in the same output_directory — no Sobol burn), batch dedup after grid snapping, and opt-in acquisition upgrades: `exploration_schedule: annealed` (eta schedule), `n_annealed_members`, `trust_region: true`, prediction-accuracy diagnostic (`gp_predictions_accurate` status flag). Hard runtime gates vs GridWalk: `dev_tests/test_vs_gridwalk.py`; variant matrix: `dev_tests/run_ablation.py` (warm-start arm needs ~16 fresh models vs 40 cold on the dummy landscape). Env smoke test: `python dev_tests/test_bayesopt_smoke.py`. Production reference config (NGC5139 xeast): `dev_tests/NGC5139_config_production.yaml` — REQUIRES `modeliterator: SplitModelIterator` (the memory cap only exists in split flow).

**Results (2026-06-17, large orblib):** BayesOpt 5814 chi2 at ml=3.88 vs GridWalk 6352 at ml=4.0 — ~540 unit advantage, and finds non-grid ml values. Small orblib (nE≤5) gives wrong landscape; use nE≥11. `LOG_PARAMS={'c-dh','f-dh'}` controls log10-transform for plotting (all_models stores linear values).

**Upstream merge note:** `iniparam_f.f90` must declare `quad_nr`, `quad_nth`, `quad_nph` as public integers and read them from unit 13. `orblib.py` must write them after `dithering`. All config YAMLs need `quad_nr: 10`, `quad_nth: 6`, `quad_nph: 6` under `orblib_settings`.
