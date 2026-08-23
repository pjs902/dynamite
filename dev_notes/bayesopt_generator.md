# BayesOptGenerator — Implementation Notes

## Overview

`BayesOptGenerator` is a `ParameterGenerator` subclass in `dynamite/parameter_space.py` (line ~1132) that replaces grid-search iteration with Bayesian Optimization: a GP surrogate model + qLogEI acquisition function from BoTorch. It is designed to find good Schwarzschild model parameters in fewer forward-model evaluations than `LegacyGridSearch` or `GridWalk`, at the cost of GP overhead per iteration.

---

## Architecture

### Warm-up modes

`warmup_mode` (generator setting, default `'sobol'`) controls the initial exploration strategy before the GP is fit.

#### `sobol` (default)

Quasi-random Sobol space-filling draws until `n_done_finite >= n_initial_random`. The GP is not fit during warm-up; `specific_generate_method` calls `_propose_random_batch`. This is equivalent to the original behaviour.

#### `initial_guess`

Axial design centred on a user-provided physical point: 1 center point + 2 axial perturbations per free parameter = `1 + 2 * n_free` warm-up models total. With 5 free params this is 11 models — fits the ~10-model RAM budget before GP kicks in.

Configuration keys:
- `warmup_mode: initial_guess`
- `initial_guess` (dict): physical parameter values for the center; omitted params default to normalized midpoint (0.5)
- `initial_step_size` (float, default 0.1): step size in normalized [0,1] space for axial probes

The queue is built at `__init__` time by `_build_axial_queue()` and consumed by `_propose_axial_batch()`. When the queue is empty, `specific_generate_method` falls through to `_gp_acquisition_batch()`.

**Important**: set `min_delta_chi2_abs` to a large negative value (e.g. −1 × 10⁶) when using `initial_guess` mode, because axial probes deliberately explore bad directions and would trigger the stopping criterion prematurely. Only `n_max_mods` and `n_max_iter` should govern the axial phase.

### Warm-up + GP phases

Generation proceeds in two phases controlled by `n_initial_random` (sobol mode) or the axial queue (initial_guess mode):

1. **Sobol warm-up** (while `n_done_finite < n_initial_random`): quasi-random space-filling via `SobolEngine(scramble=True)`. The GP is not fit yet; there are no acquisition calls.
2. **GP phase**: `SingleTaskGP` fit via BOTORCH's `fit_gpytorch_mll`; `qLogEI` acquisition maximized with `optimize_acqf`. Proposes `batch_size` candidates per iteration.

Switching between phases is in `specific_generate_method` (line ~1298). The count uses only rows where `all_done=True AND kinchi2 is finite`.

### Batch structure (`n_orblib_configs` × `n_ml_per_config`)

DYNAMITE's `SplitModelIterator` separates orbit integration from weight solving. The batch structure maps to this: `n_orblib_configs` distinct parameter sets for orbit integration, each evaluated at `n_ml_per_config` values of `ml` (reusing the same orblib). `batch_size = n_orblib_configs × n_ml_per_config`.

When `n_ml_per_config=1` (as in `ModelInnerIterator` configs), each orblib config is evaluated at exactly one ml value, drawn directly from Sobol. When `n_ml_per_config > 1`, the ml values for each orblib config are evenly spaced in normalized space: `(m + 0.5) / n_ml_per_config` for m in 0..n_ml_per_config-1.

### Triaxiality constraints

`TriaxialVisibleComponent` requires `p >= q` and `max(q/qobs, p) <= u <= min(p/qobs, 1)`. This is enforced two ways:

- **Sobol warm-up**: `_project_unit_to_feasible_qpu` clips (q,p,u) to the feasible box after Sobol draws.
- **GP phase**: `_make_triaxiality_constraints` returns BoTorch `NonlinearInequalityConstraint` objects that the acquisition optimizer must satisfy.

Both only activate when all three of q, p, u are **free** (checked via `_free_qpu_idx`). When some are fixed, there are no BoTorch constraints and the Sobol draw may land outside the feasible region. Such models are filtered by `validate_parset` → `_is_newmodel` before they enter `all_models.table`; they never reach the orbit solver.

### qobs

`get_qobs_from_system(system)` extracts `qobs` from any `TriaxialVisibleComponent` in the system. This is the projected axial ratio used to bound the triaxial viewing angles. Stored as `gen.qobs`; `None` if no triaxial component exists.

---

## Parameter naming

DYNAMITE names parameters as `{base}-{component}` (e.g. `q-stars`, `p-stars`, `u-stars`, `m-bh`). System-level parameters have no suffix (`ml`). When searching for q/p/u among free parameters, compare the base name:

```python
if p.name.split('-')[0] == axis:  # correct
if p.name == axis:                # BUG — 'q' ≠ 'q-stars'
```

`_free_qpu_idx` was initially broken (used bare name comparison), which silently disabled all triaxiality constraints in every real DYNAMITE run. Fixed in commit `09ef9d7`.

---

## Configuration

Minimum required keys in `generator_settings`:

| Key | Description |
|-----|-------------|
| `n_initial_random` | Sobol warm-up size before GP fits |
| `batch_size` | Models proposed per iteration |
| `n_orblib_configs` | Orblib configs per batch (SplitModelIterator) |
| `n_ml_per_config` | ml values per orblib config |

Recognized by `config_reader.py` under `generator_type: BayesOptGenerator`.

Additional optional keys:

| Key | Default | Description |
|-----|---------|-------------|
| `acquisition_type` | `qLogEI` | Acquisition function type |
| `max_gp_variance_threshold` | 1.0 | GP variance stopping threshold |
| `min_ei_threshold` | -1.5 | Log-EI stopping threshold |
| `warmup_mode` | `sobol` | `sobol` or `initial_guess` |
| `initial_guess` | `{}` | Dict of physical param values for axial center |
| `initial_step_size` | 0.1 | Axial step in normalized [0,1] space |
| `discretize_non_ml_params` | `False` | Snap non-ml GP proposals to par grid steps |

`discretize_non_ml_params: true` snaps each non-ml free parameter's GP proposals to the nearest value on its grid (defined by `par_generator_settings.step`). `ml` is never snapped — it stays continuous. When two GP proposals snap to the same potential-parameter grid cell, DYNAMITE's `is_new_orblib()` logic reuses the existing orblib for the second model, running only the weight solve. This can cut wall time by `n_ml_per_config`× when orblib runs dominate.

Stopping criteria (in `stopping_criteria`): `n_max_mods`, `n_max_iter`, `min_delta_chi2_abs`, `gp_max_variance_low`, `gp_min_ei_low`.

---

## Known bugs fixed during development

### 1. `_free_qpu_idx` naming (commit `09ef9d7`)
**Bug**: compared `p.name == 'q'` instead of `p.name.split('-')[0] == 'q'`. Triaxiality constraints never activated.

### 2. Sobol warm-up ml override for `n_ml_per_config=1` (commit `014e5bb`)
**Bug**: `_propose_random_batch` set `r[ml_free_j] = (m + 0.5) / n_ml_per_config`. With `n_ml_per_config=1`, this is always `0.5` — the ml midpoint — regardless of the Sobol draw. With `batch_size=2` and ml as the only free parameter, BOTH proposals had identical ml=midpoint. The second was rejected as a duplicate; BayesOpt produced only 1 unique model and stalled.
**Fix**: skip the ml override when `n_ml_per_config <= 1`; use the Sobol-drawn value directly.

### 3. Dummy run mode: `kinchi2=NaN` and `all_done=False` (commit `940cfb8`)
**Bug**: `model_iterator.py` dummy path set `mod.kinchi2 = np.nan` and left `orb_done = wts_done = False`, so `all_done=False` in the table. BayesOptGenerator reads `kinchi2` (= `which_chi2`) for GP training data — all NaN means the generator never exits warm-up, and the iteration counter never progresses.
**Fix**: in dummy mode, set `kinchi2 = chi2` (mirror the dummy function output) and `orb_done = wts_done = True`.

### 4. `config_reader.py` validated `n_batch` instead of `batch_size` (commit `5cdf48e`)
**Bug**: required-key validation checked for `n_batch` but the class uses `batch_size`.
**Fix**: corrected key name in `config_reader.py`.

---

## Upstream merge (upstream/master → bayesopt, June 2026)

### Changes required after merge

The full upstream merge (commit `70faac7`) introduced several breaking changes that required adaptation:

**1. `check_specific_stopping_criteria` typo fixed upstream**
The base class method was `check_specific_stopping_critera` (typo) and our override matched. Upstream fixed the spelling. Required renaming our override and all test call sites.

**2. `quad_nr` / `quad_nth` / `quad_nph` grid settings**
Upstream added a spherical polar quadrant grid used for recording intrinsic moments. Changes spanned three layers:
- `iniparam_f.f90`: new public integer variables + reads after `orbit_dithering` (auto-merge silently dropped these)
- `orblib.py`: writes `quad_nr`, `quad_nth`, `quad_nph` to `parameters_pot.in` after `dithering`
- All test YAMLs: `quad_nr: 10`, `quad_nth: 6`, `quad_nph: 6` added under `orblib_settings`

**3. `weight_solvers.py` NameError** (`kins`/`pops` undefined)
`origin/master` had a partial refactor in `get_observed_mass_constraints()` where `kins`/`pops` were used but never defined. Resolved by taking the full upstream version.

**4. `model_iterator.py` staging files removed + `chi2_ext` added**
Upstream removed staging files entirely and added optional `chi2_ext` to the output tuple via `self.has_chi2_ext`. Taken from upstream wholesale.

**5. `contributes_to_potential` deprecated**
Upstream changed this from a required attribute to deprecated (will be ignored). Our test YAMLs already added it during development; the deprecation warnings are benign and the attribute can be removed in a future cleanup.

---

## Comparison: BayesOpt vs GridWalk vs LegacyGridSearch (dummy mode, 2D)

Benchmark on a synthetic 2D landscape: `chi2 = 80*(q-0.4)^2 + 200*(ml-5.5)^2 + 15` with 24–25 model budget. Run via `dev_tests/plot_generator_comparison.py`.

| Generator | Models | Best chi2 | Notes |
|-----------|--------|-----------|-------|
| BayesOpt | 25 | ~16–19 | Concentrates near true minimum (q=0.4, ml=5.5) |
| GridWalk | 18 | 65.03 | Grid resolution limits; steps away from minimum |
| LegacyGridSearch | 25 | 65.00 | Fixed grid; same resolution issue |

BayesOpt reaches chi2 within ~2–4 units of the true minimum (15); grid methods are stuck at 65 due to coarse step size. The advantage grows with dimensionality and budget constraints.

Output plots: `dev_tests/generator_corner_comparison.png` (proposed models in parameter space, colored by iteration) and `dev_tests/generator_convergence.png` (running-best chi2 vs iteration).

---

## Production comparison script

`dev_tests/run_comparison_real.py` — full real-orblib comparison of all three generators on NGC6278 with 3 free parameters (ml, c-dh, f-dh — both NFW halo params; stars shape q fixed to avoid triaxiality validity conflicts at batch_size=1).

```bash
# Basic run — adjust ncpus and nmodels to match the server
python dev_tests/run_comparison_real.py --ncpus 8 --nmodels 60

# Large orblib (recommended for real science)
python dev_tests/run_comparison_real.py --ncpus 48 --nmodels 100 --nE 11 --nI2 7 --nI3 5 --dithering 3

# Resume a partial run (completed generators are skipped automatically)
python dev_tests/run_comparison_real.py --output-dir comparison_20260616_120000 --ncpus 8 --nmodels 60

# Regenerate plots only from existing results (e.g. after pulling plot fixes)
python dev_tests/run_comparison_real.py --output-dir comparison_20260616_120000 --skip-runs
```

Defaults: ncpus=4, nmodels=60, orblib=(5,5,5), dithering=1. Output goes to a timestamped directory `comparison_YYYYMMDD_HHMMSS/` containing:

| File | Contents |
|------|----------|
| `config_{gen}.yaml` | DYNAMITE config used for each generator |
| `{gen}/all_models.ecsv` | Full model table per generator |
| `corner_proposals.png` | Parameter-space proposals coloured by model index |
| `convergence.png` | Running-minimum chi2 vs cumulative model count |
| `chi2_surfaces.png` | Marginal chi2 vs each free parameter |

BayesOpt runs with `discretize_non_ml_params=True` (orblib reuse), `n_initial_random=12`, and `min_delta_chi2_abs=-1e6` (runs to n_max_mods; chi2-improvement criterion fires prematurely during Sobol warm-up). GridWalk starts from parameter default values and walks ±step. LegacyGridSearch seeds from all models within 4 chi2 units of the current minimum.

**Important:** `LOG_PARAMS = {'c-dh', 'f-dh'}` in the script controls which parameters are log10-transformed before plotting (DYNAMITE stores physical/linear values in all_models.ecsv even for logarithmic parameters).

---

## Real orblib comparison results (NGC6278, 2026-06-17)

Full run on server: `--ncpus 48 --nmodels 100 --nE 11 --nI2 7 --nI3 5 --dithering 3`

| Generator | Models | Best kinchi2 | ml | log c | log f |
|-----------|--------|-------------|-----|-------|-------|
| BayesOpt | 144 | 5813.54 | 3.88 | 2.00 | 0.50 |
| GridWalk | 46 | 6351.99 | 4.00 | 2.00 | 0.00 |
| LegacyGridSearch | 22 | 6459.82 | 5.00 | 2.00 | 0.00 |

**Key findings:**
- BayesOpt wins by ~540 chi2 units and finds ml=3.88 (between grid points — impossible for grid methods)
- Convergence structure: Sobol warm-up flat at ~10k chi2 for first 48 models, GP drops to ~6k at model 48, slow refinement to 5.8k by model 144
- All generators agree: ml≈4, log_c at lower boundary (=2.0), log_f poorly constrained
- **Small orblib (nE=5) gave a completely wrong chi2 landscape** (minimum at upper boundary, chi2~16500) — the large orblib is essential
- log_c=2.0 is the lower boundary for all three generators — parameter range should be extended: try lo=1.0 for log_c
- log_f≈0–0.5 is also near the lower boundary — dark matter total mass is poorly constrained by inner kinematics (expected)

---

## Real orblib test results (dev, 1D)

`dev_tests/run_bayesopt_real.py` — NGC6278, ml-only free (nE=2, nI2=4, nI3=3, dithering=1), single CPU, 12 model budget.

- Wall time: ~28s (MacBook M3 Pro)
- Models run: 7
- Best `kinchi2`: ~14,438 at ml≈9.0

Confirms end-to-end integration: Fortran binaries read `quad_nr/nth/nph`, Python writes them, BayesOpt proposes sensible values.

---

## Performance notes

In 1D (ml only), BayesOpt has no sample-efficiency advantage — `LegacyGridSearch` exhaustively covers the space in ~4 models. The GP overhead dominates. BayesOpt becomes valuable with ≥3 free parameters (e.g., ml + DM halo c + f), where the search space grows exponentially and grid methods are unaffordable.

Expected wall time for `bayesopt_ml_modelinner.yaml` (nE=2, nI2=4, nI3=3, ml only free, ncpus=1): ~2–5 minutes per model × 12 models ≈ 30–60 minutes total.

---

## Files

| File | Role |
|------|------|
| `dynamite/parameter_space.py:1132` | `BayesOptGenerator` class |
| `dynamite/model_iterator.py:536` | Dummy run path (do_dummy_run=True) |
| `dynamite/config_reader.py:1086` | Recognizes `BayesOptGenerator` type |
| `dynamite/orblib.py:228` | Writes `quad_nr/nth/nph` to `parameters_pot.in` |
| `legacy_fortran/iniparam_f.f90:69` | `quad_nr/nth/nph` public declarations |
| `dev_tests/bayesopt_ml_modelinner.yaml` | ml-only, ModelInnerIterator, tiny orblib |
| `dev_tests/bayesopt_ml_split.yaml` | ml-only, SplitModelIterator |
| `dev_tests/bayesopt_qml_modelinner.yaml` | q-stars+ml free, validity testing |
| `dev_tests/gridwalk_ml_modelinner.yaml` | Reference: GridWalk same criteria |
| `dev_tests/legacygrid_ml_modelinner.yaml` | Reference: LegacyGridSearch same criteria |
| `dev_tests/plot_generator_comparison.py` | Dummy-mode corner plot + convergence curve |
| `dev_tests/run_bayesopt_real.py` | Full real orblib integration test |

## v2 modernization (2026-08-22)

Production-driven upgrade, spec: docs/superpowers/specs/2026-08-22-bayesopt-modernization-design.md.

New generator_settings keys (defaults preserve v1 behavior):
- `exploration_schedule`: 'constant'|'annealed' — anneals qLogEI's indicator
  temperature eta from eta_start=0.1 down to eta_end=1e-3 over
  anneal_batches GP batches: explore early, exploit late (R1, GPry-style
  dimension-scaled schedule).
- `n_annealed_members`: batch members drawn ~ exp(mu_GP/tau) by rejection
  over the feasible set, tau annealed tau_start*tau_decay^batch, floor
  tau_min; falls back to projected Sobol on low acceptance (R2, SALE
  annealed objective).
- `trust_region` (+ tr_* keys): TuRBO-lite single trust region, entered when
  mean kNN distance around the incumbent drops below tr_trigger_frac of the
  box diagonal; grows on improvement, shrinks after tr_patience stale
  batches (R4).
- `pred_eps_rel`/`pred_eps_abs`/`pred_hits_needed`: prediction-accuracy
  counter; max(4, ceil(d/2)) consecutive accurate GP predictions raise the
  status flag `gp_predictions_accurate` — a convergence diagnostic, never a
  stop by itself (R3, GPry CorrectCounter).

Behavior changes (not behind flags):
- Triaxial feasibility now works for ANY free subset of (q,p,u); fixed axes
  use their parset values. Production case q,p-free/u-fixed=0.9999 projects
  Sobol draws and constrains GP proposals with p-q>=0 plus q<=u*qobs bounds.
- Warm-start: restarting BayesOptGenerator in a populated output_directory
  trains the GP on all historical rows; out-of-bounds rows are clipped with
  a warning; axial warm-up centers on the best historical model when
  initial_guess is absent.
- Snapped batches are deduplicated per non-ml cell (ml pairing kept - it is
  intentional orblib reuse); freed slots refill via projected Sobol.

Tests & comparisons:
- `dev_tests/test_vs_gridwalk.py` — HARD GATES: BayesOpt must reach the
  chi2 threshold in fewer models than GridWalk and keep GP overhead
  <10 s/model. Current dummy result: BO 27 fresh models vs GridWalk stalling
  at 14 without ever reaching it.
- `dev_tests/run_ablation.py` — matrix incl. GridWalk arm; warm-start arm
  needs only 16 fresh models (vs 40 cold) thanks to seeded history.

Literature: SALE (arXiv:2608.00841), GPry (2023 JCAP 10 021),
TuRBO (arXiv:1910.01739), BOLFI (2018), ALABI (2026); full list in spec.

## Surface-based generator comparison (2026-08-22)

`dev_tests/chi2_landscapes.py` defines three realistic chi2 surfaces
(shape ridge with mass-flattening tilt; production-like 4D with an
orthogonal BH bowl; the ml-halo constant-mass banana) calibrated to
NGC6278 ranges. `run_surface_comparison.py` drives GridWalk vs
BayesOptGenerator over them; `plot_surface_comparison.py` renders
trajectory projections + convergence curves into `dev_tests/surface_figs/`.

Dummy-scale results (150-model budget): production-4d is BayesOpt's clear
win (threshold crossed at ~45 fresh models vs ~180 for GridWalk, plus a
59-chi2-better optimum); shape-ridge is a tie; halo-banana is GridWalk's win
(its start point sits near the single valley - 22 models vs 64). Honest
takeaway: BO pays off when the landscape couples parameters or hides the
valley from the starting point, not on single-valley problems.
