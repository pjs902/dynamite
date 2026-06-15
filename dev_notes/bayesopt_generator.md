# BayesOptGenerator — Implementation Notes

## Overview

`BayesOptGenerator` is a `ParameterGenerator` subclass in `dynamite/parameter_space.py` (line ~1132) that replaces grid-search iteration with Bayesian Optimization: a GP surrogate model + qLogEI acquisition function from BoTorch. It is designed to find good Schwarzschild model parameters in fewer forward-model evaluations than `LegacyGridSearch` or `GridWalk`, at the cost of GP overhead per iteration.

---

## Architecture

### Warm-up + GP phases

Generation proceeds in two phases controlled by `n_initial_random`:

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

Additional optional keys: `acquisition_type` (default `qLogEI`), `max_gp_variance_threshold`, `min_ei_threshold`.

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

## Test coverage

| Test file | What it tests |
|-----------|--------------|
| `test_bayesopt_generator.py` | 14 unit tests: parameter encoding, init, Sobol warm-up, GP acquisition, stopping criteria, naming fix regression |
| `compare_generators.py` | Dummy-mode comparison: BayesOpt vs GridWalk vs LegacyGridSearch on 1D quadratic bowl (ml free). BayesOpt reaches chi2≈15 (true min); grid methods stuck at 65 (step=1.0 resolution) |
| `test_physical_validity.py` | Monkey-patches `validate_parset` to count rejections during a 2D (q-stars+ml) dummy run; asserts accepted models all have q-stars ≤ qobs |
| `test_bayesopt_e2e.py` | End-to-end GP learning test via direct `gen.generate()` calls with synthetic chi2 landscape; no ModelIterator |
| `run_bayesopt_real.py` | Full real DYNAMITE run (orbit + weights); asserts finite chi2 |
| `run_comparison_real.py` | BayesOpt vs LegacyGridSearch with real orblib; prints per-iteration comparison table |

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
| `dev_tests/bayesopt_ml_modelinner.yaml` | ml-only, ModelInnerIterator, tiny orblib |
| `dev_tests/bayesopt_ml_split.yaml` | ml-only, SplitModelIterator |
| `dev_tests/bayesopt_qml_modelinner.yaml` | q-stars+ml free, validity testing |
| `dev_tests/gridwalk_ml_modelinner.yaml` | Reference: GridWalk same criteria |
| `dev_tests/legacygrid_ml_modelinner.yaml` | Reference: LegacyGridSearch same criteria |
