# BayesOptGenerator Modernization v2 — Design

Date: 2026-08-22
Branch: `bayesopt` (post-merge of master @ 64c152e + chi2_kinmap fix)
Status: approved design, pending implementation plan

## 1. Motivation and context

The next production run is NGC5139 (omega Cen) on server hardware: 90 cores,
1416 GB RAM, four kinematic sets (MUSE BayesLOSVD + HST proper motions + MUSE
GaussHermite + Gaia proper motions), free parameters (bh mass, ml, q, p) with
u fixed at 0.9999, NFW out of the potential, adelie float32 streamed weight
solves at ~190 GiB/worker, orbit libraries of ~45k orbits. GridWalk with
n_max_mods=500 is the incumbent generator; each model costs hours, so
parameter-search efficiency is the dominant cost lever.

The `BayesOptGenerator` (BoTorch GP + qLogEI, Sobol/axial warm-up) was
validated in June 2026 on NGC6278: chi2 5814 vs GridWalk's 6352, finding
ml=3.88 between grid points. It has never run at production scale or with a
partially-free shape-parameter set.

Feasibility audit findings that motivate this work:

- **Blocker (config)**: the production YAML omits `modeliterator:
  SplitModelIterator`. The default `ModelInnerIterator` runs weight solves
  inside the ncpus=45 pool with no concurrency cap — up to 45 × 190 GiB on a
  1416 GB node. `ncpus_weights: 6` only takes effect under
  SplitModelIterator.
- **Gap (generator)**: `_project_unit_to_feasible_qpu` and
  `_make_triaxiality_constraints` no-op unless all three of q, p, u are free.
  Production frees q, p with u fixed → invalid proposals are filtered only
  after costing an iteration slot.
- **Flag (config)**: q range [0.05, 0.99] extends above the validity bound
  q ≤ u·qobs ≈ 0.9999·qobs for omega Cen-like qobs; permanently-invalid grid
  region wastes proposals.
- **Flag (config)**: header file-copy list does not cover the lvm_* and
  gaia_* kinematic input files the config declares.
- **Economics**: Sobol warm-up sat flat for ~48 models in the June run; at
  server scale that is tens of hours of wasted orblibs. In-place restart in
  the same output_directory already reloads all_models.ecsv, so switching
  generator_type after a GridWalk run gives the GP every historical row for
  free — this is the agreed warm-start mechanism.

## 2. Literature grounding

Survey of GP-surrogate and BO practice relevant to this design (ADS/arXiv,
2026-08). No published work applies BO to Schwarzschild orbit-superposition
outer loops; all ingredients have precedent elsewhere.

Adopted mechanisms:

- **SALE** (Li 2026, arXiv:2608.00841): annealed objective interpolating
  posterior sampling ↔ Thompson sampling (stable against spurious narrow GP
  spikes, unlike raw TS); curvature-adjusted k-NN distance around the
  incumbent as a cheap local-resolution proxy governing BO↔UR allocation;
  PSRF-style convergence checks at refresh times. → R2 batch sampler, R4
  trust-region trigger, stopping diagnostics.
- **GPry** (El Gammal+ 2023, JCAP 10 021): exploration relaxation ζ = d^-0.85
  balancing exploitation vs exploration by dimension; Kriging-believer batch
  guidance "batch ≈ min(dims, workers)"; centroid-seeded acquisition
  restarts once a good region is known; CorrectCounter convergence = repeated
  accurate GP predictions at newly evaluated points, requiring ~⌈d/2⌉
  consecutive hits. → R1 schedule, batch sizing guidance, IC seeding, R3
  stopping criterion.
- **TuRBO** (Eriksson+ 2019, arXiv:1910.01739): local trust regions with
  success/failure-driven resizing beat global acquisition late in runs.
  Single-TR variant adopted. → R4.

Surveyed but deferred: asynchronous/budgeted parallel BO on HPC
(aphBO-2GP-3B, Tran+ 2020), multi-armed-bandit trust-region allocation
(Huynh+ 2025), acquisition ensembles for diversity (MACE, Zhang+ 2022),
entropy-search acquisitions (PF²ES), feasibility classifiers (GPry's SVM —
we have exact algebraic constraints instead), BO-built surrogate posteriors
as the product (BOSS, Alsing+ 2023, BOLFI).

Structural anchor: Barnabè & Koopmans (2007) — cheap inner linear problem
(orbit weights via NNLS) with outer nonlinear-parameter optimization is the
canonical split our GP sits on. Science context: Lamprecht+ (2026, FCC 47)
is current-day production use of DYNAMITE + BAYES-LOSVD.

## 3. Design

### 3.1 Hardening (correctness; blocks production)

H1 · Partial-free triaxial feasibility.
Rewrite `_project_unit_to_feasible_qpu` and `_make_triaxiality_constraints`
to operate on any free subset of {q, p, u}; fixed axes take their parset
values. For u fixed: project Sobol draws with p ← max(p, q), clip
q ≤ min(q_hi, u_fixed·qobs·(1−1e-6)), and emit only the p − q ≥ 0 nonlinear
constraint (u-window constraints collapse to bounds on q given u, p).
Feasibility must hold identically in warm-up projection and GP-phase
constraints. Unit tests cover all seven free/non-free combinations plus
qobs edge cases (None, ≥ 1).

H2 · Warm-start guardrails (in-place reuse).
On generator init: count valid rows (`all_done & finite kinchi2`) and log
"warm-starting from N historical models" when N > 0; skip warm-up exactly as
today (n_valid ≥ n_initial_random) but make the log explicit. After
normalization, count training rows outside [0, 1] per axis; warn with counts
and clip them into training. `initial_guess` axial center defaults to the
best-known row (min which_chi2) when `initial_guess` is absent but history
exists.

H3 · SplitModelIterator alignment.
End-to-end dummy-mode test with `modeliterator: SplitModelIterator`
verifying: batch structure maps to n_orblib_configs × n_ml_per_config;
snapped duplicate potentials reuse one orblib via `is_new_orblib`;
weight solves stay within ncpus_weights. Add `modeliterator:
SplitModelIterator` to all shipped bayesopt YAMLs.

H4 · Dependency pinning and smoke test.
Pin botorch/gpytorch/torch to a tested range in requirements.txt. Ship
`dev_tests/test_bayesopt_smoke.py`: build a tiny GP from synthetic data,
run one qLogEI acquisition step, assert finite output. Runs without DYNAMITE
model machinery so it doubles as an environment check on new nodes.

### 3.2 Batch quality

B1 · Post-snap collision handling.
After `_snap_to_grid`, deduplicate candidates by snapped non-ml cell; keep
the first (highest acquisition value); refill freed slots with feasible Sobol
draws (through H1 projection). Deterministic, no new GP machinery. Test:
batch with forced collisions produces unique cells and full batch_size.

B2 · Configuration guidance.
Document `n_ml_per_config` pairing as the orblib-reuse lever (forced ml pairs
per potential config) vs ml-freedom trade-off in the class docstring and the
production YAML comments. Note GPry's guideline batch ≈ min(dims, workers);
production ablation decides final values.

### 3.3 Acquisition research

R1 · Dimensional exploration schedule (cheap).
Expose acquisition exploration scale as a generator_setting
(`exploration_schedule`); default anneals qLogEI's beta down over iterations
from a dimension-scaled start (GPry ζ = d^-0.85 mapping). Off-switch reverts
to today's constant beta.

R2 · Tempered-posterior batch members (centerpiece).
Each GP-phase batch splits into EI-optimized members and members drawn by
sampling exp(μ_GP(x)/τ) restricted to the feasible set via rejection
sampling (τ annealed downward across iterations; SALE's annealed-objective
mechanism simplified to mean-only, no path sampling in v1). Mix fraction is
a generator_setting (`n_annealed_members`, default ⌈batch/4⌉). Feasibility
via the same H1 projection/constraints. Rationale: diversity without the
homogenization of pure joint-qEI batches; robustness to spurious GP spikes.

R3 · Prediction-accuracy stopping (cheap, zero runtime cost).
At each completed model, compare pre-evaluation GP prediction μ_GP(x) with
the realized kinchi2; tolerance |μ − y| < ε_abs + ε_rel·|y_best − μ| (GPry
CorrectCounter adapted; defaults ε_rel = 0.01, ε_abs = 0). Require
m = max(4, ⌈d/2⌉) consecutive accurate predictions to raise
`gp_predictions_accurate`. Reported alongside existing
gp_max_variance_low / gp_min_ei_low criteria; never force-stops alone.

R4 · Trust-region refinement (medium, off by default).
Trigger: SALE-style local-resolution proxy — mean curvature-adjusted k-NN
distance around the incumbent below threshold. Action: restrict acquisition
bounds to a box around the incumbent (side length from recent step success);
shrink on failure streaks, grow on successes (TuRBO-lite, single TR). Flag:
`trust_region: true` in generator_settings; production enables it only after
ablation.

Explicitly deferred (documented in docstring): asynchronous batch BO across
the model pool; feasibility classifiers; entropy-search acquisitions;
surrogate-posterior products (evidence estimation).

### 3.4 Validation

NGC6278 remains the testing set throughout.

V1 · Unit suite: extend dev_tests/test_bayesopt_generator.py with H1
combinatorics, H2 guardrail behaviors, B1 collision handling, R3 counter,
R4 trigger mechanics (dummy mode; no GPU required).

V2 · Dummy-mode ablation matrix on the synthetic landscape: warm-up modes
{sobol, axial, warm-start} × {baseline qLogEI, R1+R2, R1–R4}; metrics:
models-to-threshold, best-chi2-at-budget, orblib-reuse rate.

V3 · Real-data comparison: rerun dev_tests/run_comparison_real.py
(GridWalk, LegacyGridSearch, upgraded BayesOpt) against June baselines —
BayesOpt 5814 chi2 is the number to beat; GridWalk 6352 the
production-relevant reference.

V4 · Production dress rehearsal: NGC5139 config at reduced orblib
(nE=11-class) end-to-end on the cluster before xeast scale, including a
warm-start continuation from the GridWalk run's all_models.ecsv.

V5 · Production config patch (YAML-only, separate commit):
add `modeliterator: SplitModelIterator`; tighten q.hi to
min(0.99, round(0.99·qobs, 3)); reconcile the input-file list with the four
declared kinematic sets (copy lvm_* and gaia_* or drop the two extra sets —
decision flagged for the user at review).

## 4. Acceptance criteria

1. All existing 38 unit tests pass unchanged except where behavior is
   deliberately extended; new tests green.
2. Partial-free triaxiality: no invalid (p, q) pair survives Sobol
   projection or GP-phase constraints for any free subset (property test).
3. Warm-start: restarting with BayesOptGenerator in a populated
   output_directory trains the GP on all historical valid rows, logs it,
   skips warm-up, and clips/warns on out-of-range rows.
4. Batch integrity: post-snap batches contain unique non-ml cells and
   full batch_size models.
5. V3 comparison reproduces-or-beats the June baseline at equal or lower
   model budget.
6. Smoke test passes on a clean env with pinned deps.
7. Production YAML parses and passes a dummy-mode end-to-end run with
   SplitModelIterator and capped weight concurrency.

## 5. Risks

- BoTorch API drift (SingleTaskVariationalGP / fit_gpytorch_mll import
  paths) — mitigated by H4 pin + smoke test as first implementation step.
- Rejection sampling efficiency for R2 if the feasible set is small relative
  to the box (u near its window edge) — cap attempts, fall back to projected
  Sobol fill.
- Trust-region shrinkage trapping the search if the landscape is multimodal
  — R4 off by default; TR growth on failure streaks; global criteria still
  monitored.
- Warm-start poisoning if historical rows used different parameter ranges —
  H2 clipping + warning keeps such rows from dominating; documented.

## 6. References

- Li 2026, arXiv:2608.00841 (SALE)
- El Gammal et al. 2023, JCAP 10 021 (GPry)
- Eriksson et al. 2019, arXiv:1910.01739 (TuRBO)
- Leclercq 2018, PhRvD 98 063511 (BOLFI)
- Birky & Barnes 2026, arXiv:2603.18259 (ALABI)
- Tran et al. 2020, arXiv:2003.09436 (aphBO-2GP-3B)
- Zhang et al. 2022, IEEE TCAD (MACE)
- Barnabè & Koopmans 2007, ApJ 666 726
- Lamprecht et al. 2026, A&A 706 A373
- den Brok et al. 2021, MNRAS 508 4786
