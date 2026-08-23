# VERA Evaluator Service for DYNAMITE Schwarzschild Campaigns

**Date:** 2026-08-23
**Status:** Draft for review
**Branch:** `slurm`
**Scope:** ω Centauri production campaign; architecture generalizes to future systems

---

## 1. Objective

Run the ω Cen Schwarzschild grid campaign on MPCDF's VERA cluster using a
Slurm-native evaluator service, replacing phased in-node iteration with a
strategist/executor split that supports classic GridWalk today and GP-driven
Bayesian optimization (`bayesopt` branch) tomorrow.

### Non-goals

- No new in-node ModelIterator. Phased execution inside one process is retired;
  Slurm is the scheduler.
- No GPU solving path. Adelie is CPU-only (verified: no CUDA in source or
  upstream); A100 nodes are out of scope except as potential GP-surrogate
  hardware for the BO agent.
- No changes to solver numerics. The adelie/fused-X/streamed-read stack ships
  as merged on `master` and is treated as a fixed, validated engine.

## 2. Measured facts this design rests on

From the local fat-node campaign (see `dev_notes/rss_fused_sprint_log.md`,
`weight_solve_rss_profile.md`, OOM post-mortem 2026-08-23):

| fact | value |
|---|---|
| solve peak RSS, float32 + streamed reads | ~190 GiB/worker |
| solve concurrency on 1416 GB node | RAM-capped at 5–6 |
| failure mode at 6 workers | global OOM kills, silent respawn-hang, 0 solves in 8 h |
| per-solve wall (production shape, cap-30) | ~106 min; full-run budget 2–3 h |
| library integration | ~45-min class, embarrassingly parallel, ~6 single-threaded procs per library |

From VERA reconnaissance (2026-08-23):

| fact | value |
|---|---|
| association / QoS | account `mia`; QoS `normal` has **no** MaxJobs/MaxSubmit/TRES caps |
| fairshare | `LevelFS ≈ 1730` (near-virgin credit; top-of-queue initially, decays with use) |
| partitions | `p.vera`: exclusive, 250 GB, 72 c, MaxTime 2 d · `p.large`: shared, 500 GB, 72 c, MaxNodes 32, MaxTime 2 d · `p.huge`: 2 TB, 1 d (unused by this design) |
| requestable memory | `RealMemory` fully allocable: 200 GB requests fit either class |
| MaxArraySize | 1001 — an entire iteration's solves fit one array job |
| billing TRES | CPU-core-hours only (no `TRESBillingWeights` anywhere); memory unpriced |
| storage | `/vera/ptmp/gc` 5 TB exclusive, **NO BACKUPS**; `/u` home 400k-file quota |

Design consequence: one f32 solve fits alone on any standard node ⇒ memory
safety is structural (cgroup-enforced `--mem`), not scheduler-managed. This is
what retires both the OOM problem and the memory-governor idea.

## 3. Architecture

```
        ┌────────────────────────────────────────────────────┐
        │ DRIVER DAEMON (login node, single process)         │
        │   SCAN → RECONCILE → SUBMIT → WAIT → OBSERVE       │
        │   single writer of all_models.ecsv                 │
        └───────┬───────────────────────────▲────────────────┘
     sbatch     │                           │ results (χ² triples)
        ▼       │                           │
┌───────────────┴───┐            ┌──────────┴─────────────┐
│ STRATEGIST        │            │ EXECUTOR (artifact-     │
│ GridWalkClassic / │◀──────────▶│ grounded, Slurm-hosted) │
│ MicroBatchWalk /  │ propose/   │ integration arrays      │
│ GP-BO adapter     │ observe    │ solve arrays            │
└───────────────────┘            └────────────────────────┘
```

Principles:

1. **Filesystem is the database.** State = `all_models.ecsv` + directory
   artifacts (`datfil/tube_box_done` sentinels, weight files) exactly as the
   local campaign already uses them. Every operation is idempotent against
   these artifacts; crash/preemption recovery = rescan.
2. **Single writer.** Only the driver mutates `all_models.ecsv`. Array tasks
   write their own private outputs (orbit libraries, weight files); the driver
   notices completion and records χ². Table writes are atomic
   (temp-file + `os.replace`).
3. **Strategist decides *what*, executor decides *where/how*.** The two talk
   only through the interface in §4.

## 4. Strategist interface contract (schema v1 — freeze on merge)

```python
class Strategist(Protocol):
    def start(self, ctx: SystemContext) -> None
    def propose(self, max_batch: int) -> list[Proposal]
    def observe(self, results: Iterable[Result]) -> None      # streaming
    def quorum_pending(self) -> int    # outstanding results wanted before
                                       # the next proposal round; 0 = ask me now
    def exhausted(self) -> bool        # stopping criteria met

Proposal = {"proposal_id": str,        # sha256(parset canonical json)[:16]
            "parset":   dict[str, float]}   # free-param values, bounds-clipped
Result   = {"proposal_id": str,
            "model_dir": str,
            "status": "done" | "failed",
            "chi2": float | None, "kinchi2": float | None,
            "kinmapchi2": float | None}
```

Rules:

- Proposals are **validated at intake** by the executor before any resource is
  spent: parameter bounds clip, then triaxial deprojection feasibility
  (`triax_pqu2tpp` pass). Invalid proposals return as failed Results without
  entering the queue. BO strategists will propose geometrically impossible
  (q,p,u) that GridWalk's clipping never produces; this gate is mandatory.
- `parset` contains only free-parameter values; fixed params are implied by
  config. Canonical-JSON hashing makes proposal identity environment-neutral,
  so a campaign can straddle the local box and VERA.
- Schema version string travels alongside every persisted proposal record;
  breaking changes bump to v2 with a translation adapter.

Three implementations planned:

| phase | strategist | quorum semantics |
|---|---|---|
| 1 | `GridWalkClassic` — port of current generator, byte-compatible proposals | whole current walk-step solved |
| 2 | `MicroBatchWalk` — re-center when ≥X% of step solved (X default 80%) | fraction |
| 3 | GP-BO: `BayesOptGenerator` (`fork/bayesopt`) behind a thin adapter | 0 (continuous) |

Phase 3 note for the BO team: `observe()` fires per completed model, not in
batches — the GP updates immediately; staleness is structurally zero.

### 4.1 TableDriven adapter pattern (BayesOptGenerator integration)

The bayesopt modernization design
(`docs/superpowers/specs/2026-08-22-bayesopt-modernization-design.md`)
implements BO *inside* DYNAMITE: `BayesOptGenerator.generate(current_models)`
trains its GP from the shared `all_models.ecsv` rows
(`extract_gp_training_data`, `all_done` mask) and appends proposals as new
rows. Observation flows through the table, not callbacks.

Consequence for schema v1: a strategist may satisfy `observe()` by reading
table rows; streaming callbacks are an optimization, not a requirement. The
Phase-3 adapter is therefore thin:

| Strategist v1 | BayesOptGenerator |
|---|---|
| `propose(n)` | instantiate generator against driver's live AllModels; call `generate()`; map appended rows → Proposals by proposal hash |
| `observe(results)` | no-op (generator re-reads table at next generate) |
| `quorum_pending()` | 0 — R3 CorrectCounter and stopping flags are evaluated per completed model |
| `exhausted()` | OR of generator status flags incl. `gp_predictions_accurate` |

Two design gifts arrive with it: **H2 warm-start** means the VERA campaign can
begin as a continuation of the local GridWalk run's table — every historical
row becomes GP training data for free; and their explicitly **deferred**
"asynchronous batch BO across the model pool" is precisely what this service
provides — the deferral was an artifact of single-node phased execution, which
VERA dissolves. Their tuning study's `batch_size=4` winner matches GPry's
min(dims, workers) guideline at d=4 free parameters, and sizes the solve-array
width nicely (K ≥ 4 keeps GP cadence tight).

## 5. Executor components

### 5.1 Work classifier (pure function)

Input: config + output tree. Output: per-model classification —

| class | criterion | action |
|---|---|---|
| `to_integrate` | row exists, dir lacks `tube_box_done`, not in-flight | submit integration |
| `to_solve` | sentinel present, weight file absent, not in-flight | submit solve |
| `done` | weight file present (χ² read from its meta) | record in table |
| `invalid` | failed intake validation | record failed Result |

This replaces `update_model_table`'s reconstruction logic with the identical
artifact rules, minus deletion: incomplete rows are never deleted on VERA,
just left unclassified-until-artifacts-appear. (The local deletion semantics
exist to prune dead weight in interactive sessions; on VERA the driver simply
resubmits.)

### 5.2 Integration array runner

- One array task = one exclusive `p.vera` node = one work package of ~12
  libraries (72 cores ÷ 6 procs/library), executed by GNU parallel.
- Each library integrates exactly as `LegacyOrbitLibrary` does today
  (chunked orbitstart/cmd_box_orbs, claim/stale-retake locking intact).
- Task writes a per-package completion stamp listing finished dirs; driver
  trusts only per-library sentinels, stamps are advisory.

### 5.3 Solve array runner

- One array task = one model = one `solve_one.py MODEL_DIR` invocation.
  Reuses `NNLS WeightSolver` directly (config + parset → weights file +
  χ² meta). No `ModelIterator` in the task path.
- Memory safety is delegated to Slurm (`--mem=200000`); no in-task governor.
- Thread budget 24 (measured solo sweet spot; node-exclusive bandwidth share
  is generous at ≤2 co-located solves).

### 5.4 Driver daemon

Login-node Python process (~200–300 lines around `squeue`/`sbatch`/`sshare`
subprocess calls + §5.1 classifier + strategist object). Loop:

```
SCAN      classify all rows; read in-flight set (own accounting, cross-checked
          against squeue every cycle)
OBSERVE   for each newly-done model: append Result → strategist.observe();
          atomically update table row
RECONCILE resubmit work whose Slurm job vanished without producing its
          artifact (attempt counter per model; >3 attempts → park row,
          alert, continue)
SUBMIT    if strategist.quorum_pending() > 0 and idle capacity: submit arrays
          for to_integrate/to_solve (array width %K adaptive, §7)
PROPOSE   if quorum_pending() == 0 and not exhausted():
          propose() → validate intake → create rows+dirs → loop
GUARDS    daily: mmlsquota check (fail-loud at 80%), nightly tar of tables to
          /u; weekly: sshare LevelFS read → adjust K ceiling
```

POLL_INTERVAL 300 s. All knobs (partitions, K bounds, thresholds, paths) come
from an env/config file so MPCDF policy shifts need zero code edits.

## 6. Slurm job specifications

```bash
# ---- integration wave (one task = one node = ~12 libraries) ----
#SBATCH --partition=p.vera --exclusive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --account=mia
#SBATCH --job-name=ocen-int

# ---- solve wave (one task = one model) ----
#SBATCH --partition=p.large
#SBATCH --mem=200000 --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --array=0-N%K                # N = backlog size; K adaptive (below)
#SBATCH --account=mia
#SBATCH --job-name=ocen-solve
```

Adaptive width policy: start K=16; raise toward 24 while LevelFS > 10 and
mean queue wait < 30 min; lower toward 4 as LevelFS approaches 1 or queue
depth grows. Width affects speed only, never correctness (idempotent
resubmission).

Density note: p.large tasks pack ≤2 per 500 GB node (2×200 GB + OS);
Slurm co-location happens naturally. If p.large throughput disappoints,
fallback profile swaps partition to `p.vera` (exclusive, 1 solve/node) via
config — no code change.

## 7. Environment build checklist (VERA)

Execution-time checklist; exact module names resolved during the build and
recorded here afterward:

1. `module av gcc` → pick gfortran ≥ 9 for `legacy_fortran` (only compiled
   piece; galahad/NNLS path permanently retired).
2. Python via conda, **env on scratch, not home** (400k-file quota):
   `conda create -p /vera/ptmp/gc/mia/pesmith/envs/dynamite python=3.12`
   Alternative if MPCDF's curated stack suffices: compiler module + venv on
   ptmp. Decision rule: prefer wheels-over-modules; pin numpy first, then
   install `adelie` wheel compatible with it, then scipy/astropy/pathos/etc.
   Phase 3 adds the BO stack per the bayesopt H4 pins: torch (CPU build),
   botorch ≥ 0.18.1, gpytorch — large wheels, one more reason the env lives
   on scratch.
3. `pip install --no-deps ./dynamite` (repo checkout lives on scratch too).
4. `make -C legacy_fortran all` with loaded gcc; verify `orbitstart` runs a
   1-orbit smoke case.
5. Record chosen versions into `runs/prod/ENV_FREEZE.txt`.

## 8. Storage layout & data policy

```
/vera/ptmp/gc/mia/pesmith/oCen/
├── dynamite/                  repo @ slurm branch
├── envs/dynamite/             conda env
├── runs/prod/                 config yaml, logs, NGC5139_production_output/
│   ├── models/orblib_*_*/     datfil RETAINED (user requirement: in-depth
│   │                          analysis later; est. 0.25–0.75 TB total)
│   └── all_models.ecsv
└── archive/                   per-iteration tar snapshots of ecsv tables
/u → $HOME/oCen_backup/         nightly ~MB-scale tar: tables + weight files
```

Guards: nightly `mmlsquota` check fails loud at 80% of 5 TB; backup tar runs
after each OBSERVE batch (tables are tiny). ptmp carries no backup warranty —
libraries are recomputable from config + sentinels, so losing them costs
compute, not science.

## 9. Failure semantics

| event | behavior |
|---|---|
| array task killed/timeout/preempted | artifact absent ⇒ RECONCILE resubmits; attempt counter parks chronic offenders after 3 tries |
| worker OOM | structurally prevented by `--mem` cgroups |
| driver crash/restart | pure restart: SCAN rebuilds world from filesystem + squeue; zero duplicate work (submission idempotence keyed by proposal hash) |
| torn `all_models.ecsv` | impossible mid-write (atomic replace); corrupt-at-rest caught by parse-check at startup → restore from nightly tar |
| NFS visibility lag on sentinels | driver requires artifact mtime age > 60 s before acting on presence; absence always re-checked next cycle |
| adelie threaded nondeterminism | acceptance gate is scientific equivalence (χ² rel. diff < 1e-9 vs reference), never bitwise — per local gate revision |

## 10. Rollout phases & acceptance gates

| phase | content | gate to pass |
|---|---|---|
| 0 | local recovery wave finishes; clean per-solve wall times recorded | ≥5 solves completing without dmesg kills; numbers into §7 freeze file |
| 1 | VERA env build + smoke campaign: `GridWalkClassic`, `n_max_mods ≈ 20`, ≤ 8 nodes | end-to-end in ≤ 1 day: 20 models integrated + solved; table complete; one solve's χ² within 1e-6 rel. of local same-parset reference (tolerance sized for cross-BLAS-kernel drift; local runs measured up to 9e-7 across code-path variants); driver kill/restart resumes with zero duplicates |
| 2 | production campaign on `GridWalkClassic`; then swap in `MicroBatchWalk` | iteration wall-time ≤ 36 h sustained over 3 iterations; quota/tar guards verified in dry-run |
| 3 | GP-BO live: merge or vendor `fork/bayesopt`, run `BayesOptGenerator` behind the §4.1 adapter (`batch_size=4` per their T1/T3 tuning); warm-start from the Phase-2 table via their H2 mechanism; A/B vs GridWalk at equal model budget | BO matches or beats GridWalk best-χ² trajectory per model count; R3 stopping flag observed firing naturally |

Local fat box remains the dev/debug rig and overflow capacity throughout; the
shared artifact format lets a campaign straddle both.

## 11. Risks & mitigations

| risk | mitigation |
|---|---|
| fairshare decay slows queues mid-campaign | adaptive K; correctness independent of width; local overflow box |
| MPCDF policy/QoS shifts | every limit is a config knob; nothing hard-coded |
| NFS metadata load from large arrays | tasks touch artifacts once; driver polls its own submission ledger + sacct, not recursive finds |
| `bayesopt` branch schema drift | Proposal/Result JSON schema frozen here at v1; adapters isolate changes. The generator is table-coupled (§4.1), so upstream drift surfaces as adapter errors, not silent science corruption |
| scratch loss (no backups) | libraries recomputable; tables mirrored to `/u` continuously |

## 12. Open items (execution-time)

- Exact VERA module names (gcc/python stacks) — commands issued, to be filled
  into §7 during Phase 1.
- BO message transport between the external agent process and the driver
  (in-process import vs file queue) — Phase 3 kickoff decision, does not
  affect schema v1.
- Empirical K ceiling from observed queue behavior — tuned online, bounded by
  §6 policy.
