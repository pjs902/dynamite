# Thrashing / Stall Diagnosis — ModelIterator pathos Grid (5× ALM-0, 24 threads)

**Date:** 2026-08-25  
**Status:** Design approved, awaiting implementation plan  
**Scope:** A (forensic read-only) + B (isolated reproduction matrix). No vera cluster.  
**Constraint:** Twins (`_verify_xeast` adelie alm100 + `_verify_scipy`) keep running undisturbed. All artefacts under `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_*`. No writes to `/tmp` or `~/.local`.

## 1. Context

### 1.1 Observed failure
- Production grid launched via `run_production.py NGC5139_config_production.yaml` with `multiprocessing_settings: total_cores 90, ncpus 45, orblib_chunks auto, ncpus_weights 5, ncpus_weights_maxtasksperchild 3`.
- Symptom: all 5 concurrent `NNLS/adelie` weight solves grind on **ALM iteration 0** for ~20 h, never advancing to iter 1. At most **24 threads total** system-wide at 100% (not 5×24 = 120), per user report. No `orbit_weights.ecsv` finishes for any of the 30 already-integrated `orblib_*` libraries.
- Earlier triage hypothesised `5 × (rayon 24 + OpenBLAS 24 + bvls 24) ≈ 360 threads` oversubscription, but the 24-thread total cap contradicts that and suggests a global thread-pool limit or a serialization lock.

### 1.2 Reference and twins
- Reference: `_reference_xeast_baseline/NGC5139_adelie_xeast_output` (single model `orblib_000_000/ml02.60`, chi²_tot 2 770 835.03, kinchi2 335 126.56) — no wall/RSS recorded (ModelIterator run).
- Probe: `_sandbox_prodshape` (prod-shape k=5 model, same library `nE=30 nI2=25 nI3=20`, adelie f32 streamed fused, 200 ALM) — solo: `t_orblib 1 168.1 s`, `solve_wall 18 980.7 s`, `total 20 148.8 s`, chi² 3 091 321.97, peak HWM ~200 GB. Tiny A/B concurrency tests showed parity, ruling out plain code rot.
- Twins (currently running, independent `ppid 1`): `_verify_xeast` (adelie alm100, 24 threads, hardlinked datfil) and `_verify_scipy` (scipy, same parset) reusing `_reference_xeast_baseline/.../datfil` via hardlinked copies to avoid `orblib_qgrid_2.6.dat` race. Both correctly short-circuit integration (`tube_box_done`) and are in `read_vel_histograms → _decompress` at 02:26.

### 1.3 Code pointers
- `dynamite/dynamite/weight_solvers.py:17` — saves/restores `sched_getaffinity` around `import adelie.solver` (rayon mis-detects core count and narrows affinity to 1 core); `weight_solvers.py:1410 n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))`, passed to adelie BVLS at `1436`.
- `dynamite/dynamite/model_iterator.py` — pathos pool for `ncpus_weights` weight solves.
- `dynamite/dynamite/orblib.py:182` — `tube_box_done` short-circuit; `770 get_orbit_ics` (Fortran `orbitstart`); `1411 _decompress` (`bunzip2 -c src > dest` unconditionally); `1952 read_orbit_base` writes `datfil/*_qgrid_2.6.dat` (ml-suffixed temp, races if shared datfil).
- `dynamite/dynamite/vera/task_model.py:76` / `dynamite/vera/solve_one.py` — correct `ml02.60` model-dir handling (`directory` vs `directory_noml`), adopted for twins.

## 2. Goals / Non-goals

**Goals**
- Explain why 5× ALM-0 stalls with only 24 threads saturated.
- Identify the minimal thread/env fix that restores 5× progress with expected per-worker CPU (no code changes yet).
- Keep twins undisturbed; all diagnostic writes under `pesmith` nexus allocation; reproducible evidence (`result.json` per cell).

**Non-goals**
- No vera/Slurm work.
- No production-grid restart or code patch in this phase.
- No `/tmp/opencode` or `~/.local` artefacts (reuse `PM_grid/_sandbox_prodshape/tools/pyspy.sh`).

## 3. Architecture

### 3.1 Phase A — Forensic read-only (5 min, zero compute)
Attach to the 5 live-but-frozen production workers (still children of `ModelIterator`, `ncpus_weights=5`). No new processes beyond `py-spy` and `ps`.

**Collect per PID:**
- `sched_getaffinity` / `taskset -p <pid>` — allowed cores (did rayon import narrow to 24)?
- `tr '\0' '\n' < /proc/<pid>/environ | grep -E OMP|OPENBLAS|RAYON|MKL` — inherited thread caps.
- `ps -o pid,ppid,psr,pcpu,stat,cmd` and `ls /proc/<pid>/task | wc -l` — core spread and thread count; system `vmstat` / `mpstat` to confirm 24-core saturation.
- `py-spy dump [--locals] --pid <pid>` — distinguish `_decompress` vs `solve_adelie_alm` (it/lam/gap) vs `communicate` serialisation.
- `cat /proc/<pid>/status | grep -E VmRSS|VmHWM` — per-worker RSS vs probe’s 190 GB.

**Decision:** 24-thread cap = global env (e.g., `RAYON_NUM_THREADS=24` or `OMP=24` inherited) vs serialization lock (4 workers at 0% u, 1 at 2400%).

### 3.2 Phase B — Isolated reproduction matrix (nexus-only, capped)
For each matrix cell, build `PM_grid/_diag_<label>/out/models/orblib_XXX/datfil` as **hardlinked copy** (`cp -al` from `NGC5139_adelie_xeast_output/models/orblib_000_000/datfil` or `NGC5139_production_output/models/orblib_001_XXX/datfil` for prod-shape). Add `mlXX.YY/vera_parset.json` and `out/mass_aper.ecsv` reuse. Launch `solve_one --model-dir orblib_XXX/mlYY` via `setsid nohup … < /dev/null > solve_*.log 2>&1 & disown` (so `ppid 1`). Pin to leftover cores (`taskset -c 96-167`, `nice 10`, `OMP=OPENBLAS=RAYON` capped) away from twins (twins ~47 threads each on `0-95`).

Sample with shared harness from `_sandbox_prodshape`: `rss_sampler.sh <root> <pattern>` (60 s) and `telemetry_collector.sh <root> <pattern>` (150 s, `py-spy` for `solve_adelie_alm`). Each cell has a `timeout 1800` so it never grinds 20 h.

### 3.3 Components
- **Forensic collector:** `PM_grid/_diag_01_forensic/collect.sh` — loops over `pgrep -f "ncpus_weights"` or known production PIDs, writes `affinity.json`, `environ.txt`, `ps_snapshot.txt`, `pyspy_*.txt`.
- **Matrix harness:** `PM_grid/_diag_02_matrix/run_cell.sh <concurrency> <omp> <rayon> <label>` — builds hardlinked datfils, writes `result.json` (`chi2_tot`, `solve_wall`, `peak_HWM_GB`, `threads_observed`, `stall_on_iter0: bool`, `telemetry_head`).
- **Shared instrumentation:** `PM_grid/_sandbox_prodshape/rss_sampler.sh`, `telemetry_collector.sh`, `status.sh`, `tools/pyspy.sh` — parameterised `<root> <pattern>`, already fixed to nexus `pyspy`.
- **Comparator:** `PM_grid/_verify_xeast/compare_twins.py` pattern reused for B cells vs reference 2 770 835.03 (rel < 1e-3 pass).

## 4. Data flow & file layout

```
PM_grid/_diag_01_forensic/
  affinity_<pid>.txt  environ_<pid>.txt  ps_snapshot.txt
  pyspy_<pid>.txt  summary.md

PM_grid/_diag_02_matrix/
  cell_1x24/   # 1 solve, OMP=RAYON=24 (baseline, should finish)
  cell_2x24/   # 2 concurrent, OMP=24 (repro 24-thread cap?)
  cell_5x24/   # 5 concurrent, OMP=24 (repro stall)
  cell_5x1/    # 5 concurrent, OMP=RAYON=OPENBLAS=1 (fix probe)
  cell_5x8/    # 5 concurrent, =8 (saturation check)
  cell_5x_affinity/ # 5× with per-worker taskset chunks
    out/models/orblib_XXX/datfil/      # hardlinked, independent _2.6.dat
    out/models/orblib_XXX/ml2.60/vera_parset.json
    solve_*.log  rss_trace.log  telemetry.csv  result.json
```

All reads from nexus, all writes to `pesmith` subdirs. No shared `_2.6.dat` race (separate datfils). `result.json` is the only cross-cell aggregation.

## 5. Experiments

| cell | concurrency | thread caps | expect | proves |
|------|-------------|-------------|--------|--------|
| 1×24 | 1 solve, OMP=RAYON=24 | completes, telemetry `it 0→1→2` in ~2 min, ~1.1 K s `t_orblib` | healthy baseline |
| 2×24 | 2 concurrent, 24 | reproduces 24-thread total cap if global (both share 24) vs 48 if per-worker | global-cap hypothesis |
| 5×24 | 5 concurrent, 24 | **stall repro:** 5× stuck on ALM-0, 24 threads total @100%, RSS ~17 GB each, no `it` advance | matches production symptom |
| 5×1 | 5 concurrent, 1 | **fix probe:** each worker 1 thread, total 5 threads, makes progress (ALM it advances) | thread-pool oversubscription / cap is causal |
| 5×8 | 5 concurrent, 8 | saturation check: 40 threads total, progress vs thrashing | optimal cap |
| 5× affinity | 5 concurrent, 24 but per-worker `taskset -c` chunks (0-23,24-47,…) | if A showed affinity narrowed to 24 cores, this restores spread | affinity vs env var |

Each cell timeout 30 min. Control: `orblib_chunks auto` vs fixed to rule out integration-phase chunking vs weight-solve threading.

## 6. Safety / Non-interference

- Twins use `ppid 1`, `47 threads` each, `~17.6 GB` HWM (climbing to ~190 GB), implicitly on cores `0-95`. Diag cells pinned to `96-167` (72 cores left on 192-core / 1.4 TB node) + `nice 10`, total diag RSS capped <20 GB per cell, 149 TB free on nexus.
- `timeout` per cell prevents 20 h grind.
- No `bunzip2` to `/tmp` — all decompress writes to hardlinked `datfil/*_qgrid_2.6.dat` in each cell’s own datfil (no shared-temp race).
- Reuse `PM_grid/_sandbox_prodshape/tools/pyspy.sh` (nexus), not `~/.local/bin/py-spy`; no `/tmp/opencode` writes.

## 7. Success criteria

- A writes `summary.md`: one paragraph root-cause (e.g., “RAYON_NUM_THREADS=24 inherited by all 5 workers from `ModelIterator` parent, capping total rayon threads to 24” or “pathos pool serialization lock at `…` holding 4 workers in `wait`”).
- B writes `result.json` per cell and a table `concurrency × thread-cap → threads_observed / stall? / chi2` showing the minimal fix that restores 5× progress with ~100% per-worker CPU and chi² within 1e-3 of reference.
- Both artefacts committed under `docs/superpowers/specs/` and `PM_grid/_diag_*`; twins remain undisturbed.

## 8. Open questions

- Exact env inheritance for production workers: does `ModelIterator` set `RAYON_NUM_THREADS` from `total_cores` or `ncpus_weights`? A will read `/proc/<pid>/environ` to confirm.
- Whether the production `ncpus_weights_maxtasksperchild: 3` recycling masks or worsens the stall (B can toggle).

## 9. Out of scope

- Any fix patch to `dynamite` code.
- Any Slurm/vera work.
- Any restart of the production grid in this phase.
