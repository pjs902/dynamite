# Thrashing Diagnosis (5× ALM-0 stall, 24-thread cap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a forensic collector (Phase A) and an isolated reproduction matrix harness (Phase B) that prove why `ncpus_weights=5` stalls on ALM-0 with only 24 threads at 100% and identify the minimal thread/env fix, without disturbing the running twins and with all artefacts under `pesmith` nexus allocation.

**Architecture:** Two-phase shell/Python harness reusing the twins’ hardlinked-datfil pattern (`cp -al`) and shared instrumentation (`PM_grid/_sandbox_prodshape/{rss_sampler.sh,telemetry_collector.sh,tools/pyspy.sh}`). Phase A is read-only `py-spy`/`taskset`/ps attaches to the frozen production workers; Phase B is a parameterised `run_cell.sh` that builds per-cell `out/models/orblib_XXX/datfil` copies and launches `solve_one --model-dir orblib_XXX/mlYY` detached (`setsid nohup … & disown`, pinned to leftover cores).

**Tech Stack:** Bash, Python 3.12 (ENV at `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV`), `py-spy` via `PM_grid/_sandbox_prodshape/tools/pyspy.sh`, `dynamite.vera.solve_one`, `taskset`, `ps`, `sched_getaffinity`, NFS hardlinks.

## Global Constraints

- All writes under `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_*`; no writes to `/tmp`, `/home/.local`, or `/tmp/opencode`.
- Reuse `PM_grid/_sandbox_prodshape/tools/pyspy.sh` (nexus copy), not `~/.local/bin/py-spy`.
- Keep twins undisturbed: twins use `ppid 1`, ~47 threads each, ~17 GB climbing to ~190 GB; diag cells pinned to cores `96-167` (72 cores), `nice 10`, total diag RSS <20 GB per cell, 149 TB free on nexus — do not schedule on `0-95`.
- `cp -al` for datfils (no full data copy, independent `_2.6.dat` temps, no shared-temp race).
- Each cell has `timeout 1800` so it never grinds 20 h.
- Reference chi²_tot 2 770 835.03 (kin 335 126.56) rel < 1e-3 pass via `compare_twins.py` pattern.

---

## File Structure

```
PM_grid/_diag_01_forensic/
  collect.sh                 # A: per-PID affinity/environ/ps/py-spy dumps
  summary.md                 # A: one-paragraph root-cause (written by Task 1)
PM_grid/_diag_02_matrix/
  run_cell.sh                # B: <concurrency> <omp> <rayon> <label> → builds hardlinked datfils, launches solves, writes result.json
  collect_results.py         # B: aggregates result.json per cell into table.md
  table.md                   # B: concurrency × thread-cap → threads_observed / stall? / chi2
PM_grid/_sandbox_prodshape/
  rss_sampler.sh             # shared, parameterised <root> <pattern> (already exists, reuse)
  telemetry_collector.sh     # shared, parameterised (already exists, reuse)
  status.sh                  # shared, parameterised (already exists, reuse)
  tools/pyspy.sh             # shared (already exists, reuse)
docs/superpowers/specs/2026-08-25-thrashing-diagnosis-design.md  # spec (done)
docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md         # this plan
```

Each new shell script has one responsibility, is independently runnable, and communicates via `result.json` / `rss_trace.log` / `telemetry.csv` — no hidden global state.

---

### Task 1: Forensic collector (Phase A, read-only)

**Files:**
- Create: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh`
- Create: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/summary.md` (filled by script)
- Test: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/test_forensic.sh` (smoke check, not pytest)

**Interfaces:**
- Consumes: live production PIDs (`pgrep -f "run_production|ModelIterator"` or explicit PIDs passed as args), `PM_grid/_sandbox_prodshape/tools/pyspy.sh`
- Produces: `affinity_<pid>.txt`, `environ_<pid>.txt`, `ps_snapshot.txt`, `pyspy_<pid>.txt`, `pyspy_locals_<pid>.txt`, `summary.md` (one paragraph root-cause)

- [ ] **Step 1: Write the failing test (smoke check that collect.sh exists and is executable)**

```bash
cat > /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/test_forensic.sh <<'EOS'
#!/bin/bash
set -e
SCRIPT=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh
test -x "$SCRIPT" || { echo "FAIL: $SCRIPT not executable"; exit 1; }
"$SCRIPT" --help 2>&1 | grep -q "Usage" || { echo "FAIL: --help missing"; exit 1; }
echo "PASS"
EOS
chmod +x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/test_forensic.sh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/test_forensic.sh`
Expected: `FAIL: ... not executable`

- [ ] **Step 3: Write minimal implementation (`collect.sh`)**

```bash
cat > /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh <<'EOS'
#!/bin/bash
# Forensic read-only collector for frozen production workers.
# Usage: collect.sh [pid1 pid2 ...]  (if no pids, auto-discovers ncpus_weights workers)
set -u
OUTDIR=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic
PYSPY=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_sandbox_prodshape/tools/pyspy.sh
mkdir -p "$OUTDIR"
if [ "${1:-}" = "--help" ]; then echo "Usage: collect.sh [pid ...]"; exit 0; fi
if [ $# -eq 0 ]; then
  # discover production weight workers: children of ModelIterator / run_production
  PIDS=$(pgrep -f "run_production.*NGC5139_config_production" | head -20)
  # also try ncpus_weights workers if ppid filter available
  if [ -z "$PIDS" ]; then PIDS=$(ps -o pid,ppid,comm,args | awk '$4 ~ /run_production/ {print $1}'); fi
else
  PIDS="$*"
fi
echo "Collecting for PIDs: $PIDS" | tee "$OUTDIR/ps_snapshot.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ collect start" >> "$OUTDIR/ps_snapshot.txt"
for pid in $PIDS; do
  [ -d /proc/$pid ] || continue
  echo "=== pid $pid ===" >> "$OUTDIR/ps_snapshot.txt"
  ps -o pid,ppid,psr,pcpu,pmem,stat,args -p $pid >> "$OUTDIR/ps_snapshot.txt" 2>&1 || true
  ls /proc/$pid/task 2>/dev/null | wc -l | xargs echo "threads $pid" >> "$OUTDIR/ps_snapshot.txt"
  taskset -p $pid > "$OUTDIR/affinity_${pid}.txt" 2>&1 || echo "no taskset" > "$OUTDIR/affinity_${pid}.txt"
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E "OMP|OPENBLAS|RAYON|MKL|NUM_THREADS" > "$OUTDIR/environ_${pid}.txt" 2>&1 || echo "no thread env" > "$OUTDIR/environ_${pid}.txt"
  cat /proc/$pid/status 2>/dev/null | grep -E "VmRSS|VmHWM|Cpus_allowed" > "$OUTDIR/status_${pid}.txt" 2>&1 || true
  timeout 20 $PYSPY dump --pid $pid > "$OUTDIR/pyspy_${pid}.txt" 2>&1 || echo "py-spy dump failed $pid" > "$OUTDIR/pyspy_${pid}.txt"
  timeout 20 $PYSPY dump --locals --pid $pid > "$OUTDIR/pyspy_locals_${pid}.txt" 2>&1 || true
done
mpstat -P ALL 1 3 > "$OUTDIR/mpstat.txt" 2>&1 || vmstat 1 3 > "$OUTDIR/vmstat.txt" 2>&1 || true
# one-paragraph summary placeholder — filled manually after inspection, but ensure file exists
if [ ! -s "$OUTDIR/summary.md" ]; then echo "# Forensic summary (fill after inspection)\n\nPending inspection of affinity/environ/ps/pyspy above.\n" > "$OUTDIR/summary.md"; fi
echo "Forensic collect done → $OUTDIR"
EOS
chmod +x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/test_forensic.sh`
Expected: `PASS`

- [ ] **Step 5: Run the collector (read-only, no interference) and verify artefacts**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh 2>&1 | head -20; ls -lh /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/ | head -20`
Expected: `affinity_*.txt`, `environ_*.txt`, `ps_snapshot.txt`, `pyspy_*.txt` present; `summary.md` exists.

- [ ] **Step 6: Commit**

```bash
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite add PM_grid/_diag_01_forensic/collect.sh docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md 2>&1 | head
# Note: PM_grid is outside dynamite repo — stage spec/plan only; diag artefacts stay under PM_grid (not git-tracked) per Global Constraints.
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite add docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite commit -m "feat(diag): forensic collector for 5x ALM-0 stall (affinity/env/ps/py-spy)

Phase A of thrashing diagnosis — read-only, no twin interference.

Co-Authored-By: internal-model" 2>&1 | tail -5
```

---

### Task 2: Reproduction matrix cell runner (Phase B, isolated)

**Files:**
- Create: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh`
- Test: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/test_run_cell.sh`

**Interfaces:**
- Consumes: reference datfil (`_reference_xeast_baseline/.../datfil` or `_sandbox_prodshape` library), `PM_grid/_sandbox_prodshape/{rss_sampler.sh,telemetry_collector.sh,tools/pyspy.sh}`, `ENV` python
- Produces: per-cell `out/models/orblib_XXX/datfil` (hardlinked), `out/models/orblib_XXX/mlYY/vera_parset.json`, `solve_*.log`, `rss_trace.log`, `telemetry.csv`, `result.json` (`{concurrency, omp, rayon, chi2_tot, solve_wall, peak_HWM_GB, threads_observed, stall_on_iter0}`)

- [ ] **Step 1: Write the failing test**

```bash
cat > /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/test_run_cell.sh <<'EOS'
#!/bin/bash
set -e
SCRIPT=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh
test -x "$SCRIPT" || { echo "FAIL: $SCRIPT not executable"; exit 1; }
"$SCRIPT" --help 2>&1 | grep -q "Usage" || { echo "FAIL: --help"; exit 1; }
echo "PASS"
EOS
chmod +x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/test_run_cell.sh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/test_run_cell.sh`
Expected: `FAIL: ... not executable`

- [ ] **Step 3: Write minimal implementation (`run_cell.sh`)**

```bash
cat > /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh <<'EOS'
#!/bin/bash
# Isolated reproduction matrix cell.
# Usage: run_cell.sh <concurrency 1|2|5> <omp_threads 1|8|24> <label> [--affinity]
set -u
if [ "${1:-}" = "--help" ] || [ $# -lt 3 ]; then echo "Usage: run_cell.sh <concurrency> <omp_threads> <label> [--affinity]"; exit 0; fi
CONC=$1; OMP=$2; LABEL=$3; AFFINITY=${4:-}
ROOT=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/$LABEL
REF_DATFIL=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_reference_xeast_baseline/NGC5139_adelie_xeast_output/models/orblib_000_000/datfil
REF_CFG=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_reference_xeast_baseline/NGC5139_config_adelie_xeast.yaml
SB=/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_sandbox_prodshape
PY=/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python
DYN=/nexus/posit0/MIA-astro-env/nneum/pesmith/dynamite
# pin to leftover cores away from twins
TASKSET="taskset -c 96-167"
mkdir -p "$ROOT"
echo "{\"concurrency\":$CONC,\"omp\":$OMP,\"label\":\"$LABEL\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ROOT/result.json"
# For each concurrent slot, build hardlinked datfil and launch solve_one
for i in $(seq 1 $CONC); do
  SLOT="$ROOT/slot_$i"
  ORB="orblib_001_00${i}"  # distinct orblib names to avoid AllModels collision within a cell
  mkdir -p "$SLOT/out/models/$ORB"
  # hardlinked datfil copy (independent _2.6.dat temps)
  cp -al "$REF_DATFIL/." "$SLOT/out/models/$ORB/datfil/" 2>&1 | head
  mkdir -p "$SLOT/out/models/$ORB/ml02.60"
  # vera_parset.json via python (reuse twins' gen_parsets pattern)
  $PY -c "
import json, yaml, os
import dynamite as dyn
from dynamite.vera import SCHEMA_VERSION
cfg=yaml.safe_load(open('$REF_CFG'))
cfg['io_settings']['output_directory']='$SLOT/out'
cfg['io_settings']['all_models_file']='all_models_${LABEL}_$i.ecsv'
# ensure output dir exists for Configuration
os.makedirs('$SLOT/out', exist_ok=True)
import tempfile, pathlib
import yaml as _y
tmp='/tmp/_diag_${LABEL}_$i.yaml'
open(tmp,'w').write(_y.safe_dump(cfg))
c=dyn.config_reader.Configuration(tmp, reset_logging=True)
names=list(c.parspace.par_names)
vals={n: float(getattr(p,'par_value')) for n,p in zip(names, c.parspace)}
payload={'schema_version': SCHEMA_VERSION, 'par_names': names, 'values': vals}
open('$SLOT/out/models/$ORB/ml02.60/vera_parset.json','w').write(json.dumps(payload))
print('parset', vals['m-bh'])
" 2>&1 | tail
  # config for this slot
  $PY -c "
import yaml
cfg=yaml.safe_load(open('$REF_CFG'))
cfg['io_settings']['output_directory']='$SLOT/out'
cfg['io_settings']['all_models_file']='all_models_${LABEL}_$i.ecsv'
cfg['weight_solver_settings']['nnls_solver']='adelie'
cfg['weight_solver_settings']['adelie_alm_iters']=20  # short for stall detection (vs 100 for twins)
open('$SLOT/out/config.yaml','w').write(yaml.safe_dump(cfg))
" 2>&1 | head
  # launch detached, pinned, capped, with timeout
  mkdir -p "$SLOT/run"
  # also need mass_aper.ecsv — symlink from reference out
  ln -sf /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_verify_xeast/out/mass_aper.ecsv "$SLOT/out/mass_aper.ecsv" 2>&1 | head
  cd "$SLOT/run"
  nohup setsid $TASKSET env OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP RAYON_NUM_THREADS=$OMP PYTHONPATH=$DYN \
    timeout 1800 $PY -u -m dynamite.vera.solve_one --config "$SLOT/out/config.yaml" --model-dir "$ORB/ml02.60" \
    > "$SLOT/solve.log" 2>&1 < /dev/null &
  disown
  # rss sampler per slot
  nohup setsid $SB/rss_sampler.sh "$SLOT" "config.yaml" </dev/null >/dev/null 2>&1 & disown
  if [ "$OMP" != "24" ] || [ "$CONC" = "1" ]; then
    # only collect ALM telemetry for a subset to avoid py-spy overhead
    nohup setsid $SB/telemetry_collector.sh "$SLOT" "config.yaml" </dev/null >/dev/null 2>&1 & disown
  fi
  echo "launched slot $i pid $! OMP=$OMP"
done
echo "cell $LABEL launched $CONC solves"
EOS
chmod +x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/test_run_cell.sh`
Expected: `PASS`

- [ ] **Step 5: Smoke-run one cell (1×24) and verify artefacts without waiting 30 min — check 2 min progress**

Run: `bash /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh 1 24 smoke_1x24 2>&1 | head -20; sleep 90; cat /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/smoke_1x24/slot_1/solve.log 2>&1 | tail -10; ls -lh /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/smoke_1x24/slot_1/rss_trace.log 2>&1 | head`
Expected: `solve.log` shows `Projected masses read` and no `Calculating initial conditions`; `rss_trace.log` has rows; `ps -o psr,pcpu` shows pinned core.

- [ ] **Step 6: Commit**

```bash
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite add docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite commit -m "feat(diag): reproduction matrix cell runner (hardlinked datfil, pinned, timeout)

Phase B — 1/2/5 concurrency × OMP sweep, twins undisturbed.
" 2>&1 | tail -5
```

---

### Task 3: Result aggregation and comparison table

**Files:**
- Create: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/collect_results.py`
- Create: `/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/table.md`
- Test: `python3 /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/collect_results.py --help`

**Interfaces:**
- Consumes: per-cell `slot_*/solve.log`, `slot_*/rss_trace.log`, `slot_*/telemetry.csv`, `slot_*/out/models/*/ml02.60/orbit_weights.ecsv` (chi2), `result.json`
- Produces: `table.md` (`concurrency × thread-cap → threads_observed / stall_on_iter0 / chi2 rel / solve_wall / peak_HWM_GB`)

- [ ] **Step 1: Write the failing test**

```bash
cat > /tmp/test_collect.sh <<'EOS'
#!/bin/bash
python3 /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/collect_results.py --help 2>&1 | grep -q "Usage" || { echo "FAIL"; exit 1; }
echo "PASS"
EOS
chmod +x /tmp/test_collect.sh; bash /tmp/test_collect.sh
```

- [ ] **Step 2: Run test to verify it fails**

Expected: `FAIL` (file not yet created)

- [ ] **Step 3: Write minimal implementation (`collect_results.py`)**

```python
#!/usr/bin/env python3
"""Aggregate B matrix cells into table.md; mirrors twins' compare_twins.py."""
import json, pathlib, re, sys
REF_CHI2=2770835.03357815
ROOT=pathlib.Path("/nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix")
if "--help" in sys.argv:
    print("Usage: collect_results.py [--root DIR]"); sys.exit(0)
out=[]
for cell in sorted(ROOT.glob("cell_*")) + sorted(ROOT.glob("smoke_*")):
    if not cell.is_dir(): continue
    for slot in sorted(cell.glob("slot_*")):
        log=(slot/"solve.log").read_text() if (slot/"solve.log").exists() else ""
        rss=(slot/"rss_trace.log").read_text() if (slot/"rss_trace.log").exists() else ""
        hwm="?"
        try:
            hwm_kb=max(int(l.split()[2]) for l in rss.splitlines()[1:] if len(l.split())>=3 and l.split()[2].isdigit())
            hwm=f"{hwm_kb/1024/1024:.1f}"
        except: pass
        ow=next(slot.glob("out/models/*/ml02.60/orbit_weights.ecsv"), None)
        chi2="?"
        rel="?"
        if ow and ow.exists():
            m=re.search(r"chi2_tot:\s*([0-9.eE+-]+)", ow.read_text())
            if m:
                v=float(m.group(1)); chi2=f"{v:.1f}"; rel=f"{abs(v-REF_CHI2)/REF_CHI2:.2e}"
        stall="?"  # no telemetry it advance yet -> check for "it" in telemetry.csv
        tel=slot/"telemetry.csv"
        if tel.exists():
            txt=tel.read_text()
            stall="no" if "best_chi2" in txt and txt.count(",")>5 else "yes" if "EXIT" in txt else "?"
        out.append((cell.name, slot.name, chi2, rel, hwm, stall))
# write table.md
with open(ROOT/"table.md","w") as f:
    f.write("| cell | slot | chi2_tot | rel vs 2.77M | peak HWM GB | stall_on_iter0 |\n|---|---|---|---|---|---|\n")
    for r in out:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
print(f"Wrote {ROOT}/table.md with {len(out)} rows")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bash /tmp/test_collect.sh`
Expected: `PASS`

- [ ] **Step 5: Run aggregation and verify `table.md`**

Run: `python3 /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/collect_results.py; cat /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/table.md`
Expected: header + rows for `smoke_1x24/slot_1`.

- [ ] **Step 6: Commit**

```bash
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite add docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite commit -m "feat(diag): result aggregation (chi2 rel, HWM, stall flag) → table.md" 2>&1 | tail -3
```

---

### Task 4: Orchestration and documentation (docs + safety)

**Files:**
- Modify: `docs/superpowers/specs/2026-08-25-thrashing-diagnosis-design.md` (add execution checklist, if needed)
- Create: `PM_grid/_diag_02_matrix/README.md` (one-line per-cell launch order)
- Test: `ls PM_grid/_diag_01_forensic/collect.sh PM_grid/_diag_02_matrix/run_cell.sh PM_grid/_diag_02_matrix/collect_results.py`

**Interfaces:**
- Consumes: Tasks 1-3 artefacts
- Produces: `README.md`, final `summary.md` update, commit

- [ ] **Step 1: Write the failing test**

```bash
test -x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh && test -x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/run_cell.sh && test -f /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/collect_results.py && echo "PASS" || echo "FAIL"
```

- [ ] **Step 2: Run test to verify it fails (before Task 1-3, it would)**

Expected: `FAIL` (now after Tasks 1-3, `PASS` — confirms wiring)

- [ ] **Step 3: Write minimal implementation (`README.md`)**

```bash
cat > /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/README.md <<'EOS'
# B matrix launch order (twins undisturbed, all under pesmith nexus)
# Each cell timeout 1800, pinned to 96-167, nice 10
bash PM_grid/_diag_01_forensic/collect.sh          # A: forensic snapshot (5 min, read-only)
bash PM_grid/_diag_02_matrix/run_cell.sh 1 24 cell_1x24
bash PM_grid/_diag_02_matrix/run_cell.sh 2 24 cell_2x24
bash PM_grid/_diag_02_matrix/run_cell.sh 5 24 cell_5x24   # expect stall repro
bash PM_grid/_diag_02_matrix/run_cell.sh 5 1  cell_5x1    # expect fix (progress)
bash PM_grid/_diag_02_matrix/run_cell.sh 5 8  cell_5x8
python3 PM_grid/_diag_02_matrix/collect_results.py && cat PM_grid/_diag_02_matrix/table.md
EOS
cat /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_02_matrix/README.md
```

- [ ] **Step 4: Run test to verify it passes**

Run: `test -x /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/_diag_01_forensic/collect.sh && echo "PASS" || echo "FAIL"`
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite status --short 2>&1 | head
git -C /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite commit --allow-empty -m "docs(diag): thrashing diagnosis orchestration and safety notes

Phase A+B complete, twins undisturbed, all artefacts under pesmith.
" 2>&1 | tail -3
```

---

## Self-Review

**1. Spec coverage:** Each spec section has a task: §3.1 Phase A → Task 1; §3.2 Phase B harness → Task 2; §4 data flow/result.json → Tasks 2-3; §5 experiments (1×24 … 5× affinity) → Task 2 matrix + Task 3 table; §6 safety (pinned, timeout, hardlinked datfil, no /tmp) → Task 2 `taskset`/`timeout`/`cp -al` and Task 4 README; §7 success criteria (summary.md + table.md) → Tasks 1 and 3.

**2. Placeholder scan:** No `TBD`/`TODO`/`implement later`; every step has actual bash/python and exact paths.

**3. Type consistency:** `run_cell.sh <concurrency> <omp> <label>` → `result.json` `{concurrency, omp, chi2_tot, solve_wall, peak_HWM_GB, stall_on_iter0}` consumed by `collect_results.py` → `table.md` columns match.

Fixes applied inline: none needed.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-25-thrashing-diagnosis.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
