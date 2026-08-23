# VERA Evaluator Service — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Slurm-native evaluator service (driver daemon, work classifier, GridWalkClassic proposer, array-job runners) and prove it end-to-end with a 20-model smoke campaign on VERA.

**Architecture:** Filesystem-as-database: all state lives in `all_models.ecsv` plus directory artifacts (`datfil/tube_box_done`, weight files). A login-node driver classifies models, submits Slurm array jobs, observes completions, and feeds a table-driven GridWalk proposer. Single writer of the ecsv table (atomic temp-file + `os.replace`). Spec: `docs/superpowers/specs/2026-08-23-vera-evaluator-service-design.md`.

**Tech Stack:** Python 3.12, stdlib only for the vera package (subprocess, json, hashlib, pathlib, tempfile, argparse); astropy Table via existing dynamite deps; Slurm CLI (`sbatch`/`squeue`/`sacct`/`sshare`) behind an injectable runner; pytest.

## Global Constraints

- Branch: all work on `slurm`. Never touch `dynamite/weight_solvers.py`, `dynamite/model_iterator.py`, `dynamite/parameter_space.py`, `dynamite/config_reader.py` (bayesopt owns those seams; zero-conflict rule).
- Schema v1 frozen per spec §4: `Proposal{proposal_id, parset}`, `Result{proposal_id, model_dir, status, chi2, kinchi2, kinmapchi2}`; `proposal_id = sha256(canonical_json)[:16]`.
- Single writer: only `VeraDriver` mutates `all_models.ecsv`; writes are tmp-file + `os.replace`.
- Artifact trust rule: presence requires file age > 60 s (NFS lag guard); absence is always re-checked next cycle.
- Attempt counter: park a model after 3 failed submissions (status logged, never silently retried forever).
- Slurm constants: solve jobs `--partition=p.large --mem=200000 --cpus-per-task=24 --time=06:00:00 --account=mia`; integration jobs `--partition=p.vera --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --time=08:00:00 --account=mia`; array throttle `%K` adaptive, start 16.
- Integration packing density: 12 libraries per node (72 cores ÷ 6 procs/library).
- `POLL_INTERVAL=300` s; `LevelFS` thresholds: raise K toward 24 when >10, lower toward 4 as →1.
- No new third-party dependencies. Tests run headless (no Slurm, no Fortran) via injected runners and tmp-path artifact trees.
- Test command everywhere below: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest <file> -v` run from the dynamite repo root.

---

## Execution tracks

- **Core track** — Tasks 1–7: the executor and `GridWalkProposer`. Local-only tests; nothing here needs Slurm or VERA.
- **Deployment track** — Tasks 8–10: smoke config, runbook, env script, cluster acceptance run (D1). May start any time after Task 7; never blocks Track C.
- **Capability track** — Tasks 11–12: `MicroBatchWalk` and the BayesOpt adapter (spec phases C2/C3). Start immediately after Task 5; gated exclusively by local tests and synthetic/minirun landscapes, per spec §10's principle that clusters are backends, not milestones.

---

### Task 1: Proposal/Result schema v1 + intake validation

**Files:**
- Create: `dynamite/vera/__init__.py`
- Create: `dynamite/vera/proposal.py`
- Create: `dev_tests/test_vera_proposal.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION = 1`; `canonical_hash(parset: dict) -> str` (16-hex); `Proposal` and `Result` dataclasses with `.to_dict()`; `validate_parset(parset: dict, config) -> tuple[dict, list[str]]` returning `(clipped_parset, violations)` — empty violations means admissible.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_proposal.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.proposal import (
    SCHEMA_VERSION, canonical_hash, Proposal, Result, validate_parset,
)

FAKE_BOUNDS = {
    "bh.m":  {"lo": 3.90, "hi": 4.78},
    "ml":    {"lo": 1.0,  "hi": 6.0},
    "q":     {"lo": 0.05, "hi": 0.72},   # hi already capped < qobs*u
    "p":     {"lo": 0.50, "hi": 0.99},
    "u":     {"lo": 0.95, "hi": 0.9999},
}
QOBS = 0.724
U_FIXED = 0.9999


def _par(q=0.46, p=0.90, u=None, bh=4.342, ml=2.6):
    d = {"bh.m": bh, "ml": ml, "q": q, "p": p}
    if u is not None:
        d["u"] = u
    return d


class FakeShape:
    """Minimal stand-in for TriaxialVisibleComponent.triax_pqu2tpp inputs."""
    qobs = QOBS


def test_schema_version_frozen():
    assert SCHEMA_VERSION == 1


def test_canonical_hash_is_key_order_insensitive():
    a = canonical_hash({"q": 0.5, "p": 0.9})
    b = canonical_hash({"p": 0.9, "q": 0.5})
    assert a == b and len(a) == 16


def test_valid_fiducial_passes():
    clipped, violations = validate_parset(_par(), FAKE_BOUNDS, qobs=QOBS,
                                          u_fixed=U_FIXED)
    assert violations == []
    assert clipped["q"] == 0.46


def test_out_of_bounds_clips_not_rejects():
    clipped, violations = validate_parset(_par(bh=9.0), FAKE_BOUNDS,
                                          qobs=QOBS, u_fixed=U_FIXED)
    assert violations == []
    assert clipped["bh.m"] == 4.78


def test_q_above_qobs_times_u_is_violation():
    # q=0.74 > 0.9999*0.724 -> geometrically impossible, must NOT be clipped
    _, violations = validate_parset(_par(q=0.74), FAKE_BOUNDS,
                                    qobs=QOBS, u_fixed=U_FIXED)
    assert any("q" in v for v in violations)


def test_p_less_than_q_is_violation():
    _, violations = validate_parset(_par(q=0.8, p=0.7), FAKE_BOUNDS,
                                    qobs=QOBS, u_fixed=U_FIXED)
    assert any("p" in v for v in violations)


def test_roundtrip_dataclasses():
    pr = Proposal(proposal_id=canonical_hash(_par()), parset=_par())
    r = Result(proposal_id=pr.proposal_id, model_dir="orblib_001_000/ml02.40",
               status="done", chi2=2770837.5, kinchi2=1.0, kinmapchi2=2.0)
    assert Proposal.from_dict(pr.to_dict()) == pr
    assert Result.from_dict(r.to_dict()) == r
    assert r.to_dict()["schema_version"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dynamite.vera'`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/__init__.py
"""VERA evaluator-service deployment tooling (spec 2026-08-23)."""
SCHEMA_VERSION = 1
```

```python
# dynamite/vera/proposal.py
"""Proposal/Result schema v1 and intake validation (spec section 4)."""
import dataclasses
import hashlib
import json
import math

from . import SCHEMA_VERSION


def canonical_hash(parset):
    payload = json.dumps(parset, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parset: dict

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION,
                "proposal_id": self.proposal_id,
                "parset": dict(self.parset)}

    @staticmethod
    def from_dict(d):
        assert d["schema_version"] == SCHEMA_VERSION
        return Proposal(proposal_id=d["proposal_id"], parset=dict(d["parset"]))


@dataclasses.dataclass(frozen=True)
class Result:
    proposal_id: str
    model_dir: str
    status: str
    chi2: float = None
    kinchi2: float = None
    kinmapchi2: float = None

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION,
                **dataclasses.asdict(self)}

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in d.items() if k != "schema_version"}
        return Result(**d)


def validate_parset(parset, bounds, qobs, u_fixed=None):
    """Clip parameters into bounds; reject geometrically impossible shapes.

    Returns (clipped_parset, violations). Bounds clipping silently repairs;
    triaxial feasibility violations (q > u*qobs, p < q, u outside window)
    are hard rejections — the caller must not spend compute on them.
    """
    clipped, violations = {}, []

    def _num(x):
        return isinstance(x, (int, float)) and math.isfinite(x)

    for name, val in parset.items():
        if not _num(val):
            violations.append(f"{name}: non-finite value {val!r}")
            continue
        lo, hi = bounds.get(name, {}).get("lo"), bounds.get(name, {}).get("hi")
        if lo is not None and val < lo:
            val = lo
        if hi is not None and val > hi:
            val = hi
        clipped[name] = float(val)

    q = clipped.get("q")
    p = clipped.get("p")
    u = clipped.get("u", u_fixed)

    if q is not None and u is not None and qobs is not None \
            and q > u * qobs * (1 - 1e-6):
        violations.append(f"q={q} exceeds axis limit u*qobs={u*qobs:.4f}")
    if q is not None and p is not None and p < q:
        violations.append(f"p={p} < q={q}: oblate-equivalent shape invalid")
    if q is not None and q <= 0.05:
        violations.append(f"q={q} outside physical range")
    return clipped, violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposal.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/__init__.py dynamite/vera/proposal.py dev_tests/test_vera_proposal.py
git commit -m "feat(vera): proposal/result schema v1 with triaxial intake validation"
```

---

### Task 2: Artifact-state work classifier

**Files:**
- Create: `dynamite/vera/classifier.py`
- Create: `dev_tests/test_vera_classifier.py`

**Interfaces:**
- Consumes: nothing (pure filesystem logic).
- Produces: `ModelState` enum (`PENDING_INTEGRATION`, `INTEGRATING`, `TO_SOLVE`, `SOLVED`, `FAILED`, `PARKED`); `classify(model_dir: Path, attempts: int, now_ts: float, min_age_s: float = 60.0) -> ModelState`. Rules: `datfil/tube_box_done` present (age > min_age_s) → TO_SOLVE unless weights file present → SOLVED; sentinel absent + recent activity (< min_age_s) → INTEGRATING; else PENDING_INTEGRATION; attempts ≥ 3 → PARKED overrides all but SOLVED; FAILED reserved for explicit intake rejection.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_classifier.py
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.classifier import ModelState, classify

NOW = 1_000_000.0


def mk(root, sentinel=False, weights=False, fresh=False):
    d = root / "orblib_001_000" / "ml02.60"
    (d / "datfil").mkdir(parents=True, exist_ok=True)
    if sentinel:
        f = d / "datfil" / "tube_box_done"
        f.write_text("")
        old = NOW - (0 if fresh else 3600)
        os.utime(f, (old, old))
    if weights:
        w = d / "weights_ecsf.dat"          # dyn.constants.weight_file name
        w.write_text("# placeholder meta\n")
        old = NOW - (0 if fresh else 3600)
        os.utime(w, (old, old))
    return d


def test_empty_dir_is_pending(tmp_path):
    d = mk(tmp_path)
    assert classify(d, attempts=0, now_ts=NOW) is ModelState.PENDING_INTEGRATION


def test_fresh_sentinel_counts_as_still_integrating(tmp_path):
    d = mk(tmp_path, sentinel=True, fresh=True)   # NFS lag guard
    assert classify(d, attempts=0, now_ts=NOW) is ModelState.INTEGRATING


def test_aged_sentinel_is_to_solve(tmp_path):
    d = mk(tmp_path, sentinel=True)
    assert classify(d, attempts=0, now_ts=NOW) is ModelState.TO_SOLVE


def test_weights_present_is_solved(tmp_path):
    d = mk(tmp_path, sentinel=True, weights=True)
    assert classify(d, attempts=0, now_ts=NOW) is ModelState.SOLVED


def test_three_attempts_park_unless_solved(tmp_path):
    d = mk(tmp_path)
    assert classify(d, attempts=3, now_ts=NOW) is ModelState.PARKED
    d2 = mk(tmp_path / "x", sentinel=True, weights=True)
    assert classify(d2, attempts=3, now_ts=NOW) is ModelState.SOLVED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_classifier.py -v`
Expected: FAIL, `No module named 'dynamite.vera.classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/classifier.py
"""Artifact-grounded model classification (spec section 5.1)."""
import enum
import os


class ModelState(enum.Enum):
    PENDING_INTEGRATION = "pending_integration"
    INTEGRATING = "integrating"
    TO_SOLVE = "to_solve"
    SOLVED = "solved"
    FAILED = "failed"
    PARKED = "parked"


SENTINEL = "datfil/tube_box_done"
WEIGHTS = "weight_ecsf.dat"      # keep in lockstep with dyn.constants.weight_file
MIN_AGE_S_DEFAULT = 60.0         # NFS metadata-lag guard
ATTEMPT_LIMIT = 3


def _age(path, now_ts):
    try:
        return now_ts - os.stat(path).st_mtime
    except OSError:
        return None


def classify(model_dir, attempts, now_ts, min_age_s=MIN_AGE_S_DEFAULT):
    sent = os.path.join(model_dir, SENTINEL)
    wght = os.path.join(model_dir, WEIGHTS)

    solved = _age(wght, now_ts)
    if solved is not None and solved > min_age_s:
        return ModelState.SOLVED

    if attempts >= ATTEMPT_LIMIT:
        return ModelState.PARKED

    sent_age = _age(sent, now_ts)
    if sent_age is not None:
        return ModelState.TO_SOLVE if sent_age > min_age_s \
            else ModelState.INTEGRATING

    # any recently touched file means a job is actively writing here
    for root, _, files in os.walk(model_dir):
        for f in files:
            a = _age(os.path.join(root, f), now_ts)
            if a is not None and a <= min_age_s:
                return ModelState.INTEGRATING
    return ModelState.PENDING_INTEGRATION
```

Note: verify the actual weight-file basename before finalizing — `grep -rn "weight_file" dynamite/constants.py` and use that exact string in `WEIGHTS` and in the test fixture.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_classifier.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/classifier.py dev_tests/test_vera_classifier.py
git commit -m "feat(vera): artifact-grounded model state classifier with NFS age guard"
```

---

### Task 3: Integration work-pack grouping

**Files:**
- Create: `dynamite/vera/pack_integration.py`
- Create: `dev_tests/test_vera_pack.py`

**Interfaces:**
- Produces: `pack_libraries(model_dirs: list[str], procs_per_lib: int = 6, cores: int = 72) -> list[list[str]]` — order-preserving groups of size `cores // procs_per_lib`.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_pack.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamite.vera.pack_integration import pack_libraries

def test_groups_of_twelve_default():
    dirs = [f"m{i}" for i in range(29)]
    packs = pack_libraries(dirs)
    assert [len(p) for p in packs] == [12, 12, 5]

def test_order_preserved():
    dirs = [f"m{i}" for i in range(15)]
    flat = [m for p in pack_libraries(dirs) for m in p]
    assert flat == dirs

def test_single_library_node_budget():
    assert pack_libraries(["only"], procs_per_lib=72, cores=72) == [["only"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_pack.py -v`
Expected: FAIL, no module `dynamite.vera.pack_integration`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/pack_integration.py
"""Group libraries into node-sized work packages (spec section 5.2)."""
PROCS_PER_LIB_DEFAULT = 6   # orblib_chunks(3) x orbit families(2), 1 thread each
CORES_PER_VERA_NODE = 72


def pack_libraries(model_dirs, procs_per_lib=PROCS_PER_LIB_DEFAULT,
                   cores=CORES_PER_VERA_NODE):
    size = max(1, cores // procs_per_lib)
    return [model_dirs[i:i + size] for i in range(0, len(model_dirs), size)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_pack.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/pack_integration.py dev_tests/test_vera_pack.py
git commit -m "feat(vera): library work-pack grouping at 12-per-node density"
```

---

### Task 4: Slurm command layer (injectable runner)

**Files:**
- Create: `dynamite/vera/slurm.py`
- Create: `dev_tests/test_vera_slurm.py`

**Interfaces:**
- Produces: `SlurmError(Exception)`; `class Runner` protocol `(argv: list[str]) -> str`; `RealRunner()`; `submit_array(runner, job_spec: dict, items: list[str]) -> int` (parses `Submitted batch job N`); `running_job_ids(runner) -> set[int]`; `levelfs(runner, user) -> float | None`; `build_solve_job_spec(array_limit_k)` and `build_integration_job_spec()` returning dicts with partition/mem/cpus/time/account keys matching Global Constraints exactly; both embed `--array=0-N%K` where N=len(items)-1.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_slurm.py
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamite.vera.slurm import (
    submit_array, running_job_ids, levelfs, build_solve_job_spec,
    build_integration_job_spec, SlurmError,
)

class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
    def __call__(self, argv):
        self.calls.append(argv)
        key = argv[0]
        if key not in self.responses:
            raise SlurmError(f"unexpected command {argv}")
        r = self.responses[key]
        if isinstance(r, Exception):
            raise r
        return r


def test_submit_array_parses_job_id():
    fr = FakeRunner({"sbatch": "Submitted batch job 424242\n"})
    jid = submit_array(fr, build_solve_job_spec(k=16), ["a", "b", "c"])
    assert jid == 424242
    argv = fr.calls[0]
    assert argv[0] == "sbatch" and "--parsable" not in argv
    joined = " ".join(argv)
    assert "--array=0-2%16" in joined and "--mem=200000" in joined
    assert "--cpus-per-task=24" in joined and "--account=mia" in joined
    assert "--partition=p.large" in joined


def test_submit_failure_raises():
    fr = FakeRunner({"sbatch": RuntimeError("sbatch: error")})
    with pytest.raises(SlurmError):
        submit_array(fr, build_solve_job_spec(k=16), ["a"])


def test_running_jobs_parsed():
    fr = FakeRunner({"squeue":
                     "424242\n424999\n" })
    assert running_job_ids(fr) == {424242, 424999}


def test_levelfs_float_or_none():
    good = FakeRunner({"sshare": "pesmith|mia|9604|0.000002|1.7307e+03\n"})
    assert abs(levelfs(good, "pesmith") - 1730.7) < 0.01
    bad = FakeRunner({"sshare": ""})
    assert levelfs(bad, "pesmith") is None


def test_integration_spec_shape():
    spec = build_integration_job_spec()
    assert spec["partition"] == "p.vera" and "--exclusive" in " ".join(spec["extra"])
    assert spec["time"] == "08:00:00"


def test_solve_spec_throttle_cap():
    spec = build_solve_job_spec(k=16, n_items=200)
    assert "--array=0-199%16" in " ".join(spec["extra"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_slurm.py -v`
Expected: FAIL, no module `dynamite.vera.slurm`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/slurm.py
"""Thin, testable wrappers around the Slurm CLI (spec section 6)."""
import subprocess

ACCOUNT = "mia"
SOLVE_SPEC = dict(partition="p.large", mem_mb=200000, cpus=24, time="06:00:00")
INT_SPEC = dict(partition="p.vera", mem_mb=None, cpus=72, time="08:00:00")


class SlurmError(RuntimeError):
    pass


class RealRunner:
    __call__ = staticmethod(
        lambda argv: subprocess.run(
            argv, capture_output=True, text=True, check=False).stdout)


def _base_flags(spec):
    flags = [f"--partition={spec['partition']}",
             f"--time={spec['time']}",
             f"--account={ACCOUNT}",
             f"--nodes=1", "--ntasks=1",
             f"--cpus-per-task={spec['cpus']}"]
    if spec.get("mem_mb"):
        flags.append(f"--mem={spec['mem_mb']}")
    if spec.get("extra"):
        flags.extend(spec["extra"])
    return flags


def build_solve_job_spec(k, n_items):
    return {**SOLVE_SPEC,
            "extra": [f"--array=0-{max(0, n_items - 1)}%{k}",
                      "--job-name=ocen-solve"]}


def build_integration_job_spec():
    return {**INT_SPEC, "extra": ["--exclusive", "--job-name=ocen-int"]}


def submit_array(runner, job_spec, items):
    argv = ["sbatch"] + _base_flags(job_spec) + ["wrap_script_placeholder"]
    out = runner(argv)
    for token in out.split():
        if token.isdigit():
            return int(token)
    raise SlurmError(f"unparseable sbatch output: {out!r}")


def running_job_ids(runner):
    out = runner(["squeue", "-u", "$USER", "-h", "-o", "%i"])
    ids = set()
    for line in out.splitlines():
        tok = line.strip().split("_")[0]
        if tok.isdigit():
            ids.add(int(tok))
    return ids


def levelfs(runner, user):
    out = runner(["sshare", "-U", "--noheader",
                  "--format=User,Account,RawUsage,NormUsage,LevelFS"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0] == user and len(parts) >= 5:
            try:
                return float(parts[4])
            except ValueError:
                return None
    return None
```

Note: the `"wrap_script_placeholder"` element is replaced in Task 6 by the real script path + item-list plumbing; its position in `argv` is what later tasks rely on. If your Slurm rejects `%K` above MaxArraySize semantics, clamp `k = min(k, 1001)` here.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_slurm.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/slurm.py dev_tests/test_vera_slurm.py
git commit -m "feat(vera): injectable slurm CLI layer with pinned job specifications"
```

---

### Task 5: GridWalkClassic proposer (TableDriven adapter)

**Files:**
- Create: `dynamite/vera/proposer_gridwalk.py`
- Create: `dev_tests/test_vera_proposer_gridwalk.py`

**Interfaces:**
- Consumes: dynamite `Configuration` (real object; the driver owns its lifecycle).
- Produces: `class GridWalkProposer: __init__(self, config)`; `.propose(max_batch) -> list[Proposal]` (calls `config.par_generator.generate(current_models=config.all_models)` — wait, generators hang off `parameter_space`; see Step 3 note — then maps newly appended rows to Proposals); `.quorum_pending() -> int`; `.exhausted() -> bool` (n_max_mods reached); `.observe(results)` (no-op; table is the channel).

Implementation note for Step 3: obtain the generator exactly as `ModelIterator.__init__` does — read `dynamite/model_iterator.py:20-60` for the `generator_type` → class lookup (`globals()[...]` pattern over `parameter_space` exports) and mirror it with a direct import instead: `from dynamite.parameter_space import GridWalk as _Gen; gen = _Gen(config=config, par_generator=..., ...)`. Copy the construction arguments verbatim from `ModelIterator.__init__`; do not invent signatures.

- [ ] **Step 1: Write the failing test**

Uses the existing NGC6278 dummy-mode fixture style from `dev_tests/` (see `test_bayesopt_generator.py` for MockAllModels patterns; alternatively point at a tiny synthetic yaml). The essential assertions:

```python
# dev_tests/test_vera_proposer_gridwalk.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.proposal import canonical_hash


def test_propose_returns_new_rows_as_proposals(minimal_config):
    strat = make_proposer(minimal_config)          # helper builds Configuration
    props = strat.propose(max_batch=500)
    assert len(props) >= 1                            # iteration 0+1 lattice
    ids = {p.proposal_id for p in props}
    assert len(ids) == len(props)                     # unique hashes
    for p in props:
        assert "bh.m" in p.parset or "ml" in p.parset


def test_quorum_counts_unsolved_proposals(minimal_config):
    strat = make_proposer(minimal_config)
    props = strat.propose(max_batch=500)
    assert strat.quorum_pending() == len(props)
    # mark one solved directly in the table (driver's job in production)
    row = find_row_for(strat, props[0])
    strat.config.all_models.table["chi2"][row] = 1234.5
    strat.config.all_models.table["kinchi2"][row] = 1200.0
    strat.config.all_models.table["kinmapchi2"][row] = 1210.0
    strat.config.all_models.table["all_done"][row] = True
    assert strat.quorum_pending() == len(props) - 1
```

Helpers `make_proposer` / `find_row_for` live at the top of the test file: build the smallest working `Configuration` available in `dev_tests/` (reuse whichever yaml `test_vs_gridwalk.py` on the bayesopt branch uses — `git show fork/bayesopt:dev_tests/test_vs_gridwalk.py` — substituting `generator_type: GridWalk`), and locate rows by matching parset values column-by-column.

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposer_gridwalk.py -v`
Expected: FAIL, no module `dynamite.vera.proposer_gridwalk`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/proposer_gridwalk.py
"""TableDriven GridWalk adapter (spec sections 4 and 4.1)."""
from .proposal import Proposal


class GridWalkProposer:
    def __init__(self, config):
        self.config = config
        self._pending_ids = set()
        gen_type = config.settings.parameter_space_settings["generator_type"]
        assert gen_type == "GridWalk", f"expected GridWalk, got {gen_type}"
        import dynamite.parameter_space as ps
        self.generator = ps.GridWalk(config=config)

    def propose(self, max_batch=1000):
        table = self.config.all_models.table
        known = {(r["directory"], ) for r in table}
        self.generator.generate(current_models=self.config.all_models)
        proposals = []
        for i, row in enumerate(self.config.all_models.table):
            if (row["directory"], ) in known or row["directory"]:
                continue                      # pre-existing or already claimed
            parset = self._row_to_parset(i)
            pid = canonical_hash(parset)
            proposals.append(Proposal(proposal_id=pid, parset=parset))
            if len(proposals) >= max_batch:
                break
        self._pending_ids |= {p.proposal_id for p in proposals}
        return proposals

    def _row_to_parset(self, row_idx):
        mod = self.config.all_models.get_model_from_row(row_idx)
        return {p.name: float(p.par_value_raw
                              if hasattr(p, "par_value_raw") else p.raw_value)
                for p in mod.parameters}

    def observe(self, results):
        return None                          # table is the channel

    def quorum_pending(self):
        t = self.config.all_models.table
        solved = int(sum(bool(t["all_done"]) for _ in range(0)))  # placeholder
        solved = sum(1 for r in t if r["all_done"])
        return max(0, len(self._pending_ids) -
                   sum(1 for r in t if r["all_done"]))

    def exhausted(self):
        stop = self.config.settings.parameter_space_settings["stopping_criteria"]
        t = self.config.all_models.table
        return int(sum(1 for r in t if r["all_done"])) >= stop["n_max_mods"]
```

The draft above intentionally shows the two spots requiring care during implementation: (a) `_row_to_parset` must read whatever value columns the real table exposes — inspect one real row and pick `par_values` accessors accordingly; (b) `quorum_pending` must count only rows belonging to *this proposer's* proposals (hash-set membership via stored mapping `proposal_id -> row index`, maintained in `propose()`), not all-done rows globally — fix the placeholder arithmetic to subtract only mapped-and-done pairs.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposer_gridwalk.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/proposer_gridwalk.py dev_tests/test_vera_proposer_gridwalk.py
git commit -m "feat(vera): GridWalkClassic tabledriven proposer adapter"
```

---

### Task 6: Driver core loop with dry-run mode

**Files:**
- Create: `dynamite/vera/driver.py`
- Create: `dev_tests/test_vera_driver.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `class VeraDriver(config, proposer, runner, run_dir, poll_interval=300, k_start=16)`; `.scan() -> dict[str, ModelState]`; `.reconcile_and_submit(dry_run=False) -> int` (jobs submitted); `.observe_completions() -> int`; `.step(dry_run) -> bool` (False = proposer exhausted); `main(argv=None)` CLI with `--config --run-dir --dry-run --once`.
- Also replaces Task 4's placeholder: `wrap_script_placeholder` becomes two elements `[script_path] + [";".join(items)]`, and `solve_one.py` (Task 7) provides `script_path`.

- [ ] **Step 1: Write the failing test**

Scenario test on a synthetic output tree (tmp_path): three models — pending, built-but-unsolved, solved; a fake runner capturing sbatch calls; a stub proposer with canned proposals.

```python
# dev_tests/test_vera_driver.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.driver import VeraDriver
from dynamite.vera.classifier import ModelState


class RecordingRunner:
    def __init__(self):
        self.sbatch_calls = []
    def __call__(self, argv):
        if argv[0] == "sbatch":
            self.sbatch_calls.append(argv)
            return "Submitted batch job 1001\n"
        if argv[0] == "squeue":
            return ""
        if argv[0] == "sshare":
            return "pesmith|mia|0|0|9999\n"
        raise AssertionError(argv)


class StubProposer:
    def __init__(self, batches):
        self.batches = list(batches)
        self.seen_results = []
    def propose(self, max_batch=1000):
        return self.batches.pop(0) if self.batches else []
    def observe(self, results):
        self.seen_results.extend(results)
    def quorum_pending(self):
        return 0 if not self.batches else 1
    def exhausted(self):
        return not self.batches


def _mk_tree(tmp):
    # returns (config_like, dirs) with three model dirs in the three states;
    # uses the same fixture helpers as test_vera_classifier
    ...

def test_first_step_submits_integration_then_solve_wave(tmp_path):
    cfg, dirs = _mk_tree(tmp)
    rr = RecordingRunner()
    drv = VeraDriver(cfg, StubProposer([]), rr, run_dir=tmp)
    submitted = drv.reconcile_and_submit(dry_run=False)
    kinds = [c[1] for c in rr.sbatch_calls]           # job-name flag position
    assert submitted == 2                             # one int wave, one solve wave
    joined = " ".join(rr.sbatch_calls[1])
    assert "--array=" in joined                        # solve wave throttled


def test_second_run_is_idempotent(tmp_path):
    cfg, dirs = _mk_tree(tmp)
    rr1 = RecordingRunner()
    VeraDriver(cfg, StubProposer([]), rr1, run_dir=tmp).reconcile_and_submit()
    n_calls = len(rr1.sbatch_calls)
    rr2 = RecordingRunner()                            # fresh world view: same fs,
    n2 = VeraDriver(cfg, StubProposer([]), rr2,     # in-flight unknown -> but
                    run_dir=tmp).reconcile_and_submit() # ledger prevents dupes
    assert n2 == 0                                     # nothing new to do
```

The `_mk_tree` helper mirrors Task 2's `mk()` plus a minimal fake `config` exposing `all_models.table` (astropy Table with the standard columns) whose rows reference those dirs — copy column construction from `AllModels.make_empty_table` (read it first) so column names match production exactly.

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_driver.py -v`
Expected: FAIL, no module `dynamite.vera.driver`

- [ ] **Step 3: Write minimal implementation**

Core responsibilities in one file (~250 lines):

```python
# dynamite/vera/driver.py
"""VERA driver daemon: scan, reconcile, submit, observe (spec section 5.4)."""
import argparse, json, os, tempfile, time
from astropy.table import Table

from .classifier import ModelState, classify, ATTEMPT_LIMIT
from .pack_integration import pack_libraries
from .slurm import (RealRunner, submit_array, running_job_ids, levelfs,
                    build_solve_job_spec, build_integration_job_spec)
from .proposal import Result

ATTEMPTS_FILE = "vera_attempts.json"
LEDGER_FILE = "vera_inflight.json"
TABLE_REL = "NGC5139_production_output/all_models.ecsv"   # from io_settings


class VeraDriver:
    def __init__(self, config, proposer, runner=None, run_dir=".",
                 poll_interval=300, k_start=16):
        self.config = config
        self.proposer = proposer
        self.runner = runner or RealRunner()
        self.run_dir = os.path.abspath(run_dir)
        self.poll_interval = poll_interval
        self.k_start = k_start
        self.attempts = self._load_json(ATTEMPTS_FILE, {})
        self.inflight = self._load_json(LEDGER_FILE, {})   # kind -> [items]
        self.output_root = config.settings.io_settings["output_directory"]

    # ---------- persistence ----------
    def _load_json(self, name, default):
        p = os.path.join(self.run_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return default

    def _dump_json(self, name, obj):
        fd, tmp = tempfile.mkstemp(dir=self.run_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, os.path.join(self.run_dir, name))

    def _save_table_atomically(self):
        tbl = self.config.all_models.table
        fd, tmp = tempfile.mkstemp(suffix=".ecsv", dir=os.curdir)
        os.close(fd)
        tbl.write(tmp, format="ascii.ecsv", overwrite=True)
        os.replace(tmp, TABLE_REL)

    # ---------- pipeline phases ----------
    def scan(self):
        states = {}
        for i, row in enumerate(self.config.all_models.table):
            d = os.path.join(self.output_root, "models", row["directory"])
            states[row["directory"]] = classify(
                d, attempts=self.attempts.get(row["directory"], 0),
                now_ts=time.time())
        return states

    def reconcile_and_submit(self, dry_run=False):
        states = self.scan()
        live = running_job_ids(self.runner) if not dry_run else set()
        self.inflight = {kind: [it for it in items
                                if _jid_of(kind, it) in live]
                         for kind, items in self.inflight.items()}
        to_int = sorted(d for d, s in states.items()
                        if s is ModelState.PENDING_INTEGRATION)
        to_sol = sorted(d for d, s in states.items() if s is ModelState.TO_SOLVE)
        submitted = 0
        if to_int:
            submitted += self._submit_wave(
                "int", pack_libraries(to_int), dry_run)
        if to_sol:
            k = self._adaptive_k()
            submitted += self._submit_wave(
                "solve", [[d] for d in to_sol], dry_run, k=k)
        return submitted

    def _submit_wave(self, kind, packages, dry_run, k=None):
        new = [pkg for pkg in packages
               if pkg not in self.inflight.get(kind, [])]
        if not new:
            return 0
        if dry_run:
            print(f"[dry-run] would submit {len(new)} {kind} package(s)")
            return 0
        if kind == "int":
            spec, items = build_integration_job_spec(), \
                [";".join(p) for p in new]
        else:
            spec, items = build_solve_job_spec(k=k, n_items=sum(len(p) for p in new)), \
                [p[0] for p in new]
        jid = submit_array(self.runner, spec, items)
        self.inflight.setdefault(kind, []).extend(new)
        self._dump_json(LEDGER_FILE, self.inflight)
        return 1

    def _adaptive_k(self):
        lf = levelfs(self.runner, "pesmith")
        if lf is None:
            return self.k_start
        if lf > 10:
            return min(24, self.k_start + 4)
        if lf < 1.0:
            return max(4, self.k_start - 4)
        return self.k_start

    def observe_completions(self):
        states = self.scan()
        done_dirs = [d for d, s in states.items() if s is ModelState.SOLVED]
        results = []
        for d in done_dirs:
            if d not in self.proposer_tracked():
                continue
            chi2s = _read_weights_meta(os.path.join(
                self.output_root, "models", d))
            results.append(Result(proposal_id=self.dir_to_pid[d],
                                  model_dir=d, status="done", **chi2s))
        if results:
            self.proposer.observe(results)
            self._sync_table_rows(results)
        return len(results)

    def step(self, dry_run=False):
        self.observe_completions()
        self.reconcile_and_submit(dry_run=dry_run)
        if self.proposer.quorum_pending() == 0 \
                and not self.proposer.exhausted():
            for p in self.proposer.propose():
                self._create_model_entry(p)
        return not self.proposer.exhausted()

    def proposer_tracked(self):
        return getattr(self, "dir_to_pid", {})

    # helpers `_create_model_entry`, `_read_weights_meta`, `_sync_table_rows`,
    # `_jid_of`, static `main(argv)` CLI: implement following spec sections
    # 5.4 and 9 — _create_model_entry appends a row with empty directory then
    # assigns orblib_%03d_%03d/ml%.2f exactly as
    # ModelInnerIterator.assign_model_directories does (read it first);
    # _read_weights_meta parses the weights-file header meta
    # (chi2_tot/chi2_kin/chi2_kinmap) with ascii.read(...).meta;
    # _sync_table_rows writes chi2 columns + all_done=True + time_modified.
```

Implementation notes (binding): read `ModelInnerIterator.assign_model_directories` and mirror its naming/format precisely; read `dyn.constants.weight_file` for the exact basename used by `_read_weights_meta`; the ledger JSON is what makes the second `reconcile_and_submit` idempotent in the test above (in-flight packages are skipped even though their artifacts are still absent). `proposer_tracked`/`dir_to_pid`: maintain `self.dir_to_pid = {model_dir: proposal_id}` populated in `propose` handling inside `step()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_driver.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the whole vera suite together (regression)**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposal.py dev_tests/test_vera_classifier.py dev_tests/test_vera_pack.py dev_tests/test_vera_slurm.py dev_tests/test_vera_proposer_gridwalk.py dev_tests/test_vera_driver.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add dynamite/vera/driver.py dev_tests/test_vera_driver.py dynamite/vera/slurm.py
git commit -m "feat(vera): driver daemon core loop with idempotent submission ledger"
```

---

### Task 7: solve_one task entry point

**Files:**
- Create: `dynamite/vera/solve_one.py`
- Create: `dev_tests/test_vera_solve_one.py`

**Interfaces:**
- Consumes: dynamite `Configuration`, `Model`, `WeightSolver` — mirror exactly what `ModelInnerIterator.create_and_run_model` does in its weights pass (read `dynamite/model_iterator.py:565-635` before writing): construct `Model` from row, `get_orblib()` (sentinel short-circuit applies), `get_weights(orblib)`.
- Produces: `main(argv) -> int` CLI: `python -m dynamite.vera.solve_one --config CFG --model-dir orblib_001_007/ml02.80 [--dry-run]`; exit 0 on success (weight file present), 3 on solver failure. On success prints a single JSON line `{"model_dir": ..., "chi2_tot": ...}` for the driver's logs.

- [ ] **Step 1: Write the failing test**

Two tests only — this task's real validation is the VERA smoke (Phase-1 gate), so unit scope stays narrow:

```python
# dev_tests/test_vera_solve_one.py
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamite.vera.solve_one import main

def test_missing_model_dir_exits_nonzero(tmp_path, capsys):
    rc = main(["--config", "nope.yaml",
               "--model-dir", "orblib_999_999/ml02.60"])
    assert rc != 0

def test_dry_run_does_not_touch_config(tmp_path):
    # points at the smoke config shipped in Task 8; asserts exit path works
    # WITHOUT constructing Fortran inputs
    ...
```

Fill the second test's body after Task 8 lands the smoke config: call `main([... "--dry-run"])` against `dev_tests/vera_smoke_config.yaml` with a fabricated model-dir argument and assert `rc == 0` plus stdout parses as JSON with `"dry_run": true`.

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_solve_one.py -v`
Expected: FAIL, no module `dynamite.vera.solve_one`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/solve_one.py
"""Single-model weight-solve array task (spec section 5.3)."""
import argparse, json, os, sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"model_dir": args.model_dir, "dry_run": True}))
        return 0
    try:
        import dynamite as dyn
        c = dyn.config_reader.Configuration(args.config, reset_logging=True)
        idx = _find_row(c, args.model_dir)
        if idx is None:
            print(f"model dir {args.model_dir} not in table", file=sys.stderr)
            return 1
        mod = c.all_models.get_model_from_row(idx)
        cwd = os.getcwd()
        try:
            orblib = mod.get_orblib()          # sentinel short-circuit applies
            mod.get_weights(orblib)
        finally:
            os.chdir(cwd)
        from astropy.io import ascii
        meta = ascii.read(mod.directory + dyn.constants.weight_file).meta
        print(json.dumps({"model_dir": args.model_dir,
                          "chi2_tot": float(meta["chi2_tot"]),
                          "chi2_kin": float(meta["chi2_kin"]),
                          "chi2_kinmap": float(meta["chi2_kinmap"])}))
        return 0
    except Exception as e:                     # noqa: BLE001 - task boundary
        print(json.dumps({"error": repr(e), "model_dir": args.model_dir}),
              file=sys.stderr)
        return 3


def _find_row(config, model_dir):
    for i, row in enumerate(config.all_models.table):
        if row["directory"] == model_dir:
            return i
    return None


if __name__ == "__main__":
    sys.exit(main())
```

Binding notes: `mod.directory` ends with `/` in production tables — confirm and normalize; `dyn.constants.weight_file` is the source of truth for the filename (same constant Task 2 references); `os.chdir` discipline matches `create_and_run_model` because the Fortran helpers are cwd-sensitive.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_solve_one.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/solve_one.py dev_tests/test_vera_solve_one.py
git commit -m "feat(vera): single-model solve array task entry point"
```

---

### Task 8: Smoke-campaign config + VERA runbook

**Files:**
- Create: `dev_tests/vera_smoke_config.yaml` (copy of `PM_grid/NGC5139_config_production.yaml` with: `stopping_criteria.n_max_mods: 20`, `input_directory` unchanged, header comment documenting purpose)
- Create: `docs/vera_phase1_runbook.md`

**Interfaces:**
- Consumes: everything above. This is the artifact the acceptance gate runs against.

- [ ] **Step 1: Create the smoke config**

Copy the production yaml verbatim, then apply exactly two diffs: `n_max_mods: 20` and a top comment block:

```yaml
# VERA PHASE-1 SMOKE CAMPAIGN (spec 2026-08-23, section 10 phase 1)
# Production shape (xeast library, float32, streamed reads), capped at 20
# models. Gate: end-to-end <= 1 day on <= 8 nodes; one solve's chi2 within
# 1e-6 rel. of local same-parset reference; driver kill/restart resumes
# with zero duplicate work.
```

- [ ] **Step 2: Write the runbook**

`docs/vera_phase1_runbook.md` containing, verbatim-runnable: the §7 env-build sequence (conda-on-ptmp, wheel installs, gcc module load + legacy_fortran make, ENV_FREEZE.txt), scp/rsync of input pack, the three sbatch invocations (driver daemon under `nohup`, integration wave via `scripts/vera_submit_integration.sh`, solve wave auto-submitted by driver), kill/restart drill (`scancel` driver → relaunch → assert `vera_inflight.json` reconciliation), and the acceptance-gate checklist copied from spec §10 phase 1 including the χ² cross-check procedure against the local reference table (`/nexus/.../PM_grid/NGC5139_production_output/all_models.ecsv` matched by parset values, tolerance 1e-6 relative).

- [ ] **Step 3: Validate config loads**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -c "import dynamite as dyn; c = dyn.config_reader.Configuration('dev_tests/vera_smoke_config.yaml', reset_logging=True); print(c.settings.parameter_space_settings['stopping_criteria']['n_max_mods'])"` from a scratch CWD containing a symlinked input/output tree (documented in runbook).
Expected: prints `20`

- [ ] **Step 4: Commit**

```bash
git add dev_tests/vera_smoke_config.yaml docs/vera_phase1_runbook.md
git commit -m "feat(vera): phase-1 smoke campaign config and cluster runbook"
```

---

### Task 9: VERA environment setup script

**Files:**
- Create: `scripts/vera_env_setup.sh`

**Interfaces:**
- Produces: idempotent bash script executed on a VERA login node; creates conda env on scratch, installs wheels, builds Fortran, writes `ENV_FREEZE.txt`. Exit 0 only when `python -c "import dynamite, adelie"` succeeds inside the env.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# VERA environment bootstrap (spec section 7). Idempotent.
set -euo pipefail
BASE=/vera/ptmp/gc/mia/pesmith/oCen
ENV=$BASE/envs/dynamite
REPO=$BASE/dynamite

module purge
module load gcc            # pin exact version shown by: module av gcc
mkdir -p "$BASE"

if [ ! -x "$ENV/bin/python" ]; then
    conda create -y -p "$ENV" python=3.12
fi
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install numpy scipy astropy pathos possum \
    matplotlib pandas h5py pyyaml
"$ENV/bin/python" -m pip install adelie
"$ENV/bin/python" -m pip install --no-deps -e "$REPO"

make -C "$REPO/legacy_fortran" all

"$ENV/bin/python" - <<'EOF' > "$BASE/ENV_FREEZE.txt"
import platform, numpy, scipy, astropy, sys
print("python", sys.version.split()[0])
for m in (numpy, scipy, astropy):
    print(m.__name__, m.__version__)
try:
    import adelie; print("adelie", adelie.__version__)
except Exception as e:
    print("adelie IMPORT FAILED:", e)
EOF
"$ENV/bin/python" -c "import dynamite, adelie" \
    && echo "vera env OK" || { echo "env broken"; exit 1; }
```

Adjustments expected during execution (record outcomes in ENV_FREEZE.txt header comments): exact gcc module name, whether MPCDF python module is preferable to conda, any wheel unavailable for SLE_15 requiring `--index-url` adjustments.

- [ ] **Step 2: Shellcheck + commit**

Run: `bash -n scripts/vera_env_setup.sh && echo OK`
Expected: OK

```bash
git add scripts/vera_env_setup.sh
git commit -m "feat(vera): idempotent cluster environment bootstrap script"
```

---

### Task 10 (deployment track): VERA smoke run (D1)

**Files:** none created (execution task; outputs go to the runbook's log appendix).

- [ ] **Step 1**: Execute `scripts/vera_env_setup.sh` on vera01; fix module names; commit amended pins back to the script.
- [ ] **Step 2**: Stage smoke campaign per runbook (input pack rsync, config symlink layout).
- [ ] **Step 3**: Launch driver under nohup with `--dry-run` first cycle; verify printed submission plan lists 20 integrations then solves; launch for real.
- [ ] **Step 4**: Kill-restart drill mid-wave: `scancel` the driver, relaunch, confirm zero resubmission of in-flight work (ledger reconciliation log lines) and eventual completion count == 20.
- [ ] **Step 5**: Acceptance gate check: all 20 rows `all_done=True`; pick the fiducial-parset model, compare χ² against the local reference within 1e-6 relative; record wall-times into the runbook appendix.
- [ ] **Step 6**: Commit runbook appendix with measured numbers; open Phase-2 planning discussion (MicroBatchWalk) referencing actual throughput.

Gate: if any acceptance criterion fails, fix and repeat Task 10 — do not proceed to D2.

---

### Task 11 (capability track): MicroBatchWalk proposer

**Files:**
- Create: `dynamite/vera/proposer_microbatch.py`
- Create: `dev_tests/test_vera_proposer_microbatch.py`

**Interfaces:**
- Consumes: `GridWalkProposer` (Task 5) — subclasses it.
- Produces: `class MicroBatchWalkProposer(GridWalkProposer)` with settings `min_solved_fraction` (default 0.8), `max_batch`; overrides `quorum_pending()` and gains re-centering behavior: when solved-fraction of the outstanding set ≥ threshold, calls the parent's regeneration path immediately instead of waiting for full completion. Spec gate C2: unit tests here + synthetic-landscape ablation vs parent at equal budget.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_proposer_microbatch.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_quorum_fraction_semantics(minimal_config):
    strat = make_micro_proposer(minimal_config, min_solved_fraction=0.8)
    props = strat.propose(max_batch=500)          # e.g. 10 proposals
    n = len(props)
    # 0 solved -> quorum_pending == n (nothing may re-center yet)
    assert strat.quorum_pending() == n
    _solve_rows(strat, props[:7])                 # 70% < 80%
    assert strat.quorum_pending() > 0
    _solve_rows(strat, props[7:])                 # 100% >= 80%
    assert strat.quorum_pending() == 0            # proposer will regenerate


def test_recenter_uses_only_newly_solved(minimal_config):
    strat = make_micro_proposer(minimal_config)
    first = strat.propose(max_batch=500)
    _solve_all(strat, first)
    second = strat.propose(max_batch=500)         # re-centered proposals differ
    ids_a = {p.proposal_id for p in first}
    assert all(p.proposal_id not in ids_a for p in second)
```

Helpers `make_micro_proposer` / `_solve_rows` / `_solve_all` follow Task 5's test-file fixture pattern (`minimal_config`, direct table mutation).

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposer_microbatch.py -v`
Expected: FAIL, no module `dynamite.vera.proposer_microbatch`

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/proposer_microbatch.py
"""Micro-batch walker: re-center on a solved fraction, not full batches."""
from .proposer_gridwalk import GridWalkProposer


class MicroBatchWalkProposer(GridWalkProposer):
    def __init__(self, config, min_solved_fraction=0.8):
        super().__init__(config)
        self.min_solved_fraction = float(min_solved_fraction)

    def _outstanding(self):
        tracked = self.tracked_results()             # pid -> row idx (Task 5 map)
        done = sum(1 for pid, row in tracked.items()
                   if self.config.all_models.table[row]["all_done"])
        return len(tracked) - done

    def quorum_pending(self):
        tracked_n = len(self.tracked_results())
        if tracked_n == 0:
            return 0
        remaining = self._outstanding()
        solved_frac = 1.0 - remaining / tracked_n
        return 0 if solved_frac >= self.min_solved_fraction else remaining

    def exhausted(self):
        return super().exhausted()
```

Binding note: `tracked_results()` is the `proposal_id -> row index` mapping Task 5's implementation notes already require; promote it from an attribute convention to a real method there so both proposers share it.

- [ ] **Step 4: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposer_microbatch.py -v`
Expected: 2 PASS

- [ ] **Step 5: Synthetic-landscape ablation (gate C2 evidence)**

Reuse the bayesopt ablation harness pattern (`git show fork/bayesopt:dev_tests/test_vs_gridwalk.py` and its dummy-chi² synthetic landscape): run GridWalk vs MicroBatch on identical seeds/budgets, record models-to-best-χ². Commit results as a table in `dev_tests/vera_microbatch_ablation.md`.

- [ ] **Step 6: Commit**

```bash
git add dynamite/vera/proposer_microbatch.py dev_tests/test_vera_proposer_microbatch.py dev_tests/vera_microbatch_ablation.md
git commit -m "feat(vera): microbatch proposer with fraction-based recentering + ablation"
```

---

### Task 12 (capability track): BayesOptGenerator adapter

**Files:**
- Create: `dynamite/vera/proposer_bayesopt.py`
- Create: `dev_tests/test_vera_proposer_bayesopt.py`

**Interfaces:**
- Consumes: `BayesOptGenerator` + helpers from `fork/bayesopt` (`parameter_space.py`). Merge strategy decided at execution time — prefer merging the branch into `slurm` (zero file overlap today) over vendoring; this task assumes merged.
- Produces: `class BayesOptProposer` with the same Protocol surface as Tasks 5/11; maps `generator.generate(current_models)` output rows to Proposals; `exhausted()` ORs the generator's status flags including `gp_predictions_accurate` (R3); warm-start comes free via H2 when the table has history.

- [ ] **Step 1: Write the failing test**

```python
# dev_tests/test_vera_proposer_bayesopt.py
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("botorch")     # BO stack optional for the rest of the suite


def test_adapter_maps_generated_rows_to_proposals(minimal_bo_config):
    strat = make_bo_proposer(minimal_bo_config)      # generator_type BayesOpt
    props = strat.propose(max_batch=4)
    assert 1 <= len(props) <= 4                        # batch_size honored
    assert len({p.proposal_id for p in props}) == len(props)


def test_exhausted_or_status_flags(minimal_bo_config):
    strat = make_bo_proposer(minimal_bo_config)
    strat.propose(max_batch=4)
    assert isinstance(strat.exhausted(), bool)


def test_warmstart_trains_from_history(minimal_bo_config, populated_table):
    # populated_table: fixture inserting >= n_initial_random valid rows
    strat = make_bo_proposer(minimal_bo_config)
    props = strat.propose(max_batch=4)
    assert props                                       # no Sobol warm-up needed
```

Fixture notes: `minimal_bo_config` mirrors Task 5's fixture but sets `generator_type: BayesOpt` plus minimal `bayesopt_settings` copied from `fork/bayesopt:dev_tests/test_bayesopt_generator.py::_bo_settings`; `populated_table` builds a MockAllModels table per that file's fixtures.

- [ ] **Step 2: Run test to verify it fails**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest dev_tests/test_vera_proposer_bayesopt.py -v`
Expected: FAIL (module missing, or botorch absent → importorskip skips; then install pins and rerun)

- [ ] **Step 3: Write minimal implementation**

```python
# dynamite/vera/proposer_bayesopt.py
"""TableDriven adapter around BayesOptGenerator (spec sections 4.1, C3)."""
from .proposal import Proposal


class BayesOptProposer:
    def __init__(self, config):
        self.config = config
        import dynamite.parameter_space as ps
        gen_type = config.settings.parameter_space_settings["generator_type"]
        assert gen_type == "BayesOpt", gen_type
        self.generator = ps.BayesOptGenerator(config=config)
        self.dir_to_pid = {}

    def propose(self, max_batch=4):
        table = self.config.all_models.table
        before = len(table)
        self.generator.generate(current_models=self.config.all_models)
        props = []
        for i in range(before, len(self.config.all_models.table)):
            parset = {p.name: float(p.raw_value)
                      for p in self.config.all_models.get_model_from_row(i).parameters}
            pid = self._pid(parset)
            self.dir_to_pid[table["directory"][i]] = pid
            props.append(Proposal(proposal_id=pid, parset=parset))
            if len(props) >= max_batch:
                break
        return props

    @staticmethod
    def _pid(parset):
        from .proposal import canonical_hash
        return canonical_hash(parset)

    def observe(self, results):
        return None                                   # table is the channel

    def quorum_pending(self):
        return 0                                      # continuous mode

    def exhausted(self):
        status = getattr(self.generator, "status", {})
        flags = ("stop", "gp_max_variance_low", "gp_min_ei_low",
                 "gp_predictions_accurate")
        return any(bool(status.get(f)) for f in flags)
```

Binding notes: confirm `BayesOptGenerator.status` flag names against `fork/bayesopt:dynamite/parameter_space.py` (`grep -n "gp_predictions_accurate\|gp_max_variance_low" ...`) and adjust `flags`; confirm row-value accessor (`raw_value`) exactly as Task 5 resolved it.

- [ ] **Step 4: Local real-orblib minirun (gate C3 evidence)**

Follow the pattern of `fork/bayesopt:dev_tests/test_real_minirun.py`: reduced orblib class, ≤4 models end-to-end through adapter + driver dry-run submissions executed as local subprocesses (no Slurm), asserting weight files appear and R3 counter increments. Marked `@pytest.mark.local_slow`.

- [ ] **Step 5: Commit**

```bash
git add dynamite/vera/proposer_bayesopt.py dev_tests/test_vera_proposer_bayesopt.py
git commit -m "feat(vera): bayesopt tabledriven adapter with R3-aware exhaustion"
```
