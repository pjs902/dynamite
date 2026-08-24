"""Regression tests for the vera code-review fixes.

Deliberately avoids the stubs in test_vera_driver.py: those gave the fake
collaborators a WIDER interface than the real ones (a dir_to_pid the
proposers never had, per-task Slurm job ids that arrays never issue), which
is why the defects here were invisible to the suite.
"""

import json
import os
import time

import numpy as np
import pytest

from dynamite.vera.classifier import ModelState
from dynamite.vera.driver import VeraDriver
from dynamite.vera.proposal import Result
from dynamite.vera.slurm import MAX_ARRAY_SIZE, build_solve_job_spec
from test_vera_proposer_gridwalk import build_minimal_config

NOW = time.time()
WEIGHTS_ECSV = (
    "# %ECSV 1.0\n# ---\n# datatype:\n# - {name: w, datatype: float64}\n"
    "# meta: !!omap\n# - {chi2_tot: 1.0}\n# - {chi2_kin: 2.0}\n"
    "# - {chi2_kinmap: 3.0}\nw\n1.0\n"
)


class ArrayRunner:
    """Slurm as it really behaves: one array job has ONE id, and squeue
    reports its tasks as <jid>_<n>, which running_job_ids truncates."""

    def __init__(self, live=()):
        self.live = set(live)
        self.sbatch = []

    def __call__(self, argv):
        if argv[0] == "sbatch":
            self.sbatch.append(argv)
            return "Submitted batch job %d\n" % (7000 + len(self.sbatch))
        if argv[0] == "squeue":
            return "".join(f"{j}_0\n" for j in sorted(self.live))
        if argv[0] == "sshare":
            return "%s|mia|0|0|9999\n" % os.environ.get("USER", "u")
        raise AssertionError(argv)


class RealisticProposer:
    """Only what GridWalkProposer/BayesOptProposer actually expose."""

    def __init__(self):
        self.pid_to_row = {}
        self.failed_pids = set()

    def propose(self, max_batch=1000):
        return []

    def observe(self, results):
        for r in results:
            if r.status == "failed":
                self.failed_pids.add(r.proposal_id)

    def quorum_pending(self):
        return 0

    def exhausted(self):
        return False


def _aged(p):
    os.utime(p, (NOW - 3600, NOW - 3600))


@pytest.fixture()
def world(tmp_path):
    """n model dirs; caller adds artifacts. Returns (cfg, dirs, run_dir)."""

    def build(specs):
        cfg = build_minimal_config()
        outroot = tmp_path / "out"
        dirs = []
        t = cfg.all_models.table
        for i, artifacts in enumerate(specs):
            rel = f"orblib_001_{i:03d}/ml02.60"
            (outroot / "models" / rel / "datfil").mkdir(parents=True, exist_ok=True)
            noml = outroot / "models" / f"orblib_001_{i:03d}" / "datfil"
            noml.mkdir(parents=True, exist_ok=True)
            for art in artifacts:
                f = noml / art if art.endswith("_done") else outroot / "models" / rel / art
                f.write_text(WEIGHTS_ECSV if art.endswith(".ecsv") else "# fixture\n")
                _aged(f)
            t.add_row([2.6, 0.46, 0.90, np.nan, np.nan, np.nan, "", False, False, False, 1, rel])
            dirs.append(rel)
        cfg.settings.io_settings = {
            "output_directory": str(outroot),
            "all_models_file": "all_models.ecsv",
        }
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        return cfg, dirs, str(run_dir)

    return build


def _drv(cfg, run_dir, runner, proposer=None):
    return VeraDriver(
        cfg, proposer or RealisticProposer(), runner=runner,
        run_dir=run_dir, clock=lambda: NOW,
    )


def test_observe_survives_driver_restart(world):
    """dir_to_pid is persisted; the row index must be derived, not cached."""
    cfg, dirs, run_dir = world([["tube_box_done", "orbit_weights.ecsv"]])
    with open(os.path.join(run_dir, "vera_dirs.json"), "w") as f:
        json.dump({"dir_to_pid": {dirs[0]: "abc123"}}, f)
    # a fresh driver over the same run dir, as after a daemon restart
    drv = _drv(cfg, run_dir, ArrayRunner())
    n = drv.observe_completions()
    assert n == 1, "the solved model should have been observed"
    t = cfg.all_models.table
    row = next(i for i, r in enumerate(t) if str(r["directory"]) == dirs[0])
    assert bool(t["all_done"][row])
    assert float(t["kinchi2"][row]) == 2.0


def test_parked_model_reports_failure(world):
    """The PARKED branch must reach the driver's own attribution map."""
    cfg, dirs, run_dir = world([[]])
    with open(os.path.join(run_dir, "vera_attempts.json"), "w") as f:
        json.dump({dirs[0]: 5}, f)
    with open(os.path.join(run_dir, "vera_dirs.json"), "w") as f:
        json.dump({"dir_to_pid": {dirs[0]: "deadbeef"}}, f)
    prop = RealisticProposer()
    drv = _drv(cfg, run_dir, ArrayRunner(), prop)
    assert drv.scan()[dirs[0]] is ModelState.PARKED
    drv.observe_completions()
    assert "deadbeef" in prop.failed_pids


def test_live_array_tasks_are_not_resubmitted(world):
    """Every task of an array shares the base jid."""
    cfg, dirs, run_dir = world([["tube_box_done"]] * 3)
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    assert len(runner.sbatch) == 1
    led = json.load(open(os.path.join(run_dir, "vera_ledger_jids.json")))
    assert set(led.values()) == {7001}, f"array tasks must share one jid: {led}"

    runner.live = {7001}  # the array job is still running
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    assert len(runner.sbatch) == 1, "live work was resubmitted underneath itself"


def test_vanished_job_charges_an_attempt(world):
    """A job that disappears without artifacts is a failed attempt."""
    cfg, dirs, run_dir = world([["tube_box_done"]])
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    runner.live = set()  # job gone, weights never appeared
    drv = _drv(cfg, run_dir, runner)
    drv.reconcile_and_submit()
    attempts = json.load(open(os.path.join(run_dir, "vera_attempts.json")))
    assert attempts.get(dirs[0], 0) >= 1, "no attempt was charged"


def test_array_size_bounded_not_just_throttle():
    """MaxArraySize limits the array, which is what Slurm rejects over."""
    spec = build_solve_job_spec(k=16, n_items=5000)
    flag = next(e for e in spec["extra"] if e.startswith("--array="))
    upper = int(flag.split("=")[1].split("%")[0].split("-")[1])
    assert upper <= MAX_ARRAY_SIZE - 1, flag


def test_result_from_dict_rejects_foreign_payloads():
    r = Result(proposal_id="a", model_dir="d", status="done", chi2=1.0)
    assert Result.from_dict(r.to_dict()) == r
    with pytest.raises(ValueError):
        Result.from_dict({**r.to_dict(), "schema_version": 99})
    with pytest.raises(ValueError):
        Result.from_dict({**r.to_dict(), "surprise": 1})


def test_each_array_task_gets_its_own_item(tmp_path):
    """Slurm hands every task identical argv; the item comes from the index.

    Drives the real script helper: a task that read $1 as its model dir
    would receive the entire wave instead.
    """
    import subprocess

    from dynamite.vera.slurm import write_manifest

    scripts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dynamite", "vera", "scripts",
    )
    items = [f"orblib_001_{i:03d}/ml02.60" for i in range(3)]
    manifest = write_manifest(str(tmp_path), "solve", items)
    probe = tmp_path / "probe.sh"
    probe.write_text(
        f'#!/bin/bash\nset -euo pipefail\n. "{scripts}/_select_item.sh"\necho "$ITEM"\n'
    )
    for idx, expected in enumerate(items):
        out = subprocess.run(
            ["bash", str(probe), manifest],
            capture_output=True, text=True,
            env={**os.environ, "SLURM_ARRAY_TASK_ID": str(idx)},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expected, f"task {idx} got {out.stdout!r}"


def test_wave_larger_than_max_array_is_chunked_not_truncated(world):
    """Every item must reach a real array; a truncated tail would sit in the
    in-flight ledger forever, never run and never retried."""
    from dynamite.vera.slurm import MAX_ARRAY_SIZE

    cfg, dirs, run_dir = world([["tube_box_done"]])
    runner = ArrayRunner()
    drv = _drv(cfg, run_dir, runner)
    n = MAX_ARRAY_SIZE + 5
    drv._submit_wave("solve", [[f"d{i}"] for i in range(n)], dry_run=False)
    scheduled = sum(
        int(next(e for e in call if e.startswith("--array=")).split("%")[0].rsplit("-", 1)[-1]) + 1
        for call in runner.sbatch
    )
    assert len(runner.sbatch) == 2, "wave should span two array jobs"
    assert scheduled == n, f"only {scheduled} of {n} items actually scheduled"


def test_missing_ledger_entry_does_not_stall(world):
    """A lost jid must read as dead (resubmit), not live-forever (stall)."""
    cfg, dirs, run_dir = world([["tube_box_done"]])
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    assert len(runner.sbatch) == 1
    os.remove(os.path.join(run_dir, "vera_ledger_jids.json"))  # ledger lost
    runner.live = {7001}  # slurm still says the job is alive
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    assert len(runner.sbatch) == 2, "work stalled instead of being retried"
    attempts = json.load(open(os.path.join(run_dir, "vera_attempts.json")))
    assert attempts.get(dirs[0], 0) >= 1


def test_successful_integration_is_not_billed_a_failure(world):
    """A finished integration leaves its model in TO_SOLVE -- that is success
    for an int job, and must not count against ATTEMPT_LIMIT."""
    cfg, dirs, run_dir = world([[]])  # nothing built yet -> PENDING_INTEGRATION
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()  # int wave goes out
    # the integration succeeds: the sentinel appears, so the model is TO_SOLVE
    sentinel = os.path.join(
        cfg.settings.io_settings["output_directory"], "models",
        dirs[0].split("/")[0], "datfil", "tube_box_done",
    )
    open(sentinel, "w").write("done\n")
    _aged(sentinel)
    runner.live = set()  # the int job has finished and left the queue
    drv = _drv(cfg, run_dir, runner)
    assert drv.scan()[dirs[0]] is ModelState.TO_SOLVE
    drv.reconcile_and_submit()
    attempts_path = os.path.join(run_dir, "vera_attempts.json")
    attempts = json.load(open(attempts_path)) if os.path.isfile(attempts_path) else {}
    assert attempts.get(dirs[0], 0) == 0, "successful integration was billed a failure"


def test_finished_work_is_pruned_from_the_ledger(world):
    """Entries are removed once their job is gone, so the ledger does not
    accumulate one dead key per model for the life of the campaign."""
    cfg, dirs, run_dir = world([["tube_box_done"]])
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()
    ledger = os.path.join(run_dir, "vera_ledger_jids.json")
    assert len(json.load(open(ledger))) == 1

    # the solve finishes: weights land, so there is nothing left to resubmit
    weights = os.path.join(
        cfg.settings.io_settings["output_directory"], "models", dirs[0],
        "orbit_weights.ecsv",
    )
    open(weights, "w").write(WEIGHTS_ECSV)
    _aged(weights)
    runner.live = set()  # and the job has left the queue
    drv = _drv(cfg, run_dir, runner)
    assert drv.scan()[dirs[0]] is ModelState.SOLVED
    drv.reconcile_and_submit()
    assert json.load(open(ledger)) == {}, "dead entry was never pruned"


def test_repacked_wave_does_not_resubmit_live_models(world):
    """In-flight membership is per model dir, not per joined package string.

    A wave repacks whenever its composition changes; a model still running
    inside a live package then appears in a differently-composed package and
    used to look like new work.
    """
    cfg, dirs, run_dir = world([[], [], []])  # three PENDING_INTEGRATION models
    runner = ArrayRunner()
    drv = _drv(cfg, run_dir, runner)
    # pack all three into ONE package, as pack_libraries would
    drv._submit_wave("int", [list(dirs)], dry_run=False)
    assert len(runner.sbatch) == 1
    assert set(drv.inflight["int"]) == set(dirs), drv.inflight

    # next cycle: the wave repacks differently (say one model per package)
    # while the original array job is still running
    runner.live = {7001}
    drv2 = _drv(cfg, run_dir, runner)
    drv2._submit_wave("int", [[d] for d in dirs], dry_run=False)
    assert len(runner.sbatch) == 1, "live models were resubmitted after a repack"


def test_unreadable_weights_do_not_kill_the_daemon(world):
    """A partially-flushed ecsv is readable next cycle, not a fatal error."""
    cfg, dirs, run_dir = world([["tube_box_done", "orbit_weights.ecsv"]])
    weights = os.path.join(
        cfg.settings.io_settings["output_directory"], "models", dirs[0],
        "orbit_weights.ecsv",
    )
    with open(weights, "w") as f:
        f.write("# %ECSV 1.0\n# ---\n# datatype:\n")  # truncated mid-write
    _aged(weights)
    with open(os.path.join(run_dir, "vera_dirs.json"), "w") as f:
        json.dump({"dir_to_pid": {dirs[0]: "abc123"}}, f)
    drv = _drv(cfg, run_dir, ArrayRunner())
    assert drv.observe_completions() == 0  # skipped, not raised
    assert not bool(cfg.all_models.table["all_done"][-1])


def test_crash_mid_integration_still_charges_an_attempt(world):
    """A library that dies mid-run leaves fresh files, so it classifies
    INTEGRATING at reconcile time. Testing for the not-started state let it
    resubmit forever with attempts stuck at 0."""
    cfg, dirs, run_dir = world([[]])
    runner = ArrayRunner()
    _drv(cfg, run_dir, runner).reconcile_and_submit()  # int wave out
    # the job crashed, but only after writing scratch files a moment ago
    scratch = os.path.join(
        cfg.settings.io_settings["output_directory"], "models", dirs[0],
        "datfil", "partial.dat",
    )
    open(scratch, "w").write("half a library\n")
    os.utime(scratch, (NOW - 1, NOW - 1))  # fresh -> INTEGRATING
    runner.live = set()  # job is gone from the queue
    drv = _drv(cfg, run_dir, runner)
    assert drv.scan()[dirs[0]] is ModelState.INTEGRATING
    drv.reconcile_and_submit()
    attempts = json.load(open(os.path.join(run_dir, "vera_attempts.json")))
    assert attempts.get(dirs[0], 0) >= 1, "a crashed integration was never charged"


def test_transient_squeue_failure_does_not_resubmit_everything(world):
    """A flaky squeue must not read as 'every job died'."""
    from dynamite.vera.slurm import SlurmError

    cfg, dirs, run_dir = world([["tube_box_done"]])

    class Flaky(ArrayRunner):
        def __call__(self, argv):
            if argv[0] == "squeue":
                raise SlurmError("squeue: Socket timed out")
            return super().__call__(argv)

    runner = Flaky()
    drv = _drv(cfg, run_dir, runner)
    assert drv.reconcile_and_submit() == 0
    assert runner.sbatch == [], "submitted work without knowing what was live"
