"""Tests for dynamite.vera.driver - synthetic tree, recording runner.

Idempotence semantics (spec section 9): work whose Slurm job is still alive
is never resubmitted; work whose job vanished without artifacts IS resubmitted
(crash recovery); completed work is never touched again.
"""

import os
import sys
import time

import numpy as np
import pytest
from astropy.table import Column

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.driver import VeraDriver  # noqa: E402
from test_vera_proposer_gridwalk import build_minimal_config  # noqa: E402

NOW = time.time()


class RecordingRunner:
    def __init__(self, live_jids=()):
        self.live_jids = set(live_jids)
        self.sbatch_calls = []

    def __call__(self, argv):
        if argv[0] == "sbatch":
            self.sbatch_calls.append(argv)
            return "Submitted batch job %d\n" % (7000 + len(self.sbatch_calls))
        if argv[0] == "squeue":
            return "".join(f"{j}\n" for j in sorted(self.live_jids))
        if argv[0] == "sshare":
            return "pesmith|mia|0|0|9999\n"
        raise AssertionError(f"unexpected command {argv}")


class StubProposer:
    def __init__(self):
        self.dir_to_pid = {}
        self.dir_to_row = {}

    def propose(self, max_batch=1000):
        return []

    def observe(self, results):
        self.seen = getattr(self, "seen", []) + list(results)

    def quorum_pending(self):
        return 0

    def exhausted(self):
        return False


def _aged(path):
    os.utime(path, (NOW - 3600, NOW - 3600))


@pytest.fixture()
def world(tmp_path):
    cfg = build_minimal_config()
    outroot = tmp_path / "NGC5139_production_output"
    models = outroot / "models"

    # distinct orblib parents: in reality each library owns its noml dir;
    # sharing one would leak sentinels between models
    specs = {"pending": ("orblib_001_000/ml02.pe", []),
             "built": ("orblib_001_001/ml02.bu", ["tube_box_done"]),
             "solved": ("orblib_001_002/ml02.so",
                        ["tube_box_done", "orbit_weights.ecsv"])}
    dirs = {}
    for kind, (rel, artifacts) in specs.items():
        d = models / rel
        (d / "datfil").mkdir(parents=True)
        noml_datfil = models / rel.split("/")[0] / "datfil"
        noml_datfil.mkdir(parents=True, exist_ok=True)
        for art in artifacts:
            # sentinels live at the noml level (shared library); weights at ml
            f = noml_datfil / art if art.endswith("_done") else d / art
            if not f.exists():
                f.write_text("# fixture\n")
            _aged(f)
        dirs[kind] = rel

    t = cfg.all_models.table
    for lab in dirs.values():
        t.add_row([2.6, 0.46, 0.90, np.nan, np.nan, np.nan,
                   "", False, False, False, 1, lab])

    cfg.settings.io_settings = {"output_directory": str(outroot),
                                "all_models_file": "all_models.ecsv"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return cfg, dirs, run_dir


def test_first_pass_submits_int_and_solve_waves(world):
    cfg, dirs, run_dir = world
    rr = RecordingRunner()
    drv = VeraDriver(cfg, StubProposer(), runner=rr, run_dir=str(run_dir))
    n = drv.reconcile_and_submit()
    assert n == 2  # one int wave + one solve wave
    kinds = [" ".join(a) for a in rr.sbatch_calls]
    assert any("--job-name=ocen-int" in k for k in kinds)
    assert any("--job-name=ocen-solve" in k for k in kinds)
    # pending dir goes into the integration wave; built dir into solve wave.
    # The items live in the manifest, NOT on the command line: every array
    # task shares one argv, so per-task arguments must be looked up by index.
    def _manifest_body(call):
        path = next(a for a in call if a.endswith(".txt"))
        return open(path).read()

    int_call = next(c for c in rr.sbatch_calls if "--job-name=ocen-int" in " ".join(c))
    solve_call = next(c for c in rr.sbatch_calls if "--job-name=ocen-solve" in " ".join(c))
    assert dirs["pending"] in _manifest_body(int_call)
    assert dirs["built"] in _manifest_body(solve_call)
    assert dirs["solved"] not in _manifest_body(solve_call)


def test_alive_jobs_not_resubmitted(world):
    cfg, dirs, run_dir = world
    rr1 = RecordingRunner()
    VeraDriver(
        cfg, StubProposer(), runner=rr1, run_dir=str(run_dir)
    ).reconcile_and_submit()
    jids = tuple(7001 + i for i in range(3))  # array ids + per-task offsets
    rr2 = RecordingRunner(live_jids=jids)  # slurm says all alive
    n2 = VeraDriver(
        cfg, StubProposer(), runner=rr2, run_dir=str(run_dir)
    ).reconcile_and_submit()
    assert n2 == 0
    assert rr2.sbatch_calls == []


def test_completed_work_never_resubmitted(world):
    cfg, dirs, run_dir = world
    rr = RecordingRunner(live_jids=(7001, 7002))  # pretend first wave alive
    drv = VeraDriver(cfg, StubProposer(), runner=rr, run_dir=str(run_dir))
    drv.reconcile_and_submit()  # ledger records 7001/7002
    before = len(rr.sbatch_calls)
    drv.reconcile_and_submit()  # same world, jobs alive
    assert len(rr.sbatch_calls) == before  # no duplicates while alive


def test_dry_run_submits_nothing(world, capsys):
    cfg, dirs, run_dir = world
    rr = RecordingRunner()
    n = VeraDriver(
        cfg, StubProposer(), runner=rr, run_dir=str(run_dir)
    ).reconcile_and_submit(dry_run=True)
    assert n == 0
    assert rr.sbatch_calls == []
    assert "[dry-run]" in capsys.readouterr().out
