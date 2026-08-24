"""Tests for dynamite.vera.slurm - injectable CLI layer, pinned job specs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from dynamite.vera.slurm import (  # noqa: E402
    submit_array,
    write_manifest,
    running_job_ids,
    levelfs,
    build_solve_job_spec,
    build_integration_job_spec,
    SlurmError,
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


def test_submit_array_parses_job_id_and_flags(tmp_path):
    fr = FakeRunner({"sbatch": "Submitted batch job 424242\n"})
    spec = build_solve_job_spec(k=16, n_items=3)
    items = ["m/a", "m/b", "m/c"]
    manifest = write_manifest(str(tmp_path), "solve", items)
    jid = submit_array(fr, spec, "/path/solve_task.sh", items, manifest)
    assert jid == 424242
    joined = " ".join(fr.calls[0])
    assert "--array=0-2%16" in joined
    assert "--mem=200000" in joined
    assert "--cpus-per-task=24" in joined
    assert "--account=mia" in joined
    assert "--partition=p.large" in joined
    assert "--time=06:00:00" in joined
    # the per-task argument CANNOT ride on argv: every array task gets the
    # same command line, so it is looked up from the manifest by index
    assert manifest in joined
    assert "m/a;m/b;m/c" not in joined
    assert open(manifest).read().splitlines() == items


def test_submit_failure_raises():
    fr = FakeRunner({"sbatch": RuntimeError("sbatch: error")})
    with pytest.raises(SlurmError):
        submit_array(fr, build_solve_job_spec(k=16, n_items=1), "/path/x.sh", ["a"], "/m.txt")


def test_array_throttle_is_passed_through():
    """The %k throttle is a concurrency limit -- it cannot overflow anything,
    so it is passed through, only floored at 1. MaxArraySize bounds the array
    size instead; see test_array_size_bounded_not_just_throttle."""
    assert "%5000" in " ".join(build_solve_job_spec(k=5000, n_items=200)["extra"])
    assert "%1" in " ".join(build_solve_job_spec(k=0, n_items=5)["extra"])


def test_running_jobs_parsed_with_array_suffixes():
    fr = FakeRunner({"squeue": "424242\n424999_7\n"})
    assert running_job_ids(fr) == {424242, 424999}


def test_levelfs_float_or_none():
    good = FakeRunner({"sshare": "pesmith|mia|9604|0.000002|1.7307e+03\n"})
    assert abs(levelfs(good, "pesmith") - 1730.7) < 0.01
    bad = FakeRunner({"sshare": ""})
    assert levelfs(bad, "pesmith") is None


def test_integration_spec_exclusive_vera():
    fr = FakeRunner({"sbatch": "Submitted batch job 7\n"})
    spec = build_integration_job_spec(n_items=2)
    jid = submit_array(fr, spec, "/path/int_task.sh", ["pkg0", "pkg1"], "/m.txt")
    assert jid == 7
    joined = " ".join(fr.calls[0])
    assert "--exclusive" in joined
    assert "--partition=p.vera" in joined
    assert "--cpus-per-task=72" in joined
    assert "--array=0-1%8" in joined


def test_real_runner_rejects_on_error():
    import shutil

    if shutil.which("sbatch") is None:
        pytest.skip("no slurm on this host (local dev box)")
    from dynamite.vera.slurm import RealRunner

    with pytest.raises(SlurmError):
        RealRunner()(["sbatch", "--definitely-not-a-flag"])
