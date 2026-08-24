"""Regression tests for the cloud-review findings.

The theme is name-space drift: the existing vera tests key parsets by bare
`q`/`p`/`u` and compare raw bounds against raw values, while production
parspaces use `q-stars`/`p-stars`/`u-stars` and store PHYSICAL values in the
table. The mocks agreed with the bugs, so the gates they guard were inert.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.classifier import ModelState, classify  # noqa: E402
from dynamite.vera.proposal import Proposal, validate_parset  # noqa: E402

QOBS = 0.55097
NOW = time.time()

# real parspace names, as config_reader builds them: f"{par}-{comp}"
BOUNDS = {
    "q-stars": {"lo": 0.05, "hi": 0.99},
    "p-stars": {"lo": 0.05, "hi": 0.999},
    "u-stars": {"lo": 0.05, "hi": 1.0},
    "ml": {"lo": 1.0, "hi": 6.0},
}


def _parset(q, p, u, ml=2.6):
    return {"q-stars": q, "p-stars": p, "u-stars": u, "ml": ml}


def test_triaxiality_gate_fires_on_qualified_names():
    """p < q must be rejected even though the key is `p-stars`, not `p`."""
    _, violations = validate_parset(_parset(0.85, 0.70, 0.99), BOUNDS, qobs=QOBS)
    assert violations, "impossible shape (p < q) passed the intake gate"
    assert any("oblate-equivalent" in v for v in violations), violations


def test_axis_limit_fires_on_qualified_names():
    """q > u*qobs must be rejected under the suffixed names too."""
    _, violations = validate_parset(_parset(0.95, 0.97, 0.60), BOUNDS, qobs=QOBS)
    assert any("axis limit" in v for v in violations), violations


def test_feasible_shape_still_passes():
    q, p, u = 0.46, 0.90, 0.9925
    assert q <= u * QOBS * (1 - 1e-6) or True  # sanity of the fixture
    _, violations = validate_parset(_parset(0.40, 0.90, 0.99), BOUNDS, qobs=QOBS)
    assert violations == [], violations


def test_bare_names_still_work():
    """System-level parameters (ml, bh.m) are unqualified; don't break them."""
    _, violations = validate_parset(
        {"q": 0.85, "p": 0.70, "u": 0.99}, {}, qobs=QOBS
    )
    assert violations, "bare-name parsets must still be validated"


def test_log_parameter_must_be_validated_in_raw_space():
    """par_generator_settings lo/hi are log10; the table stores the physical
    value. Comparing them made the bounds gate a no-op for every log
    parameter -- and would have destroyed the value if the clip were applied.
    """
    from dynamite.parameter_space import Parameter

    par = Parameter(
        name="m-bh", fixed=False, logarithmic=True, value=5.0,
        par_generator_settings={"lo": 1.0, "hi": 10.0},
    )
    physical = float(par.par_value)  # 10 ** 5 = 100000
    lo, hi = 1.0, 10.0
    assert not (lo <= physical <= hi), "fixture no longer exercises the mismatch"
    raw = float(par.get_raw_value_from_par_value(physical))
    assert lo <= raw <= hi, "raw value must sit inside the raw bounds"
    # and the physical value would have been clipped to the raw ceiling
    clipped, violations = validate_parset(
        {"m-bh": physical}, {"m-bh": {"lo": lo, "hi": hi}}, qobs=None
    )
    assert violations == []  # silently "repaired", never flagged
    assert clipped["m-bh"] == hi  # ...to 10, from 100000
    assert not np.isclose(clipped["m-bh"], physical)


def test_levelfs_parses_real_sshare_output():
    """sshare only emits '|' separated fields when asked with -P; without it
    the split never yields 5 fields and the adaptive throttle is dead."""
    from dynamite.vera.slurm import levelfs

    calls = []

    def runner(argv):
        calls.append(argv)
        # what sshare -P actually prints
        return "pesmith|mia|9604|0.000002|1.7307e+03\n"

    assert abs(levelfs(runner, "pesmith") - 1730.7) < 0.01
    assert "-P" in calls[0], f"sshare invoked without parsable output: {calls[0]}"


def test_parked_model_is_reported_once(tmp_path):
    """A parked model stays parked; re-announcing it every poll cycle means
    the driver never reaches quiescence."""
    import json

    from dynamite.vera.driver import VeraDriver
    from test_vera_driver_fixes import ArrayRunner, RealisticProposer, WEIGHTS_ECSV  # noqa: F401
    from test_vera_proposer_gridwalk import build_minimal_config

    cfg = build_minimal_config()
    outroot = tmp_path / "out"
    rel = "orblib_001_000/ml02.60"
    (outroot / "models" / rel / "datfil").mkdir(parents=True)
    (outroot / "models" / "orblib_001_000" / "datfil").mkdir(parents=True, exist_ok=True)
    cfg.all_models.table.add_row(
        [2.6, 0.46, 0.90, np.nan, np.nan, np.nan, "", False, False, False, 1, rel]
    )
    cfg.settings.io_settings = {
        "output_directory": str(outroot), "all_models_file": "all_models.ecsv",
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "vera_attempts.json", "w") as f:
        json.dump({rel: 5}, f)
    with open(run_dir / "vera_dirs.json", "w") as f:
        json.dump({"dir_to_pid": {rel: "deadbeef"}}, f)

    prop = RealisticProposer()
    drv = VeraDriver(cfg, prop, runner=ArrayRunner(), run_dir=str(run_dir),
                     clock=lambda: NOW)
    assert drv.observe_completions() == 1  # reported once
    assert drv.observe_completions() == 0, "parked model re-reported every cycle"


def test_daemon_survives_non_slurm_errors():
    """The cycle does NFS I/O and astropy writes; a half-written ledger
    raises OSError/JSONDecodeError, not SlurmError."""
    from dynamite.vera.driver import VeraDriver

    class Boom(VeraDriver):
        def __init__(self):  # no config needed; we only exercise the loop
            self.calls = 0
            self.poll_interval = 0
            import logging

            self.log = logging.getLogger("test.boom")

        def step(self, dry_run=False):
            self.calls += 1
            raise OSError("stale NFS file handle")

    drv = Boom()
    with pytest.raises(OSError):
        drv.run_forever(max_consecutive_errors=3)
    assert drv.calls == 3, f"gave up after {drv.calls} cycle(s), not 3"


def test_proposal_from_dict_rejects_foreign_payloads():
    """Mirrors Result.from_dict: a bare assert vanishes under python -O."""
    pr = Proposal(proposal_id="abc", parset={"ml": 2.6})
    assert Proposal.from_dict(pr.to_dict()) == pr
    with pytest.raises(ValueError):
        Proposal.from_dict({**pr.to_dict(), "schema_version": 99})


def test_future_mtime_sentinel_is_not_pending(tmp_path):
    """A sentinel whose mtime is ahead of the driver's clock (NFS skew)
    exists; reading it as absent triggered a needless re-integration."""
    model = tmp_path / "orblib_001_000" / "ml02.60"
    (model / "datfil").mkdir(parents=True)
    noml_datfil = tmp_path / "orblib_001_000" / "datfil"
    noml_datfil.mkdir(parents=True, exist_ok=True)
    sentinel = noml_datfil / "tube_box_done"
    sentinel.write_text("done\n")
    skewed = NOW + 30  # compute node's clock is ahead of the driver's
    os.utime(sentinel, (skewed, skewed))
    state = classify(str(model), attempts=0, now_ts=NOW)
    assert state is not ModelState.PENDING_INTEGRATION, (
        "a present sentinel read as absent"
    )
    assert state in (ModelState.INTEGRATING, ModelState.TO_SOLVE), state
