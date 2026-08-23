"""Tests for dynamite.vera.classifier - artifact-state classification."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.classifier import ModelState, classify  # noqa: E402

NOW = 1_000_000.0
WEIGHTS = "orbit_weights.ecsv"  # dyn.constants.weight_file


def mk(root, sentinel=False, weights=False, fresh=False, other_fresh=False):
    d = root / "orblib_001_000" / "ml02.60"
    (d / "datfil").mkdir(parents=True, exist_ok=True)
    noml_datfil = root / "orblib_001_000" / "datfil"
    noml_datfil.mkdir(parents=True, exist_ok=True)
    old = NOW - 3600.0
    new = NOW - 5.0
    ts = new if fresh else old
    if sentinel:
        f = noml_datfil / "tube_box_done"
        f.write_text("")
        os.utime(f, (ts, ts))
    if weights:
        w = d / WEIGHTS
        w.write_text("# placeholder meta\n")
        os.utime(w, (old, old))  # weights always aged here
    if other_fresh:
        g = d / "datfil" / "begin.dat"
        g.write_text("")
        os.utime(g, (new, new))
    return d


def test_empty_dir_is_pending(tmp_path):
    assert (
        classify(mk(tmp_path), attempts=0, now_ts=NOW) is ModelState.PENDING_INTEGRATION
    )


def test_fresh_sentinel_counts_as_still_integrating(tmp_path):
    d = mk(tmp_path, sentinel=True, fresh=True)
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


def test_fresh_begin_dat_means_active_integration(tmp_path):
    d = mk(tmp_path, other_fresh=True)
    assert classify(d, attempts=0, now_ts=NOW) is ModelState.INTEGRATING
