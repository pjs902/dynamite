"""Tests for dynamite.vera.proposal - schema v1 and intake validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.proposal import (  # noqa: E402
    SCHEMA_VERSION,
    canonical_hash,
    Proposal,
    Result,
    validate_parset,
)

FAKE_BOUNDS = {
    "bh.m": {"lo": 3.90, "hi": 4.78},
    "ml": {"lo": 1.0, "hi": 6.0},
    "q": {"lo": 0.05, "hi": 0.72},  # hi already capped < qobs*u
    "p": {"lo": 0.50, "hi": 0.99},
    "u": {"lo": 0.95, "hi": 0.9999},
}
# this file's parsets use BARE names; production passes the qualified ones
BARE_SHAPE = {"q": "q", "p": "p", "u": "u"}
QOBS = 0.724
U_FIXED = 0.9999


def _par(q=0.46, p=0.90, u=None, bh=4.342, ml=2.6):
    d = {"bh.m": bh, "ml": ml, "q": q, "p": p}
    if u is not None:
        d["u"] = u
    return d


def test_schema_version_frozen():
    assert SCHEMA_VERSION == 1


def test_canonical_hash_is_key_order_insensitive():
    a = canonical_hash({"q": 0.5, "p": 0.9})
    b = canonical_hash({"p": 0.9, "q": 0.5})
    assert a == b
    assert len(a) == 16


def test_valid_fiducial_passes():
    clipped, violations = validate_parset(
        _par(), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED
    )
    assert violations == []
    assert clipped["q"] == 0.46


def test_out_of_bounds_clips_not_rejects():
    clipped, violations = validate_parset(
        _par(bh=9.0), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED
    )
    assert violations == []
    assert clipped["bh.m"] == 4.78


def test_q_above_qobs_times_u_is_violation():
    # Uncapped bounds: the config-mistake scenario the geometric gate exists
    # for. Clipping cannot repair what permissive bounds permit.
    loose = {**FAKE_BOUNDS, "q": {"lo": 0.05, "hi": 0.99}}
    _, violations = validate_parset(_par(q=0.74), loose, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED)
    assert any("q" in v for v in violations)


def test_capped_bounds_repair_geometry_silently():
    # production-style bounds: q.hi < qobs*u means clipping fixes it and no
    # violation surfaces
    clipped, violations = validate_parset(
        _par(q=0.74), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED
    )
    assert violations == []
    assert clipped["q"] == FAKE_BOUNDS["q"]["hi"]


def test_p_less_than_q_is_violation():
    _, violations = validate_parset(
        _par(q=0.8, p=0.7), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED
    )
    assert any("p" in v for v in violations)


def test_nonfinite_value_rejected():
    _, violations = validate_parset(
        _par(ml=float("nan")), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=U_FIXED
    )
    assert any("ml" in v for v in violations)


def test_roundtrip_dataclasses():
    pr = Proposal(proposal_id=canonical_hash(_par()), parset=_par())
    r = Result(
        proposal_id=pr.proposal_id,
        model_dir="orblib_001_000/ml02.40",
        status="done",
        chi2=2770837.5,
        kinchi2=1.0,
        kinmapchi2=2.0,
    )
    assert Proposal.from_dict(pr.to_dict()) == pr
    assert Result.from_dict(r.to_dict()) == r
    assert r.to_dict()["schema_version"] == 1


def test_u_free_uses_parset_u_not_fixed():
    # u free and low enough that q=0.74 becomes admissible (0.95*0.724=0.688?
    # no: 0.688 < 0.74 -> still violation; use q=0.68 instead)
    clipped, violations = validate_parset(
        _par(q=0.68, u=0.95), FAKE_BOUNDS, qobs=QOBS, shape_names=BARE_SHAPE, u_fixed=None
    )
    assert violations == []
    assert clipped["u"] == 0.95
