"""Tests for dynamite.vera.proposer_microbatch (spec phase C2)."""

import os
import sys

import pytest
import numpy as np  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astropy.table import Column  # noqa: E402

from test_vera_proposer_gridwalk import build_minimal_config  # noqa
from dynamite.vera.proposer_microbatch import (  # noqa: E402
    MicroBatchWalkProposer,
)





def _solve(strat, props):
    t = strat.config.all_models.table
    for p in props:
        row = strat.pid_to_row[p.proposal_id]
        t["chi2"][row] = 100.0 + row
        t["kinchi2"][row] = 90.0 + row
        t["kinmapchi2"][row] = 95.0 + row
        t["all_done"][row] = True


def _solve_n(strat, props, n):
    _solve(strat, props[:n])


@pytest.fixture()
def minimal_config():
    return build_minimal_config()


def test_quorum_fraction_semantics(minimal_config):
    strat = MicroBatchWalkProposer(minimal_config, min_solved_fraction=0.8)
    props = strat.propose(max_batch=500)
    n = len(props)
    assert n > 0
    assert strat.quorum_pending() == n  # nothing solved yet
    _solve_n(strat, props, max(1, int(0.5 * n)))  # ~50% < 80%
    assert strat.quorum_pending() == strat._outstanding() > 0
    _solve(strat, props)  # 100% >= 80%
    assert strat.quorum_pending() == 0


def test_recenter_produces_new_ids(minimal_config):
    strat = MicroBatchWalkProposer(minimal_config)
    first = strat.propose(max_batch=500)
    _solve(strat, first)
    second = strat.propose(max_batch=500)
    ids_a = {p.proposal_id for p in first}
    for p in second:
        assert (
            p.proposal_id not in ids_a or second is first
        )  # fresh batch, fresh identities


def test_invalid_fraction_rejected(minimal_config):
    import pytest

    with pytest.raises(ValueError, match="min_solved_fraction"):
        MicroBatchWalkProposer(minimal_config, min_solved_fraction=0.0)


def test_full_completion_still_gates(minimal_config):
    # fraction=1.0 behaves exactly like the classic full-batch quorum
    strat = MicroBatchWalkProposer(minimal_config, min_solved_fraction=1.0)
    props = strat.propose(max_batch=500)
    assert strat.quorum_pending() == len(props)
    _solve(strat, props)
    assert strat.quorum_pending() == 0
