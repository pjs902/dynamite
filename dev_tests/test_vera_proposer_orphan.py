"""A generator row that never becomes a proposal is lost from the campaign."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_vera_proposer_gridwalk import build_minimal_config  # noqa: E402

from dynamite.vera.proposer_gridwalk import GridWalkProposer  # noqa: E402


def test_no_generator_row_is_orphaned():
    """Every row the generator appends must get a proposal_id.

    A row left behind has no pid and no directory: scan() skips it,
    quorum_pending() never counts it, and the next propose() starts past
    it -- the parameter set silently vanishes.
    """
    cfg = build_minimal_config()
    prop = GridWalkProposer(cfg)
    t = cfg.all_models.table
    before = len(t)
    props = prop.propose(max_batch=1)  # advisory only
    appended = len(t) - before
    assert appended > 1, "need a multi-row generator batch to exercise this"
    assert len(props) == appended, f"{appended - len(props)} row(s) orphaned"
    for i in range(before, len(t)):
        assert i in prop.pid_to_row.values(), f"row {i} has no proposal_id"
