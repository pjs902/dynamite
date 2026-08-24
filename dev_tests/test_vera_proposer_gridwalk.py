"""Tests for dynamite.vera.proposer_gridwalk - mock-driven, no real configs.

Mock pattern proven on the bayesopt branch: real ParameterSpace over a
MockSystem carrying real Parameter objects; MockAllModels with the standard
astropy columns. GridWalk.generate() runs for real against these mocks.
"""

import os
import sys
import types

import numpy as np  # noqa: F401
import pytest
from astropy.table import Table, Column

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.parameter_space import (  # noqa: E402
    Parameter,
    ParameterSpace,
)
from dynamite.vera.proposer_gridwalk import GridWalkProposer  # noqa: E402


# --------------------------------------------------------------------------
# Mock infrastructure (bayesopt-branch pattern)
# --------------------------------------------------------------------------
class MockComponent:
    def __init__(self, name="cmp"):
        self.name = name
        self.parameters = []

    def get_parname(self, name):
        return name

    def validate_parset(self, par):
        return True


class MockSystem:
    def __init__(self, params):
        cmp = MockComponent("stars")
        cmp.parameters = list(params)
        self.cmp_list = [cmp]
        self.parameters = []

    def validate_parset(self, par):
        return True


def _mk_param(name, lo, hi, value, step=0.1, fixed=False):
    return Parameter(
        name=name,
        fixed=fixed,
        logarithmic=False,
        value=value,
        par_generator_settings={"lo": lo, "hi": hi, "step": step},
    )


def build_minimal_config(n_max_mods=12):
    ml = _mk_param("ml", 1.0, 6.0, 2.6, step=0.2)
    q = _mk_param("q", 0.30, 0.72, 0.46, step=0.04)
    p = _mk_param("p", 0.50, 0.99, 0.90, step=0.03)
    pspace = ParameterSpace(MockSystem([ml, q, p]))
    pss = {
        "generator_type": "GridWalk",
        "which_chi2": "kinchi2",
        "stopping_criteria": {
            "n_max_mods": n_max_mods,
            "n_max_iter": 200,
            "min_delta_chi2_abs": 0.01,
        },
    }

    class FakeSettings:
        parameter_space_settings = pss

    class FakeAllModels:
        def __init__(self):
            names = ["ml", "q", "p"]
            t = Table()
            for n in names:
                t[n] = Column([], dtype=float)
            t["chi2"] = Column([], dtype=float)
            t["kinchi2"] = Column([], dtype=float)
            t["kinmapchi2"] = Column([], dtype=float)
            t["time_modified"] = Column([], dtype="U64")
            t["orblib_done"] = Column([], dtype=bool)
            t["weights_done"] = Column([], dtype=bool)
            t["all_done"] = Column([], dtype=bool)
            t["which_iter"] = Column([], dtype=int)
            t["directory"] = Column([], dtype="U256")
            # seed iteration-0 center row so the walker has a center
            t.add_row(
                [
                    2.6,
                    0.46,
                    0.90,
                    np.nan,
                    np.nan,
                    np.nan,
                    "",
                    False,
                    False,
                    False,
                    0,
                    "",
                ]
            )
            self.table = t

    class FakeConfig:
        settings = FakeSettings()
        all_models = FakeAllModels()
        parspace = pspace

    return FakeConfig()


@pytest.fixture()
def minimal_config():
    return build_minimal_config()


def _solve_all(strat):
    t = strat.config.all_models.table
    for pid, row in list(strat.pid_to_row.items()):
        t["chi2"][row] = 100.0 + row
        t["kinchi2"][row] = 90.0 + row
        t["kinmapchi2"][row] = 95.0 + row
        t["all_done"][row] = True


def test_propose_returns_new_rows_as_proposals(minimal_config):
    strat = GridWalkProposer(minimal_config)
    props = strat.propose()
    assert len(props) >= 1
    assert len({p.proposal_id for p in props}) == len(props)
    for p in props:
        assert set(p.parset) == {"ml", "q", "p"}


def test_quorum_counts_only_tracked_unsolved(minimal_config):
    strat = GridWalkProposer(minimal_config)
    props = strat.propose()
    n = len(props)
    assert strat.quorum_pending() == n
    # solve exactly one tracked proposal directly in the table
    pid, row = next(iter(strat.pid_to_row.items()))
    strat.config.all_models.table["all_done"][row] = True
    strat.config.all_models.table["chi2"][row] = 1.0
    strat.config.all_models.table["kinchi2"][row] = 1.0
    strat.config.all_models.table["kinmapchi2"][row] = 1.0
    assert strat.quorum_pending() == n - 1


def test_exhausted_on_n_max_mods(minimal_config):
    strat = GridWalkProposer(minimal_config)
    assert not strat.exhausted()
    t = minimal_config.all_models.table
    t["all_done"][0] = True
    for _ in range(11):  # reach n_max_mods=12 done rows
        t.add_row(
            [2.6, 0.46, 0.9, 1.0, 1.0, 1.0, "", True, True, True, 1, f"x{len(t)}"]
        )
    assert strat.exhausted()


def test_wrong_generator_type_rejected(minimal_config):
    minimal_config.settings.parameter_space_settings["generator_type"] = (
        "LegacyGridSearch"
    )
    with pytest.raises(ValueError, match="generator_type"):
        GridWalkProposer(minimal_config)


def test_observe_is_noop_but_channel_consistent(minimal_config):
    strat = GridWalkProposer(minimal_config)
    strat.propose()
    assert strat.observe([]) is None
    assert strat.pid_to_row  # the table is the observation channel
    assert not strat.ready_to_propose()  # nothing solved yet
