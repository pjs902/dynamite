"""Tests for dynamite.vera.proposer_bayesopt (spec phase C3).

Skipped cleanly when the BO stack is absent from the environment; the
adapter itself only needs botorch at generate()-time, so construction
tests exercise the mapping logic with the generator stubbed where noted.
"""

import os
import sys
import types

import pytest
from astropy.table import Column, Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import botorch  # noqa: F401,E402
    from dynamite.parameter_space import (  # noqa: E402
        Parameter,
        ParameterSpace,
    )
except Exception as _e:  # noqa: BLE001 - any import-order ABI failure counts
    # Known host quirk: once torch loads its own libstdc++/libicu, the
    # subsequent pymc->sqlite3 C-extension import can fail an ABI check -
    # and depending on which side imports first, different exception types
    # surface. The BO stack gets a clean env on VERA (spec section 7).
    pytest.skip(f"BO stack unusable on this host: {_e!r}",
                allow_module_level=True)

from dynamite.vera.proposer_bayesopt import BayesOptProposer  # noqa: E402


class MockComponent:
    def __init__(self, name="stars"):
        self.name = name
        self.parameters = []

    def get_parname(self, name):
        return name

    def validate_parset(self, par):
        return True


MockComponent.__name__ = "TriaxialVisibleComponent"
MockComponent.__qualname__ = "TriaxialVisibleComponent"


class MockSystem:
    def __init__(self, params):
        cmp = MockComponent()
        cmp.parameters = list(params)
        self.cmp_list = [cmp]
        self.parameters = []

    def validate_parset(self, par):
        return True


def _mk_param(name, lo, hi, value, step=0.1):
    return Parameter(
        name=name,
        fixed=False,
        logarithmic=False,
        value=value,
        par_generator_settings={"lo": lo, "hi": hi, "step": step},
    )


@pytest.fixture()
def bo_config():
    params = [
        _mk_param("ml", 4.0, 6.0, 5.0),
        _mk_param("q", 0.30, 0.72, 0.5),
        _mk_param("p", 0.55, 0.99, 0.9),
    ]
    pspace = ParameterSpace(MockSystem(params))
    pss = {
        "generator_type": "BayesOpt",
        "which_chi2": "kinchi2",
        "stopping_criteria": {
            "n_max_mods": 10,
            "n_max_iter": 20,
            "min_delta_chi2_abs": 0.01,
        },
        "bayesopt_settings": {"batch_size": 2, "n_initial_random": 2},
    }
    t = Table()
    for n in ("ml", "q", "p"):
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

    class FakeAllModels:
        pass

    class FakeSettings:
        parameter_space_settings = pss

    class FakeConfig:
        pass

    fam = FakeAllModels()
    fam.table = t
    fcfg = FakeConfig()
    fcfg.settings = FakeSettings()
    fcfg.all_models = fam
    fcfg.parspace = pspace
    return fcfg


def test_wrong_generator_type_rejected(bo_config):
    bo_config.settings.parameter_space_settings["generator_type"] = "GridWalk"
    with pytest.raises(ValueError, match="generator_type"):
        BayesOptProposer(bo_config)


def test_exhausted_reads_status_flags(bo_config, monkeypatch):
    strat = BayesOptProposer.__new__(BayesOptProposer)  # skip heavy __init__
    strat.generator = types.SimpleNamespace(status={"gp_predictions_accurate": False})
    strat.config = bo_config
    assert not strat.exhausted()
    strat.generator.status["gp_predictions_accurate"] = True
    assert strat.exhausted()


def test_quorum_counts_tracked_unsolved_rows_only(bo_config):
    strat = BayesOptProposer.__new__(BayesOptProposer)
    strat.config = bo_config
    strat.par_names = ["ml", "q", "p"]
    strat.pid_to_row = {}
    strat.log = __import__("logging").getLogger("test")
    t = bo_config.all_models.table
    t.add_row([5.0, 0.5, 0.9, 1.0, 1.0, 1.0, "", True, True, True, 0, "d0"])
    t.add_row([5.2, 0.6, 0.95, 2.0, 2.0, 2.0, "", False, False, False, 0, "d1"])
    from dynamite.vera.proposal import canonical_hash

    strat.pid_to_row = {
        canonical_hash({"ml": 5.0, "q": 0.5, "p": 0.9}): 0,
        canonical_hash({"ml": 5.2, "q": 0.6, "p": 0.95}): 1,
    }
    assert strat.quorum_pending() == 1  # only row 1 unsolved
