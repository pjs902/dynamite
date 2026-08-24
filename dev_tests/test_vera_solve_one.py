"""Tests for the array-task entry points - CLI surface and failure paths only.

The real validation for these modules is the cluster smoke run (D1); unit
scope here is intentionally narrow so nothing heavy runs on the dev box.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera import SCHEMA_VERSION  # noqa: E402
from dynamite.vera.integrate_one import main as integrate_main  # noqa: E402
from dynamite.vera.solve_one import main as solve_main  # noqa: E402


@pytest.mark.parametrize("entry", [solve_main, integrate_main])
def test_missing_config_file_fails_gracefully(entry, tmp_path, capsys):
    """Both tasks share one main() now; both must still report, not die.

    A traceback escaping here takes the array task down with an opaque
    non-zero exit; the driver charges an attempt either way, but only the
    structured line says which model and why.
    """
    rc = entry(
        [
            "--config",
            str(tmp_path / "nope.yaml"),
            "--model-dir",
            "orblib_001_000/ml02.60",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert '"error"' in err  # structured failure line
    assert "orblib_001_000/ml02.60" in err


def test_parset_schema_mismatch_is_rejected(tmp_path):
    """A worker must refuse a vera_parset.json it does not understand.

    The version field used to be written and never read; on a cluster a
    stale $PYTHONPATH can put a worker of one vintage against a driver of
    another, and silently reading `values` under the wrong schema spends a
    node-hour producing a wrong answer.
    """
    from dynamite.vera.task_model import build_model

    model_dir = "orblib_001_000/ml02.60"
    d = tmp_path / "models" / model_dir
    d.mkdir(parents=True)
    (d / "vera_parset.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION + 1,
                "par_names": ["ml"],
                "values": {"ml": 2.6},
            }
        )
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"io_settings:\n  output_directory: {tmp_path}\n")

    # The gate runs before Configuration() is built, so this stub config --
    # which is far too thin to construct one -- never gets that far. That
    # ordering is the point: a check that only fires after a successful
    # Configuration() is no check at all on the paths that matter.
    with pytest.raises(ValueError, match="schema_version"):
        build_model(str(cfg), model_dir)
