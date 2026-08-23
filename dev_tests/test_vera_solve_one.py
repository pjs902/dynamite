"""Tests for dynamite.vera.solve_one - CLI surface and failure paths only.

The real validation for this module is the cluster smoke run (D1); unit
scope here is intentionally narrow so nothing heavy runs on the dev box.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.solve_one import main  # noqa: E402


def test_dry_run_emits_json_and_exits_zero(capsys):
    rc = main(
        [
            "--config",
            "unused.yaml",
            "--model-dir",
            "orblib_001_007/ml02.80",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed == {"model_dir": "orblib_001_007/ml02.80", "dry_run": True}


def test_missing_config_file_fails_gracefully(tmp_path, capsys):
    rc = main(
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

