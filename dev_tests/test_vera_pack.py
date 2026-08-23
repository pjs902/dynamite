"""Tests for dynamite.vera.pack_integration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamite.vera.pack_integration import pack_libraries  # noqa: E402


def test_groups_of_twelve_default():
    dirs = [f"m{i}" for i in range(29)]
    packs = pack_libraries(dirs)
    assert [len(p) for p in packs] == [12, 12, 5]


def test_order_preserved():
    dirs = [f"m{i}" for i in range(15)]
    flat = [m for p in pack_libraries(dirs) for m in p]
    assert flat == dirs


def test_single_library_node_budget():
    assert pack_libraries(["only"], procs_per_lib=72, cores=72) == [["only"]]
