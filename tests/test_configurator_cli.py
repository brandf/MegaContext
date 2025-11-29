import os
import runpy
import sys

import pytest


def _run_configurator(argv, initial_globals):
    """Execute nanochat/configurator.py with custom argv/globals."""
    old_argv = sys.argv
    sys.argv = ["configurator.py"] + argv
    try:
        runpy.run_path(
            os.path.join("nanochat", "configurator.py"),
            init_globals=initial_globals,
        )
    finally:
        sys.argv = old_argv
    return initial_globals


def test_configurator_accepts_float_override():
    initial = {"mc_auto_batch_safety": 2.0}
    result = _run_configurator(["--mc_auto_batch_safety=2.0"], initial.copy())
    assert result["mc_auto_batch_safety"] == 2.0


def test_configurator_rejects_int_for_float():
    initial = {"mc_auto_batch_safety": 2.0}
    with pytest.raises(AssertionError):
        _run_configurator(["--mc_auto_batch_safety=2"], initial.copy())
