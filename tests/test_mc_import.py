import importlib


def test_mc_controller_importable():
    mc_module = importlib.import_module("mc.runtime")
    assert hasattr(mc_module, "MCController"), "MCController missing from mc.runtime"
