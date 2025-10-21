import os
import sys

import megacontext.utils.env as env


def clear_env_caches() -> None:
    env.in_colab_env.cache_clear()
    env.in_notebook_env.cache_clear()


def test_in_notebook_env_false_when_ipython_missing(monkeypatch) -> None:
    clear_env_caches()
    monkeypatch.setattr(env, "get_ipython", lambda: None, raising=False)
    assert env.in_notebook_env() is False


def test_in_notebook_env_true_for_jupyter(monkeypatch) -> None:
    clear_env_caches()

    shell_type = type("ZMQInteractiveShell", (), {})
    shell_instance = shell_type()
    monkeypatch.setattr(env, "get_ipython", lambda: shell_instance, raising=False)

    assert env.in_notebook_env() is True


def test_in_colab_env_detects_module(monkeypatch) -> None:
    clear_env_caches()
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert env.in_colab_env() is True
    monkeypatch.delitem(sys.modules, "google.colab")
    clear_env_caches()
    os.environ.pop("COLAB_GPU", None)
    assert env.in_colab_env() is False
