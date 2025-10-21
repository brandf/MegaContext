"""
Environment detection helpers shared across the project.

These functions let runtime modules differentiate between notebook (Colab/Jupyter)
and CLI contexts without sprinkling ad-hoc checks throughout the codebase.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

try:  # pragma: no cover - optional dependency
    from IPython import get_ipython  # type: ignore
except ImportError:  # pragma: no cover - IPython not installed
    get_ipython = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def in_colab_env() -> bool:
    """
    Return ``True`` when running inside Google Colab.

    Detection is based on the ``google.colab`` module import or the presence of the
    ``COLAB_GPU`` environment variable that Colab sets by default.
    """

    if "google.colab" in sys.modules:
        return True
    return os.environ.get("COLAB_GPU") is not None


@lru_cache(maxsize=1)
def in_notebook_env() -> bool:
    """
    Return ``True`` when executing inside a Jupyter-style notebook shell.

    The check looks for IPython's interactive shells (`ZMQInteractiveShell` for
    Jupyter/Colab). Plain Python executables and terminal IPython shells return
    ``False`` so CLI workflows remain unaffected.
    """

    if get_ipython is None:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    shell_name = ip.__class__.__name__
    if shell_name in {"ZMQInteractiveShell", "Shell"}:
        return True
    return False


__all__ = ["in_colab_env", "in_notebook_env"]
