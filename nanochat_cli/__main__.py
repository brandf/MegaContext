from __future__ import annotations

import importlib
import subprocess
import sys
from typing import List


def _bootstrap() -> None:
    """Install minimal deps needed to launch the TUI if they're missing."""
    needed = {
        "textual": "textual==0.79.1",
        "rich": "rich>=13.3.3",
        "yaml": "pyyaml>=6.0.2",
    }
    missing: List[str] = []
    for mod, spec in needed.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(spec)
    if not missing:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *missing]
    print(f"[nanochat-cli] Installing runtime deps: {' '.join(missing)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def main() -> None:
    _bootstrap()
    from .app import NanochatApp

    NanochatApp().run()


if __name__ == "__main__":
    main()
