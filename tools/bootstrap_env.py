"""Bootstrap a development environment for the MegaContext POC."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMANDS: list[list[str]] = [
    ["uv", "venv"],
    ["uv", "pip", "install", "-r", "requirements.txt"],
    ["uv", "pip", "install", "-e", ".[dev]"],
    ["uv", "run", "pre-commit", "install"],
]


def run(command: list[str], *, dry_run: bool = False) -> None:
    display = " ".join(command)
    print(f"→ {display}")
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def ensure_uv_available() -> None:
    if shutil.which("uv"):
        return
    print("uv is required but was not found on PATH.", file=sys.stderr)
    print(
        "Install from https://github.com/astral-sh/uv and re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap MegaContext dev environment."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    ensure_uv_available()

    for command in COMMANDS:
        run(command, dry_run=args.dry_run)

    print("\nEnvironment ready! Suggested next steps:")
    print("  uv run ruff check")
    print("  uv run pytest --maxfail=1 --disable-warnings")


if __name__ == "__main__":
    main()
