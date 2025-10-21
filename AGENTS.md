# Repository Guidelines

## Project Structure & Module Organization
- `README.md` is the architecture contract—always start a new chat by reading it fully.  Revise it alongside behavior changes. Active workstreams are tracked in `planning/POC_PLAN.md`, `planning/PAPER_PLAN.md`, and `planning/FUTURE_PLAN.md`; note progress and new follow-ups before hand-off.
- Runtime code lives in `src/` (e.g., `src/focus/allocator.py`) with mirror tests in `tests/`. Keep exploratory notebooks in `research/` until they stabilize.
- House shared diagrams under `assets/` (e.g., `assets/megacontext.png`); reference images with relative paths.
- Store configuration schemas in `configs/` as YAML and document each field inline, preferring enums for mode switches.

## Build, Test, and Development Commands
- `uv venv` creates the local environment; follow with `uv sync` (or `uv pip install -r requirements.txt`) to install dependencies.
- `uv run pytest --maxfail=1 --disable-warnings` is the canonical test entry point. Extend with `--cov=src --cov-report=term-missing` before publishing coverage figures.
- `uv run ruff check src tests` enforces linting, and `uv run black src tests` keeps formatting at the project-wide 88-column limit.
- `uv run python tools/bootstrap_env.py` provisions demo assets, and `uv run python tools/decode_demo.py --help` exposes the end-to-end inference harness.

## Coding Style & Naming Conventions
- Target Python 3.11 semantics. Keep imports sorted and code formatted via the `ruff` + `black` toolchain; wire both through pre-commit when you touch configuration.
- Modules use snake_case names, classes use PascalCase, and public functions begin with an action verb in snake_case (e.g., `allocate_focus()`).

## Testing Guidelines
- Mirror new runtime modules with `tests/test_<module>.py`, covering both success and failure paths. Use parametrization to cover focus edge cases.
- Maintain ≥90 % statement coverage for focus allocation and summarization flows (`uv run pytest --cov=src --cov-report=term-missing`).
- Seed randomness (`PYTHONHASHSEED=0`, `torch.manual_seed(42)`) in any model-facing tests to stabilize CI.

## Commit & Pull Request Guidelines
- Commits follow imperative, capitalized subjects (e.g., `Align Allocator API`), combining related changes into one reviewable unit.
- Pull requests must summarize the problem, outline the solution, and paste the latest `uv run pytest --maxfail=1 --disable-warnings` output. Attach plots or tensors when behavior shifts.
- Link issues with `Fixes #ID` and request review from both architecture and implementation maintainers whenever shared interfaces move.

## Documentation & Communication
- Update `README.md` diagrams or references when APIs change, adding new visuals under `assets/`.
- Record roadmap shifts in the relevant `planning/*.md` file so the next agent can resume without rediscovery.
