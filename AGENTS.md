# Repository Guidelines

## Project Structure & Module Organization
- `README.md` holds the definitive architecture spec; update it alongside any code changes so the high-level contract stays current.
- Keep visual assets such as `megacontext.png` (and future diagrams) in an `assets/` subtree; reference them from docs with relative paths.
- When adding implementation work, organize runtime code under `src/` and supporting experiments under `research/`; mirror that structure in `tests/` so each module has a focused test target.

## Build, Test, and Development Commands
- Prefer Python-native workflows: document setup via `uv` or `poetry` (e.g., `uv venv`, `uv pip install -r requirements.txt`, `uv run python -m <module>`).
- Provide CLI entry points or `python -m` commands for routine tasks (`python -m tools.format`, `uv run ruff check`, `uv run black .`), and consolidate common tasks into scripts under `tools/`.
- Use `pytest` for unit and integration coverage: standard command is `uv run pytest --maxfail=1 --disable-warnings`.
- Prototype notebooks should rely on `uv` or `poetry` lockfiles—document any new dependencies in `pyproject.toml` and regenerate the lock before committing.

## Coding Style & Naming Conventions
- Default to Python 3.11; enforce `ruff` + `black` via pre-commit to keep imports sorted and code wrapped at 88 columns.
- Name modules with snake_case (`lens_controller.py`), classes in PascalCase (`LensController`), and public functions in snake_case with verb-first names (`allocate_focus()`).
- Store configuration schemas as `.yaml` under `configs/` and document each field inside the file; prefer explicit enums over booleans for allocator modes.

## Testing Guidelines
- Mirror each `src/` module with a `tests/test_<module>.py` file and include both happy-path and failure-path assertions.
- Target >=90% statement coverage for core allocator and summarizer logic; measure via `pytest --cov=src --cov-report=term-missing`.
- When adding models, include deterministic smoke tests that seed RNG (`PYTHONHASHSEED=0`, `torch.manual_seed(42)`) to stabilize CI results.

## Commit & Pull Request Guidelines
- Follow the existing imperative, capitalized commit style (`Enhance README with...`); group related edits into a single logical commit.
- PRs must describe the problem, the solution, and validation steps; attach terminal output for the canonical test command (e.g., `uv run pytest --maxfail=1 --disable-warnings`) and relevant plots or tensors as screenshots when behavior changes.
- Link issues with `Fixes #ID` to trigger auto-closure, and request review from both architecture and implementation maintainers when changes touch shared interfaces.
