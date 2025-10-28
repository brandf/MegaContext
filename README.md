# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

---

## TL;DR — MegaContext
MegaContext virtualizes context by pairing a disk-backed gist tree called the MegaContext Tree with a budgeted working context governed by GistNet, LensNet, and the Focus Allocator.

It separates a model’s context into a MegaContext Tree (stored on disk) and a Working Context (on GPU). A learned GistNet model is used to build the MegaContext Tree as a hierarchy of gists. The working context compresses the MegaContext Tree into a fixed-size mix of tokens and gists that are used for inference.

To dynamically adapt level of detail, a learned LensNet model, continuously/incrementally refocuses the MegaContext Tree onto the Working Context, giving the model effectively infinite memory at constant compute with automatic context management.

---

## Documentation

You can view/read the documentation [here](https://brandf.github.io/MegaContext/).

For the best editing/contributing experience open in Obsidian client, open a the vault in the `obsidian/` folder;


---

## Development & Contribution

- Follow the conventions in `AGENTS.md` for directory layout, testing, linting, and communication.
- Bootstrap: `uv venv`, `uv sync`, then `uv run pytest --maxfail=1 --disable-warnings` for smoke tests. Lint/format via `uv run ruff check src tests` and `uv run black src tests`.
- Tooling and demos live under `tools/`; see the corresponding Obsidian notes for command examples.
- Update progress in the Obsidian plan notes before hand-off so the next contributor has full context.

---

# License

MIT License (suggested). PRs welcome—please include reproducible tests for GistNet, LensNet, the focus allocator, and end-to-end demos.

---

*MegaContext virtualizes sequence memory just as MegaTexture virtualized textures—focusing detailed computation only where needed. It opens a path to persistent, updatable, and truly lifelong language models.*
