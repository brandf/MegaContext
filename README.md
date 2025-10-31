# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandf/MegaContext/blob/main/notebooks/megacontext.ipynb)

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

## Runtime Requirements & Setup

- Python 3.10 or newer
- CUDA-capable GPU (recommended for training)

### Local (Linux/macOS)
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and ensure `python3.10+` is available.
2. Run `uv venv` followed by `uv sync` to install the runtime and development dependencies.
3. Launch Jupyter with `uv run jupyter lab` and open [`notebooks/megacontext.ipynb`](./notebooks/megacontext.ipynb).

### Google Colab
1. Open the notebook via the badge above or [this link](https://colab.research.google.com/github/brandf/MegaContext/blob/main/notebooks/megacontext.ipynb).
2. Select a GPU runtime (`Runtime → Change runtime type → T4/L4/A100`).
3. Run the **Quick Start** bootstrap cell at the top of the notebook; it clones the repo, installs dependencies, and wires up widget support.
4. Continue through the numbered sections (environment snapshot → data prep → training).

The bootstrap script is idempotent—rerun it whenever you reconnect to a fresh Colab session.

### Artifact Storage & Resuming Runs
- Point `MEGACONTEXT_ARTIFACT_ROOT` to mounted network storage (e.g. Novita.ai volumes) before launching the notebook. All checkpoints, logs, and summaries flow there by default.
- Use `MEGACONTEXT_DATA_ROOT` if you want dataset shards on a different mount than the git checkout.
- The notebook’s **Configure Storage** cell lets you override paths interactively; it creates directories as needed.
- Checkpoint selection is handled in **Checkpoints & Resume**—pick `Do not resume` for a fresh run or choose any `.ckpt` discovered under the artifact root.
- Reproducibility defaults to seed `42`; set `MEGACONTEXT_SEED` to pin a different seed per experiment.
- Set `MEGACONTEXT_FORCE_REINSTALL=1` before running the bootstrap cell if you need to rebuild the editable install in-place (otherwise cached installs are reused to avoid Colab restarts).

---

## Development & Contribution

### Agents
  - follow the conventions in `AGENTS.md` for directory layout, testing, linting, and communication.
  - otherwise use the same workflow as a human contributor

### Humans
  - Background information:
    + Watch Karpathy videos
      * [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g)
      * [Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI)
      * [Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
    + Read [Megacontext Documentation](https://brandf.github.io/MegaContext/)
  - Follow [SETUP.md](./SETUP.md) instruction in a linux environment.
    + Such as a rented GPU from https://novita.ai
  - Use the Jupyter [notebook](./notebooks/megacontext.ipynb) for training.

---

# License

MIT [License](./LICENSE). PRs welcome.
