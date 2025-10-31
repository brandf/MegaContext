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
