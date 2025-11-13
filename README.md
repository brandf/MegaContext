# MegaContext — Learned Context Compression & Focus for Frozen LLMs

*A system architecture for virtualized LLM memory.*

---

## TL;DR — MegaContext
MegaContext virtualizes an LLM’s context window by pairing an unbounded MegaContext Tree with a fixed-size Working Context. Learned components (GistNet, LensNet, Focus Allocator) keep the GPU budget constant while swapping detail in and out of focus.

For the complete narrative, start with [`obsidian/index.md`](./obsidian/index.md) and then read [`obsidian/getting started/How MegaContext Works.md`](./obsidian/getting%20started/How%20MegaContext%20Works.md). This README focuses on repo logistics and runtime setup.

---

## Documentation

- Public site: [brandf.github.io/MegaContext](https://brandf.github.io/MegaContext/)
- Obsidian vault: `obsidian/`
- Implementation roadmap: `obsidian/plans/Implementation Roadmap.md`
- Current work tracker: `obsidian/TODO.md`

---

## Runtime Requirements & Setup

Baseline requirements (matches upstream nanochat):

- Python 3.11 + CUDA 12.x drivers
- `uv` for dependency management (`uv venv && uv sync --extra gpu`)
- WANDB/HF tokens exported before long runs

The detailed operating guide (env prep, telemetry, troubleshooting) lives in [`obsidian/ops/Training & Operations.md`](./obsidian/ops/Training%20&%20Operations.md). Use it as the single source of truth. Quick reference:

| Scenario | Command |
| --- | --- |
| Single GPU (32 GB class) | `bash run10.sh --gpu 5090` |
| Single H100 | `bash run10.sh --gpu h100` |
| $100 tier (8×H100) | `bash speedrun.sh` |
| $1000 tier (8×H100) | `bash run1000.sh` |

These scripts run tokenizer → base → mid → chat SFT end-to-end, drop checkpoints in `~/.cache/nanochat`, and generate `report/report.md`. For chat/web demos after training, follow [[Base Runtime]](./obsidian/ops/Base%20Runtime.md). New box? Run `./mc_setup` once and follow the prompts— it installs `uv`, syncs deps, installs OTEL exporters, and writes `.mc_env` so `run10.sh` inherits your WANDB/HF/telemetry settings automatically.

> 🆕 Set `--mc` (e.g. `bash run10.sh --gpu 5090 --mc`) to enable the MegaContext controller. The knobs below configure the tree/working-context components and their auxiliary losses:

- `MCController.process_batch(...)` now returns `cached_embeddings` alongside the positional cache and auxiliary losses, so advanced callers can reuse the exact `[B, T, D]` embeddings when experimenting with counterfactual Working Context edits without re-running the token embedding layer.
- The same call also exposes `positional_caches`, a dict keyed by session id that stores per-sample `(cos, sin, alibi)` tuples; `scripts/base_train.py` now stitches these into batched overrides so MC RoPE applies even when `B > 1`.
> - `--block_size` (default 32) controls how many tokens feed each gist.
> - `--gistnet_type transformer|mean`, `--gistnet_layers {2,4}`, `--gistnet_pooling mean|query|cls`, `--gistnet_head linear|mlp` (defaults transformer/2/mean/mlp).
> - `--lensnet_type transformer`, `--lensnet_layers {2,4,8}`, `--lensnet_head linear|mlp` (defaults transformer/2/mlp).
> - `--allocator greedy|stochastic_greedy` picks the focus policy. `stochastic_greedy` samples among the top-|score| candidates (tune via `--allocator_sample_top_k`, `--allocator_sample_temperature`) while respecting recent-token protection and soft length. Advanced knobs—`--allocator_soft_max`, `--allocator_recent_tokens`, `--allocator_expand_threshold`, `--allocator_collapse_threshold`, `--allocator_max_replacements`, `--allocator_iterations`—are exposed through `scripts/base_train.py` for fine-grained control.
> - `--mc_tree ram` selects the MegaContext backing store (current release accepts only `ram`). A future `disk` mode is reserved for a MegaCache-backed implementation.
> - `--mc_num_random_variants`, `--mc_train_wc_length`, `--mc_max_counterfactuals` tune how many random WC compressions we sample per sequence and how aggressively each one shrinks toward the training working-context length.
> - MC training now samples a baseline WC plus `N` random variants per sequence and applies the standard next-token loss to each, so you can focus on how many variants you want rather than juggling extra horizon knobs.
> - `--mc_auto_batch 1` (default) automatically shrinks `device_batch_size` and increases `num_iterations` based on the variant multiplier so flipping `--mc` on doesn’t unexpectedly OOM; set it to `0` if you want to control batch math yourself.
> - `--mc_eval_soft_max_length`, `--mc_infer_allocator_*`, `--mc_infer_refocus_interval` let you tune the validation/inference WC size and refocus cadence independently from training so reported `val/bpb` matches runtime behavior.
> - `--mc_lens_loss_weight` adjusts how strongly LensNet supervision contributes alongside the core next-token objective.
> - `--mc_aux_dtype auto|fp32|bf16` controls the precision used for GistNet/LensNet. `auto` picks bf16 on GPUs that support it (H100/A100) and falls back to fp32 elsewhere.
> LensNet scores are tanh-clamped floats (positive ⇒ expand, negative ⇒ collapse). When `--mc` is active we also build Gaussian RoPE positional caches using MegaContext global positions/LOD metadata. The flagless path continues to match upstream nanochat.

When comparing MC-enabled vs. vanilla runs, normalize by tokens, FLOPs, or wall-clock time rather than raw `step` counts (MC batches perform more work per update). `scripts/base_train.py` already logs `total_training_flops`, `total_training_time`, and `mc/*` loss metrics so you can overlay both training curves fairly in W&B or Grafana.

Telemetry: the training scripts now instantiate the built-in OpenTelemetry provider (OTLP exporter) whenever `--mc` is set. Point it at Tempo/Grafana (or any OTLP-compatible backend) by exporting `MC_OTEL_ENDPOINT` (e.g., `http://localhost:4318`) and `MC_OTEL_INSECURE=1` if needed. Each MC session emits structured spans (`mc_tree_snapshot`, `working_context_snapshot`, `focus_allocator`, `mc_timing`, etc.) that you can visualize alongside WANDB metrics, and `scripts/base_train.py` logs `mc/time_controller_ms` so you can catch controller slowdowns.

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
  - Follow [SETUP.md](./SETUP.md) instruction in a linux environment (Python 3.11 + CUDA 12.x as noted above).
    + Such as a rented GPU from https://novita.ai
  - Use the nanochat scripts (`run10.sh`, `speedrun.sh`, `run1000.sh`, or the component-level `scripts/*.py` modules) for training and evaluation.

---

# License

MIT [License](./LICENSE). PRs welcome.

---
This is a fork of:

# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Talk to it

To get a sense of the endpoint of this repo, you can currently find [nanochat d32](https://github.com/karpathy/nanochat/discussions/8) hosted on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/). "d32" means that this model has 32 layers in the Transformer neural network. This model has 1.9 billion parameters, it was trained on 38 billion tokens by simply running the single script [run1000.sh](run1000.sh), and the total cost of training was ~$800 (about 33 hours training time on 8XH100 GPU node). While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of modern Large Language Models like GPT-5. When talking to these micro models, you'll see that they make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing. But what makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end. To train and talk to your own, we turn to...

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Running on CPU / MPS

nanochat can be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

## Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

Additionally, to add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── configurator.py             # A superior alternative to argparse
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── rustbpe                         # Custom Rust BPE tokenizer trainer
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # see for why this even exists
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_rustbpe.py
└── uv.lock
```

## Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
