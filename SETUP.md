# Cloud GPU Setup (Novita.ai)

This guide walks through provisioning a Novita.ai GPU instance and running the
Jupyter notebook for training. The environment mirrors the upstream nanochat
requirements we plan to import: **Python 3.11**, **PyTorch 2.2+ (cu121)**, and a
CUDA 12.x-capable GPU/driver stack.

## 1. Provision the instance
- Sign in to <https://novita.ai/> and add credits.
- Launch a GPU instance.
- Attach your SSH key (or note the password) and record the public IP.

## 2. First-time login
```bash
ssh ubuntu@<INSTANCE_IP>
sudo apt update
sudo apt install -y git curl build-essential python3.11 python3.11-dev python3-distutils
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```

## 3. Clone the repository
```bash
mkdir -p ~/src && cd ~/src
git clone git@github.com:brandf/MegaContext.git
cd MegaContext
```

## 4. Bootstrap the repository

Create the venv, install deps, and run a quick smoke of tests locally:

```bash
uv venv
uv sync --extra gpu
source .venv/bin/activate
uv run pytest -q
```

Operational guidance (training recipes, telemetry, troubleshooting) now lives in `obsidian/ops/Training & Operations.md` and `README.md`. Follow those for `run10.sh`/`speedrun.sh`/`run1000.sh` usage.

## 5. Shut down when idle
- Use `nvidia-smi` to confirm utilisation during runs.
- Stop or snapshot the instance when you’re done to avoid extra charges.
