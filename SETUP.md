# Cloud GPU Setup (Novita.ai)

This guide walks through provisioning a Novita.ai GPU instance and running the
Phase 1 decode demo against a real base model.

## 1. Provision the instance
- Sign in to <https://novita.ai/> and add credits.
- Launch a GPU instance (RTX 4090 24 GB to begin, Ubuntu image, ≥80 GB SSD).
- Attach your SSH key (or note the password) and record the public IP.

## 2. First-time login
```bash
ssh ubuntu@<INSTANCE_IP>
sudo apt update
sudo apt install -y git curl build-essential python3-dev python3-distutils
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```

## 3. Clone the repository
```bash
mkdir -p ~/src && cd ~/src
git clone git@github.com:<your-handle>/MegaContext.git
cd MegaContext
```
(Use HTTPS if you prefer.)

## 4. Python environment
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .[dev]
uv run pre-commit install  # optional
```

## 5. Hugging Face authentication (if needed)
```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
uv run huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"
```

## 6. Sanity checks
```bash
uv run ruff check
uv run pytest --maxfail=1 --disable-warnings
```

## 7. Dataset prep smoke test
```bash
uv run python -m tools.prepare_dataset --config configs/data/sample_text.yaml
```

## 8. Run the decode demo
```bash
export MEGACONTEXT_ENABLE_WANDB=1    # optional
export WANDB_MODE=online             # or "offline"
uv run python -m tools.decode_demo --config configs/runs/base_llm.yaml
```

You should see:
- Generation output printed to stdout.
- Log file in `artifacts/run_logs/base_llm_demo-<timestamp>.log`.
- (Optional) W&B run under the `megacontext-poc` project.

## 9. Shut down when idle
- Use `nvidia-smi` to confirm utilisation during runs.
- Stop or snapshot the instance when you’re done to avoid extra charges.
