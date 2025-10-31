#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/setup_megacontext.sh [options]

Interactive helper for provisioning a MegaContext development environment.

Options:
  -y, --yes                Run non-interactively with default answers (where safe).
  --install-uv             Force installation of uv even if already present.
  --skip-tests             Skip linting and pytest smoke tests.
  --huggingface-login [TOKEN]
                           Run ``huggingface-cli login`` (reads token from argument,
                           prompt, or existing environment variable).
  --prep-dataset           Prepare a dataset shard (default config unless overridden).
  --dataset-config PATH    Dataset config to use when preparing shards.
  --decode-demo            Run the base decode demo after setup.
  --enable-wandb           Export MEGACONTEXT_ENABLE_WANDB=1 during demo.
  --help                   Show this message.

Examples:
  bash scripts/setup_megacontext.sh --prep-dataset --dataset-config configs/Gutenberg_SmolLM3.yaml
  bash scripts/setup_megacontext.sh -y --skip-tests --decode-demo
EOF
}

AUTO_CONFIRM=false
FORCE_INSTALL_UV=false
RUN_TESTS=true
RUN_DATASET=false
DATASET_CONFIG="configs/SampleText_TinyGPT2.yaml"
RUN_DEMO=false
ENABLE_WANDB=false
RUN_HF_LOGIN=false
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) AUTO_CONFIRM=true ;;
  --install-uv) FORCE_INSTALL_UV=true ;;
  --skip-tests) RUN_TESTS=false ;;
  --prep-dataset) RUN_DATASET=true ;;
  --dataset-config)
    shift || { echo "Missing value for --dataset-config" >&2; exit 1; }
    DATASET_CONFIG="$1"
    ;;
  --decode-demo) RUN_DEMO=true ;;
  --enable-wandb) ENABLE_WANDB=true ;;
  --huggingface-login)
    RUN_HF_LOGIN=true
    if [[ $# -gt 1 ]]; then
      case "$2" in
        -*) ;;
        *)
          HF_TOKEN="$2"
          shift
          ;;
      esac
    fi
    ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

confirm() {
  local prompt="$1"
  local default="${2:-y}"
  local answer
  if "$AUTO_CONFIRM"; then
    [[ "$default" =~ ^[Yy]$ ]]
    return
  fi
  read -r -p "${prompt} " answer
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy]$ ]]
}

run_step() {
  local description="$1"
  shift
  echo ""
  echo "=== ${description} ==="
  "$@"
}

ensure_command() {
  local cmd="$1"
  local install_hint="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  echo "Command '$cmd' is required."
  if confirm "Install $cmd now? [Y/n]" "y"; then
    eval "$install_hint"
  else
    echo "Cannot continue without '$cmd'." >&2
    exit 1
  fi
}

cd "$REPO_ROOT"

echo "MegaContext setup helper"
echo "Repository root: $REPO_ROOT"

INSTALL_UV=false
if "$FORCE_INSTALL_UV" || ! command -v uv >/dev/null 2>&1; then
  INSTALL_UV=true
fi

if "$INSTALL_UV"; then
  if confirm "Install uv package manager? [Y/n]" "y"; then
    run_step "Installing uv" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "uv installation skipped; please ensure it is available before continuing."
  fi
fi

ensure_command "uv" "curl -LsSf https://astral.sh/uv/install.sh | sh"

if [[ ! -d ".venv" ]]; then
  if confirm "Create uv virtual environment (.venv)? [Y/n]" "y"; then
    run_step "Creating virtual environment" uv venv
  fi
fi

run_step "Synchronising dependencies via uv sync" uv sync

if confirm "Install project in editable mode with dev extras? [Y/n]" "y"; then
  run_step "Installing project" uv pip install -e .[dev]
fi

if confirm "Install pre-commit hooks? [y/N]" "n"; then
  run_step "Installing pre-commit hooks" uv run pre-commit install
fi

if "$RUN_TESTS"; then
  if confirm "Run lint (ruff) and pytest smoke tests? [Y/n]" "y"; then
    run_step "Running ruff check" uv run ruff check
    run_step "Running pytest smoke tests" uv run pytest --maxfail=1 --disable-warnings
  fi
fi

if ! "$RUN_HF_LOGIN" && ! "$AUTO_CONFIRM"; then
  if confirm "Login to Hugging Face now? [y/N]" "n"; then
    RUN_HF_LOGIN=true
  fi
fi

if "$RUN_HF_LOGIN"; then
  if [[ -z "$HF_TOKEN" ]]; then
    if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
      HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
    elif "$AUTO_CONFIRM"; then
      echo "Hugging Face token not provided; skipping login."
      RUN_HF_LOGIN=false
    else
      read -rsp "Enter Hugging Face token (leave blank to skip): " HF_TOKEN
      echo ""
    fi
  fi
  if [[ -n "$HF_TOKEN" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    run_step "Logging into Hugging Face" uv run huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"
  else
    echo "Skipping Hugging Face login."
  fi
fi

if "$RUN_DATASET" || ( ! "$AUTO_CONFIRM" && confirm "Prepare sample dataset shard now? [y/N]" "n"); then
  RUN_DATASET=true
fi

if "$RUN_DATASET"; then
  if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Dataset config '$DATASET_CONFIG' not found." >&2
    exit 1
  fi
  run_step "Preparing dataset shard ($DATASET_CONFIG)" uv run python -m tools.prepare_dataset --config "$DATASET_CONFIG"
fi

if "$ENABLE_WANDB"; then
  export MEGACONTEXT_ENABLE_WANDB=1
  echo "Enabled Weights & Biases logging for subsequent demo."
fi

if "$RUN_DEMO" || ( ! "$AUTO_CONFIRM" && confirm "Run base decode demo? [y/N]" "n"); then
  RUN_DEMO=true
fi

if "$RUN_DEMO"; then
  run_step "Running base decode demo" uv run python -m tools.decode_demo --config "$DATASET_CONFIG"
fi

cat <<'EOF'

Setup complete.
Next steps:
  - For GistNet training, launch Jupyter and open notebooks/megacontext.ipynb.
  - To prepare alternative datasets, rerun this script with --prep-dataset and --dataset-config <path>.
EOF
