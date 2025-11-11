#!/bin/bash

# A single-GPU nanochat training recipe meant to validate ideas quickly
# before committing to the larger speedrun/run1000 pipelines.

set -euo pipefail

# Load optional environment prepared by mc_setup
if [ -f ".mc_env" ]; then
    # shellcheck source=/dev/null
    source .mc_env
fi

usage() {
    cat <<'EOF'
Usage: bash run10.sh [--gpu 5090|h100] [--mc]

Defaults to --gpu h100. Both profiles run on a single GPU but dial batch size
and iteration count to match the available VRAM / throughput.
EOF
}

GPU_PROFILE="h100"
MC_ENABLED=0
BLOCK_SIZE=32
GISTNET_TYPE="transformer"
GISTNET_LAYERS=2
GISTNET_POOLING="mean"
GISTNET_HEAD="mlp"
LENSNET_TYPE="transformer"
LENSNET_LAYERS=2
LENSNET_HEAD="mlp"
# Use the only supported allocator implementation
ALLOCATOR_TYPE="greedy"
POSITIONAL_TYPE="gaussian"
MC_TREE_TYPE="ram"
SKIP_DATA_PREP=${SKIP_DATA_PREP:-0}
MC_AUX_DTYPE="auto"
MC_AUTO_BATCH=1
MC_EVAL_SOFT_MAX_LENGTH=""
MC_INFER_ALLOCATOR_MAX_REPLACEMENTS=""
MC_INFER_ALLOCATOR_ITERATIONS=""
MC_INFER_REFOCUS_INTERVAL=32
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gpu" >&2; exit 1; }
            GPU_PROFILE="$1"
            ;;
        --mc)
            MC_ENABLED=1
            ;;
        --gistnet)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gistnet" >&2; exit 1; }
            GISTNET_TYPE="$1"
            ;;
        --gistnet_layers)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gistnet_layers" >&2; exit 1; }
            GISTNET_LAYERS="$1"
            ;;
        --gistnet_pooling)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gistnet_pooling" >&2; exit 1; }
            GISTNET_POOLING="$1"
            ;;
        --gistnet_head)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gistnet_head" >&2; exit 1; }
            GISTNET_HEAD="$1"
            ;;
        --lensnet)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --lensnet" >&2; exit 1; }
            LENSNET_TYPE="$1"
            ;;
        --lensnet_layers)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --lensnet_layers" >&2; exit 1; }
            LENSNET_LAYERS="$1"
            ;;
        --lensnet_head)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --lensnet_head" >&2; exit 1; }
            LENSNET_HEAD="$1"
            ;;
        --allocator)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --allocator" >&2; exit 1; }
            ALLOCATOR_TYPE="$1"
            ;;
        --mc_tree)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_tree" >&2; exit 1; }
            MC_TREE_TYPE="$1"
            ;;
        --positional)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --positional" >&2; exit 1; }
            POSITIONAL_TYPE="$1"
            ;;
        --mc_aux_dtype)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_aux_dtype" >&2; exit 1; }
            MC_AUX_DTYPE="$1"
            ;;
        --mc_auto_batch)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_auto_batch" >&2; exit 1; }
            MC_AUTO_BATCH="$1"
            ;;
        --mc_eval_soft_max_length)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_eval_soft_max_length" >&2; exit 1; }
            MC_EVAL_SOFT_MAX_LENGTH="$1"
            ;;
        --mc_infer_allocator_max_replacements)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_infer_allocator_max_replacements" >&2; exit 1; }
            MC_INFER_ALLOCATOR_MAX_REPLACEMENTS="$1"
            ;;
        --mc_infer_allocator_iterations)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_infer_allocator_iterations" >&2; exit 1; }
            MC_INFER_ALLOCATOR_ITERATIONS="$1"
            ;;
        --mc_infer_refocus_interval)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_infer_refocus_interval" >&2; exit 1; }
            MC_INFER_REFOCUS_INTERVAL="$1"
            ;;
        --block_size)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --block_size" >&2; exit 1; }
            BLOCK_SIZE="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

DEPTH=12
MAX_SEQ_LEN=2048

case "$GPU_PROFILE" in
    5090)
        DEVICE_BATCH_SIZE=20
        NUM_ITERATIONS=75500
        ;;
    h100)
        DEVICE_BATCH_SIZE=40
        NUM_ITERATIONS=37750
        ;;
    *)
        echo "Unsupported GPU profile: $GPU_PROFILE (expected 5090 or h100)" >&2
        exit 1
        ;;
esac

TOKENIZER_SHARDS=8
TOKENIZER_MAX_CHARS=2000000000
PRETRAIN_SHARDS=160

TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN))

echo "Run10 profile: $GPU_PROFILE"
echo "Depth: $DEPTH  |  Seq len: $MAX_SEQ_LEN"
echo "Device batch: $DEVICE_BATCH_SIZE  |  Total batch: $TOTAL_BATCH_SIZE tokens"
echo "Iterations: $NUM_ITERATIONS"
echo "MegaContext enabled: $MC_ENABLED"

# -----------------------------------------------------------------------------
# Environment + dependencies

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN="dummy"
fi

python -m nanochat.report reset

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# -----------------------------------------------------------------------------
# Dataset + tokenizer

if [ "$SKIP_DATA_PREP" -eq 1 ]; then
    echo "Skipping dataset/tokenizer prep (SKIP_DATA_PREP=1). Ensure shards/checkpoints already exist."
else
    python -m nanochat.dataset -n "$TOKENIZER_SHARDS"
    python -m nanochat.dataset -n "$PRETRAIN_SHARDS" &
    DATASET_DOWNLOAD_PID=$!

    python -m scripts.tok_train --max_chars="$TOKENIZER_MAX_CHARS"
    python -m scripts.tok_eval

    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
fi

# -----------------------------------------------------------------------------
# Training + eval

NPROC_PER_NODE=1

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max_seq_len="$MAX_SEQ_LEN" \
    --block_size="$BLOCK_SIZE" \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --total_batch_size="$TOTAL_BATCH_SIZE" \
    --num_iterations="$NUM_ITERATIONS" \
    --run="$WANDB_RUN" \
    --mc_enabled="$MC_ENABLED" \
    --gistnet_type="$GISTNET_TYPE" \
    --gistnet_layers="$GISTNET_LAYERS" \
    --gistnet_pooling="$GISTNET_POOLING" \
    --gistnet_head="$GISTNET_HEAD" \
    --lensnet_type="$LENSNET_TYPE" \
    --lensnet_layers="$LENSNET_LAYERS" \
    --lensnet_head="$LENSNET_HEAD" \
    --allocator_type="$ALLOCATOR_TYPE" \
    --mc_tree_type="$MC_TREE_TYPE" \
    --positional_type="$POSITIONAL_TYPE" \
    --mc_aux_dtype="$MC_AUX_DTYPE" \
    --mc_auto_batch="$MC_AUTO_BATCH" \
    --mc_eval_soft_max_length="$MC_EVAL_SOFT_MAX_LENGTH" \
    --mc_infer_allocator_max_replacements="$MC_INFER_ALLOCATOR_MAX_REPLACEMENTS" \
    --mc_infer_allocator_iterations="$MC_INFER_ALLOCATOR_ITERATIONS" \
    --mc_infer_refocus_interval="$MC_INFER_REFOCUS_INTERVAL"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss -- \
    --device_batch_size="$DEVICE_BATCH_SIZE"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.mid_train -- \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --max_seq_len="$MAX_SEQ_LEN" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft

python -m nanochat.report generate

echo "Run finished. Chat with the checkpoint via:"
echo "  python -m scripts.chat_cli -p \"Why is the sky blue?\""
echo "  python -m scripts.chat_web"
