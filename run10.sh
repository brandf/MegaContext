#!/bin/bash

# A single-GPU nanochat training recipe meant to validate ideas quickly
# before committing to the larger speedrun/run1000 pipelines.

set -euo pipefail

# Capture pre-existing overrides so .mc_env doesn't clobber them
USER_WANDB_RUN=${WANDB_RUN:-""}
USER_MODEL_TAG=${MODEL_TAG:-""}

# Load optional environment prepared by mc_setup
if [ -f ".mc_env" ]; then
    # shellcheck source=/dev/null
    source .mc_env
fi

# Restore user-provided overrides after sourcing .mc_env
if [ -n "$USER_WANDB_RUN" ]; then
    WANDB_RUN="$USER_WANDB_RUN"
fi
if [ -n "$USER_MODEL_TAG" ]; then
    MODEL_TAG="$USER_MODEL_TAG"
fi

WANDB_RUN=${WANDB_RUN:-"dummy"}
MODEL_TAG=${MODEL_TAG:-""}

usage() {
    cat <<'EOF'
Usage: bash run10.sh [--gpu 5090|h100] [--mc] [--profile baseline|mc]

Defaults to --gpu h100. Both profiles run on a single GPU but dial batch size
and iteration count to match the available VRAM / throughput.
EOF
}

GPU_PROFILE="h100"
PROFILE=""
MC_ENABLED=0
BLOCK_SIZE=32
DEVICE_BATCH_SIZE=""
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
MC_LOG_TIMERS=0
MC_EVAL_SOFT_MAX_LENGTH=""
MC_INFER_ALLOCATOR_MAX_REPLACEMENTS=""
MC_INFER_ALLOCATOR_ITERATIONS=""
MC_INFER_REFOCUS_INTERVAL=32
ALLOCATOR_SOFT_MAX=""
MC_TRAIN_REPORT=0
MC_VAL_REPORT=1
MC_LOG_LOD_ASCII_TRAIN=0
MC_LOG_LOD_ASCII_VAL=0
MC_LOG_LENS_DEBUG=0
MC_LENS_RANK_WEIGHT=0.5
MC_LENS_BUDGET_WEIGHT=0.1
MC_LENS_MARGIN=0.1
MC_DISABLE_VAL=0
MC_LENS_COLLAPSE_WEIGHT=1.0
MC_LENS_TEMPERATURE=1.0
MC_TRAIN_WC_LENGTH=""
MC_NUM_RANDOM_VARIANTS=2
MC_RANDOM_VARIANT_ITERATIONS=2
MC_GIST_DELTA_WEIGHT=0.1
MC_MAX_LENS_PAIRS=8
MC_LENS_KL_WEIGHT=0.0
MC_LENS_ADV_NORM_BETA=0.9
MC_LENS_BUDGET_SMOOTH_WEIGHT=0.0
MC_LENS_BUDGET_SMOOTH_BETA=0.9
MC_LENS_HARD_NEGATIVE_RATIO=1.0
MC_COMPILE_GISTNET=1
MC_COMPILE_LENSNET=1
GRAD_CLIP=1.0
WARMUP_RATIO=0.02
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --gpu" >&2; exit 1; }
            GPU_PROFILE="$1"
            ;;
        --profile)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --profile" >&2; exit 1; }
            PROFILE="$1"
            case "$PROFILE" in
                baseline)
                    MC_ENABLED=0
                    ;;
                mc)
                    MC_ENABLED=1
                    ;;
                *)
                    echo "Unsupported profile: $PROFILE (expected baseline or mc)" >&2
                    exit 1
                    ;;
            esac
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
        --device_batch_size)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --device_batch_size" >&2; exit 1; }
            DEVICE_BATCH_SIZE="$1"
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
        --allocator_soft_max)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --allocator_soft_max" >&2; exit 1; }
            ALLOCATOR_SOFT_MAX="$1"
            ;;
        --mc_train_report)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_train_report" >&2; exit 1; }
            MC_TRAIN_REPORT="$1"
            ;;
        --mc_val_report)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_val_report" >&2; exit 1; }
            MC_VAL_REPORT="$1"
            ;;
        --mc_log_timers)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_log_timers" >&2; exit 1; }
            MC_LOG_TIMERS="$1"
            ;;
        --mc_log_lod_ascii_train)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_log_lod_ascii_train" >&2; exit 1; }
            MC_LOG_LOD_ASCII_TRAIN="$1"
            ;;
        --mc_log_lod_ascii_val)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_log_lod_ascii_val" >&2; exit 1; }
            MC_LOG_LOD_ASCII_VAL="$1"
            ;;
        --mc_log_lens_debug)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_log_lens_debug" >&2; exit 1; }
            MC_LOG_LENS_DEBUG="$1"
            ;;
        --mc_lens_rank_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_rank_weight" >&2; exit 1; }
            MC_LENS_RANK_WEIGHT="$1"
            ;;
        --mc_lens_budget_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_budget_weight" >&2; exit 1; }
            MC_LENS_BUDGET_WEIGHT="$1"
            ;;
        --mc_lens_margin)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_margin" >&2; exit 1; }
            MC_LENS_MARGIN="$1"
            ;;
        --mc_disable_val)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_disable_val" >&2; exit 1; }
            MC_DISABLE_VAL="$1"
            ;;
        --mc_lens_collapse_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_collapse_weight" >&2; exit 1; }
            MC_LENS_COLLAPSE_WEIGHT="$1"
            ;;
        --mc_lens_temperature)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_temperature" >&2; exit 1; }
            MC_LENS_TEMPERATURE="$1"
            ;;
        --mc_train_wc_length)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_train_wc_length" >&2; exit 1; }
            MC_TRAIN_WC_LENGTH="$1"
            ;;
        --mc_num_random_variants)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_num_random_variants" >&2; exit 1; }
            MC_NUM_RANDOM_VARIANTS="$1"
            ;;
        --mc_random_variant_iterations)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_random_variant_iterations" >&2; exit 1; }
            MC_RANDOM_VARIANT_ITERATIONS="$1"
            ;;
        --mc_gist_delta_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_gist_delta_weight" >&2; exit 1; }
            MC_GIST_DELTA_WEIGHT="$1"
            ;;
        --mc_max_lens_pairs)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_max_lens_pairs" >&2; exit 1; }
            MC_MAX_LENS_PAIRS="$1"
            ;;
        --mc_lens_kl_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_kl_weight" >&2; exit 1; }
            MC_LENS_KL_WEIGHT="$1"
            ;;
        --mc_lens_adv_norm_beta)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_adv_norm_beta" >&2; exit 1; }
            MC_LENS_ADV_NORM_BETA="$1"
            ;;
        --mc_lens_budget_smooth_weight)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_budget_smooth_weight" >&2; exit 1; }
            MC_LENS_BUDGET_SMOOTH_WEIGHT="$1"
            ;;
        --mc_lens_budget_smooth_beta)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_budget_smooth_beta" >&2; exit 1; }
            MC_LENS_BUDGET_SMOOTH_BETA="$1"
            ;;
        --mc_lens_hard_negative_ratio)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_lens_hard_negative_ratio" >&2; exit 1; }
            MC_LENS_HARD_NEGATIVE_RATIO="$1"
            ;;
        --mc_compile_gistnet)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_compile_gistnet" >&2; exit 1; }
            MC_COMPILE_GISTNET="$1"
            ;;
        --mc_compile_lensnet)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --mc_compile_lensnet" >&2; exit 1; }
            MC_COMPILE_LENSNET="$1"
            ;;
        --model_tag)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --model_tag" >&2; exit 1; }
            MODEL_TAG="$1"
            ;;
        --grad_clip)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --grad_clip" >&2; exit 1; }
            GRAD_CLIP="$1"
            ;;
        --warmup_ratio)
            shift
            [[ $# -gt 0 ]] || { echo "Missing value for --warmup_ratio" >&2; exit 1; }
            WARMUP_RATIO="$1"
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

if [ -z "$PROFILE" ]; then
    if [ "$MC_ENABLED" -eq 1 ]; then
        PROFILE="mc"
    else
        PROFILE="baseline"
    fi
fi

if [ -z "$DEVICE_BATCH_SIZE" ]; then
    case "$GPU_PROFILE" in
        5090)
            DEVICE_BATCH_SIZE=20
            ;;
        h100)
            DEVICE_BATCH_SIZE=88
            ;;
        *)
            echo "Unsupported GPU profile: $GPU_PROFILE (expected 5090 or h100)" >&2
            exit 1
            ;;
    esac
fi

if [ -z "$ALLOCATOR_SOFT_MAX" ]; then
    ALLOCATOR_SOFT_MAX="$MAX_SEQ_LEN"
fi

TOKENIZER_SHARDS=8
TOKENIZER_MAX_CHARS=2000000000
PRETRAIN_SHARDS=160

TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
TRAIN_TOKENS_TARGET=${TRAIN_TOKENS_TARGET:-4329472000}
NUM_ITERATIONS=$(( (TRAIN_TOKENS_TARGET + TOTAL_BATCH_SIZE - 1) / TOTAL_BATCH_SIZE ))
ACTUAL_TRAIN_TOKENS=$((NUM_ITERATIONS * TOTAL_BATCH_SIZE))

echo "Run10 profile: $GPU_PROFILE"
echo "Mode: $PROFILE"
echo "Depth: $DEPTH  |  Seq len: $MAX_SEQ_LEN"
echo "Device batch: $DEVICE_BATCH_SIZE  |  Total batch: $TOTAL_BATCH_SIZE tokens"
echo "Iterations: $NUM_ITERATIONS (target tokens: $TRAIN_TOKENS_TARGET actual: $ACTUAL_TRAIN_TOKENS)"
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
    --model_tag="$MODEL_TAG" \
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
    --allocator_soft_max="$ALLOCATOR_SOFT_MAX" \
    --mc_auto_batch="$MC_AUTO_BATCH" \
    --mc_eval_soft_max_length="$MC_EVAL_SOFT_MAX_LENGTH" \
    --mc_infer_allocator_max_replacements="$MC_INFER_ALLOCATOR_MAX_REPLACEMENTS" \
    --mc_infer_allocator_iterations="$MC_INFER_ALLOCATOR_ITERATIONS" \
    --mc_infer_refocus_interval="$MC_INFER_REFOCUS_INTERVAL" \
    --mc_train_report="$MC_TRAIN_REPORT" \
    --mc_val_report="$MC_VAL_REPORT" \
    --mc_log_timers="$MC_LOG_TIMERS" \
    --mc_log_lod_ascii_train="$MC_LOG_LOD_ASCII_TRAIN" \
    --mc_log_lod_ascii_val="$MC_LOG_LOD_ASCII_VAL" \
    --mc_log_lens_debug="$MC_LOG_LENS_DEBUG" \
    --mc_lens_rank_weight="$MC_LENS_RANK_WEIGHT" \
    --mc_lens_budget_weight="$MC_LENS_BUDGET_WEIGHT" \
    --mc_lens_margin="$MC_LENS_MARGIN" \
    --mc_disable_val="$MC_DISABLE_VAL" \
    --mc_lens_collapse_weight="$MC_LENS_COLLAPSE_WEIGHT" \
    --mc_lens_temperature="$MC_LENS_TEMPERATURE" \
    --mc_train_wc_length="$MC_TRAIN_WC_LENGTH" \
    --mc_num_random_variants="$MC_NUM_RANDOM_VARIANTS" \
    --mc_random_variant_iterations="$MC_RANDOM_VARIANT_ITERATIONS" \
    --mc_gist_delta_weight="$MC_GIST_DELTA_WEIGHT" \
    --mc_max_lens_pairs="$MC_MAX_LENS_PAIRS" \
    --mc_lens_kl_weight="$MC_LENS_KL_WEIGHT" \
    --mc_lens_adv_norm_beta="$MC_LENS_ADV_NORM_BETA" \
    --mc_lens_budget_smooth_weight="$MC_LENS_BUDGET_SMOOTH_WEIGHT" \
    --mc_lens_budget_smooth_beta="$MC_LENS_BUDGET_SMOOTH_BETA" \
    --mc_lens_hard_negative_ratio="$MC_LENS_HARD_NEGATIVE_RATIO" \
    --mc_compile_gistnet="$MC_COMPILE_GISTNET" \
    --mc_compile_lensnet="$MC_COMPILE_LENSNET" \
    --grad_clip="$GRAD_CLIP" \
    --warmup_ratio="$WARMUP_RATIO"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_loss -- \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --model_tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model_tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.mid_train -- \
    --device_batch_size="$DEVICE_BATCH_SIZE" \
    --max_seq_len="$MAX_SEQ_LEN" \
    --run="$WANDB_RUN" \
    --model_tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i mid --model_tag "$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --run="$WANDB_RUN" \
    --model_tag="$MODEL_TAG"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft --model_tag "$MODEL_TAG"

python -m nanochat.report generate

echo "Run finished. Chat with the checkpoint via:"
echo "  python -m scripts.chat_cli -p \"Why is the sky blue?\""
echo "  python -m scripts.chat_web"
