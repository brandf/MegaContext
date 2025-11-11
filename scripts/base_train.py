"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
from contextlib import nullcontext

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

try:
    from mc.config import MCConfig
    from mc.runtime import MCController, WorkingContextVariant
    from mc.telemetry import OpenTelemetryProvider, NoOpTelemetryProvider
except ImportError:  # pragma: no cover - optional dependency during early development
    MCController = None
    MCConfig = None
    OpenTelemetryProvider = None
    NoOpTelemetryProvider = None
print_banner()

if NoOpTelemetryProvider is None:
    class _FallbackTelemetry:
        def log_event(self, event):
            return
    NoOpTelemetryProvider = _FallbackTelemetry  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
block_size = 32 # MegaContext block size (tokens per gist)
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb (overridden to 25 when mc_enabled=1)
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# MegaContext (phase 1 integration)
mc_enabled = 0 # set to 1 to enable MegaContext instrumentation
gistnet_type = "transformer"  # transformer | mean
gistnet_layers = 2
gistnet_pooling = "mean"  # mean | query | cls
gistnet_head = "mlp"  # mlp | linear
lensnet_type = "transformer"
lensnet_layers = 2
lensnet_head = "mlp"
allocator_type = "greedy"
positional_type = "gaussian"
allocator_soft_max = max_seq_len
allocator_recent_tokens = 128
allocator_expand_threshold = 0.1
allocator_collapse_threshold = 0.1
allocator_max_replacements = 4
allocator_iterations = 2
allocator_sample_top_k = 4
allocator_sample_temperature = 1.0
mc_aux_dtype = "auto"
mc_tree_type = "ram"
mc_initial_wcs = 4
mc_max_counterfactuals = 8
mc_lens_loss_weight = 0.1
mc_auto_batch = 1
mc_eval_soft_max_length = None
mc_infer_allocator_max_replacements = None
mc_infer_allocator_iterations = None
mc_infer_refocus_interval = 32
mc_infer_rebuild_max_replacements = None
mc_infer_rebuild_iterations = None
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
def _parse_optional_int(value):
    if value in (None, "", "None"):
        return None
    return int(value)


mc_eval_soft_max_length = _parse_optional_int(mc_eval_soft_max_length)
mc_infer_allocator_max_replacements = _parse_optional_int(mc_infer_allocator_max_replacements)
mc_infer_allocator_iterations = _parse_optional_int(mc_infer_allocator_iterations)
mc_infer_rebuild_max_replacements = _parse_optional_int(mc_infer_rebuild_max_replacements)
mc_infer_rebuild_iterations = _parse_optional_int(mc_infer_rebuild_iterations)
mc_infer_refocus_interval = int(mc_infer_refocus_interval)
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Auto-adjust batch size for MC variant amplification
if mc_enabled and mc_auto_batch:
    variant_multiplier = max(1, mc_max_counterfactuals)
    if variant_multiplier > 1:
        original_device_batch_size = device_batch_size
        original_total_batch_size = total_batch_size
        original_num_iterations = num_iterations
        device_batch_size = max(1, device_batch_size // variant_multiplier)
        total_batch_size = device_batch_size * max_seq_len
        target_tokens = original_total_batch_size * max(1, original_num_iterations)
        if total_batch_size > 0:
            num_iterations = max(1, math.ceil(target_tokens / total_batch_size))
        print0(
            "[MegaContext] auto batch adjust: "
            f"device_batch_size {original_device_batch_size} -> {device_batch_size}, "
            f"total_batch_size {original_total_batch_size} -> {total_batch_size}, "
            f"num_iterations {original_num_iterations} -> {num_iterations}"
        )

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
def _training_autocast():
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)
mc_controller = None

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
if mc_enabled:
    if MCController is None or MCConfig is None:
        raise ImportError("mc package is required when --mc_enabled=1")
    if OpenTelemetryProvider is None:
        raise ImportError("opentelemetry dependencies are required when --mc_enabled=1")
    mc_config = MCConfig(
        embed_dim=model_config.n_embd,
        max_seq_len=max_seq_len,
        block_size=block_size,
        device=str(device),
        gistnet_type=gistnet_type,
        gistnet_layers=gistnet_layers,
        gistnet_pooling=gistnet_pooling,
        gistnet_head=gistnet_head,
        lensnet_type=lensnet_type,
        lensnet_layers=lensnet_layers,
        lensnet_head=lensnet_head,
        allocator_type=allocator_type,
        mc_tree_type=mc_tree_type,
        initial_working_contexts=mc_initial_wcs,
        max_counterfactuals=mc_max_counterfactuals,
        lens_loss_weight=mc_lens_loss_weight,
        soft_max_length=allocator_soft_max,
        eval_soft_max_length=mc_eval_soft_max_length,
        infer_allocator_max_replacements=mc_infer_allocator_max_replacements,
        infer_allocator_iterations=mc_infer_allocator_iterations,
        infer_refocus_interval=mc_infer_refocus_interval,
        infer_rebuild_max_replacements=mc_infer_rebuild_max_replacements,
        infer_rebuild_iterations=mc_infer_rebuild_iterations,
        allocator_recent_tokens=allocator_recent_tokens,
        allocator_expand_threshold=allocator_expand_threshold,
        allocator_collapse_threshold=allocator_collapse_threshold,
        allocator_max_replacements=allocator_max_replacements,
        allocator_iterations=allocator_iterations,
        allocator_sample_top_k=allocator_sample_top_k,
        allocator_sample_temperature=allocator_sample_temperature,
        num_heads=num_heads,
        positional_type=positional_type,
        auxiliary_dtype=mc_aux_dtype,
        )
    otel_endpoint = os.getenv("MC_OTEL_ENDPOINT")
    otel_insecure = os.getenv("MC_OTEL_INSECURE", "0") == "1"
    disable_otel = os.getenv("MC_DISABLE_TELEMETRY", "0").lower() in {"1", "true", "yes"}
    if disable_otel or not otel_endpoint or OpenTelemetryProvider is None:
        telemetry_provider = NoOpTelemetryProvider()
    else:
        telemetry_provider = OpenTelemetryProvider(
            service_name="megacontext-train",
            endpoint=otel_endpoint,
            insecure=otel_insecure,
            resource_attributes={"run.id": run},
        )
    mc_controller = MCController(model, mc_config, telemetry_provider=telemetry_provider)
    eval_every = 25
orig_model = model # original, uncompiled model, for saving raw model state_dict
model = torch.compile(model, dynamic=False) # TODO: dynamic True/False think through
eval_model = orig_model if mc_enabled else model
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# Initialize the DataLoaders for train/val
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


def _mean_or_none(values):
    return float(sum(values) / len(values)) if values else None


def _assemble_positional_override(x, mc_result, step, device, mc_controller):
    if mc_result is None:
        return None, None
    batch = x.size(0)
    if mc_result.positional_cache is not None and batch == 1:
        cos, sin = mc_result.positional_cache[:2]
        alibi = mc_result.positional_cache[2] if len(mc_result.positional_cache) > 2 else None
        return (cos.to(device), sin.to(device)), (alibi.to(device) if alibi is not None else None)
    cache_map = getattr(mc_result, "positional_caches", None)
    if not cache_map:
        return None, None
    cos_list, sin_list, alibi_list = [], [], []
    for b in range(batch):
        key = f"train_step_{step}_sample_{b}"
        if key not in cache_map:
            return None, None
        cos_b, sin_b, alibi_b = cache_map[key]
        cos_list.append(cos_b.to(device))
        sin_list.append(sin_b.to(device))
        alibi_list.append(alibi_b.to(device) if alibi_b is not None else None)
    cos = torch.cat(cos_list, dim=0)
    sin = torch.cat(sin_list, dim=0)
    alibi = None
    if any(a is not None for a in alibi_list):
        H = mc_controller.config.num_heads
        T = cos_list[0].size(1)
        dtype = cos_list[0].dtype
        dev = cos_list[0].device
        filled = []
        for a in alibi_list:
            if a is None:
                filled.append(torch.zeros((1, H, T, T), dtype=dtype, device=dev))
            else:
                filled.append(a.to(device))
        alibi = torch.cat(filled, dim=0)
    return (cos, sin), alibi


def _assemble_mc_variant_batch(mc_result, device):
    variants: list[WorkingContextVariant] = getattr(mc_result, "variants", [])
    if not variants:
        return None
    entries = []
    for variant in variants:
        embeddings = variant.working_context.to_tensor().to(device)
        entries.append(
            {
                "variant": variant,
                "embeddings": embeddings,
            }
        )
    return entries


def _run_variant_batch_forward(
    model,
    original_inputs,
    original_labels,
    variant_entries,
    cos_sin_override,
    alibi_override,
):
    total_loss = 0.0
    lod0_loss = None
    for entry in variant_entries:
        variant = entry["variant"]
        embeddings = entry["embeddings"]
        batch_idx = variant.batch_index
        max_seq = original_inputs.size(1)
        if embeddings.shape[1] > max_seq:
            embeddings = embeddings[:, -max_seq:, :]
        seq_len = embeddings.shape[1]
        inputs_slice = original_inputs[batch_idx : batch_idx + 1, :seq_len]
        labels_slice = original_labels[batch_idx : batch_idx + 1, :seq_len]
        if inputs_slice.shape[1] < seq_len:
            pad = torch.zeros(
                (1, seq_len - inputs_slice.shape[1]),
                dtype=inputs_slice.dtype,
                device=inputs_slice.device,
            )
            inputs_slice = torch.cat([pad, inputs_slice], dim=1)
            labels_pad = torch.full(
                (1, seq_len - labels_slice.shape[1]),
                -1,
                dtype=labels_slice.dtype,
                device=labels_slice.device,
            )
            labels_slice = torch.cat([labels_pad, labels_slice], dim=1)
        loss = model(
            inputs_slice,
            labels_slice,
            cos_sin_override=cos_sin_override,
            alibi_override=alibi_override,
            inputs_embeds=embeddings,
        )
        variant.token_loss_value = loss.detach()
        total_loss = total_loss + loss
        if lod0_loss is None and variant.lod_hint == 0:
            lod0_loss = float(loss.detach())
    total_loss = total_loss / max(1, len(variant_entries))
    if lod0_loss is None:
        lod0_loss = float(total_loss.detach())
    return total_loss, lod0_loss


@torch.no_grad()
def evaluate_bpb_with_mc(model, controller, batches, steps, token_bytes, device):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    report_printed = False
    for eval_idx in range(steps):
        x, y = next(batch_iter)
        x = x.to(device)
        y = y.to(device)
        session_id = controller.begin_inference_session(x, rebuild=True)
        if not report_printed:
            report = controller.get_inference_report() if hasattr(controller, "get_inference_report") else None
            if report:
                lod_counts = report.get("lod_counts", {})
                lod_parts = [f"LOD{lod}:{count}" for lod, count in sorted(lod_counts.items())]
                lod_line = ", ".join(lod_parts) if lod_parts else "none"
                pref_iters = report.get("prefill_iterations", 0)
                pref_swaps = report.get("prefill_replacements", 0)
                ref_updates = report.get("refocus_updates", 0)
                ref_iters = report.get("refocus_iterations", 0)
                ref_swaps = report.get("refocus_replacements", 0)
                print0(
                    "✨ MegaContext Val Report\n"
                    f"   📜 Original seq: {report.get('original_length', 'n/a')} tokens\n"
                    f"   🗂️ Working context: {report.get('wc_length', 'n/a')} tokens\n"
                    f"   🧩 LOD mix: {lod_line}\n"
                    f"   ⚙️ Prefocus: {pref_iters} iterations / {pref_swaps} replacements\n"
                    f"   🔁 Refocus: {ref_updates} updates / {ref_iters} iterations / {ref_swaps} replacements"
                )
                report_printed = True
        wc = controller.get_inference_working_context()
        if wc is None:
            raise RuntimeError("Inference working context is None during validation")
        wc_tensor = wc.to_tensor()
        wc_len = wc_tensor.shape[1]
        cos_sin_override = controller._build_wc_positional(wc)
        alibi_override = cos_sin_override[2]
        cos_sin_override = cos_sin_override[:2]
        inputs_embeds_override = wc_tensor
        tokens_slice = y[:, -wc_len:].contiguous()
        input_tokens = x[:, -wc_len:].contiguous()
        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float32
        autocast_ctx = nullcontext()
        if autocast_enabled:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
        with _training_autocast():
            embeds = inputs_embeds_override
            target_dtype = autocast_dtype if autocast_enabled else next(model.parameters()).dtype
            if embeds.dtype != target_dtype:
                embeds = embeds.to(target_dtype)
            loss2d = model(
                input_tokens,
                tokens_slice,
                loss_reduction="none",
                cos_sin_override=cos_sin_override,
                alibi_override=alibi_override,
                inputs_embeds=embeds,
            )
        loss2d = loss2d.view(-1)
        targets_flat = tokens_slice.view(-1)
        if (targets_flat.int() < 0).any():
            valid = targets_flat >= 0
            targets_safe = torch.where(valid, targets_flat, torch.zeros_like(targets_flat))
            num_bytes2d = torch.where(
                valid,
                token_bytes[targets_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[targets_flat]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
# note that we run +1 steps only so that we can eval and save at the end
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step
    mc_lens_loss_val = None
    mc_lens_loss_samples = []
    mc_time_samples: list[float] = []
    forward_time_samples: list[float] = []
    optimizer_time_samples: list[float] = []
    loader_time_samples: list[float] = []
    mc_variant_counts: list[int] = []
    vanilla_loss_samples: list[float] = []

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        if mc_controller is not None:
            val_bpb = evaluate_bpb_with_mc(eval_model, mc_controller, val_loader, eval_steps, token_bytes, device)
        else:
            with _training_autocast():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with _training_autocast():
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with _training_autocast():
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        mc_result = None
        positional_cache = None
        if mc_controller is not None:
            try:
                t_mc0 = time.time()
                with _training_autocast():
                    mc_result = mc_controller.process_batch(x.detach(), step, context="train")
                t_mc1 = time.time()
                # Collect MC controller timing (ms) for logging later
                mc_time_samples.append((t_mc1 - t_mc0) * 1000.0)
                profile = getattr(mc_controller, "last_batch_profile", None)
                if profile:
                    mc_variant_counts.extend(profile.get("variant_counts", []))
            except Exception as exc: # pragma: no cover - guard against optional path regressions
                print0(f"[MegaContext] controller error at step {step}: {exc}")
                import traceback

                traceback.print_exc()
            else:
                if mc_result is not None:
                    positional_cache = mc_result.positional_cache
                    if mc_result.lens_loss is not None:
                        mc_lens_loss_samples.append(float(mc_result.lens_loss.detach()))
        with _training_autocast():
            cos_sin_override = None
            alibi_override = None
            if positional_cache is not None:
                # Single-sample fast path: controller provided a batch-level cache
                cos_sin_override = positional_cache[:2]
                if len(positional_cache) > 2:
                    alibi_override = positional_cache[2]
            elif mc_result is not None and hasattr(mc_result, "positional_caches") and mc_result.positional_caches:
                # Batched path: assemble per-sample caches returned by the controller
                try:
                    B = x.size(0)
                    cos_list, sin_list, alibi_list = [], [], []
                    for b in range(B):
                        key = f"train_step_{step}_sample_{b}"
                        if key not in mc_result.positional_caches:
                            raise KeyError(key)
                        cos_b, sin_b, alibi_b = mc_result.positional_caches[key]
                        cos_list.append(cos_b.to(device))
                        sin_list.append(sin_b.to(device))
                        alibi_list.append(alibi_b.to(device) if alibi_b is not None else None)
                    cos = torch.cat(cos_list, dim=0)
                    sin = torch.cat(sin_list, dim=0)
                    cos_sin_override = (cos, sin)
                    if any(a is not None for a in alibi_list):
                        # Fill missing per-sample ALiBi with zeros of the correct shape
                        T = cos_list[0].size(1)
                        H = mc_controller.config.num_heads if mc_controller is not None else 1
                        filled = []
                        for a in alibi_list:
                            if a is None:
                                filled.append(torch.zeros((1, H, T, T), dtype=torch.bfloat16, device=device))
                            else:
                                filled.append(a.to(device))
                        alibi_override = torch.cat(filled, dim=0)
                except Exception as _:
                    # Fallback to default RoPE if assembly fails; training still proceeds
                    cos_sin_override = None
                    alibi_override = None
            inputs_embeds_override = None
            if mc_result is not None and mc_result.cached_embeddings is not None:
                inputs_embeds_override = mc_result.cached_embeddings.to(device)
            t_fw0 = time.time()
            if mc_result is not None and mc_result.variant_loss is not None:
                base_loss = mc_result.variant_loss
                lod0_loss = mc_result.lod0_loss if mc_result.lod0_loss is not None else float(base_loss.detach())
                vanilla_loss_samples.append(lod0_loss)
            else:
                base_loss = model(
                    x,
                    y,
                    cos_sin_override=cos_sin_override,
                    alibi_override=alibi_override,
                    inputs_embeds=inputs_embeds_override,
                )
                vanilla_loss_samples.append(float(base_loss.detach()))
            mc_extra_loss = base_loss.new_zeros(())
            if mc_controller is not None and mc_result is not None:
                if mc_result.lens_loss is not None:
                    mc_extra_loss = mc_extra_loss + mc_result.lens_loss * mc_controller.config.lens_loss_weight
            loss = base_loss + mc_extra_loss
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        forward_time_samples.append((time.time() - t_fw0) * 1000.0)
        t_loader0 = time.time()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        loader_time_samples.append((time.time() - t_loader0) * 1000.0)
    t_opt0 = time.time()
    mc_lens_loss_val = _mean_or_none(mc_lens_loss_samples)
    vanilla_loss_val = _mean_or_none(vanilla_loss_samples)
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    optimizer_time_samples.append((time.time() - t_opt0) * 1000.0)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    log_interval = 5 if mc_controller is not None else 100
    if step % log_interval == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        if mc_time_samples:
            log_data["mc/time_controller_ms"] = sum(mc_time_samples) / len(mc_time_samples)
            mc_time_samples.clear()
        if forward_time_samples:
            log_data["time/forward_ms"] = sum(forward_time_samples) / len(forward_time_samples)
            forward_time_samples.clear()
        if optimizer_time_samples:
            log_data["time/optimizer_ms"] = sum(optimizer_time_samples) / len(optimizer_time_samples)
            optimizer_time_samples.clear()
        if loader_time_samples:
            log_data["time/dataloader_ms"] = sum(loader_time_samples) / len(loader_time_samples)
            loader_time_samples.clear()
        if mc_variant_counts:
            log_data["mc/variants_mean"] = sum(mc_variant_counts) / len(mc_variant_counts)
            log_data["mc/variants_total"] = sum(mc_variant_counts)
            mc_variant_counts.clear()
        if mc_result is not None and mc_result.delta_mean is not None:
            log_data["mc/delta_mean"] = mc_result.delta_mean
        if mc_result is not None and mc_result.delta_p95 is not None:
            log_data["mc/delta_p95"] = mc_result.delta_p95
        if mc_result is not None and mc_result.lod_metrics:
            for lod, val in mc_result.lod_metrics.items():
                log_data[f"mc/lod_loss/{lod}"] = val
        if mc_result is not None and mc_result.lod_counts:
            for lod, count in mc_result.lod_counts.items():
                log_data[f"mc/lod_count/{lod}"] = count
        if vanilla_loss_val is not None:
            log_data["train/loss_lod0"] = vanilla_loss_val
        if mc_lens_loss_val is not None:
            log_data["mc/lens_loss"] = mc_lens_loss_val
        wandb_run.log(log_data)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
