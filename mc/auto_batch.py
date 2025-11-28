import math


def compute_variant_multiplier(
    max_seq_len: int,
    mc_num_random_variants: int,
    mc_train_wc_length: int | None,
) -> int:
    """
    Estimate how many micro-batches we need to accommodate MC variants.

    We start with the baseline variant count (baseline + random variants)
    and then inflate by the effective token ratio introduced by shortened
    working contexts so we don't under-estimate VRAM usage.
    """
    base_variant_multiplier = max(1, mc_num_random_variants + 1)
    train_wc_target = mc_train_wc_length if mc_train_wc_length is not None else int(round(max_seq_len * 0.75))
    train_wc_target = max(1, min(max_seq_len, int(train_wc_target)))
    tokens_ratio = 1.0 + (mc_num_random_variants * train_wc_target) / max_seq_len
    combined_multiplier = math.ceil(tokens_ratio * base_variant_multiplier)
    return max(base_variant_multiplier, combined_multiplier)


def choose_micro_batch_divisor(per_rank_budget: int, desired_multiplier: int) -> tuple[int, int]:
    """
    Pick the largest per-rank micro-batch size (device batch) that:
      1. Divides the per-rank token budget exactly so grad accumulation stays integral.
      2. Does not exceed the requested shrink factor (desired_multiplier).

    Returns:
        device_batch_size, grad_accum_steps
    """
    if per_rank_budget <= 0:
        raise ValueError("per_rank_budget must be > 0")
    desired_multiplier = max(1, desired_multiplier)
    # Convert the desired multiplier into a maximum per-rank micro-batch size.
    # Clamp to at least 1 so we never underflow.
    max_device_batch = max(1, per_rank_budget // desired_multiplier)
    device_batch_size = max_device_batch
    # Search downward until we find a divisor of per_rank_budget.
    while per_rank_budget % device_batch_size != 0 and device_batch_size > 1:
        device_batch_size -= 1
    if per_rank_budget % device_batch_size != 0:
        # Fall back to device_batch_size=1 which always divides per_rank_budget.
        device_batch_size = 1
    grad_accum_steps = per_rank_budget // device_batch_size
    return device_batch_size, grad_accum_steps
