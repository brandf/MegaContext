import math


def _effective_variant_length_ratio(
    max_seq_len: int,
    mc_train_wc_length: int | None,
) -> float:
    """
    Estimate the average relative length of a random variant vs. the baseline WC.

    Variants anneal from ~80% of max_seq_len down to mc_train_wc_length, so we
    approximate the average as the midpoint between those endpoints.
    """
    start_len = int(round(max_seq_len * 0.8))
    target_len = mc_train_wc_length if mc_train_wc_length is not None else int(round(max_seq_len * 0.2))
    target_len = max(1, min(max_seq_len, target_len))
    avg_len = 0.5 * (start_len + target_len)
    # Clamp to avoid pathological ratios; variants should always be a fraction of baseline cost.
    return max(0.1, min(1.0, avg_len / max_seq_len))


def compute_variant_multiplier(
    max_seq_len: int,
    mc_num_random_variants: int,
    mc_train_wc_length: int | None,
    mc_random_variant_iterations: int,
) -> float:
    """
    Estimate the relative memory multiplier introduced by MC variants.

    Baseline WC costs 1. Each random variant costs roughly the average
    length ratio of a variant vs. the baseline. We scale by the larger of:
      - length-weighted load across all variants and their iterations
      - a simple count floor of (1 + num_variants)
    """
    if mc_num_random_variants <= 0:
        return 1.0
    variant_cost = mc_num_random_variants * _effective_variant_length_ratio(max_seq_len, mc_train_wc_length)
    # Account for multiple passes over each variant during training.
    length_weighted = 1.0 + variant_cost * max(1, mc_random_variant_iterations + 1)
    count_weighted = 1.0 + mc_num_random_variants
    return max(length_weighted, count_weighted)


def estimate_mc_device_batch(
    total_mem_bytes: int,
    baseline_device_batch: int,
    max_seq_len: int,
    mc_num_random_variants: int,
    mc_random_variant_iterations: int,
    mc_train_wc_length: int | None,
    vram_target_fraction: float = 0.9,
    vram_reserve_bytes: int = 3_000_000_000,
) -> tuple[int, float, float]:
    """
    Estimate the maximum per-device micro-batch size that fits within a VRAM budget.

    Args:
        total_mem_bytes: reported device memory (bytes).
        baseline_device_batch: the baseline non-MC micro-batch size.
        max_seq_len: sequence length.
        mc_num_random_variants: number of random variants per sample.
        mc_random_variant_iterations: how many times each variant is replayed.
        mc_train_wc_length: target WC length (None => default heuristic).
        vram_target_fraction: fraction of total_mem_bytes to aim for.
        vram_reserve_bytes: bytes to reserve for fragmentation/overheads.

    Returns:
        (device_batch_size, target_bytes, per_sample_bytes)
    """
    target_bytes = max(0.0, total_mem_bytes * vram_target_fraction - vram_reserve_bytes)
    if baseline_device_batch <= 0 or target_bytes <= 0:
        return 1, target_bytes, target_bytes

    base_per_sample = target_bytes / baseline_device_batch
    base_length_ratio = _effective_variant_length_ratio(max_seq_len, mc_train_wc_length)
    # Calibrate length impact so small-N configs aren't overly penalized but larger N still shrink sufficiently.
    length_ratio = base_length_ratio * (0.5 + 0.5 * min(1.0, mc_num_random_variants / 4.0))
    # Total per-sample load given variants and variant iterations.
    per_sample_mc = base_per_sample * (1.0 + mc_num_random_variants * length_ratio) * max(
        1.0, mc_random_variant_iterations + 1
    )
    if per_sample_mc <= 0:
        return 1, target_bytes, base_per_sample

    device_batch_size = int(max(1, target_bytes // per_sample_mc))
    return device_batch_size, target_bytes, per_sample_mc


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
