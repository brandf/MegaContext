from mc.auto_batch import compute_variant_multiplier, choose_micro_batch_divisor, estimate_mc_device_batch


def test_compute_variant_multiplier_inflates_with_variants():
    multiplier = compute_variant_multiplier(max_seq_len=2048, mc_num_random_variants=1, mc_train_wc_length=None, mc_random_variant_iterations=2)
    assert multiplier >= 2.0  # at least baseline + 1 variant
    multiplier_more = compute_variant_multiplier(max_seq_len=2048, mc_num_random_variants=3, mc_train_wc_length=512, mc_random_variant_iterations=2)
    assert multiplier_more >= 4.0  # at least baseline + 3 variants
    assert multiplier_more > multiplier


def test_estimate_mc_device_batch_scales_with_variants_and_iters():
    # Baseline: 80GB card, baseline batch 64.
    device_batch, target_bytes, per_sample = estimate_mc_device_batch(
        total_mem_bytes=80 * 1024**3,
        baseline_device_batch=64,
        max_seq_len=2048,
        mc_num_random_variants=0,
        mc_random_variant_iterations=0,
        mc_train_wc_length=None,
    )
    assert device_batch == 64  # should match baseline when no variants

    # With variants and iterations we expect a smaller batch.
    device_batch_mc, _, _ = estimate_mc_device_batch(
        total_mem_bytes=80 * 1024**3,
        baseline_device_batch=64,
        max_seq_len=2048,
        mc_num_random_variants=4,
        mc_random_variant_iterations=2,
        mc_train_wc_length=None,
    )
    assert device_batch_mc < device_batch


def test_choose_micro_batch_keeps_tokens_constant():
    # With a perfect divisor the result should match the target shrink factor exactly.
    micro, accum = choose_micro_batch_divisor(per_rank_budget=64, desired_multiplier=8)
    assert micro == 8
    assert accum == 8
    assert micro * accum == 64

    # When the target multiplier doesn't divide the budget we still keep the product constant.
    micro, accum = choose_micro_batch_divisor(per_rank_budget=88, desired_multiplier=3)
    assert micro * accum == 88
    # We should shrink more aggressively than requested to maintain divisibility.
    assert accum >= 3


def test_choose_micro_batch_handles_extreme_multiplier():
    micro, accum = choose_micro_batch_divisor(per_rank_budget=10, desired_multiplier=100)
    assert micro == 1
    assert accum == 10
