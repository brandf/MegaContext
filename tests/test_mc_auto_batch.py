from mc.auto_batch import compute_variant_multiplier, choose_micro_batch_divisor


def test_compute_variant_multiplier_inflates_with_variants():
    # Baseline multiplier (1 variant => multiplier 2) with modest WC target.
    multiplier = compute_variant_multiplier(max_seq_len=2048, mc_num_random_variants=1, mc_train_wc_length=None)
    assert multiplier >= 2
    # Larger random variant count should strictly increase the multiplier.
    multiplier_more = compute_variant_multiplier(max_seq_len=2048, mc_num_random_variants=3, mc_train_wc_length=512)
    assert multiplier_more > multiplier


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
