---
tags:
  - algorithms
  - strategies
summary: Comparison of allocation strategies for the Focus Allocator, from greedy POC to learned differentiable approaches.
---

The [[Focus Allocator]] transforms [[LensNet]] utilities into concrete expand/collapse actions. This document compares the **greedy strategy** used in the current [[POC]] against future **learned policies** and explores trade-offs between approaches.

---

## Overview

The allocator's core challenge: given signed focus scores from [[LensNet]], decide which blocks to expand (increase detail) or collapse (reduce detail) while maintaining:
- [[Working Context]] contiguity
- Budget constraints (`W_max`)
- Block alignment (32-token boundaries)
- System stability (no oscillation)

Three primary strategies exist:
1. **Greedy algorithm** (current POC)
2. **Learned/differentiable allocator** (future)
3. **Optimization-based variants** (future)

---

## Greedy Strategy (POC)

### Core Algorithm

The greedy approach uses alternating priority queues to select actions based on magnitude:

```pseudocode
function greedy_allocate(focus_scores, working_context, W_max, N_diff):
    # Step 1: Collect candidates
    expand_queue = PriorityQueue()  # max-heap
    collapse_queue = PriorityQueue()  # min-heap

    for block in working_context:
        score = aggregate_score(focus_scores, block)

        if score > τ_expand and can_expand(block):
            expand_queue.push(score, block)
        elif score < -τ_collapse and can_collapse(block):
            collapse_queue.push(score, block)

    # Step 2: Apply actions alternately
    actions_applied = 0
    current_budget = compute_budget(working_context)

    while actions_applied < N_diff:
        # Try expand if budget allows
        if not expand_queue.empty():
            score, block = expand_queue.peek()
            cost = expansion_cost(block)

            if current_budget + cost <= W_max:
                if not is_on_cooldown(block, 'expand'):
                    expand_queue.pop()
                    expand(block)
                    current_budget += cost
                    set_cooldown(block, 'expand')
                    actions_applied++

        # Try collapse to refund budget
        if not collapse_queue.empty():
            score, block = collapse_queue.peek()
            if not is_on_cooldown(block, 'collapse'):
                collapse_queue.pop()
                collapse(block)
                current_budget -= collapse_refund(block)
                set_cooldown(block, 'collapse')
                actions_applied++

        # Exit if both queues exhausted
        if expand_queue.empty() and collapse_queue.empty():
            break

    return working_context
```

### Characteristics

**Strengths:**
- **Simple implementation:** Easy to debug and understand
- **Fast execution:** O(n log n) complexity for sorting candidates
- **Predictable behavior:** Actions always take highest-magnitude scores
- **No training required:** Works immediately without data collection
- **Stable defaults:** Thresholds (`τ_expand = 0.2`, `τ_collapse = 0.2`) work across documents

**Weaknesses:**
- **Myopic decisions:** Each action considers only local score, not global optimality
- **Fixed thresholds:** Cannot adapt to document characteristics or user behavior
- **Alternating bias:** Strict alternation may not reflect optimal expand/collapse ratio
- **No lookahead:** Cannot anticipate how current actions affect future iterations
- **Suboptimal budget use:** May leave budget unused if only small expands remain

### Hysteresis & Stability

The greedy strategy incorporates guardrails to prevent oscillation:

```pseudocode
# Cooldown mechanism
cooldown_tracker = {}  # block_id -> (action_type, remaining_steps)

function is_on_cooldown(block, action_type):
    if block.id in cooldown_tracker:
        last_action, steps = cooldown_tracker[block.id]
        # Prevent opposite action for cooldown_steps iterations
        if last_action != action_type and steps > 0:
            return True
    return False

function set_cooldown(block, action_type):
    cooldown_tracker[block.id] = (action_type, cooldown_steps)

function decrement_cooldowns():
    for block_id in cooldown_tracker:
        action, steps = cooldown_tracker[block_id]
        if steps > 0:
            cooldown_tracker[block_id] = (action, steps - 1)
```

**Legality masks** prevent invalid operations:
- L0 blocks (raw tokens) cannot expand further
- Maximum [[LOD]] blocks (root level) cannot collapse
- Mixed-LOD siblings are rejected to maintain [[GistNet]] alignment

### Parameter Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `τ_expand` | 0.20 | Minimum score to trigger expansion |
| `τ_collapse` | 0.20 | Minimum negative score to trigger collapse |
| `N_diff` | 4 | Max actions per iteration (caps churn) |
| `cooldown_steps` | 2 | Iterations before action reversal allowed |
| `lens_update_interval` | 32 tokens | How often [[LensNet]] runs |
| `tail_gist_window` | 5 L1 + 1 L2 | [[LensNet]] conditioning context |

---

## Learned/Differentiable Allocator (Future)

### Motivation

A **learned focus router** could overcome greedy limitations by:
- **Global optimization:** Jointly optimize all expand/collapse decisions
- **Adaptive thresholds:** Learn document-specific or user-specific parameters
- **Predictive planning:** Anticipate how actions affect future [[LensNet]] outputs
- **Efficient budgeting:** Maximize utility per token used

### Proposed Architecture

```pseudocode
# Neural network-based allocator
class LearnedAllocator(Module):
    def __init__(self, hidden_dim=256):
        self.block_encoder = TransformerEncoder(layers=2)
        self.action_head = Linear(hidden_dim, 3)  # {expand, keep, collapse}

    def forward(self, focus_scores, context_embeddings, budget_state):
        # Encode each block with its score and context
        features = self.block_encoder(
            scores=focus_scores,
            embeddings=context_embeddings,
            budget=budget_state
        )

        # Predict action probabilities for each block
        action_logits = self.action_head(features)
        action_probs = softmax(action_logits, dim=-1)

        return action_probs

# Differentiable action selection
function differentiable_allocate(focus_scores, working_context, budget):
    # Get soft action assignments
    action_probs = model(focus_scores, working_context, budget)

    # Sample actions (training) or take argmax (inference)
    if training:
        actions = gumbel_softmax(action_probs, tau=0.5)
    else:
        actions = argmax(action_probs, dim=-1)

    # Apply constraint projection
    valid_actions = project_to_feasible(
        actions,
        budget_constraint=W_max,
        alignment_constraints=block_boundaries
    )

    return valid_actions
```

### Training Approach

**Reward signal:**
```pseudocode
function compute_reward(episode):
    # Measure quality of resulting context
    reward = 0

    # 1. Task performance (downstream LLM quality)
    reward += λ_task * task_success_rate(episode)

    # 2. Budget efficiency
    tokens_used = episode.final_budget
    reward += λ_budget * (1 - tokens_used / W_max)

    # 3. Stability (fewer actions preferred)
    reward -= λ_churn * episode.total_actions

    # 4. Relevance (focus on user cursor region)
    reward += λ_relevance * cursor_coverage(episode)

    return reward
```

**Training loop (reinforcement learning):**
```pseudocode
# Policy gradient / PPO approach
for episode in training_data:
    states = []
    actions = []
    rewards = []

    context = initialize_context(episode.document)

    for timestep in episode:
        # Get action distribution from learned policy
        action_probs = model(context.scores, context.embeddings)
        action = sample(action_probs)

        # Apply action and observe outcome
        new_context = apply_action(context, action)
        reward = compute_reward_step(new_context, timestep)

        states.append(context)
        actions.append(action)
        rewards.append(reward)

        context = new_context

    # Compute policy gradient
    loss = -sum(log_prob(actions) * discounted_rewards(rewards))
    loss += entropy_bonus(action_probs)  # encourage exploration

    optimizer.step(loss)
```

### Characteristics

**Potential strengths:**
- **Optimal allocation:** Can learn to maximize long-term utility
- **Adaptive behavior:** Personalizes to document types and users
- **Joint optimization:** Considers all blocks simultaneously
- **Smooth interpolation:** Soft assignments during training enable gradients

**Challenges:**
- **Training complexity:** Requires large dataset of edit sessions
- **Reward engineering:** Defining good reward signal is non-trivial
- **Inference cost:** Neural network forward pass adds latency
- **Stability concerns:** Policy may drift or collapse during training
- **Interpretability:** Hard to debug when allocator makes unexpected choices

---

## Optimization-Based Variants (Future)

### Balanced Mass Matching

Instead of greedy selection, solve for optimal expand/collapse balance:

```pseudocode
function mass_matching_allocate(focus_scores, W_max):
    # Formulate as assignment problem
    expand_candidates = [(score, cost) for score > τ if can_expand]
    collapse_candidates = [(score, refund) for score < -τ if can_collapse]

    # Objective: maximize total utility subject to budget
    # maximize: sum(score_i * x_i) for all candidates
    # subject to: sum(cost_i * x_i) <= W_max
    #             x_i in {0, 1}

    solution = solve_knapsack(expand_candidates, collapse_candidates, W_max)
    return solution
```

**Advantages:**
- Better budget utilization than greedy
- Considers expand/collapse balance holistically
- Still deterministic and interpretable

**Disadvantages:**
- Higher computational cost (NP-hard in general)
- May require approximation for real-time use
- Does not adapt over time without retuning

### Adaptive Thresholds

Learn `τ_expand` and `τ_collapse` based on recent utilization:

```pseudocode
function adaptive_thresholds(recent_history):
    # If consistently leaving budget unused, lower τ_expand
    if avg_budget_used(recent_history) < 0.8 * W_max:
        τ_expand *= 0.95

    # If consistently over-budget, raise τ_expand
    if avg_budget_used(recent_history) > 0.95 * W_max:
        τ_expand *= 1.05

    # Mirror for collapse threshold
    τ_collapse = τ_expand  # keep symmetric initially

    return τ_expand, τ_collapse
```

---

## Strategy Comparison

| Criterion | Greedy (POC) | Learned Policy | Optimization |
|-----------|--------------|----------------|--------------|
| **Implementation complexity** | Low | High | Medium |
| **Training required** | No | Yes (large dataset) | No |
| **Runtime cost** | Fast (O(n log n)) | Medium (neural forward) | Slow (optimization) |
| **Optimality** | Local maxima | Potential global | Near-optimal |
| **Adaptability** | Fixed thresholds | Learns user patterns | Adapts per-doc |
| **Interpretability** | High | Low | Medium |
| **Stability** | Good (with cooldowns) | Uncertain | Deterministic |
| **Production readiness** | Ready now | Research phase | Research phase |

---

## When to Use Each Strategy

### Use Greedy (Current POC) when:
- ✅ Rapid prototyping and iteration
- ✅ Need predictable, debuggable behavior
- ✅ Limited training data available
- ✅ Real-time latency critical (< 1ms allocation overhead)
- ✅ System stability paramount

### Consider Learned Policy when:
- 🔬 Large dataset of user editing sessions available
- 🔬 Can afford training infrastructure and time
- 🔬 User behavior patterns are complex and worth modeling
- 🔬 Willing to trade interpretability for performance
- 🔬 Can handle occasional policy failures gracefully

### Consider Optimization when:
- 🔬 Budget utilization is critical bottleneck
- 🔬 Can afford 10-100ms allocation latency
- 🔬 Problem size is small (< 100 candidate blocks)
- 🔬 Determinism and reproducibility required
- 🔬 Objective function is well-defined

---

## Future Enhancements

Based on [[Focus Allocator]] future directions:

### 1. Hybrid Approaches
```pseudocode
function hybrid_allocator(scores, context):
    # Use greedy for initial pruning
    candidates = greedy_filter(scores, top_k=50)

    # Run optimization on filtered set
    actions = optimize_subset(candidates, W_max)

    return actions
```

Combines greedy speed with optimization quality on smaller problem.

### 2. Multi-Objective Optimization
Jointly optimize:
- **Utility maximization:** Expand high-focus regions
- **Budget adherence:** Stay near `W_max` without waste
- **Churn minimization:** Reduce action count
- **Cursor proximity:** Prefer changes near user attention

### 3. Soft Assignments
Instead of discrete expand/collapse:
```pseudocode
# Allow fractional detail levels
action = {
    'expand': 0.7,   # 70% expanded
    'keep': 0.2,     # 20% unchanged
    'collapse': 0.1  # 10% collapsed
}

# Interpolate LOD representations
blended_embedding = (
    0.7 * expand_embedding(block) +
    0.2 * current_embedding(block) +
    0.1 * collapse_embedding(block)
)
```

Enables smoother transitions and potentially better gradients for learning.

### 4. Hierarchical Planning
```pseudocode
# Two-stage allocation
stage1 = high_level_policy(document_structure)  # coarse LOD decisions
stage2 = fine_grained_policy(stage1_result)     # per-block refinement
```

Reduces search space for complex documents.

### 5. Meta-Learning
Train allocator to quickly adapt to new documents or users:
```pseudocode
# Few-shot adaptation
pretrained_policy = load_pretrained()
adapted_policy = meta_adapt(
    pretrained_policy,
    few_examples=user_first_5_edits
)
```

---

## Implementation Roadmap

**Phase 1 (Current):** Greedy allocator with fixed thresholds
- ✅ Simple priority queue implementation
- ✅ Cooldown-based stability
- ✅ Hard budget constraints

**Phase 2:** Enhanced greedy
- ⏳ Adaptive thresholds based on utilization
- ⏳ Better tie-breaking (cursor proximity, recency)
- ⏳ Soft budget constraints with preference

**Phase 3:** Optimization variant
- 📋 Mass-matching allocator
- 📋 Small linear program solver integration
- 📋 A/B test against greedy baseline

**Phase 4:** Learned policy
- 📋 Collect training data from POC usage
- 📋 Design reward function and training loop
- 📋 Implement differentiable router
- 📋 Production deployment with safety fallback to greedy

---

## Related Pages

### Core Components
- [[Focus Allocator]] — Main allocator documentation and greedy algorithm implementation
- [[LensNet]] — Produces signed focus scores that drive allocation decisions
- [[LensNet Scoring]] — How utilities are computed and calibrated for allocation
- [[LensNet Training]] — Training regime that shapes score distributions and budget balancing

### Working Context
- [[Working Context]] — The fixed-size target data structure modified by allocator actions
- [[Working Context Assembly]] — Initial context construction before allocation begins
- [[Working Context Refocusing]] — Complete refocus workflow integrating LensNet + allocator

### System Architecture
- [[MegaContext Tree]] — Source of spans and LOD candidates for allocation operations
- [[GistNet]] — Compression mechanism defining the block structure allocator navigates
- [[Invariants]] — Budget, contiguity, and legality constraints allocator must enforce

### Implementation & Operations
- [[POC Implementation]] — Current greedy strategy parameters and system constraints
- [[Runtime Loop]] — How allocator fits into the decode-ingest-score-allocate cycle
- [[Alternating Optimization]] — Training considerations for future learned policies
- [[Telemetry]] — Metrics for measuring allocation quality (swap rate, budget efficiency)

### Future Directions
- [[Training & Operations]] — Infrastructure for training learned allocator policies
- [[System Properties]] — Performance targets and scaling considerations for strategies

---

## References (Academic Context)

**Greedy allocation:**
- Efficient for online decision-making
- Used in cache replacement, attention routing
- Trade-off: speed vs. optimality

**Learned policies:**
- Policy gradient methods (REINFORCE, PPO)
- Differentiable planning (Gumbel-Softmax)
- Meta-learning for fast adaptation (MAML, Reptile)

**Optimization:**
- Knapsack variants for budget-constrained selection
- Assignment problems for mass matching
- Linear programming for continuous relaxation
