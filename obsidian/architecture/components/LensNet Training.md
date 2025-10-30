---
tags:
  - components
summary: Complete training methodology for LensNet including counterfactual labeling, multi-component loss functions, data generation, and optimization procedures.
---
# LensNet Training

[[LensNet]] learns to predict signed focus scores that guide the [[Focus Allocator]] to expand or collapse regions within the [[Working Context]], maintaining relevance while keeping compute constant. Training relies on **counterfactual supervision** derived from NLL deltas and multi-objective loss functions that balance accuracy, budget constraints, and legal actions.

## Overview

The training process teaches [[LensNet]] [1,2] to:
- **Regress** signed utility values `u_i` that quantify the benefit of expanding/collapsing each entry
- **Rank** entries correctly (higher-value expansions should score higher than lower-value ones)
- **Maintain budget neutrality** (total expansion cost ≈ total collapse refund)
- **Respect legality constraints** (L0 tokens cannot expand; L2 gists cannot collapse)

Training data is generated from **trace logs** of [[MegaContext Tree]] operations, where counterfactual ΔNLL measurements quantify the utility of each possible expand/collapse action.

## Counterfactual Labeling Methodology

### Target Utility Computation

Each entry in the [[Working Context]] receives a **signed target utility** `y_i` based on measuring the impact of focus actions on language modeling performance:

#### Expandable Items (L1/L2 Gists)
For gists that can be expanded into finer-grained children:

```python
def compute_expand_utility(gist_node, context, base_model):
    """
    Measure benefit of expanding a gist into its children.

    Returns:
        y_i > 0: Positive utility proportional to ΔNLL improvement
    """
    # Baseline: current context with gist
    nll_baseline = evaluate_nll(context, base_model)

    # Counterfactual: replace gist with its children
    context_expanded = replace_gist_with_children(context, gist_node)
    nll_expanded = evaluate_nll(context_expanded, base_model)

    # Positive utility = NLL reduction from expansion
    delta_nll = nll_baseline - nll_expanded
    y_i = max(0.0, delta_nll)  # Clip to positive

    return y_i
```

**Intuition:** If expanding a gist into detailed tokens improves the model's ability to predict upcoming text (lower NLL), that gist should receive a high positive utility score.

#### Collapsible Items (L0/L1 Spans)
For token spans that can be collapsed into parent gists:

```python
def compute_collapse_utility(token_span, context, base_model):
    """
    Measure cost of collapsing tokens into parent gist.

    Returns:
        y_i < 0: Negative utility proportional to NLL degradation
    """
    # Baseline: current context with detailed tokens
    nll_baseline = evaluate_nll(context, base_model)

    # Counterfactual: replace tokens with parent gist
    context_collapsed = replace_tokens_with_gist(context, token_span)
    nll_collapsed = evaluate_nll(context_collapsed, base_model)

    # Negative utility = NLL increase from collapse
    delta_nll = nll_collapsed - nll_baseline
    y_i = -max(0.0, delta_nll)  # Negative, magnitude = cost

    return y_i
```

**Intuition:** If collapsing tokens into a lossy gist increases NLL (worse predictions), that span should receive a negative utility indicating the cost of compression.

#### Neutral Items
Entries that cannot be expanded or collapsed receive `y_i = 0` and are masked during loss computation.

### Batched Label Generation

During training data collection:

```python
def generate_training_batch(trace_log, base_model):
    """
    Generate supervised training batch from trace log.

    Args:
        trace_log: Sequence of (context_state, actions_taken, nll_deltas)
        base_model: Frozen base LLM for NLL evaluation

    Returns:
        batch: {
            'context_embeddings': [B, N, d],
            'tail_gists': [B, K, d],
            'levels': [B, N],
            'span_width': [B, N],
            'distance_to_cursor': [B, N],
            'target_utilities': [B, N],
            'action_mask': [B, N]  # Which entries are actionable
        }
    """
    batches = []

    for snapshot in trace_log:
        context = snapshot.working_context
        N = len(context.entries)

        # Initialize targets
        y = torch.zeros(N)
        mask = torch.zeros(N, dtype=torch.bool)

        # Compute expand utilities for gists
        for i, entry in enumerate(context.entries):
            if entry.level in [1, 2] and entry.has_children():
                y[i] = compute_expand_utility(entry, context, base_model)
                mask[i] = True

        # Compute collapse utilities for token spans
        for i, entry in enumerate(context.entries):
            if entry.level in [0, 1] and entry.has_parent():
                y[i] = compute_collapse_utility(entry, context, base_model)
                mask[i] = True

        batches.append({
            'context_embeddings': context.get_embeddings(),
            'tail_gists': context.get_tail_gists(K=6),
            'levels': torch.tensor([e.level for e in context.entries]),
            'span_width': torch.tensor([e.span_width for e in context.entries]),
            'distance_to_cursor': torch.tensor([e.distance for e in context.entries]),
            'target_utilities': y,
            'action_mask': mask
        })

    return collate_batch(batches)
```

## Loss Functions

The total training loss combines four components that balance multiple objectives:

```python
L_total = L_reg + λ_rank * L_rank + λ_budget * L_budget + L_illegal
```

Where `λ_rank = 0.5`, `λ_budget = 0.1` in the POC.

### 1. Regression Loss (`L_reg`)

**Objective:** Predict the exact utility value for each entry.

```python
def regression_loss(predictions, targets, mask):
    """
    Mean squared error on actionable entries.

    Args:
        predictions: [B, N] - LensNet output scores u_i
        targets: [B, N] - Counterfactual utilities y_i
        mask: [B, N] - Boolean mask for actionable entries

    Returns:
        L_reg: Scalar loss
    """
    masked_pred = predictions[mask]
    masked_target = targets[mask]

    squared_errors = (masked_pred - masked_target) ** 2
    L_reg = squared_errors.mean()

    return L_reg
```

**Formula:**
$$
L_{\text{reg}} = \frac{1}{|M|} \sum_{i \in M} (u_i - y_i)^2
$$

Where:
- `u_i`: Predicted score from [[LensNet]]
- `y_i`: Target utility from counterfactual NLL delta
- `M`: Set of actionable entries (mask = True)

### 2. Ranking Loss (`L_rank`)

**Objective:** Ensure relative ordering matches utility magnitudes. High-value expansions should score above low-value expansions; low-cost collapses should score below high-cost collapses.

```python
def ranking_loss(predictions, targets, mask, margin=0.1):
    """
    Pairwise ranking loss using softplus hinge.

    For every pair (i, j) where y_i > y_j, we want u_i > u_j.
    """
    masked_pred = predictions[mask]
    masked_target = targets[mask]

    # Generate all pairs where target_i > target_j
    n = masked_target.shape[0]
    pairs = []

    for i in range(n):
        for j in range(n):
            if masked_target[i] > masked_target[j] + margin:
                # Penalize if predicted order is wrong
                violation = -(masked_pred[i] - masked_pred[j])
                loss = torch.nn.functional.softplus(violation)
                pairs.append(loss)

    L_rank = torch.stack(pairs).mean() if pairs else torch.tensor(0.0)
    return L_rank
```

**Formula (per pair):**
$$
L_{\text{rank}} = \text{softplus}(-(u_i - u_j)) \quad \text{where } y_i > y_j
$$

**Softplus function:**
$$
\text{softplus}(x) = \log(1 + e^x)
$$

This provides a smooth, differentiable approximation to the hinge loss `max(0, -x)`.

### 3. Budget Regularizer (`L_budget`)

**Objective:** Encourage net-zero token change per block to maintain constant [[Working Context]] size.

```python
def budget_loss(predictions, span_width, levels, eps=1e-6):
    """
    Zero-sum budget regularizer.

    Computes total expansion cost and collapse refund,
    penalizes imbalance.

    Args:
        predictions: [B, N] - Signed scores u_i
        span_width: [B, N] - Token count per entry
        levels: [B, N] - Hierarchy level (0/1/2)
        eps: Numerical stability constant

    Returns:
        L_budget: Scalar loss
    """
    # Expansion cost (positive scores on L1/L2)
    expandable_mask = (levels >= 1)
    expand_scores = torch.relu(predictions) * expandable_mask

    # Cost = score * children_width
    # Approximate: assume L1 expands 8x, L2 expands 64x
    expansion_multiplier = torch.where(
        levels == 1,
        torch.tensor(8.0),
        torch.where(levels == 2, torch.tensor(64.0), torch.tensor(1.0))
    )

    c_plus = span_width * (expansion_multiplier - 1)  # Net token increase
    P = (expand_scores * c_plus).sum()

    # Collapse refund (negative scores on L0/L1)
    collapsible_mask = (levels <= 1)
    collapse_scores = torch.relu(-predictions) * collapsible_mask

    # Refund = score * current_width (tokens freed)
    c_minus = span_width
    N = (collapse_scores * c_minus).sum()

    # Penalize imbalance
    balance_ratio = (P - N) / (eps + P + N)
    L_budget = balance_ratio ** 2

    return L_budget
```

**Formula:**
$$
P = \sum_i c_i^+ \cdot \text{ReLU}(u_i)
$$
$$
N = \sum_i c_i^- \cdot \text{ReLU}(-u_i)
$$
$$
L_{\text{budget}} = \left( \frac{P - N}{\epsilon + P + N} \right)^2
$$

Where:
- `c_i^+`: Token cost if entry `i` is expanded (children_width - current_width)
- `c_i^-`: Token refund if entry `i` is collapsed (current_width - parent_width)
- `P`: Total expansion pressure
- `N`: Total collapse pressure
- `ε`: Stability constant (`1e-6`)

**Intuition:** When `P ≈ N`, the ratio approaches zero and loss is minimal. When expansion >> collapse (or vice versa), loss grows quadratically.

### 4. Illegality Penalties (`L_illegal`)

**Objective:** Discourage impossible actions that violate hierarchy constraints.

```python
def illegality_loss(predictions, levels, alpha=0.3, beta=0.3):
    """
    Penalize illegal focus directions.

    - L0 tokens cannot expand (no children exist)
    - L2 gists cannot collapse (no parent exists)

    Args:
        predictions: [B, N] - Signed scores u_i
        levels: [B, N] - Hierarchy level
        alpha: Penalty weight for illegal expansions
        beta: Penalty weight for illegal collapses

    Returns:
        L_illegal: Scalar loss
    """
    # Penalize positive scores on L0 (cannot expand)
    l0_mask = (levels == 0)
    illegal_expand = torch.relu(predictions[l0_mask]).sum()

    # Penalize negative scores on L2 (cannot collapse)
    l2_mask = (levels == 2)
    illegal_collapse = torch.relu(-predictions[l2_mask]).sum()

    L_illegal = alpha * illegal_expand + beta * illegal_collapse

    return L_illegal
```

**Formula:**
$$
L_{\text{illegal}} = \alpha \sum_{i \in L_0} \text{ReLU}(u_i) + \beta \sum_{i \in L_2} \text{ReLU}(-u_i)
$$

Where `α = β = 0.3` in the POC.

**Note:** At inference, illegal directions are **hard-masked to zero** (not soft-penalized), but training uses soft penalties to provide gradient signal.

### Total Loss Implementation

```python
class LensNetLoss(torch.nn.Module):
    def __init__(self, lambda_rank=0.5, lambda_budget=0.1,
                 alpha_illegal=0.3, beta_illegal=0.3):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_budget = lambda_budget
        self.alpha_illegal = alpha_illegal
        self.beta_illegal = beta_illegal

    def forward(self, predictions, batch):
        """
        Compute total loss.

        Args:
            predictions: [B, N] - LensNet output scores
            batch: Dict with keys:
                - target_utilities: [B, N]
                - action_mask: [B, N]
                - levels: [B, N]
                - span_width: [B, N]
        """
        L_reg = regression_loss(
            predictions,
            batch['target_utilities'],
            batch['action_mask']
        )

        L_rank = ranking_loss(
            predictions,
            batch['target_utilities'],
            batch['action_mask']
        )

        L_budget = budget_loss(
            predictions,
            batch['span_width'],
            batch['levels']
        )

        L_illegal = illegality_loss(
            predictions,
            batch['levels'],
            alpha=self.alpha_illegal,
            beta=self.beta_illegal
        )

        L_total = (L_reg +
                   self.lambda_rank * L_rank +
                   self.lambda_budget * L_budget +
                   L_illegal)

        return {
            'loss': L_total,
            'L_reg': L_reg.item(),
            'L_rank': L_rank.item(),
            'L_budget': L_budget.item(),
            'L_illegal': L_illegal.item()
        }
```

## Training Data Generation

### Trace Log Collection

Training data is generated from production traces of [[MegaContext Tree]] usage:

```python
class TraceCollector:
    """
    Collects training examples from live MegaContext sessions.
    """
    def __init__(self, base_model, gistnet, tree):
        self.base_model = base_model
        self.gistnet = gistnet
        self.tree = tree
        self.traces = []

    def collect_snapshot(self, working_context, query_tokens):
        """
        Capture state and compute counterfactual utilities.

        Called every K tokens (32 in POC) when LensNet would run.
        """
        N = len(working_context.entries)

        # Evaluate baseline NLL
        nll_baseline = self.evaluate_nll(
            working_context,
            query_tokens
        )

        # Generate counterfactual contexts
        counterfactuals = []

        for i, entry in enumerate(working_context.entries):
            # Try expansion
            if entry.can_expand():
                ctx_expanded = self.simulate_expand(working_context, i)
                nll_expanded = self.evaluate_nll(ctx_expanded, query_tokens)
                delta_nll_expand = nll_baseline - nll_expanded
                counterfactuals.append({
                    'index': i,
                    'action': 'expand',
                    'utility': max(0.0, delta_nll_expand)
                })

            # Try collapse
            if entry.can_collapse():
                ctx_collapsed = self.simulate_collapse(working_context, i)
                nll_collapsed = self.evaluate_nll(ctx_collapsed, query_tokens)
                delta_nll_collapse = nll_collapsed - nll_baseline
                counterfactuals.append({
                    'index': i,
                    'action': 'collapse',
                    'utility': -max(0.0, delta_nll_collapse)
                })

        # Store training example
        self.traces.append({
            'context_embeddings': working_context.get_embeddings(),
            'tail_gists': working_context.get_tail_gists(K=6),
            'levels': [e.level for e in working_context.entries],
            'span_width': [e.span_width for e in working_context.entries],
            'distance_to_cursor': [e.distance_to_cursor for e in working_context.entries],
            'utilities': counterfactuals,
            'timestamp': time.time()
        })

    def evaluate_nll(self, context, query_tokens):
        """
        Measure language modeling loss on upcoming query tokens.
        """
        with torch.no_grad():
            logits = self.base_model(context.embeddings)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                query_tokens.view(-1)
            )
        return loss.item()
```

### Data Augmentation

To increase training diversity:

```python
def augment_training_data(traces):
    """
    Apply augmentations to training traces.

    Techniques:
    - Temporal jitter: Sample snapshots at varying cadences
    - Subsampling: Use partial context windows
    - Utility smoothing: Add small noise to y_i values
    """
    augmented = []

    for trace in traces:
        # Original example
        augmented.append(trace)

        # Temporal jitter: Skip every other token
        if random.random() < 0.3:
            jittered = subsample_context(trace, stride=2)
            augmented.append(jittered)

        # Utility smoothing: Add Gaussian noise
        if random.random() < 0.2:
            noisy = add_utility_noise(trace, std=0.05)
            augmented.append(noisy)

    return augmented
```

## Optimization Settings

### POC Training Configuration

```python
# Model architecture (Perceiver-inspired [1,2])
model_config = {
    'd_model': 1024,          # Base model embedding dim
    'd_lens': 512,            # LensNet projection dim
    'n_heads': 8,             # Attention heads
    'n_stacks': 2,            # Dual cross-attention layers
    'dropout': 0.1
}

# Optimization (LoRA adapters [3])
optimizer_config = {
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 0.01
}

# Loss weights
loss_config = {
    'lambda_rank': 0.5,
    'lambda_budget': 0.1,
    'alpha_illegal': 0.3,
    'beta_illegal': 0.3
}

# Training dynamics
training_config = {
    'batch_size': 16,
    'gradient_clip': 1.0,
    'warmup_steps': 1000,
    'total_steps': 50000,
    'eval_interval': 500,
    'save_interval': 2000
}
```

### Learning Rate Schedule

```python
def get_lr_schedule(optimizer, warmup_steps=1000):
    """
    Linear warmup followed by cosine annealing.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Gradient Clipping

```python
def training_step(model, batch, optimizer, max_grad_norm=1.0):
    """
    Single training iteration with gradient clipping.
    """
    optimizer.zero_grad()

    predictions = model(
        context=batch['context_embeddings'],
        tail_gists=batch['tail_gists'],
        levels=batch['levels'],
        span_width=batch['span_width'],
        distance_to_cursor=batch['distance_to_cursor']
    )

    loss_dict = criterion(predictions, batch)
    loss = loss_dict['loss']

    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return loss_dict
```

## Training Loop

### Main Training Procedure

```python
def train_lensnet(model, train_loader, val_loader, config):
    """
    Complete training loop for LensNet.

    Args:
        model: LensNet module
        train_loader: DataLoader for training traces
        val_loader: DataLoader for validation traces
        config: Training configuration dict
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=config['betas'],
        weight_decay=config['weight_decay']
    )

    scheduler = get_lr_schedule(optimizer, config['warmup_steps'])
    criterion = LensNetLoss(
        lambda_rank=config['lambda_rank'],
        lambda_budget=config['lambda_budget'],
        alpha_illegal=config['alpha_illegal'],
        beta_illegal=config['beta_illegal']
    )

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss_dict = training_step(
                model, batch, optimizer,
                max_grad_norm=config['gradient_clip']
            )
            scheduler.step()

            epoch_losses.append(loss_dict['loss'])
            global_step += 1

            # Logging
            if global_step % 100 == 0:
                log_training_metrics(global_step, loss_dict, scheduler)

            # Validation
            if global_step % config['eval_interval'] == 0:
                val_metrics = validate(model, val_loader, criterion)

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    save_checkpoint(model, optimizer, global_step, 'best')

                model.train()

            # Checkpointing
            if global_step % config['save_interval'] == 0:
                save_checkpoint(model, optimizer, global_step, f'step_{global_step}')

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    return model
```

### Validation Procedure

```python
def validate(model, val_loader, criterion):
    """
    Evaluate model on validation set.

    Returns:
        metrics: Dict with loss components and ranking accuracy
    """
    model.eval()

    total_loss = 0.0
    component_losses = {'L_reg': 0.0, 'L_rank': 0.0,
                        'L_budget': 0.0, 'L_illegal': 0.0}
    ranking_correct = 0
    ranking_total = 0

    with torch.no_grad():
        for batch in val_loader:
            predictions = model(
                context=batch['context_embeddings'],
                tail_gists=batch['tail_gists'],
                levels=batch['levels'],
                span_width=batch['span_width'],
                distance_to_cursor=batch['distance_to_cursor']
            )

            loss_dict = criterion(predictions, batch)
            total_loss += loss_dict['loss']

            for key in component_losses:
                component_losses[key] += loss_dict[key]

            # Compute ranking accuracy
            acc, count = compute_ranking_accuracy(
                predictions, batch['target_utilities'], batch['action_mask']
            )
            ranking_correct += acc
            ranking_total += count

    n_batches = len(val_loader)
    metrics = {
        'loss': total_loss / n_batches,
        'ranking_accuracy': ranking_correct / max(1, ranking_total),
        **{k: v / n_batches for k, v in component_losses.items()}
    }

    return metrics
```

## Evaluation Metrics

### Primary Metrics

```python
def compute_ranking_accuracy(predictions, targets, mask):
    """
    Fraction of correctly ordered pairs.

    For all pairs (i, j) where y_i > y_j, check if u_i > u_j.
    """
    masked_pred = predictions[mask]
    masked_target = targets[mask]

    correct = 0
    total = 0

    for i in range(len(masked_pred)):
        for j in range(len(masked_pred)):
            if masked_target[i] > masked_target[j]:
                total += 1
                if masked_pred[i] > masked_pred[j]:
                    correct += 1

    return correct, total
```

### Utility Correlation

```python
def compute_utility_correlation(predictions, targets, mask):
    """
    Pearson correlation between predicted and target utilities.
    """
    masked_pred = predictions[mask].cpu().numpy()
    masked_target = targets[mask].cpu().numpy()

    if len(masked_pred) < 2:
        return 0.0

    corr, _ = scipy.stats.pearsonr(masked_pred, masked_target)
    return corr
```

### Budget Balance Score

```python
def compute_budget_balance(predictions, span_width, levels):
    """
    Measure how close expansion/collapse masses are to balanced.

    Returns:
        balance: 0.0 = perfectly balanced, 1.0 = maximally imbalanced
    """
    # Same computation as budget loss
    P, N = compute_budget_terms(predictions, span_width, levels)

    if P + N < 1e-6:
        return 0.0

    balance = abs(P - N) / (P + N)
    return balance
```

### Legality Compliance

```python
def compute_legality_rate(predictions, levels):
    """
    Fraction of predictions that respect hierarchy constraints.
    """
    violations = 0
    total = len(predictions)

    # Check L0 expansions
    l0_mask = (levels == 0)
    violations += (predictions[l0_mask] > 0).sum().item()

    # Check L2 collapses
    l2_mask = (levels == 2)
    violations += (predictions[l2_mask] < 0).sum().item()

    compliance = 1.0 - (violations / total)
    return compliance
```

### Comprehensive Evaluation

```python
def evaluate_model(model, test_loader):
    """
    Full evaluation with all metrics.
    """
    model.eval()

    all_metrics = {
        'mse': [],
        'ranking_accuracy': [],
        'utility_correlation': [],
        'budget_balance': [],
        'legality_compliance': []
    }

    with torch.no_grad():
        for batch in test_loader:
            predictions = model(
                context=batch['context_embeddings'],
                tail_gists=batch['tail_gists'],
                levels=batch['levels'],
                span_width=batch['span_width'],
                distance_to_cursor=batch['distance_to_cursor']
            )

            # Compute metrics
            mse = F.mse_loss(
                predictions[batch['action_mask']],
                batch['target_utilities'][batch['action_mask']]
            )
            all_metrics['mse'].append(mse.item())

            correct, total = compute_ranking_accuracy(
                predictions, batch['target_utilities'], batch['action_mask']
            )
            all_metrics['ranking_accuracy'].append(correct / max(1, total))

            corr = compute_utility_correlation(
                predictions, batch['target_utilities'], batch['action_mask']
            )
            all_metrics['utility_correlation'].append(corr)

            balance = compute_budget_balance(
                predictions, batch['span_width'], batch['levels']
            )
            all_metrics['budget_balance'].append(balance)

            compliance = compute_legality_rate(predictions, batch['levels'])
            all_metrics['legality_compliance'].append(compliance)

    # Aggregate
    final_metrics = {
        key: np.mean(values) for key, values in all_metrics.items()
    }

    return final_metrics
```

## Training Summary

| Aspect | Configuration |
|--------|---------------|
| **Supervision** | Counterfactual ΔNLL utilities from trace logs |
| **Label Sign** | Positive for expand, negative for collapse |
| **Primary Loss** | MSE regression (`L_reg`) |
| **Ranking Loss** | Softplus pairwise hinge (`L_rank`, weight 0.5) |
| **Budget Loss** | Quadratic balance penalty (`L_budget`, weight 0.1) |
| **Illegality Loss** | Soft penalties on invalid actions (`L_illegal`, weights 0.3) |
| **Optimizer** | AdamW with lr=1e-4, weight decay=0.01 |
| **Schedule** | Linear warmup (1k steps) + cosine annealing |
| **Batch Size** | 16 |
| **Gradient Clip** | 1.0 |
| **Training Steps** | ~50k |
| **Validation** | Every 500 steps |

**Key Insight:** The multi-objective loss balances conflicting goals (accuracy vs. budget neutrality vs. legality), enabling [[LensNet]] to make practical, deployable focus decisions that maintain constant [[Working Context]] size while maximizing relevance for the [[Focus Allocator]].

## Related Pages

### Core Components
- [[LensNet]] — Main architecture and design overview
- [[LensNet Scoring]] — How scores are computed and interpreted at inference time
- [[Focus Allocator]] — Consumes LensNet utilities to execute expand/collapse actions
- [[Working Context]] — The fixed-size GPU window optimized by LensNet predictions
- [[GistNet]] — Produces the gist embeddings that LensNet learns to expand/collapse

### Training & Optimization
- [[Alternating Optimization]] — Joint training regime coordinating GistNet and LensNet updates
- [[Training & Operations]] — Overall training strategy and operational procedures
- [[GistNet Training]] — Parallel training methodology for the compression component

### System Context
- [[MegaContext Tree]] — Source of trace logs and counterfactual evaluation contexts
- [[Working Context Refocusing]] — The runtime loop where trained LensNet predictions are applied
- [[POC Implementation]] — Concrete parameter values, batch sizes, and optimization settings

### Data & Metrics
- [[Telemetry]] — Metrics collection including ranking accuracy and budget balance
- [[Storage Format]] — Where trace logs and training checkpoints are persisted
- [[Invariants]] — Constraints that loss functions enforce (budget, legality, contiguity)

## References

1. **Perceiver** (Jaegle et al., 2021) — [[papers/Perceiver - 2103.03206v2|Analysis]] — Latent cross-attention bottleneck architecture
2. **Perceiver IO** (Jaegle et al., 2021) — [[papers/Perceiver IO - 2107.14795v3|Analysis]] — Query-based decoding for arbitrary structured outputs
3. **LoRA** (Hu et al., 2021) — [[papers/LoRA|Analysis]] — Low-rank adaptation used in GistNet/LensNet training

See [[Related Work]] for the complete bibliography of all research papers referenced throughout the documentation.
