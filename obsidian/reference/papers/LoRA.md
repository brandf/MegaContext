---
tags:
  - papers
  - reference
  - parameter-efficient-training
  - fine-tuning
summary: Low-Rank Adaptation for efficient fine-tuning of large language models with minimal parameter updates
---

# LoRA: Low-Rank Adaptation of Large Language Models

**PDF**: [LoRA - 2106.09685.pdf](LoRA%20-%202106.09685.pdf)

## Paper Metadata

- **Title**: LoRA: Low-Rank Adaptation of Large Language Models
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Affiliation**: Microsoft Corporation
- **Publication**: ICLR 2022
- **Year**: 2021 (arXiv preprint June 2021)
- **ArXiv ID**: 2106.09685
- **URL**: https://arxiv.org/abs/2106.09685
- **Key Contributions**: Low-rank decomposition for adapter modules, parameter-efficient fine-tuning, inference-time efficiency

---

## Overview

### What the Paper Introduces

LoRA (Low-Rank Adaptation) is a **parameter-efficient fine-tuning** technique that freezes pretrained model weights and injects trainable **low-rank decomposition matrices** into each layer. Instead of fine-tuning all parameters, LoRA adds trainable pairs of rank decomposition matrices (A, B) to the frozen weight matrices, reducing trainable parameters by 10,000× while maintaining or exceeding full fine-tuning performance.

### Key Innovation

The core insight is that **weight updates during adaptation have low "intrinsic rank"**—the effective dimensionality of changes needed for task adaptation is much smaller than the full parameter space. By parameterizing weight updates as low-rank matrices:

```
W_adapted = W_frozen + ΔW
where ΔW = B·A (low-rank factorization)
```

LoRA achieves dramatic parameter reduction:
- **Full fine-tuning**: All parameters updated (e.g., 175B for GPT-3)
- **LoRA**: Only low-rank matrices updated (e.g., 37M for GPT-3, **4,736× reduction**)

### Key Results

1. **GPT-3 175B on natural language tasks**: LoRA matches or exceeds full fine-tuning with 10,000× fewer trainable parameters
2. **GPT-2 on E2E NLG**: Better performance than adapters, prefix tuning, and full fine-tuning with rank r=4
3. **RoBERTa on GLUE**: Comparable or better than full fine-tuning with r=8
4. **No inference latency**: Unlike adapters (add sequential layers), LoRA merges into frozen weights at inference time
5. **Task switching**: Store multiple LoRA weights (small) and swap between tasks efficiently

All results demonstrate that **low-rank updates are sufficient** for effective adaptation across diverse tasks.

---

## Core Technical Concepts

### 1. Low-Rank Decomposition

**Problem**: Fine-tuning a pretrained weight matrix `W ∈ ℝ^(d×k)` requires updating all d×k parameters.

**LoRA Solution**: Represent the weight update as a low-rank decomposition:

```
h = W₀·x + ΔW·x = W₀·x + B·A·x
```

Where:
- `W₀ ∈ ℝ^(d×k)` = frozen pretrained weights
- `B ∈ ℝ^(d×r)` = trainable down-projection
- `A ∈ ℝ^(r×k)` = trainable up-projection
- `r << min(d, k)` = rank (e.g., r=1, 2, 4, 8)

**Parameter Count**:
- Full fine-tuning: d×k parameters
- LoRA: d×r + r×k = r×(d+k) parameters
- **Reduction ratio**: (d×k) / (r×(d+k)) ≈ **min(d,k) / r** when d≈k

**Example** (GPT-3 attention layer):
- d=k=12,288 (model dimension)
- Full: 12,288² = 150M parameters
- LoRA (r=4): 4×(12,288+12,288) = 98k parameters
- **Reduction: 1,536×**

### 2. Initialization and Scaling

**Initialization Strategy**:
- `A` is initialized with **Gaussian random** (similar to standard initialization)
- `B` is initialized to **zero**
- Result: `ΔW = B·A = 0` at initialization → LoRA starts as identity (no change)

**Scaling Factor**:
LoRA outputs are scaled by `α/r` where α is a constant (typically α=r or α=2r):

```
h = W₀·x + (α/r)·B·A·x
```

**Rationale**:
- Keeps activation magnitudes consistent across different ranks
- When switching rank r, don't need to retune learning rate
- α=r means LoRA updates have same initial magnitude regardless of rank

### 3. Which Layers to Adapt

LoRA can be applied to any dense layers, but **different layers have different adaptation needs**:

**Transformer Attention Layers**:
- Query projection: `W_q`
- Key projection: `W_k`
- Value projection: `W_v`
- Output projection: `W_o`

**Paper's Findings**:
- **Best results**: Adapt only `W_q` and `W_v` (query and value)
- Adapting all four (W_q, W_k, W_v, W_o) gives similar performance but doubles parameters
- Adapting only W_q gives worse results (query alone insufficient)
- **Not recommended**: Adapting MLP layers provides minimal benefit with higher cost

**MegaContext Implication**: Focus LoRA on attention layers, not feedforward MLPs.

### 4. Rank Selection

**Key Question**: How low can rank r go before performance degrades?

**Paper's Findings**:
- **r=1 to r=4**: Often sufficient for most tasks
- **r=8**: Matches or exceeds full fine-tuning on GLUE/SuperGLUE
- **r=64**: Diminishing returns; little improvement over r=8
- **Task dependence**: Some tasks (e.g., summarization) benefit from slightly higher rank

**Guidelines**:
- Start with **r=4 or r=8** (good default)
- Increase to r=16 or r=32 only if validation loss plateaus
- Rarely need r>64

### 5. Inference-Time Efficiency

**Merging at Inference**:
At deployment, LoRA weights can be merged into the frozen model:

```
W_deployed = W₀ + B·A
```

**Benefits**:
- **No additional latency**: Merged model has same size and speed as original
- **No architectural changes**: Standard transformer architecture preserved
- **Task switching**: Store multiple (B, A) pairs, swap by computing different W_deployed

**Comparison to Other Methods**:
- **Adapters**: Add sequential bottleneck layers → **inference slowdown**
- **Prefix tuning**: Reduce effective sequence length → **less capacity**
- **LoRA**: Zero inference overhead when merged

### 6. Multi-Task Support

**Scenario**: Deploy a single base model serving multiple tasks (e.g., translation, summarization, Q&A).

**LoRA Solution**:
1. Train separate (B_task1, A_task1), (B_task2, A_task2), ... for each task
2. At inference, load appropriate LoRA weights:
   ```
   W_task_i = W₀ + B_task_i · A_task_i
   ```
3. Switch tasks by swapping LoRA modules (small memory overhead)

**Storage Efficiency**:
- Base model W₀: 175B parameters (350GB at float16)
- Each LoRA task: ~37M parameters (74MB at float16)
- **1,000 tasks**: 350GB + 74GB = 424GB (only 21% overhead!)

---

## Relevance to MegaContext

### Direct Training Applications

MegaContext involves training **two small neural networks** atop a **frozen base LLM**:

1. **[[GistNet]]**: 32→1→32→1 compression network (~10M parameters)
2. **[[LensNet]]**: Cross-attention controller (~5-10M parameters)

**LoRA's Relevance**:
- Both networks need **base model adaptation** to align with the frozen LLM's embedding space
- Full fine-tuning of the base model is impractical (billions of parameters)
- LoRA provides **efficient adaptation** with minimal overhead

### Application 1: Base Model Adaptation Layer

**Current POC Design**: Small LoRA on top of frozen base model for MegaContext-specific adjustments.

**LoRA Configuration**:
```python
# SmolLM3-3B base model (3B parameters, frozen)
# LoRA adapter for MegaContext context processing

lora_config = {
    "target_modules": ["q_proj", "v_proj"],  # Only adapt attention
    "rank": 8,  # Low rank for minimal overhead
    "alpha": 16,  # Scaling factor (2×rank)
    "dropout": 0.05,  # Light regularization
}

# Trainable parameters: ~2M (0.067% of base model)
```

**Purpose**:
- Adapt base model to **work with gist embeddings** (which have different distributional properties than token embeddings)
- Learn to **integrate working context layouts** with varying LOD levels
- Fine-tune **positional encoding handling** for teleported spans

**Training**:
- Freeze base model
- Train LoRA + GistNet + LensNet jointly
- LoRA learns to bridge gist → base model interface

### Application 2: GistNet Initialization

**Challenge**: [[GistNet]] must produce embeddings in the **base model's embedding space** (d=2,560 for SmolLM3-3B).

**LoRA-Inspired Approach**: Initialize GistNet's final projection as a **low-rank bottleneck**:

```python
class GistNet:
    def __init__(self, d_model=2560, d_hidden=512, rank=32):
        # Encoder: 32 tokens → 1 slot query
        self.encoder = SlotAttentionEncoder(d_model, d_hidden)

        # Low-rank projection to base model space
        self.to_gist_A = nn.Linear(d_hidden, rank, bias=False)
        self.to_gist_B = nn.Linear(rank, d_model, bias=False)

        # Initialize like LoRA: A~N(0,σ²), B=0
        nn.init.normal_(self.to_gist_A.weight, std=0.02)
        nn.init.zeros_(self.to_gist_B.weight)

    def forward(self, tokens):
        slot = self.encoder(tokens)  # [1, d_hidden]
        gist = self.to_gist_B(self.to_gist_A(slot))  # [1, d_model]
        return gist
```

**Benefits**:
- **Stable initialization**: Starts with identity (gist ≈ mean of input tokens)
- **Intrinsic rank constraint**: Forces gist to use low-dimensional subspace (improves generalization)
- **Faster training**: Fewer parameters in bottleneck → faster convergence

**Rationale**: Gist embeddings likely live in a **low-dimensional manifold** within the full d-dimensional space (similar to weight updates in LoRA).

### Application 3: LensNet Efficiency

**Challenge**: [[LensNet]] performs cross-attention over working context entries (100-1000 entries) to produce focus scores.

**Current Design**:
```python
class LensNet:
    def __init__(self, d_model=2560, d_attn=256):
        # Cross-attention layers
        self.query_proj = nn.Linear(d_model, d_attn)
        self.key_proj = nn.Linear(d_model, d_attn)
        self.value_proj = nn.Linear(d_model, d_attn)
        self.out_proj = nn.Linear(d_attn, 1)  # Focus score
```

**LoRA Enhancement**:
Instead of full-rank projections, use **low-rank factorizations**:

```python
class LensNetLoRA:
    def __init__(self, d_model=2560, d_attn=256, rank=16):
        # Low-rank query/key/value projections
        self.q_down = nn.Linear(d_model, rank, bias=False)
        self.q_up = nn.Linear(rank, d_attn, bias=False)

        self.k_down = nn.Linear(d_model, rank, bias=False)
        self.k_up = nn.Linear(rank, d_attn, bias=False)

        # Similar for value
        # ...

    def forward(self, wc_entries):
        Q = self.q_up(self.q_down(wc_entries))  # Low-rank query
        K = self.k_up(self.k_down(wc_entries))  # Low-rank key
        # ... attention computation
```

**Savings**:
- Full-rank: `d_model × d_attn = 2,560 × 256 = 655k` parameters per projection
- Low-rank (r=16): `2,560×16 + 16×256 = 45k` parameters per projection
- **Reduction: 14.5×** with minimal performance loss

### Application 4: Task-Specific Gisting

**Use Case**: Different domains (code, narrative, structured data) may need **specialized gisting strategies**.

**LoRA Solution**: Train **domain-specific LoRA modules** for GistNet:

```python
# Base GistNet (trained on mixed data)
gist_net_base = GistNet(d_model=2560)

# Domain-specific LoRA adaptations
lora_code = LoRAModule(rank=8)       # Code compression
lora_narrative = LoRAModule(rank=8)  # Prose compression
lora_structured = LoRAModule(rank=8) # JSON/XML compression

# At runtime, select appropriate LoRA
def create_gist(tokens, domain="general"):
    features = gist_net_base.encode(tokens)
    if domain == "code":
        features = features + lora_code(features)
    elif domain == "narrative":
        features = features + lora_narrative(features)
    # ...
    return gist_net_base.decode(features)
```

**Benefits**:
- **Shared base**: One GistNet handles all domains (general capability)
- **Specialization**: Each LoRA adds domain-specific refinements
- **Efficiency**: Each LoRA is tiny (~100k parameters), enabling many domain adapters

### Application 5: Continual Learning

**Scenario**: As MegaContext is deployed, users may want to **adapt to new domains** without retraining from scratch.

**LoRA Approach**:
1. **Freeze** GistNet_base and LensNet_base (trained on general data)
2. **Train new LoRA adapters** on domain-specific data
3. **Compose** base + LoRA for domain-adapted MegaContext

**Example** (adapting to medical documents):
```bash
# Training
python train_lora.py \
    --base-model megacontext-v1 \
    --domain medical \
    --lora-rank 8 \
    --lora-alpha 16 \
    --data medical_corpus.jsonl \
    --output lora_medical.pt

# Inference
megacontext_system = MegaContext.load("megacontext-v1")
megacontext_system.load_lora("lora_medical.pt")
# Now processes medical documents with specialized compression
```

**Advantages**:
- **No catastrophic forgetting**: Base model unchanged
- **Fast adaptation**: Train only ~1M LoRA parameters (hours, not days)
- **Multi-domain**: Load multiple LoRAs simultaneously if memory permits

---

## What We Can Use

### 1. LoRA-Initialized GistNet Projection

**Implementation**:
```python
class GistNetWithLoRAProjection(nn.Module):
    def __init__(self, d_model=2560, d_slot=512, rank=32):
        super().__init__()
        self.rank = rank

        # Slot attention encoder (32 tokens → 1 slot)
        self.slot_encoder = SlotAttentionBlock(
            n_slots=1,
            d_model=d_model,
            d_slot=d_slot,
        )

        # LoRA-style low-rank projection
        self.gist_proj_down = nn.Linear(d_slot, rank, bias=False)
        self.gist_proj_up = nn.Linear(rank, d_model, bias=False)

        # LoRA initialization
        nn.init.kaiming_uniform_(self.gist_proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gist_proj_up.weight)  # Start at identity

        # Scaling factor
        self.alpha = rank * 2  # Typical: 2×rank

    def forward(self, token_embeddings):
        # token_embeddings: [batch, 32, d_model]
        slot = self.slot_encoder(token_embeddings)  # [batch, 1, d_slot]

        # Low-rank projection with scaling
        hidden = self.gist_proj_down(slot)  # [batch, 1, rank]
        gist = self.gist_proj_up(hidden)  # [batch, 1, d_model]
        gist = gist * (self.alpha / self.rank)  # Scale

        return gist
```

**Training Benefits**:
- **Stable start**: Zero initialization means early training doesn't corrupt base model embeddings
- **Gradual learning**: Low rank forces gradual, structured exploration of embedding space
- **Regularization**: Intrinsic low-rank constraint prevents overfitting to training data

**Experimentation**:
- Try ranks r ∈ {8, 16, 32, 64}
- Measure ΔNLL@H vs. rank (find minimum sufficient rank)
- Compare to full-rank projection (d_slot → d_model directly)

### 2. Efficient LensNet with Low-Rank Attention

**Problem**: [[LensNet]] cross-attention has high parameter count (Q, K, V projections each d_model × d_attn).

**LoRA Solution**:
```python
class LoRALinear(nn.Module):
    """LoRA-enhanced linear layer."""
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Frozen "base" projection (can be pretrained or zero)
        self.base = nn.Linear(in_features, out_features, bias=True)
        self.base.weight.requires_grad = False

        # Trainable low-rank adaptation
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # LoRA init
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x))
        return base_out + (self.alpha / self.rank) * lora_out

class LensNetWithLoRA(nn.Module):
    def __init__(self, d_model=2560, d_attn=256, rank=16):
        super().__init__()
        # Use LoRA for all attention projections
        self.q_proj = LoRALinear(d_model, d_attn, rank=rank)
        self.k_proj = LoRALinear(d_model, d_attn, rank=rank)
        self.v_proj = LoRALinear(d_model, d_attn, rank=rank)
        self.out_proj = nn.Linear(d_attn, 1)  # Focus score head

    def forward(self, wc_entries, conditioning):
        Q = self.q_proj(conditioning)  # Query from current state
        K = self.k_proj(wc_entries)    # Keys from WC entries
        V = self.v_proj(wc_entries)    # Values from WC entries

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Produce focus scores
        return self.out_proj(context)  # [batch, n_entries, 1]
```

**Parameter Savings**:
- Full LensNet: `3 × (2,560 × 256) ≈ 2M` parameters (Q, K, V)
- LoRA LensNet (r=16): `3 × (2,560×16 + 16×256) ≈ 135k` parameters
- **Reduction: 14.8×**

### 3. Base Model LoRA Adapter

**Purpose**: Adapt frozen base LLM to work with MegaContext gist embeddings and working context layouts.

**Configuration**:
```python
from peft import LoraConfig, get_peft_model

# Load frozen base model
base_model = AutoModelForCausalLM.from_pretrained("SmolLM3-3B")
for param in base_model.parameters():
    param.requires_grad = False

# Add LoRA adapter
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,                        # Rank (start small)
    lora_alpha=16,              # Scaling (2×rank typical)
    lora_dropout=0.05,          # Light regularization
    target_modules=["q_proj", "v_proj"],  # Only attention
    bias="none",                # Don't adapt biases
)

model_with_lora = get_peft_model(base_model, lora_config)
print(f"Trainable params: {model_with_lora.print_trainable_parameters()}")
# Output: Trainable params: 2.1M / 3B (0.07%)
```

**Training Loop**:
```python
# Joint training: LoRA + GistNet + LensNet
optimizer = AdamW([
    {"params": model_with_lora.parameters(), "lr": 1e-4},
    {"params": gist_net.parameters(), "lr": 3e-4},
    {"params": lens_net.parameters(), "lr": 3e-4},
])

for batch in train_loader:
    # 1. Create gists
    gists = gist_net(batch["tokens"])

    # 2. Assemble working context (mix LOD0/LOD1/LOD2)
    wc = assemble_working_context(gists, batch["focus_layout"])

    # 3. Forward through base model with LoRA
    logits = model_with_lora(wc)

    # 4. Compute loss
    loss = compute_megacontext_loss(logits, batch["targets"], gists)

    # 5. Backprop through all components
    loss.backward()
    optimizer.step()
```

**Inference** (merge LoRA for deployment):
```python
# Merge LoRA into base model
merged_model = model_with_lora.merge_and_unload()
# Now `merged_model` is standard transformer, no LoRA overhead
```

---

## Limitations & Risks

### 1. Low-Rank Bottleneck Capacity

**LoRA Limitation**: If rank r is too low, the model cannot express necessary updates, leading to performance degradation.

**MegaContext Context**: [[GistNet]] must compress 32 diverse tokens into 1 embedding. If gist rank is too low, information is lost (high ΔNLL@H).

**Mitigation**: Run ablation studies on rank ∈ {8, 16, 32, 64}, measure ΔNLL@H vs. rank to find minimum sufficient rank.

### 2. Initialization Sensitivity

**LoRA Limitation**: Zero initialization of B ensures starting at identity, but convergence speed depends on initialization of A and scaling factor α.

**Mitigation**: Follow LoRA recipe (Kaiming init for A, zero for B), use α=2r, apply gradient clipping.

### 3. Multi-Task Merging Overhead

**LoRA Limitation**: Multi-task deployment requires either separate merged models (high memory) or runtime LoRA switching (latency overhead).

**Mitigation**: Use hybrid approach—merge top-3 most common domains, keep rare domains as LoRA modules.

---

## Potential Follow-Up Reading

### LoRA Extensions

1. **"QLoRA: Efficient Finetuning of Quantized LLMs"** (2023, Dettmers et al.) - Combines LoRA with quantization
2. **"AdaLoRA: Adaptive Budget Allocation"** (2023, Zhang et al.) - Per-layer rank allocation
3. **"LoRA-FA: Memory-Efficient Low-Rank Adaptation"** (2023) - Frozen-A variant

### Parameter-Efficient Alternatives

4. **"Prefix Tuning"** (2021, Li & Liang) - Comparison point
5. **"Adapter Layers"** (2019, Houlsby et al.) - Sequential bottlenecks
6. **"Compacter"** (2021, Mahabadi et al.) - Hypercomplex adapters

### Theory

7. **"Intrinsic Dimensionality Explains Fine-Tuning"** (2020, Aghajanyan et al.) - Theoretical basis for LoRA
8. **"Geometry of Loss Surfaces"** (2017, Pennington & Worah) - When low-rank works

---

## Open Questions for MegaContext

### 1. Optimal Rank for GistNet Projection

**Question**: What is the minimum rank for gist projection (d_slot → d_model) that preserves substitutability?

**Experiment**: Train variants with ranks r ∈ {8, 16, 32, 64, 128}, measure ΔNLL@H, find elbow point.

### 2. Layer-Wise Rank Allocation in LensNet

**Question**: Should all [[LensNet]] layers use the same rank, or vary by depth?

**Hypothesis**: Early layers (broad context) need low rank (r=8), later layers (fine scoring) need higher rank (r=32).

### 3. LoRA for Base Model: Which Layers?

**Question**: Adapt all transformer layers or only first/last layers?

**Experiment**: Compare all-layers vs. ends-only vs. last-only configurations.

### 4. Intrinsic Rank of Gist Embeddings

**Question**: What is the true intrinsic dimensionality of gist embeddings?

**Method**: Collect 100k gists, apply PCA, find rank for 95% explained variance.

### 5. Domain-Specific LoRA Training Order

**Question**: If training LoRAs sequentially, does order matter to minimize interference?

**Hypothesis**: Train from most-similar-to-general to most-different.

### 6. Merged vs. Modular Deployment

**Question**: For production, merge LoRAs (fast) or keep modular (flexible)?

**Recommendation**: Hybrid—merge top-3 common domains, keep rare as LoRA modules.

### 7. LoRA for Continual Learning

**Question**: Can users train custom LoRA adapters after MegaContext release?

**Approach**: Freeze base, train small user-specific LoRA (rank 8) on user data (~1M tokens).

### 8. LoRA Scaling Factor Tuning

**Question**: Is α=2r optimal for MegaContext, or should we use different scaling?

**Experiment**: Fix r=16, try α ∈ {8, 16, 32, 64}, measure ΔNLL@H and embedding norms.

---

## Related Pages

### Core MegaContext Components
- [[GistNet]] (main application area)
- [[GistNet Training]] (LoRA-enhanced training)
- [[LensNet]] (low-rank attention)
- [[LensNet Training]] (LoRA adaptation)
- [[Glossary#Base Model|Base Model]] (LoRA adapter target)
- [[Working Context]]

### Training & Optimization
- [[Training & Operations]]
- [[MegaContext End-to-End Training]]
- [[GistNet Architecture Details]]

### Related Papers
- [[reference/papers/Perceiver - 2103.03206v2.md|Perceiver]] (cross-attention efficiency)
- [[reference/papers/Perceiver IO - 2107.14795v3.md|Perceiver IO]] (multi-modal bottleneck)
- [[reference/papers/Gist Tokens - 2304.08467v3.md|Gist Tokens]] (prompt compression)
- [[Knowledge Distillation]] (teacher-student training)

### Concepts
- [[Glossary#Base Model]]
- [[Glossary#GistNet]]
- [[Glossary#LensNet]]

---

## Summary

LoRA enables **parameter-efficient fine-tuning** through low-rank decomposition of weight updates, achieving 10,000× parameter reduction with minimal quality loss. For MegaContext, LoRA is directly applicable to:

1. **Base model adaptation** - Small LoRA (r=8, ~2M params) adapts frozen LLM to gist embeddings
2. **GistNet projection** - Low-rank bottleneck (r=32) for stable initialization and efficient gisting
3. **LensNet efficiency** - Low-rank Q/K/V projections (r=16) reduce parameters by 14×
4. **Domain specialization** - Multiple tiny LoRAs (~100k params each) enable multi-domain support
5. **Continual learning** - Users can train custom LoRAs without retraining entire system

The key insight—**updates occupy low-dimensional subspaces**—aligns perfectly with MegaContext's compression philosophy. By using LoRA-inspired techniques throughout the architecture, we achieve training efficiency, deployment flexibility, and multi-task capability while maintaining the frozen base model approach that makes MegaContext practical.
