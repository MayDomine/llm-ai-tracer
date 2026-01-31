# LLM Training Parallelism Strategies

This document provides a comprehensive overview of parallelism strategies for large-scale LLM training, including memory optimization, communication costs, and batch size calculations.

## Table of Contents

1. [Batch Size Concepts](#batch-size-concepts)
2. [Parallelism Strategies Overview](#parallelism-strategies-overview)
3. [Data Parallelism (DP)](#data-parallelism-dp)
4. [Tensor Parallelism (TP)](#tensor-parallelism-tp)
5. [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
6. [Sequence Parallelism (SP)](#sequence-parallelism-sp)
7. [Expert Parallelism (EP)](#expert-parallelism-ep)
8. [ZeRO / FSDP Memory Optimization](#zero--fsdp-memory-optimization)
9. [Activation Recomputation](#activation-recomputation)
10. [Memory and Communication Formulas](#memory-and-communication-formulas)
11. [Strategy Selection Guidelines](#strategy-selection-guidelines)

---

## Batch Size Concepts

### Terminology

| Term | Definition |
|------|------------|
| **Global Batch Size (GBS)** | Total samples processed per training step across all GPUs |
| **Micro Batch Size (MBS)** | Samples processed per GPU per forward/backward pass |
| **Gradient Accumulation (GA)** | Number of micro-batches before gradient sync |
| **Effective Batch Size** | Samples contributing to one gradient update |

### Fundamental Relationship

```
Global Batch Size = Data Parallel Size × Micro Batch Size × Gradient Accumulation

GBS = DP × MBS × GA
```

### GPU Topology

```
Total GPUs = DP × TP × PP × EP

Where:
- DP: Data Parallel degree
- TP: Tensor Parallel degree  
- PP: Pipeline Parallel degree
- EP: Expert Parallel degree (for MoE models)
```

### Example Calculation

```
Configuration:
- Total GPUs: 128
- TP = 8, PP = 4, DP = 4 (128 = 4 × 8 × 4)
- Micro Batch Size = 2
- Gradient Accumulation = 8

Global Batch Size = 4 × 2 × 8 = 64 samples per step
```

---

## Parallelism Strategies Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Parallelism Taxonomy                         │
├─────────────────┬───────────────────┬───────────────────────────┤
│   Category      │   Strategy        │   What is Partitioned     │
├─────────────────┼───────────────────┼───────────────────────────┤
│ Data            │ DP (DDP)          │ Training data (batch)     │
│ Parallelism     │ FSDP/ZeRO         │ + Model states (optional) │
├─────────────────┼───────────────────┼───────────────────────────┤
│ Model           │ TP                │ Intra-layer (horizontal)  │
│ Parallelism     │ PP                │ Inter-layer (vertical)    │
│                 │ SP                │ Sequence dimension        │
│                 │ EP                │ Expert networks (MoE)     │
├─────────────────┼───────────────────┼───────────────────────────┤
│ Memory          │ Recomputation     │ Activation memory         │
│ Optimization    │ Offloading        │ To CPU/NVMe               │
└─────────────────┴───────────────────┴───────────────────────────┘
```

---

## Data Parallelism (DP)

### Concept

Each GPU holds a complete copy of the model and processes a different portion of the batch. Gradients are synchronized via AllReduce after each step.

### Communication Pattern

```
Forward:  No communication
Backward: AllReduce gradients across DP ranks

Communication Volume = 2 × Parameters × BytesPerElement
                     = 2 × P × B  (for ring AllReduce)
                     
Time = 2 × (DP-1)/DP × P × B / Bandwidth
```

### Memory Usage

```
Per GPU Memory = Model Weights + Gradients + Optimizer States + Activations
              = P×B + P×B + 2×P×4 + Activations  (for FP16 + Adam)
              = P×B + P×B + 8×P + Activations
```

### When to Use

- Model fits in single GPU memory
- High-bandwidth interconnect (NVLink/InfiniBand)
- Large batch training

---

## Tensor Parallelism (TP)

### Concept

Splits individual layers horizontally across GPUs. Each GPU computes a portion of the tensor operations.

### Implementation for Transformers

**Attention Module:**
```
Q, K, V projections: Column-parallel (split output dimension)
Output projection: Row-parallel (split input dimension)

Communication: 2 AllReduce per layer (forward + backward each)
```

**FFN Module:**
```
Up/Gate projection: Column-parallel
Down projection: Row-parallel

Communication: 2 AllReduce per layer
```

### Communication Volume

```
Per Layer (Forward):
- Attention: AllReduce of [B × S × d_model]
- FFN: AllReduce of [B × S × d_model]
Total = 4 × B × S × d × BytesPerElement × NumLayers

Per Layer (Backward):
- Same as forward (2 more AllReduce per layer)
Total = 8 × B × S × d × BytesPerElement × NumLayers
```

### Memory Savings

```
Weight Memory / GPU = TotalWeights / TP
- Q, K, V, O projections: divided by TP
- FFN gate, up, down: divided by TP
- Embedding: NOT divided (replicated)

Activation Memory:
- Attention heads: divided by TP (each GPU handles H/TP heads)
- FFN intermediate: divided by TP
```

### Constraints

```
- TP must divide num_attention_heads evenly
- TP must divide num_kv_heads evenly (for GQA)
- TP should be power of 2 for efficiency
- TP <= 8 typically (NVLink domain)
```

---

## Pipeline Parallelism (PP)

### Concept

Splits model vertically by layers. Each GPU (stage) holds a subset of consecutive layers.

### Schedules

**GPipe / All-Forward-All-Backward:**
```
Stage 0: [F0][F1][F2][F3]...[F_m][B_m]...[B3][B2][B1][B0]
Stage 1:    [F0][F1][F2]...[F_m-1][B_m-1]...[B2][B1][B0]
...

Bubble ratio = (PP - 1) / (PP - 1 + m) ≈ (PP-1) / m
```

**1F1B (One-Forward-One-Backward):**
```
Stage 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2]...
Stage 1:    [F0][F1][F2][B0][F3][B1][F4][B2]...

Bubble ratio = (PP - 1) / m  (same but better memory)
Memory = PP × LayersPerStage activations (not m×)
```

**Interleaved 1F1B:**
```
Each stage holds multiple non-consecutive layer chunks.
Reduces bubble to (PP - 1) / (v × m) where v = virtual stages
```

### Communication Pattern

```
Point-to-Point (P2P):
- Forward: Send activations to next stage
- Backward: Send gradients to previous stage

Volume per micro-batch = B × S × d_model × BytesPerElement × 2
                       (forward activations + backward gradients)
```

### Bubble Overhead

```
Bubble Time = (PP - 1) × time_per_microbatch
Useful Compute = m × PP × time_per_microbatch

Efficiency = m × PP / (m × PP + PP - 1)
           = m / (m + (PP-1)/PP)
           ≈ 1 - (PP-1)/(m×PP)  for large m
```

### Memory Savings

```
Weights / GPU = TotalWeights / PP
Activations = Activations per micro-batch × num_in_flight
            (1F1B: num_in_flight = PP, not m)
```

### Constraints

```
- PP must divide num_layers evenly
- Requires sufficient micro-batches: m >= 4×PP recommended
- Latency increases with PP (more stages = more bubbles)
```

---

## Sequence Parallelism (SP)

### Concept

Partitions the sequence dimension across GPUs. Works in conjunction with TP to reduce activation memory.

### How It Works

```
Without SP: LayerNorm and Dropout activations are replicated across TP ranks
With SP:    These activations are partitioned along sequence dimension

Activation Memory Reduction = TP × (for non-TP operations)
```

### Communication Pattern

```
- After LayerNorm: AllGather to reconstruct full sequence for attention
- After Dropout: ReduceScatter to partition sequence

Volume = 2 × B × S × d × BytesPerElement × NumLayers
       (AllGather + ReduceScatter = AllReduce equivalent)
```

### Memory Savings

```
Activation Memory with SP ≈ Activation Memory without SP / TP

Combined with selective recomputation:
- 5× activation memory reduction
- 90% reduction in recomputation overhead
```

### When to Use

- Always enable SP when using TP
- Essential for long sequences (8K+ tokens)
- Critical for large hidden dimensions

---

## Expert Parallelism (EP)

### Concept

For Mixture of Experts (MoE) models, distributes different expert networks across GPUs.

### Communication Pattern

```
All-to-All for token routing:
1. Each GPU sends tokens to the experts they're routed to
2. Each GPU receives tokens from all other GPUs for its experts
3. Expert computation
4. All-to-All to return processed tokens

Volume = 2 × B × S × d × top_k × BytesPerElement × NumMoELayers
       (dispatch + combine)
```

### GPU Assignment

```
With EP degree = E:
- Each GPU holds num_experts / E experts
- Tokens are shuffled via All-to-All based on routing

Total GPUs = DP × TP × PP × EP
```

### Communication Overhead

```
All-to-All time ≈ 15-60% of total MoE training time

Optimizations:
- Token locality optimization (minimize cross-GPU routing)
- Overlapping All-to-All with computation
- Hierarchical All-to-All (intra-node then inter-node)
```

---

## ZeRO / FSDP Memory Optimization

### ZeRO Stages

| Stage | Partitions | Memory Per GPU | Communication |
|-------|-----------|----------------|---------------|
| **Stage 0** | None (DDP) | P×B + P×B + 8P + Act | 2P (AllReduce grad) |
| **Stage 1** | Optimizer States | P×B + P×B + 8P/DP + Act | 2P (AllReduce grad) |
| **Stage 2** | + Gradients | P×B + P×B/DP + 8P/DP + Act | 2P (ReduceScatter + AllGather grad) |
| **Stage 3** | + Parameters | P×B/DP + P×B/DP + 8P/DP + Act | 3P (+ AllGather params each step) |

### Memory Formulas

```
FP16 Training with Adam Optimizer:

DDP (Stage 0):
  Memory = 2P + 2P + 4P + 4P + Activations
         = 12P + Activations
  (weights FP16 + grads FP16 + master weights FP32 + m FP32 + v FP32)

ZeRO-1:
  Memory = 2P + 2P + 8P/DP + Activations
         = 4P + 8P/DP + Activations

ZeRO-2:
  Memory = 2P + (2P + 8P)/DP + Activations
         = 2P + 10P/DP + Activations

ZeRO-3:
  Memory = (2P + 2P + 8P)/DP + Activations
         = 12P/DP + Activations
```

### Communication Formulas

```
ZeRO-1: Same as DDP
  Volume = 2P (AllReduce gradients)

ZeRO-2: 
  Volume = 2P (ReduceScatter + AllGather gradients)

ZeRO-3:
  Volume = 3P per step
  - 1P: AllGather parameters before forward
  - 1P: AllGather parameters before backward
  - 1P: ReduceScatter gradients after backward
```

### FSDP (PyTorch)

FSDP is PyTorch's implementation of ZeRO-3 with additional features:
- Full sharding of parameters, gradients, and optimizer states
- Automatic parameter gathering/sharding
- Mixed precision support
- CPU offloading option

---

## Activation Recomputation

### Concept

Trade compute for memory by discarding activations during forward pass and recomputing them during backward pass.

### Strategies

**Full Recomputation:**
```
- Checkpoint input to each transformer layer
- Recompute entire layer during backward
- Compute overhead: ~33% per layer
- Memory: O(num_layers) → O(sqrt(num_layers)) or O(1)
```

**Selective Recomputation:**
```
- Keep "cheap to store" activations: QKV outputs, attention output
- Recompute "expensive to store": attention scores, softmax output

Memory savings: ~5× reduction
Compute overhead: reduced from 33% to ~5%
```

**Block Recomputation:**
```
- Checkpoint every N layers instead of every layer
- Tradeoff between memory and compute
```

### Memory Impact

```
Without recomputation:
  Activation Memory = B × S × d × factor × NumLayers
  factor ≈ 12-16 (depending on architecture)

With full recomputation:
  Activation Memory = B × S × d × factor × 1
  (only need to store 1 layer's activations at a time)

With selective recomputation + SP + TP:
  Activation Memory = B × S × d × factor / TP × 1
```

---

## Memory and Communication Formulas

### Complete Memory Model

```
Total GPU Memory = Weights + Gradients + Optimizer + Activations + KV Cache + Overhead

Weights:
  = P × BytesPerElement / (TP × PP × ZeRO_param_shard)

Gradients:
  = P × BytesPerElement / (TP × PP × ZeRO_grad_shard)

Optimizer (Adam FP32):
  = 8 × P / (TP × PP × DP)  [with ZeRO-1+]

Activations per layer:
  = B_micro × S × d × bytes × factor / (TP if SP else 1) × (1 if recomp else layers_stored)

  factor includes:
  - Input: 1
  - Q, K, V: 3 × (d/TP)
  - Attention scores: H/TP × S × S (if not recomputed)
  - Attention output: 1
  - FFN intermediate: intermediate_size/TP × 2 (if gated)
  - Residuals: 2
```

### Complete Communication Model

```
Per Training Step:

Data Parallel (gradient sync):
  Volume = 2 × P × B / (TP × PP)  [AllReduce]
  Time = 2 × (DP-1)/DP × Volume / Bandwidth_DP

Tensor Parallel (per layer):
  Volume = 4 × B_micro × S × d × B × NumLayers  [4 AllReduce: 2 fwd + 2 bwd]
  Time = 4 × (TP-1)/TP × B_micro × S × d × B × NumLayers / Bandwidth_TP

Pipeline Parallel:
  Volume = 2 × B_micro × S × d × B × (PP-1) × num_microbatches  [P2P]
  Bubble = (PP-1) / num_microbatches × compute_time

Expert Parallel (MoE):
  Volume = 2 × B_micro × S × d × B × top_k × NumMoELayers  [All-to-All]
  Time = Volume / Bandwidth_EP

ZeRO-3 (parameter gather):
  Volume = 2 × P × B / (TP × PP)  [AllGather × 2: fwd + bwd]
  Time = 2 × (DP-1)/DP × Volume / Bandwidth_DP
```

---

## Strategy Selection Guidelines

### Decision Tree

```
1. Can model fit in single GPU?
   YES → Use DDP (DP only)
   NO → Continue

2. How big is the model?
   < 10B params → Try ZeRO-2/3 or TP=2-4
   10B-100B → TP + PP + ZeRO-1
   > 100B → TP + PP + ZeRO + EP (if MoE)

3. What's your sequence length?
   < 4K → Standard approach
   4K-32K → Enable SP, consider selective recomputation
   > 32K → Mandatory SP, ring attention, aggressive recomputation

4. What's your interconnect?
   NVLink → Prefer TP within node
   InfiniBand → Prefer PP across nodes
   Ethernet → Minimize cross-node communication
```

### Recommended Configurations

**Small Model (1-7B), 8 GPUs:**
```
TP=1, PP=1, DP=8, ZeRO-2
Or: TP=2, PP=1, DP=4 (if memory constrained)
```

**Medium Model (7-30B), 32 GPUs:**
```
TP=4, PP=2, DP=4, ZeRO-1
Enable SP with TP
```

**Large Model (30-100B), 128 GPUs:**
```
TP=8, PP=4, DP=4, ZeRO-1
SP enabled, selective recomputation
```

**Very Large Model (100B+), 512+ GPUs:**
```
TP=8, PP=8-16, DP=4-8, ZeRO-1
Full recomputation, SP enabled
```

**MoE Model with Experts:**
```
Add EP dimension: total_gpus = DP × TP × PP × EP
Route experts efficiently within EP domain
```

### Performance Tuning

```
1. Maximize MBS while fitting in memory
2. Increase GA to reach target GBS
3. Ensure num_microbatches >= 4 × PP for pipeline efficiency
4. Place TP within NVLink domain (single node)
5. Place DP across nodes (gradient sync is less frequent)
6. Enable SP whenever using TP
7. Use selective recomputation before full recomputation
```

---

## References

### Core Papers

1. **Megatron-LM**
   - Title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
   - Authors: Shoeybi, Patwary, Puri, et al. (NVIDIA)
   - Year: 2019
   - arXiv: [1909.08053](https://arxiv.org/abs/1909.08053)
   - Key contributions: Tensor Parallelism, Pipeline Parallelism

2. **ZeRO**
   - Title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
   - Authors: Rajbhandari, Rasley, Ruwase, He (Microsoft)
   - Year: 2020
   - Venue: SC20
   - arXiv: [1910.02054](https://arxiv.org/abs/1910.02054)
   - Key formulas: ZeRO-1/2/3 memory formulas

3. **Reducing Activation Recomputation**
   - Title: "Reducing Activation Recomputation in Large Transformer Models"
   - Authors: Korthikanti, Casper, Lym, et al. (NVIDIA)
   - Year: 2023
   - Venue: MLSys 2023
   - arXiv: [2205.05198](https://arxiv.org/abs/2205.05198)
   - Key contributions: Selective recomputation, Sequence Parallelism

4. **GPipe**
   - Title: "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"
   - Authors: Huang, Cheng, Bapna, et al. (Google)
   - Year: 2019
   - Venue: NeurIPS 2019
   - arXiv: [1811.06965](https://arxiv.org/abs/1811.06965)
   - Key formula: Bubble ratio = (PP-1)/num_microbatches

5. **DeepSpeed Ulysses**
   - Title: "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"
   - Authors: Jacobs, et al. (Microsoft)
   - Year: 2023
   - arXiv: [2309.14509](https://arxiv.org/abs/2309.14509)
   - Key contributions: Context Parallelism with all-to-all

6. **Ring Attention**
   - Title: "Ring Attention with Blockwise Transformers for Near-Infinite Context"
   - Authors: Liu, Zaharia, Abbeel
   - Year: 2023
   - arXiv: [2310.01889](https://arxiv.org/abs/2310.01889)
   - Key contributions: Ring-based context parallelism

7. **AMSP**
   - Title: "AMSP: Reducing Communication Overhead of ZeRO for Efficient LLM Training"
   - Authors: Various
   - Year: 2024
   - Key contributions: Optimized ZeRO communication

### Implementation References

8. **FSDP Documentation**
   - PyTorch Fully Sharded Data Parallel
   - URL: https://pytorch.org/docs/stable/fsdp.html
   - ZeRO-3 style sharding in PyTorch

9. **TorchTitan**
   - Title: "TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training"
   - Authors: Meta/PyTorch team
   - Year: 2024
   - URL: https://github.com/pytorch/torchtitan
   - Key contributions: 3D parallelism reference implementation

10. **NVIDIA NCCL**
    - NVIDIA Collective Communications Library
    - URL: https://docs.nvidia.com/deeplearning/nccl/
    - Key formulas: Ring AllReduce, AllGather, ReduceScatter
