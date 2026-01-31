# TFLOPS Calculation Methodology

This document details the FLOPs (Floating-Point Operations) calculation methodology used in LLM AI Tracer, with comparisons to industry-standard implementations from HuggingFace, vLLM, and academic references.

## References

- **OpenAI Scaling Laws** (Kaplan et al., 2020): `C_forward+backward ≈ 6N`
- **Chinchilla Scaling Laws** (DeepMind, Hoffmann et al., 2022): Detailed per-operation breakdown
- **PaLM** (Google, Chowdhery et al., 2022): Model FLOPs Utilization (MFU) definition
- **HuggingFace calflops**: https://github.com/MrYxJ/calculate-flops.pytorch
- **Transformer FLOPs** (Adam Casson): https://www.adamcasson.com/posts/transformer-flops

---

## 1. High-Level FLOPs Estimation

### OpenAI Method (per token, forward pass)

```
C_forward ≈ 2N + 2 × n_layers × n_ctx × d_attn
         ≈ 2N  (when d_model > n_ctx/12)
```

Where:
- `N` = number of non-embedding parameters
- `n_layers` = number of transformer layers
- `n_ctx` = context/sequence length
- `d_attn` = attention output dimension (typically = d_model)

### Training FLOPs

```
C_forward+backward ≈ 6N per token
C_with_activation_checkpointing ≈ 8N per token
```

The factor breakdown:
- **Forward pass**: 2N (multiply-accumulate = 2 ops per parameter)
- **Backward pass**: 4N (2× matmuls for gradient computation)
- **Activation checkpointing**: +2N (recompute forward during backward)

---

## 2. Detailed Per-Operation FLOPs

### 2.1 GEMM (General Matrix Multiplication)

```
FLOPs = 2 × M × N × K
```

Where the matrices are [M × K] × [K × N] = [M × N].

The factor of 2 comes from multiply-accumulate (1 multiply + 1 add per element).

### 2.2 Attention Module

For batch size `B`, sequence length `S`, hidden size `d`, number of heads `H`, head dimension `d_h`:

| Operation | FLOPs | Our Implementation |
|-----------|-------|-------------------|
| Q Projection | 2 × B × S × d × d | ✅ `analyzeGEMM` |
| K Projection | 2 × B × S × d × (h_kv × d_h) | ✅ GQA-aware |
| V Projection | 2 × B × S × d × (h_kv × d_h) | ✅ GQA-aware |
| QK^T | 2 × B × H × S × S × d_h | ✅ |
| Softmax | 5 × B × H × S × S | ✅ (max + exp + sum + div + sub) |
| Attention × V | 2 × B × H × S × d_h × S | ✅ |
| O Projection | 2 × B × S × d × d | ✅ |

**Note on GQA (Grouped Query Attention):**
- Q projection uses all `H` heads
- K/V projections use `h_kv` heads (fewer than Q)
- Our implementation correctly accounts for this difference

### 2.3 Feedforward Network (FFN)

#### GPT-style (2 linear layers)
```
FLOPs = 2 × tokens × d × d_ff + 2 × tokens × d_ff × d
      = 4 × tokens × d × d_ff
```

#### LLaMA-style (Gated, 3 linear layers)
```
FLOPs = 2 × tokens × d × d_ff (gate)
      + 2 × tokens × d × d_ff (up)  
      + 2 × tokens × d_ff × d (down)
      = 6 × tokens × d × d_ff
```

Plus activation function (SiLU ≈ 4 ops per element) and element-wise multiply.

#### MoE (Mixture of Experts)
```
FLOPs = Router: 2 × tokens × d × n_experts
      + Active experts: k × (gate + up + down per expert)
```

Where `k` = top-K activated experts per token.

### 2.4 Other Operations

| Operation | FLOPs Estimate | Notes |
|-----------|----------------|-------|
| Embedding | 0 | Table lookup only |
| LayerNorm/RMSNorm | 5 × elements | mean, var, normalize |
| GELU | 8 × elements | Approximation ops |
| SiLU | 4 × elements | sigmoid × x |
| Softmax | 5 × elements | max, exp, sum, div, sub |

---

## 3. Comparison with Industry Implementations

### 3.1 HuggingFace calflops

Our implementation aligns with HuggingFace calflops:
- ✅ GEMM FLOPs: `2 × M × N × K`
- ✅ Backward multiplier: 2× forward (or 3× with activation checkpointing)
- ✅ Per-operation breakdown (attention, FFN, embedding)

### 3.2 DeepMind Chinchilla Method

The DeepMind method (per sequence) includes more detailed attention components:

| Component | DeepMind Formula | Our Implementation |
|-----------|-----------------|-------------------|
| QK logits | 2 × n_ctx × n_ctx × d_key × n_heads | ✅ QK^T GEMM |
| Softmax | 3 × n_heads × n_ctx × n_ctx | ✅ (we use 5 ops) |
| Reduction | 2 × n_ctx × n_ctx × d_key × n_heads | ✅ Attn × V GEMM |

**Difference:** DeepMind uses 3 ops for softmax, we use 5 (more accurate for max-subtraction numerical stability).

### 3.3 vLLM

vLLM follows the OpenAI scaling law approach:
- Uses `6N` approximation for training
- Our detailed breakdown provides more accurate per-operation analysis

---

## 4. Training vs Inference FLOPs

### Forward Pass (Inference)

```
C_forward = Σ_layers (Attention + FFN) + Embedding + LM_Head
```

### Training (Forward + Backward)

```
C_training = 3 × C_forward (standard)
           = 4 × C_forward (with activation checkpointing)
```

### Decode vs Prefill

| Mode | Attention FLOPs | Notes |
|------|----------------|-------|
| Prefill | O(S²) | Full attention matrix |
| Decode | O(S) | Single query against KV cache |

Our implementation correctly handles both modes with different sequence length calculations:
- `querySeqLen`: 1 for decode, S for prefill
- `kvSeqLen`: cached_len + 1 for decode, S for prefill

---

## 5. MFU (Model FLOPs Utilization)

### Definition (PaLM Paper)

```
MFU = (C × D) / P
```

Where:
- `C` = Model FLOPs per token (6N for training)
- `D` = Observed tokens per second
- `P` = Theoretical peak FLOPS

### Our Implementation

```typescript
// Global batch FLOPs
const totalUsefulFlops = 6 * params.total * globalBatchSize * seqLength;

// Per-step time
const stepTime = forwardTime + backwardTime + recomputationTime + communicationTime;

// MFU
const mfu = (totalUsefulFlops / stepTime) / (numGPUs * peakTFLOPS);
```

### Realistic MFU Ranges

Based on industry benchmarks:
- **10-30%**: Typical for memory-bound or communication-heavy setups
- **30-50%**: Well-optimized single-node training
- **40-65%**: Highly optimized large-scale training (PaLM, GPT-4 class)

---

## 6. Roofline Model Integration

Our implementation uses the roofline model to determine compute vs memory bound:

```
Ridge Point = Peak FLOPS / Peak Bandwidth
Arithmetic Intensity = FLOPs / Memory Bytes

If AI < Ridge Point: Memory Bound
If AI > Ridge Point: Compute Bound
```

### Per-Operation Analysis

| Operation | Typical AI | Bound |
|-----------|-----------|-------|
| GEMM (large) | 100-1000 | Compute |
| GEMM (small batch) | 1-10 | Memory |
| Softmax | 5 | Memory |
| LayerNorm | 5 | Memory |
| Embedding | 0 | Memory |

---

## 7. Validation

To validate our calculations:

1. **Parameter Count**: Compare with `model.num_parameters()` from HuggingFace
2. **FLOPs per Token**: Should match `calflops` library within 5%
3. **MFU**: Should be in 10-65% range for realistic workloads

### Test Case: Qwen3-8B

```
Parameters: ~8.3B
FLOPs per token (forward): ~16.6 TFLOPs
FLOPs per token (training): ~49.8 TFLOPs
```

---

## 8. Known Limitations

1. **Softmax Approximation**: We use 5 ops, DeepMind uses 3 - difference is negligible
2. **Flash Attention**: Reduces memory but not compute FLOPs
3. **Quantization**: INT8/INT4 have different FLOP characteristics (ops vs equivalent fp16 ops)
4. **Communication**: Not counted in model FLOPs, affects MFU via time

---

## References

1. Kaplan et al. "Scaling Laws for Neural Language Models" (2020)
2. Hoffmann et al. "Training Compute-Optimal Large Language Models" (2022)
3. Chowdhery et al. "PaLM: Scaling Language Modeling with Pathways" (2022)
4. Adam Casson. "Transformer FLOPs" (2023)
5. HuggingFace calflops: https://github.com/MrYxJ/calculate-flops.pytorch
