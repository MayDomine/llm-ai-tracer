/**
 * Unified Activation Memory Calculator
 * 
 * This module provides a single source of truth for activation memory calculations,
 * used by both memoryCalculator.ts (inference) and trainingCalculator.ts (training).
 * 
 * Citation: "Reducing Activation Recomputation in Large Transformer Models"
 * (Korthikanti et al., MLSys 2023) - arXiv:2205.05198
 * 
 * Key insight: FlashAttention eliminates O(S²) attention score storage,
 * reducing activation memory from ~34× hidden to ~10-12× hidden per layer.
 */

import type { ModelConfig } from '../types/model';
import type { RecomputationStrategy } from '../types/training';

// ============ Configurable Parameters ============

export interface ActivationCalculatorConfig {
  // Whether FlashAttention is used (eliminates O(S²) attention score storage)
  useFlashAttention: boolean;
  
  // Recomputation strategy
  recomputation: RecomputationStrategy;
  
  // Tensor parallel degree (activations sharded across TP ranks)
  tensorParallel: number;
  
  // Sequence parallel (activations sharded along sequence dimension with TP)
  sequenceParallel: boolean;
  
  // Context parallel degree (sequence split across CP ranks)
  contextParallel: number;
  
  // Pipeline parallel degree (affects stored layers)
  pipelineParallel: number;
  
  // Precision in bytes
  bytesPerElement: number;
}

export const DEFAULT_ACTIVATION_CONFIG: ActivationCalculatorConfig = {
  useFlashAttention: true,
  recomputation: 'none',
  tensorParallel: 1,
  sequenceParallel: false,
  contextParallel: 1,
  pipelineParallel: 1,
  bytesPerElement: 2,  // FP16/BF16
};

// ============ Per-Tensor Breakdown ============

export interface ActivationTensor {
  name: string;
  shape: string;                  // Human-readable shape like "[B, S, d]"
  shapeValues: number[];          // Actual numeric values
  elements: number;               // Total element count
  bytesTotal: number;             // Total memory (before sharding)
  bytesPerGPU: number;            // Memory per GPU (after sharding)
  isTpSharded: boolean;           // Sharded by tensor parallel
  isSpSharded: boolean;           // Sharded by sequence parallel
  isStored: boolean;              // Whether stored (vs recomputed)
  description: string;
}

export interface PerLayerActivationBreakdown {
  tensors: ActivationTensor[];
  totalBytesPerLayer: number;
  totalBytesPerGPU: number;
  formula: string;
}

export interface TotalActivationBreakdown {
  perLayerBreakdown: PerLayerActivationBreakdown;
  
  // Number of layers stored (affected by PP and recomputation)
  layersStored: number;
  layersPerGPU: number;
  
  // LM Head logits (always stored, cannot recompute)
  lmHeadLogitsBytes: number;
  
  // Total activation memory per GPU
  totalBytesPerGPU: number;
  
  // Detailed formula
  formula: string;
}

// ============ Core Calculation Functions ============

/**
 * Calculate detailed per-layer activation memory
 * 
 * This is the authoritative activation calculation used by both
 * training and inference memory estimation.
 */
export function calculatePerLayerActivations(
  model: ModelConfig,
  batchSize: number,
  seqLength: number,
  config: ActivationCalculatorConfig = DEFAULT_ACTIVATION_CONFIG
): PerLayerActivationBreakdown {
  const B = batchSize;
  // With Context Parallel, each GPU only handles S/CP tokens
  const S = Math.ceil(seqLength / config.contextParallel);
  const d = model.hiddenSize;
  const H = model.numAttentionHeads;
  const H_kv = model.numKVHeads;
  const d_h = model.headDim;
  const d_ff = model.intermediateSize;
  const bytes = config.bytesPerElement;
  const TP = config.tensorParallel;
  const useRecompute = config.recomputation === 'full' || config.recomputation === 'block';
  const useSelective = config.recomputation === 'selective';
  
  const tensors: ActivationTensor[] = [];
  
  // Helper to add a tensor with proper sharding
  const addTensor = (
    name: string,
    shape: string,
    shapeValues: number[],
    isTpSharded: boolean = false,
    isSpSharded: boolean = false,
    isStored: boolean = true,
    description: string = ''
  ) => {
    const elements = shapeValues.reduce((a, b) => a * b, 1);
    const bytesTotal = elements * bytes;
    
    let bytesPerGPU = bytesTotal;
    if (isTpSharded && TP > 1) {
      bytesPerGPU = bytesTotal / TP;
    }
    if (isSpSharded && config.sequenceParallel && TP > 1) {
      bytesPerGPU = bytesPerGPU / TP;
    }
    
    tensors.push({
      name,
      shape,
      shapeValues,
      elements,
      bytesTotal,
      bytesPerGPU,
      isTpSharded,
      isSpSharded,
      isStored,
      description: description || name,
    });
  };
  
  // ============ Attention Module ============
  
  // Layer input (always stored for residual + backward)
  addTensor(
    'Layer Input',
    '[B, S, d]',
    [B, S, d],
    false, config.sequenceParallel, true,
    'Input to transformer layer (stored for residual connection)'
  );
  
  // Pre-attention LayerNorm output
  if (!useRecompute) {
    addTensor(
      'Pre-Attn LN Output',
      '[B, S, d]',
      [B, S, d],
      false, config.sequenceParallel, true,
      'LayerNorm output before attention'
    );
  }
  
  // Q, K, V projections (TP sharded across heads)
  if (!useRecompute) {
    addTensor(
      'Q Projection',
      '[B, S, H, d_h]',
      [B, S, H, d_h],
      true, false, true,
      `Q projection: ${H} heads × ${d_h} head dim`
    );
    
    addTensor(
      'K Projection',
      '[B, S, H_kv, d_h]',
      [B, S, H_kv, d_h],
      true, false, true,
      `K projection: ${H_kv} KV heads × ${d_h} head dim`
    );
    
    addTensor(
      'V Projection',
      '[B, S, H_kv, d_h]',
      [B, S, H_kv, d_h],
      true, false, true,
      `V projection: ${H_kv} KV heads × ${d_h} head dim`
    );
  }
  
  // Attention Scores: [B, H, S, S]
  // NOT stored with FlashAttention - this is the key memory saving!
  if (!config.useFlashAttention && !useRecompute && !useSelective) {
    addTensor(
      'Attention Scores (QK^T)',
      '[B, H, S, S]',
      [B, H, S, S],
      true, false, true,
      `⚠️ O(S²) attention scores - use FlashAttention to avoid!`
    );
    
    addTensor(
      'Softmax Output',
      '[B, H, S, S]',
      [B, H, S, S],
      true, false, true,
      'Softmax attention probabilities'
    );
  }
  
  // Attention output
  if (!useRecompute) {
    addTensor(
      'Attention Output',
      '[B, S, d]',
      [B, S, d],
      true, false, true,
      'Output of softmax × V'
    );
    
    addTensor(
      'Output Projection',
      '[B, S, d]',
      [B, S, d],
      false, config.sequenceParallel, true,
      'Output of O projection'
    );
  }
  
  // ============ FFN Module ============
  
  if (!useRecompute) {
    addTensor(
      'Pre-FFN LN Output',
      '[B, S, d]',
      [B, S, d],
      false, config.sequenceParallel, true,
      'LayerNorm output before FFN'
    );
  }
  
  // Gated FFN (SwiGLU) - gate and up projections
  if (model.ffnType === 'gated' && !useRecompute) {
    addTensor(
      'Gate Projection',
      '[B, S, d_ff]',
      [B, S, d_ff],
      true, false, true,
      `Gate projection to ${d_ff} intermediate`
    );
  }
  
  if (!useRecompute) {
    addTensor(
      'Up Projection',
      '[B, S, d_ff]',
      [B, S, d_ff],
      true, false, true,
      `Up projection to ${d_ff} intermediate`
    );
    
    addTensor(
      'FFN Activation',
      '[B, S, d_ff]',
      [B, S, d_ff],
      true, false, true,
      'SiLU/GELU activation output'
    );
    
    addTensor(
      'Down Projection',
      '[B, S, d]',
      [B, S, d],
      false, config.sequenceParallel, true,
      'Down projection to hidden size'
    );
  }
  
  // Calculate totals
  const storedTensors = tensors.filter(t => t.isStored);
  const totalBytesPerLayer = storedTensors.reduce((sum, t) => sum + t.bytesTotal, 0);
  const totalBytesPerGPU = storedTensors.reduce((sum, t) => sum + t.bytesPerGPU, 0);
  
  // Generate formula description
  let formulaParts: string[] = [];
  if (config.useFlashAttention) {
    formulaParts.push('FlashAttention enabled (no O(S²) storage)');
  }
  if (config.recomputation !== 'none') {
    formulaParts.push(`Recomputation: ${config.recomputation}`);
  }
  if (config.sequenceParallel) {
    formulaParts.push(`Sequence Parallel with TP=${TP}`);
  }
  if (config.contextParallel > 1) {
    formulaParts.push(`Context Parallel: S/${config.contextParallel} = ${S} tokens/GPU`);
  }
  
  const formula = formulaParts.length > 0 ? formulaParts.join(', ') : 'Standard activation storage';
  
  return {
    tensors,
    totalBytesPerLayer,
    totalBytesPerGPU,
    formula,
  };
}

/**
 * Calculate LM Head logits memory
 * 
 * Shape: [B, S, vocab_size]
 * This CANNOT be recomputed - needed for cross-entropy loss backward pass
 */
export function calculateLMHeadLogitsMemory(
  model: ModelConfig,
  batchSize: number,
  seqLength: number,
  config: ActivationCalculatorConfig = DEFAULT_ACTIVATION_CONFIG
): { bytes: number; bytesPerGPU: number; formula: string } {
  const B = batchSize;
  const S = Math.ceil(seqLength / config.contextParallel);
  const V = model.vocabSize;
  const bytes = config.bytesPerElement;
  
  const totalBytes = B * S * V * bytes;
  
  // With TP, vocab is sharded
  const bytesPerGPU = config.tensorParallel > 1 ? totalBytes / config.tensorParallel : totalBytes;
  
  return {
    bytes: totalBytes,
    bytesPerGPU,
    formula: `LM Head Logits: ${B} × ${S} × ${V} × ${bytes} bytes = ${(totalBytes / 1e9).toFixed(2)} GB (CANNOT recompute)`,
  };
}

/**
 * Calculate total activation memory for the model
 */
export function calculateTotalActivationMemory(
  model: ModelConfig,
  batchSize: number,
  seqLength: number,
  config: ActivationCalculatorConfig = DEFAULT_ACTIVATION_CONFIG
): TotalActivationBreakdown {
  const perLayerBreakdown = calculatePerLayerActivations(model, batchSize, seqLength, config);
  const lmHeadLogits = calculateLMHeadLogitsMemory(model, batchSize, seqLength, config);
  
  // Calculate layers stored based on recomputation strategy and PP
  const layersPerGPU = Math.ceil(model.numLayers / config.pipelineParallel);
  let layersStored: number;
  
  switch (config.recomputation) {
    case 'full':
      layersStored = 1; // Only store model input, recompute all layers
      break;
    case 'block':
      layersStored = layersPerGPU; // Store layer inputs, recompute internals
      break;
    default:
      layersStored = layersPerGPU; // Store all activations
  }
  
  const totalBytesPerGPU = perLayerBreakdown.totalBytesPerGPU * layersStored + lmHeadLogits.bytesPerGPU;
  
  const formula = `${(perLayerBreakdown.totalBytesPerGPU / 1e6).toFixed(1)} MB/layer × ${layersStored} layers + ${(lmHeadLogits.bytesPerGPU / 1e9).toFixed(2)} GB (LM Head) = ${(totalBytesPerGPU / 1e9).toFixed(2)} GB`;
  
  return {
    perLayerBreakdown,
    layersStored,
    layersPerGPU,
    lmHeadLogitsBytes: lmHeadLogits.bytesPerGPU,
    totalBytesPerGPU,
    formula,
  };
}

// ============ Activation Factor Calculation ============

/**
 * Calculate the activation factor (multiplier of B × S × d × bytes)
 * 
 * This provides the "activation factor" used in simplified calculations.
 * The factor varies based on:
 * - FlashAttention: ~10-12 (no S² term)
 * - Without FlashAttention: ~34+ (includes B × H × S × S)
 * - With Sequence Parallel: divided by TP
 */
export function calculateActivationFactor(
  model: ModelConfig,
  seqLength: number,
  useFlashAttention: boolean = true,
  sequenceParallel: boolean = false,
  tensorParallel: number = 1
): { factor: number; breakdown: string } {
  const d = model.hiddenSize;
  const H = model.numAttentionHeads;
  const H_kv = model.numKVHeads;
  const d_h = model.headDim;
  const d_ff = model.intermediateSize;
  const S = seqLength;
  
  // Base factor for d-sized tensors
  // Layer input: 1
  // Pre-attn LN: 1
  // Q: H × d_h / d = 1 (since H × d_h = d)
  // K: H_kv × d_h / d
  // V: H_kv × d_h / d
  // Attn output: 1
  // O proj: 1
  // Pre-FFN LN: 1
  // Gate: d_ff / d
  // Up: d_ff / d
  // Activation: d_ff / d
  // Down: 1
  // Total: 6 + (H_kv × d_h / d) × 2 + (d_ff / d) × 3
  
  const kvRatio = (H_kv * d_h) / d;
  const ffnRatio = d_ff / d;
  
  let factor = 6 + 2 * kvRatio + 3 * ffnRatio;
  
  // Add attention scores if not using FlashAttention
  if (!useFlashAttention) {
    // Attention scores: [B, H, S, S] = H × S² / d tensors equivalent
    // Two of these: QK^T and softmax output
    const attnScoreFactor = 2 * H * S / d;
    factor += attnScoreFactor;
  }
  
  // Sequence parallel divides non-TP tensors by TP
  if (sequenceParallel && tensorParallel > 1) {
    // Roughly half the tensors are SP-sharded (the d-sized ones)
    factor = factor * 0.5 + factor * 0.5 / tensorParallel;
  }
  
  const breakdown = [
    `Base: 6 (layer input, LN outputs, Q, attn out, O proj, down)`,
    `KV: 2 × ${kvRatio.toFixed(2)} = ${(2 * kvRatio).toFixed(2)}`,
    `FFN: 3 × ${ffnRatio.toFixed(2)} = ${(3 * ffnRatio).toFixed(2)}`,
    useFlashAttention ? 'FlashAttention: no S² term' : `Attention scores: +${(2 * H * S / d).toFixed(2)}`,
    sequenceParallel ? `SP reduction: /${tensorParallel}` : '',
  ].filter(Boolean).join(', ');
  
  return { factor, breakdown };
}
