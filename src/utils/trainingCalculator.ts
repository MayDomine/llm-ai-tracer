/**
 * Training Calculator
 * 
 * Calculates memory, communication, and compute requirements for distributed LLM training.
 * Implements formulas from Megatron-LM, ZeRO, and related research.
 */

import type { ModelConfig } from '../types/model';
import type {
  BatchConfig,
  ParallelismConfig,
  TrainingConfig,
  ClusterTopology,
  TrainingMemoryBreakdown,
  CommunicationBreakdown,
  TrainingStepAnalysis,
  RecomputationStrategy,
} from '../types/training';
import { calculateModelParameters } from './calculator';

// ============ Batch Size Calculations ============

/**
 * Calculate effective batch configuration from global batch size
 */
export function calculateBatchConfig(
  globalBatchSize: number,
  parallelism: ParallelismConfig,
  microBatchSize: number
): BatchConfig {
  const { dataParallel } = parallelism;
  
  // Global Batch = DP × MBS × GA
  // GA = GBS / (DP × MBS)
  const gradientAccumulation = Math.ceil(globalBatchSize / (dataParallel * microBatchSize));
  
  // Adjust micro batch if needed
  const actualMBS = Math.min(microBatchSize, Math.ceil(globalBatchSize / dataParallel));
  
  return {
    globalBatchSize,
    microBatchSize: actualMBS,
    gradientAccumulation,
  };
}

/**
 * Validate batch configuration
 */
export function validateBatchConfig(
  batch: BatchConfig,
  parallelism: ParallelismConfig
): string[] {
  const errors: string[] = [];
  
  const expectedGBS = parallelism.dataParallel * batch.microBatchSize * batch.gradientAccumulation;
  if (expectedGBS !== batch.globalBatchSize) {
    errors.push(`GBS mismatch: DP(${parallelism.dataParallel}) × MBS(${batch.microBatchSize}) × GA(${batch.gradientAccumulation}) = ${expectedGBS} ≠ ${batch.globalBatchSize}`);
  }
  
  if (batch.microBatchSize < 1) {
    errors.push('Micro batch size must be at least 1');
  }
  
  if (batch.gradientAccumulation < 1) {
    errors.push('Gradient accumulation must be at least 1');
  }
  
  return errors;
}

// ============ Parallelism Validation ============

/**
 * Validate parallelism configuration against model config
 */
export function validateParallelism(
  parallelism: ParallelismConfig,
  model: ModelConfig,
  cluster: ClusterTopology
): string[] {
  const errors: string[] = [];
  const { dataParallel, tensorParallel, pipelineParallel, expertParallel, contextParallel, contextParallelType } = parallelism;
  
  // Check total GPUs (now includes CP)
  const requiredGPUs = dataParallel * tensorParallel * pipelineParallel * expertParallel * contextParallel;
  if (requiredGPUs > cluster.totalGPUs) {
    errors.push(`Configuration requires ${requiredGPUs} GPUs but only ${cluster.totalGPUs} available`);
  }
  
  // TP must divide attention heads
  if (model.numAttentionHeads % tensorParallel !== 0) {
    errors.push(`TP=${tensorParallel} must divide num_attention_heads=${model.numAttentionHeads}`);
  }
  
  // TP must divide KV heads
  if (model.numKVHeads % tensorParallel !== 0) {
    errors.push(`TP=${tensorParallel} must divide num_kv_heads=${model.numKVHeads}`);
  }
  
  // PP must divide layers
  if (model.numLayers % pipelineParallel !== 0) {
    errors.push(`PP=${pipelineParallel} must divide num_layers=${model.numLayers}`);
  }
  
  // EP must divide experts (for MoE)
  if (model.ffnType === 'moe' && model.numExperts) {
    if (model.numExperts % expertParallel !== 0) {
      errors.push(`EP=${expertParallel} must divide num_experts=${model.numExperts}`);
    }
  }
  
  // ============ Context Parallel Validation ============
  if (contextParallel > 1) {
    // Effective KV heads after TP (KV heads is the limiting factor for Ulysses in GQA)
    const effectiveKVHeads = model.numKVHeads / tensorParallel;
    
    if (contextParallelType === 'ulysses') {
      // Ulysses: CP must divide the number of KV heads (more restrictive in GQA)
      // After TP, each GPU has effectiveKVHeads, Ulysses further divides by CP
      if (effectiveKVHeads % contextParallel !== 0) {
        errors.push(`Ulysses CP=${contextParallel} must divide KV heads after TP (${effectiveKVHeads}). Consider using Ring or Hybrid.`);
      }
      if (contextParallel > effectiveKVHeads) {
        errors.push(`Ulysses CP=${contextParallel} exceeds available KV heads (${effectiveKVHeads}). Use Ring attention instead.`);
      }
    } else if (contextParallelType === 'hybrid') {
      // Hybrid: Ulysses up to KV heads limit, then Ring for the rest
      // Calculate Ulysses degree (must divide KV heads)
      const maxUlysses = effectiveKVHeads;
      if (contextParallel > maxUlysses) {
        // This is okay for hybrid - will use Ring for the extra parallelism
        // Just a warning that Ring will be used
      }
    }
    // Ring attention has no head limitation
    
    // CP should be within NVLink domain for best performance
    if (contextParallel > cluster.gpusPerNode && contextParallelType === 'ulysses') {
      errors.push(`Ulysses CP=${contextParallel} crosses node boundary. All-to-all is latency-sensitive.`);
    }
  }
  
  // TP should be within NVLink domain
  if (tensorParallel > cluster.gpusPerNode) {
    errors.push(`TP=${tensorParallel} crosses node boundary (${cluster.gpusPerNode} GPUs/node). This will be slow.`);
  }
  
  return errors;
}

// ============ Memory Calculations ============

/**
 * Get bytes per element for precision
 */
function getPrecisionBytes(precision: 'fp32' | 'fp16' | 'bf16'): number {
  return precision === 'fp32' ? 4 : 2;
}

/**
 * Detailed activation memory calculation per layer with tensor breakdown
 * 
 * Reference: "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al.)
 * FlashAttention: No need to store attention scores (B × H × S × S)
 */
export interface ActivationTensorBreakdown {
  name: string;
  shape: string;
  shapeValues: number[];
  bytes: number;
  bytesPerGPU: number;
  formula: string;
  stored: boolean;  // Whether this is stored (vs recomputed)
}

export interface LayerActivationBreakdown {
  tensors: ActivationTensorBreakdown[];
  totalBytes: number;
  totalBytesPerGPU: number;
  formula: string;
}

export function calculateDetailedActivationMemory(
  model: ModelConfig,
  microBatchSize: number,
  seqLength: number,
  precision: 'fp32' | 'fp16' | 'bf16',
  tensorParallel: number,
  sequenceParallel: boolean,
  recomputation: RecomputationStrategy,
  flashAttention: boolean,
  contextParallel: number = 1  // CP splits sequence across GPUs
): LayerActivationBreakdown {
  const B = microBatchSize;
  // With Context Parallel, each GPU only handles S/CP tokens
  const S = Math.ceil(seqLength / contextParallel);
  const d = model.hiddenSize;
  const H = model.numAttentionHeads;
  const H_kv = model.numKVHeads;
  const d_h = model.headDim;
  const d_ff = model.intermediateSize;
  const elemBytes = getPrecisionBytes(precision);
  
  const tensors: ActivationTensorBreakdown[] = [];
  
  // Helper to add tensor with TP/SP consideration
  const addTensor = (
    name: string, 
    shape: string, 
    shapeValues: number[], 
    formula: string,
    isTPSharded: boolean = false,
    isSPSharded: boolean = false,
    stored: boolean = true
  ) => {
    const elements = shapeValues.reduce((a, b) => a * b, 1);
    const bytes = elements * elemBytes;
    
    let bytesPerGPU = bytes;
    if (isTPSharded && tensorParallel > 1) {
      bytesPerGPU = bytes / tensorParallel;
    }
    if (isSPSharded && sequenceParallel && tensorParallel > 1) {
      bytesPerGPU = bytesPerGPU / tensorParallel;
    }
    
    tensors.push({ name, shape, shapeValues, bytes, bytesPerGPU, formula, stored });
  };
  
  const useRecompute = recomputation === 'full' || recomputation === 'block';
  const useSelective = recomputation === 'selective';
  
  // ============ Attention Module ============
  
  // Layer input (always stored for backward)
  addTensor(
    'Layer Input', 
    '[B, S, d]', 
    [B, S, d],
    `${B} × ${S} × ${d} × ${elemBytes} bytes`,
    false, sequenceParallel, true
  );
  
  // Pre-attention LayerNorm output (for backward)
  if (!useRecompute) {
    addTensor(
      'Pre-Attn LN Output',
      '[B, S, d]',
      [B, S, d],
      `${B} × ${S} × ${d} × ${elemBytes} bytes`,
      false, sequenceParallel, true
    );
  }
  
  // Q projection output: [B, S, H, d_h]
  if (!useRecompute) {
    addTensor(
      'Q Projection',
      '[B, S, H, d_h]',
      [B, S, H, d_h],
      `${B} × ${S} × ${H} × ${d_h} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // K projection output: [B, S, H_kv, d_h]
  if (!useRecompute) {
    addTensor(
      'K Projection',
      '[B, S, H_kv, d_h]',
      [B, S, H_kv, d_h],
      `${B} × ${S} × ${H_kv} × ${d_h} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // V projection output: [B, S, H_kv, d_h]
  if (!useRecompute) {
    addTensor(
      'V Projection',
      '[B, S, H_kv, d_h]',
      [B, S, H_kv, d_h],
      `${B} × ${S} × ${H_kv} × ${d_h} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Attention Scores: [B, H, S, S] - NOT stored with FlashAttention!
  if (!flashAttention && !useRecompute && !useSelective) {
    addTensor(
      'Attention Scores (QK^T)',
      '[B, H, S, S]',
      [B, H, S, S],
      `${B} × ${H} × ${S} × ${S} × ${elemBytes} bytes (LARGE! Use FlashAttention to avoid)`,
      true, false, true
    );
    
    // Softmax output (same shape)
    addTensor(
      'Softmax Output',
      '[B, H, S, S]',
      [B, H, S, S],
      `${B} × ${H} × ${S} × ${S} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Attention output after V multiplication: [B, S, d]
  if (!useRecompute) {
    addTensor(
      'Attention Output (softmax @ V)',
      '[B, S, d]',
      [B, S, d],
      `${B} × ${S} × ${d} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Output projection result: [B, S, d]
  if (!useRecompute) {
    addTensor(
      'Output Projection',
      '[B, S, d]',
      [B, S, d],
      `${B} × ${S} × ${d} × ${elemBytes} bytes`,
      false, sequenceParallel, true
    );
  }
  
  // Post-attention residual (same as input typically)
  
  // ============ FFN Module ============
  
  // Pre-FFN LayerNorm output
  if (!useRecompute) {
    addTensor(
      'Pre-FFN LN Output',
      '[B, S, d]',
      [B, S, d],
      `${B} × ${S} × ${d} × ${elemBytes} bytes`,
      false, sequenceParallel, true
    );
  }
  
  // Gate projection (for gated FFN like SwiGLU)
  if (model.ffnType === 'gated' && !useRecompute) {
    addTensor(
      'Gate Projection',
      '[B, S, d_ff]',
      [B, S, d_ff],
      `${B} × ${S} × ${d_ff} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Up projection
  if (!useRecompute) {
    addTensor(
      'Up Projection',
      '[B, S, d_ff]',
      [B, S, d_ff],
      `${B} × ${S} × ${d_ff} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Activation function output (after SiLU/GELU)
  if (!useRecompute) {
    addTensor(
      'FFN Activation Output',
      '[B, S, d_ff]',
      [B, S, d_ff],
      `${B} × ${S} × ${d_ff} × ${elemBytes} bytes`,
      true, false, true
    );
  }
  
  // Down projection result
  if (!useRecompute) {
    addTensor(
      'Down Projection',
      '[B, S, d]',
      [B, S, d],
      `${B} × ${S} × ${d} × ${elemBytes} bytes`,
      false, sequenceParallel, true
    );
  }
  
  // Calculate totals
  const storedTensors = tensors.filter(t => t.stored);
  const totalBytes = storedTensors.reduce((sum, t) => sum + t.bytes, 0);
  const totalBytesPerGPU = storedTensors.reduce((sum, t) => sum + t.bytesPerGPU, 0);
  
  return {
    tensors,
    totalBytes,
    totalBytesPerGPU,
    formula: `Sum of stored activation tensors per layer (S=${S} tokens per GPU with CP=${contextParallel})`,
  };
}

/**
 * Calculate LM Head logits memory - this is a MAJOR memory consumer
 * Shape: [B, S, vocab_size] - cannot be recomputed efficiently
 * Must be stored for cross-entropy loss backward pass
 */
export function calculateLMHeadLogitsMemory(
  model: ModelConfig,
  microBatchSize: number,
  seqLength: number,
  precision: 'fp32' | 'fp16' | 'bf16',
  tensorParallel: number,
  contextParallel: number = 1
): { bytes: number; bytesPerGPU: number; formula: string; shape: string } {
  const B = microBatchSize;
  // With CP, each GPU handles S/CP tokens
  const S = Math.ceil(seqLength / contextParallel);
  const V = model.vocabSize;
  const elemBytes = getPrecisionBytes(precision);
  
  // Logits shape: [B, S, vocab_size]
  // This is stored for the backward pass through cross-entropy loss
  const bytes = B * S * V * elemBytes;
  
  // With TP, vocab is sharded across GPUs
  const bytesPerGPU = tensorParallel > 1 ? bytes / tensorParallel : bytes;
  
  return {
    bytes,
    bytesPerGPU,
    formula: `LM Head Logits: ${B} × ${S} × ${V} × ${elemBytes} bytes = ${(bytes / 1e9).toFixed(2)} GB (vocab=${V}, cannot recompute)`,
    shape: `[${B}, ${S}, ${V}]`,
  };
}

/**
 * Calculate total training memory breakdown with detailed precision info
 * 
 * Mixed Precision Training Memory Layout:
 * - Model weights: User-defined dtype (FP16/BF16/FP32)
 * - Master weights: Always FP32 (for numerical stability in optimizer)
 * - Optimizer momentum (m): Always FP32
 * - Optimizer variance (v): Always FP32  
 * - Gradients: User-defined dtype during backward
 * - Gradient accumulation buffer: FP32 (for accumulation across micro-batches)
 * - Activations: User-defined dtype
 */
export function calculateTrainingMemory(
  model: ModelConfig,
  config: TrainingConfig,
  _cluster: ClusterTopology
): TrainingMemoryBreakdown {
  const params = calculateModelParameters(model);
  const totalParams = params.total;
  
  const { parallelism, memoryOptimization, batch } = config;
  const { tensorParallel, pipelineParallel, dataParallel, sequenceParallel, zeroStage } = parallelism;
  
  const modelPrecisionBytes = getPrecisionBytes(config.mixedPrecision);
  const fp32Bytes = 4;
  
  // Effective params per GPU after TP/PP sharding
  const paramsAfterTPPP = totalParams / (tensorParallel * pipelineParallel);
  
  // ============ Model Weights ============
  // Stored in user-defined dtype
  const modelWeightsTotal = totalParams * modelPrecisionBytes;
  let modelWeightsPerGPU = paramsAfterTPPP * modelPrecisionBytes;
  if (zeroStage === 3) {
    modelWeightsPerGPU = modelWeightsPerGPU / dataParallel;
  }
  
  // ============ Master Weights (FP32) ============
  // Always FP32 for mixed precision training - optimizer updates in FP32
  const masterWeightsTotal = totalParams * fp32Bytes;
  let masterWeightsPerGPU = paramsAfterTPPP * fp32Bytes;
  if (zeroStage >= 1) {
    masterWeightsPerGPU = masterWeightsPerGPU / dataParallel;
  }
  // Only needed if using mixed precision (FP16/BF16)
  const needsMasterWeights = config.mixedPrecision !== 'fp32';
  
  // ============ Optimizer States (FP32) ============
  // Adam: momentum (m) and variance (v), always FP32
  const momentumTotal = totalParams * fp32Bytes;
  const varianceTotal = totalParams * fp32Bytes;
  let momentumPerGPU = paramsAfterTPPP * fp32Bytes;
  let variancePerGPU = paramsAfterTPPP * fp32Bytes;
  if (zeroStage >= 1) {
    momentumPerGPU = momentumPerGPU / dataParallel;
    variancePerGPU = variancePerGPU / dataParallel;
  }
  
  // ============ Gradients ============
  // Computed in user-defined dtype during backward
  const gradientsTotal = totalParams * modelPrecisionBytes;
  let gradientsPerGPU = paramsAfterTPPP * modelPrecisionBytes;
  if (zeroStage >= 2) {
    gradientsPerGPU = gradientsPerGPU / dataParallel;
  }
  
  // ============ Gradient Accumulation Buffer (FP32) ============
  // FP32 for numerical stability during accumulation
  // Only needed when gradient_accumulation > 1
  const needsGradAccumBuffer = batch.gradientAccumulation > 1;
  let gradAccumBufferPerGPU = 0;
  let gradAccumBufferTotal = 0;
  if (needsGradAccumBuffer) {
    gradAccumBufferTotal = totalParams * fp32Bytes;
    gradAccumBufferPerGPU = paramsAfterTPPP * fp32Bytes;
    if (zeroStage >= 2) {
      gradAccumBufferPerGPU = gradAccumBufferPerGPU / dataParallel;
    }
  }
  
  // ============ Activation Memory (Detailed) ============
  const flashAttention = memoryOptimization.flashAttention ?? true; // Default to FlashAttention
  const { contextParallel } = parallelism;
  
  // Per-layer activation memory (accounts for CP sequence split)
  const activationBreakdown = calculateDetailedActivationMemory(
    model,
    batch.microBatchSize,
    config.maxSeqLength,
    config.mixedPrecision,
    tensorParallel,
    sequenceParallel,
    memoryOptimization.recomputation,
    flashAttention,
    contextParallel  // Pass CP to reduce S by factor of CP
  );
  
  const activationsPerLayer = activationBreakdown.totalBytesPerGPU;
  
  // ============ LM Head Logits Memory (MAJOR consumer!) ============
  // Shape: [B, S/CP, vocab_size] - stored for cross-entropy loss backward
  // This CANNOT be saved by recomputation - always needed for loss gradient
  const lmHeadLogits = calculateLMHeadLogitsMemory(
    model,
    batch.microBatchSize,
    config.maxSeqLength,
    config.mixedPrecision,
    tensorParallel,
    contextParallel
  );
  
  // Total activations depends on recomputation strategy
  // - None/Selective: Store all layer activations (intermediate tensors)
  // - Block: Store layer INPUTS for all layers, recompute attention/FFN within each block during backward
  // - Full: Store only model input, recompute everything
  // With PP, we need to store activations for microbatches in flight
  const layersPerGPU = model.numLayers / pipelineParallel;
  let layersStored: number;
  if (memoryOptimization.recomputation === 'full') {
    // Full recomputation: only store model input, recompute all layers
    layersStored = 1;
  } else if (memoryOptimization.recomputation === 'block') {
    // Block recomputation: store layer INPUT for each Transformer block
    // Attention/FFN are recomputed during backward, but we need to store 
    // the input to each block so we can recompute it
    layersStored = layersPerGPU; // All layers, but only storing layer inputs (not intermediate tensors)
  } else {
    // Without recomputation (none/selective), store all layers with their intermediate tensors
    layersStored = layersPerGPU;
  }
  
  // Total activation memory = per-layer × layers + LM Head logits
  const activationsTotal = activationsPerLayer * layersStored + lmHeadLogits.bytesPerGPU;
  const numInFlight = pipelineParallel;
  const peakActivations = activationsPerLayer * Math.min(layersStored, numInFlight) + lmHeadLogits.bytesPerGPU;
  
  // ============ Communication Buffers ============
  // Double buffering for overlap
  const communicationBuffers = batch.microBatchSize * config.maxSeqLength * model.hiddenSize * 
    modelPrecisionBytes * 2;
  
  // ============ Build Component Breakdown with Formulas ============
  const P = totalParams;
  const TP = tensorParallel;
  const PP = pipelineParallel;
  const DP = dataParallel;
  
  const components = {
    modelWeights: {
      name: 'Model Weights',
      bytes: modelWeightsTotal,
      bytesPerGPU: modelWeightsPerGPU,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Model parameters in ${config.mixedPrecision.toUpperCase()}`,
      formula: zeroStage === 3 
        ? `P / (TP × PP × DP) × ${modelPrecisionBytes} = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × ${modelPrecisionBytes}`
        : `P / (TP × PP) × ${modelPrecisionBytes} = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP}) × ${modelPrecisionBytes}`,
      isRequired: true,
    },
    masterWeights: {
      name: 'Master Weights (FP32)',
      bytes: needsMasterWeights ? masterWeightsTotal : 0,
      bytesPerGPU: needsMasterWeights ? masterWeightsPerGPU : 0,
      precision: 'fp32' as const,
      description: 'FP32 copy for optimizer updates (required for mixed precision)',
      formula: needsMasterWeights 
        ? `P / (TP × PP × DP) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × 4 bytes`
        : 'Not needed (training in FP32)',
      isRequired: needsMasterWeights,
    },
    optimizerMomentum: {
      name: 'Optimizer Momentum (m)',
      bytes: momentumTotal,
      bytesPerGPU: momentumPerGPU,
      precision: 'fp32' as const,
      description: 'Adam first moment estimate (always FP32)',
      formula: `P / (TP × PP × DP) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × 4 bytes`,
      isRequired: true,
    },
    optimizerVariance: {
      name: 'Optimizer Variance (v)',
      bytes: varianceTotal,
      bytesPerGPU: variancePerGPU,
      precision: 'fp32' as const,
      description: 'Adam second moment estimate (always FP32)',
      formula: `P / (TP × PP × DP) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × 4 bytes`,
      isRequired: true,
    },
    gradients: {
      name: 'Gradients',
      bytes: gradientsTotal,
      bytesPerGPU: gradientsPerGPU,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Gradient tensors in ${config.mixedPrecision.toUpperCase()}`,
      formula: zeroStage >= 2
        ? `P / (TP × PP × DP) × ${modelPrecisionBytes} = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × ${modelPrecisionBytes}`
        : `P / (TP × PP) × ${modelPrecisionBytes} = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP}) × ${modelPrecisionBytes}`,
      isRequired: true,
    },
    gradientAccumBuffer: {
      name: 'Gradient Accum Buffer',
      bytes: gradAccumBufferTotal,
      bytesPerGPU: gradAccumBufferPerGPU,
      precision: 'fp32' as const,
      description: 'FP32 buffer for gradient accumulation (stability)',
      formula: needsGradAccumBuffer 
        ? `P / (TP × PP × DP) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${DP}) × 4 bytes`
        : 'Not needed (GA=1)',
      isRequired: needsGradAccumBuffer,
    },
    activations: {
      name: 'Activations',
      bytes: activationsTotal,
      bytesPerGPU: activationsTotal,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Layer activations (${layersStored} layers, CP=${contextParallel}) + LM Head logits`,
      formula: `${formatBytes(activationsPerLayer)}/layer × ${layersStored} layers + LM Head ${formatBytes(lmHeadLogits.bytesPerGPU)} = ${formatBytes(activationsTotal)}`,
      isRequired: true,
      tensors: [
        // Per-layer activation tensors
        ...activationBreakdown.tensors.filter(t => t.stored).map(t => ({
          name: t.name,
          shape: t.shape,
          shapeValues: t.shapeValues,
          elementCount: t.shapeValues.reduce((a, b) => a * b, 1),
          bytes: t.bytes,
          bytesPerGPU: t.bytesPerGPU,
          precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
          formula: t.formula,
          description: t.name,
        })),
        // LM Head logits - major memory consumer, cannot be recomputed
        {
          name: 'LM Head Logits',
          shape: lmHeadLogits.shape,
          shapeValues: [batch.microBatchSize, Math.ceil(config.maxSeqLength / contextParallel), model.vocabSize],
          elementCount: batch.microBatchSize * Math.ceil(config.maxSeqLength / contextParallel) * model.vocabSize,
          bytes: lmHeadLogits.bytes,
          bytesPerGPU: lmHeadLogits.bytesPerGPU,
          precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
          formula: lmHeadLogits.formula,
          description: 'LM Head output logits (stored for loss backward, CANNOT recompute)',
        },
      ],
    },
    communicationBuffers: {
      name: 'Communication Buffers',
      bytes: communicationBuffers,
      bytesPerGPU: communicationBuffers,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: 'AllReduce/AllGather double buffers',
      formula: `B × S × d × ${modelPrecisionBytes} × 2 = ${batch.microBatchSize} × ${config.maxSeqLength} × ${model.hiddenSize} × ${modelPrecisionBytes} × 2`,
      isRequired: true,
    },
  };
  
  // ============ Calculate Totals ============
  const modelStatesBytes = modelWeightsPerGPU + 
    (needsMasterWeights ? masterWeightsPerGPU : 0) + 
    momentumPerGPU + variancePerGPU + 
    gradientsPerGPU + gradAccumBufferPerGPU;
  const activationsBytes = activationsTotal;
  const buffersBytes = communicationBuffers;
  
  const totalPerGPU = modelStatesBytes + activationsBytes + buffersBytes;
  
  // Breakdown by precision
  const byPrecision = {
    fp32: (needsMasterWeights ? masterWeightsPerGPU : 0) + 
          momentumPerGPU + variancePerGPU + 
          gradAccumBufferPerGPU,
    fp16: config.mixedPrecision === 'fp16' ? 
          (modelWeightsPerGPU + gradientsPerGPU + activationsTotal + communicationBuffers) : 0,
    bf16: config.mixedPrecision === 'bf16' ? 
          (modelWeightsPerGPU + gradientsPerGPU + activationsTotal + communicationBuffers) : 0,
  };
  if (config.mixedPrecision === 'fp32') {
    byPrecision.fp32 += modelWeightsPerGPU + gradientsPerGPU + activationsTotal + communicationBuffers;
  }
  
  // Legacy fields for backward compatibility
  const parametersLegacy = modelWeightsTotal;
  const gradientsLegacy = gradientsTotal;
  const optimizerStatesLegacy = momentumTotal + varianceTotal + (needsMasterWeights ? masterWeightsTotal : 0);
  const optimizerPerGPU = momentumPerGPU + variancePerGPU + (needsMasterWeights ? masterWeightsPerGPU : 0);
  
  return {
    // New detailed breakdown
    components,
    modelStatesBytes,
    activationsBytes,
    buffersBytes,
    totalBytes: totalPerGPU,
    totalPerGPU,
    byPrecision,
    
    // Recomputation info
    layersStored,
    layersPerGPU,
    
    // Legacy fields
    parameters: parametersLegacy,
    gradients: gradientsLegacy,
    optimizerStates: optimizerStatesLegacy,
    activationsPerLayer,
    activationsTotal,
    peakActivations,
    communicationBuffers,
    parametersPerGPU: modelWeightsPerGPU,
    gradientsPerGPU,
    optimizerPerGPU,
    activationsPerGPU: activationsTotal,
  };
}

// ============ Communication Calculations ============

/**
 * Calculate Ring AllReduce time
 */
function ringAllReduceTime(dataBytes: number, numRanks: number, bandwidth: number): number {
  if (numRanks <= 1) return 0;
  const factor = 2 * (numRanks - 1) / numRanks;
  return (dataBytes * factor) / (bandwidth * 1e9) * 1000; // ms
}

/**
 * Calculate AllGather time
 */
function allGatherTime(dataBytes: number, numRanks: number, bandwidth: number): number {
  if (numRanks <= 1) return 0;
  const factor = (numRanks - 1) / numRanks;
  return (dataBytes * factor) / (bandwidth * 1e9) * 1000; // ms
}

/**
 * Calculate P2P time
 */
function p2pTime(dataBytes: number, bandwidth: number): number {
  return dataBytes / (bandwidth * 1e9) * 1000; // ms
}

/**
 * Calculate All-to-All time (for Ulysses)
 * All-to-All sends N/P elements to each of P-1 peers
 */
function allToAllTime(dataBytes: number, numRanks: number, bandwidth: number): number {
  if (numRanks <= 1) return 0;
  // Each rank sends (P-1)/P of its data
  const factor = (numRanks - 1) / numRanks;
  return (dataBytes * factor) / (bandwidth * 1e9) * 1000; // ms
}

/**
 * Calculate communication breakdown
 */
export function calculateCommunication(
  model: ModelConfig,
  config: TrainingConfig,
  cluster: ClusterTopology
): CommunicationBreakdown {
  const params = calculateModelParameters(model);
  const { parallelism, batch } = config;
  const { dataParallel, tensorParallel, pipelineParallel, expertParallel, 
          contextParallel, contextParallelType, zeroStage } = parallelism;
  const bytes = getPrecisionBytes(config.gradientPrecision);
  
  // Effective parameters per DP rank (after TP/PP sharding)
  const paramsPerRank = params.total / (tensorParallel * pipelineParallel);
  
  // Effective sequence length per GPU with Context Parallel
  const seqLenPerGPU = config.maxSeqLength / contextParallel;
  
  // Data Parallel communication (gradient sync)
  let dataParallelVolume = 0;
  if (zeroStage <= 1) {
    // DDP or ZeRO-1: AllReduce gradients
    dataParallelVolume = 2 * paramsPerRank * bytes;
  } else {
    // ZeRO-2/3: ReduceScatter + AllGather
    dataParallelVolume = 2 * paramsPerRank * bytes;
  }
  
  // ZeRO-3 additional parameter gather
  let zeroVolume = 0;
  if (zeroStage === 3) {
    // AllGather params before forward and backward
    zeroVolume = 2 * paramsPerRank * bytes;
  }
  
  // Tensor Parallel communication (per layer, per micro-batch)
  // Note: with CP, each GPU has seqLenPerGPU tokens
  const activationSize = batch.microBatchSize * seqLenPerGPU * model.hiddenSize * bytes;
  // 4 AllReduce per layer (2 forward: attn + ffn, 2 backward: attn + ffn)
  const tpVolume = 4 * activationSize * (model.numLayers / pipelineParallel);
  const tensorParallelVolume = tpVolume * batch.gradientAccumulation;
  
  // Pipeline Parallel communication
  const ppVolume = 2 * activationSize * (pipelineParallel - 1); // P2P per micro-batch
  const numMicrobatches = batch.gradientAccumulation * dataParallel;
  const pipelineParallelVolume = ppVolume * numMicrobatches;
  
  // Expert Parallel communication (for MoE)
  let expertParallelVolume = 0;
  if (model.ffnType === 'moe' && expertParallel > 1) {
    const topK = model.numExpertsPerToken || 2;
    const moeLayerCount = model.numLayers / pipelineParallel;
    // All-to-All: dispatch + combine
    expertParallelVolume = 2 * batch.microBatchSize * seqLenPerGPU * 
      model.hiddenSize * topK * bytes * moeLayerCount * batch.gradientAccumulation;
  }
  
  // ============ Context Parallel Communication ============
  // Reference: DeepSpeed Ulysses paper, Ring Attention papers
  let contextParallelVolume = 0;
  if (contextParallel > 1) {
    const layersPerStage = model.numLayers / pipelineParallel;
    
    if (contextParallelType === 'ulysses') {
      // Ulysses: All-to-all for Q, K, V before attention, then all-to-all for output
      // Per layer: 4 all-to-all operations (Q, K, V input + output)
      // Each all-to-all moves (N/CP) * d elements
      // Total per layer: 4 * (N/CP) * d * bytes
      // For forward + backward: 2x
      const perLayerVolume = 4 * seqLenPerGPU * model.hiddenSize * bytes;
      contextParallelVolume = 2 * perLayerVolume * layersPerStage * batch.gradientAccumulation;
    } else if (contextParallelType === 'ring') {
      // Ring Attention: P2P ring communication of KV blocks
      // Each GPU sends KV to next neighbor for CP-1 steps
      // Per step: 2 * (N/CP) * d_kv * bytes (K + V)
      // Total steps: CP - 1
      // For forward + backward: 2x
      const kvHeadDim = model.numKVHeads * model.headDim / tensorParallel;
      const perStepVolume = 2 * seqLenPerGPU * kvHeadDim * bytes;
      const numSteps = contextParallel - 1;
      contextParallelVolume = 2 * perStepVolume * numSteps * layersPerStage * batch.gradientAccumulation;
    } else if (contextParallelType === 'hybrid') {
      // Hybrid: Ulysses within smaller groups, Ring across groups
      // Calculate effective Ulysses degree (limited by KV heads)
      const effectiveKVHeads = model.numKVHeads / tensorParallel;
      const ulyssesDegree = Math.min(contextParallel, effectiveKVHeads);
      const ringDegree = contextParallel / ulyssesDegree;
      
      // Ulysses component
      const ulyssesSeqLen = config.maxSeqLength / ulyssesDegree;
      const ulyssesVolume = 4 * ulyssesSeqLen * model.hiddenSize * bytes;
      
      // Ring component
      const ringSeqLen = ulyssesSeqLen / ringDegree;
      const kvHeadDim = model.numKVHeads * model.headDim / tensorParallel / ulyssesDegree;
      const ringPerStep = 2 * ringSeqLen * kvHeadDim * bytes;
      const ringVolume = ringPerStep * (ringDegree - 1);
      
      contextParallelVolume = 2 * (ulyssesVolume + ringVolume) * layersPerStage * batch.gradientAccumulation;
    }
  }
  
  const totalVolume = dataParallelVolume + tensorParallelVolume + 
    pipelineParallelVolume + expertParallelVolume + contextParallelVolume + zeroVolume;
  
  // Calculate times
  const intraNodeBW = cluster.intraNodeBandwidth;
  const interNodeBW = cluster.interNodeBandwidth;
  
  // DP typically crosses nodes
  const dpBandwidth = dataParallel > cluster.gpusPerNode ? interNodeBW : intraNodeBW;
  const dataParallelTime = ringAllReduceTime(dataParallelVolume, dataParallel, dpBandwidth);
  
  // TP should be within node
  const tpBandwidth = tensorParallel <= cluster.gpusPerNode ? intraNodeBW : interNodeBW;
  const tensorParallelTime = ringAllReduceTime(
    tensorParallelVolume / batch.gradientAccumulation, 
    tensorParallel, 
    tpBandwidth
  ) * batch.gradientAccumulation;
  
  // PP communication
  const ppBandwidth = pipelineParallel <= cluster.gpusPerNode ? intraNodeBW : interNodeBW;
  const pipelineParallelTime = p2pTime(pipelineParallelVolume, ppBandwidth);
  
  // EP communication
  const epBandwidth = expertParallel <= cluster.gpusPerNode ? intraNodeBW : interNodeBW;
  const expertParallelTime = expertParallelVolume > 0 ? 
    (expertParallelVolume / (epBandwidth * 1e9)) * 1000 : 0;
  
  // CP communication
  // Ulysses benefits from NVLink (all-to-all is latency sensitive)
  // Ring can work across nodes (P2P is bandwidth bound)
  const cpBandwidth = contextParallel <= cluster.gpusPerNode ? intraNodeBW : interNodeBW;
  let contextParallelTime = 0;
  if (contextParallel > 1) {
    if (contextParallelType === 'ulysses') {
      // All-to-all time
      contextParallelTime = allToAllTime(contextParallelVolume, contextParallel, cpBandwidth);
    } else {
      // Ring/Hybrid: P2P time
      contextParallelTime = p2pTime(contextParallelVolume, cpBandwidth);
    }
  }
  
  // ZeRO-3 time
  const zeroTime = zeroVolume > 0 ? 
    allGatherTime(zeroVolume / 2, dataParallel, dpBandwidth) * 2 : 0;
  
  const totalTime = dataParallelTime + tensorParallelTime + 
    pipelineParallelTime + expertParallelTime + contextParallelTime + zeroTime;
  
  // Pipeline bubble
  const bubbleRatio = pipelineParallel > 1 ? 
    (pipelineParallel - 1) / numMicrobatches : 0;
  
  return {
    dataParallelVolume,
    tensorParallelVolume,
    pipelineParallelVolume,
    expertParallelVolume,
    contextParallelVolume,
    zeroVolume,
    totalVolume,
    dataParallelTime,
    tensorParallelTime,
    pipelineParallelTime,
    expertParallelTime,
    contextParallelTime,
    zeroTime,
    totalTime,
    bubbleRatio,
    bubbleTime: 0, // Will be calculated with compute time
  };
}

// ============ Training Step Analysis ============

/**
 * Calculate complete training step analysis
 */
export function analyzeTrainingStep(
  model: ModelConfig,
  config: TrainingConfig,
  cluster: ClusterTopology
): TrainingStepAnalysis {
  const params = calculateModelParameters(model);
  
  // Compute FLOPs
  const tokens = config.batch.microBatchSize * config.maxSeqLength;
  const forwardFlops = 2 * params.total * tokens; // ~2N per token
  const backwardFlops = 2 * forwardFlops; // Backward is ~2x forward
  const totalFlops = forwardFlops + backwardFlops;
  
  // Recomputation overhead
  let recomputationMultiplier = 1;
  if (config.memoryOptimization.recomputation === 'full') {
    recomputationMultiplier = 1.33; // ~33% overhead
  } else if (config.memoryOptimization.recomputation === 'selective') {
    recomputationMultiplier = 1.05; // ~5% overhead with selective
  }
  
  // Compute time per micro-batch
  const effectiveCompute = cluster.gpuComputeTFLOPS * 1e12;
  const forwardTime = forwardFlops / effectiveCompute * 1000; // ms
  const backwardTime = backwardFlops / effectiveCompute * 1000;
  const computeTime = (forwardTime + backwardTime) * recomputationMultiplier;
  const recomputationOverhead = (forwardTime + backwardTime) * (recomputationMultiplier - 1);
  
  // Memory breakdown
  const memoryBreakdown = calculateTrainingMemory(model, config, cluster);
  
  // Communication breakdown
  const communicationBreakdown = calculateCommunication(model, config, cluster);
  
  // Update bubble time with compute time
  communicationBreakdown.bubbleTime = computeTime * config.batch.gradientAccumulation * 
    communicationBreakdown.bubbleRatio;
  
  // Total time per step
  // TP/PP/CP communication can partially overlap with compute
  const overlapFactor = 0.5; // Assume 50% overlap
  const effectiveCommTime = communicationBreakdown.dataParallelTime + 
    (communicationBreakdown.tensorParallelTime + 
     communicationBreakdown.pipelineParallelTime + 
     communicationBreakdown.expertParallelTime +
     communicationBreakdown.contextParallelTime) * (1 - overlapFactor) +
    communicationBreakdown.zeroTime;
  
  const totalComputeTime = computeTime * config.batch.gradientAccumulation;
  const timePerStep = totalComputeTime + effectiveCommTime + communicationBreakdown.bubbleTime;
  
  // Efficiency metrics
  const theoreticalPeakFlops = cluster.gpuComputeTFLOPS * 1e12 * 
    (timePerStep / 1000) * config.parallelism.totalGPUs;
  const mfu = (totalFlops * config.batch.gradientAccumulation) / theoreticalPeakFlops;
  const hfu = mfu * recomputationMultiplier; // HFU includes recomputation
  
  const computeEfficiency = totalComputeTime / timePerStep;
  const memoryEfficiency = memoryBreakdown.totalPerGPU / (cluster.gpuMemoryGB * 1e9);
  
  // Throughput
  const tokensPerSecond = (config.batch.globalBatchSize * config.maxSeqLength) / (timePerStep / 1000);
  const samplesPerSecond = config.batch.globalBatchSize / (timePerStep / 1000);
  
  return {
    forwardFlops,
    backwardFlops,
    totalFlops: totalFlops * config.batch.gradientAccumulation,
    forwardTime,
    backwardTime,
    computeTime: totalComputeTime,
    recomputationOverhead: recomputationOverhead * config.batch.gradientAccumulation,
    memoryBreakdown,
    communicationBreakdown,
    mfu,
    hfu,
    computeEfficiency,
    memoryEfficiency,
    tokensPerSecond,
    samplesPerSecond,
    timePerStep,
  };
}

// ============ Format Utilities ============

export function formatBytes(bytes: number): string {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
  return bytes.toFixed(0) + ' B';
}

export function formatThroughput(tokensPerSec: number): string {
  if (tokensPerSec >= 1e6) return (tokensPerSec / 1e6).toFixed(2) + 'M';
  if (tokensPerSec >= 1e3) return (tokensPerSec / 1e3).toFixed(1) + 'K';
  return tokensPerSec.toFixed(0);
}
