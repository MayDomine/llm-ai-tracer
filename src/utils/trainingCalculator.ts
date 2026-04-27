/**
 * Training Calculator
 * 
 * Calculates memory, communication, and compute requirements for distributed LLM training.
 * Implements formulas from Megatron-LM, ZeRO, and related research.
 * 
 * Formula References:
 * - Transformer training FLOPs use a Megatron-style decomposition with
 *   attention, FFN/MoE, and vocab terms instead of a raw 6N shortcut
 * - Ring AllReduce: 2×(n-1)/n × data/BW (NCCL)
 * - Pipeline Bubble: (PP-1)/microbatches (GPipe, Huang et al., 2019)
 * - ZeRO Memory Formulas (Rajbhandari et al., 2020)
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
  CalculatorAssumptions,
  NetworkLatencyConfig,
} from '../types/training';
import { DEFAULT_CALCULATOR_ASSUMPTIONS, DEFAULT_NETWORK_LATENCY } from '../types/training';
import {
  calculateModelParameters,
  estimateTrainingFlops,
  estimateTrainingFlopsPerToken,
} from './calculator';
// Unified activation calculator is available for detailed calculations:
// import { calculatePerLayerActivations, calculateLMHeadLogitsMemory } from './activationCalculator';

/**
 * Get merged assumptions (user overrides + defaults)
 */
function getAssumptions(config: TrainingConfig): CalculatorAssumptions {
  return {
    ...DEFAULT_CALCULATOR_ASSUMPTIONS,
    ...config.assumptions,
  };
}

/**
 * Get network latency config with defaults
 */
function getNetworkLatencyConfig(cluster: ClusterTopology): NetworkLatencyConfig {
  return cluster.networkLatency ?? DEFAULT_NETWORK_LATENCY;
}

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
  // EP doesn't consume additional GPUs - experts are distributed within the DP×TP group
  const requiredGPUs = dataParallel * tensorParallel * pipelineParallel * contextParallel;
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
  const { tensorParallel, pipelineParallel, dataParallel, sequenceParallel, zeroStage, effectiveDataParallel, expertParallel } = parallelism;
  
  // ZeRO/FSDP shards across the entire DP group = DP_batch × CP
  // CP is a special form of DP that splits sequence instead of batch
  const zeroShardingDimension = effectiveDataParallel || dataParallel;
  
  const modelPrecisionBytes = getPrecisionBytes(config.mixedPrecision);
  const fp32Bytes = 4;
  
  // For MoE models, attention and FFN have different sharding:
  // - Attention: ÷ (TP × PP)
  // - FFN/Experts: ÷ (EP × PP) for MoE, ÷ (TP × PP) for dense
  // - Embedding, LM Head, LayerNorm: follow attention sharding
  const isMoE = model.ffnType === 'moe';
  const EP = isMoE ? expertParallel : tensorParallel; // EP for FFN, TP for dense
  
  // Params breakdown by module type (all divide by PP for pipeline stages)
  const attnParamsPerGPU = params.attention / (tensorParallel * pipelineParallel);
  const ffnParamsPerGPU = params.ffn / (EP * pipelineParallel); // EP for MoE, TP for dense
  const embeddingParamsPerGPU = params.embedding / pipelineParallel; // Embedding not sharded by TP typically
  const lmHeadParamsPerGPU = params.lmHead / (tensorParallel * pipelineParallel);
  const layerNormParamsPerGPU = params.layerNorm / pipelineParallel;
  
  // Total params per GPU after parallelism sharding
  const paramsAfterTPPP = attnParamsPerGPU + ffnParamsPerGPU + embeddingParamsPerGPU + lmHeadParamsPerGPU + layerNormParamsPerGPU;
  
  // ============ Model Weights ============
  // Stored in user-defined dtype
  const modelWeightsTotal = totalParams * modelPrecisionBytes;
  let modelWeightsPerGPU = paramsAfterTPPP * modelPrecisionBytes;
  if (zeroStage === 3) {
    // ZeRO-3/FSDP shards parameters across DP × CP
    modelWeightsPerGPU = modelWeightsPerGPU / zeroShardingDimension;
  }
  
  // ============ Master Weights (FP32) ============
  // Always FP32 for mixed precision training - optimizer updates in FP32
  const masterWeightsTotal = totalParams * fp32Bytes;
  let masterWeightsPerGPU = paramsAfterTPPP * fp32Bytes;
  if (zeroStage >= 1) {
    // ZeRO-1/2/3: optimizer states sharded across DP × CP
    masterWeightsPerGPU = masterWeightsPerGPU / zeroShardingDimension;
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
    // ZeRO-1/2/3: optimizer states sharded across DP × CP
    momentumPerGPU = momentumPerGPU / zeroShardingDimension;
    variancePerGPU = variancePerGPU / zeroShardingDimension;
  }
  
  // ============ Gradients ============
  // Computed in user-defined dtype during backward
  const gradientsTotal = totalParams * modelPrecisionBytes;
  let gradientsPerGPU = paramsAfterTPPP * modelPrecisionBytes;
  if (zeroStage >= 2) {
    // ZeRO-2/3: gradients sharded across DP × CP
    gradientsPerGPU = gradientsPerGPU / zeroShardingDimension;
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
      // ZeRO-2/3: gradient accum buffer sharded across DP × CP
      gradAccumBufferPerGPU = gradAccumBufferPerGPU / zeroShardingDimension;
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
  
  // Per-layer activation memory (NOT including LM Head - that's separate)
  const activationsPerLayerTotal = activationsPerLayer * layersStored;
  const numInFlight = pipelineParallel;
  const peakActivations = activationsPerLayer * Math.min(layersStored, numInFlight);
  
  // LM Head is computed ONCE (not per layer) - separate from layer activations
  const lmHeadLogitsBytes = lmHeadLogits.bytesPerGPU;
  
  // Total activations = layers + LM Head (but tracked separately for clarity)
  const activationsTotal = activationsPerLayerTotal + lmHeadLogitsBytes;
  
  // ============ Communication Buffers ============
  // Double buffering for overlap
  const communicationBuffers = batch.microBatchSize * config.maxSeqLength * model.hiddenSize * 
    modelPrecisionBytes * 2;
  
  // ============ Build Component Breakdown with Formulas ============
  const P = totalParams;
  const TP = tensorParallel;
  const PP = pipelineParallel;
  const DP = dataParallel;
  const CP = contextParallel;
  const DP_eff = zeroShardingDimension; // DP × CP for ZeRO sharding
  const EPVal = expertParallel;
  
  // Format the sharding dimension for formulas
  const shardingLabel = CP > 1 ? `DP×CP` : `DP`;
  const shardingValue = CP > 1 ? `${DP}×${CP}=${DP_eff}` : `${DP}`;
  
  // For MoE, show different sharding for Attention vs FFN
  const moeNote = isMoE 
    ? `\n(Attn: TP×PP=${TP}×${PP}, FFN: EP×PP=${EPVal}×${PP})`
    : '';
  
  const components = {
    modelWeights: {
      name: 'Model Weights',
      bytes: modelWeightsTotal,
      bytesPerGPU: modelWeightsPerGPU,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Model parameters in ${config.mixedPrecision.toUpperCase()}${isMoE ? ' (MoE: Attn÷TP, FFN÷EP)' : ''}`,
      formula: zeroStage === 3 
        ? `Attn/(TP×PP×${shardingLabel}) + FFN/(${isMoE ? 'EP' : 'TP'}×PP×${shardingLabel})${moeNote}`
        : `Attn/(TP×PP) + FFN/(${isMoE ? 'EP' : 'TP'}×PP)${moeNote}`,
      isRequired: true,
    },
    masterWeights: {
      name: 'Master Weights (FP32)',
      bytes: needsMasterWeights ? masterWeightsTotal : 0,
      bytesPerGPU: needsMasterWeights ? masterWeightsPerGPU : 0,
      precision: 'fp32' as const,
      description: 'FP32 copy for optimizer updates (required for mixed precision)',
      formula: needsMasterWeights 
        ? `P / (TP × PP × ${shardingLabel}) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${shardingValue}) × 4 bytes`
        : 'Not needed (training in FP32)',
      isRequired: needsMasterWeights,
    },
    optimizerMomentum: {
      name: 'Optimizer Momentum (m)',
      bytes: momentumTotal,
      bytesPerGPU: momentumPerGPU,
      precision: 'fp32' as const,
      description: 'Adam first moment estimate (always FP32)',
      formula: `P / (TP × PP × ${shardingLabel}) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${shardingValue}) × 4 bytes`,
      isRequired: true,
    },
    optimizerVariance: {
      name: 'Optimizer Variance (v)',
      bytes: varianceTotal,
      bytesPerGPU: variancePerGPU,
      precision: 'fp32' as const,
      description: 'Adam second moment estimate (always FP32)',
      formula: `P / (TP × PP × ${shardingLabel}) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${shardingValue}) × 4 bytes`,
      isRequired: true,
    },
    gradients: {
      name: 'Gradients',
      bytes: gradientsTotal,
      bytesPerGPU: gradientsPerGPU,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Gradient tensors in ${config.mixedPrecision.toUpperCase()}`,
      formula: zeroStage >= 2
        ? `P / (TP × PP × ${shardingLabel}) × ${modelPrecisionBytes} = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${shardingValue}) × ${modelPrecisionBytes}`
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
        ? `P / (TP × PP × ${shardingLabel}) × 4 = ${(P/1e9).toFixed(2)}B / (${TP} × ${PP} × ${shardingValue}) × 4 bytes`
        : 'Not needed (GA=1)',
      isRequired: needsGradAccumBuffer,
    },
    activations: {
      name: 'Layer Activations',
      bytes: activationsPerLayerTotal,
      bytesPerGPU: activationsPerLayerTotal,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Per-layer activations (${layersStored} layers × S/${contextParallel} tokens)`,
      formula: `${formatBytes(activationsPerLayer)}/layer × ${layersStored} layers = ${formatBytes(activationsPerLayerTotal)}`,
      isRequired: true,
      tensors: activationBreakdown.tensors.filter(t => t.stored).map(t => ({
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
    },
    lmHeadLogits: {
      name: 'LM Head Logits',
      bytes: lmHeadLogits.bytes,
      bytesPerGPU: lmHeadLogitsBytes,
      precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
      description: `Output logits [B, S/${contextParallel}, V=${model.vocabSize}] - CANNOT recompute`,
      formula: lmHeadLogits.formula,
      isRequired: true,
      tensors: [{
        name: 'LM Head Logits',
        shape: lmHeadLogits.shape,
        shapeValues: [batch.microBatchSize, Math.ceil(config.maxSeqLength / contextParallel), model.vocabSize],
        elementCount: batch.microBatchSize * Math.ceil(config.maxSeqLength / contextParallel) * model.vocabSize,
        bytes: lmHeadLogits.bytes,
        bytesPerGPU: lmHeadLogitsBytes,
        precision: config.mixedPrecision as 'fp32' | 'fp16' | 'bf16',
        formula: lmHeadLogits.formula,
        description: 'Stored for cross-entropy loss backward pass',
      }],
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
  const activationsBytes = activationsPerLayerTotal + lmHeadLogitsBytes; // Layer activations + LM Head
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
 * Communication time model with latency and bandwidth
 * 
 * Total time = alpha + beta * data_size
 * where:
 *   alpha = latency (fixed overhead per message)
 *   beta = 1/bandwidth (time per byte)
 * 
 * For collective operations, we also consider:
 *   - Number of hops in the network
 *   - NCCL/RCCL protocol overhead
 * 
 * Citation: NCCL documentation, LogP model (Culler et al., 1993)
 */
interface CommTimeParams {
  tensorBytes: number;
  numRanks: number;
  numCalls?: number;
  bandwidthGBs: number;
  latencyUs?: number;      // Per-hop latency in microseconds
  numHops?: number;        // Number of network hops
  ncclOverhead?: number;   // NCCL overhead factor (default: 1.1)
}

/**
 * Calculate Ring AllReduce time with latency
 * Formula: 2 × (n-1)/n × data/BW + latency × hops × 2
 * 
 * Citation: NCCL documentation
 */
function ringAllReduceTime(params: CommTimeParams): number {
  const {
    tensorBytes,
    numRanks,
    numCalls = 1,
    bandwidthGBs,
    latencyUs = 0,
    numHops = 1,
    ncclOverhead = 1.1,
  } = params;
  if (numRanks <= 1 || tensorBytes <= 0 || numCalls <= 0) return 0;
  
  const ringSteps = numRanks - 1;
  const factor = 2 * ringSteps / numRanks;
  const bandwidthTimeMs = numCalls * (tensorBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  
  // Ring all-reduce has 2*(n-1) startup steps.
  const latencyTimeMs = numCalls * (latencyUs * numHops * 2 * ringSteps) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
}

/**
 * Calculate AllGather time with latency
 */
function allGatherTime(params: CommTimeParams): number {
  const {
    tensorBytes,
    numRanks,
    numCalls = 1,
    bandwidthGBs,
    latencyUs = 0,
    numHops = 1,
    ncclOverhead = 1.1,
  } = params;
  if (numRanks <= 1 || tensorBytes <= 0 || numCalls <= 0) return 0;
  
  const ringSteps = numRanks - 1;
  const factor = ringSteps / numRanks;
  const bandwidthTimeMs = numCalls * (tensorBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  const latencyTimeMs = numCalls * (latencyUs * numHops * ringSteps) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
}

/**
 * Calculate ReduceScatter time with latency
 */
function reduceScatterTime(params: CommTimeParams): number {
  return allGatherTime(params);
}

/**
 * Calculate P2P time with latency
 */
function p2pTime(params: CommTimeParams): number {
  const {
    tensorBytes,
    numCalls = 1,
    bandwidthGBs,
    latencyUs = 0,
    numHops = 1,
  } = params;
  if (tensorBytes <= 0 || numCalls <= 0) return 0;
  
  const bandwidthTimeMs = numCalls * tensorBytes / (bandwidthGBs * 1e9) * 1000;
  const latencyTimeMs = numCalls * (latencyUs * numHops) / 1000;
  
  return bandwidthTimeMs + latencyTimeMs;
}

/**
 * Calculate All-to-All time (for Ulysses)
 * All-to-All sends N/P elements to each of P-1 peers
 * 
 * Citation: DeepSpeed Ulysses paper
 */
function allToAllTime(params: CommTimeParams): number {
  const {
    tensorBytes,
    numRanks,
    numCalls = 1,
    bandwidthGBs,
    latencyUs = 0,
    numHops = 1,
    ncclOverhead = 1.1,
  } = params;
  if (numRanks <= 1 || tensorBytes <= 0 || numCalls <= 0) return 0;
  
  const pairwiseSteps = numRanks - 1;
  const factor = pairwiseSteps / numRanks;
  const bandwidthTimeMs = numCalls * (tensorBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  
  const latencyTimeMs = numCalls * (latencyUs * numHops * pairwiseSteps) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
}

interface LocalShardedParameterCounts {
  attention: number;
  ffn: number;
  embedding: number;
  lmHead: number;
  layerNorm: number;
  total: number;
  transformerOnly: number;
}

interface TrainingMemoryTraffic {
  forwardBytes: number;
  backwardBytes: number;
}

function getMoELayerCounts(model: ModelConfig): { numDenseLayers: number; numMoeLayers: number } {
  if (model.ffnType !== 'moe') {
    return { numDenseLayers: model.numLayers, numMoeLayers: 0 };
  }

  const numDenseLayers = Math.max(0, Math.min(model.numLayers, model.numDenseLayers ?? 0));
  return {
    numDenseLayers,
    numMoeLayers: model.numLayers - numDenseLayers,
  };
}

function getDenseIntermediateSize(model: ModelConfig): number {
  return model.denseIntermediateSize ?? model.intermediateSize;
}

function getSharedExpertIntermediateSize(model: ModelConfig): number {
  return model.sharedExpertIntermediateSize
    ?? ((model.sharedExpertNum ?? 0) * model.intermediateSize);
}

function getLocalShardedParameterCounts(
  model: ModelConfig,
  parallelism: ParallelismConfig
): LocalShardedParameterCounts {
  const params = calculateModelParameters(model);
  const ffnShardDegree = model.ffnType === 'moe'
    ? parallelism.expertParallel
    : parallelism.tensorParallel;

  const attention = params.attention / (parallelism.tensorParallel * parallelism.pipelineParallel);
  const ffn = params.ffn / (ffnShardDegree * parallelism.pipelineParallel);
  const embedding = params.embedding / parallelism.pipelineParallel;
  const lmHead = params.lmHead / (parallelism.tensorParallel * parallelism.pipelineParallel);
  const layerNorm = params.layerNorm / parallelism.pipelineParallel;

  return {
    attention,
    ffn,
    embedding,
    lmHead,
    layerNorm,
    total: attention + ffn + embedding + lmHead + layerNorm,
    transformerOnly: attention + ffn + layerNorm,
  };
}

function zeroTraffic(): TrainingMemoryTraffic {
  return { forwardBytes: 0, backwardBytes: 0 };
}

function addTraffic(...items: TrainingMemoryTraffic[]): TrainingMemoryTraffic {
  return items.reduce(
    (sum, item) => ({
      forwardBytes: sum.forwardBytes + item.forwardBytes,
      backwardBytes: sum.backwardBytes + item.backwardBytes,
    }),
    zeroTraffic()
  );
}

function scaleTraffic(traffic: TrainingMemoryTraffic, factor: number): TrainingMemoryTraffic {
  return {
    forwardBytes: traffic.forwardBytes * factor,
    backwardBytes: traffic.backwardBytes * factor,
  };
}

function linearTrainingTraffic(
  tokenCount: number,
  inputDim: number,
  outputDim: number,
  elemBytes: number
): TrainingMemoryTraffic {
  const inputBytes = tokenCount * inputDim * elemBytes;
  const weightBytes = inputDim * outputDim * elemBytes;
  const outputBytes = tokenCount * outputDim * elemBytes;
  const forwardBytes = inputBytes + weightBytes + outputBytes;

  return {
    forwardBytes,
    backwardBytes: 2 * forwardBytes,
  };
}

function layerNormTrainingTraffic(elements: number, elemBytes: number): TrainingMemoryTraffic {
  const tensorBytes = elements * elemBytes;
  return {
    forwardBytes: 3 * tensorBytes,
    backwardBytes: 4 * tensorBytes,
  };
}

function residualTrainingTraffic(elements: number, elemBytes: number): TrainingMemoryTraffic {
  const tensorBytes = elements * elemBytes;
  return {
    forwardBytes: 3 * tensorBytes,
    backwardBytes: 3 * tensorBytes,
  };
}

function unaryActivationTrainingTraffic(elements: number, elemBytes: number): TrainingMemoryTraffic {
  const tensorBytes = elements * elemBytes;
  return {
    forwardBytes: 2 * tensorBytes,
    backwardBytes: 3 * tensorBytes,
  };
}

function multiplyTrainingTraffic(elements: number, elemBytes: number): TrainingMemoryTraffic {
  const tensorBytes = elements * elemBytes;
  return {
    forwardBytes: 3 * tensorBytes,
    backwardBytes: 4 * tensorBytes,
  };
}

function softmaxTrainingTraffic(elements: number, elemBytes: number): TrainingMemoryTraffic {
  const tensorBytes = elements * elemBytes;
  return {
    forwardBytes: 2 * tensorBytes,
    backwardBytes: 3 * tensorBytes,
  };
}

function flashAttentionTrainingTraffic(
  tokenCount: number,
  localQueryDim: number,
  localKvDim: number,
  localHeads: number,
  elemBytes: number
): TrainingMemoryTraffic {
  const qBytes = tokenCount * localQueryDim * elemBytes;
  const kBytes = tokenCount * localKvDim * elemBytes;
  const vBytes = tokenCount * localKvDim * elemBytes;
  const oBytes = tokenCount * localQueryDim * elemBytes;
  const statsBytes = tokenCount * localHeads * 4; // FP32 softmax stats

  return {
    forwardBytes: qBytes + kBytes + vBytes + oBytes + statsBytes,
    backwardBytes: 2 * (qBytes + kBytes + vBytes + oBytes) + statsBytes,
  };
}

function naiveAttentionTrainingTraffic(
  batchSize: number,
  seqLenPerRank: number,
  localHeads: number,
  tokenCount: number,
  localQueryDim: number,
  localKvDim: number,
  elemBytes: number
): TrainingMemoryTraffic {
  const qBytes = tokenCount * localQueryDim * elemBytes;
  const kBytes = tokenCount * localKvDim * elemBytes;
  const vBytes = tokenCount * localKvDim * elemBytes;
  const oBytes = tokenCount * localQueryDim * elemBytes;
  const scoreBytes = batchSize * localHeads * seqLenPerRank * seqLenPerRank * elemBytes;

  return {
    forwardBytes: qBytes + kBytes + vBytes + oBytes + 4 * scoreBytes,
    backwardBytes: 2 * (qBytes + kBytes + vBytes + oBytes) + 6 * scoreBytes,
  };
}

function estimateTrainingMemoryTraffic(
  model: ModelConfig,
  config: TrainingConfig
): TrainingMemoryTraffic {
  const { batch, parallelism, memoryOptimization } = config;
  const {
    tensorParallel,
    pipelineParallel,
    expertParallel,
    contextParallel,
    sequenceParallel,
  } = parallelism;
  const elemBytes = getPrecisionBytes(config.mixedPrecision);
  const seqLenPerRank = Math.ceil(config.maxSeqLength / contextParallel);
  const tokenCount = batch.microBatchSize * seqLenPerRank;
  const normTokenCount = sequenceParallel && tensorParallel > 1
    ? tokenCount / tensorParallel
    : tokenCount;
  const normElements = normTokenCount * model.hiddenSize;
  const localQueryDim = model.hiddenSize / tensorParallel;
  const localKvDim = (model.numKVHeads * model.headDim) / tensorParallel;
  const localHeads = model.numAttentionHeads / tensorParallel;
  const { numDenseLayers, numMoeLayers } = getMoELayerCounts(model);
  const denseLayersPerStage = numDenseLayers / pipelineParallel;
  const moeLayersPerStage = numMoeLayers / pipelineParallel;
  const denseIntermediate = getDenseIntermediateSize(model);
  const localDenseIntermediate = denseIntermediate / tensorParallel;
  const layerNormTraffic = layerNormTrainingTraffic(normElements, elemBytes);
  const residualTraffic = residualTrainingTraffic(normElements, elemBytes);
  const attentionCoreTraffic = memoryOptimization.flashAttention
    ? flashAttentionTrainingTraffic(
        tokenCount,
        localQueryDim,
        localKvDim,
        localHeads,
        elemBytes
      )
    : naiveAttentionTrainingTraffic(
        batch.microBatchSize,
        seqLenPerRank,
        localHeads,
        tokenCount,
        localQueryDim,
        localKvDim,
        elemBytes
      );

  const denseFfnTraffic = model.ffnType === 'gpt'
    ? addTraffic(
        linearTrainingTraffic(tokenCount, model.hiddenSize, localDenseIntermediate, elemBytes),
        unaryActivationTrainingTraffic(tokenCount * localDenseIntermediate, elemBytes),
        linearTrainingTraffic(tokenCount, localDenseIntermediate, model.hiddenSize, elemBytes)
      )
    : addTraffic(
        linearTrainingTraffic(tokenCount, model.hiddenSize, localDenseIntermediate, elemBytes),
        linearTrainingTraffic(tokenCount, model.hiddenSize, localDenseIntermediate, elemBytes),
        unaryActivationTrainingTraffic(tokenCount * localDenseIntermediate, elemBytes),
        multiplyTrainingTraffic(tokenCount * localDenseIntermediate, elemBytes),
        linearTrainingTraffic(tokenCount, localDenseIntermediate, model.hiddenSize, elemBytes)
      );

  const denseLayerTraffic = addTraffic(
    layerNormTraffic,
    linearTrainingTraffic(tokenCount, model.hiddenSize, localQueryDim, elemBytes),
    linearTrainingTraffic(tokenCount, model.hiddenSize, localKvDim, elemBytes),
    linearTrainingTraffic(tokenCount, model.hiddenSize, localKvDim, elemBytes),
    attentionCoreTraffic,
    linearTrainingTraffic(tokenCount, localQueryDim, model.hiddenSize, elemBytes),
    residualTraffic,
    layerNormTraffic,
    denseFfnTraffic,
    residualTraffic
  );

  let moeLayerTraffic = zeroTraffic();
  if (moeLayersPerStage > 0) {
    const nExperts = model.numExperts || 8;
    const topK = model.numExpertsPerToken || 2;
    const routedTokenCopiesPerRank = (tokenCount * topK) / expertParallel;
    const routedIntermediateElements = routedTokenCopiesPerRank * model.intermediateSize;
    const sharedIntermediate = getSharedExpertIntermediateSize(model);
    const sharedIntermediateElements = tokenCount * sharedIntermediate;

    const routerTraffic = addTraffic(
      linearTrainingTraffic(tokenCount, model.hiddenSize, nExperts, elemBytes),
      softmaxTrainingTraffic(tokenCount * nExperts, elemBytes)
    );

    const routedExpertTraffic = addTraffic(
      linearTrainingTraffic(routedTokenCopiesPerRank, model.hiddenSize, model.intermediateSize, elemBytes),
      linearTrainingTraffic(routedTokenCopiesPerRank, model.hiddenSize, model.intermediateSize, elemBytes),
      unaryActivationTrainingTraffic(routedIntermediateElements, elemBytes),
      multiplyTrainingTraffic(routedIntermediateElements, elemBytes),
      linearTrainingTraffic(routedTokenCopiesPerRank, model.intermediateSize, model.hiddenSize, elemBytes)
    );

    const sharedExpertTraffic = sharedIntermediate > 0
      ? addTraffic(
          linearTrainingTraffic(tokenCount, model.hiddenSize, sharedIntermediate, elemBytes),
          linearTrainingTraffic(tokenCount, model.hiddenSize, sharedIntermediate, elemBytes),
          unaryActivationTrainingTraffic(sharedIntermediateElements, elemBytes),
          multiplyTrainingTraffic(sharedIntermediateElements, elemBytes),
          linearTrainingTraffic(tokenCount, sharedIntermediate, model.hiddenSize, elemBytes)
        )
      : zeroTraffic();

    moeLayerTraffic = addTraffic(
      layerNormTraffic,
      linearTrainingTraffic(tokenCount, model.hiddenSize, localQueryDim, elemBytes),
      linearTrainingTraffic(tokenCount, model.hiddenSize, localKvDim, elemBytes),
      linearTrainingTraffic(tokenCount, model.hiddenSize, localKvDim, elemBytes),
      attentionCoreTraffic,
      linearTrainingTraffic(tokenCount, localQueryDim, model.hiddenSize, elemBytes),
      residualTraffic,
      layerNormTraffic,
      routerTraffic,
      routedExpertTraffic,
      sharedExpertTraffic,
      residualTraffic
    );
  }

  const lmHeadTraffic = addTraffic(
    layerNormTraffic,
    linearTrainingTraffic(tokenCount, model.hiddenSize, model.vocabSize / tensorParallel, elemBytes),
    softmaxTrainingTraffic(tokenCount * (model.vocabSize / tensorParallel), elemBytes)
  );

  return addTraffic(
    scaleTraffic(denseLayerTraffic, denseLayersPerStage),
    scaleTraffic(moeLayerTraffic, moeLayersPerStage),
    scaleTraffic(lmHeadTraffic, 1 / pipelineParallel)
  );
}


/**
 * Calculate communication breakdown
 */
export function calculateCommunication(
  model: ModelConfig,
  config: TrainingConfig,
  cluster: ClusterTopology
): CommunicationBreakdown {
  const { parallelism, batch } = config;
  const {
    dataParallel,
    tensorParallel,
    pipelineParallel,
    expertParallel,
    contextParallel,
    contextParallelType,
    zeroStage,
    effectiveDataParallel,
  } = parallelism;
  const gradientElemBytes = getPrecisionBytes(config.gradientPrecision);
  const activationElemBytes = getPrecisionBytes(config.mixedPrecision);
  const localParamCounts = getLocalShardedParameterCounts(model, parallelism);
  const { numMoeLayers } = getMoELayerCounts(model);
  
  // Effective DP for gradient sync and ZeRO sharding = DP × CP
  // CP is a special form of DP that splits sequence instead of batch
  // The gradient AllReduce must happen across ALL data parallel ranks (DP × CP)
  const dpForGradSync = effectiveDataParallel || (dataParallel * contextParallel);

  const seqLenPerGPU = Math.ceil(config.maxSeqLength / contextParallel);
  const activationTensorBytes =
    batch.microBatchSize * seqLenPerGPU * model.hiddenSize * activationElemBytes;
  const localGradientTensorBytes = localParamCounts.total * gradientElemBytes;
  const localParameterTensorBytes = localParamCounts.total * activationElemBytes;
  const localTransformerGradientBytes = localParamCounts.transformerOnly * gradientElemBytes;
  const localTransformerParameterBytes = localParamCounts.transformerOnly * activationElemBytes;
  const layersPerStage = model.numLayers / pipelineParallel;
  const perLayerGradientBytes = layersPerStage > 0
    ? localTransformerGradientBytes / layersPerStage
    : 0;
  const perLayerParameterBytes = layersPerStage > 0
    ? localTransformerParameterBytes / layersPerStage
    : 0;
  const numMicrobatches = batch.gradientAccumulation;

  let dataParallelVolume = 0;
  let zeroVolume = 0;
  let dataParallelTime = 0;
  let zeroTime = 0;

  // Data-parallel gradient synchronization / ZeRO sharding.
  if (dpForGradSync > 1) {
    if (zeroStage === 0 || zeroStage === 1) {
      dataParallelVolume = localGradientTensorBytes;
    } else if (zeroStage === 2) {
      dataParallelVolume = localGradientTensorBytes * numMicrobatches;
      zeroVolume = localParameterTensorBytes;
    } else {
      const gradCalls = layersPerStage * numMicrobatches;
      const paramCalls = 2 * layersPerStage * numMicrobatches;
      dataParallelVolume = perLayerGradientBytes * gradCalls;
      zeroVolume = perLayerParameterBytes * paramCalls;
    }
  }

  const tpCalls = tensorParallel > 1 ? 4 * layersPerStage * numMicrobatches : 0;
  const tensorParallelVolume = activationTensorBytes * tpCalls;

  const ppCalls = pipelineParallel > 1 ? 2 * numMicrobatches : 0;
  const pipelineParallelVolume = activationTensorBytes * ppCalls;

  let expertParallelVolume = 0;
  if (model.ffnType === 'moe' && expertParallel > 1) {
    const topK = model.numExpertsPerToken || 2;
    const moeLayersPerStage = numMoeLayers / pipelineParallel;
    const expertMessageBytes =
      batch.microBatchSize * seqLenPerGPU * model.hiddenSize * topK * activationElemBytes;
    const expertCalls = 4 * moeLayersPerStage * numMicrobatches;
    expertParallelVolume = expertMessageBytes * expertCalls;
  }

  let contextParallelVolume = 0;
  let contextParallelTime = 0;
  if (contextParallel > 1) {
    if (contextParallelType === 'ulysses') {
      const cpMessageBytes = activationTensorBytes;
      const cpCalls = 8 * layersPerStage * numMicrobatches;
      contextParallelVolume = cpMessageBytes * cpCalls;
    } else if (contextParallelType === 'ring') {
      const kvMessageBytes =
        batch.microBatchSize * seqLenPerGPU
        * ((model.numKVHeads * model.headDim * 2) / tensorParallel)
        * activationElemBytes;
      const cpCalls = 2 * (contextParallel - 1) * layersPerStage * numMicrobatches;
      contextParallelVolume = kvMessageBytes * cpCalls;
    } else {
      const effectiveKVHeads = model.numKVHeads / tensorParallel;
      const ulyssesDegree = Math.max(1, Math.min(contextParallel, effectiveKVHeads));
      const ringDegree = Math.max(1, Math.ceil(contextParallel / ulyssesDegree));
      const ulyssesMessageBytes =
        batch.microBatchSize * Math.ceil(config.maxSeqLength / ulyssesDegree) * model.hiddenSize * activationElemBytes;
      const ringMessageBytes =
        batch.microBatchSize
        * Math.ceil(config.maxSeqLength / contextParallel)
        * ((model.numKVHeads * model.headDim * 2) / (tensorParallel * ulyssesDegree))
        * activationElemBytes;
      const ulyssesCalls = 8 * layersPerStage * numMicrobatches;
      const ringCalls = 2 * Math.max(0, ringDegree - 1) * layersPerStage * numMicrobatches;

      contextParallelVolume = ulyssesMessageBytes * ulyssesCalls + ringMessageBytes * ringCalls;
    }
  }
  
  const totalVolume = dataParallelVolume + tensorParallelVolume + 
    pipelineParallelVolume + expertParallelVolume + contextParallelVolume + zeroVolume;
  
  // Calculate times with latency modeling
  const intraNodeBW = cluster.intraNodeBandwidth;
  const interNodeBW = cluster.interNodeBandwidth;
  // Get network latency configuration (uses defaults if not specified)
  const networkLatency = getNetworkLatencyConfig(cluster);
  const commAssumptions = getAssumptions(config);
  
  // Helper to get bandwidth, latency, and hops based on whether comm crosses node boundary
  const getCommParams = (
    parallelSize: number,
    tensorBytes: number,
    numRanks: number,
    numCalls: number = 1
  ) => {
    const isIntraNode = parallelSize <= cluster.gpusPerNode;
    return {
      tensorBytes,
      numRanks,
      numCalls,
      bandwidthGBs: isIntraNode ? intraNodeBW : interNodeBW,
      latencyUs: isIntraNode ? networkLatency.intraNodeLatencyUs : networkLatency.interNodeLatencyUs,
      numHops: isIntraNode ? networkLatency.intraNodeHops : networkLatency.interNodeHops,
      ncclOverhead: commAssumptions.ncclOverheadFactor,
    };
  };
  
  // Recompute collective times with per-call message sizes so latency scales with call count.
  if (dpForGradSync > 1) {
    if (zeroStage === 0 || zeroStage === 1) {
      dataParallelTime = ringAllReduceTime(
        getCommParams(dpForGradSync, localGradientTensorBytes, dpForGradSync, 1)
      );
    } else if (zeroStage === 2) {
      dataParallelTime = reduceScatterTime(
        getCommParams(dpForGradSync, localGradientTensorBytes, dpForGradSync, numMicrobatches)
      );
      zeroTime = allGatherTime(
        getCommParams(dpForGradSync, localParameterTensorBytes, dpForGradSync, 1)
      );
    } else {
      const gradCalls = layersPerStage * numMicrobatches;
      const paramCalls = 2 * layersPerStage * numMicrobatches;
      dataParallelTime = reduceScatterTime(
        getCommParams(dpForGradSync, perLayerGradientBytes, dpForGradSync, gradCalls)
      );
      zeroTime = allGatherTime(
        getCommParams(dpForGradSync, perLayerParameterBytes, dpForGradSync, paramCalls)
      );
    }
  }

  const tensorParallelTime = tensorParallel > 1
    ? ringAllReduceTime(
        getCommParams(tensorParallel, activationTensorBytes, tensorParallel, tpCalls)
      )
    : 0;

  const pipelineParallelTime = pipelineParallel > 1
    ? p2pTime(
        getCommParams(pipelineParallel, activationTensorBytes, 2, ppCalls)
      )
    : 0;

  let expertParallelTime = 0;
  if (expertParallelVolume > 0) {
    const topK = model.numExpertsPerToken || 2;
    const moeLayersPerStage = numMoeLayers / pipelineParallel;
    const expertMessageBytes =
      batch.microBatchSize * seqLenPerGPU * model.hiddenSize * topK * activationElemBytes;
    const expertCalls = 4 * moeLayersPerStage * numMicrobatches;
    expertParallelTime = allToAllTime(
      getCommParams(expertParallel, expertMessageBytes, expertParallel, expertCalls)
    );
  }

  if (contextParallel > 1) {
    if (contextParallelType === 'ulysses') {
      const cpMessageBytes = activationTensorBytes;
      const cpCalls = 8 * layersPerStage * numMicrobatches;
      contextParallelTime = allToAllTime(
        getCommParams(contextParallel, cpMessageBytes, contextParallel, cpCalls)
      );
    } else if (contextParallelType === 'ring') {
      const kvMessageBytes =
        batch.microBatchSize * seqLenPerGPU
        * ((model.numKVHeads * model.headDim * 2) / tensorParallel)
        * activationElemBytes;
      const cpCalls = 2 * (contextParallel - 1) * layersPerStage * numMicrobatches;
      contextParallelTime = p2pTime(
        getCommParams(contextParallel, kvMessageBytes, 2, cpCalls)
      );
    } else {
      const effectiveKVHeads = model.numKVHeads / tensorParallel;
      const ulyssesDegree = Math.max(1, Math.min(contextParallel, effectiveKVHeads));
      const ringDegree = Math.max(1, Math.ceil(contextParallel / ulyssesDegree));
      const ulyssesMessageBytes =
        batch.microBatchSize * Math.ceil(config.maxSeqLength / ulyssesDegree) * model.hiddenSize * activationElemBytes;
      const ringMessageBytes =
        batch.microBatchSize
        * Math.ceil(config.maxSeqLength / contextParallel)
        * ((model.numKVHeads * model.headDim * 2) / (tensorParallel * ulyssesDegree))
        * activationElemBytes;
      const ulyssesCalls = 8 * layersPerStage * numMicrobatches;
      const ringCalls = 2 * Math.max(0, ringDegree - 1) * layersPerStage * numMicrobatches;

      contextParallelTime =
        allToAllTime(
          getCommParams(ulyssesDegree, ulyssesMessageBytes, ulyssesDegree, ulyssesCalls)
        )
        + p2pTime(
          getCommParams(ringDegree, ringMessageBytes, 2, ringCalls)
        );
    }
  }
  
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
  const { tensorParallel, pipelineParallel, contextParallel } = config.parallelism;
  
  // Compute FLOPs per micro-batch using a Megatron-style Transformer/MoE estimator.
  // This keeps routed MoE compute proportional to active experts (top-k), not total experts.
  const totalFlops = estimateTrainingFlops(
    model,
    config.batch.microBatchSize,
    config.maxSeqLength
  );
  const forwardFlops = totalFlops / 3;
  const backwardFlops = totalFlops - forwardFlops;
  
  // With TP/PP/CP, each GPU only computes a fraction of the model.
  const modelParallelDegree = tensorParallel * pipelineParallel * contextParallel;
  const forwardFlopsPerGPU = forwardFlops / modelParallelDegree;
  const backwardFlopsPerGPU = backwardFlops / modelParallelDegree;
  
  // Recomputation overhead
  // Citation: "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al., 2023)
  const stepAssumptions = getAssumptions(config);
  let recomputationMultiplier = 1;
  if (config.memoryOptimization.recomputation === 'full') {
    recomputationMultiplier = 1 + stepAssumptions.recomputationFullOverhead; // ~33% overhead
  } else if (config.memoryOptimization.recomputation === 'selective') {
    recomputationMultiplier = 1 + stepAssumptions.recomputationSelectiveOverhead; // ~5% overhead
  } else if (config.memoryOptimization.recomputation === 'block') {
    recomputationMultiplier = 1 + stepAssumptions.recomputationBlockOverhead; // ~15% overhead
  }
  
  // Compute time per micro-batch per GPU using a roofline-style model with
  // explicit training HBM traffic estimates instead of a single 6N approximation.
  const effectiveCompute =
    cluster.gpuComputeTFLOPS * 1e12 * stepAssumptions.kernelEfficiency;
  const effectiveBandwidth =
    cluster.gpuMemoryBandwidthTBs * 1e12 * stepAssumptions.memoryBandwidthEfficiency;

  const memoryTraffic = estimateTrainingMemoryTraffic(model, config);
  const forwardMemoryBytes = memoryTraffic.forwardBytes;
  const backwardMemoryBytes = memoryTraffic.backwardBytes;
  
  // Compute-bound time (FLOPs / peak compute)
  const forwardComputeTime = forwardFlopsPerGPU / effectiveCompute * 1000; // ms
  const backwardComputeTime = backwardFlopsPerGPU / effectiveCompute * 1000;
  
  // Memory-bound time (bytes / peak bandwidth)
  const forwardMemoryTime = forwardMemoryBytes / effectiveBandwidth * 1000; // ms
  const backwardMemoryTime = backwardMemoryBytes / effectiveBandwidth * 1000;
  
  // Roofline: actual time = max(compute time, memory time)
  // In practice, there's some overlap, so we use a weighted combination
  // For GEMM-dominated workloads, it's usually compute-bound
  // The arithmetic intensity determines which bound applies
  const forwardAI = forwardFlopsPerGPU / forwardMemoryBytes;
  const backwardAI = backwardFlopsPerGPU / backwardMemoryBytes;
  const ridgePoint = effectiveCompute / effectiveBandwidth; // FLOPs/byte at ridge point
  
  // Below ridge point: memory bound, above: compute bound
  const forwardTime = forwardAI < ridgePoint 
    ? forwardMemoryTime  // Memory bound
    : forwardComputeTime; // Compute bound
  const backwardTime = backwardAI < ridgePoint
    ? backwardMemoryTime
    : backwardComputeTime;
  
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
  // Citation: Megatron-LM achieves 70-80% overlap with async AllReduce
  const assumptions = getAssumptions(config);
  const overlapFactor = assumptions.communicationOverlapFactor;
  const effectiveCommTime = communicationBreakdown.dataParallelTime + 
    (communicationBreakdown.tensorParallelTime + 
     communicationBreakdown.pipelineParallelTime + 
     communicationBreakdown.expertParallelTime +
     communicationBreakdown.contextParallelTime) * (1 - overlapFactor) +
    communicationBreakdown.zeroTime;
  
  const totalComputeTime = computeTime * config.batch.gradientAccumulation;
  const timePerStep = totalComputeTime + effectiveCommTime + communicationBreakdown.bubbleTime;
  
  // Efficiency metrics
  // MFU (Model FLOPs Utilization) = Useful FLOPs / Peak FLOPs
  // Citation: "PaLM: Scaling Language Modeling with Pathways" (Chowdhery et al., 2022)
  // 
  // Useful FLOPs per step are estimated directly from the current training batch,
  // so MoE uses active-expert compute rather than total-expert params.
  // 
  // Peak FLOPs = totalGPUs * gpuPeakFLOPS * timePerStep
  const theoreticalPeakFlops = cluster.gpuComputeTFLOPS * 1e12 * 
    (timePerStep / 1000) * config.parallelism.totalGPUs;
  
  // MFU = Useful work / Peak capacity
  const totalUsefulFlops = estimateTrainingFlops(
    model,
    config.batch.globalBatchSize,
    config.maxSeqLength
  );
  const mfu = totalUsefulFlops / theoreticalPeakFlops;
  
  // HFU includes recomputation overhead in "useful" work
  const hfu = (totalUsefulFlops * recomputationMultiplier) / theoreticalPeakFlops;
  
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

export interface TrainingTimeEstimate {
  totalTokens: number;
  tokensPerStep: number;
  totalSteps: number;
  tokensPerSecond: number;
  samplesPerSecond: number;
  timeSeconds: number;
  gpuHours: number;
  gpuDays: number;
  activeGPUs: number;
}

/**
 * Estimate end-to-end pretraining time from a target MFU and token budget.
 *
 * Assumptions:
 * - Pretraining FLOPs/token come from the explicit Transformer/MoE estimator
 * - Sequence length is fixed for the run, so steps = total_tokens / (GBS * seq_len)
 */
export function estimateTrainingTime(
  model: ModelConfig,
  config: TrainingConfig,
  cluster: ClusterTopology,
  targetMfu: number,
  totalTokens: number
): TrainingTimeEstimate | null {
  const activeGPUs = config.parallelism.totalGPUs;
  const tokensPerStep = config.batch.globalBatchSize * config.maxSeqLength;

  if (!Number.isFinite(targetMfu) || targetMfu <= 0 || targetMfu > 1) {
    return null;
  }
  if (!Number.isFinite(totalTokens) || totalTokens <= 0) {
    return null;
  }
  if (!Number.isFinite(activeGPUs) || activeGPUs <= 0) {
    return null;
  }
  if (!Number.isFinite(tokensPerStep) || tokensPerStep <= 0) {
    return null;
  }

  const flopsPerToken = estimateTrainingFlopsPerToken(model, config.maxSeqLength);
  if (!Number.isFinite(flopsPerToken) || flopsPerToken <= 0) {
    return null;
  }

  const totalTrainFlops = flopsPerToken * totalTokens;
  const effectiveSystemFlopsPerSecond =
    cluster.gpuComputeTFLOPS * 1e12 * activeGPUs * targetMfu;

  if (!Number.isFinite(effectiveSystemFlopsPerSecond) || effectiveSystemFlopsPerSecond <= 0) {
    return null;
  }

  const timeSeconds = totalTrainFlops / effectiveSystemFlopsPerSecond;
  const tokensPerSecond = totalTokens / timeSeconds;
  const samplesPerSecond = tokensPerSecond / config.maxSeqLength;
  const totalSteps = totalTokens / tokensPerStep;
  const gpuHours = (timeSeconds * activeGPUs) / 3600;
  const gpuDays = gpuHours / 24;

  return {
    totalTokens,
    tokensPerStep,
    totalSteps,
    tokensPerSecond,
    samplesPerSecond,
    timeSeconds,
    gpuHours,
    gpuDays,
    activeGPUs,
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
