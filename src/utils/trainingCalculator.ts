/**
 * Training Calculator
 * 
 * Calculates memory, communication, and compute requirements for distributed LLM training.
 * Implements formulas from Megatron-LM, ZeRO, and related research.
 * 
 * Formula References:
 * - Forward FLOPs ≈ 2N×T (OpenAI Scaling Laws, Kaplan et al., 2020)
 * - Backward FLOPs ≈ 2× Forward (Megatron-LM, NVIDIA)
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
import { calculateModelParameters } from './calculator';
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
  dataBytes: number;
  numRanks: number;
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
  const { dataBytes, numRanks, bandwidthGBs, latencyUs = 0, numHops = 1, ncclOverhead = 1.1 } = params;
  if (numRanks <= 1) return 0;
  
  // Bandwidth component: reduce-scatter + all-gather
  const factor = 2 * (numRanks - 1) / numRanks;
  const bandwidthTimeMs = (dataBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  
  // Latency component: 2 phases × hops
  const latencyTimeMs = (latencyUs * numHops * 2) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
}

/**
 * Calculate AllGather time with latency
 */
function allGatherTime(params: CommTimeParams): number {
  const { dataBytes, numRanks, bandwidthGBs, latencyUs = 0, numHops = 1, ncclOverhead = 1.1 } = params;
  if (numRanks <= 1) return 0;
  
  const factor = (numRanks - 1) / numRanks;
  const bandwidthTimeMs = (dataBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  const latencyTimeMs = (latencyUs * numHops) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
}

/**
 * Calculate P2P time with latency
 */
function p2pTime(params: CommTimeParams): number {
  const { dataBytes, bandwidthGBs, latencyUs = 0, numHops = 1 } = params;
  
  const bandwidthTimeMs = dataBytes / (bandwidthGBs * 1e9) * 1000;
  const latencyTimeMs = (latencyUs * numHops) / 1000;
  
  return bandwidthTimeMs + latencyTimeMs;
}

/**
 * Calculate All-to-All time (for Ulysses)
 * All-to-All sends N/P elements to each of P-1 peers
 * 
 * Citation: DeepSpeed Ulysses paper
 */
function allToAllTime(params: CommTimeParams): number {
  const { dataBytes, numRanks, bandwidthGBs, latencyUs = 0, numHops = 1, ncclOverhead = 1.1 } = params;
  if (numRanks <= 1) return 0;
  
  // Each rank sends (P-1)/P of its data
  const factor = (numRanks - 1) / numRanks;
  const bandwidthTimeMs = (dataBytes * factor) / (bandwidthGBs * 1e9) * 1000;
  
  // All-to-All has higher latency due to many concurrent messages
  const latencyTimeMs = (latencyUs * numHops * (numRanks - 1)) / 1000;
  
  return (bandwidthTimeMs + latencyTimeMs) * ncclOverhead;
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
          contextParallel, contextParallelType, zeroStage, effectiveDataParallel } = parallelism;
  const bytes = getPrecisionBytes(config.gradientPrecision);
  
  // Effective DP for gradient sync and ZeRO sharding = DP × CP
  // CP is a special form of DP that splits sequence instead of batch
  // The gradient AllReduce must happen across ALL data parallel ranks (DP × CP)
  const dpForGradSync = effectiveDataParallel || (dataParallel * contextParallel);
  
  // Effective parameters per DP rank (after TP/PP sharding)
  const paramsPerRank = params.total / (tensorParallel * pipelineParallel);
  
  // Effective sequence length per GPU with Context Parallel
  const seqLenPerGPU = config.maxSeqLength / contextParallel;
  
  // Data Parallel communication (gradient sync across DP × CP ranks)
  // Even with CP, gradients must be synchronized because each CP rank
  // processes different sequence chunks but needs the same model update
  //
  // ZeRO-2 with Gradient Accumulation (GA):
  // To avoid keeping full gradient buffer (which degrades to ZeRO-1 memory),
  // Megatron-LM/DeepSpeed perform ReduceScatter EVERY micro-batch.
  // This increases communication but maintains ZeRO-2 memory benefits.
  // Reference: DeepSpeed ZeRO-2 implementation, Megatron-LM distributed optimizer
  let dataParallelVolume = 0;
  if (dpForGradSync > 1) {
    if (zeroStage === 0) {
      // DDP: AllReduce gradients once per step (after all GA micro-batches)
      dataParallelVolume = 2 * paramsPerRank * bytes;
    } else if (zeroStage === 1) {
      // ZeRO-1: AllReduce gradients once per step
      dataParallelVolume = 2 * paramsPerRank * bytes;
    } else if (zeroStage === 2) {
      // ZeRO-2: ReduceScatter + AllGather per MICRO-BATCH
      // To avoid keeping full gradient buffer, ReduceScatter after each micro-batch
      // Then AllGather at the end for optimizer step
      // Volume per micro-batch: ReduceScatter = params/DP (send) + params/DP (recv) = 2*P/DP per micro-batch
      // At step end: AllGather = 2*P/DP
      // Total = GA * ReduceScatter + AllGather = GA * 2*P/DP + 2*P/DP = 2*P/DP * (GA + 1)
      // But simplified: ReduceScatter every micro-batch = 2*P (ring-based) * GA times
      const reduceScatterPerMicroBatch = 2 * paramsPerRank * bytes / dpForGradSync * (dpForGradSync - 1);
      const allGatherAtEnd = 2 * paramsPerRank * bytes / dpForGradSync * (dpForGradSync - 1);
      dataParallelVolume = reduceScatterPerMicroBatch * batch.gradientAccumulation + allGatherAtEnd;
    } else {
      // ZeRO-3: AllGather params handled separately in zeroVolume
      // Gradients: ReduceScatter per micro-batch
      const reduceScatterPerMicroBatch = 2 * paramsPerRank * bytes / dpForGradSync * (dpForGradSync - 1);
      dataParallelVolume = reduceScatterPerMicroBatch * batch.gradientAccumulation;
    }
  }
  
  // ZeRO-3 additional parameter gather (across DP × CP)
  let zeroVolume = 0;
  if (zeroStage === 3 && dpForGradSync > 1) {
    // AllGather params before each forward and backward pass
    // Per layer: AllGather params = 2 * layer_params / DP * (DP-1)
    // For all layers across all GA micro-batches
    const layerParams = paramsPerRank / (model.numLayers / pipelineParallel);
    const allGatherPerLayer = 2 * layerParams * bytes / dpForGradSync * (dpForGradSync - 1);
    // Forward + Backward for each micro-batch
    zeroVolume = 2 * allGatherPerLayer * (model.numLayers / pipelineParallel) * batch.gradientAccumulation;
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
  
  // Calculate times with latency modeling
  const intraNodeBW = cluster.intraNodeBandwidth;
  const interNodeBW = cluster.interNodeBandwidth;
  // Get network latency configuration (uses defaults if not specified)
  const networkLatency = getNetworkLatencyConfig(cluster);
  const commAssumptions = getAssumptions(config);
  
  // Helper to get bandwidth, latency, and hops based on whether comm crosses node boundary
  const getCommParams = (parallelSize: number, dataBytes: number, numRanks: number) => {
    const isIntraNode = parallelSize <= cluster.gpusPerNode;
    return {
      dataBytes,
      numRanks,
      bandwidthGBs: isIntraNode ? intraNodeBW : interNodeBW,
      latencyUs: isIntraNode ? networkLatency.intraNodeLatencyUs : networkLatency.interNodeLatencyUs,
      numHops: isIntraNode ? networkLatency.intraNodeHops : networkLatency.interNodeHops,
      ncclOverhead: commAssumptions.ncclOverheadFactor,
    };
  };
  
  // DP gradient sync across DP × CP ranks (typically crosses nodes)
  const dataParallelTime = dpForGradSync > 1 ? ringAllReduceTime(
    getCommParams(dpForGradSync, dataParallelVolume, dpForGradSync)
  ) : 0;
  
  // TP should be within node
  const tensorParallelTime = ringAllReduceTime(
    getCommParams(tensorParallel, tensorParallelVolume / batch.gradientAccumulation, tensorParallel)
  ) * batch.gradientAccumulation;
  
  // PP communication
  const pipelineParallelTime = p2pTime(
    getCommParams(pipelineParallel, pipelineParallelVolume, pipelineParallel)
  );
  
  // EP communication
  let expertParallelTime = 0;
  if (expertParallelVolume > 0) {
    expertParallelTime = allToAllTime(
      getCommParams(expertParallel, expertParallelVolume, expertParallel)
    );
  }
  
  // CP communication
  // Ulysses benefits from NVLink (all-to-all is latency sensitive)
  // Ring can work across nodes (P2P is bandwidth bound)
  let contextParallelTime = 0;
  if (contextParallel > 1) {
    if (contextParallelType === 'ulysses') {
      // All-to-all time
      contextParallelTime = allToAllTime(
        getCommParams(contextParallel, contextParallelVolume, contextParallel)
      );
    } else {
      // Ring/Hybrid: P2P time
      contextParallelTime = p2pTime(
        getCommParams(contextParallel, contextParallelVolume, contextParallel)
      );
    }
  }
  
  // ZeRO-3 time (AllGather across DP × CP)
  const zeroTime = zeroVolume > 0 ? 
    allGatherTime(getCommParams(dpForGradSync, zeroVolume / 2, dpForGradSync)) * 2 : 0;
  
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
  const { tensorParallel, pipelineParallel, dataParallel, contextParallel } = config.parallelism;
  
  // Compute FLOPs per micro-batch (total model FLOPs, before parallelism splitting)
  // Citation: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
  // Forward: ~2N FLOPs per token, Backward: ~4N FLOPs per token
  const tokens = config.batch.microBatchSize * config.maxSeqLength;
  const forwardFlops = 2 * params.total * tokens; // ~2N per token
  const backwardFlops = 2 * forwardFlops; // Backward is ~2x forward
  const totalFlops = forwardFlops + backwardFlops; // 6N per token
  
  // With TP/PP, each GPU only computes a fraction of the model
  // TP splits each layer across GPUs, PP splits layers across stages
  const modelParallelDegree = tensorParallel * pipelineParallel;
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
  
  // Compute time per micro-batch per GPU using Roofline Model
  // The actual time is max(compute-bound time, memory-bound time)
  // 
  // For training, memory access per forward pass:
  // - Read weights: P/(TP*PP) bytes
  // - Read activations: B * S * H * L bytes (per layer)
  // - Write activations: B * S * H * L bytes (per layer)
  // For backward, memory access is roughly 2x forward
  const effectiveCompute = cluster.gpuComputeTFLOPS * 1e12; // FLOPS
  const effectiveBandwidth = cluster.gpuMemoryBandwidthTBs * 1e12; // bytes/s
  
  // Params per GPU after TP/PP
  const paramsPerGPU = params.total / modelParallelDegree;
  const paramBytes = paramsPerGPU * getPrecisionBytes(config.mixedPrecision);
  
  // Activations memory access per micro-batch (rough estimate)
  // Each layer: read input (B*S*H), compute, write output (B*S*H)
  // With CP, sequence length per GPU is S/CP
  const seqLenPerGPU = config.maxSeqLength / contextParallel;
  const activationBytesPerLayer = 2 * config.batch.microBatchSize * seqLenPerGPU * 
    model.hiddenSize * getPrecisionBytes(config.mixedPrecision);
  const layersPerGPU = model.numLayers / pipelineParallel;
  
  // Forward memory access: read weights once + read/write activations per layer
  const forwardMemoryBytes = paramBytes + activationBytesPerLayer * layersPerGPU;
  // Backward memory access: ~2x forward (read gradients, write gradients, read weights, read activations)
  const backwardMemoryBytes = forwardMemoryBytes * 2;
  
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
  // Useful FLOPs per step = 6 * N * globalBatch * S
  // Where: N = total params, globalBatch = microBatch * DP * gradAccum, S = seqLen
  // Note: DP ranks each process different data, so total useful FLOPs = 6N * globalBatch * S
  // 
  // Peak FLOPs = totalGPUs * gpuPeakFLOPS * timePerStep
  const effectiveDP = dataParallel * contextParallel; // CP is part of DP dimension
  const theoreticalPeakFlops = cluster.gpuComputeTFLOPS * 1e12 * 
    (timePerStep / 1000) * config.parallelism.totalGPUs;
  
  // MFU = Useful work / Peak capacity
  // For training: useful work = 6 * N * globalBatch * S per step
  const globalBatchSize = config.batch.microBatchSize * effectiveDP * config.batch.gradientAccumulation;
  const totalUsefulFlops = 6 * params.total * globalBatchSize * config.maxSeqLength;
  const mfu = totalUsefulFlops / theoreticalPeakFlops;
  
  // HFU includes recomputation overhead in "useful" work
  const hfu = mfu * recomputationMultiplier;
  
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
