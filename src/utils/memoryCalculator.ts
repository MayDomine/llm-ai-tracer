/**
 * GPU Memory Calculator for LLM Models
 * 
 * Calculates memory requirements for:
 * - Model weights
 * - KV Cache (inference)
 * - Activations
 * - Gradients (training)
 * - Optimizer states (training)
 * 
 * References:
 * - vLLM memory modeling
 * - Megatron-LM memory estimation
 */

import type { 
  ModelConfig, 
  HardwareConfig, 
  InferenceConfig, 
  MemoryAnalysis,
  MemoryBreakdown,
  ParallelConfig
} from '../types/model';
import { DTYPE_BYTES, calculateModelParameters } from './calculator';

// 颜色映射
const MEMORY_COLORS = {
  modelWeights: '#6366f1',      // indigo
  kvCache: '#06b6d4',           // cyan
  activations: '#8b5cf6',       // purple
  gradients: '#f59e0b',         // amber
  optimizerStates: '#ef4444',   // red
  frameworkOverhead: '#6b7280', // gray
};

/**
 * 计算模型权重内存
 */
export function calculateModelWeightsMemory(
  modelConfig: ModelConfig,
  dtype: InferenceConfig['dtype'],
  parallelConfig?: ParallelConfig
): number {
  const params = calculateModelParameters(modelConfig);
  const bytesPerElement = DTYPE_BYTES[dtype];
  
  let totalParams = params.total;
  
  // 考虑张量并行: 权重被分片到多个GPU
  if (parallelConfig && parallelConfig.tensorParallel > 1) {
    // Attention和FFN权重被TP分片
    const shardedParams = params.attention + params.ffn;
    const unshardedParams = params.embedding + params.lmHead + params.layerNorm;
    totalParams = shardedParams / parallelConfig.tensorParallel + unshardedParams;
  }
  
  // 考虑流水线并行: 只存储部分层
  if (parallelConfig && parallelConfig.pipelineParallel > 1) {
    const layersPerStage = modelConfig.numLayers / parallelConfig.pipelineParallel;
    const perLayerParams = (params.attention + params.ffn + params.layerNorm) / modelConfig.numLayers;
    const stageParams = perLayerParams * layersPerStage;
    // 每个stage只有部分层,但embedding和lm_head分布在首尾stage
    totalParams = stageParams + (params.embedding + params.lmHead) / parallelConfig.pipelineParallel;
  }
  
  return totalParams * bytesPerElement;
}

/**
 * 计算KV Cache内存
 * KV Cache = 2 × batch × seq_len × num_layers × num_kv_heads × head_dim × bytes
 */
export function calculateKVCacheMemory(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig?: ParallelConfig
): number {
  const { numLayers, numKVHeads, headDim } = modelConfig;
  const { batchSize, seqLen, kvCacheLen, mode, dtype } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[dtype];
  
  // 根据模式确定KV Cache长度
  const cacheSeqLen = mode === 'decode' ? kvCacheLen + 1 : seqLen;
  
  // 2 for K and V
  let kvCacheSize = 2 * batchSize * cacheSeqLen * numLayers * numKVHeads * headDim * bytesPerElement;
  
  // 考虑张量并行: KV heads可能被分片
  if (parallelConfig && parallelConfig.tensorParallel > 1) {
    kvCacheSize = kvCacheSize / parallelConfig.tensorParallel;
  }
  
  // 考虑流水线并行: 每个stage只有部分层
  if (parallelConfig && parallelConfig.pipelineParallel > 1) {
    kvCacheSize = kvCacheSize / parallelConfig.pipelineParallel;
  }
  
  return kvCacheSize;
}

/**
 * 计算激活内存 (前向传播中的中间张量)
 * 
 * Per layer activations (approximate):
 * - Input to layer: batch × seq × hidden
 * - QKV projections: batch × seq × (3 × hidden) for MHA
 * - Attention scores: batch × heads × seq × seq
 * - Attention output: batch × seq × hidden
 * - FFN intermediate: batch × seq × intermediate_size
 * - Various residual connections
 * 
 * Total per layer ≈ batch × seq × hidden × (10-14) depending on implementation
 */
export function calculateActivationsMemory(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  useGradientCheckpointing: boolean = false,
  parallelConfig?: ParallelConfig
): number {
  const { hiddenSize, numLayers, numAttentionHeads, intermediateSize } = modelConfig;
  const { batchSize, seqLen, mode, dtype } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[dtype];
  
  // 实际序列长度
  const effectiveSeqLen = mode === 'decode' ? 1 : seqLen;
  
  // 基础激活内存因子 (经验值)
  // 详细breakdown:
  // - Layer input: 1 × hidden
  // - Q, K, V: 3 × hidden
  // - Attention scores: heads × seq (对于长序列这个可能很大)
  // - Attention output: 1 × hidden
  // - FFN up: 1 × intermediate
  // - FFN gate (gated): 1 × intermediate
  // - FFN down input: 1 × intermediate
  // - Residuals: 2 × hidden
  // 总计约 10-14 × hidden (不含attention scores)
  
  const activationFactor = 12; // 保守估计
  
  let perLayerActivations = batchSize * effectiveSeqLen * hiddenSize * activationFactor * bytesPerElement;
  
  // 加上attention scores内存 (可能很大)
  // 对于prefill/training: batch × heads × seq × seq
  if (mode !== 'decode') {
    const attentionScoresMemory = batchSize * numAttentionHeads * seqLen * seqLen * bytesPerElement;
    perLayerActivations += attentionScoresMemory;
  }
  
  // 加上FFN intermediate内存
  const ffnIntermediateMemory = batchSize * effectiveSeqLen * intermediateSize * 2 * bytesPerElement;
  perLayerActivations += ffnIntermediateMemory;
  
  let totalActivations: number;
  
  if (useGradientCheckpointing) {
    // 使用gradient checkpointing时,只保存每个checkpoint边界的激活
    // 通常每层checkpoint一次,所以只需要存储 ~2层 的激活
    totalActivations = perLayerActivations * 2;
  } else if (inferenceConfig.mode === 'training') {
    // 训练时需要保存所有层的激活用于反向传播
    totalActivations = perLayerActivations * numLayers;
  } else {
    // 推理时可以逐层处理,只需要1层的激活
    totalActivations = perLayerActivations;
  }
  
  // 考虑张量并行: 激活被分片
  if (parallelConfig && parallelConfig.tensorParallel > 1) {
    // 部分激活被分片 (FFN intermediate, attention heads)
    totalActivations = totalActivations * 0.7 + totalActivations * 0.3 / parallelConfig.tensorParallel;
  }
  
  // 考虑序列并行
  if (parallelConfig?.sequenceParallel && parallelConfig.tensorParallel > 1) {
    // 序列并行进一步减少激活内存
    totalActivations = totalActivations / parallelConfig.tensorParallel;
  }
  
  return totalActivations;
}

/**
 * 计算梯度内存 (训练模式)
 * 梯度大小 = 参数数量 × 字节数
 */
export function calculateGradientsMemory(
  modelConfig: ModelConfig,
  dtype: InferenceConfig['dtype'],
  parallelConfig?: ParallelConfig
): number {
  // 梯度与权重大小相同
  return calculateModelWeightsMemory(modelConfig, dtype, parallelConfig);
}

/**
 * 计算优化器状态内存 (训练模式)
 * Adam/AdamW: 2 × params × 4 bytes (FP32 momentum + variance)
 */
export function calculateOptimizerStatesMemory(
  modelConfig: ModelConfig,
  optimizerType: 'adam' | 'sgd' | 'adafactor' = 'adam',
  parallelConfig?: ParallelConfig
): number {
  const params = calculateModelParameters(modelConfig);
  let effectiveParams = params.total;
  
  // 考虑并行分片
  if (parallelConfig) {
    if (parallelConfig.tensorParallel > 1) {
      const shardedParams = params.attention + params.ffn;
      const unshardedParams = params.embedding + params.lmHead + params.layerNorm;
      effectiveParams = shardedParams / parallelConfig.tensorParallel + unshardedParams;
    }
    if (parallelConfig.pipelineParallel > 1) {
      effectiveParams = effectiveParams / parallelConfig.pipelineParallel;
    }
    // ZeRO-style data parallel也会分片优化器状态
    if (parallelConfig.dataParallel > 1) {
      // ZeRO Stage 1: 优化器状态分片
      effectiveParams = effectiveParams / parallelConfig.dataParallel;
    }
  }
  
  switch (optimizerType) {
    case 'adam':
      // Adam: momentum (FP32) + variance (FP32) = 8 bytes per param
      return effectiveParams * 8;
    case 'sgd':
      // SGD with momentum: just momentum (FP32) = 4 bytes per param
      return effectiveParams * 4;
    case 'adafactor':
      // Adafactor: row + column factors ≈ 2 bytes per param
      return effectiveParams * 2;
    default:
      return effectiveParams * 8;
  }
}

/**
 * 估算框架开销 (CUDA context, PyTorch buffers, etc.)
 */
export function calculateFrameworkOverhead(
  baseMemory: number
): number {
  // 经验值: 约5-10%的额外开销,最少500MB
  const overheadRatio = 0.08;
  const minOverhead = 500 * 1024 * 1024; // 500MB
  return Math.max(baseMemory * overheadRatio, minOverhead);
}

/**
 * 完整的GPU内存分析
 */
export function analyzeMemory(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  options: {
    parallelConfig?: ParallelConfig;
    useGradientCheckpointing?: boolean;
    optimizerType?: 'adam' | 'sgd' | 'adafactor';
  } = {}
): MemoryAnalysis {
  const { parallelConfig, useGradientCheckpointing = false, optimizerType = 'adam' } = options;
  const isTraining = inferenceConfig.mode === 'training';
  
  // 计算各组件
  const modelWeights = calculateModelWeightsMemory(modelConfig, inferenceConfig.dtype, parallelConfig);
  const kvCache = isTraining ? 0 : calculateKVCacheMemory(modelConfig, inferenceConfig, parallelConfig);
  const activations = calculateActivationsMemory(modelConfig, inferenceConfig, useGradientCheckpointing, parallelConfig);
  const gradients = isTraining ? calculateGradientsMemory(modelConfig, inferenceConfig.dtype, parallelConfig) : undefined;
  const optimizerStates = isTraining ? calculateOptimizerStatesMemory(modelConfig, optimizerType, parallelConfig) : undefined;
  
  const subtotal = modelWeights + kvCache + activations + (gradients || 0) + (optimizerStates || 0);
  const frameworkOverhead = calculateFrameworkOverhead(subtotal);
  const total = subtotal + frameworkOverhead;
  
  // Peak memory考虑训练时的临时缓冲区
  const peakMemory = isTraining ? total * 1.1 : total;
  
  // 构建breakdown
  const breakdown: MemoryBreakdown[] = [];
  
  breakdown.push({
    name: 'Model Weights',
    bytes: modelWeights,
    percentage: (modelWeights / total) * 100,
    color: MEMORY_COLORS.modelWeights,
  });
  
  if (kvCache > 0) {
    breakdown.push({
      name: 'KV Cache',
      bytes: kvCache,
      percentage: (kvCache / total) * 100,
      color: MEMORY_COLORS.kvCache,
    });
  }
  
  breakdown.push({
    name: 'Activations',
    bytes: activations,
    percentage: (activations / total) * 100,
    color: MEMORY_COLORS.activations,
  });
  
  if (gradients !== undefined) {
    breakdown.push({
      name: 'Gradients',
      bytes: gradients,
      percentage: (gradients / total) * 100,
      color: MEMORY_COLORS.gradients,
    });
  }
  
  if (optimizerStates !== undefined) {
    breakdown.push({
      name: 'Optimizer States',
      bytes: optimizerStates,
      percentage: (optimizerStates / total) * 100,
      color: MEMORY_COLORS.optimizerStates,
    });
  }
  
  breakdown.push({
    name: 'Framework Overhead',
    bytes: frameworkOverhead,
    percentage: (frameworkOverhead / total) * 100,
    color: MEMORY_COLORS.frameworkOverhead,
  });
  
  // GPU容量检查
  const gpuCapacity = (hardwareConfig as any).memoryCapacity 
    ? (hardwareConfig as any).memoryCapacity * 1e9 
    : undefined;
  const utilizationPercent = gpuCapacity ? (peakMemory / gpuCapacity) * 100 : undefined;
  const fitsInGPU = gpuCapacity ? peakMemory <= gpuCapacity * 0.95 : undefined; // 留5%余量
  
  return {
    modelWeights,
    kvCache,
    activations,
    gradients,
    optimizerStates,
    frameworkOverhead,
    total,
    peakMemory,
    breakdown,
    gpuCapacity,
    utilizationPercent,
    fitsInGPU,
  };
}

/**
 * 格式化内存大小
 */
export function formatMemorySize(bytes: number): string {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
  return bytes.toFixed(0) + ' B';
}

/**
 * 计算最大可支持的batch size
 */
export function calculateMaxBatchSize(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig?: ParallelConfig
): number {
  const gpuCapacity = (hardwareConfig as any).memoryCapacity 
    ? (hardwareConfig as any).memoryCapacity * 1e9 * 0.9 // 使用90%容量
    : 80e9; // 默认80GB
  
  // 固定开销 (模型权重 + 框架)
  const fixedMemory = calculateModelWeightsMemory(modelConfig, inferenceConfig.dtype, parallelConfig);
  const overhead = calculateFrameworkOverhead(fixedMemory);
  const availableMemory = gpuCapacity - fixedMemory - overhead;
  
  if (availableMemory <= 0) return 0;
  
  // 每个batch item的内存 (KV cache + activations)
  const singleBatchConfig = { ...inferenceConfig, batchSize: 1 };
  const kvCachePerBatch = calculateKVCacheMemory(modelConfig, singleBatchConfig, parallelConfig);
  const activationsPerBatch = calculateActivationsMemory(modelConfig, singleBatchConfig, false, parallelConfig);
  const memoryPerBatch = kvCachePerBatch + activationsPerBatch;
  
  if (memoryPerBatch <= 0) return 1;
  
  return Math.floor(availableMemory / memoryPerBatch);
}
