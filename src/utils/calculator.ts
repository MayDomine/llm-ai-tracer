/**
 * LLM FLOPs 和内存访问计算器
 * 基于 Roofline 模型分析算术密度
 * 
 * 公式参考:
 * - OpenAI Scaling Laws: Forward FLOPs ≈ 2N, Training FLOPs ≈ 6N
 * - GEMM FLOPs: 2 * M * N * K (multiply-accumulate)
 * - Memory Access: (M*K + K*N + M*N) * bytes_per_element
 */

import type { 
  ModelConfig, 
  HardwareConfig, 
  InferenceConfig, 
  OperationAnalysis,
  ModuleAnalysis,
  ModelAnalysis 
} from '../types/model';

// 数据类型字节数映射
export const DTYPE_BYTES: Record<InferenceConfig['dtype'], number> = {
  'fp32': 4,
  'fp16': 2,
  'bf16': 2,
  'int8': 1,
  'int4': 0.5,
};

/**
 * 计算模型参数数量
 */
export function calculateModelParameters(modelConfig: ModelConfig): {
  embedding: number;
  attention: number;
  ffn: number;
  lmHead: number;
  layerNorm: number;
  total: number;
} {
  const { 
    hiddenSize: d, 
    numLayers: L, 
    vocabSize: V, 
    numKVHeads: h_kv,
    headDim: d_h,
    intermediateSize: d_ff,
    ffnType,
    numExperts,
    sharedExpertNum,
    denseIntermediateSize,
    sharedExpertIntermediateSize,
    numDenseLayers: configuredDenseLayers,
  } = modelConfig;

  // Embedding: vocab_size × hidden_size
  const embedding = V * d;

  // Attention per layer: Q, K, V, O projections
  // Q: d × d, K: d × (h_kv × d_h), V: d × (h_kv × d_h), O: d × d
  const d_kv = h_kv * d_h;
  const attentionPerLayer = d * d + d * d_kv + d * d_kv + d * d; // Q + K + V + O
  const attention = attentionPerLayer * L;

  const totalDenseLayers = Math.max(0, Math.min(L, configuredDenseLayers ?? 0));
  const totalMoeLayers = Math.max(0, L - totalDenseLayers);
  const denseDff = denseIntermediateSize ?? d_ff;
  const totalSharedIntermediate = sharedExpertIntermediateSize ?? ((sharedExpertNum || 0) * d_ff);

  // FFN per layer
  let ffn: number;
  if (ffnType === 'gpt') {
    // GPT-style: up (d × d_ff) + down (d_ff × d)
    ffn = L * 2 * d * d_ff;
  } else if (ffnType === 'gated') {
    // LLaMA-style: gate (d × d_ff) + up (d × d_ff) + down (d_ff × d)
    ffn = L * 3 * d * d_ff;
  } else if (ffnType === 'moe') {
    // MoE: dense prefix/suffix layers + routed experts + shared experts
    const nExperts = numExperts || 8;
    const denseFfnPerLayer = 3 * d * denseDff;
    const router = d * nExperts;
    // Each expert: 3 × d × d_ff (gated)
    const expertsParams = nExperts * 3 * d * d_ff;
    const sharedParams = 3 * d * totalSharedIntermediate;
    const moeFfnPerLayer = router + expertsParams + sharedParams;
    ffn = totalDenseLayers * denseFfnPerLayer + totalMoeLayers * moeFfnPerLayer;
  } else {
    ffn = L * 2 * d * d_ff;
  }

  // LM Head: hidden_size × vocab_size (often tied with embedding)
  const lmHead = d * V;

  // LayerNorm/RMSNorm: 2 per layer (pre-attn, pre-ffn) + 1 final
  // Each has hidden_size parameters (scale) + optional bias
  const layerNormPerLayer = 2 * d;
  const layerNorm = layerNormPerLayer * L + d; // +1 for final norm

  const total = embedding + attention + ffn + lmHead + layerNorm;

  return {
    embedding,
    attention,
    ffn,
    lmHead,
    layerNorm,
    total
  };
}

function getTrainingGatedLinearMultiplier(modelConfig: ModelConfig): number {
  return modelConfig.ffnType === 'gpt' ? 1 : 1.5;
}

function getMoELayerCounts(modelConfig: ModelConfig): { numDenseLayers: number; numMoeLayers: number } {
  if (modelConfig.ffnType !== 'moe') {
    return { numDenseLayers: modelConfig.numLayers, numMoeLayers: 0 };
  }

  const numDenseLayers = Math.max(
    0,
    Math.min(modelConfig.numLayers, modelConfig.numDenseLayers ?? 0)
  );

  return {
    numDenseLayers,
    numMoeLayers: modelConfig.numLayers - numDenseLayers,
  };
}

function getMoEIntermediateSizes(modelConfig: ModelConfig): {
  denseIntermediateSize: number;
  moeIntermediateSize: number;
  sharedExpertIntermediateSize: number;
} {
  const moeIntermediateSize = modelConfig.intermediateSize;
  return {
    denseIntermediateSize: modelConfig.denseIntermediateSize ?? moeIntermediateSize,
    moeIntermediateSize,
    sharedExpertIntermediateSize:
      modelConfig.sharedExpertIntermediateSize
      ?? ((modelConfig.sharedExpertNum ?? 0) * moeIntermediateSize),
  };
}

export interface TrainingParameterCounts {
  totalParameters: number;
  activeParameters: number;
  totalTransformerParameters: number;
  activeTransformerParameters: number;
  vocabParameters: number;
  numDenseLayers: number;
  numMoeLayers: number;
}

/**
 * Estimate training-active parameter counts using a Megatron-style MoE view:
 * routed experts contribute by top-k to active params, while total params still
 * include every expert for optimizer/state memory.
 */
export function calculateTrainingParameterCounts(modelConfig: ModelConfig): TrainingParameterCounts {
  const {
    hiddenSize: d,
    vocabSize: V,
    numKVHeads: h_kv,
    headDim: d_h,
    numExperts,
  } = modelConfig;

  const d_kv = h_kv * d_h;
  const attnParamsPerLayer = d * d + d * d_kv + d * d_kv + d * d;
  const gatedLinearMultiplier = getTrainingGatedLinearMultiplier(modelConfig);
  const { numDenseLayers, numMoeLayers } = getMoELayerCounts(modelConfig);
  const {
    denseIntermediateSize,
    moeIntermediateSize,
    sharedExpertIntermediateSize,
  } = getMoEIntermediateSizes(modelConfig);

  const denseMlpParamsPerLayer = 2 * d * denseIntermediateSize * gatedLinearMultiplier;

  if (modelConfig.ffnType !== 'moe') {
    const totalTransformerParameters =
      modelConfig.numLayers * (attnParamsPerLayer + denseMlpParamsPerLayer);
    const vocabParameters = d * V;
    return {
      totalParameters: totalTransformerParameters + vocabParameters,
      activeParameters: totalTransformerParameters + vocabParameters,
      totalTransformerParameters,
      activeTransformerParameters: totalTransformerParameters,
      vocabParameters,
      numDenseLayers: modelConfig.numLayers,
      numMoeLayers: 0,
    };
  }

  const nExperts = numExperts || 8;
  const topK = modelConfig.numExpertsPerToken || 2;
  const routerParamsPerLayer = d * nExperts;
  const moeTotalMlpParamsPerLayer =
    routerParamsPerLayer
    + (2 * d * moeIntermediateSize * gatedLinearMultiplier * nExperts)
    + (2 * d * sharedExpertIntermediateSize * gatedLinearMultiplier);
  const moeActiveMlpParamsPerLayer =
    routerParamsPerLayer
    + (2 * d * moeIntermediateSize * gatedLinearMultiplier * topK)
    + (2 * d * sharedExpertIntermediateSize * gatedLinearMultiplier);

  const totalTransformerParameters =
    numDenseLayers * (attnParamsPerLayer + denseMlpParamsPerLayer)
    + numMoeLayers * (attnParamsPerLayer + moeTotalMlpParamsPerLayer);
  const activeTransformerParameters =
    numDenseLayers * (attnParamsPerLayer + denseMlpParamsPerLayer)
    + numMoeLayers * (attnParamsPerLayer + moeActiveMlpParamsPerLayer);
  const vocabParameters = d * V;

  return {
    totalParameters: totalTransformerParameters + vocabParameters,
    activeParameters: activeTransformerParameters + vocabParameters,
    totalTransformerParameters,
    activeTransformerParameters,
    vocabParameters,
    numDenseLayers,
    numMoeLayers,
  };
}

/**
 * Estimate training FLOPs for a batch with a Megatron-style Transformer/MoE formula.
 *
 * This follows the same structure as Megatron's `num_floating_point_operations()`
 * for standard Transformer + MoE training, so routed experts contribute by top-k
 * rather than by total expert count.
 */
export function estimateTrainingFlops(
  modelConfig: ModelConfig,
  batchSize: number,
  seqLength: number
): number {
  const {
    hiddenSize: h,
    vocabSize: v,
    numAttentionHeads,
    numKVHeads,
    headDim,
  } = modelConfig;

  const { numDenseLayers, numMoeLayers } = getMoELayerCounts(modelConfig);
  const {
    denseIntermediateSize,
    moeIntermediateSize,
    sharedExpertIntermediateSize,
  } = getMoEIntermediateSizes(modelConfig);
  const gatedLinearMultiplier = getTrainingGatedLinearMultiplier(modelConfig);
  const queryProjectionSize = headDim * numAttentionHeads;
  const queryProjectionToHiddenSizeRatio = queryProjectionSize / h;
  const expansionFactor = 12; // 3 * 2 * 2, same as Megatron estimator

  const selfAttentionTerm =
    expansionFactor
    * modelConfig.numLayers
    * h
    * h
    * (
      (
        1
        + (numKVHeads / numAttentionHeads)
        + (seqLength / h / 2)
      )
      * queryProjectionToHiddenSizeRatio
    );

  const denseFeedForwardTerm =
    numDenseLayers * denseIntermediateSize * gatedLinearMultiplier;
  const moeFeedForwardTerm =
    numMoeLayers
    * (
      moeIntermediateSize * (modelConfig.numExpertsPerToken || 2) * gatedLinearMultiplier
      + sharedExpertIntermediateSize * gatedLinearMultiplier
    );

  const perTokenFlops =
    expansionFactor * h * (denseFeedForwardTerm + moeFeedForwardTerm)
    + selfAttentionTerm
    + 6 * h * v;

  return batchSize * seqLength * perTokenFlops;
}

export function estimateTrainingFlopsPerToken(
  modelConfig: ModelConfig,
  seqLength: number
): number {
  return estimateTrainingFlops(modelConfig, 1, seqLength) / seqLength;
}

/**
 * 计算训练时的反向传播FLOPs倍数
 */
export function getBackwardMultiplier(useActivationCheckpointing: boolean): number {
  // Backward pass requires ~2x forward FLOPs
  // With activation checkpointing, we recompute forward once more
  return useActivationCheckpointing ? 3 : 2; // Total: 4x or 3x forward for full training
}

// 生成唯一ID
let idCounter = 0;
const generateId = () => `op_${++idCounter}`;

/**
 * 计算GEMM的FLOPs和内存访问
 * FLOPs = 2 * M * N * K (乘加各算一次)
 * Memory = (M*K + K*N + M*N) * bytes_per_element
 */
function analyzeGEMM(
  name: string,
  module: OperationAnalysis['module'],
  subModule: string,
  M: number,
  N: number,
  K: number,
  bytesPerElement: number,
  hardwareConfig: HardwareConfig
): OperationAnalysis {
  const flops = 2 * M * N * K;
  // 输入矩阵A (M x K) + 权重矩阵B (K x N) + 输出矩阵C (M x N)
  const memoryBytes = (M * K + K * N + M * N) * bytesPerElement;
  const arithmeticIntensity = flops / memoryBytes;
  
  // Roofline模型判断
  // 拐点 = 计算能力 / 内存带宽
  const rooflinePoint = (hardwareConfig.computeCapability * 1e12) / (hardwareConfig.memoryBandwidth * 1e12);
  const isComputeBound = arithmeticIntensity >= rooflinePoint;
  
  // 计算理论执行时间
  const computeTime = flops / (hardwareConfig.computeCapability * 1e12) * 1000; // ms
  const memoryTime = memoryBytes / (hardwareConfig.memoryBandwidth * 1e12) * 1000; // ms
  const theoreticalTime = Math.max(computeTime, memoryTime);
  
  return {
    id: generateId(),
    name,
    type: 'gemm',
    module,
    subModule,
    flops,
    memoryBytes,
    arithmeticIntensity,
    isComputeBound,
    theoreticalTime,
    bottleneck: isComputeBound ? 'compute' : 'memory',
    shape: { m: M, n: N, k: K },
    description: `GEMM [${M} x ${K}] × [${K} x ${N}] = [${M} x ${N}]`,
  };
}

/**
 * 分析非GEMM操作
 */
function analyzeOperation(
  name: string,
  type: OperationAnalysis['type'],
  module: OperationAnalysis['module'],
  subModule: string,
  flops: number,
  memoryBytes: number,
  hardwareConfig: HardwareConfig,
  description: string
): OperationAnalysis {
  const arithmeticIntensity = memoryBytes > 0 ? flops / memoryBytes : 0;
  const rooflinePoint = (hardwareConfig.computeCapability * 1e12) / (hardwareConfig.memoryBandwidth * 1e12);
  const isComputeBound = arithmeticIntensity >= rooflinePoint;
  
  const computeTime = flops / (hardwareConfig.computeCapability * 1e12) * 1000;
  const memoryTime = memoryBytes / (hardwareConfig.memoryBandwidth * 1e12) * 1000;
  const theoreticalTime = Math.max(computeTime, memoryTime);
  
  return {
    id: generateId(),
    name,
    type,
    module,
    subModule,
    flops,
    memoryBytes,
    arithmeticIntensity,
    isComputeBound,
    theoreticalTime,
    bottleneck: isComputeBound ? 'compute' : 'memory',
    description,
  };
}

/**
 * 获取当前模式下的token数量
 */
function getTokenCount(inferenceConfig: InferenceConfig): number {
  const { mode, batchSize, seqLen } = inferenceConfig;
  // decode模式每次只处理1个新token
  return mode === 'decode' ? batchSize : batchSize * seqLen;
}

/**
 * 获取attention的query序列长度
 */
function getQuerySeqLen(inferenceConfig: InferenceConfig): number {
  const { mode, seqLen } = inferenceConfig;
  // decode模式query长度为1
  return mode === 'decode' ? 1 : seqLen;
}

/**
 * 获取attention的KV序列长度
 */
function getKVSeqLen(inferenceConfig: InferenceConfig): number {
  const { mode, seqLen, kvCacheLen } = inferenceConfig;
  if (mode === 'decode') {
    // decode时需要访问所有之前的KV cache + 当前token
    return kvCacheLen + 1;
  }
  // training/prefill时KV长度等于序列长度
  return seqLen;
}

/**
 * 分析Embedding层
 */
function analyzeEmbedding(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareConfig: HardwareConfig
): ModuleAnalysis {
  const { vocabSize, hiddenSize } = modelConfig;
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  
  // Embedding查找: 从词表中查找对应向量
  // FLOPs ≈ 0 (只是查表)
  // Memory = 读取embedding table部分 + 写入输出
  const numTokens = getTokenCount(inferenceConfig);
  const memoryBytes = (numTokens * hiddenSize + vocabSize * hiddenSize) * bytesPerElement;
  
  const modeLabel = inferenceConfig.mode === 'decode' ? ' (1 token/batch)' : '';
  
  const embeddingOp = analyzeOperation(
    'Token Embedding',
    'embedding',
    'embedding',
    'token_embedding',
    0, // embedding本质是查表，没有计算
    memoryBytes,
    hardwareConfig,
    `Embed ${numTokens} tokens from vocab size ${vocabSize}${modeLabel}`
  );
  
  return {
    name: 'Embedding',
    operations: [embeddingOp],
    totalFlops: embeddingOp.flops,
    totalMemoryBytes: embeddingOp.memoryBytes,
    avgArithmeticIntensity: embeddingOp.arithmeticIntensity,
    computeBoundOps: embeddingOp.isComputeBound ? 1 : 0,
    memoryBoundOps: embeddingOp.isComputeBound ? 0 : 1,
  };
}

/**
 * 分析Attention模块
 */
function analyzeAttention(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareConfig: HardwareConfig
): OperationAnalysis[] {
  const { hiddenSize, numAttentionHeads, numKVHeads, headDim } = modelConfig;
  const { batchSize, mode } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  
  const operations: OperationAnalysis[] = [];
  const B = batchSize;
  const H = numAttentionHeads;
  const D = headDim;
  const KVH = numKVHeads;
  
  // 获取不同模式下的序列长度
  const querySeqLen = getQuerySeqLen(inferenceConfig);  // Q的序列长度
  const kvSeqLen = getKVSeqLen(inferenceConfig);        // KV的序列长度
  const numTokens = getTokenCount(inferenceConfig);      // 当前处理的token数
  
  const isDecodeMode = mode === 'decode';
  const modePrefix = isDecodeMode ? '[Decode] ' : '';
  
  // Q projection: [B*querySeqLen, hidden] x [hidden, H*D] = [B*querySeqLen, H*D]
  operations.push(analyzeGEMM(
    `${modePrefix}Q Projection`,
    'attention',
    'q_proj',
    numTokens,
    H * D,
    hiddenSize,
    bytesPerElement,
    hardwareConfig
  ));
  
  // K projection: [B*querySeqLen, hidden] x [hidden, KVH*D] = [B*querySeqLen, KVH*D]
  operations.push(analyzeGEMM(
    `${modePrefix}K Projection`,
    'attention',
    'k_proj',
    numTokens,
    KVH * D,
    hiddenSize,
    bytesPerElement,
    hardwareConfig
  ));
  
  // V projection: [B*querySeqLen, hidden] x [hidden, KVH*D] = [B*querySeqLen, KVH*D]
  operations.push(analyzeGEMM(
    `${modePrefix}V Projection`,
    'attention',
    'v_proj',
    numTokens,
    KVH * D,
    hiddenSize,
    bytesPerElement,
    hardwareConfig
  ));
  
  // Decode模式需要读取KV Cache
  if (isDecodeMode) {
    // 读取KV Cache: 需要读取所有之前的K和V
    // KV Cache大小: 2 * numLayers * B * kvCacheLen * KVH * D
    // 但这里只分析单层，所以是 2 * B * kvCacheLen * KVH * D
    const kvCacheReadSize = 2 * B * inferenceConfig.kvCacheLen * KVH * D * bytesPerElement;
    operations.push(analyzeOperation(
      'KV Cache Read',
      'embedding', // 用embedding类型表示内存操作
      'attention',
      'kv_cache_read',
      0, // 没有计算
      kvCacheReadSize,
      hardwareConfig,
      `Read KV cache: ${inferenceConfig.kvCacheLen} tokens × ${KVH} heads × ${D} dim`
    ));
    
    // 写入新的KV到Cache
    const kvCacheWriteSize = 2 * B * 1 * KVH * D * bytesPerElement;
    operations.push(analyzeOperation(
      'KV Cache Write',
      'embedding',
      'attention',
      'kv_cache_write',
      0,
      kvCacheWriteSize,
      hardwareConfig,
      `Write new KV to cache: 1 token × ${KVH} heads × ${D} dim`
    ));
  }
  
  // QK^T attention scores
  // Training/Prefill: [B*H, S, D] x [B*H, D, S] = [B*H, S, S]
  // Decode: [B*H, 1, D] x [B*H, D, kvSeqLen] = [B*H, 1, kvSeqLen]
  const kvReplicationFactor = H / KVH;
  operations.push(analyzeGEMM(
    `${modePrefix}QK^T (Attention Scores)`,
    'attention',
    'qk_matmul',
    B * H * querySeqLen,
    kvSeqLen,
    D,
    bytesPerElement,
    hardwareConfig
  ));
  
  // Softmax: 对每行进行softmax
  // Training/Prefill: S x S, Decode: 1 x kvSeqLen
  // Softmax FLOPs per element: max(1) + exp(2) + sum(1) + div(1) = 5 ops
  const softmaxElements = B * H * querySeqLen * kvSeqLen;
  const softmaxFlops = 5 * softmaxElements;
  const softmaxMemory = 2 * softmaxElements * bytesPerElement; // Read attention scores + Write normalized
  operations.push(analyzeOperation(
    `${modePrefix}Softmax`,
    'softmax',
    'attention',
    'softmax',
    softmaxFlops,
    softmaxMemory,
    hardwareConfig,
    `Softmax over ${querySeqLen}x${kvSeqLen} attention matrix`
  ));
  
  // Attention x V
  // Training/Prefill: [B*H, S, S] x [B*H, S, D] = [B*H, S, D]
  // Decode: [B*H, 1, kvSeqLen] x [B*H, kvSeqLen, D] = [B*H, 1, D]
  operations.push(analyzeGEMM(
    `${modePrefix}Attention × V`,
    'attention',
    'attn_v_matmul',
    B * H * querySeqLen,
    D,
    kvSeqLen,
    bytesPerElement,
    hardwareConfig
  ));
  
  // Output projection: [B*querySeqLen, H*D] x [H*D, hidden] = [B*querySeqLen, hidden]
  operations.push(analyzeGEMM(
    `${modePrefix}Output Projection`,
    'attention',
    'o_proj',
    numTokens,
    hiddenSize,
    H * D,
    bytesPerElement,
    hardwareConfig
  ));
  
  // 添加GQA标注信息
  if (kvReplicationFactor > 1) {
    operations[0].description += ` (GQA: ${H} Q heads, ${KVH} KV heads)`;
  }
  
  return operations;
}

/**
 * 分析FFN模块
 */
function analyzeFFN(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareConfig: HardwareConfig
): OperationAnalysis[] {
  const { hiddenSize, intermediateSize, ffnType, numExperts, numExpertsPerToken, sharedExpertNum } = modelConfig;
  const { mode } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  
  const operations: OperationAnalysis[] = [];
  const tokens = getTokenCount(inferenceConfig);
  const isDecodeMode = mode === 'decode';
  const modePrefix = isDecodeMode ? '[Decode] ' : '';
  
  if (ffnType === 'gpt') {
    // GPT-style: 两个Linear
    // Up: [tokens, hidden] x [hidden, intermediate] = [tokens, intermediate]
    operations.push(analyzeGEMM(
      `${modePrefix}FFN Up`,
      'ffn',
      'up_proj',
      tokens,
      intermediateSize,
      hiddenSize,
      bytesPerElement,
      hardwareConfig
    ));
    
    // GELU activation
    const geluFlops = 8 * tokens * intermediateSize; // GELU约8个操作
    const geluMemory = 2 * tokens * intermediateSize * bytesPerElement;
    operations.push(analyzeOperation(
      `${modePrefix}GELU`,
      'gelu',
      'ffn',
      'activation',
      geluFlops,
      geluMemory,
      hardwareConfig,
      `GELU activation on ${tokens}x${intermediateSize} tensor`
    ));
    
    // Down: [tokens, intermediate] x [intermediate, hidden] = [tokens, hidden]
    operations.push(analyzeGEMM(
      `${modePrefix}FFN Down`,
      'ffn',
      'down_proj',
      tokens,
      hiddenSize,
      intermediateSize,
      bytesPerElement,
      hardwareConfig
    ));
  } else if (ffnType === 'gated') {
    // LLaMA-style: 三个Linear (Gated)
    // Gate: [tokens, hidden] x [hidden, intermediate]
    operations.push(analyzeGEMM(
      `${modePrefix}FFN Gate`,
      'ffn',
      'gate_proj',
      tokens,
      intermediateSize,
      hiddenSize,
      bytesPerElement,
      hardwareConfig
    ));
    
    // Up: [tokens, hidden] x [hidden, intermediate]
    operations.push(analyzeGEMM(
      `${modePrefix}FFN Up`,
      'ffn',
      'up_proj',
      tokens,
      intermediateSize,
      hiddenSize,
      bytesPerElement,
      hardwareConfig
    ));
    
    // SiLU activation on gate output
    const siluFlops = 4 * tokens * intermediateSize;
    const siluMemory = 2 * tokens * intermediateSize * bytesPerElement;
    operations.push(analyzeOperation(
      `${modePrefix}SiLU`,
      'silu',
      'ffn',
      'activation',
      siluFlops,
      siluMemory,
      hardwareConfig,
      `SiLU activation on ${tokens}x${intermediateSize} tensor`
    ));
    
    // Element-wise multiply: gate * up
    const mulFlops = tokens * intermediateSize;
    const mulMemory = 3 * tokens * intermediateSize * bytesPerElement; // 2 reads + 1 write
    operations.push(analyzeOperation(
      `${modePrefix}Gate × Up`,
      'elementwise',
      'ffn',
      'gate_multiply',
      mulFlops,
      mulMemory,
      hardwareConfig,
      `Element-wise multiply ${tokens}x${intermediateSize}`
    ));
    
    // Down: [tokens, intermediate] x [intermediate, hidden]
    operations.push(analyzeGEMM(
      `${modePrefix}FFN Down`,
      'ffn',
      'down_proj',
      tokens,
      hiddenSize,
      intermediateSize,
      bytesPerElement,
      hardwareConfig
    ));
  } else if (ffnType === 'moe') {
    // MoE: Router + 多个专家
    const nExperts = numExperts || 8;
    const topK = numExpertsPerToken || 2;
    const sharedExperts = sharedExpertNum || 0;
    
    // Router: [tokens, hidden] x [hidden, numExperts]
    operations.push(analyzeGEMM(
      `${modePrefix}MoE Router`,
      'ffn',
      'router',
      tokens,
      nExperts,
      hiddenSize,
      bytesPerElement,
      hardwareConfig
    ));
    
    // Router softmax
    const routerSoftmaxFlops = 5 * tokens * nExperts;
    const routerSoftmaxMem = 2 * tokens * nExperts * bytesPerElement;
    operations.push(analyzeOperation(
      `${modePrefix}Router Softmax`,
      'softmax',
      'ffn',
      'router_softmax',
      routerSoftmaxFlops,
      routerSoftmaxMem,
      hardwareConfig,
      `Softmax for routing ${tokens} tokens to ${nExperts} experts`
    ));
    
    // 每个token激活topK个专家
    // 假设tokens均匀分布到各专家
    const tokensPerExpert = Math.ceil(tokens * topK / nExperts);
    
    // 激活的专家 (Gate + Up + Down for each)
    for (let i = 0; i < topK; i++) {
      operations.push(analyzeGEMM(
        `${modePrefix}Expert ${i + 1} Gate`,
        'ffn',
        `expert_${i}_gate`,
        tokensPerExpert,
        intermediateSize,
        hiddenSize,
        bytesPerElement,
        hardwareConfig
      ));
      
      operations.push(analyzeGEMM(
        `${modePrefix}Expert ${i + 1} Up`,
        'ffn',
        `expert_${i}_up`,
        tokensPerExpert,
        intermediateSize,
        hiddenSize,
        bytesPerElement,
        hardwareConfig
      ));
      
      operations.push(analyzeGEMM(
        `${modePrefix}Expert ${i + 1} Down`,
        'ffn',
        `expert_${i}_down`,
        tokensPerExpert,
        hiddenSize,
        intermediateSize,
        bytesPerElement,
        hardwareConfig
      ));
    }
    
    // 共享专家 (如果有)
    if (sharedExperts > 0) {
      operations.push(analyzeGEMM(
        `${modePrefix}Shared Expert Gate`,
        'ffn',
        'shared_expert_gate',
        tokens,
        intermediateSize,
        hiddenSize,
        bytesPerElement,
        hardwareConfig
      ));
      
      operations.push(analyzeGEMM(
        `${modePrefix}Shared Expert Up`,
        'ffn',
        'shared_expert_up',
        tokens,
        intermediateSize,
        hiddenSize,
        bytesPerElement,
        hardwareConfig
      ));
      
      operations.push(analyzeGEMM(
        `${modePrefix}Shared Expert Down`,
        'ffn',
        'shared_expert_down',
        tokens,
        hiddenSize,
        intermediateSize,
        bytesPerElement,
        hardwareConfig
      ));
    }
  }
  
  return operations;
}

/**
 * 分析单个Transformer Block
 */
function analyzeTransformerBlock(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareConfig: HardwareConfig
): ModuleAnalysis {
  const { hiddenSize } = modelConfig;
  const { mode } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  
  const operations: OperationAnalysis[] = [];
  const tokens = getTokenCount(inferenceConfig);
  const isDecodeMode = mode === 'decode';
  const modePrefix = isDecodeMode ? '[Decode] ' : '';
  
  // Pre-attention RMSNorm
  const rmsNormFlops = 5 * tokens * hiddenSize;
  const rmsNormMemory = 3 * tokens * hiddenSize * bytesPerElement;
  operations.push(analyzeOperation(
    `${modePrefix}Pre-Attn RMSNorm`,
    'rmsnorm',
    'attention',
    'pre_norm',
    rmsNormFlops,
    rmsNormMemory,
    hardwareConfig,
    `RMSNorm on ${tokens}x${hiddenSize} tensor`
  ));
  
  // Attention
  operations.push(...analyzeAttention(modelConfig, inferenceConfig, hardwareConfig));
  
  // Residual add
  const residualFlops = tokens * hiddenSize;
  const residualMemory = 3 * tokens * hiddenSize * bytesPerElement;
  operations.push(analyzeOperation(
    `${modePrefix}Residual Add (Attn)`,
    'elementwise',
    'attention',
    'residual',
    residualFlops,
    residualMemory,
    hardwareConfig,
    `Residual connection ${tokens}x${hiddenSize}`
  ));
  
  // Pre-FFN RMSNorm
  operations.push(analyzeOperation(
    `${modePrefix}Pre-FFN RMSNorm`,
    'rmsnorm',
    'ffn',
    'pre_norm',
    rmsNormFlops,
    rmsNormMemory,
    hardwareConfig,
    `RMSNorm on ${tokens}x${hiddenSize} tensor`
  ));
  
  // FFN
  operations.push(...analyzeFFN(modelConfig, inferenceConfig, hardwareConfig));
  
  // Residual add
  operations.push(analyzeOperation(
    `${modePrefix}Residual Add (FFN)`,
    'elementwise',
    'ffn',
    'residual',
    residualFlops,
    residualMemory,
    hardwareConfig,
    `Residual connection ${tokens}x${hiddenSize}`
  ));
  
  // 汇总统计
  const totalFlops = operations.reduce((sum, op) => sum + op.flops, 0);
  const totalMemoryBytes = operations.reduce((sum, op) => sum + op.memoryBytes, 0);
  const computeBoundOps = operations.filter(op => op.isComputeBound).length;
  const memoryBoundOps = operations.filter(op => !op.isComputeBound).length;
  
  return {
    name: 'Transformer Block',
    operations,
    totalFlops,
    totalMemoryBytes,
    avgArithmeticIntensity: totalFlops / totalMemoryBytes,
    computeBoundOps,
    memoryBoundOps,
  };
}

/**
 * 分析LM Head
 */
function analyzeLMHead(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  hardwareConfig: HardwareConfig
): ModuleAnalysis {
  const { vocabSize, hiddenSize } = modelConfig;
  const { mode } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  
  const tokens = getTokenCount(inferenceConfig);
  const isDecodeMode = mode === 'decode';
  const modePrefix = isDecodeMode ? '[Decode] ' : '';
  
  // Final RMSNorm
  const rmsNormFlops = 5 * tokens * hiddenSize;
  const rmsNormMemory = 3 * tokens * hiddenSize * bytesPerElement;
  const normOp = analyzeOperation(
    `${modePrefix}Final RMSNorm`,
    'rmsnorm',
    'lm_head',
    'final_norm',
    rmsNormFlops,
    rmsNormMemory,
    hardwareConfig,
    `RMSNorm on ${tokens}x${hiddenSize} tensor`
  );
  
  // LM Head projection: [tokens, hidden] x [hidden, vocab] = [tokens, vocab]
  const lmHeadOp = analyzeGEMM(
    `${modePrefix}LM Head`,
    'lm_head',
    'projection',
    tokens,
    vocabSize,
    hiddenSize,
    bytesPerElement,
    hardwareConfig
  );
  
  const operations = [normOp, lmHeadOp];
  const totalFlops = operations.reduce((sum, op) => sum + op.flops, 0);
  const totalMemoryBytes = operations.reduce((sum, op) => sum + op.memoryBytes, 0);
  
  return {
    name: 'LM Head',
    operations,
    totalFlops,
    totalMemoryBytes,
    avgArithmeticIntensity: totalFlops / totalMemoryBytes,
    computeBoundOps: operations.filter(op => op.isComputeBound).length,
    memoryBoundOps: operations.filter(op => !op.isComputeBound).length,
  };
}

/**
 * 完整模型分析
 */
export function analyzeModel(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig
): ModelAnalysis {
  // 重置ID计数器
  idCounter = 0;
  
  // 计算模型参数量
  const parameterCount = calculateModelParameters(modelConfig);
  
  const embedding = analyzeEmbedding(modelConfig, inferenceConfig, hardwareConfig);
  const transformerBlock = analyzeTransformerBlock(modelConfig, inferenceConfig, hardwareConfig);
  const lmHead = analyzeLMHead(modelConfig, inferenceConfig, hardwareConfig);
  
  // 计算总量 (考虑多层) - Forward Pass
  const forwardFlops = 
    embedding.totalFlops + 
    transformerBlock.totalFlops * modelConfig.numLayers + 
    lmHead.totalFlops;
  
  const totalMemoryBytes = 
    embedding.totalMemoryBytes + 
    transformerBlock.totalMemoryBytes * modelConfig.numLayers + 
    lmHead.totalMemoryBytes;
  
  // 根据模式计算总FLOPs
  let totalFlops: number;
  let backwardFlops: number | undefined;
  
  if (inferenceConfig.mode === 'training') {
    // Training mode: forward + backward (backward ≈ 2x forward)
    backwardFlops = forwardFlops * 2;
    totalFlops = forwardFlops + backwardFlops;
  } else {
    // Inference mode: forward only
    totalFlops = forwardFlops;
    backwardFlops = undefined;
  }
  
  const overallArithmeticIntensity = totalFlops / totalMemoryBytes;
  
  // Roofline拐点
  const rooflinePoint = hardwareConfig.computeCapability / hardwareConfig.memoryBandwidth;
  
  // 估计延迟
  const computeTime = totalFlops / (hardwareConfig.computeCapability * 1e12) * 1000;
  const memoryTime = totalMemoryBytes / (hardwareConfig.memoryBandwidth * 1e12) * 1000;
  const estimatedLatency = Math.max(computeTime, memoryTime);
  
  // 计算tokens数量
  const numTokens = inferenceConfig.mode === 'decode' 
    ? inferenceConfig.batchSize 
    : inferenceConfig.batchSize * inferenceConfig.seqLen;
  
  // 计算每token的FLOPs
  const flopsPerToken = totalFlops / numTokens;
  
  // 使用 2N 近似验证
  const approximateFlops = 2 * parameterCount.total * numTokens;
  const approximationError = Math.abs(totalFlops - approximateFlops) / approximateFlops;
  
  return {
    modelConfig,
    hardwareConfig,
    inferenceConfig,
    embedding,
    transformerBlock,
    lmHead,
    totalFlops,
    forwardFlops,
    backwardFlops,
    totalMemoryBytes,
    overallArithmeticIntensity,
    estimatedLatency,
    rooflinePoint,
    parameterCount,
    flopsPerToken,
    numTokens,
    approximateFlops,
    approximationError,
  };
}

/**
 * 格式化数字显示
 */
export function formatNumber(num: number): string {
  if (num >= 1e15) return (num / 1e15).toFixed(2) + ' P';
  if (num >= 1e12) return (num / 1e12).toFixed(2) + ' T';
  if (num >= 1e9) return (num / 1e9).toFixed(2) + ' G';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + ' M';
  if (num >= 1e3) return (num / 1e3).toFixed(2) + ' K';
  return num.toFixed(2);
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
  return bytes.toFixed(2) + ' B';
}

export function formatTime(ms: number): string {
  if (ms >= 1000) return (ms / 1000).toFixed(3) + ' s';
  if (ms >= 1) return ms.toFixed(3) + ' ms';
  return (ms * 1000).toFixed(3) + ' μs';
}
