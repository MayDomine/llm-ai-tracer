/**
 * Communication Cost Calculator for Distributed LLM Training/Inference
 * 
 * Models communication overhead for:
 * - Data Parallel (DP): AllReduce gradients
 * - Tensor Parallel (TP): AllReduce/AllGather activations
 * - Pipeline Parallel (PP): Point-to-point activations/gradients
 * 
 * References:
 * - Megatron-LM parallelism
 * - NCCL collective operations
 */

import type { 
  ModelConfig, 
  HardwareConfig, 
  InferenceConfig, 
  ParallelConfig,
  CommunicationAnalysis
} from '../types/model';
import { DTYPE_BYTES, calculateModelParameters } from './calculator';

// 扩展HardwareConfig类型以包含网络带宽
interface ExtendedHardwareConfig extends HardwareConfig {
  memoryCapacity?: number;       // GB
  nvlinkBandwidth?: number;      // GB/s (intra-node)
  networkBandwidth?: number;     // GB/s (inter-node)
  gpusPerNode?: number;          // GPUs per node
}

/**
 * 获取有效带宽
 * 根据通信是否跨节点选择合适的带宽
 */
function getEffectiveBandwidth(
  hardware: ExtendedHardwareConfig,
  parallelSize: number,
  gpusPerNode: number = 8
): number {
  // 如果并行度 <= 单节点GPU数，使用NVLink带宽
  if (parallelSize <= gpusPerNode) {
    return hardware.nvlinkBandwidth || 300; // 默认300 GB/s (H100 NVLink)
  }
  // 跨节点使用网络带宽
  return hardware.networkBandwidth || 50; // 默认50 GB/s (InfiniBand)
}

/**
 * Ring AllReduce 通信时间
 * 
 * Ring AllReduce: 2 * (n-1)/n * data_size / bandwidth
 * 其中 n 是参与的GPU数量
 */
export function calculateRingAllReduceTime(
  dataSize: number,        // bytes
  numGPUs: number,
  bandwidth: number        // GB/s
): number {
  if (numGPUs <= 1) return 0;
  
  // Ring AllReduce: 发送和接收各需要 (n-1)/n 的数据量
  const factor = 2 * (numGPUs - 1) / numGPUs;
  const timeSeconds = (dataSize * factor) / (bandwidth * 1e9);
  return timeSeconds * 1000; // 转换为毫秒
}

/**
 * AllGather 通信时间
 * 
 * AllGather: (n-1)/n * data_size / bandwidth
 */
export function calculateAllGatherTime(
  dataSize: number,
  numGPUs: number,
  bandwidth: number
): number {
  if (numGPUs <= 1) return 0;
  
  const factor = (numGPUs - 1) / numGPUs;
  const timeSeconds = (dataSize * factor) / (bandwidth * 1e9);
  return timeSeconds * 1000;
}

/**
 * Point-to-Point 通信时间
 */
export function calculateP2PTime(
  dataSize: number,
  bandwidth: number
): number {
  const timeSeconds = dataSize / (bandwidth * 1e9);
  return timeSeconds * 1000;
}

/**
 * 计算数据并行通信开销
 * 
 * 每个训练步需要AllReduce所有梯度
 */
export function calculateDataParallelCommunication(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig: ParallelConfig,
  hardware: ExtendedHardwareConfig
): { bytes: number; timeMs: number } {
  const { dataParallel, tensorParallel, pipelineParallel } = parallelConfig;
  
  if (dataParallel <= 1) {
    return { bytes: 0, timeMs: 0 };
  }
  
  // 计算需要同步的参数量 (考虑TP和PP分片)
  const params = calculateModelParameters(modelConfig);
  let effectiveParams = params.total;
  
  // TP分片减少需要同步的参数
  if (tensorParallel > 1) {
    const shardedParams = params.attention + params.ffn;
    const unshardedParams = params.embedding + params.lmHead + params.layerNorm;
    effectiveParams = shardedParams / tensorParallel + unshardedParams;
  }
  
  // PP分片减少需要同步的参数
  if (pipelineParallel > 1) {
    effectiveParams = effectiveParams / pipelineParallel;
  }
  
  const bytesPerElement = DTYPE_BYTES[inferenceConfig.dtype];
  const gradientBytes = effectiveParams * bytesPerElement;
  
  // DP通常跨节点,使用网络带宽
  const bandwidth = getEffectiveBandwidth(hardware, dataParallel, hardware.gpusPerNode || 8);
  const timeMs = calculateRingAllReduceTime(gradientBytes, dataParallel, bandwidth);
  
  return { bytes: gradientBytes, timeMs };
}

/**
 * 计算张量并行通信开销
 * 
 * 每层需要:
 * - Attention: 1次AllReduce (output projection后)
 * - FFN: 1次AllReduce (down projection后)
 * 
 * 如果使用序列并行:
 * - LayerNorm后需要AllGather
 * - Dropout后需要ReduceScatter
 */
export function calculateTensorParallelCommunication(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig: ParallelConfig,
  hardware: ExtendedHardwareConfig
): { bytes: number; timeMs: number } {
  const { tensorParallel } = parallelConfig;
  
  if (tensorParallel <= 1) {
    return { bytes: 0, timeMs: 0 };
  }
  
  const { hiddenSize, numLayers } = modelConfig;
  const { batchSize, seqLen, mode, dtype } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[dtype];
  
  const effectiveSeqLen = mode === 'decode' ? 1 : seqLen;
  const tokens = batchSize * effectiveSeqLen;
  
  // 每层2次AllReduce (attention output, FFN output)
  // AllReduce数据量: tokens × hidden_size
  const allReduceDataPerLayer = tokens * hiddenSize * bytesPerElement;
  const allReducePerStep = allReduceDataPerLayer * 2; // attention + FFN
  
  // 如果是训练,反向传播也需要通信
  const isTraining = mode === 'training';
  const numPasses = isTraining ? 2 : 1; // forward + backward
  
  const totalBytesPerLayer = allReducePerStep * numPasses;
  const totalBytes = totalBytesPerLayer * numLayers;
  
  // TP通常在节点内,使用NVLink带宽
  const bandwidth = hardware.nvlinkBandwidth || 300;
  
  // 每次AllReduce的时间
  const timePerAllReduce = calculateRingAllReduceTime(allReduceDataPerLayer, tensorParallel, bandwidth);
  const totalTimeMs = timePerAllReduce * 2 * numPasses * numLayers;
  
  return { bytes: totalBytes, timeMs: totalTimeMs };
}

/**
 * 计算流水线并行通信开销
 * 
 * 每个micro-batch需要:
 * - 前向: 发送激活到下一个stage
 * - 反向: 发送梯度到上一个stage
 * 
 * Bubble开销: (pp_size - 1) / num_microbatches
 */
export function calculatePipelineParallelCommunication(
  modelConfig: ModelConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig: ParallelConfig,
  hardware: ExtendedHardwareConfig
): { bytes: number; timeMs: number; bubbleRatio: number } {
  const { pipelineParallel, microBatchSize, gradientAccumulation } = parallelConfig;
  
  if (pipelineParallel <= 1) {
    return { bytes: 0, timeMs: 0, bubbleRatio: 0 };
  }
  
  const { hiddenSize } = modelConfig;
  const { batchSize, seqLen, mode, dtype } = inferenceConfig;
  const bytesPerElement = DTYPE_BYTES[dtype];
  
  const effectiveSeqLen = mode === 'decode' ? 1 : seqLen;
  const isTraining = mode === 'training';
  
  // 计算micro-batch数量
  const effectiveMicroBatchSize = microBatchSize || batchSize;
  const numMicroBatches = Math.max(1, Math.ceil(batchSize / effectiveMicroBatchSize) * (gradientAccumulation || 1));
  
  // 每个micro-batch在stage边界传递的激活大小
  // Shape: [micro_batch_size, seq_len, hidden_size]
  const activationSize = effectiveMicroBatchSize * effectiveSeqLen * hiddenSize * bytesPerElement;
  
  // 前向和反向各传递一次 (训练时)
  const passesPerMicroBatch = isTraining ? 2 : 1;
  
  // 总共需要传递 (pp_size - 1) 次 (每个stage边界)
  const stageTransitions = pipelineParallel - 1;
  
  const bytesPerMicroBatch = activationSize * passesPerMicroBatch * stageTransitions;
  const totalBytes = bytesPerMicroBatch * numMicroBatches;
  
  // PP可能跨节点,根据配置选择带宽
  const bandwidth = getEffectiveBandwidth(hardware, pipelineParallel, hardware.gpusPerNode || 8);
  const timePerTransfer = calculateP2PTime(activationSize, bandwidth);
  const totalTimeMs = timePerTransfer * passesPerMicroBatch * stageTransitions * numMicroBatches;
  
  // 计算bubble开销
  // 1F1B schedule: bubble = (pp_size - 1) / num_microbatches
  const bubbleRatio = (pipelineParallel - 1) / numMicroBatches;
  
  return { bytes: totalBytes, timeMs: totalTimeMs, bubbleRatio };
}

/**
 * 完整的通信分析
 */
export function analyzeCommunication(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig: ParallelConfig,
  computeTimeMs: number
): CommunicationAnalysis {
  const extendedHardware = hardwareConfig as ExtendedHardwareConfig;
  
  // 计算各种并行的通信
  const dpComm = calculateDataParallelCommunication(
    modelConfig, inferenceConfig, parallelConfig, extendedHardware
  );
  
  const tpComm = calculateTensorParallelCommunication(
    modelConfig, inferenceConfig, parallelConfig, extendedHardware
  );
  
  const ppComm = calculatePipelineParallelCommunication(
    modelConfig, inferenceConfig, parallelConfig, extendedHardware
  );
  
  const totalCommBytes = dpComm.bytes + tpComm.bytes + ppComm.bytes;
  const totalCommTime = dpComm.timeMs + tpComm.timeMs + ppComm.timeMs;
  
  // 计算效率
  // 注意: TP通信与计算可以部分重叠,PP通信也可以与计算重叠
  // 这里使用保守估计,假设50%的通信可以被隐藏
  const effectiveCommTime = dpComm.timeMs + (tpComm.timeMs + ppComm.timeMs) * 0.5;
  const totalTime = computeTimeMs + effectiveCommTime + computeTimeMs * ppComm.bubbleRatio;
  
  const efficiency = computeTimeMs / totalTime;
  const computeCommRatio = totalCommTime > 0 ? computeTimeMs / totalCommTime : Infinity;
  
  return {
    dataParallelComm: dpComm.bytes,
    tensorParallelComm: tpComm.bytes,
    pipelineParallelComm: ppComm.bytes,
    totalCommBytes,
    
    dataParallelTime: dpComm.timeMs,
    tensorParallelTime: tpComm.timeMs,
    pipelineParallelTime: ppComm.timeMs,
    totalCommTime,
    
    bubbleOverhead: ppComm.bubbleRatio * 100, // 转换为百分比
    computeCommRatio,
    efficiency,
  };
}

/**
 * 估算分布式训练/推理的吞吐量
 */
export function estimateThroughput(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  parallelConfig: ParallelConfig,
  singleGPUTimeMs: number
): {
  tokensPerSecond: number;
  samplesPerSecond: number;
  effectiveGPUUtilization: number;
  timePerStepMs: number;
} {
  const { dataParallel, tensorParallel, pipelineParallel, gradientAccumulation = 1 } = parallelConfig;
  const totalGPUs = dataParallel * tensorParallel * pipelineParallel;
  
  // 每个GPU处理的计算量减少 (TP和PP分担计算)
  const computeTimeMs = singleGPUTimeMs / (tensorParallel * pipelineParallel);
  
  // 计算通信开销
  const commAnalysis = analyzeCommunication(
    modelConfig, hardwareConfig, inferenceConfig, parallelConfig, computeTimeMs
  );
  
  // 总时间 = 计算时间 + 通信时间 + bubble开销
  const effectiveCommTime = commAnalysis.dataParallelTime + 
    (commAnalysis.tensorParallelTime + commAnalysis.pipelineParallelTime) * 0.5;
  const bubbleTime = computeTimeMs * commAnalysis.bubbleOverhead / 100;
  const timePerStepMs = computeTimeMs + effectiveCommTime + bubbleTime;
  
  // 每步处理的样本数
  const samplesPerStep = inferenceConfig.batchSize * dataParallel * gradientAccumulation;
  const tokensPerStep = samplesPerStep * (inferenceConfig.mode === 'decode' ? 1 : inferenceConfig.seqLen);
  
  // 吞吐量
  const tokensPerSecond = (tokensPerStep / timePerStepMs) * 1000;
  const samplesPerSecond = (samplesPerStep / timePerStepMs) * 1000;
  
  // GPU利用率 = 理想时间 / 实际时间
  const idealTime = singleGPUTimeMs / totalGPUs;
  const effectiveGPUUtilization = idealTime / timePerStepMs;
  
  return {
    tokensPerSecond,
    samplesPerSecond,
    effectiveGPUUtilization,
    timePerStepMs,
  };
}

/**
 * 格式化带宽
 */
export function formatBandwidth(gbps: number): string {
  if (gbps >= 1000) return (gbps / 1000).toFixed(1) + ' TB/s';
  return gbps.toFixed(1) + ' GB/s';
}

/**
 * 格式化通信量
 */
export function formatCommBytes(bytes: number): string {
  if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + ' TB';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
  return bytes.toFixed(0) + ' B';
}
