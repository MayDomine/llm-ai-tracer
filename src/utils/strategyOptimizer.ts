/**
 * Strategy Optimizer for Distributed LLM Training/Inference
 * 
 * Finds optimal parallel configurations based on:
 * - GPU memory constraints
 * - Communication overhead
 * - Throughput/Latency objectives
 */

import type { 
  ModelConfig, 
  HardwareConfig, 
  InferenceConfig, 
  ParallelConfig,
  StrategyConfig,
  StrategyAnalysis,
  StrategyComparison,
  MemoryAnalysis,
  CommunicationAnalysis
} from '../types/model';
import { analyzeMemory } from './memoryCalculator';
import { analyzeCommunication, estimateThroughput } from './communicationCalculator';
import { analyzeModel } from './calculator';

// 扩展硬件配置类型
interface ExtendedHardwareConfig extends HardwareConfig {
  memoryCapacity?: number;
  nvlinkBandwidth?: number;
  networkBandwidth?: number;
  gpusPerNode?: number;
}

/**
 * 生成所有有效的并行配置组合
 */
export function generateValidConfigurations(
  numGPUs: number,
  modelConfig: ModelConfig,
  options: {
    maxTP?: number;
    maxPP?: number;
    allowedTPSizes?: number[];
  } = {}
): ParallelConfig[] {
  const {
    maxTP = Math.min(8, modelConfig.numAttentionHeads),
    maxPP = modelConfig.numLayers,
    allowedTPSizes = [1, 2, 4, 8]
  } = options;
  
  const configs: ParallelConfig[] = [];
  
  // 获取numGPUs的所有因子组合
  for (const tp of allowedTPSizes) {
    if (tp > maxTP || tp > numGPUs) continue;
    if (numGPUs % tp !== 0) continue;
    
    // TP必须能整除attention heads
    if (modelConfig.numAttentionHeads % tp !== 0) continue;
    
    const remainingGPUs = numGPUs / tp;
    
    for (let pp = 1; pp <= Math.min(maxPP, remainingGPUs); pp++) {
      if (remainingGPUs % pp !== 0) continue;
      
      // PP必须能整除层数
      if (modelConfig.numLayers % pp !== 0) continue;
      
      const dp = remainingGPUs / pp;
      
      // 验证 DP * TP * PP = numGPUs
      if (dp * tp * pp !== numGPUs) continue;
      
      // 生成不同的micro batch和gradient accumulation配置
      const microBatchSizes = [1, 2, 4, 8, 16, 32];
      const gradAccumSteps = [1, 2, 4, 8];
      
      for (const microBatchSize of microBatchSizes) {
        for (const gradientAccumulation of gradAccumSteps) {
          configs.push({
            dataParallel: dp,
            tensorParallel: tp,
            pipelineParallel: pp,
            microBatchSize,
            gradientAccumulation,
          });
        }
      }
    }
  }
  
  // 去重 (基于DP, TP, PP, 保留micro batch最小的配置用于初始评估)
  const uniqueConfigs = new Map<string, ParallelConfig>();
  for (const config of configs) {
    const key = `${config.dataParallel}-${config.tensorParallel}-${config.pipelineParallel}`;
    if (!uniqueConfigs.has(key)) {
      // 使用合理的默认值
      uniqueConfigs.set(key, {
        ...config,
        microBatchSize: Math.max(1, Math.floor(32 / config.pipelineParallel)),
        gradientAccumulation: config.pipelineParallel > 1 ? config.pipelineParallel * 2 : 1,
      });
    }
  }
  
  return Array.from(uniqueConfigs.values());
}

/**
 * 验证配置是否满足约束
 */
export function validateConfiguration(
  config: ParallelConfig,
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig
): { isValid: boolean; errors: string[] } {
  const errors: string[] = [];
  const extendedHW = hardwareConfig as ExtendedHardwareConfig;
  
  // 检查内存约束
  const memoryAnalysis = analyzeMemory(modelConfig, hardwareConfig, inferenceConfig, {
    parallelConfig: config,
  });
  
  if (extendedHW.memoryCapacity) {
    const gpuMemoryBytes = extendedHW.memoryCapacity * 1e9;
    if (memoryAnalysis.peakMemory > gpuMemoryBytes * 0.95) {
      errors.push(`Memory exceeds GPU capacity: ${(memoryAnalysis.peakMemory / 1e9).toFixed(1)}GB > ${extendedHW.memoryCapacity}GB`);
    }
  }
  
  // 检查TP约束
  if (modelConfig.numAttentionHeads % config.tensorParallel !== 0) {
    errors.push(`TP size ${config.tensorParallel} does not divide attention heads ${modelConfig.numAttentionHeads}`);
  }
  
  // 检查PP约束
  if (modelConfig.numLayers % config.pipelineParallel !== 0) {
    errors.push(`PP size ${config.pipelineParallel} does not divide layers ${modelConfig.numLayers}`);
  }
  
  // 检查micro batch size
  if (config.microBatchSize > inferenceConfig.batchSize) {
    errors.push(`Micro batch size ${config.microBatchSize} exceeds batch size ${inferenceConfig.batchSize}`);
  }
  
  // PP需要足够的micro batches来隐藏bubble
  if (config.pipelineParallel > 1) {
    const numMicroBatches = Math.ceil(inferenceConfig.batchSize / config.microBatchSize) * config.gradientAccumulation;
    if (numMicroBatches < config.pipelineParallel * 2) {
      errors.push(`Too few micro-batches (${numMicroBatches}) for PP=${config.pipelineParallel}, recommend >= ${config.pipelineParallel * 2}`);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * 分析单个策略配置
 */
export function analyzeStrategy(
  config: ParallelConfig,
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig
): StrategyAnalysis {
  // 验证配置
  const validation = validateConfiguration(config, modelConfig, hardwareConfig, inferenceConfig);
  
  // 计算内存
  const memoryAnalysis = analyzeMemory(modelConfig, hardwareConfig, inferenceConfig, {
    parallelConfig: config,
    useGradientCheckpointing: config.pipelineParallel > 1, // PP通常需要gradient checkpointing
  });
  
  // 计算单GPU时间
  const modelAnalysis = analyzeModel(modelConfig, hardwareConfig, inferenceConfig);
  const singleGPUTimeMs = modelAnalysis.estimatedLatency;
  
  // 计算通信开销
  const communicationAnalysis = analyzeCommunication(
    modelConfig, hardwareConfig, inferenceConfig, config, singleGPUTimeMs
  );
  
  // 估算吞吐量
  const throughputEstimate = estimateThroughput(
    modelConfig, hardwareConfig, inferenceConfig, config, singleGPUTimeMs
  );
  
  // 构建策略配置
  const strategyConfig: StrategyConfig = {
    ...config,
    name: `DP${config.dataParallel}-TP${config.tensorParallel}-PP${config.pipelineParallel}`,
  };
  
  // 计算综合评分
  const score = calculateStrategyScore(
    throughputEstimate.tokensPerSecond,
    throughputEstimate.timePerStepMs,
    memoryAnalysis,
    communicationAnalysis,
    validation.isValid
  );
  
  return {
    config: strategyConfig,
    memoryAnalysis,
    communicationAnalysis,
    throughputTokensPerSec: throughputEstimate.tokensPerSecond,
    latencyMs: throughputEstimate.timePerStepMs,
    isValid: validation.isValid,
    validationErrors: validation.errors,
    score,
  };
}

/**
 * 计算策略综合评分
 */
function calculateStrategyScore(
  throughput: number,
  latency: number,
  memory: MemoryAnalysis,
  comm: CommunicationAnalysis,
  isValid: boolean
): number {
  if (!isValid) return 0;
  
  // 归一化各指标 (假设合理范围)
  const throughputScore = Math.min(throughput / 100000, 1); // 100k tokens/s为满分
  const latencyScore = Math.max(0, 1 - latency / 1000);     // 1s以下为正分
  const efficiencyScore = comm.efficiency;                   // 0-1
  const memoryScore = memory.utilizationPercent 
    ? Math.max(0, 1 - (memory.utilizationPercent - 50) / 50) // 50-100%为负分
    : 0.5;
  
  // 加权平均
  const weights = {
    throughput: 0.4,
    latency: 0.2,
    efficiency: 0.3,
    memory: 0.1,
  };
  
  return (
    throughputScore * weights.throughput +
    latencyScore * weights.latency +
    efficiencyScore * weights.efficiency +
    memoryScore * weights.memory
  );
}

/**
 * 找到最优并行策略
 */
export function findOptimalStrategies(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  numGPUs: number,
  objective: 'throughput' | 'latency' | 'memory' = 'throughput'
): StrategyComparison {
  // 生成所有可能的配置
  const configs = generateValidConfigurations(numGPUs, modelConfig);
  
  // 分析每个配置
  const strategies: StrategyAnalysis[] = configs.map(config => 
    analyzeStrategy(config, modelConfig, hardwareConfig, inferenceConfig)
  );
  
  // 根据目标排序
  const sortedStrategies = [...strategies].sort((a, b) => {
    // 无效配置排在最后
    if (!a.isValid && b.isValid) return 1;
    if (a.isValid && !b.isValid) return -1;
    if (!a.isValid && !b.isValid) return 0;
    
    switch (objective) {
      case 'throughput':
        return b.throughputTokensPerSec - a.throughputTokensPerSec;
      case 'latency':
        return a.latencyMs - b.latencyMs;
      case 'memory':
        return a.memoryAnalysis.peakMemory - b.memoryAnalysis.peakMemory;
      default:
        return b.score - a.score;
    }
  });
  
  // 选择推荐策略
  const validStrategies = sortedStrategies.filter(s => s.isValid);
  const recommendedStrategy = validStrategies.length > 0 ? validStrategies[0] : null;
  
  return {
    strategies: sortedStrategies,
    recommendedStrategy,
    optimizationObjective: objective,
  };
}

/**
 * 快速推荐策略 (不枚举所有配置)
 */
export function quickRecommendStrategy(
  modelConfig: ModelConfig,
  hardwareConfig: HardwareConfig,
  inferenceConfig: InferenceConfig,
  numGPUs: number
): ParallelConfig {
  const extendedHW = hardwareConfig as ExtendedHardwareConfig;
  const gpuMemory = (extendedHW.memoryCapacity || 80) * 1e9;
  
  // 估算单GPU内存需求
  const singleGPUMemory = analyzeMemory(modelConfig, hardwareConfig, inferenceConfig, {
    parallelConfig: { dataParallel: 1, tensorParallel: 1, pipelineParallel: 1, microBatchSize: 1, gradientAccumulation: 1 },
  });
  
  // 确定需要的TP大小 (基于内存)
  let tp = 1;
  while (tp < 8 && singleGPUMemory.modelWeights / tp > gpuMemory * 0.6) {
    tp *= 2;
  }
  // 确保TP能整除attention heads
  while (modelConfig.numAttentionHeads % tp !== 0 && tp > 1) {
    tp /= 2;
  }
  tp = Math.min(tp, numGPUs);
  
  // 确定PP大小 (如果TP不够,使用PP)
  let pp = 1;
  const remainingAfterTP = numGPUs / tp;
  if (singleGPUMemory.modelWeights / tp > gpuMemory * 0.7 && remainingAfterTP > 1) {
    // 需要更多分片
    pp = Math.min(4, remainingAfterTP);
    while (modelConfig.numLayers % pp !== 0 && pp > 1) {
      pp--;
    }
  }
  
  // 剩余GPU用于DP
  const dp = numGPUs / (tp * pp);
  
  // 确定micro batch size
  const microBatchSize = pp > 1 ? Math.max(1, Math.floor(inferenceConfig.batchSize / (pp * 4))) : inferenceConfig.batchSize;
  
  // 确定gradient accumulation
  const gradientAccumulation = pp > 1 ? pp * 2 : 1;
  
  return {
    dataParallel: dp,
    tensorParallel: tp,
    pipelineParallel: pp,
    microBatchSize,
    gradientAccumulation,
  };
}

/**
 * 生成策略说明
 */
export function generateStrategyDescription(config: ParallelConfig): string {
  const parts: string[] = [];
  
  if (config.dataParallel > 1) {
    parts.push(`Data Parallel (${config.dataParallel} replicas)`);
  }
  if (config.tensorParallel > 1) {
    parts.push(`Tensor Parallel (${config.tensorParallel}-way split)`);
  }
  if (config.pipelineParallel > 1) {
    parts.push(`Pipeline Parallel (${config.pipelineParallel} stages)`);
  }
  
  if (parts.length === 0) {
    return 'Single GPU (no parallelism)';
  }
  
  return parts.join(' + ');
}

/**
 * 获取策略优缺点
 */
export function getStrategyProsCons(config: ParallelConfig): {
  pros: string[];
  cons: string[];
} {
  const pros: string[] = [];
  const cons: string[] = [];
  
  const totalGPUs = config.dataParallel * config.tensorParallel * config.pipelineParallel;
  
  if (config.dataParallel > 1) {
    pros.push('Linear scaling with batch size');
    pros.push('Good for large batch training');
    cons.push('Requires gradient synchronization');
    if (config.dataParallel >= 8) {
      cons.push('May require high network bandwidth');
    }
  }
  
  if (config.tensorParallel > 1) {
    pros.push('Reduces per-GPU memory for model weights');
    pros.push('Low latency for inference');
    cons.push('Frequent intra-layer communication');
    if (config.tensorParallel > 4) {
      cons.push('Diminishing returns at high TP');
    }
  }
  
  if (config.pipelineParallel > 1) {
    pros.push('Enables training very large models');
    pros.push('Lower communication volume than TP');
    cons.push('Pipeline bubble overhead');
    cons.push('Requires careful micro-batch tuning');
    if (config.pipelineParallel > 8) {
      cons.push('High bubble overhead');
    }
  }
  
  if (totalGPUs === 1) {
    pros.push('No communication overhead');
    pros.push('Simple to implement and debug');
    cons.push('Limited by single GPU memory and compute');
  }
  
  return { pros, cons };
}
