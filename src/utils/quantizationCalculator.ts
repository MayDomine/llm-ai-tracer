/**
 * Quantization Calculator
 * 
 * Calculates memory and compute requirements for quantized models.
 * Supports INT8, INT4, FP8, and mixed precision configurations.
 * 
 * References:
 * - GPTQ: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
 * - AWQ: "AWQ: Activation-aware Weight Quantization"
 * - QLoRA: "QLoRA: Efficient Finetuning of Quantized LLMs"
 * - FP8: "FP8 Formats for Deep Learning" (NVIDIA)
 */

import type { ModelConfig, QuantizationConfig, PrecisionType } from '../types/model';
import { getPrecisionBytes, getPrecisionThroughputMultiplier } from '../types/model';
import { calculateModelParameters } from './calculator';
import { QUANTIZATION_SPECS } from '../config/formulaConstants';

// ============ Quantization Memory Calculation ============

export interface QuantizedMemoryBreakdown {
  // Weight memory
  modelWeightsBytes: number;
  modelWeightsQuantizedBytes: number;
  
  // Additional quantization overhead
  scaleFactorsBytes: number;     // Per-group/per-channel scales
  zeroPointsBytes: number;       // For asymmetric quantization
  
  // KV cache (inference)
  kvCacheBytes: number;
  
  // Activations
  activationsBytes: number;
  
  // Total
  totalBytes: number;
  
  // Compression ratio
  compressionRatio: number;      // vs FP16 baseline
  
  // Breakdown by component
  breakdown: {
    name: string;
    bytes: number;
    precision: string;
  }[];
}

/**
 * Calculate memory for quantized model weights
 * 
 * For block-wise quantization (INT4/INT8 with groups):
 * - Weights: params × bits / 8
 * - Scales: params / group_size × 2 bytes (FP16)
 * - Zero points: params / group_size × 0.5 bytes (INT4) or 1 byte (INT8)
 */
export function calculateQuantizedWeightsMemory(
  modelConfig: ModelConfig,
  quantConfig: QuantizationConfig
): { weightsBytes: number; scalesBytes: number; zeroPointsBytes: number } {
  const params = calculateModelParameters(modelConfig);
  const totalParams = params.total;
  
  const weightBytes = getPrecisionBytes(quantConfig.weightPrecision);
  const weightsBytes = totalParams * weightBytes;
  
  // Calculate scale and zero point overhead for block-wise quantization
  let scalesBytes = 0;
  let zeroPointsBytes = 0;
  
  if (quantConfig.weightPrecision === 'int4' || 
      quantConfig.weightPrecision === 'nf4' || 
      quantConfig.weightPrecision === 'int8') {
    
    const groupSize = quantConfig.groupSize || 128; // Default group size
    const numGroups = Math.ceil(totalParams / groupSize);
    
    // Scales are typically FP16
    scalesBytes = numGroups * 2;
    
    // Zero points for asymmetric quantization
    if (quantConfig.method !== 'awq') { // AWQ uses symmetric quantization
      if (quantConfig.weightPrecision === 'int4') {
        zeroPointsBytes = numGroups * 0.5; // 4-bit zero points
      } else if (quantConfig.weightPrecision === 'int8') {
        zeroPointsBytes = numGroups * 1; // 8-bit zero points
      }
    }
  }
  
  return {
    weightsBytes,
    scalesBytes,
    zeroPointsBytes,
  };
}

/**
 * Calculate KV cache memory with potential quantization
 * 
 * Quantized KV cache can significantly reduce memory:
 * - FP8: 2x reduction vs FP16
 * - INT8: 2x reduction vs FP16
 * - INT4: 4x reduction vs FP16 (with some accuracy loss)
 */
export function calculateQuantizedKVCacheMemory(
  modelConfig: ModelConfig,
  batchSize: number,
  seqLen: number,
  kvCachePrecision: PrecisionType
): number {
  const { numLayers, numKVHeads, headDim } = modelConfig;
  const bytesPerElement = getPrecisionBytes(kvCachePrecision);
  
  // KV Cache = 2 × B × S × L × h_kv × d_h × bytes
  return 2 * batchSize * seqLen * numLayers * numKVHeads * headDim * bytesPerElement;
}

/**
 * Calculate complete quantized model memory
 */
export function calculateQuantizedMemory(
  modelConfig: ModelConfig,
  quantConfig: QuantizationConfig,
  batchSize: number,
  seqLen: number,
  isTraining: boolean = false
): QuantizedMemoryBreakdown {
  // Weights
  const { weightsBytes, scalesBytes, zeroPointsBytes } = 
    calculateQuantizedWeightsMemory(modelConfig, quantConfig);
  
  // Calculate FP16 baseline for comparison
  const params = calculateModelParameters(modelConfig);
  const fp16WeightsBytes = params.total * 2;
  
  // KV Cache
  const kvCacheBytes = calculateQuantizedKVCacheMemory(
    modelConfig, batchSize, seqLen, quantConfig.kvCachePrecision
  );
  
  // Activations (typically kept at higher precision)
  const activationBytes = getPrecisionBytes(quantConfig.activationPrecision);
  const { hiddenSize } = modelConfig;
  const activationFactor = 12; // Conservative estimate
  const activationsBytes = batchSize * seqLen * hiddenSize * activationFactor * activationBytes;
  
  // Training-specific memory
  let gradientBytes = 0;
  let optimizerBytes = 0;
  
  if (isTraining) {
    const gradPrecision = quantConfig.gradientPrecision || 'fp16';
    gradientBytes = params.total * getPrecisionBytes(gradPrecision);
    
    // Optimizer states (always FP32 for stability)
    // For QLoRA: only LoRA adapter parameters need optimizer states
    if (quantConfig.method === 'qlora') {
      // LoRA typically adds 0.1-1% of parameters
      const loraRatio = 0.01;
      optimizerBytes = params.total * loraRatio * 8; // momentum + variance
    } else {
      optimizerBytes = params.total * 8; // Full Adam
    }
  }
  
  const totalBytes = weightsBytes + scalesBytes + zeroPointsBytes + 
    kvCacheBytes + activationsBytes + gradientBytes + optimizerBytes;
  
  const compressionRatio = fp16WeightsBytes / (weightsBytes + scalesBytes + zeroPointsBytes);
  
  const breakdown = [
    { name: 'Model Weights', bytes: weightsBytes, precision: quantConfig.weightPrecision },
    { name: 'Scale Factors', bytes: scalesBytes, precision: 'fp16' },
    { name: 'Zero Points', bytes: zeroPointsBytes, precision: quantConfig.weightPrecision },
    { name: 'KV Cache', bytes: kvCacheBytes, precision: quantConfig.kvCachePrecision },
    { name: 'Activations', bytes: activationsBytes, precision: quantConfig.activationPrecision },
  ];
  
  if (isTraining) {
    breakdown.push({ name: 'Gradients', bytes: gradientBytes, precision: quantConfig.gradientPrecision || 'fp16' });
    breakdown.push({ name: 'Optimizer', bytes: optimizerBytes, precision: 'fp32' });
  }
  
  return {
    modelWeightsBytes: fp16WeightsBytes,
    modelWeightsQuantizedBytes: weightsBytes + scalesBytes + zeroPointsBytes,
    scaleFactorsBytes: scalesBytes,
    zeroPointsBytes,
    kvCacheBytes,
    activationsBytes,
    totalBytes,
    compressionRatio,
    breakdown,
  };
}

// ============ Quantization Compute Analysis ============

export interface QuantizedComputeAnalysis {
  // Effective TFLOPS with quantization
  effectiveTFLOPS: number;
  
  // Throughput improvement vs FP16
  throughputMultiplier: number;
  
  // Memory bandwidth improvement
  memoryBandwidthSavings: number;
  
  // Accuracy impact estimate
  expectedAccuracyImpact: string;
  
  // Hardware requirements
  requiresSpecialHardware: boolean;
  supportedGPUs: string[];
}

/**
 * Analyze compute characteristics with quantization
 */
export function analyzeQuantizedCompute(
  baseTFLOPS: number,
  quantConfig: QuantizationConfig,
  gpuArchitecture?: string
): QuantizedComputeAnalysis {
  const spec = QUANTIZATION_SPECS[quantConfig.weightPrecision.toUpperCase()] || 
               QUANTIZATION_SPECS.FP16;
  
  const throughputMultiplier = getPrecisionThroughputMultiplier(quantConfig.weightPrecision);
  const effectiveTFLOPS = baseTFLOPS * throughputMultiplier;
  
  const memoryBandwidthSavings = spec.memoryReductionFactor;
  
  // Check hardware compatibility
  let requiresSpecialHardware = spec.requiresSpecialHardware;
  let supportedGPUs = spec.supportedGPUs;
  
  // FP8 requires Hopper or later
  if (quantConfig.weightPrecision.startsWith('fp8')) {
    requiresSpecialHardware = true;
    supportedGPUs = ['H100', 'H200', 'B200', 'L40S'];
    
    if (gpuArchitecture && !['Hopper', 'Ada Lovelace', 'Blackwell'].includes(gpuArchitecture)) {
      requiresSpecialHardware = true;
    }
  }
  
  return {
    effectiveTFLOPS,
    throughputMultiplier,
    memoryBandwidthSavings,
    expectedAccuracyImpact: spec.typicalAccuracyLoss,
    requiresSpecialHardware,
    supportedGPUs,
  };
}

// ============ Quantization Recommendations ============

export interface QuantizationRecommendation {
  config: QuantizationConfig;
  memorySavingsPercent: number;
  throughputGainPercent: number;
  accuracyImpact: 'none' | 'minimal' | 'moderate' | 'significant';
  recommendation: 'highly_recommended' | 'recommended' | 'consider' | 'not_recommended';
  rationale: string;
}

/**
 * Get quantization recommendations based on use case
 */
export function getQuantizationRecommendations(
  modelConfig: ModelConfig,
  targetGpuMemoryGB: number,
  useCase: 'inference' | 'training' | 'finetuning',
  gpuArchitecture?: string
): QuantizationRecommendation[] {
  const recommendations: QuantizationRecommendation[] = [];
  const params = calculateModelParameters(modelConfig);
  const modelSizeGB = (params.total * 2) / 1e9; // FP16 size
  
  // Calculate how much quantization is needed
  const memoryRatio = modelSizeGB / targetGpuMemoryGB;
  
  if (useCase === 'inference') {
    // FP8 for Hopper+ (best quality/speed tradeoff)
    if (['Hopper', 'Ada Lovelace', 'Blackwell'].includes(gpuArchitecture || '')) {
      recommendations.push({
        config: {
          weightPrecision: 'fp8_e4m3',
          activationPrecision: 'fp8_e5m2',
          kvCachePrecision: 'fp8_e5m2',
        },
        memorySavingsPercent: 50,
        throughputGainPercent: 100,
        accuracyImpact: 'minimal',
        recommendation: 'highly_recommended',
        rationale: 'FP8 provides 2x throughput with minimal accuracy loss on Hopper/Blackwell GPUs',
      });
    }
    
    // INT8 for general inference
    recommendations.push({
      config: {
        weightPrecision: 'int8',
        activationPrecision: 'int8',
        kvCachePrecision: 'fp16',
        method: 'ptq',
      },
      memorySavingsPercent: 50,
      throughputGainPercent: 100,
      accuracyImpact: 'minimal',
      recommendation: memoryRatio > 0.5 ? 'highly_recommended' : 'recommended',
      rationale: 'INT8 provides good balance of memory savings and accuracy',
    });
    
    // INT4 for memory-constrained scenarios
    if (memoryRatio > 0.8) {
      recommendations.push({
        config: {
          weightPrecision: 'int4',
          activationPrecision: 'fp16',
          kvCachePrecision: 'fp16',
          method: 'awq',
          groupSize: 128,
        },
        memorySavingsPercent: 75,
        throughputGainPercent: 50,
        accuracyImpact: 'moderate',
        recommendation: 'consider',
        rationale: 'INT4 (AWQ) for when model barely fits in memory. Consider GPTQ for better accuracy.',
      });
    }
  } else if (useCase === 'finetuning') {
    // QLoRA for efficient finetuning
    recommendations.push({
      config: {
        weightPrecision: 'nf4',
        activationPrecision: 'bf16',
        kvCachePrecision: 'bf16',
        gradientPrecision: 'bf16',
        method: 'qlora',
        groupSize: 64,
      },
      memorySavingsPercent: 70,
      throughputGainPercent: 0, // Training speed similar
      accuracyImpact: 'minimal',
      recommendation: 'highly_recommended',
      rationale: 'QLoRA enables finetuning large models on consumer GPUs with minimal accuracy loss',
    });
  }
  
  return recommendations;
}

/**
 * Format quantization configuration for display
 */
export function formatQuantizationConfig(config: QuantizationConfig): string {
  const parts = [`Weights: ${config.weightPrecision.toUpperCase()}`];
  
  if (config.activationPrecision !== config.weightPrecision) {
    parts.push(`Activations: ${config.activationPrecision.toUpperCase()}`);
  }
  
  if (config.kvCachePrecision !== config.activationPrecision) {
    parts.push(`KV Cache: ${config.kvCachePrecision.toUpperCase()}`);
  }
  
  if (config.method) {
    parts.push(`Method: ${config.method.toUpperCase()}`);
  }
  
  if (config.groupSize) {
    parts.push(`Group size: ${config.groupSize}`);
  }
  
  return parts.join(', ');
}
