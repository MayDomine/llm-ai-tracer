/**
 * 预设模型配置
 * 包含 Qwen3、LLaMA2/3、MiniCPM 等主流模型
 */

import type { ModelConfig, HardwareConfig } from '../types/model';

// ============ Qwen3 系列 ============

export const QWEN3_0_6B: ModelConfig = {
  name: 'Qwen3-0.6B',
  hiddenSize: 1024,
  numLayers: 28,
  vocabSize: 151936,
  numAttentionHeads: 16,
  numKVHeads: 8,
  headDim: 64,
  attentionType: 'gqa',
  intermediateSize: 3072,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

export const QWEN3_1_7B: ModelConfig = {
  name: 'Qwen3-1.7B',
  hiddenSize: 2048,
  numLayers: 28,
  vocabSize: 151936,
  numAttentionHeads: 16,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 6144,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

export const QWEN3_4B: ModelConfig = {
  name: 'Qwen3-4B',
  hiddenSize: 2560,
  numLayers: 36,
  vocabSize: 151936,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 80,
  attentionType: 'gqa',
  intermediateSize: 9216,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

export const QWEN3_8B: ModelConfig = {
  name: 'Qwen3-8B',
  hiddenSize: 4096,
  numLayers: 36,
  vocabSize: 151936,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 12288,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

export const QWEN3_14B: ModelConfig = {
  name: 'Qwen3-14B',
  hiddenSize: 5120,
  numLayers: 40,
  vocabSize: 151936,
  numAttentionHeads: 40,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 13696,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

export const QWEN3_32B: ModelConfig = {
  name: 'Qwen3-32B',
  hiddenSize: 5120,
  numLayers: 64,
  vocabSize: 151936,
  numAttentionHeads: 64,
  numKVHeads: 8,
  headDim: 80,
  attentionType: 'gqa',
  intermediateSize: 25600,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

// Qwen3 MoE 系列
export const QWEN3_30B_A3B: ModelConfig = {
  name: 'Qwen3-30B-A3B (MoE)',
  hiddenSize: 2048,
  numLayers: 48,
  vocabSize: 151936,
  numAttentionHeads: 16,
  numKVHeads: 4,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 2048, // 每个专家的intermediate size
  ffnType: 'moe',
  numExperts: 128,
  numExpertsPerToken: 8,
  sharedExpertNum: 1,
  maxSeqLen: 32768,
};

export const QWEN3_235B_A22B: ModelConfig = {
  name: 'Qwen3-235B-A22B (MoE)',
  hiddenSize: 4096,
  numLayers: 94,
  vocabSize: 151936,
  numAttentionHeads: 64,
  numKVHeads: 4,
  headDim: 64,
  attentionType: 'gqa',
  intermediateSize: 3072,
  ffnType: 'moe',
  numExperts: 128,
  numExpertsPerToken: 8,
  sharedExpertNum: 1,
  maxSeqLen: 32768,
};

// ============ LLaMA 2 系列 ============

export const LLAMA2_7B: ModelConfig = {
  name: 'LLaMA2-7B',
  hiddenSize: 4096,
  numLayers: 32,
  vocabSize: 32000,
  numAttentionHeads: 32,
  numKVHeads: 32,
  headDim: 128,
  attentionType: 'mha',
  intermediateSize: 11008,
  ffnType: 'gated',
  maxSeqLen: 4096,
};

export const LLAMA2_13B: ModelConfig = {
  name: 'LLaMA2-13B',
  hiddenSize: 5120,
  numLayers: 40,
  vocabSize: 32000,
  numAttentionHeads: 40,
  numKVHeads: 40,
  headDim: 128,
  attentionType: 'mha',
  intermediateSize: 13824,
  ffnType: 'gated',
  maxSeqLen: 4096,
};

export const LLAMA2_70B: ModelConfig = {
  name: 'LLaMA2-70B',
  hiddenSize: 8192,
  numLayers: 80,
  vocabSize: 32000,
  numAttentionHeads: 64,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 28672,
  ffnType: 'gated',
  maxSeqLen: 4096,
};

// ============ LLaMA 3 系列 ============

export const LLAMA3_8B: ModelConfig = {
  name: 'LLaMA3-8B',
  hiddenSize: 4096,
  numLayers: 32,
  vocabSize: 128256,
  numAttentionHeads: 32,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 14336,
  ffnType: 'gated',
  maxSeqLen: 8192,
};

export const LLAMA3_70B: ModelConfig = {
  name: 'LLaMA3-70B',
  hiddenSize: 8192,
  numLayers: 80,
  vocabSize: 128256,
  numAttentionHeads: 64,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 28672,
  ffnType: 'gated',
  maxSeqLen: 8192,
};

export const LLAMA3_405B: ModelConfig = {
  name: 'LLaMA3-405B',
  hiddenSize: 16384,
  numLayers: 126,
  vocabSize: 128256,
  numAttentionHeads: 128,
  numKVHeads: 8,
  headDim: 128,
  attentionType: 'gqa',
  intermediateSize: 53248,
  ffnType: 'gated',
  maxSeqLen: 8192,
};

// ============ MiniCPM 系列 ============

export const MINICPM_2B: ModelConfig = {
  name: 'MiniCPM-2B',
  hiddenSize: 2304,
  numLayers: 40,
  vocabSize: 122753,
  numAttentionHeads: 36,
  numKVHeads: 36,
  headDim: 64,
  attentionType: 'mha',
  intermediateSize: 5760,
  ffnType: 'gated',
  maxSeqLen: 4096,
};

export const MINICPM_3B: ModelConfig = {
  name: 'MiniCPM3-4B',
  hiddenSize: 2560,
  numLayers: 62,
  vocabSize: 73440,
  numAttentionHeads: 40,
  numKVHeads: 8,
  headDim: 64,
  attentionType: 'gqa',
  intermediateSize: 6400,
  ffnType: 'gated',
  maxSeqLen: 32768,
};

// ============ 所有预设模型 ============

export const PRESET_MODELS: ModelConfig[] = [
  // Qwen3 Dense
  QWEN3_0_6B,
  QWEN3_1_7B,
  QWEN3_4B,
  QWEN3_8B,
  QWEN3_14B,
  QWEN3_32B,
  // Qwen3 MoE
  QWEN3_30B_A3B,
  QWEN3_235B_A22B,
  // LLaMA 2
  LLAMA2_7B,
  LLAMA2_13B,
  LLAMA2_70B,
  // LLaMA 3
  LLAMA3_8B,
  LLAMA3_70B,
  LLAMA3_405B,
  // MiniCPM
  MINICPM_2B,
  MINICPM_3B,
];

// ============ 扩展硬件配置类型 ============

import { GPU_SPECS, type GPUSpec, QUANTIZATION_SPECS, type QuantizationSpec } from './formulaConstants';

export interface ExtendedHardwareConfig extends HardwareConfig {
  memoryCapacity: number;      // GPU显存容量 (GB)
  nvlinkBandwidth?: number;    // NVLink带宽 (GB/s, 节点内)
  networkBandwidth?: number;   // 网络带宽 (GB/s, 跨节点)
  gpusPerNode?: number;        // 每节点GPU数量
  
  // Extended specs from GPU database
  architecture?: string;
  fp32TFLOPS?: number;
  int8TOPS?: number;
  fp8TFLOPS?: number;
  memoryType?: string;
  tdpWatts?: number;
  rooflinePointFP16?: number;  // FLOP/Byte
}

/**
 * Convert GPUSpec to ExtendedHardwareConfig
 */
export function gpuSpecToHardwareConfig(spec: GPUSpec, gpusPerNode: number = 8): ExtendedHardwareConfig {
  return {
    name: spec.name,
    computeCapability: spec.fp16TFLOPS,
    memoryBandwidth: spec.memoryBandwidthTBs,
    memoryCapacity: spec.memoryGB,
    nvlinkBandwidth: spec.nvlinkBandwidthGBs,
    networkBandwidth: 50,  // Default InfiniBand HDR
    gpusPerNode,
    architecture: spec.architecture,
    fp32TFLOPS: spec.fp32TFLOPS,
    int8TOPS: spec.int8TOPS,
    fp8TFLOPS: spec.fp8TFLOPS,
    memoryType: spec.memoryType,
    tdpWatts: spec.tdpWatts,
    rooflinePointFP16: spec.rooflinePointFP16,
  };
}

// Re-export for convenience
export { GPU_SPECS, QUANTIZATION_SPECS };
export type { GPUSpec, QuantizationSpec };

// ============ 预设硬件配置 ============

export const NVIDIA_A100_80G: ExtendedHardwareConfig = {
  name: 'NVIDIA A100 (80GB)',
  computeCapability: 312,      // TFLOPS FP16
  memoryBandwidth: 2.0,        // TB/s
  memoryCapacity: 80,          // GB
  nvlinkBandwidth: 600,        // GB/s (NVLink 3.0)
  networkBandwidth: 50,        // GB/s (InfiniBand HDR)
  gpusPerNode: 8,
};

export const NVIDIA_A100_40G: ExtendedHardwareConfig = {
  name: 'NVIDIA A100 (40GB)',
  computeCapability: 312,
  memoryBandwidth: 1.6,
  memoryCapacity: 40,
  nvlinkBandwidth: 600,
  networkBandwidth: 50,
  gpusPerNode: 8,
};

export const NVIDIA_H100_SXM: ExtendedHardwareConfig = {
  name: 'NVIDIA H100 SXM',
  computeCapability: 989,      // TFLOPS FP16
  memoryBandwidth: 3.35,       // TB/s
  memoryCapacity: 80,          // GB
  nvlinkBandwidth: 900,        // GB/s (NVLink 4.0)
  networkBandwidth: 100,       // GB/s (InfiniBand NDR)
  gpusPerNode: 8,
};

export const NVIDIA_H100_PCIE: ExtendedHardwareConfig = {
  name: 'NVIDIA H100 PCIe',
  computeCapability: 756,
  memoryBandwidth: 2.0,
  memoryCapacity: 80,
  nvlinkBandwidth: 0,          // 无NVLink
  networkBandwidth: 50,
  gpusPerNode: 4,
};

export const NVIDIA_H200_SXM: ExtendedHardwareConfig = {
  name: 'NVIDIA H200 SXM',
  computeCapability: 989,      // 与H100相同
  memoryBandwidth: 4.8,        // TB/s (HBM3e)
  memoryCapacity: 141,         // GB
  nvlinkBandwidth: 900,
  networkBandwidth: 100,
  gpusPerNode: 8,
};

export const NVIDIA_4090: ExtendedHardwareConfig = {
  name: 'NVIDIA RTX 4090',
  computeCapability: 165,      // TFLOPS FP16 (Tensor)
  memoryBandwidth: 1.01,
  memoryCapacity: 24,          // GB
  nvlinkBandwidth: 0,          // 消费级无NVLink
  networkBandwidth: 12.5,      // PCIe 4.0 x16
  gpusPerNode: 2,              // 典型配置
};

export const NVIDIA_3090: ExtendedHardwareConfig = {
  name: 'NVIDIA RTX 3090',
  computeCapability: 71,       // TFLOPS FP16
  memoryBandwidth: 0.936,
  memoryCapacity: 24,
  nvlinkBandwidth: 112,        // NVLink Bridge (2卡)
  networkBandwidth: 12.5,
  gpusPerNode: 2,
};

export const NVIDIA_L40S: ExtendedHardwareConfig = {
  name: 'NVIDIA L40S',
  computeCapability: 362,
  memoryBandwidth: 0.864,
  memoryCapacity: 48,          // GB
  nvlinkBandwidth: 0,
  networkBandwidth: 50,
  gpusPerNode: 8,
};

export const AMD_MI300X: ExtendedHardwareConfig = {
  name: 'AMD MI300X',
  computeCapability: 1307,     // TFLOPS FP16
  memoryBandwidth: 5.3,        // TB/s
  memoryCapacity: 192,         // GB HBM3
  nvlinkBandwidth: 896,        // Infinity Fabric
  networkBandwidth: 100,
  gpusPerNode: 8,
};

export const NVIDIA_B200: ExtendedHardwareConfig = {
  name: 'NVIDIA B200',
  computeCapability: 2250,     // TFLOPS FP16 (estimated)
  memoryBandwidth: 8.0,        // TB/s (HBM3e)
  memoryCapacity: 192,         // GB
  nvlinkBandwidth: 1800,       // NVLink 5.0
  networkBandwidth: 200,       // 400G network
  gpusPerNode: 8,
};

export const PRESET_HARDWARE: ExtendedHardwareConfig[] = [
  NVIDIA_H100_SXM,
  NVIDIA_H100_PCIE,
  NVIDIA_H200_SXM,
  NVIDIA_B200,
  NVIDIA_A100_80G,
  NVIDIA_A100_40G,
  NVIDIA_L40S,
  NVIDIA_4090,
  NVIDIA_3090,
  AMD_MI300X,
];

// 模型分类
export const MODEL_CATEGORIES = {
  'Qwen3': [QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, QWEN3_14B, QWEN3_32B],
  'Qwen3 MoE': [QWEN3_30B_A3B, QWEN3_235B_A22B],
  'LLaMA 2': [LLAMA2_7B, LLAMA2_13B, LLAMA2_70B],
  'LLaMA 3': [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B],
  'MiniCPM': [MINICPM_2B, MINICPM_3B],
};
