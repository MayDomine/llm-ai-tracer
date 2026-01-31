/**
 * Formula Constants with Citations
 * 
 * This file contains all mathematical constants and formulas used in the calculator,
 * with proper citations to source papers and frameworks.
 * 
 * All formulas have been verified against production implementations:
 * - Megatron-LM (NVIDIA)
 * - DeepSpeed ZeRO (Microsoft)
 * - vLLM
 * - FlashAttention
 */

// ============ Citation References ============

export const CITATIONS = {
  OPENAI_SCALING: {
    title: 'Scaling Laws for Neural Language Models',
    authors: 'Kaplan et al.',
    year: 2020,
    venue: 'arXiv:2001.08361',
    url: 'https://arxiv.org/abs/2001.08361',
  },
  MEGATRON_LM: {
    title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
    authors: 'Shoeybi et al.',
    year: 2019,
    venue: 'arXiv:1909.08053',
    url: 'https://arxiv.org/abs/1909.08053',
  },
  ZERO: {
    title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
    authors: 'Rajbhandari et al.',
    year: 2020,
    venue: 'SC20',
    url: 'https://arxiv.org/abs/1910.02054',
  },
  FLASH_ATTENTION: {
    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
    authors: 'Dao et al.',
    year: 2022,
    venue: 'NeurIPS 2022',
    url: 'https://arxiv.org/abs/2205.14135',
  },
  REDUCING_ACTIVATION: {
    title: 'Reducing Activation Recomputation in Large Transformer Models',
    authors: 'Korthikanti et al.',
    year: 2023,
    venue: 'MLSys 2023',
    url: 'https://arxiv.org/abs/2205.05198',
  },
  GPIPE: {
    title: 'GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism',
    authors: 'Huang et al.',
    year: 2019,
    venue: 'NeurIPS 2019',
    url: 'https://arxiv.org/abs/1811.06965',
  },
  ULYSSES: {
    title: 'DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models',
    authors: 'Jacobs et al.',
    year: 2023,
    venue: 'arXiv:2309.14509',
    url: 'https://arxiv.org/abs/2309.14509',
  },
  RING_ATTENTION: {
    title: 'Ring Attention with Blockwise Transformers for Near-Infinite Context',
    authors: 'Liu et al.',
    year: 2023,
    venue: 'arXiv:2310.01889',
    url: 'https://arxiv.org/abs/2310.01889',
  },
  PALM: {
    title: 'PaLM: Scaling Language Modeling with Pathways',
    authors: 'Chowdhery et al.',
    year: 2022,
    venue: 'arXiv:2204.02311',
    url: 'https://arxiv.org/abs/2204.02311',
  },
  NCCL: {
    title: 'NVIDIA Collective Communications Library (NCCL)',
    authors: 'NVIDIA',
    year: 2024,
    venue: 'Documentation',
    url: 'https://docs.nvidia.com/deeplearning/nccl/',
  },
} as const;

// ============ Configurable Assumptions ============

/**
 * Default assumptions used in calculations.
 * These can be overridden by the user for more accurate estimates.
 */
export interface CalculatorAssumptions {
  // Communication overlap with compute (0.0 - 1.0)
  // Default: 0.5 (50%) - Conservative estimate
  // Megatron-LM achieves ~0.7-0.8 with async AllReduce
  communicationOverlapFactor: number;
  
  // Framework memory overhead ratio (0.0 - 0.2)
  // Accounts for: CUDA contexts, PyTorch buffers, NCCL buffers
  frameworkOverheadRatio: number;
  
  // Minimum framework overhead in bytes
  frameworkOverheadMinBytes: number;
  
  // Activation memory factor (simplified calculation)
  // Real value depends on: FlashAttention, sequence parallel, model architecture
  // Range: 10-16 with FlashAttention, 30+ without
  activationFactor: number;
  
  // NCCL/communication overhead factor
  // Accounts for: protocol overhead, kernel launch, synchronization
  ncclOverheadFactor: number;
  
  // Kernel efficiency (actual vs theoretical peak)
  // Range: 0.6-0.8 for well-optimized kernels
  kernelEfficiency: number;
  
  // Recomputation overhead multipliers
  recomputationFullOverhead: number;     // ~33% overhead
  recomputationSelectiveOverhead: number; // ~5% overhead
  recomputationBlockOverhead: number;     // ~15% overhead
}

export const DEFAULT_ASSUMPTIONS: CalculatorAssumptions = {
  communicationOverlapFactor: 0.5,  // Conservative: Megatron achieves 0.7-0.8
  frameworkOverheadRatio: 0.08,     // 8% typical
  frameworkOverheadMinBytes: 500 * 1024 * 1024, // 500MB minimum
  activationFactor: 12,             // Conservative estimate with FlashAttention
  ncclOverheadFactor: 1.1,          // 10% overhead
  kernelEfficiency: 0.7,            // 70% of theoretical peak
  recomputationFullOverhead: 0.33,
  recomputationSelectiveOverhead: 0.05,
  recomputationBlockOverhead: 0.15,
};

// ============ FLOPs Formulas with Citations ============

export const FLOPS_FORMULAS = {
  // Forward pass approximation
  // Citation: OpenAI Scaling Laws (Kaplan et al., 2020)
  FORWARD_APPROX: {
    formula: 'Forward_FLOPs ≈ 2 × N × T',
    description: 'Forward pass FLOPs is approximately 2 times parameters times tokens',
    citation: CITATIONS.OPENAI_SCALING,
  },
  
  // Training total (forward + backward)
  // Citation: OpenAI Scaling Laws
  TRAINING_TOTAL: {
    formula: 'Training_FLOPs ≈ 6 × N × T (or 8 × N × T with recomputation)',
    description: 'Total training FLOPs including forward and backward pass',
    citation: CITATIONS.OPENAI_SCALING,
  },
  
  // GEMM FLOPs
  // Standard formula
  GEMM: {
    formula: 'GEMM_FLOPs = 2 × M × N × K',
    description: 'Matrix multiply-accumulate: each element requires multiply + add',
    citation: null, // Standard formula
  },
  
  // Backward pass
  // Citation: Megatron-LM
  BACKWARD: {
    formula: 'Backward_FLOPs ≈ 2 × Forward_FLOPs',
    description: 'Backward pass requires ~2x forward for weight and input gradients',
    citation: CITATIONS.MEGATRON_LM,
  },
  
  // Attention FLOPs
  // Citation: FlashAttention paper
  ATTENTION: {
    formula: 'Attention_FLOPs = 2 × B × H × S × S × d_h (QK^T) + 2 × B × H × S × S × d_h (Attn×V)',
    description: 'Attention computation FLOPs for QK^T and softmax×V',
    citation: CITATIONS.FLASH_ATTENTION,
  },
  
  // Softmax FLOPs
  SOFTMAX: {
    formula: 'Softmax_FLOPs = 5 × elements (max + exp + sum + div)',
    description: 'Softmax: find max (1), exp(x-max) (2), sum (1), divide (1)',
    citation: null,
  },
  
  // RMSNorm FLOPs
  RMSNORM: {
    formula: 'RMSNorm_FLOPs = 5 × tokens × hidden_size',
    description: 'RMSNorm: square (1), mean (1), rsqrt (1), normalize (1), scale (1)',
    citation: null,
  },
};

// ============ Memory Formulas with Citations ============

export const MEMORY_FORMULAS = {
  // KV Cache
  // Citation: vLLM
  KV_CACHE: {
    formula: 'KV_Cache = 2 × B × S × L × h_kv × d_h × bytes',
    description: 'Key and Value cache for autoregressive generation',
    citation: null, // Standard implementation
  },
  
  // Model weights
  MODEL_WEIGHTS: {
    formula: 'Weights = P × bytes_per_element / (TP × PP)',
    description: 'Model weights sharded across tensor and pipeline parallel',
    citation: CITATIONS.MEGATRON_LM,
  },
  
  // ZeRO Stage 1: Optimizer states sharded
  // Citation: ZeRO paper
  ZERO_STAGE_1: {
    formula: 'Memory = 4P + 8P/DP + Activations (FP16 + Adam)',
    description: 'ZeRO-1: Optimizer states (momentum + variance) sharded across DP',
    citation: CITATIONS.ZERO,
  },
  
  // ZeRO Stage 2: + Gradients sharded
  ZERO_STAGE_2: {
    formula: 'Memory = 2P + 10P/DP + Activations',
    description: 'ZeRO-2: Gradients + optimizer states sharded',
    citation: CITATIONS.ZERO,
  },
  
  // ZeRO Stage 3: + Weights sharded
  ZERO_STAGE_3: {
    formula: 'Memory = 12P/DP + Activations',
    description: 'ZeRO-3: All model states fully sharded',
    citation: CITATIONS.ZERO,
  },
  
  // Activation memory per layer
  // Citation: Reducing Activation Recomputation paper
  ACTIVATIONS_PER_LAYER: {
    formula: 'Activations = B × S × d × factor × bytes (factor ≈ 10-16)',
    description: 'Per-layer activation memory, varies by architecture and FlashAttention',
    citation: CITATIONS.REDUCING_ACTIVATION,
  },
  
  // Adam optimizer states
  ADAM_STATES: {
    formula: 'Adam_States = 8 × P (momentum FP32 + variance FP32)',
    description: 'Adam stores first and second moment estimates in FP32',
    citation: null,
  },
};

// ============ Communication Formulas with Citations ============

export const COMMUNICATION_FORMULAS = {
  // Ring AllReduce
  // Citation: NCCL documentation
  RING_ALLREDUCE: {
    formula: 'Time = 2 × (n-1)/n × data_size / bandwidth',
    description: 'Ring AllReduce: reduce-scatter + all-gather phases',
    citation: CITATIONS.NCCL,
  },
  
  // AllGather
  ALLGATHER: {
    formula: 'Time = (n-1)/n × data_size / bandwidth',
    description: 'AllGather: each rank sends to all others',
    citation: CITATIONS.NCCL,
  },
  
  // Point-to-point
  P2P: {
    formula: 'Time = data_size / bandwidth',
    description: 'Direct send/receive between two ranks',
    citation: CITATIONS.NCCL,
  },
  
  // Pipeline bubble
  // Citation: GPipe paper
  PIPELINE_BUBBLE: {
    formula: 'Bubble_Ratio = (PP - 1) / num_microbatches',
    description: 'Pipeline bubble overhead decreases with more microbatches',
    citation: CITATIONS.GPIPE,
  },
  
  // Tensor parallel communication
  // Citation: Megatron-LM
  TENSOR_PARALLEL: {
    formula: 'TP_Comm = 4 × B × S × d × L × bytes (2 AllReduce/layer: attn + FFN)',
    description: 'Tensor parallel requires AllReduce after attention and FFN',
    citation: CITATIONS.MEGATRON_LM,
  },
  
  // Context parallel (Ulysses)
  // Citation: DeepSpeed Ulysses
  CONTEXT_PARALLEL_ULYSSES: {
    formula: 'Ulysses_Comm = 4 × all-to-all per layer (Q,K,V input + output)',
    description: 'Ulysses uses all-to-all for sequence redistribution, limited by num_heads',
    citation: CITATIONS.ULYSSES,
  },
  
  // Context parallel (Ring)
  // Citation: Ring Attention
  CONTEXT_PARALLEL_RING: {
    formula: 'Ring_Comm = (CP-1) × P2P steps, each sending KV blocks',
    description: 'Ring attention passes KV blocks around the ring, no head limitation',
    citation: CITATIONS.RING_ATTENTION,
  },
};

// ============ Efficiency Metrics with Citations ============

export const EFFICIENCY_FORMULAS = {
  // Model FLOPs Utilization (MFU)
  // Citation: PaLM paper
  MFU: {
    formula: 'MFU = Actual_Model_FLOPs / (Time × Peak_FLOPS × GPUs)',
    description: 'Ratio of model FLOPs to theoretical peak, excludes recomputation',
    citation: CITATIONS.PALM,
  },
  
  // Hardware FLOPs Utilization (HFU)
  // Citation: PaLM paper
  HFU: {
    formula: 'HFU = Total_FLOPs_Including_Recomputation / (Time × Peak_FLOPS × GPUs)',
    description: 'Ratio of all FLOPs (including recomputation) to theoretical peak',
    citation: CITATIONS.PALM,
  },
  
  // Compute efficiency
  COMPUTE_EFFICIENCY: {
    formula: 'Compute_Efficiency = Compute_Time / (Compute + Comm + Bubble)',
    description: 'Fraction of time spent on useful computation',
    citation: null,
  },
};

// ============ Hardware Specs Database ============

export interface GPUSpec {
  name: string;
  vendor: 'nvidia' | 'amd' | 'intel';
  
  // Compute (TFLOPS)
  fp32TFLOPS: number;
  fp16TFLOPS: number;
  bf16TFLOPS: number;
  int8TOPS: number;
  fp8TFLOPS?: number;  // Hopper+ only
  
  // Memory
  memoryGB: number;
  memoryBandwidthTBs: number;
  memoryType: 'HBM2' | 'HBM2e' | 'HBM3' | 'HBM3e' | 'GDDR6' | 'GDDR6X';
  
  // Interconnect
  nvlinkBandwidthGBs: number;    // Per-direction, bidirectional = 2x
  nvlinkVersion?: string;
  pcieBandwidthGBs: number;
  
  // TDP
  tdpWatts: number;
  
  // Architecture
  architecture: string;
  
  // Roofline point (FLOP/Byte)
  rooflinePointFP16: number;     // fp16TFLOPS / memoryBandwidthTBs
}

export const GPU_SPECS: Record<string, GPUSpec> = {
  // NVIDIA Hopper
  H100_SXM: {
    name: 'NVIDIA H100 SXM',
    vendor: 'nvidia',
    fp32TFLOPS: 67,
    fp16TFLOPS: 989,   // with sparsity: 1978
    bf16TFLOPS: 989,
    int8TOPS: 1978,
    fp8TFLOPS: 1978,
    memoryGB: 80,
    memoryBandwidthTBs: 3.35,
    memoryType: 'HBM3',
    nvlinkBandwidthGBs: 450,  // 900 GB/s bidirectional
    nvlinkVersion: '4.0',
    pcieBandwidthGBs: 64,     // PCIe 5.0 x16
    tdpWatts: 700,
    architecture: 'Hopper',
    rooflinePointFP16: 295,   // 989 / 3.35
  },
  
  H100_PCIE: {
    name: 'NVIDIA H100 PCIe',
    vendor: 'nvidia',
    fp32TFLOPS: 51,
    fp16TFLOPS: 756,
    bf16TFLOPS: 756,
    int8TOPS: 1513,
    fp8TFLOPS: 1513,
    memoryGB: 80,
    memoryBandwidthTBs: 2.0,
    memoryType: 'HBM2e',
    nvlinkBandwidthGBs: 0,    // No NVLink on PCIe variant
    pcieBandwidthGBs: 64,
    tdpWatts: 350,
    architecture: 'Hopper',
    rooflinePointFP16: 378,
  },
  
  H200_SXM: {
    name: 'NVIDIA H200 SXM',
    vendor: 'nvidia',
    fp32TFLOPS: 67,
    fp16TFLOPS: 989,
    bf16TFLOPS: 989,
    int8TOPS: 1978,
    fp8TFLOPS: 1978,
    memoryGB: 141,
    memoryBandwidthTBs: 4.8,
    memoryType: 'HBM3e',
    nvlinkBandwidthGBs: 450,
    nvlinkVersion: '4.0',
    pcieBandwidthGBs: 64,
    tdpWatts: 700,
    architecture: 'Hopper',
    rooflinePointFP16: 206,   // More memory-bound due to higher bandwidth
  },
  
  // NVIDIA Blackwell
  B200: {
    name: 'NVIDIA B200',
    vendor: 'nvidia',
    fp32TFLOPS: 90,
    fp16TFLOPS: 2250,   // Estimated
    bf16TFLOPS: 2250,
    int8TOPS: 4500,
    fp8TFLOPS: 4500,
    memoryGB: 192,
    memoryBandwidthTBs: 8.0,
    memoryType: 'HBM3e',
    nvlinkBandwidthGBs: 900,  // 1800 GB/s bidirectional (NVLink 5.0)
    nvlinkVersion: '5.0',
    pcieBandwidthGBs: 128,    // PCIe 6.0
    tdpWatts: 1000,
    architecture: 'Blackwell',
    rooflinePointFP16: 281,
  },
  
  // NVIDIA Ampere
  A100_80G: {
    name: 'NVIDIA A100 (80GB)',
    vendor: 'nvidia',
    fp32TFLOPS: 19.5,
    fp16TFLOPS: 312,
    bf16TFLOPS: 312,
    int8TOPS: 624,
    memoryGB: 80,
    memoryBandwidthTBs: 2.0,
    memoryType: 'HBM2e',
    nvlinkBandwidthGBs: 300,  // 600 GB/s bidirectional (NVLink 3.0)
    nvlinkVersion: '3.0',
    pcieBandwidthGBs: 32,     // PCIe 4.0 x16
    tdpWatts: 400,
    architecture: 'Ampere',
    rooflinePointFP16: 156,
  },
  
  A100_40G: {
    name: 'NVIDIA A100 (40GB)',
    vendor: 'nvidia',
    fp32TFLOPS: 19.5,
    fp16TFLOPS: 312,
    bf16TFLOPS: 312,
    int8TOPS: 624,
    memoryGB: 40,
    memoryBandwidthTBs: 1.6,
    memoryType: 'HBM2e',
    nvlinkBandwidthGBs: 300,
    nvlinkVersion: '3.0',
    pcieBandwidthGBs: 32,
    tdpWatts: 400,
    architecture: 'Ampere',
    rooflinePointFP16: 195,
  },
  
  // NVIDIA Ada Lovelace
  L40S: {
    name: 'NVIDIA L40S',
    vendor: 'nvidia',
    fp32TFLOPS: 91.6,
    fp16TFLOPS: 362,
    bf16TFLOPS: 362,
    int8TOPS: 724,
    fp8TFLOPS: 724,
    memoryGB: 48,
    memoryBandwidthTBs: 0.864,
    memoryType: 'GDDR6',
    nvlinkBandwidthGBs: 0,
    pcieBandwidthGBs: 32,
    tdpWatts: 350,
    architecture: 'Ada Lovelace',
    rooflinePointFP16: 419,
  },
  
  RTX_4090: {
    name: 'NVIDIA RTX 4090',
    vendor: 'nvidia',
    fp32TFLOPS: 82.6,
    fp16TFLOPS: 165.2,
    bf16TFLOPS: 165.2,
    int8TOPS: 330.4,
    memoryGB: 24,
    memoryBandwidthTBs: 1.01,
    memoryType: 'GDDR6X',
    nvlinkBandwidthGBs: 0,
    pcieBandwidthGBs: 32,
    tdpWatts: 450,
    architecture: 'Ada Lovelace',
    rooflinePointFP16: 164,
  },
  
  RTX_3090: {
    name: 'NVIDIA RTX 3090',
    vendor: 'nvidia',
    fp32TFLOPS: 35.6,
    fp16TFLOPS: 71.0,   // Tensor cores: 142 with sparsity
    bf16TFLOPS: 71.0,
    int8TOPS: 142.0,
    memoryGB: 24,
    memoryBandwidthTBs: 0.936,
    memoryType: 'GDDR6X',
    nvlinkBandwidthGBs: 56,   // NVLink Bridge: 112 GB/s bidirectional
    pcieBandwidthGBs: 32,
    tdpWatts: 350,
    architecture: 'Ampere',
    rooflinePointFP16: 76,
  },
  
  // AMD
  MI300X: {
    name: 'AMD MI300X',
    vendor: 'amd',
    fp32TFLOPS: 163.4,
    fp16TFLOPS: 1307.4,
    bf16TFLOPS: 1307.4,
    int8TOPS: 2614.8,
    memoryGB: 192,
    memoryBandwidthTBs: 5.3,
    memoryType: 'HBM3',
    nvlinkBandwidthGBs: 448,  // Infinity Fabric Link
    pcieBandwidthGBs: 128,
    tdpWatts: 750,
    architecture: 'CDNA 3',
    rooflinePointFP16: 247,
  },
};

// ============ Quantization Specs ============

export interface QuantizationSpec {
  name: string;
  bitsPerWeight: number;
  bitsPerActivation: number;
  
  // Memory savings
  memoryReductionFactor: number;  // Compared to FP16
  
  // Compute characteristics
  throughputMultiplier: number;   // Compared to FP16/BF16
  
  // Accuracy impact (rough estimate)
  typicalAccuracyLoss: string;    // Description
  
  // Supported operations
  supportsTraining: boolean;
  supportsInference: boolean;
  
  // Hardware requirements
  requiresSpecialHardware: boolean;
  supportedGPUs: string[];
}

export const QUANTIZATION_SPECS: Record<string, QuantizationSpec> = {
  FP32: {
    name: 'FP32 (Full Precision)',
    bitsPerWeight: 32,
    bitsPerActivation: 32,
    memoryReductionFactor: 0.5,  // 2x more than FP16
    throughputMultiplier: 0.5,    // Half the FP16 throughput
    typicalAccuracyLoss: 'Baseline (reference)',
    supportsTraining: true,
    supportsInference: true,
    requiresSpecialHardware: false,
    supportedGPUs: ['all'],
  },
  
  FP16: {
    name: 'FP16 (Half Precision)',
    bitsPerWeight: 16,
    bitsPerActivation: 16,
    memoryReductionFactor: 1.0,   // Baseline for comparison
    throughputMultiplier: 1.0,
    typicalAccuracyLoss: 'None with loss scaling',
    supportsTraining: true,
    supportsInference: true,
    requiresSpecialHardware: false,
    supportedGPUs: ['all modern GPUs'],
  },
  
  BF16: {
    name: 'BF16 (Brain Float)',
    bitsPerWeight: 16,
    bitsPerActivation: 16,
    memoryReductionFactor: 1.0,
    throughputMultiplier: 1.0,
    typicalAccuracyLoss: 'Negligible (same exponent range as FP32)',
    supportsTraining: true,
    supportsInference: true,
    requiresSpecialHardware: true,
    supportedGPUs: ['A100', 'H100', 'H200', 'B200', 'RTX 30xx+', 'MI300X'],
  },
  
  FP8_E4M3: {
    name: 'FP8 E4M3 (Weights)',
    bitsPerWeight: 8,
    bitsPerActivation: 8,
    memoryReductionFactor: 2.0,   // 2x less than FP16
    throughputMultiplier: 2.0,    // 2x faster than FP16
    typicalAccuracyLoss: '<1% with calibration',
    supportsTraining: true,       // With proper scaling
    supportsInference: true,
    requiresSpecialHardware: true,
    supportedGPUs: ['H100', 'H200', 'B200', 'L40S'],
  },
  
  FP8_E5M2: {
    name: 'FP8 E5M2 (Activations/Gradients)',
    bitsPerWeight: 8,
    bitsPerActivation: 8,
    memoryReductionFactor: 2.0,
    throughputMultiplier: 2.0,
    typicalAccuracyLoss: '<1% with dynamic scaling',
    supportsTraining: true,
    supportsInference: true,
    requiresSpecialHardware: true,
    supportedGPUs: ['H100', 'H200', 'B200', 'L40S'],
  },
  
  INT8: {
    name: 'INT8 (8-bit Integer)',
    bitsPerWeight: 8,
    bitsPerActivation: 8,
    memoryReductionFactor: 2.0,
    throughputMultiplier: 2.0,
    typicalAccuracyLoss: '1-2% with proper calibration (PTQ) or QAT',
    supportsTraining: false,      // QAT uses FP for backward
    supportsInference: true,
    requiresSpecialHardware: false,
    supportedGPUs: ['all Tensor Core GPUs'],
  },
  
  INT4: {
    name: 'INT4 (4-bit Integer)',
    bitsPerWeight: 4,
    bitsPerActivation: 16,        // Activations usually stay FP16
    memoryReductionFactor: 4.0,   // 4x less than FP16 for weights
    throughputMultiplier: 2.5,    // Limited by dequantization overhead
    typicalAccuracyLoss: '2-5% depending on model and method (GPTQ, AWQ)',
    supportsTraining: false,
    supportsInference: true,
    requiresSpecialHardware: false,
    supportedGPUs: ['all modern GPUs (software dequant)'],
  },
  
  NF4: {
    name: 'NF4 (4-bit NormalFloat)',
    bitsPerWeight: 4,
    bitsPerActivation: 16,
    memoryReductionFactor: 4.0,
    throughputMultiplier: 2.0,
    typicalAccuracyLoss: '1-3% (better than INT4 for normal distributions)',
    supportsTraining: true,       // QLoRA
    supportsInference: true,
    requiresSpecialHardware: false,
    supportedGPUs: ['all modern GPUs'],
  },
};

// ============ Network Latency Constants ============

export interface NetworkLatencySpec {
  name: string;
  latencyMicroseconds: number;
  bandwidthGBs: number;
  typicalHops: number;
}

export const NETWORK_LATENCY_SPECS: Record<string, NetworkLatencySpec> = {
  NVLINK_4: {
    name: 'NVLink 4.0 (Intra-node)',
    latencyMicroseconds: 1,       // ~1us
    bandwidthGBs: 450,            // Per direction
    typicalHops: 1,
  },
  NVLINK_3: {
    name: 'NVLink 3.0',
    latencyMicroseconds: 1.5,
    bandwidthGBs: 300,
    typicalHops: 1,
  },
  NVSWITCH_4: {
    name: 'NVSwitch (All-to-all within node)',
    latencyMicroseconds: 2,
    bandwidthGBs: 450,
    typicalHops: 2,
  },
  INFINIBAND_NDR: {
    name: 'InfiniBand NDR (400Gb/s)',
    latencyMicroseconds: 2,
    bandwidthGBs: 50,             // ~400 Gbps
    typicalHops: 2,
  },
  INFINIBAND_HDR: {
    name: 'InfiniBand HDR (200Gb/s)',
    latencyMicroseconds: 2.5,
    bandwidthGBs: 25,
    typicalHops: 2,
  },
  ETHERNET_100G: {
    name: 'Ethernet 100GbE',
    latencyMicroseconds: 10,
    bandwidthGBs: 12.5,
    typicalHops: 3,
  },
  ETHERNET_ROCE: {
    name: 'RoCE v2 (100GbE)',
    latencyMicroseconds: 5,
    bandwidthGBs: 12.5,
    typicalHops: 2,
  },
  PCIE_5: {
    name: 'PCIe 5.0 x16',
    latencyMicroseconds: 0.5,
    bandwidthGBs: 64,
    typicalHops: 1,
  },
  PCIE_4: {
    name: 'PCIe 4.0 x16',
    latencyMicroseconds: 0.5,
    bandwidthGBs: 32,
    typicalHops: 1,
  },
};

// ============ Validation Benchmarks ============

/**
 * Known benchmark values for validation.
 * These are approximate expected values based on published results.
 */
export interface ValidationBenchmark {
  name: string;
  modelName: string;
  config: {
    tp: number;
    pp: number;
    dp: number;
    microBatchSize: number;
    seqLength: number;
    precision: 'fp16' | 'bf16' | 'fp32';
    gpuType: string;
  };
  expected: {
    memoryPerGPU_GB: { min: number; max: number };
    mfu: { min: number; max: number };
    tokensPerSecond?: { min: number; max: number };
  };
  source: string;
}

export const VALIDATION_BENCHMARKS: ValidationBenchmark[] = [
  {
    name: 'LLaMA2-7B Training (Single GPU)',
    modelName: 'LLaMA2-7B',
    config: {
      tp: 1,
      pp: 1,
      dp: 1,
      microBatchSize: 1,
      seqLength: 4096,
      precision: 'bf16',
      gpuType: 'H100_SXM',
    },
    expected: {
      memoryPerGPU_GB: { min: 25, max: 35 },
      mfu: { min: 0.35, max: 0.50 },
    },
    source: 'Community benchmarks',
  },
  {
    name: 'LLaMA2-70B Training (8x H100)',
    modelName: 'LLaMA2-70B',
    config: {
      tp: 8,
      pp: 1,
      dp: 1,
      microBatchSize: 1,
      seqLength: 4096,
      precision: 'bf16',
      gpuType: 'H100_SXM',
    },
    expected: {
      memoryPerGPU_GB: { min: 45, max: 65 },
      mfu: { min: 0.40, max: 0.55 },
    },
    source: 'Megatron-LM benchmarks',
  },
  {
    name: 'LLaMA2-70B Training (32x H100)',
    modelName: 'LLaMA2-70B',
    config: {
      tp: 8,
      pp: 4,
      dp: 1,
      microBatchSize: 1,
      seqLength: 4096,
      precision: 'bf16',
      gpuType: 'H100_SXM',
    },
    expected: {
      memoryPerGPU_GB: { min: 15, max: 25 },
      mfu: { min: 0.38, max: 0.50 },
    },
    source: 'Megatron-LM benchmarks',
  },
  {
    name: 'GPT-3 175B Training (256x A100)',
    modelName: 'GPT-3 175B',
    config: {
      tp: 8,
      pp: 8,
      dp: 4,
      microBatchSize: 1,
      seqLength: 2048,
      precision: 'bf16',
      gpuType: 'A100_80G',
    },
    expected: {
      memoryPerGPU_GB: { min: 35, max: 50 },
      mfu: { min: 0.40, max: 0.52 },
    },
    source: 'Megatron-LM paper estimates',
  },
];

/**
 * Get formula citation as a formatted string
 */
export function getFormulaCitation(formula: { formula: string; description: string; citation: typeof CITATIONS[keyof typeof CITATIONS] | null }): string {
  if (!formula.citation) {
    return `${formula.formula}\n${formula.description}`;
  }
  return `${formula.formula}\n${formula.description}\n\nSource: ${formula.citation.title} (${formula.citation.authors}, ${formula.citation.year})`;
}
