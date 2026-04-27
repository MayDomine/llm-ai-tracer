/**
 * Training-specific type definitions
 * Focused on parallelism strategies, batch size calculations, and memory optimization
 */

// ============ Calculator Assumptions (Configurable) ============

/**
 * Configurable assumptions used in calculations.
 * These can be adjusted by users for more accurate estimates based on their setup.
 * 
 * Reference values:
 * - communicationOverlapFactor: 0.5 conservative, 0.7-0.8 with Megatron-style async
 * - frameworkOverheadRatio: 0.05-0.10 typical
 * - ncclOverheadFactor: 1.05-1.15 depending on message size
 */
export interface CalculatorAssumptions {
  // Communication overlap with compute (0.0 - 1.0)
  // Higher = more overlap = better performance
  // Citation: Megatron-LM achieves ~70-80% overlap with async AllReduce
  communicationOverlapFactor: number;
  
  // Framework memory overhead ratio (0.0 - 0.2)
  // Accounts for: CUDA contexts (~200-500MB), PyTorch buffers, NCCL buffers
  frameworkOverheadRatio: number;
  
  // Minimum framework overhead in bytes (default: 500MB)
  frameworkOverheadMinBytes: number;
  
  // NCCL/communication overhead factor (1.0 - 1.2)
  // Accounts for: protocol overhead, kernel launch, synchronization
  ncclOverheadFactor: number;
  
  // Kernel efficiency (0.5 - 0.9)
  // Actual kernel performance vs theoretical peak
  // Range: 0.6-0.8 for well-optimized kernels
  kernelEfficiency: number;

  // Achieved HBM bandwidth vs theoretical peak
  // Range: 0.75-0.95 depending on kernel mix and fusion quality
  memoryBandwidthEfficiency: number;
  
  // Activation memory factor (simplified calculation path)
  // Real value depends on: FlashAttention, sequence parallel, model architecture
  // Range: 10-16 with FlashAttention, 30+ without
  activationFactor: number;
  
  // Recomputation overhead multipliers (fraction of forward time)
  recomputationFullOverhead: number;      // ~0.33 (33% overhead)
  recomputationSelectiveOverhead: number; // ~0.05 (5% overhead)
  recomputationBlockOverhead: number;     // ~0.15 (15% overhead)
}

/**
 * Default assumptions - conservative estimates
 */
export const DEFAULT_CALCULATOR_ASSUMPTIONS: CalculatorAssumptions = {
  communicationOverlapFactor: 0.5,          // Conservative: Megatron achieves 0.7-0.8
  frameworkOverheadRatio: 0.08,             // 8% typical
  frameworkOverheadMinBytes: 500 * 1024 * 1024, // 500MB minimum
  ncclOverheadFactor: 1.1,                  // 10% overhead
  kernelEfficiency: 0.7,                    // 70% of theoretical peak
  memoryBandwidthEfficiency: 0.85,          // Typical achieved HBM bandwidth
  activationFactor: 12,                     // Conservative with FlashAttention
  recomputationFullOverhead: 0.33,
  recomputationSelectiveOverhead: 0.05,
  recomputationBlockOverhead: 0.15,
};

// ============ Batch Size Configuration ============

export interface BatchConfig {
  globalBatchSize: number;      // Total samples per training step
  microBatchSize: number;       // Samples per GPU per forward/backward
  gradientAccumulation: number; // Number of micro-batches before gradient sync
}

// ============ Parallelism Configuration ============

export type ZeROStage = 0 | 1 | 2 | 3;

// Context Parallel types
export type ContextParallelType = 
  | 'ulysses'     // DeepSpeed Ulysses: all-to-all, limited by num_heads
  | 'ring'        // Ring Attention: P2P ring, no head limit
  | 'hybrid';     // Ulysses + Ring combined

export interface ParallelismConfig {
  // Core parallelism dimensions
  dataParallel: number;         // DP_batch size (batch dimension parallelism)
  tensorParallel: number;       // TP size
  pipelineParallel: number;     // PP size
  expertParallel: number;       // EP size (for MoE)
  
  // Context Parallel (for long sequences)
  // Note: CP is a special form of DP that splits sequence instead of batch
  // The ZeRO/FSDP sharding group = DP_batch × CP
  contextParallel: number;      // CP size (sequence dimension parallelism)
  contextParallelType: ContextParallelType; // Ulysses vs Ring vs Hybrid
  
  // Effective DP for ZeRO/FSDP sharding = DP_batch × CP
  // This is the total data parallel dimension across both batch and sequence splits
  effectiveDataParallel: number;
  
  // Additional options
  sequenceParallel: boolean;    // SP (usually enabled with TP)
  zeroStage: ZeROStage;         // ZeRO optimization level
  
  // Derived
  totalGPUs: number;            // DP_batch × CP × TP × PP (EP doesn't consume extra GPUs)
}

// ============ Memory Optimization ============

export type RecomputationStrategy = 
  | 'none'           // No recomputation, store all activations
  | 'full'           // Checkpoint all layer inputs, recompute everything
  | 'selective'      // Selective recomputation (keep cheap, recompute expensive)
  | 'block';         // Checkpoint every N layers

export interface MemoryOptimizationConfig {
  recomputation: RecomputationStrategy;
  recomputationBlocks?: number;   // For 'block' strategy
  cpuOffloading: boolean;         // Offload optimizer states to CPU
  nvmeOffloading: boolean;        // Offload to NVMe (ZeRO-Infinity)
  flashAttention: boolean;        // Use FlashAttention (no softmax matrix storage)
}

// ============ Training Configuration ============

export interface TrainingConfig {
  batch: BatchConfig;
  parallelism: ParallelismConfig;
  memoryOptimization: MemoryOptimizationConfig;
  
  // Sequence configuration
  maxSeqLength: number;
  
  // Precision
  mixedPrecision: 'fp32' | 'fp16' | 'bf16';
  gradientPrecision: 'fp32' | 'fp16' | 'bf16';
  
  // Configurable assumptions (optional - uses defaults if not provided)
  // These allow users to tune the calculator for their specific setup
  assumptions?: Partial<CalculatorAssumptions>;
}

// ============ Hardware Topology ============

export interface NetworkLatencyConfig {
  // Intra-node latency (microseconds)
  // NVLink 4.0: ~1us, NVLink 3.0: ~1.5us
  intraNodeLatencyUs: number;
  
  // Inter-node latency (microseconds)
  // InfiniBand NDR: ~2us, HDR: ~2.5us, Ethernet: ~10us
  interNodeLatencyUs: number;
  
  // Typical hops for collective operations
  intraNodeHops: number;         // Usually 1-2 with NVSwitch
  interNodeHops: number;         // Usually 2-3 with fat-tree
}

export const DEFAULT_NETWORK_LATENCY: NetworkLatencyConfig = {
  intraNodeLatencyUs: 1.5,       // Conservative NVLink estimate
  interNodeLatencyUs: 3,         // InfiniBand estimate
  intraNodeHops: 1,
  interNodeHops: 2,
};

export interface ClusterTopology {
  totalGPUs: number;
  gpusPerNode: number;
  numNodes: number;
  
  // Bandwidth (GB/s)
  intraNodeBandwidth: number;    // NVLink / NVSwitch
  interNodeBandwidth: number;    // InfiniBand / Ethernet
  
  // GPU specs
  gpuMemoryGB: number;
  gpuComputeTFLOPS: number;
  gpuMemoryBandwidthTBs: number;
  
  // Network latency (optional, uses defaults if not provided)
  networkLatency?: NetworkLatencyConfig;
}

// ============ Memory Breakdown ============

export type MemoryPrecision = 'fp32' | 'fp16' | 'bf16' | 'mixed';

export interface TensorInfo {
  name: string;
  shape: string;           // e.g., "[B, S, H]"
  shapeValues: number[];   // actual values
  elementCount: number;
  bytes: number;
  bytesPerGPU: number;
  precision: MemoryPrecision;
  formula: string;         // How this was calculated
  description: string;
}

export interface ModuleMemory {
  name: string;
  tensors: TensorInfo[];
  totalBytes: number;
  totalBytesPerGPU: number;
  formula: string;
}

export interface MemoryComponent {
  name: string;
  bytes: number;
  bytesPerGPU: number;
  precision: MemoryPrecision;
  description: string;
  formula: string;         // Calculation formula
  isRequired: boolean;     // Always needed vs optional optimization
  tensors?: TensorInfo[];  // Detailed tensor breakdown
}

export interface TrainingMemoryBreakdown {
  // Detailed breakdown with precision info
  components: {
    // Model weights
    modelWeights: MemoryComponent;
    
    // Optimizer states (always FP32)
    masterWeights: MemoryComponent;      // FP32 copy of weights
    optimizerMomentum: MemoryComponent;  // Adam first moment (m)
    optimizerVariance: MemoryComponent;  // Adam second moment (v)
    
    // Gradients
    gradients: MemoryComponent;
    gradientAccumBuffer: MemoryComponent; // FP32 for accumulation
    
    // Activations (per-layer, multiplied by stored layers)
    activations: MemoryComponent;
    
    // LM Head Logits (NOT per-layer, only computed once)
    lmHeadLogits: MemoryComponent;
    
    // Buffers
    communicationBuffers: MemoryComponent;
  };
  
  // Summary by category
  modelStatesBytes: number;       // weights + optimizer + gradients
  activationsBytes: number;
  buffersBytes: number;
  
  // Total
  totalBytes: number;
  totalPerGPU: number;
  
  // Breakdown by precision (for pie chart)
  byPrecision: {
    fp32: number;
    fp16: number;
    bf16: number;
  };
  
  // Recomputation info
  layersStored: number;           // Number of layers stored (affected by recomputation)
  layersPerGPU: number;           // Total layers per GPU stage (numLayers / PP)
  
  // Legacy fields for compatibility
  parameters: number;
  gradients: number;
  optimizerStates: number;
  activationsPerLayer: number;
  activationsTotal: number;
  peakActivations: number;
  communicationBuffers: number;
  parametersPerGPU: number;
  gradientsPerGPU: number;
  optimizerPerGPU: number;
  activationsPerGPU: number;
}

// ============ Communication Analysis ============

export interface CommunicationBreakdown {
  // Per-step communication volume (bytes)
  dataParallelVolume: number;     // Gradient AllReduce/ReduceScatter
  tensorParallelVolume: number;   // Activation AllReduce
  pipelineParallelVolume: number; // P2P activations/gradients
  expertParallelVolume: number;   // All-to-All for MoE
  contextParallelVolume: number;  // Context Parallel (Ulysses all-to-all or Ring P2P)
  zeroVolume: number;             // Parameter AllGather (ZeRO-3)
  
  totalVolume: number;
  
  // Per-step communication time (ms)
  dataParallelTime: number;
  tensorParallelTime: number;
  pipelineParallelTime: number;
  expertParallelTime: number;
  contextParallelTime: number;    // Context Parallel communication time
  zeroTime: number;
  
  totalTime: number;
  
  // Pipeline bubble
  bubbleRatio: number;            // (PP-1) / num_microbatches
  bubbleTime: number;             // ms
}

// ============ Training Step Analysis ============

export interface TrainingStepAnalysis {
  // Compute
  forwardFlops: number;
  backwardFlops: number;
  totalFlops: number;
  
  // Compute time
  forwardTime: number;            // ms
  backwardTime: number;           // ms
  computeTime: number;            // ms (without recomputation overhead)
  recomputationOverhead: number;  // ms (additional compute for recomputation)
  
  // Memory
  memoryBreakdown: TrainingMemoryBreakdown;
  
  // Communication
  communicationBreakdown: CommunicationBreakdown;
  
  // Efficiency metrics
  mfu: number;                    // Model FLOPs Utilization
  hfu: number;                    // Hardware FLOPs Utilization
  computeEfficiency: number;      // compute / (compute + comm + bubble)
  memoryEfficiency: number;       // used / available
  
  // Throughput
  tokensPerSecond: number;
  samplesPerSecond: number;
  timePerStep: number;            // ms
}

// ============ Strategy Comparison ============

export interface TrainingStrategy {
  name: string;
  config: TrainingConfig;
  analysis: TrainingStepAnalysis;
  
  // Validation
  isValid: boolean;
  validationErrors: string[];
  warnings: string[];
  
  // Score for ranking
  score: number;
}

export interface StrategyRecommendation {
  strategies: TrainingStrategy[];
  recommended: TrainingStrategy | null;
  objective: 'throughput' | 'memory' | 'efficiency';
}

// ============ Constraints ============

export interface TrainingConstraints {
  maxGPUMemoryGB: number;
  minMicroBatchSize: number;
  maxMicroBatchSize: number;
  targetGlobalBatchSize: number;
  
  // Optional constraints
  maxTPSize?: number;
  maxPPSize?: number;
  preferZeROStage?: ZeROStage;
}
