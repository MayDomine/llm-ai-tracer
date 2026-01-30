/**
 * Training-specific type definitions
 * Focused on parallelism strategies, batch size calculations, and memory optimization
 */

// ============ Batch Size Configuration ============

export interface BatchConfig {
  globalBatchSize: number;      // Total samples per training step
  microBatchSize: number;       // Samples per GPU per forward/backward
  gradientAccumulation: number; // Number of micro-batches before gradient sync
}

// ============ Parallelism Configuration ============

export type ZeROStage = 0 | 1 | 2 | 3;

export interface ParallelismConfig {
  // Core parallelism dimensions
  dataParallel: number;         // DP size
  tensorParallel: number;       // TP size
  pipelineParallel: number;     // PP size
  expertParallel: number;       // EP size (for MoE)
  
  // Additional options
  sequenceParallel: boolean;    // SP (usually enabled with TP)
  zeroStage: ZeROStage;         // ZeRO optimization level
  
  // Derived
  totalGPUs: number;            // DP × TP × PP × EP
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
}

// ============ Hardware Topology ============

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
    
    // Activations
    activations: MemoryComponent;
    
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
  zeroVolume: number;             // Parameter AllGather (ZeRO-3)
  
  totalVolume: number;
  
  // Per-step communication time (ms)
  dataParallelTime: number;
  tensorParallelTime: number;
  pipelineParallelTime: number;
  expertParallelTime: number;
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
