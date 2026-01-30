// 模型配置类型定义

export type FFNType = 'gpt' | 'gated' | 'moe';
export type AttentionType = 'mha' | 'gqa';

export interface ModelConfig {
  name: string;
  // 基础参数
  hiddenSize: number;           // 隐藏层维度 d_model
  numLayers: number;            // Transformer层数
  vocabSize: number;            // 词表大小
  
  // Attention参数
  numAttentionHeads: number;    // 注意力头数
  numKVHeads: number;           // KV头数 (GQA时 < numAttentionHeads)
  headDim: number;              // 每个头的维度
  attentionType: AttentionType;
  
  // FFN参数
  intermediateSize: number;     // FFN中间层维度
  ffnType: FFNType;
  
  // MoE参数 (仅当ffnType为'moe'时有效)
  numExperts?: number;          // 专家数量
  numExpertsPerToken?: number;  // 每个token激活的专家数
  sharedExpertNum?: number;     // 共享专家数量 (Qwen3 MoE)
  
  // 序列参数
  maxSeqLen: number;            // 最大序列长度
}

export interface HardwareConfig {
  computeCapability: number;    // 计算能力 (TFLOPS)
  memoryBandwidth: number;      // 内存带宽 (TB/s)
  name: string;
}

export type InferenceMode = 'training' | 'prefill' | 'decode';

export interface InferenceConfig {
  mode: InferenceMode;
  batchSize: number;
  seqLen: number;               // 序列长度 (training/prefill用)
  kvCacheLen: number;           // KV Cache长度 (decode用，已生成的token数)
  dtype: 'fp16' | 'bf16' | 'fp32' | 'int8' | 'int4';
}

// 运算类型
export type OperationType = 
  | 'gemm'           // 矩阵乘法
  | 'attention'      // 注意力计算
  | 'softmax'        // Softmax
  | 'layernorm'      // LayerNorm
  | 'rmsnorm'        // RMSNorm
  | 'embedding'      // Embedding查找
  | 'silu'           // SiLU激活
  | 'gelu'           // GELU激活
  | 'elementwise';   // 逐元素运算

// 单个运算的分析结果
export interface OperationAnalysis {
  id: string;
  name: string;
  type: OperationType;
  module: 'embedding' | 'attention' | 'ffn' | 'lm_head';
  subModule?: string;
  
  // 计算量和内存访问
  flops: number;                // 浮点运算次数
  memoryBytes: number;          // 内存访问字节数
  arithmeticIntensity: number;  // 算术密度 = flops / memoryBytes
  
  // 性能分析
  isComputeBound: boolean;      // 是否计算密集
  theoreticalTime: number;      // 理论执行时间 (ms)
  bottleneck: 'compute' | 'memory';
  
  // 详细信息
  shape?: {
    m?: number;
    n?: number;
    k?: number;
  };
  description: string;
}

// 模块级别的分析结果
export interface ModuleAnalysis {
  name: string;
  operations: OperationAnalysis[];
  totalFlops: number;
  totalMemoryBytes: number;
  avgArithmeticIntensity: number;
  computeBoundOps: number;
  memoryBoundOps: number;
}

// 参数数量统计
export interface ParameterCount {
  embedding: number;
  attention: number;
  ffn: number;
  lmHead: number;
  layerNorm: number;
  total: number;
}

// 完整模型分析结果
export interface ModelAnalysis {
  modelConfig: ModelConfig;
  hardwareConfig: HardwareConfig;
  inferenceConfig: InferenceConfig;
  
  embedding: ModuleAnalysis;
  transformerBlock: ModuleAnalysis;  // 单个block
  lmHead: ModuleAnalysis;
  
  // 汇总统计
  totalFlops: number;
  forwardFlops: number;              // 前向传播FLOPs
  backwardFlops?: number;            // 反向传播FLOPs (仅training模式)
  totalMemoryBytes: number;
  overallArithmeticIntensity: number;
  estimatedLatency: number;          // 估计延迟 (ms)
  
  // Roofline相关
  rooflinePoint: number;             // 硬件的roofline拐点
  
  // 新增: 参数统计
  parameterCount: ParameterCount;
  
  // 新增: 每token统计
  flopsPerToken: number;
  numTokens: number;
  
  // 新增: 验证信息
  approximateFlops: number;          // 使用 2N 近似计算的FLOPs
  approximationError: number;        // 近似误差百分比
}

// ============ 并行策略相关类型 ============

export interface ParallelConfig {
  dataParallel: number;              // DP size
  tensorParallel: number;            // TP size  
  pipelineParallel: number;          // PP size
  microBatchSize: number;            // 微批次大小
  gradientAccumulation: number;      // 梯度累积步数
  sequenceParallel?: boolean;        // 是否使用序列并行
}

export interface CommunicationAnalysis {
  // 各种并行的通信量 (bytes per step)
  dataParallelComm: number;
  tensorParallelComm: number;
  pipelineParallelComm: number;
  totalCommBytes: number;
  
  // 通信时间 (ms)
  dataParallelTime: number;
  tensorParallelTime: number;
  pipelineParallelTime: number;
  totalCommTime: number;
  
  // 效率指标
  bubbleOverhead: number;            // Pipeline bubble 开销百分比
  computeCommRatio: number;          // 计算/通信比
  efficiency: number;                // 整体效率 = 计算时间 / (计算时间 + 通信时间)
}

// ============ GPU内存相关类型 ============

export interface MemoryBreakdown {
  name: string;
  bytes: number;
  percentage: number;
  color: string;
}

export interface MemoryAnalysis {
  // 各组件内存占用 (bytes)
  modelWeights: number;
  kvCache: number;
  activations: number;
  gradients?: number;                // 仅training
  optimizerStates?: number;          // 仅training
  frameworkOverhead: number;
  
  // 总计
  total: number;
  peakMemory: number;
  
  // 分解
  breakdown: MemoryBreakdown[];
  
  // 与GPU容量对比
  gpuCapacity?: number;
  utilizationPercent?: number;
  fitsInGPU?: boolean;
}

// ============ 策略比较相关类型 ============

export interface StrategyConfig extends ParallelConfig {
  name: string;
}

export interface StrategyAnalysis {
  config: StrategyConfig;
  memoryAnalysis: MemoryAnalysis;
  communicationAnalysis: CommunicationAnalysis;
  
  // 性能指标
  throughputTokensPerSec: number;
  latencyMs: number;
  
  // 约束检查
  isValid: boolean;
  validationErrors: string[];
  
  // 评分
  score: number;
}

export interface StrategyComparison {
  strategies: StrategyAnalysis[];
  recommendedStrategy: StrategyAnalysis | null;
  optimizationObjective: 'throughput' | 'latency' | 'memory';
}
