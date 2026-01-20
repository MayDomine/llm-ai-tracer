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
  totalMemoryBytes: number;
  overallArithmeticIntensity: number;
  estimatedLatency: number;         // 估计延迟 (ms)
  
  // Roofline相关
  rooflinePoint: number;            // 硬件的roofline拐点
}
