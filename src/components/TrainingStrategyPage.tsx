import { useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Server,
  Layers,
  Network,
  Database,
  Zap,
  ChevronDown,
  AlertTriangle,
  CheckCircle,
  Info,
  BarChart3,
  Cpu,
} from 'lucide-react';
import type { ModelConfig, HardwareConfig } from '../types/model';
import type {
  TrainingConfig,
  ClusterTopology,
  ZeROStage,
  RecomputationStrategy,
} from '../types/training';
import {
  analyzeTrainingStep,
  validateParallelism,
  validateBatchConfig,
  formatBytes,
  formatThroughput,
} from '../utils/trainingCalculator';
import { ExtendedHardwareConfig } from '../config/presets';

interface TrainingStrategyPageProps {
  modelConfig: ModelConfig;
  hardwareConfig: HardwareConfig;
}

// Default cluster topology from hardware config
function createClusterTopology(
  hardware: ExtendedHardwareConfig,
  numGPUs: number
): ClusterTopology {
  const gpusPerNode = hardware.gpusPerNode || 8;
  return {
    totalGPUs: numGPUs,
    gpusPerNode,
    numNodes: Math.ceil(numGPUs / gpusPerNode),
    intraNodeBandwidth: hardware.nvlinkBandwidth || 300,
    interNodeBandwidth: hardware.networkBandwidth || 50,
    gpuMemoryGB: hardware.memoryCapacity || 80,
    gpuComputeTFLOPS: hardware.computeCapability,
    gpuMemoryBandwidthTBs: hardware.memoryBandwidth,
  };
}


export function TrainingStrategyPage({ modelConfig, hardwareConfig }: TrainingStrategyPageProps) {
  // State
  const [numGPUs, setNumGPUs] = useState(8);
  const [globalBatchSize, setGlobalBatchSize] = useState(1024);
  const [microBatchSize, setMicroBatchSize] = useState(1);
  const [maxSeqLength, setMaxSeqLength] = useState(2048);
  
  // Parallelism state
  const [tensorParallel, setTensorParallel] = useState(1);
  const [pipelineParallel, setPipelineParallel] = useState(1);
  const [expertParallel] = useState(1);
  const [contextParallel, setContextParallel] = useState(1);
  const [contextParallelType, setContextParallelType] = useState<'ulysses' | 'ring' | 'hybrid'>('ulysses');
  const [sequenceParallel, setSequenceParallel] = useState(false);
  const [zeroStage, setZeroStage] = useState<ZeROStage>(0);
  
  // Memory optimization state
  const [recomputation, setRecomputation] = useState<RecomputationStrategy>('none');
  const [precision, setPrecision] = useState<'fp32' | 'fp16' | 'bf16'>('bf16');
  const [flashAttention, setFlashAttention] = useState(true);
  
  // Expanded sections
  const [expandedSection, setExpandedSection] = useState<string | null>('parallelism');

  // Calculate derived values
  const dataParallel = useMemo(() => {
    return Math.floor(numGPUs / (tensorParallel * pipelineParallel * expertParallel * contextParallel));
  }, [numGPUs, tensorParallel, pipelineParallel, expertParallel, contextParallel]);

  const gradientAccumulation = useMemo(() => {
    return Math.max(1, Math.ceil(globalBatchSize / (dataParallel * microBatchSize)));
  }, [globalBatchSize, dataParallel, microBatchSize]);

  // Build config
  const trainingConfig: TrainingConfig = useMemo(() => ({
    batch: {
      globalBatchSize,
      microBatchSize,
      gradientAccumulation,
    },
    parallelism: {
      dataParallel,
      tensorParallel,
      pipelineParallel,
      expertParallel,
      contextParallel,
      contextParallelType,
      sequenceParallel,
      zeroStage,
      totalGPUs: dataParallel * tensorParallel * pipelineParallel * expertParallel * contextParallel,
    },
    memoryOptimization: {
      recomputation,
      cpuOffloading: false,
      nvmeOffloading: false,
      flashAttention,
    },
    maxSeqLength,
    mixedPrecision: precision,
    gradientPrecision: 'fp32',
  }), [
    globalBatchSize, microBatchSize, gradientAccumulation,
    dataParallel, tensorParallel, pipelineParallel, expertParallel, contextParallel, contextParallelType,
    sequenceParallel, zeroStage, recomputation, maxSeqLength, precision, flashAttention
  ]);

  const cluster = useMemo(() => 
    createClusterTopology(hardwareConfig as ExtendedHardwareConfig, numGPUs),
    [hardwareConfig, numGPUs]
  );

  // Validation
  const parallelismErrors = useMemo(() => 
    validateParallelism(trainingConfig.parallelism, modelConfig, cluster),
    [trainingConfig.parallelism, modelConfig, cluster]
  );

  const batchErrors = useMemo(() => 
    validateBatchConfig(trainingConfig.batch, trainingConfig.parallelism),
    [trainingConfig.batch, trainingConfig.parallelism]
  );

  const allErrors = [...parallelismErrors, ...batchErrors];
  const isValid = allErrors.length === 0;

  // Analysis
  const analysis = useMemo(() => {
    if (!isValid) return null;
    try {
      return analyzeTrainingStep(modelConfig, trainingConfig, cluster);
    } catch {
      return null;
    }
  }, [modelConfig, trainingConfig, cluster, isValid]);

  // Available options
  const tpOptions = [1, 2, 4, 8].filter(tp => 
    modelConfig.numAttentionHeads % tp === 0 && tp <= numGPUs
  );

  const ppOptions = [1, 2, 4, 8, 16].filter(pp => 
    modelConfig.numLayers % pp === 0 && pp <= numGPUs
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold">Training Strategy Configuration</h2>
            <p className="text-sm text-gray-400">
              Configure parallelism, batch size, and memory optimization for distributed training
            </p>
          </div>
        </div>

        {/* Quick Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-xs text-gray-500">Model</div>
            <div className="font-medium">{modelConfig.name}</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-xs text-gray-500">Hardware</div>
            <div className="font-medium">{hardwareConfig.name}</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-xs text-gray-500">Total GPUs</div>
            <div className="font-medium">{numGPUs}</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-xs text-gray-500">Config</div>
            <div className="font-medium font-mono">
              DP{dataParallel}×TP{tensorParallel}×PP{pipelineParallel}
              {contextParallel > 1 && `×CP${contextParallel}`}
              {expertParallel > 1 && `×EP${expertParallel}`}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Configuration Panel (4 cols ≈ 30%) */}
        <div className="lg:col-span-4 space-y-3">
          {/* GPU Count and Global Batch Size */}
          <ConfigSection
            title="Cluster & Batch"
            icon={<Server className="w-3.5 h-3.5" />}
            isExpanded={expandedSection === 'cluster'}
            onToggle={() => setExpandedSection(expandedSection === 'cluster' ? null : 'cluster')}
          >
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1.5">Total GPUs</label>
                <div className="grid grid-cols-4 gap-1">
                  {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].map((n) => (
                    <button
                      key={n}
                      onClick={() => setNumGPUs(n)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        numGPUs === n
                          ? 'bg-green-600/50 text-green-300 border border-green-500/50'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <NumberInput
                  label="GBS"
                  hint="samples"
                  value={globalBatchSize}
                  onChange={setGlobalBatchSize}
                  min={1}
                  validate={(v) => v >= 1 ? null : 'Must be at least 1'}
                />
                <NumberInput
                  label="MBS"
                  hint="per GPU"
                  value={microBatchSize}
                  onChange={setMicroBatchSize}
                  min={1}
                  validate={(v) => v >= 1 ? null : 'Must be at least 1'}
                />
              </div>

              <NumberInput
                label="Seq Length"
                value={maxSeqLength}
                onChange={setMaxSeqLength}
                min={128}
                validate={(v) => v >= 128 ? null : 'Must be at least 128'}
              />

              <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/30">
                <div className="flex items-start gap-1.5">
                  <Info className="w-3 h-3 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="text-xs">
                    <div className="text-blue-400 font-medium">GBS = DP × MBS × GA</div>
                    <div className="text-gray-400 mt-0.5 font-mono text-[10px]">
                      {globalBatchSize} = {dataParallel} × {microBatchSize} × {gradientAccumulation}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </ConfigSection>

          {/* Parallelism Configuration */}
          <ConfigSection
            title="Parallelism"
            icon={<Network className="w-3.5 h-3.5" />}
            isExpanded={expandedSection === 'parallelism'}
            onToggle={() => setExpandedSection(expandedSection === 'parallelism' ? null : 'parallelism')}
          >
            <div className="space-y-3">
              {/* TP */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">TP</label>
                  <span className="font-mono text-xs text-green-400">{tensorParallel}</span>
                </div>
                <div className="flex gap-0.5">
                  {tpOptions.map((tp) => (
                    <button
                      key={tp}
                      onClick={() => {
                        setTensorParallel(tp);
                        if (tp > 1) setSequenceParallel(true);
                      }}
                      className={`flex-1 px-1 py-1 text-[10px] rounded transition-all ${
                        tensorParallel === tp
                          ? 'bg-green-600/50 text-green-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {tp}
                    </button>
                  ))}
                </div>
              </div>

              {/* PP */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">PP</label>
                  <span className="font-mono text-xs text-green-400">{pipelineParallel}</span>
                </div>
                <div className="grid grid-cols-4 gap-0.5">
                  {ppOptions.map((pp) => (
                    <button
                      key={pp}
                      onClick={() => setPipelineParallel(pp)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        pipelineParallel === pp
                          ? 'bg-green-600/50 text-green-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {pp}
                    </button>
                  ))}
                </div>
              </div>

              {/* DP (computed) */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">DP = GPUs/(TP×PP)</span>
                <span className="font-mono text-blue-400">{dataParallel}</span>
              </div>

              {/* SP */}
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-400">SP</label>
                <button
                  onClick={() => setSequenceParallel(!sequenceParallel)}
                  disabled={tensorParallel <= 1}
                  className={`px-2 py-0.5 text-[10px] rounded transition-all ${
                    sequenceParallel && tensorParallel > 1
                      ? 'bg-green-600/30 text-green-300'
                      : tensorParallel <= 1
                        ? 'bg-gray-800/30 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-800/50 text-gray-400'
                  }`}
                >
                  {sequenceParallel ? 'On' : 'Off'}
                </button>
              </div>

              {/* Context Parallel (CP) */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">CP (Context Parallel)</label>
                  <span className="font-mono text-[10px] text-orange-400">{contextParallel}</span>
                </div>
                <div className="grid grid-cols-4 gap-0.5 mb-1">
                  {[1, 2, 4, 8].map((cp) => (
                    <button
                      key={cp}
                      onClick={() => setContextParallel(cp)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        contextParallel === cp
                          ? 'bg-orange-600/50 text-orange-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {cp}
                    </button>
                  ))}
                </div>
                {contextParallel > 1 && (
                  <>
                    <div className="grid grid-cols-3 gap-0.5 mb-1">
                      {(['ulysses', 'ring', 'hybrid'] as const).map((type) => (
                        <button
                          key={type}
                          onClick={() => setContextParallelType(type)}
                          className={`px-1 py-0.5 text-[10px] rounded transition-all capitalize ${
                            contextParallelType === type
                              ? 'bg-orange-600/40 text-orange-300'
                              : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                          }`}
                        >
                          {type}
                        </button>
                      ))}
                    </div>
                    <div className="text-[10px] text-gray-500">
                      {contextParallelType === 'ulysses' && `All-to-all (limited by KV heads: ${modelConfig.numKVHeads / tensorParallel})`}
                      {contextParallelType === 'ring' && 'Ring P2P (no head limit, higher comm)'}
                      {contextParallelType === 'hybrid' && 'Ulysses + Ring combined'}
                    </div>
                  </>
                )}
              </div>

              {/* ZeRO Stage */}
              <div>
                <label className="text-xs text-gray-400 block mb-1">ZeRO</label>
                <div className="grid grid-cols-4 gap-0.5">
                  {([0, 1, 2, 3] as ZeROStage[]).map((stage) => (
                    <button
                      key={stage}
                      onClick={() => setZeroStage(stage)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        zeroStage === stage
                          ? 'bg-purple-600/50 text-purple-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {stage === 0 ? 'DDP' : `Z${stage}`}
                    </button>
                  ))}
                </div>
                <div className="text-[10px] text-gray-500 mt-1">
                  {zeroStage === 0 && 'Standard DDP - no optimizer sharding'}
                  {zeroStage === 1 && 'Shard optimizer states across DP ranks'}
                  {zeroStage === 2 && 'Shard optimizer + gradients'}
                  {zeroStage === 3 && 'Shard optimizer + gradients + parameters (FSDP)'}
                </div>
              </div>
            </div>
          </ConfigSection>

          {/* Memory Optimization */}
          <ConfigSection
            title="Optimization"
            icon={<Database className="w-3.5 h-3.5" />}
            isExpanded={expandedSection === 'memory'}
            onToggle={() => setExpandedSection(expandedSection === 'memory' ? null : 'memory')}
          >
            <div className="space-y-3">
              {/* Precision */}
              <div>
                <label className="text-xs text-gray-400 block mb-1">Precision</label>
                <div className="grid grid-cols-3 gap-0.5">
                  {(['fp32', 'fp16', 'bf16'] as const).map((p) => (
                    <button
                      key={p}
                      onClick={() => setPrecision(p)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        precision === p
                          ? 'bg-cyan-600/50 text-cyan-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* Recomputation */}
              <div>
                <label className="text-xs text-gray-400 block mb-1">Recomputation</label>
                <div className="grid grid-cols-2 gap-0.5">
                  {([
                    { value: 'none', label: 'None' },
                    { value: 'selective', label: 'Selective' },
                    { value: 'full', label: 'Full' },
                    { value: 'block', label: 'Block' },
                  ] as const).map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => setRecomputation(opt.value)}
                      className={`px-1 py-1 text-[10px] rounded transition-all ${
                        recomputation === opt.value
                          ? 'bg-orange-600/30 text-orange-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* FlashAttention */}
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-400">FlashAttention</label>
                <button
                  onClick={() => setFlashAttention(!flashAttention)}
                  className={`px-2 py-0.5 text-[10px] rounded transition-all ${
                    flashAttention
                      ? 'bg-green-600/50 text-green-300'
                      : 'bg-gray-700/50 text-gray-400'
                  }`}
                >
                  {flashAttention ? 'On' : 'Off'}
                </button>
              </div>
            </div>
          </ConfigSection>

          {/* Validation Errors */}
          {allErrors.length > 0 && (
            <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/30">
              <div className="flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <div className="text-xs text-red-400 font-medium">Errors</div>
                  <ul className="text-[10px] text-gray-400 mt-1 space-y-0.5">
                    {allErrors.map((error, i) => (
                      <li key={i}>{error}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Center: Memory Breakdown - Main Area (5 cols ≈ 45%) */}
        <div className="lg:col-span-5 space-y-4">
          {analysis ? (
            <MemoryModuleSection
              memoryBreakdown={analysis.memoryBreakdown}
              memoryEfficiency={analysis.memoryEfficiency}
              precision={precision}
              gpuMemoryGB={cluster.gpuMemoryGB}
            />
          ) : (
            <div className="glass rounded-xl p-12 text-center">
              <Database className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-500">Fix configuration errors to see memory breakdown</p>
            </div>
          )}
        </div>

        {/* Right: Stats Panel (3 cols ≈ 25%) - Compact fonts */}
        <div className="lg:col-span-3 space-y-2">
          {/* Status */}
          <div className={`glass rounded-lg p-2 ${isValid ? 'border-green-500/30' : 'border-red-500/30'} border`}>
            <div className="flex items-center gap-1.5">
              {isValid ? (
                <CheckCircle className="w-3.5 h-3.5 text-green-400" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              )}
              <span className={`text-xs ${isValid ? 'text-green-400' : 'text-red-400'}`}>
                {isValid ? 'Valid' : 'Invalid'}
              </span>
            </div>
          </div>

          {analysis && (
            <>
              {/* Throughput */}
              <div className="glass rounded-lg p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Zap className="w-3.5 h-3.5 text-yellow-400" />
                  <h3 className="font-medium text-xs">Throughput</h3>
                </div>
                <div className="text-lg font-bold font-mono text-yellow-400">
                  {formatThroughput(analysis.tokensPerSecond)} tok/s
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5">
                  {analysis.samplesPerSecond.toFixed(1)} samples/s · {analysis.timePerStep.toFixed(1)} ms/step
                </div>
              </div>

              {/* Efficiency */}
              <div className="glass rounded-lg p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <BarChart3 className="w-3.5 h-3.5 text-purple-400" />
                  <h3 className="font-medium text-xs">Efficiency</h3>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">MFU</span>
                    <span className="font-mono text-purple-400">{(analysis.mfu * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Compute</span>
                    <span className="font-mono text-blue-400">{(analysis.computeEfficiency * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Bubble</span>
                    <span className="font-mono text-orange-400">{(analysis.communicationBreakdown.bubbleRatio * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Communication */}
              <div className="glass rounded-lg p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Network className="w-3.5 h-3.5 text-green-400" />
                  <h3 className="font-medium text-xs">Communication</h3>
                </div>
                <div className="space-y-0.5 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-gray-500">DP</span>
                    <span className="font-mono text-gray-400">{analysis.communicationBreakdown.dataParallelTime.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">TP</span>
                    <span className="font-mono text-gray-400">{analysis.communicationBreakdown.tensorParallelTime.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">PP</span>
                    <span className="font-mono text-gray-400">{analysis.communicationBreakdown.pipelineParallelTime.toFixed(1)} ms</span>
                  </div>
                  {contextParallel > 1 && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">CP</span>
                      <span className="font-mono text-orange-400">{analysis.communicationBreakdown.contextParallelTime.toFixed(1)} ms</span>
                    </div>
                  )}
                  <div className="flex justify-between pt-0.5 border-t border-gray-700">
                    <span className="text-gray-400">Total</span>
                    <span className="font-mono text-green-400">{analysis.communicationBreakdown.totalTime.toFixed(1)} ms</span>
                  </div>
                </div>
              </div>

              {/* Memory Summary */}
              <div className="glass rounded-lg p-3">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Database className="w-3.5 h-3.5 text-cyan-400" />
                  <h3 className="font-medium text-xs">Memory/GPU</h3>
                </div>
                <div className="text-lg font-bold font-mono text-cyan-400">
                  {formatBytes(analysis.memoryBreakdown.totalPerGPU)}
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5">
                  / {cluster.gpuMemoryGB} GB ({(analysis.memoryEfficiency * 100).toFixed(0)}%)
                </div>
                <div className="h-1.5 bg-gray-700 rounded-full mt-1.5 overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      analysis.memoryEfficiency > 0.95 ? 'bg-red-500' :
                      analysis.memoryEfficiency > 0.8 ? 'bg-yellow-500' : 'bg-cyan-500'
                    }`}
                    style={{ width: `${Math.min(100, analysis.memoryEfficiency * 100)}%` }}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper component for config sections - compact style
function ConfigSection({
  title,
  icon,
  isExpanded,
  onToggle,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <motion.div className="glass rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2.5 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="text-green-400">{icon}</div>
          <h3 className="font-medium text-sm">{title}</h3>
        </div>
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-4 h-4 text-gray-400" />
        </motion.div>
      </button>
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="px-3 pb-3 border-t border-gray-700/50 pt-3 text-sm">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Memory Module Section - Uses module cards like ArithIntensity tab
function MemoryModuleSection({
  memoryBreakdown,
  memoryEfficiency,
  precision,
  gpuMemoryGB,
}: {
  memoryBreakdown: TrainingMemoryBreakdown;
  memoryEfficiency: number;
  precision: 'fp32' | 'fp16' | 'bf16';
  gpuMemoryGB: number;
}) {
  const [expandedModule, setExpandedModule] = useState<string | null>('activations');
  
  // Use layersStored from calculator (accounts for recomputation)
  const layersStored = memoryBreakdown.layersStored;
  const layersPerGPU = memoryBreakdown.layersPerGPU;
  
  // Group activation tensors by module
  const activationTensors = memoryBreakdown.components.activations.tensors || [];
  const attentionTensors = activationTensors.filter(t => 
    t.name.includes('Attn') || t.name.includes('Q ') || t.name.includes('K ') || 
    t.name.includes('V ') || t.name.includes('Attention') || t.name.includes('Output Projection')
  );
  const ffnTensors = activationTensors.filter(t => 
    t.name.includes('FFN') || t.name.includes('Gate') || t.name.includes('Up') || t.name.includes('Down')
  );
  const otherTensors = activationTensors.filter(t => 
    !attentionTensors.includes(t) && !ffnTensors.includes(t)
  );

  // Calculate per-layer memory for attention and FFN (use layersStored, not layersPerGPU!)
  const attentionMemoryPerLayer = attentionTensors.reduce((sum, t) => sum + t.bytesPerGPU, 0);
  const ffnMemoryPerLayer = ffnTensors.reduce((sum, t) => sum + t.bytesPerGPU, 0);
  const otherMemoryPerLayer = otherTensors.reduce((sum, t) => sum + t.bytesPerGPU, 0);

  return (
    <div className="space-y-4">
      {/* Memory Summary Header */}
      <div className="glass rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Memory Breakdown per GPU</h2>
              <p className="text-sm text-gray-400">
                {layersPerGPU} layers per GPU · {layersStored < layersPerGPU ? `${layersStored} stored (recompute)` : 'all stored'} · {precision.toUpperCase()}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold font-mono text-cyan-400">
              {formatBytes(memoryBreakdown.totalPerGPU)}
            </div>
            <div className="text-sm text-gray-500">
              / {gpuMemoryGB} GB ({(memoryEfficiency * 100).toFixed(0)}% utilized)
            </div>
          </div>
        </div>
        
        {/* Memory bar */}
        <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${
              memoryEfficiency > 0.95 ? 'bg-red-500' :
              memoryEfficiency > 0.8 ? 'bg-yellow-500' :
              'bg-gradient-to-r from-cyan-500 to-blue-500'
            }`}
            style={{ width: `${Math.min(100, memoryEfficiency * 100)}%` }}
          />
        </div>
        
        {memoryEfficiency > 0.9 && (
          <div className="mt-3 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-yellow-400">
                Memory usage is high. Consider enabling ZeRO-2/3 or activation recomputation.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Module Cards - Similar to ArithIntensity ModuleCard */}
      <div className="space-y-3">
        {/* Model States Module */}
        <MemoryModuleCard
          name="Model States"
          icon={<Layers className="w-5 h-5" />}
          color="from-orange-500 to-red-500"
          totalBytes={memoryBreakdown.modelStatesBytes}
          description="Weights, Gradients, Optimizer States"
          isExpanded={expandedModule === 'model-states'}
          onToggle={() => setExpandedModule(expandedModule === 'model-states' ? null : 'model-states')}
        >
          <div className="divide-y divide-gray-800">
            {[
              memoryBreakdown.components.modelWeights,
              memoryBreakdown.components.masterWeights,
              memoryBreakdown.components.optimizerMomentum,
              memoryBreakdown.components.optimizerVariance,
              memoryBreakdown.components.gradients,
              memoryBreakdown.components.gradientAccumBuffer,
            ].filter(c => c.bytesPerGPU > 0).map((comp, idx) => (
              <div key={idx} className="px-4 py-3 flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{comp.name}</span>
                    <PrecisionBadge precision={comp.precision} />
                    {comp.formula && <FormulaButton formula={comp.formula} title={comp.name} />}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">{comp.description}</p>
                </div>
                <div className="font-mono text-sm">{formatBytes(comp.bytesPerGPU)}</div>
              </div>
            ))}
          </div>
        </MemoryModuleCard>

        {/* Attention Activations Module */}
        <MemoryModuleCard
          name="Attention Activations"
          icon={<Cpu className="w-5 h-5" />}
          color="from-indigo-500 to-purple-500"
          totalBytes={attentionMemoryPerLayer * layersStored}
          description={`${attentionTensors.length} tensors × ${layersStored} stored layers${layersStored < layersPerGPU ? ` (of ${layersPerGPU})` : ''}`}
          isExpanded={expandedModule === 'attention'}
          onToggle={() => setExpandedModule(expandedModule === 'attention' ? null : 'attention')}
        >
          <div className="divide-y divide-gray-800">
            {attentionTensors.map((tensor, idx) => (
              <div key={idx} className="px-4 py-3 flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{tensor.name}</span>
                    <span className="font-mono text-xs text-gray-500">{tensor.shape}</span>
                    <PrecisionBadge precision={precision} />
                    <FormulaButton formula={tensor.formula} title={tensor.name} />
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {tensor.shapeValues.map(v => v.toLocaleString()).join(' × ')} = {tensor.elementCount.toLocaleString()} elements
                  </p>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm">{formatBytes(tensor.bytesPerGPU)}</div>
                  <div className="text-xs text-gray-500">×{layersStored} = {formatBytes(tensor.bytesPerGPU * layersStored)}</div>
                </div>
              </div>
            ))}
          </div>
        </MemoryModuleCard>

        {/* FFN Activations Module */}
        <MemoryModuleCard
          name="FFN Activations"
          icon={<Cpu className="w-5 h-5" />}
          color="from-purple-500 to-pink-500"
          totalBytes={ffnMemoryPerLayer * layersStored}
          description={`${ffnTensors.length} tensors × ${layersStored} stored layers${layersStored < layersPerGPU ? ` (of ${layersPerGPU})` : ''}`}
          isExpanded={expandedModule === 'ffn'}
          onToggle={() => setExpandedModule(expandedModule === 'ffn' ? null : 'ffn')}
        >
          <div className="divide-y divide-gray-800">
            {ffnTensors.map((tensor, idx) => (
              <div key={idx} className="px-4 py-3 flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{tensor.name}</span>
                    <span className="font-mono text-xs text-gray-500">{tensor.shape}</span>
                    <PrecisionBadge precision={precision} />
                    <FormulaButton formula={tensor.formula} title={tensor.name} />
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {tensor.shapeValues.map(v => v.toLocaleString()).join(' × ')} = {tensor.elementCount.toLocaleString()} elements
                  </p>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm">{formatBytes(tensor.bytesPerGPU)}</div>
                  <div className="text-xs text-gray-500">×{layersStored} = {formatBytes(tensor.bytesPerGPU * layersStored)}</div>
                </div>
              </div>
            ))}
          </div>
        </MemoryModuleCard>

        {/* Other Activations (if any) */}
        {otherTensors.length > 0 && (
          <MemoryModuleCard
            name="Other Activations"
            icon={<Layers className="w-5 h-5" />}
            color="from-gray-500 to-gray-600"
            totalBytes={otherMemoryPerLayer * layersStored}
            description={`${otherTensors.length} tensors × ${layersStored} stored layers${layersStored < layersPerGPU ? ` (of ${layersPerGPU})` : ''}`}
            isExpanded={expandedModule === 'other'}
            onToggle={() => setExpandedModule(expandedModule === 'other' ? null : 'other')}
          >
            <div className="divide-y divide-gray-800">
              {otherTensors.map((tensor, idx) => (
                <div key={idx} className="px-4 py-3 flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{tensor.name}</span>
                      <span className="font-mono text-xs text-gray-500">{tensor.shape}</span>
                      <FormulaButton formula={tensor.formula} title={tensor.name} />
                    </div>
                  </div>
                  <div className="font-mono text-sm">{formatBytes(tensor.bytesPerGPU)}</div>
                </div>
              ))}
            </div>
          </MemoryModuleCard>
        )}

        {/* Communication Buffers Module */}
        <MemoryModuleCard
          name="Communication Buffers"
          icon={<Network className="w-5 h-5" />}
          color="from-green-500 to-teal-500"
          totalBytes={memoryBreakdown.components.communicationBuffers.bytesPerGPU}
          description="AllReduce / AllGather double buffering"
          isExpanded={expandedModule === 'buffers'}
          onToggle={() => setExpandedModule(expandedModule === 'buffers' ? null : 'buffers')}
        >
          <div className="px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{memoryBreakdown.components.communicationBuffers.name}</span>
                <PrecisionBadge precision={memoryBreakdown.components.communicationBuffers.precision} />
                {memoryBreakdown.components.communicationBuffers.formula && (
                  <FormulaButton 
                    formula={memoryBreakdown.components.communicationBuffers.formula} 
                    title={memoryBreakdown.components.communicationBuffers.name} 
                  />
                )}
              </div>
              <div className="font-mono text-sm">
                {formatBytes(memoryBreakdown.components.communicationBuffers.bytesPerGPU)}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {memoryBreakdown.components.communicationBuffers.description}
            </p>
          </div>
        </MemoryModuleCard>
      </div>

      {/* Memory by Precision */}
      <div className="glass rounded-xl p-4">
        <h3 className="text-sm font-medium mb-3">Memory by Precision</h3>
        <div className="grid grid-cols-3 gap-3">
          {memoryBreakdown.byPrecision.fp32 > 0 && (
            <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/30 text-center">
              <div className="text-xs text-orange-400 font-medium">FP32</div>
              <div className="font-mono text-sm text-orange-300 mt-1">{formatBytes(memoryBreakdown.byPrecision.fp32)}</div>
              <div className="text-xs text-orange-400/60">{((memoryBreakdown.byPrecision.fp32 / memoryBreakdown.totalPerGPU) * 100).toFixed(0)}%</div>
            </div>
          )}
          {memoryBreakdown.byPrecision.fp16 > 0 && (
            <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30 text-center">
              <div className="text-xs text-blue-400 font-medium">FP16</div>
              <div className="font-mono text-sm text-blue-300 mt-1">{formatBytes(memoryBreakdown.byPrecision.fp16)}</div>
              <div className="text-xs text-blue-400/60">{((memoryBreakdown.byPrecision.fp16 / memoryBreakdown.totalPerGPU) * 100).toFixed(0)}%</div>
            </div>
          )}
          {memoryBreakdown.byPrecision.bf16 > 0 && (
            <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30 text-center">
              <div className="text-xs text-purple-400 font-medium">BF16</div>
              <div className="font-mono text-sm text-purple-300 mt-1">{formatBytes(memoryBreakdown.byPrecision.bf16)}</div>
              <div className="text-xs text-purple-400/60">{((memoryBreakdown.byPrecision.bf16 / memoryBreakdown.totalPerGPU) * 100).toFixed(0)}%</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Memory Module Card - Similar to ModuleCard from ArithIntensity
function MemoryModuleCard({
  name,
  icon,
  color,
  totalBytes,
  description,
  isExpanded,
  onToggle,
  children,
}: {
  name: string;
  icon: React.ReactNode;
  color: string;
  totalBytes: number;
  description: string;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <motion.div
      layout
      className="glass rounded-xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-4 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center text-white`}>
            {icon}
          </div>
          <div className="text-left">
            <h3 className="font-semibold">{name}</h3>
            <p className="text-sm text-gray-400">{description}</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="font-mono text-lg text-cyan-400">{formatBytes(totalBytes)}</div>
          </div>
          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-5 h-5 text-gray-400" />
          </motion.div>
        </div>
      </button>

      {/* Expanded content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="border-t border-gray-700/50">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Precision Badge Component
function PrecisionBadge({ precision }: { precision: string }) {
  const colors: Record<string, string> = {
    fp32: 'bg-orange-500/20 text-orange-400',
    fp16: 'bg-blue-500/20 text-blue-400',
    bf16: 'bg-purple-500/20 text-purple-400',
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${colors[precision] || 'bg-gray-500/20 text-gray-400'}`}>
      {precision.toUpperCase()}
    </span>
  );
}

// Import TrainingMemoryBreakdown type for the component
import type { TrainingMemoryBreakdown } from '../types/training';

// Reusable number input component with validation
function NumberInput({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  validate,
}: {
  label: string;
  hint?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  validate?: (value: number) => string | null;
}) {
  const [inputValue, setInputValue] = useState(value.toString());
  const [error, setError] = useState<string | null>(null);
  
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    
    const num = parseInt(newValue, 10);
    if (isNaN(num)) {
      setError('Must be a number');
      return;
    }
    
    if (min !== undefined && num < min) {
      setError(`Must be at least ${min}`);
      return;
    }
    
    if (max !== undefined && num > max) {
      setError(`Must be at most ${max}`);
      return;
    }
    
    if (validate) {
      const validationError = validate(num);
      if (validationError) {
        setError(validationError);
        return;
      }
    }
    
    setError(null);
    onChange(num);
  };
  
  const handleBlur = () => {
    // Reset to valid value on blur if invalid
    if (error) {
      setInputValue(value.toString());
      setError(null);
    }
  };
  
  return (
    <div>
      <label className="text-xs text-gray-400 block mb-1">
        {label}
        {hint && <span className="text-gray-600 ml-1">({hint})</span>}
      </label>
      <input
        type="text"
        value={inputValue}
        onChange={handleChange}
        onBlur={handleBlur}
        className={`w-full px-2 py-1.5 text-sm bg-gray-800 rounded-lg border transition-colors font-mono ${
          error 
            ? 'border-red-500 border-dashed focus:border-red-400' 
            : 'border-gray-700 focus:border-green-500'
        } focus:outline-none`}
      />
      {error && (
        <p className="text-xs text-red-400 mt-1">{error}</p>
      )}
    </div>
  );
}

// Formula display component - uses React Portal to escape parent transforms
function FormulaButton({
  formula,
  title,
}: {
  formula: string;
  title: string;
}) {
  const [showFormula, setShowFormula] = useState(false);
  
  // Split formula into lines for better display
  const formulaLines = formula.split('\n');
  
  return (
    <>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setShowFormula(!showFormula);
        }}
        className="w-4 h-4 flex items-center justify-center text-[10px] text-blue-400 hover:text-blue-300 bg-blue-500/20 rounded-full flex-shrink-0"
        title="Show calculation formula"
      >
        ?
      </button>
      {showFormula && createPortal(
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-[9999] bg-black/50"
            onClick={() => setShowFormula(false)}
          />
          {/* Modal */}
          <div 
            className="fixed z-[10000] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 p-5 bg-gray-800 border border-gray-600 rounded-xl shadow-2xl max-w-lg w-[90vw] max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="text-base font-medium text-gray-200">{title}</div>
              <button 
                onClick={() => setShowFormula(false)}
                className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded-full transition-colors"
              >
                ×
              </button>
            </div>
            <div className="bg-gray-900/70 rounded-lg p-4 border border-gray-700">
              <div className="text-xs text-gray-500 mb-2">Calculation Formula:</div>
              <div className="space-y-1">
                {formulaLines.map((line, idx) => (
                  <div key={idx} className="font-mono text-sm leading-relaxed">
                    {line.includes('=') ? (
                      <>
                        <span className="text-cyan-400">{line.split('=')[0]}</span>
                        <span className="text-gray-400">=</span>
                        <span className="text-green-400">{line.split('=').slice(1).join('=')}</span>
                      </>
                    ) : (
                      <span className="text-green-400">{line}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500 text-right">
              Click outside or × to close
            </div>
          </div>
        </>,
        document.body
      )}
    </>
  );
}

