import { motion } from 'framer-motion';
import { Cpu, HardDrive, Gauge, Clock, Zap, TrendingUp, Play, GraduationCap } from 'lucide-react';
import type { ModelAnalysis } from '../types/model';
import { formatNumber, formatBytes, formatTime } from '../utils/calculator';

interface StatsPanelProps {
  analysis: ModelAnalysis;
}

const MODE_LABELS = {
  training: { label: 'Training', icon: GraduationCap, color: 'text-green-400' },
  prefill: { label: 'Prefill', icon: Play, color: 'text-blue-400' },
  decode: { label: 'Decode', icon: Zap, color: 'text-orange-400' },
};

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
  color: string;
  delay?: number;
}

function StatCard({ icon, label, value, subValue, color, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      className="glass rounded-xl p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
    >
      <div className="flex items-start justify-between">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color}`}>
          {icon}
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-400 uppercase tracking-wide">{label}</div>
          <div className="text-xl font-semibold font-mono mt-1">{value}</div>
          {subValue && (
            <div className="text-xs text-gray-500 mt-0.5">{subValue}</div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export function StatsPanel({ analysis }: StatsPanelProps) {
  const { totalFlops, totalMemoryBytes, overallArithmeticIntensity, estimatedLatency, rooflinePoint, inferenceConfig, hardwareConfig } = analysis;
  
  const computeBoundTotal = 
    analysis.embedding.computeBoundOps + 
    analysis.transformerBlock.computeBoundOps * analysis.modelConfig.numLayers + 
    analysis.lmHead.computeBoundOps;
  
  const memoryBoundTotal = 
    analysis.embedding.memoryBoundOps + 
    analysis.transformerBlock.memoryBoundOps * analysis.modelConfig.numLayers + 
    analysis.lmHead.memoryBoundOps;
  
  const totalOps = computeBoundTotal + memoryBoundTotal;
  const computeRatio = totalOps > 0 ? (computeBoundTotal / totalOps * 100) : 0;

  const modeConfig = MODE_LABELS[inferenceConfig.mode];
  const ModeIcon = modeConfig.icon;
  
  // Calculate detailed latency breakdown
  const computeTime = totalFlops / (hardwareConfig.computeCapability * 1e12) * 1000; // ms
  const memoryTime = totalMemoryBytes / (hardwareConfig.memoryBandwidth * 1e12) * 1000; // ms
  const isComputeBottleneck = computeTime > memoryTime;

  return (
    <div className="space-y-4">
      {/* 当前模式指示 */}
      <motion.div
        className="glass rounded-xl p-3 flex items-center justify-between"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3">
          <ModeIcon className={`w-5 h-5 ${modeConfig.color}`} />
          <div>
            <span className={`font-semibold ${modeConfig.color}`}>{modeConfig.label}</span>
            <span className="text-gray-400 text-sm ml-2">模式</span>
          </div>
        </div>
        <div className="text-sm text-gray-400">
          {inferenceConfig.mode === 'decode' ? (
            <span>
              B={inferenceConfig.batchSize} × 1 token, KV Cache={inferenceConfig.kvCacheLen}
            </span>
          ) : (
            <span>
              B={inferenceConfig.batchSize} × S={inferenceConfig.seqLen} = {inferenceConfig.batchSize * inferenceConfig.seqLen} tokens
            </span>
          )}
        </div>
      </motion.div>

      {/* 主统计卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Cpu className="w-5 h-5 text-indigo-400" />}
          label="总 FLOPs"
          value={formatNumber(totalFlops)}
          subValue="浮点运算"
          color="bg-indigo-500/20"
          delay={0}
        />
        <StatCard
          icon={<HardDrive className="w-5 h-5 text-cyan-400" />}
          label="内存访问"
          value={formatBytes(totalMemoryBytes)}
          subValue="总数据量"
          color="bg-cyan-500/20"
          delay={0.05}
        />
        <StatCard
          icon={<Gauge className="w-5 h-5 text-purple-400" />}
          label="算术密度"
          value={overallArithmeticIntensity.toFixed(2)}
          subValue="FLOP/Byte"
          color="bg-purple-500/20"
          delay={0.1}
        />
        <StatCard
          icon={<Clock className="w-5 h-5 text-orange-400" />}
          label="理论延迟"
          value={formatTime(estimatedLatency)}
          subValue="估算时间"
          color="bg-orange-500/20"
          delay={0.15}
        />
      </div>

      {/* 性能瓶颈分析 */}
      <motion.div
        className="glass rounded-xl p-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold">性能瓶颈分析</h3>
          </div>
          <div className="text-sm text-gray-400">
            Roofline 拐点: <span className="font-mono text-white">{rooflinePoint.toFixed(1)}</span> FLOP/Byte
          </div>
        </div>

        {/* 操作类型分布条 */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">操作类型分布</span>
            <span className="text-gray-500">{totalOps} 个操作 / 每层</span>
          </div>
          
          <div className="h-6 rounded-full overflow-hidden bg-gray-800 flex">
            <motion.div
              className="h-full bg-gradient-to-r from-compute-bound to-compute-bound-light flex items-center justify-center"
              initial={{ width: 0 }}
              animate={{ width: `${computeRatio}%` }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {computeRatio > 15 && (
                <span className="text-xs font-medium text-white px-2">
                  {computeRatio.toFixed(0)}%
                </span>
              )}
            </motion.div>
            <motion.div
              className="h-full bg-gradient-to-r from-memory-bound to-memory-bound-light flex items-center justify-center"
              initial={{ width: 0 }}
              animate={{ width: `${100 - computeRatio}%` }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {(100 - computeRatio) > 15 && (
                <span className="text-xs font-medium text-white px-2">
                  {(100 - computeRatio).toFixed(0)}%
                </span>
              )}
            </motion.div>
          </div>

          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-compute-bound" />
              <span className="text-gray-400">计算密集</span>
              <span className="font-mono text-compute-bound">{computeBoundTotal} 个</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-memory-bound">{memoryBoundTotal} 个</span>
              <span className="text-gray-400">内存密集</span>
              <span className="w-3 h-3 rounded-full bg-memory-bound" />
            </div>
          </div>
        </div>

        {/* 瓶颈提示 */}
        <div className="mt-4 p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
          <div className="flex items-start gap-2">
            <Zap className={`w-4 h-4 mt-0.5 ${overallArithmeticIntensity >= rooflinePoint ? 'text-compute-bound' : 'text-memory-bound'}`} />
            <div className="text-sm">
              {overallArithmeticIntensity >= rooflinePoint ? (
                <>
                  <span className="text-compute-bound font-medium">整体为计算密集型</span>
                  <p className="text-gray-400 mt-1">
                    算术密度 ({overallArithmeticIntensity.toFixed(2)}) ≥ Roofline拐点 ({rooflinePoint.toFixed(1)})，
                    性能主要受限于 GPU 算力。可通过提升计算能力或使用算子融合来优化。
                  </p>
                </>
              ) : (
                <>
                  <span className="text-memory-bound font-medium">整体为内存密集型</span>
                  <p className="text-gray-400 mt-1">
                    算术密度 ({overallArithmeticIntensity.toFixed(2)}) &lt; Roofline拐点 ({rooflinePoint.toFixed(1)})，
                    性能主要受限于内存带宽。可通过增大 batch size、使用量化或 FlashAttention 等技术来优化。
                  </p>
                </>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* 延迟分解 (Latency Breakdown) */}
      <motion.div
        className="glass rounded-xl p-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-orange-400" />
            <h3 className="font-semibold">延迟分解</h3>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="p-3 rounded-lg bg-gray-800/50 text-center">
            <div className="text-xs text-gray-500 mb-1">Compute Time</div>
            <div className={`text-lg font-mono ${isComputeBottleneck ? 'text-red-400' : 'text-gray-400'}`}>
              {formatTime(computeTime)}
            </div>
            {isComputeBottleneck && (
              <div className="text-xs text-red-400 mt-1">← Bottleneck</div>
            )}
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 text-center">
            <div className="text-xs text-gray-500 mb-1">Memory Time</div>
            <div className={`text-lg font-mono ${!isComputeBottleneck ? 'text-red-400' : 'text-gray-400'}`}>
              {formatTime(memoryTime)}
            </div>
            {!isComputeBottleneck && (
              <div className="text-xs text-red-400 mt-1">← Bottleneck</div>
            )}
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 text-center">
            <div className="text-xs text-gray-500 mb-1">Total Latency</div>
            <div className="text-lg font-mono text-orange-400">
              {formatTime(estimatedLatency)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              = max(compute, memory)
            </div>
          </div>
        </div>

        {/* Per-module latency */}
        <div className="pt-4 border-t border-gray-700/50">
          <div className="text-xs text-gray-500 mb-2">Per-Module Latency</div>
          <div className="space-y-1">
            {[
              { name: 'Embedding', time: analysis.embedding.operations.reduce((sum, op) => sum + op.theoreticalTime, 0), color: 'text-blue-400' },
              { name: 'Transformer (×' + analysis.modelConfig.numLayers + ')', time: analysis.transformerBlock.operations.reduce((sum, op) => sum + op.theoreticalTime, 0) * analysis.modelConfig.numLayers, color: 'text-purple-400' },
              { name: 'LM Head', time: analysis.lmHead.operations.reduce((sum, op) => sum + op.theoreticalTime, 0), color: 'text-orange-400' },
            ].map((module) => {
              const percentage = (module.time / estimatedLatency) * 100;
              return (
                <div key={module.name} className="flex items-center gap-2">
                  <div className="w-36 text-xs text-gray-400 truncate">{module.name}</div>
                  <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-full bg-current ${module.color}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${percentage}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  <div className={`w-20 text-right text-xs font-mono ${module.color}`}>
                    {formatTime(module.time)}
                  </div>
                  <div className="w-12 text-right text-xs text-gray-500">
                    {percentage.toFixed(1)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
