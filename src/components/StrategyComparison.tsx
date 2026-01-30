import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, 
  Trophy, 
  AlertCircle, 
  CheckCircle, 
  Clock, 
  HardDrive,
  Network,
  ArrowUpDown,
  ChevronDown,
  X
} from 'lucide-react';
import type { 
  ModelConfig, 
  HardwareConfig, 
  InferenceConfig, 
  StrategyAnalysis
} from '../types/model';
import { findOptimalStrategies } from '../utils/strategyOptimizer';
import { formatMemorySize } from '../utils/memoryCalculator';
import { formatCommBytes } from '../utils/communicationCalculator';

interface StrategyComparisonProps {
  modelConfig: ModelConfig;
  hardwareConfig: HardwareConfig;
  inferenceConfig: InferenceConfig;
  numGPUs: number;
  isOpen: boolean;
  onClose: () => void;
  onSelectStrategy: (config: StrategyAnalysis['config']) => void;
}

type SortKey = 'throughput' | 'latency' | 'memory' | 'efficiency' | 'score';
type SortDirection = 'asc' | 'desc';

export function StrategyComparisonModal({
  modelConfig,
  hardwareConfig,
  inferenceConfig,
  numGPUs,
  isOpen,
  onClose,
  onSelectStrategy,
}: StrategyComparisonProps) {
  const [sortKey, setSortKey] = useState<SortKey>('throughput');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [showInvalid, setShowInvalid] = useState(false);
  const [objective, setObjective] = useState<'throughput' | 'latency' | 'memory'>('throughput');

  // 分析所有策略
  const comparison = useMemo(() => {
    return findOptimalStrategies(
      modelConfig, hardwareConfig, inferenceConfig, numGPUs, objective
    );
  }, [modelConfig, hardwareConfig, inferenceConfig, numGPUs, objective]);

  // 排序策略
  const sortedStrategies = useMemo(() => {
    const strategies = showInvalid 
      ? comparison.strategies 
      : comparison.strategies.filter(s => s.isValid);

    return [...strategies].sort((a, b) => {
      let aVal: number, bVal: number;
      
      switch (sortKey) {
        case 'throughput':
          aVal = a.throughputTokensPerSec;
          bVal = b.throughputTokensPerSec;
          break;
        case 'latency':
          aVal = a.latencyMs;
          bVal = b.latencyMs;
          break;
        case 'memory':
          aVal = a.memoryAnalysis.peakMemory;
          bVal = b.memoryAnalysis.peakMemory;
          break;
        case 'efficiency':
          aVal = a.communicationAnalysis.efficiency;
          bVal = b.communicationAnalysis.efficiency;
          break;
        case 'score':
          aVal = a.score;
          bVal = b.score;
          break;
        default:
          aVal = a.score;
          bVal = b.score;
      }

      const multiplier = sortDirection === 'asc' ? 1 : -1;
      return (aVal - bVal) * multiplier;
    });
  }, [comparison.strategies, sortKey, sortDirection, showInvalid]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection(key === 'latency' || key === 'memory' ? 'asc' : 'desc');
    }
  };

  const formatThroughput = (tokensPerSec: number) => {
    if (tokensPerSec >= 1e6) return (tokensPerSec / 1e6).toFixed(2) + 'M';
    if (tokensPerSec >= 1e3) return (tokensPerSec / 1e3).toFixed(1) + 'K';
    return tokensPerSec.toFixed(0);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="glass rounded-2xl w-full max-w-5xl max-h-[85vh] overflow-hidden m-4"
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <div>
                <h2 className="font-semibold">Strategy Comparison</h2>
                <p className="text-xs text-gray-400">
                  {numGPUs} GPUs • {comparison.strategies.filter(s => s.isValid).length} valid configurations
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-700/50 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Controls */}
          <div className="px-6 py-3 border-b border-gray-700/30 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">Optimize for:</span>
                <div className="flex gap-1">
                  {(['throughput', 'latency', 'memory'] as const).map((obj) => (
                    <button
                      key={obj}
                      onClick={() => setObjective(obj)}
                      className={`px-3 py-1 text-xs rounded transition-all ${
                        objective === obj
                          ? 'bg-purple-600/50 text-purple-300'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                      }`}
                    >
                      {obj.charAt(0).toUpperCase() + obj.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-400">
              <input
                type="checkbox"
                checked={showInvalid}
                onChange={(e) => setShowInvalid(e.target.checked)}
                className="rounded"
              />
              Show invalid configs
            </label>
          </div>

          {/* Recommended Strategy */}
          {comparison.recommendedStrategy && (
            <div className="px-6 py-4 bg-purple-600/10 border-b border-purple-500/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Trophy className="w-5 h-5 text-yellow-400" />
                  <div>
                    <div className="text-sm font-medium text-yellow-400">
                      Recommended: {comparison.recommendedStrategy.config.name}
                    </div>
                    <div className="text-xs text-gray-400">
                      {formatThroughput(comparison.recommendedStrategy.throughputTokensPerSec)} tokens/s • 
                      {comparison.recommendedStrategy.latencyMs.toFixed(2)}ms latency • 
                      {formatMemorySize(comparison.recommendedStrategy.memoryAnalysis.peakMemory)} memory
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => onSelectStrategy(comparison.recommendedStrategy!.config)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-sm rounded-lg transition-colors"
                >
                  Apply
                </button>
              </div>
            </div>
          )}

          {/* Table */}
          <div className="overflow-auto max-h-[50vh]">
            <table className="w-full text-sm">
              <thead className="bg-gray-800/50 sticky top-0">
                <tr>
                  <th className="px-4 py-3 text-left text-gray-400 font-medium">Strategy</th>
                  <th 
                    className="px-4 py-3 text-right text-gray-400 font-medium cursor-pointer hover:text-gray-200"
                    onClick={() => handleSort('throughput')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Throughput
                      {sortKey === 'throughput' && (
                        <ArrowUpDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-4 py-3 text-right text-gray-400 font-medium cursor-pointer hover:text-gray-200"
                    onClick={() => handleSort('latency')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Latency
                      {sortKey === 'latency' && (
                        <ArrowUpDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-4 py-3 text-right text-gray-400 font-medium cursor-pointer hover:text-gray-200"
                    onClick={() => handleSort('memory')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Memory
                      {sortKey === 'memory' && (
                        <ArrowUpDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                  <th 
                    className="px-4 py-3 text-right text-gray-400 font-medium cursor-pointer hover:text-gray-200"
                    onClick={() => handleSort('efficiency')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Efficiency
                      {sortKey === 'efficiency' && (
                        <ArrowUpDown className="w-3 h-3" />
                      )}
                    </div>
                  </th>
                  <th className="px-4 py-3 text-center text-gray-400 font-medium">Status</th>
                  <th className="px-4 py-3"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700/30">
                {sortedStrategies.map((strategy) => (
                  <StrategyRow
                    key={strategy.config.name}
                    strategy={strategy}
                    isRecommended={strategy === comparison.recommendedStrategy}
                    onSelect={() => onSelectStrategy(strategy.config)}
                    formatThroughput={formatThroughput}
                  />
                ))}
              </tbody>
            </table>
          </div>

          {/* Footer */}
          <div className="px-6 py-3 border-t border-gray-700/50 text-xs text-gray-500">
            Click on any strategy to apply it. Metrics are theoretical estimates based on the roofline model.
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

interface StrategyRowProps {
  strategy: StrategyAnalysis;
  isRecommended: boolean;
  onSelect: () => void;
  formatThroughput: (n: number) => string;
}

function StrategyRow({ strategy, isRecommended, onSelect, formatThroughput }: StrategyRowProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { config, memoryAnalysis, communicationAnalysis, throughputTokensPerSec, latencyMs, isValid, validationErrors } = strategy;

  return (
    <>
      <tr 
        className={`hover:bg-gray-800/30 transition-colors cursor-pointer ${
          isRecommended ? 'bg-purple-600/10' : ''
        } ${!isValid ? 'opacity-50' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            {isRecommended && <Trophy className="w-4 h-4 text-yellow-400" />}
            <div>
              <div className="font-medium text-gray-200">{config.name}</div>
              <div className="text-xs text-gray-500">
                MB={config.microBatchSize} GA={config.gradientAccumulation}
              </div>
            </div>
          </div>
        </td>
        <td className="px-4 py-3 text-right">
          <div className="font-mono text-green-400">
            {formatThroughput(throughputTokensPerSec)}/s
          </div>
        </td>
        <td className="px-4 py-3 text-right">
          <div className="font-mono text-blue-400">
            {latencyMs.toFixed(2)}ms
          </div>
        </td>
        <td className="px-4 py-3 text-right">
          <div className="font-mono text-cyan-400">
            {formatMemorySize(memoryAnalysis.peakMemory)}
          </div>
          {memoryAnalysis.utilizationPercent && (
            <div className={`text-xs ${
              memoryAnalysis.utilizationPercent > 90 ? 'text-red-400' :
              memoryAnalysis.utilizationPercent > 70 ? 'text-yellow-400' :
              'text-gray-500'
            }`}>
              {memoryAnalysis.utilizationPercent.toFixed(0)}%
            </div>
          )}
        </td>
        <td className="px-4 py-3 text-right">
          <div className="font-mono text-purple-400">
            {(communicationAnalysis.efficiency * 100).toFixed(1)}%
          </div>
        </td>
        <td className="px-4 py-3 text-center">
          {isValid ? (
            <CheckCircle className="w-4 h-4 text-green-400 mx-auto" />
          ) : (
            <AlertCircle className="w-4 h-4 text-red-400 mx-auto" />
          )}
        </td>
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSelect();
              }}
              disabled={!isValid}
              className={`px-3 py-1 text-xs rounded transition-all ${
                isValid
                  ? 'bg-purple-600/30 hover:bg-purple-600/50 text-purple-300'
                  : 'bg-gray-700/30 text-gray-600 cursor-not-allowed'
              }`}
            >
              Apply
            </button>
            <motion.div
              animate={{ rotate: isExpanded ? 180 : 0 }}
            >
              <ChevronDown className="w-4 h-4 text-gray-400" />
            </motion.div>
          </div>
        </td>
      </tr>

      {/* Expanded Details */}
      <AnimatePresence>
        {isExpanded && (
          <motion.tr
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
          >
            <td colSpan={7} className="px-4 py-0 bg-gray-800/20">
              <motion.div
                className="py-3 space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {/* Validation Errors */}
                {validationErrors.length > 0 && (
                  <div className="p-2 rounded bg-red-500/10 border border-red-500/30">
                    <div className="text-xs text-red-400">
                      {validationErrors.map((err, i) => (
                        <div key={i}>{err}</div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Details Grid */}
                <div className="grid grid-cols-3 gap-4 text-xs">
                  {/* Memory Breakdown */}
                  <div className="space-y-1">
                    <div className="text-gray-400 font-medium flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      Memory Breakdown
                    </div>
                    {memoryAnalysis.breakdown.map(item => (
                      <div key={item.name} className="flex justify-between">
                        <span className="text-gray-500">{item.name}</span>
                        <span className="font-mono text-gray-300">
                          {formatMemorySize(item.bytes)}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Communication */}
                  <div className="space-y-1">
                    <div className="text-gray-400 font-medium flex items-center gap-1">
                      <Network className="w-3 h-3" />
                      Communication
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">DP AllReduce</span>
                      <span className="font-mono text-gray-300">
                        {formatCommBytes(communicationAnalysis.dataParallelComm)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">TP AllReduce</span>
                      <span className="font-mono text-gray-300">
                        {formatCommBytes(communicationAnalysis.tensorParallelComm)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">PP P2P</span>
                      <span className="font-mono text-gray-300">
                        {formatCommBytes(communicationAnalysis.pipelineParallelComm)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Bubble Overhead</span>
                      <span className="font-mono text-gray-300">
                        {communicationAnalysis.bubbleOverhead.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Timing */}
                  <div className="space-y-1">
                    <div className="text-gray-400 font-medium flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Timing Breakdown
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">DP Comm Time</span>
                      <span className="font-mono text-gray-300">
                        {communicationAnalysis.dataParallelTime.toFixed(2)}ms
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">TP Comm Time</span>
                      <span className="font-mono text-gray-300">
                        {communicationAnalysis.tensorParallelTime.toFixed(2)}ms
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">PP Comm Time</span>
                      <span className="font-mono text-gray-300">
                        {communicationAnalysis.pipelineParallelTime.toFixed(2)}ms
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Total Comm Time</span>
                      <span className="font-mono text-gray-300">
                        {communicationAnalysis.totalCommTime.toFixed(2)}ms
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            </td>
          </motion.tr>
        )}
      </AnimatePresence>
    </>
  );
}

// 简化版策略选择器 (用于侧边栏)
interface QuickStrategyPickerProps {
  modelConfig: ModelConfig;
  hardwareConfig: HardwareConfig;
  inferenceConfig: InferenceConfig;
  numGPUs: number;
  onOpenComparison: () => void;
}

export function QuickStrategyPicker({
  modelConfig,
  hardwareConfig,
  inferenceConfig,
  numGPUs,
  onOpenComparison,
}: QuickStrategyPickerProps) {
  const comparison = useMemo(() => {
    return findOptimalStrategies(
      modelConfig, hardwareConfig, inferenceConfig, numGPUs, 'throughput'
    );
  }, [modelConfig, hardwareConfig, inferenceConfig, numGPUs]);

  const validCount = comparison.strategies.filter(s => s.isValid).length;
  const recommended = comparison.recommendedStrategy;

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-purple-400" />
          <h3 className="font-medium text-sm">Strategy Analysis</h3>
        </div>
        <span className="text-xs text-gray-500">{validCount} valid</span>
      </div>

      {recommended && (
        <div className="p-2 rounded-lg bg-purple-600/10 border border-purple-500/30 mb-3">
          <div className="flex items-center gap-2">
            <Trophy className="w-4 h-4 text-yellow-400" />
            <span className="text-sm font-medium text-yellow-400">
              {recommended.config.name}
            </span>
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {(recommended.throughputTokensPerSec / 1000).toFixed(1)}K tok/s • 
            {(recommended.communicationAnalysis.efficiency * 100).toFixed(0)}% efficiency
          </div>
        </div>
      )}

      <button
        onClick={onOpenComparison}
        className="w-full py-2 text-sm bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 rounded-lg transition-colors"
      >
        Compare All Strategies
      </button>
    </div>
  );
}
