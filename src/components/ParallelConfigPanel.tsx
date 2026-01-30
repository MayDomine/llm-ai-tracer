import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Network, Layers, Zap, ChevronRight, Wand2, AlertCircle } from 'lucide-react';
import type { ParallelConfig, ModelConfig, HardwareConfig, InferenceConfig } from '../types/model';
import { quickRecommendStrategy, generateStrategyDescription, getStrategyProsCons } from '../utils/strategyOptimizer';

interface ParallelConfigPanelProps {
  modelConfig: ModelConfig;
  hardwareConfig: HardwareConfig;
  inferenceConfig: InferenceConfig;
  parallelConfig: ParallelConfig;
  onConfigChange: (config: ParallelConfig) => void;
  numGPUs: number;
  onNumGPUsChange: (n: number) => void;
}

export function ParallelConfigPanel({
  modelConfig,
  hardwareConfig,
  inferenceConfig,
  parallelConfig,
  onConfigChange,
  numGPUs,
  onNumGPUsChange,
}: ParallelConfigPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // 验证当前配置
  const validationErrors = useMemo(() => {
    const errors: string[] = [];
    const { dataParallel, tensorParallel, pipelineParallel } = parallelConfig;
    const totalGPUs = dataParallel * tensorParallel * pipelineParallel;

    if (totalGPUs !== numGPUs) {
      errors.push(`DP × TP × PP (${totalGPUs}) ≠ Total GPUs (${numGPUs})`);
    }
    if (modelConfig.numAttentionHeads % tensorParallel !== 0) {
      errors.push(`TP (${tensorParallel}) must divide attention heads (${modelConfig.numAttentionHeads})`);
    }
    if (modelConfig.numLayers % pipelineParallel !== 0) {
      errors.push(`PP (${pipelineParallel}) must divide layers (${modelConfig.numLayers})`);
    }

    return errors;
  }, [parallelConfig, numGPUs, modelConfig]);

  // 获取策略说明和优缺点
  const strategyDescription = useMemo(() => 
    generateStrategyDescription(parallelConfig), [parallelConfig]);
  
  const { pros, cons } = useMemo(() => 
    getStrategyProsCons(parallelConfig), [parallelConfig]);

  // 可用的TP选项 (必须能整除attention heads且为2的幂)
  const tpOptions = useMemo(() => {
    const options: number[] = [1];
    for (let tp = 2; tp <= Math.min(8, numGPUs); tp *= 2) {
      if (modelConfig.numAttentionHeads % tp === 0) {
        options.push(tp);
      }
    }
    return options;
  }, [modelConfig.numAttentionHeads, numGPUs]);

  // 可用的PP选项 (必须能整除layers)
  const ppOptions = useMemo(() => {
    const options: number[] = [];
    for (let pp = 1; pp <= Math.min(modelConfig.numLayers, numGPUs); pp++) {
      if (modelConfig.numLayers % pp === 0 && numGPUs % pp === 0) {
        options.push(pp);
      }
    }
    return options.slice(0, 8); // 限制选项数量
  }, [modelConfig.numLayers, numGPUs]);

  // 自动推荐策略
  const handleAutoOptimize = () => {
    const recommended = quickRecommendStrategy(
      modelConfig, hardwareConfig, inferenceConfig, numGPUs
    );
    onConfigChange(recommended);
  };

  // 更新配置时自动计算DP
  const handleTPChange = (tp: number) => {
    const remaining = numGPUs / tp;
    const pp = Math.min(parallelConfig.pipelineParallel, remaining);
    const dp = remaining / pp;
    onConfigChange({
      ...parallelConfig,
      tensorParallel: tp,
      pipelineParallel: pp,
      dataParallel: dp,
    });
  };

  const handlePPChange = (pp: number) => {
    const remaining = numGPUs / parallelConfig.tensorParallel;
    const dp = remaining / pp;
    onConfigChange({
      ...parallelConfig,
      pipelineParallel: pp,
      dataParallel: dp,
    });
  };

  return (
    <motion.div
      className="glass rounded-xl overflow-hidden"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
    >
      <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-purple-400" />
          <h3 className="font-medium text-sm">Parallel Strategy</h3>
        </div>
        <button
          onClick={handleAutoOptimize}
          className="flex items-center gap-1 px-2 py-1 text-xs bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 rounded transition-colors"
        >
          <Wand2 className="w-3 h-3" />
          Auto
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* GPU数量 */}
        <div>
          <label className="text-sm text-gray-400 block mb-2">Total GPUs</label>
          <div className="grid grid-cols-6 gap-1">
            {[1, 2, 4, 8, 16, 32].map((n) => (
              <button
                key={n}
                onClick={() => {
                  onNumGPUsChange(n);
                  // 重新计算配置
                  const newConfig = quickRecommendStrategy(
                    modelConfig, hardwareConfig, inferenceConfig, n
                  );
                  onConfigChange(newConfig);
                }}
                className={`px-2 py-1.5 text-xs rounded transition-all ${
                  numGPUs === n
                    ? 'bg-purple-600/50 text-purple-300 border border-purple-500/50'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* 当前策略描述 */}
        <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
          <div className="text-sm font-medium text-gray-300">{strategyDescription}</div>
          <div className="text-xs text-gray-500 mt-1">
            {parallelConfig.dataParallel} × {parallelConfig.tensorParallel} × {parallelConfig.pipelineParallel} = {numGPUs} GPUs
          </div>
        </div>

        {/* 验证错误 */}
        {validationErrors.length > 0 && (
          <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/30">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-red-400">
                {validationErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 并行配置滑块 */}
        <div className="space-y-3">
          {/* Tensor Parallel */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                Tensor Parallel (TP)
              </label>
              <span className="font-mono text-sm text-purple-400">
                {parallelConfig.tensorParallel}
              </span>
            </div>
            <div className="flex gap-1">
              {tpOptions.map((tp) => (
                <button
                  key={tp}
                  onClick={() => handleTPChange(tp)}
                  disabled={numGPUs % tp !== 0}
                  className={`flex-1 px-2 py-1 text-xs rounded transition-all ${
                    parallelConfig.tensorParallel === tp
                      ? 'bg-purple-600/50 text-purple-300'
                      : numGPUs % tp !== 0
                        ? 'bg-gray-800/30 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                  }`}
                >
                  {tp}
                </button>
              ))}
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Splits model layers horizontally across GPUs
            </div>
          </div>

          {/* Pipeline Parallel */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                <Layers className="w-3 h-3" />
                Pipeline Parallel (PP)
              </label>
              <span className="font-mono text-sm text-purple-400">
                {parallelConfig.pipelineParallel}
              </span>
            </div>
            <div className="flex gap-1 flex-wrap">
              {ppOptions.map((pp) => {
                const remainingGPUs = numGPUs / parallelConfig.tensorParallel;
                const isValid = remainingGPUs % pp === 0;
                return (
                  <button
                    key={pp}
                    onClick={() => handlePPChange(pp)}
                    disabled={!isValid}
                    className={`px-2 py-1 text-xs rounded transition-all ${
                      parallelConfig.pipelineParallel === pp
                        ? 'bg-purple-600/50 text-purple-300'
                        : !isValid
                          ? 'bg-gray-800/30 text-gray-600 cursor-not-allowed'
                          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                    }`}
                  >
                    {pp}
                  </button>
                );
              })}
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Splits model layers vertically into pipeline stages
            </div>
          </div>

          {/* Data Parallel (自动计算) */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400 flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Data Parallel (DP)
              </label>
              <span className="font-mono text-sm text-green-400">
                {parallelConfig.dataParallel} (auto)
              </span>
            </div>
            <div className="text-xs text-gray-600">
              Calculated as Total GPUs ÷ (TP × PP)
            </div>
          </div>
        </div>

        {/* 高级选项 */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-sm text-gray-400 hover:text-gray-300 transition-colors"
        >
          <motion.div
            animate={{ rotate: showAdvanced ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight className="w-4 h-4" />
          </motion.div>
          Advanced Options
        </button>

        {showAdvanced && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="space-y-3 pt-2"
          >
            {/* Micro Batch Size */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">Micro Batch Size</label>
                <span className="font-mono text-sm text-gray-300">
                  {parallelConfig.microBatchSize}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max={Math.max(1, inferenceConfig.batchSize)}
                value={parallelConfig.microBatchSize}
                onChange={(e) => onConfigChange({
                  ...parallelConfig,
                  microBatchSize: Number(e.target.value),
                })}
                className="w-full"
              />
            </div>

            {/* Gradient Accumulation */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">Gradient Accumulation</label>
                <span className="font-mono text-sm text-gray-300">
                  {parallelConfig.gradientAccumulation}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="32"
                value={parallelConfig.gradientAccumulation}
                onChange={(e) => onConfigChange({
                  ...parallelConfig,
                  gradientAccumulation: Number(e.target.value),
                })}
                className="w-full"
              />
            </div>

            {/* Sequence Parallel */}
            <div className="flex items-center justify-between">
              <label className="text-sm text-gray-400">Sequence Parallel</label>
              <button
                onClick={() => onConfigChange({
                  ...parallelConfig,
                  sequenceParallel: !parallelConfig.sequenceParallel,
                })}
                className={`px-3 py-1 text-xs rounded transition-all ${
                  parallelConfig.sequenceParallel
                    ? 'bg-green-600/30 text-green-300'
                    : 'bg-gray-800/50 text-gray-400'
                }`}
              >
                {parallelConfig.sequenceParallel ? 'Enabled' : 'Disabled'}
              </button>
            </div>
          </motion.div>
        )}

        {/* 策略优缺点 */}
        {(pros.length > 0 || cons.length > 0) && (
          <div className="pt-3 border-t border-gray-700/50 space-y-2">
            {pros.length > 0 && (
              <div>
                <div className="text-xs text-green-400 mb-1">Advantages:</div>
                <ul className="text-xs text-gray-400 space-y-0.5">
                  {pros.slice(0, 3).map((pro, i) => (
                    <li key={i} className="flex items-start gap-1">
                      <span className="text-green-400">+</span>
                      {pro}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {cons.length > 0 && (
              <div>
                <div className="text-xs text-orange-400 mb-1">Trade-offs:</div>
                <ul className="text-xs text-gray-400 space-y-0.5">
                  {cons.slice(0, 3).map((con, i) => (
                    <li key={i} className="flex items-start gap-1">
                      <span className="text-orange-400">-</span>
                      {con}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
}
