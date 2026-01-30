import { motion } from 'framer-motion';
import { HardDrive, AlertTriangle, CheckCircle, Database } from 'lucide-react';
import type { MemoryAnalysis } from '../types/model';
import { formatMemorySize } from '../utils/memoryCalculator';

interface MemoryPanelProps {
  memoryAnalysis: MemoryAnalysis;
  gpuName?: string;
}

export function MemoryPanel({ memoryAnalysis }: MemoryPanelProps) {
  const {
    breakdown,
    total,
    peakMemory,
    gpuCapacity,
    utilizationPercent,
    fitsInGPU,
  } = memoryAnalysis;

  return (
    <motion.div
      className="glass rounded-xl overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-cyan-400" />
          <h3 className="font-semibold">GPU Memory Analysis</h3>
        </div>
        {gpuCapacity && (
          <div className="flex items-center gap-2">
            {fitsInGPU ? (
              <CheckCircle className="w-4 h-4 text-green-400" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-red-400" />
            )}
            <span className={`text-sm ${fitsInGPU ? 'text-green-400' : 'text-red-400'}`}>
              {fitsInGPU ? 'Fits in GPU' : 'Exceeds GPU Memory'}
            </span>
          </div>
        )}
      </div>

      <div className="p-4 space-y-4">
        {/* 总内存使用 */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-400">Total Memory Required</div>
            <div className="text-2xl font-bold font-mono">{formatMemorySize(total)}</div>
            {peakMemory !== total && (
              <div className="text-xs text-gray-500">
                Peak: {formatMemorySize(peakMemory)}
              </div>
            )}
          </div>
          {gpuCapacity && (
            <div className="text-right">
              <div className="text-sm text-gray-400">GPU Capacity</div>
              <div className="text-xl font-mono">{formatMemorySize(gpuCapacity)}</div>
              <div className={`text-sm font-mono ${
                utilizationPercent! > 90 ? 'text-red-400' : 
                utilizationPercent! > 70 ? 'text-yellow-400' : 
                'text-green-400'
              }`}>
                {utilizationPercent?.toFixed(1)}% used
              </div>
            </div>
          )}
        </div>

        {/* GPU容量条 */}
        {gpuCapacity && (
          <div className="relative h-8 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className={`h-full ${
                utilizationPercent! > 95 ? 'bg-red-500' :
                utilizationPercent! > 80 ? 'bg-yellow-500' :
                'bg-gradient-to-r from-cyan-500 to-blue-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(utilizationPercent!, 100)}%` }}
              transition={{ duration: 0.5 }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
              {formatMemorySize(peakMemory)} / {formatMemorySize(gpuCapacity)}
            </div>
          </div>
        )}

        {/* 内存分解图 */}
        <div className="space-y-2">
          <div className="text-sm text-gray-400">Memory Breakdown</div>
          
          {/* 堆叠条形图 */}
          <div className="h-10 rounded-lg overflow-hidden flex">
            {breakdown.map((item, index) => (
              <motion.div
                key={item.name}
                className="h-full flex items-center justify-center text-xs font-medium text-white overflow-hidden"
                style={{ backgroundColor: item.color }}
                initial={{ width: 0 }}
                animate={{ width: `${item.percentage}%` }}
                transition={{ duration: 0.5, delay: index * 0.05 }}
                title={`${item.name}: ${formatMemorySize(item.bytes)} (${item.percentage.toFixed(1)}%)`}
              >
                {item.percentage > 10 && (
                  <span className="truncate px-1">{item.percentage.toFixed(0)}%</span>
                )}
              </motion.div>
            ))}
          </div>

          {/* 图例 */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-3">
            {breakdown.map((item) => (
              <div key={item.name} className="flex items-center gap-2 text-sm">
                <div 
                  className="w-3 h-3 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: item.color }}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-gray-300 truncate">{item.name}</div>
                  <div className="text-xs text-gray-500 font-mono">
                    {formatMemorySize(item.bytes)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 详细数据表格 */}
        <div className="mt-4 pt-4 border-t border-gray-700/50">
          <div className="text-sm text-gray-400 mb-2">Detailed Breakdown</div>
          <div className="space-y-1">
            {breakdown.map((item) => (
              <div 
                key={item.name}
                className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-gray-800/50"
              >
                <div className="flex items-center gap-2">
                  <div 
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-300">{item.name}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-sm font-mono text-gray-400">
                    {formatMemorySize(item.bytes)}
                  </span>
                  <span className="text-sm font-mono text-gray-500 w-16 text-right">
                    {item.percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 优化建议 */}
        {(utilizationPercent && utilizationPercent > 90) && (
          <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <div className="text-red-400 font-medium">Memory Warning</div>
                <p className="text-gray-400 mt-1">
                  GPU memory usage is very high. Consider:
                </p>
                <ul className="text-gray-400 mt-1 list-disc list-inside">
                  <li>Using tensor parallelism to split model across GPUs</li>
                  <li>Enabling gradient checkpointing (activation recomputation)</li>
                  <li>Reducing batch size or sequence length</li>
                  <li>Using quantization (INT8/INT4)</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// 简化版内存指示器 (用于侧边栏)
interface MemoryIndicatorProps {
  memoryAnalysis: MemoryAnalysis;
}

export function MemoryIndicator({ memoryAnalysis }: MemoryIndicatorProps) {
  const { total, gpuCapacity, utilizationPercent, fitsInGPU } = memoryAnalysis;

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Database className="w-4 h-4 text-cyan-400" />
        <h3 className="font-medium text-sm">Memory Usage</h3>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-400">Required</span>
          <span className="font-mono text-sm">{formatMemorySize(total)}</span>
        </div>

        {gpuCapacity && (
          <>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">GPU Memory</span>
              <span className="font-mono text-sm">{formatMemorySize(gpuCapacity)}</span>
            </div>

            <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full ${
                  utilizationPercent! > 95 ? 'bg-red-500' :
                  utilizationPercent! > 80 ? 'bg-yellow-500' :
                  'bg-cyan-500'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(utilizationPercent!, 100)}%` }}
              />
            </div>

            <div className="flex justify-between items-center text-xs">
              <span className={fitsInGPU ? 'text-green-400' : 'text-red-400'}>
                {fitsInGPU ? 'OK' : 'OOM Risk'}
              </span>
              <span className="text-gray-500">{utilizationPercent?.toFixed(1)}%</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
