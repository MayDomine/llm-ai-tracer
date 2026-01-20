import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Cpu, HardDrive } from 'lucide-react';
import type { ModuleAnalysis } from '../types/model';
import { formatNumber, formatBytes } from '../utils/calculator';
import { OperationList } from './OperationList';

interface ModuleCardProps {
  module: ModuleAnalysis;
  isExpanded: boolean;
  onToggle: () => void;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
  layerMultiplier?: number;
}

export function ModuleCard({
  module,
  isExpanded,
  onToggle,
  icon,
  color,
  subtitle,
  layerMultiplier = 1,
}: ModuleCardProps) {
  const totalFlops = module.totalFlops * layerMultiplier;
  const totalMemory = module.totalMemoryBytes * layerMultiplier;
  const avgAI = module.totalFlops / module.totalMemoryBytes;

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
            <div className="flex items-center gap-2">
              <h3 className="font-semibold">{module.name}</h3>
              {subtitle && (
                <span className="px-2 py-0.5 text-xs rounded-full bg-gray-700 text-gray-300">
                  {subtitle}
                </span>
              )}
              {layerMultiplier > 1 && (
                <span className="px-2 py-0.5 text-xs rounded-full bg-indigo-900/50 text-indigo-300">
                  ×{layerMultiplier} layers
                </span>
              )}
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-400 mt-1">
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                {formatNumber(totalFlops)}FLOPs
              </span>
              <span className="flex items-center gap-1">
                <HardDrive className="w-3 h-3" />
                {formatBytes(totalMemory)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* 算术密度指示器 */}
          <div className="hidden sm:flex items-center gap-3">
            <div className="text-right">
              <div className="text-xs text-gray-500">算术密度</div>
              <div className="font-mono text-sm">{avgAI.toFixed(2)} FLOP/B</div>
            </div>
            <div className="flex gap-1">
              <div className="flex items-center gap-1 px-2 py-1 rounded bg-compute-bound/20 text-compute-bound text-xs">
                <span className="w-2 h-2 rounded-full bg-compute-bound" />
                {module.computeBoundOps}
              </div>
              <div className="flex items-center gap-1 px-2 py-1 rounded bg-memory-bound/20 text-memory-bound text-xs">
                <span className="w-2 h-2 rounded-full bg-memory-bound" />
                {module.memoryBoundOps}
              </div>
            </div>
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
              <OperationList operations={module.operations} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
