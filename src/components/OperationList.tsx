import { motion } from 'framer-motion';
import { Clock, ArrowRight } from 'lucide-react';
import type { OperationAnalysis } from '../types/model';
import { formatNumber, formatBytes, formatTime } from '../utils/calculator';

interface OperationListProps {
  operations: OperationAnalysis[];
}

const TYPE_ICONS: Record<string, string> = {
  gemm: '⊗',
  attention: '👁',
  softmax: 'σ',
  layernorm: 'N',
  rmsnorm: 'R',
  embedding: 'E',
  silu: 'ƒ',
  gelu: 'ƒ',
  elementwise: '⊕',
};

const TYPE_LABELS: Record<string, string> = {
  gemm: 'GEMM',
  attention: 'Attention',
  softmax: 'Softmax',
  layernorm: 'LayerNorm',
  rmsnorm: 'RMSNorm',
  embedding: 'Embedding',
  silu: 'SiLU',
  gelu: 'GELU',
  elementwise: 'Elementwise',
};

export function OperationList({ operations }: OperationListProps) {
  return (
    <div className="divide-y divide-gray-800">
      {operations.map((op, index) => (
        <motion.div
          key={op.id}
          className={`px-4 py-3 ${op.isComputeBound ? 'op-compute-bound' : 'op-memory-bound'}`}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.03 }}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 flex items-center justify-center rounded bg-gray-800 text-xs font-mono">
                  {TYPE_ICONS[op.type] || '?'}
                </span>
                <span className="font-medium text-sm truncate">{op.name}</span>
                <span className="px-1.5 py-0.5 text-[10px] rounded bg-gray-800 text-gray-400 uppercase tracking-wide">
                  {TYPE_LABELS[op.type] || op.type}
                </span>
              </div>
              
              {op.shape && (
                <div className="mt-1 flex items-center gap-1 text-xs text-gray-500 font-mono">
                  <span>[{op.shape.m?.toLocaleString()}, {op.shape.k?.toLocaleString()}]</span>
                  <span className="text-gray-600">×</span>
                  <span>[{op.shape.k?.toLocaleString()}, {op.shape.n?.toLocaleString()}]</span>
                  <ArrowRight className="w-3 h-3 mx-1" />
                  <span>[{op.shape.m?.toLocaleString()}, {op.shape.n?.toLocaleString()}]</span>
                </div>
              )}
              
              {!op.shape && (
                <p className="mt-1 text-xs text-gray-500 truncate">{op.description}</p>
              )}
            </div>

            <div className="flex items-center gap-4 text-xs shrink-0">
              {/* FLOPs */}
              <div className="text-right">
                <div className="text-gray-500">FLOPs</div>
                <div className="font-mono">{formatNumber(op.flops)}</div>
              </div>

              {/* Memory */}
              <div className="text-right">
                <div className="text-gray-500">Memory</div>
                <div className="font-mono">{formatBytes(op.memoryBytes)}</div>
              </div>

              {/* AI */}
              <div className="text-right w-16">
                <div className="text-gray-500">AI</div>
                <div className="font-mono">{op.arithmeticIntensity.toFixed(1)}</div>
              </div>

              {/* Time */}
              <div className="text-right w-20">
                <div className="text-gray-500 flex items-center gap-1 justify-end">
                  <Clock className="w-3 h-3" /> Time
                </div>
                <div className="font-mono">{formatTime(op.theoreticalTime)}</div>
              </div>

              {/* Bound indicator */}
              <div className={`w-20 px-2 py-1 rounded text-center text-xs font-medium ${
                op.isComputeBound 
                  ? 'bg-compute-bound/20 text-compute-bound' 
                  : 'bg-memory-bound/20 text-memory-bound'
              }`}>
                {op.isComputeBound ? 'Compute' : 'Memory'}
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
