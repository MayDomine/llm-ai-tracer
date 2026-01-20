import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, HardDrive, Zap, Clock, Layers, Settings2, BarChart3 } from 'lucide-react';

import { analyzeModel } from './utils/calculator';
import { PRESET_MODELS, PRESET_HARDWARE, MODEL_CATEGORIES, NVIDIA_H100_SXM } from './config/presets';
import type { ModelConfig, HardwareConfig, InferenceConfig, ModelAnalysis, OperationAnalysis } from './types/model';

import { ModuleCard } from './components/ModuleCard';
import { StatsPanel } from './components/StatsPanel';
import { ConfigPanel } from './components/ConfigPanel';
import { RooflineChart } from './components/RooflineChart';
import { ModelImportModal } from './components/ModelImportModal';

function App() {
  // 状态管理
  const [selectedModel, setSelectedModel] = useState<ModelConfig>(PRESET_MODELS[3]); // Qwen3-8B
  const [selectedHardware, setSelectedHardware] = useState<HardwareConfig>(NVIDIA_H100_SXM);
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>({
    mode: 'prefill',
    batchSize: 1,
    seqLen: 512,
    kvCacheLen: 512,
    dtype: 'fp16',
  });
  const [expandedModule, setExpandedModule] = useState<string | null>('attention');
  const [showRoofline, setShowRoofline] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [customModelsVersion, setCustomModelsVersion] = useState(0);

  // 计算分析结果
  const analysis: ModelAnalysis = useMemo(() => {
    return analyzeModel(selectedModel, selectedHardware, inferenceConfig);
  }, [selectedModel, selectedHardware, inferenceConfig]);

  // 获取所有操作用于Roofline图
  const allOperations = useMemo(() => {
    const ops: OperationAnalysis[] = [];
    ops.push(...analysis.embedding.operations);
    ops.push(...analysis.transformerBlock.operations);
    ops.push(...analysis.lmHead.operations);
    return ops;
  }, [analysis]);

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <header className="sticky top-0 z-50 glass border-b border-gray-700/50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight">LLM AI Tracer</h1>
                <p className="text-xs text-gray-400">大模型算术密度可视化分析</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button
                onClick={() => setShowRoofline(!showRoofline)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  showRoofline 
                    ? 'bg-indigo-600 text-white' 
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span className="text-sm font-medium">Roofline</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* 左侧配置面板 */}
          <div className="lg:col-span-1 space-y-4">
            <ConfigPanel
              selectedModel={selectedModel}
              selectedHardware={selectedHardware}
              inferenceConfig={inferenceConfig}
              modelCategories={MODEL_CATEGORIES}
              presetHardware={PRESET_HARDWARE}
              onModelChange={setSelectedModel}
              onHardwareChange={setSelectedHardware}
              onInferenceConfigChange={setInferenceConfig}
              onOpenImportModal={() => setShowImportModal(true)}
              customModelsVersion={customModelsVersion}
            />
          </div>

          {/* 右侧主内容 */}
          <div className="lg:col-span-3 space-y-6">
            {/* 统计面板 */}
            <StatsPanel analysis={analysis} />

            {/* Roofline 图表 */}
            <AnimatePresence>
              {showRoofline && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <RooflineChart
                    operations={allOperations}
                    hardwareConfig={selectedHardware}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            {/* 模块结构展示 */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-indigo-400" />
                <h2 className="text-lg font-semibold">模型结构分析</h2>
                <span className="text-sm text-gray-400">
                  (共 {selectedModel.numLayers} 层 Transformer)
                </span>
                {inferenceConfig.mode === 'decode' && (
                  <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30">
                    Decode: 1 token + {inferenceConfig.kvCacheLen} KV Cache
                  </span>
                )}
              </div>

              {/* Embedding */}
              <ModuleCard
                module={analysis.embedding}
                isExpanded={expandedModule === 'embedding'}
                onToggle={() => setExpandedModule(expandedModule === 'embedding' ? null : 'embedding')}
                icon={<HardDrive className="w-5 h-5" />}
                color="from-blue-500 to-cyan-500"
                layerMultiplier={1}
              />

              {/* Transformer Block */}
              <div className="relative">
                <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-indigo-500/50 via-purple-500/50 to-pink-500/50" />
                
                <div className="ml-0 space-y-3">
                  {/* Attention */}
                  <ModuleCard
                    module={{
                      name: 'Multi-Head Attention',
                      operations: analysis.transformerBlock.operations.filter(op => op.module === 'attention'),
                      totalFlops: analysis.transformerBlock.operations.filter(op => op.module === 'attention').reduce((sum, op) => sum + op.flops, 0),
                      totalMemoryBytes: analysis.transformerBlock.operations.filter(op => op.module === 'attention').reduce((sum, op) => sum + op.memoryBytes, 0),
                      avgArithmeticIntensity: 0,
                      computeBoundOps: analysis.transformerBlock.operations.filter(op => op.module === 'attention' && op.isComputeBound).length,
                      memoryBoundOps: analysis.transformerBlock.operations.filter(op => op.module === 'attention' && !op.isComputeBound).length,
                    }}
                    isExpanded={expandedModule === 'attention'}
                    onToggle={() => setExpandedModule(expandedModule === 'attention' ? null : 'attention')}
                    icon={<Cpu className="w-5 h-5" />}
                    color="from-indigo-500 to-purple-500"
                    subtitle={selectedModel.attentionType.toUpperCase()}
                    layerMultiplier={selectedModel.numLayers}
                  />

                  {/* FFN */}
                  <ModuleCard
                    module={{
                      name: 'Feed-Forward Network',
                      operations: analysis.transformerBlock.operations.filter(op => op.module === 'ffn'),
                      totalFlops: analysis.transformerBlock.operations.filter(op => op.module === 'ffn').reduce((sum, op) => sum + op.flops, 0),
                      totalMemoryBytes: analysis.transformerBlock.operations.filter(op => op.module === 'ffn').reduce((sum, op) => sum + op.memoryBytes, 0),
                      avgArithmeticIntensity: 0,
                      computeBoundOps: analysis.transformerBlock.operations.filter(op => op.module === 'ffn' && op.isComputeBound).length,
                      memoryBoundOps: analysis.transformerBlock.operations.filter(op => op.module === 'ffn' && !op.isComputeBound).length,
                    }}
                    isExpanded={expandedModule === 'ffn'}
                    onToggle={() => setExpandedModule(expandedModule === 'ffn' ? null : 'ffn')}
                    icon={<Settings2 className="w-5 h-5" />}
                    color="from-purple-500 to-pink-500"
                    subtitle={selectedModel.ffnType.toUpperCase()}
                    layerMultiplier={selectedModel.numLayers}
                  />
                </div>
              </div>

              {/* LM Head */}
              <ModuleCard
                module={analysis.lmHead}
                isExpanded={expandedModule === 'lm_head'}
                onToggle={() => setExpandedModule(expandedModule === 'lm_head' ? null : 'lm_head')}
                icon={<Zap className="w-5 h-5" />}
                color="from-orange-500 to-red-500"
                layerMultiplier={1}
              />
            </div>

            {/* 图例说明 */}
            <div className="glass rounded-xl p-4 mt-6">
              <h3 className="text-sm font-medium text-gray-300 mb-3">图例说明</h3>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-compute-bound" />
                  <span className="text-gray-400">计算密集 (Compute Bound)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-memory-bound" />
                  <span className="text-gray-400">内存密集 (Memory Bound)</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-400">理论延迟</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3">
                算术密度 = FLOPs / Memory Bytes。当算术密度 ≥ Roofline拐点 ({(selectedHardware.computeCapability / selectedHardware.memoryBandwidth).toFixed(1)} FLOP/Byte) 时为计算密集，否则为内存密集。
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          <p>LLM AI Tracer - 基于 Roofline 模型的大模型性能分析工具</p>
        </div>
      </footer>

      {/* 模型导入模态框 */}
      <ModelImportModal
        isOpen={showImportModal}
        onClose={() => setShowImportModal(false)}
        onModelImported={() => setCustomModelsVersion(v => v + 1)}
      />
    </div>
  );
}

export default App;
