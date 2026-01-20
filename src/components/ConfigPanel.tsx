import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Settings, Cpu, Server, Sliders, ChevronRight, Play, Zap, GraduationCap, Plus, Trash2 } from 'lucide-react';
import type { ModelConfig, HardwareConfig, InferenceConfig, InferenceMode } from '../types/model';
import { getCustomModels, deleteCustomModel } from '../utils/storage';

const MODE_CONFIG = {
  training: {
    label: 'Training',
    description: '完整前向传播',
    icon: GraduationCap,
    color: 'from-green-500 to-emerald-500',
  },
  prefill: {
    label: 'Prefill',
    description: 'Prompt处理阶段',
    icon: Play,
    color: 'from-blue-500 to-cyan-500',
  },
  decode: {
    label: 'Decode',
    description: '逐Token生成',
    icon: Zap,
    color: 'from-orange-500 to-red-500',
  },
};

interface ConfigPanelProps {
  selectedModel: ModelConfig;
  selectedHardware: HardwareConfig;
  inferenceConfig: InferenceConfig;
  modelCategories: Record<string, ModelConfig[]>;
  presetHardware: HardwareConfig[];
  onModelChange: (model: ModelConfig) => void;
  onHardwareChange: (hardware: HardwareConfig) => void;
  onInferenceConfigChange: (config: InferenceConfig) => void;
  onOpenImportModal: () => void;
  customModelsVersion: number; // 用于触发重新加载
}

export function ConfigPanel({
  selectedModel,
  selectedHardware,
  inferenceConfig,
  modelCategories,
  presetHardware,
  onModelChange,
  onHardwareChange,
  onInferenceConfigChange,
  onOpenImportModal,
  customModelsVersion,
}: ConfigPanelProps) {
  const [expandedCategory, setExpandedCategory] = useState<string | null>('Qwen3');
  const [customModels, setCustomModels] = useState<ModelConfig[]>([]);

  // 加载自定义模型
  useEffect(() => {
    setCustomModels(getCustomModels());
  }, [customModelsVersion]);

  const handleDeleteCustomModel = (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteCustomModel(name);
    setCustomModels(getCustomModels());
  };

  return (
    <div className="space-y-4">
      {/* 模型选择 */}
      <motion.div
        className="glass rounded-xl overflow-hidden"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        <div className="px-4 py-3 border-b border-gray-700/50 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          <h3 className="font-medium text-sm">模型选择</h3>
        </div>
        
        <div className="p-2 max-h-80 overflow-y-auto">
          {/* 预设模型 */}
          {Object.entries(modelCategories).map(([category, models]) => (
            <div key={category} className="mb-1">
              <button
                onClick={() => setExpandedCategory(expandedCategory === category ? null : category)}
                className="w-full px-3 py-2 flex items-center justify-between text-sm hover:bg-gray-800/50 rounded-lg transition-colors"
              >
                <span className="text-gray-300">{category}</span>
                <motion.div
                  animate={{ rotate: expandedCategory === category ? 90 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronRight className="w-4 h-4 text-gray-500" />
                </motion.div>
              </button>
              
              {expandedCategory === category && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="ml-2 space-y-0.5"
                >
                  {models.map((model) => (
                    <button
                      key={model.name}
                      onClick={() => onModelChange(model)}
                      className={`w-full px-3 py-2 text-left text-sm rounded-lg transition-all ${
                        selectedModel.name === model.name
                          ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/50'
                          : 'hover:bg-gray-800/50 text-gray-400'
                      }`}
                    >
                      <div className="font-medium">{model.name}</div>
                      <div className="text-xs text-gray-500 mt-0.5">
                        {model.numLayers}L / {model.hiddenSize}d / {model.attentionType.toUpperCase()}
                        {model.ffnType === 'moe' && ` / ${model.numExperts}E`}
                      </div>
                    </button>
                  ))}
                </motion.div>
              )}
            </div>
          ))}

          {/* 自定义模型 */}
          {customModels.length > 0 && (
            <div className="mb-1">
              <button
                onClick={() => setExpandedCategory(expandedCategory === 'custom' ? null : 'custom')}
                className="w-full px-3 py-2 flex items-center justify-between text-sm hover:bg-gray-800/50 rounded-lg transition-colors"
              >
                <span className="text-amber-400">⭐ 自定义模型</span>
                <motion.div
                  animate={{ rotate: expandedCategory === 'custom' ? 90 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronRight className="w-4 h-4 text-gray-500" />
                </motion.div>
              </button>
              
              {expandedCategory === 'custom' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="ml-2 space-y-0.5"
                >
                  {customModels.map((model) => (
                    <div
                      key={model.name}
                      className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all ${
                        selectedModel.name === model.name
                          ? 'bg-amber-600/30 text-amber-300 border border-amber-500/50'
                          : 'hover:bg-gray-800/50 text-gray-400'
                      }`}
                    >
                      <button
                        onClick={() => onModelChange(model)}
                        className="flex-1 text-left"
                      >
                        <div className="font-medium">{model.name}</div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          {model.numLayers}L / {model.hiddenSize}d / {model.attentionType.toUpperCase()}
                          {model.ffnType === 'moe' && ` / ${model.numExperts}E`}
                        </div>
                      </button>
                      <button
                        onClick={(e) => handleDeleteCustomModel(model.name, e)}
                        className="p-1 text-red-400 hover:bg-red-500/20 rounded transition-colors"
                        title="删除"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </motion.div>
              )}
            </div>
          )}
        </div>

        {/* 导入按钮 */}
        <div className="p-2 border-t border-gray-700/50">
          <button
            onClick={onOpenImportModal}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" />
            导入/创建自定义模型
          </button>
        </div>
      </motion.div>

      {/* 硬件选择 */}
      <motion.div
        className="glass rounded-xl overflow-hidden"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="px-4 py-3 border-b border-gray-700/50 flex items-center gap-2">
          <Server className="w-4 h-4 text-cyan-400" />
          <h3 className="font-medium text-sm">硬件配置</h3>
        </div>
        
        <div className="p-2">
          <select
            value={selectedHardware.name}
            onChange={(e) => {
              const hw = presetHardware.find(h => h.name === e.target.value);
              if (hw) onHardwareChange(hw);
            }}
            className="w-full"
          >
            {presetHardware.map((hw) => (
              <option key={hw.name} value={hw.name}>
                {hw.name}
              </option>
            ))}
          </select>
          
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 rounded-lg bg-gray-800/50">
              <div className="text-gray-500">算力 (FP16)</div>
              <div className="font-mono text-cyan-400">{selectedHardware.computeCapability} TFLOPS</div>
            </div>
            <div className="p-2 rounded-lg bg-gray-800/50">
              <div className="text-gray-500">内存带宽</div>
              <div className="font-mono text-cyan-400">{selectedHardware.memoryBandwidth} TB/s</div>
            </div>
          </div>

          {/* 自定义硬件 */}
          <div className="mt-3 space-y-2">
            <div className="text-xs text-gray-500">自定义参数</div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-400 block mb-1">算力 (TFLOPS)</label>
                <input
                  type="number"
                  value={selectedHardware.computeCapability}
                  onChange={(e) => onHardwareChange({
                    ...selectedHardware,
                    name: 'Custom',
                    computeCapability: Number(e.target.value) || 1,
                  })}
                  className="w-full text-sm"
                  min="1"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">带宽 (TB/s)</label>
                <input
                  type="number"
                  step="0.1"
                  value={selectedHardware.memoryBandwidth}
                  onChange={(e) => onHardwareChange({
                    ...selectedHardware,
                    name: 'Custom',
                    memoryBandwidth: Number(e.target.value) || 0.1,
                  })}
                  className="w-full text-sm"
                  min="0.1"
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 推理模式 */}
      <motion.div
        className="glass rounded-xl overflow-hidden"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="px-4 py-3 border-b border-gray-700/50 flex items-center gap-2">
          <Sliders className="w-4 h-4 text-purple-400" />
          <h3 className="font-medium text-sm">推理模式</h3>
        </div>
        
        <div className="p-3">
          {/* 模式选择 */}
          <div className="grid grid-cols-3 gap-2 mb-4">
            {(Object.entries(MODE_CONFIG) as [InferenceMode, typeof MODE_CONFIG.training][]).map(([mode, config]) => {
              const Icon = config.icon;
              const isActive = inferenceConfig.mode === mode;
              return (
                <button
                  key={mode}
                  onClick={() => onInferenceConfigChange({ ...inferenceConfig, mode })}
                  className={`p-2 rounded-lg transition-all text-center ${
                    isActive
                      ? `bg-gradient-to-br ${config.color} text-white shadow-lg`
                      : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                  }`}
                >
                  <Icon className="w-4 h-4 mx-auto mb-1" />
                  <div className="text-xs font-medium">{config.label}</div>
                </button>
              );
            })}
          </div>
          
          {/* 模式说明 */}
          <div className="text-xs text-gray-500 mb-4 p-2 rounded-lg bg-gray-800/30">
            {inferenceConfig.mode === 'training' && (
              <p>🎓 <strong>Training</strong>: 完整序列前向传播，用于训练或评估整个序列的计算量。</p>
            )}
            {inferenceConfig.mode === 'prefill' && (
              <p>▶️ <strong>Prefill</strong>: 处理输入prompt，生成所有token的KV Cache。类似training但用于推理。</p>
            )}
            {inferenceConfig.mode === 'decode' && (
              <p>⚡ <strong>Decode</strong>: 自回归生成阶段，每次生成1个token，需要读取KV Cache。</p>
            )}
          </div>
        </div>
      </motion.div>

      {/* 输入参数 */}
      <motion.div
        className="glass rounded-xl overflow-hidden"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.25 }}
      >
        <div className="px-4 py-3 border-b border-gray-700/50 flex items-center gap-2">
          <Settings className="w-4 h-4 text-amber-400" />
          <h3 className="font-medium text-sm">输入参数</h3>
          <span className="text-xs text-gray-500 ml-auto">
            {MODE_CONFIG[inferenceConfig.mode].label} 模式
          </span>
        </div>
        
        <div className="p-4 space-y-4">
          {/* Batch Size */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400">Batch Size</label>
              <span className="font-mono text-sm text-purple-400">{inferenceConfig.batchSize}</span>
            </div>
            <input
              type="range"
              min="1"
              max="128"
              value={inferenceConfig.batchSize}
              onChange={(e) => onInferenceConfigChange({
                ...inferenceConfig,
                batchSize: Number(e.target.value),
              })}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>1</span>
              <span>128</span>
            </div>
          </div>

          {/* 根据模式显示不同的参数 */}
          {inferenceConfig.mode !== 'decode' ? (
            /* Training / Prefill: 序列长度 */
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">序列长度 (Seq Length)</label>
                <span className="font-mono text-sm text-purple-400">{inferenceConfig.seqLen}</span>
              </div>
              <input
                type="range"
                min="128"
                max="16384"
                step="128"
                value={inferenceConfig.seqLen}
                onChange={(e) => onInferenceConfigChange({
                  ...inferenceConfig,
                  seqLen: Number(e.target.value),
                })}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>128</span>
                <span>16384</span>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                处理的token数 = Batch × SeqLen = {inferenceConfig.batchSize * inferenceConfig.seqLen}
              </p>
            </div>
          ) : (
            /* Decode: KV Cache长度 */
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-gray-400">KV Cache 长度</label>
                <span className="font-mono text-sm text-orange-400">{inferenceConfig.kvCacheLen}</span>
              </div>
              <input
                type="range"
                min="1"
                max="16384"
                step="1"
                value={inferenceConfig.kvCacheLen}
                onChange={(e) => onInferenceConfigChange({
                  ...inferenceConfig,
                  kvCacheLen: Number(e.target.value),
                })}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>1</span>
                <span>16384</span>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                每次生成 <span className="text-orange-400 font-mono">1</span> 个新token，需要访问 <span className="text-orange-400 font-mono">{inferenceConfig.kvCacheLen}</span> 个历史KV
              </p>
            </div>
          )}

          {/* Data Type */}
          <div>
            <label className="text-sm text-gray-400 block mb-2">数据类型</label>
            <div className="grid grid-cols-5 gap-1">
              {(['fp32', 'fp16', 'bf16', 'int8', 'int4'] as const).map((dtype) => (
                <button
                  key={dtype}
                  onClick={() => onInferenceConfigChange({ ...inferenceConfig, dtype })}
                  className={`px-2 py-1.5 text-xs rounded-lg transition-all ${
                    inferenceConfig.dtype === dtype
                      ? 'bg-purple-600/50 text-purple-300 border border-purple-500/50'
                      : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                  }`}
                >
                  {dtype.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* 当前模型信息 */}
      <motion.div
        className="glass rounded-xl p-4"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Settings className="w-4 h-4 text-gray-400" />
          <h3 className="font-medium text-sm">模型详情</h3>
        </div>
        
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Hidden Size</span>
            <span className="font-mono">{selectedModel.hiddenSize}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Layers</span>
            <span className="font-mono">{selectedModel.numLayers}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Attention Heads</span>
            <span className="font-mono">{selectedModel.numAttentionHeads} / {selectedModel.numKVHeads} KV</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Head Dim</span>
            <span className="font-mono">{selectedModel.headDim}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Intermediate Size</span>
            <span className="font-mono">{selectedModel.intermediateSize.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Vocab Size</span>
            <span className="font-mono">{selectedModel.vocabSize.toLocaleString()}</span>
          </div>
          {selectedModel.ffnType === 'moe' && (
            <>
              <div className="flex justify-between">
                <span className="text-gray-500">Experts</span>
                <span className="font-mono">{selectedModel.numExperts}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Top-K Experts</span>
                <span className="font-mono">{selectedModel.numExpertsPerToken}</span>
              </div>
              {selectedModel.sharedExpertNum && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Shared Experts</span>
                  <span className="font-mono">{selectedModel.sharedExpertNum}</span>
                </div>
              )}
            </>
          )}
        </div>
      </motion.div>
    </div>
  );
}
