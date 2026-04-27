import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Upload, Download, Plus, Trash2, Check, AlertCircle, Globe, FileJson, Loader2 } from 'lucide-react';
import type { ModelConfig } from '../types/model';
import { 
  getCustomModels, 
  saveCustomModel, 
  deleteCustomModel, 
  importCustomModels,
  createModelTemplate,
  importHuggingFaceConfig,
  fetchHuggingFaceConfig,
} from '../utils/storage';

interface ModelImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  onModelImported: () => void;
}

type TabType = 'huggingface' | 'import' | 'create' | 'manage';

export function ModelImportModal({ isOpen, onClose, onModelImported }: ModelImportModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('huggingface');
  const [jsonInput, setJsonInput] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'warning'; text: string } | null>(null);
  const [customModels, setCustomModels] = useState<ModelConfig[]>([]);
  const [editingModel, setEditingModel] = useState<ModelConfig>(createModelTemplate());
  
  // HuggingFace tab state
  const [hfInput, setHfInput] = useState('');
  const [hfLoading, setHfLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [parsedConfig, setParsedConfig] = useState<ModelConfig | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setCustomModels(getCustomModels());
      setMessage(null);
      setParsedConfig(null);
      setHfInput('');
    }
  }, [isOpen]);
  
  // File drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, []);
  
  const handleFileUpload = (file: File) => {
    if (!file.name.endsWith('.json')) {
      setMessage({ type: 'error', text: '请上传 JSON 文件' });
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      processHuggingFaceConfig(content);
    };
    reader.onerror = () => {
      setMessage({ type: 'error', text: '文件读取失败' });
    };
    reader.readAsText(file);
  };
  
  const processHuggingFaceConfig = (jsonContent: string, customName?: string) => {
    const result = importHuggingFaceConfig(jsonContent, customName);
    
    if (result.success && result.model) {
      setParsedConfig(result.model);
      if (result.warnings && result.warnings.length > 0) {
        setMessage({ type: 'warning', text: result.warnings.join('; ') });
      } else {
        setMessage({ type: 'success', text: `已解析模型: ${result.model.name}` });
      }
      setCustomModels(getCustomModels());
      onModelImported();
    } else {
      setMessage({ type: 'error', text: result.error || '导入失败' });
      setParsedConfig(null);
    }
  };
  
  const handleHfUrlImport = async () => {
    if (!hfInput.trim()) {
      setMessage({ type: 'error', text: '请输入模型 ID 或 URL' });
      return;
    }
    
    setHfLoading(true);
    setMessage(null);
    
    try {
      const result = await fetchHuggingFaceConfig(hfInput.trim());
      
      if (result.success && result.config) {
        const imported = importHuggingFaceConfig(result.config, result.modelId);

        if (imported.success && imported.model) {
          setParsedConfig(imported.model);
          if (imported.warnings && imported.warnings.length > 0) {
            setMessage({ type: 'warning', text: imported.warnings.join('; ') });
          } else {
            setMessage({ type: 'success', text: `已导入模型: ${imported.model.name}` });
          }
          setCustomModels(getCustomModels());
          onModelImported();
        } else {
          setMessage({ type: 'error', text: imported.error || '导入失败' });
        }
      } else {
        setMessage({ type: 'error', text: result.error || '导入失败' });
      }
    } catch {
      setMessage({ type: 'error', text: '网络请求失败，请检查网络连接' });
    } finally {
      setHfLoading(false);
    }
  };
  
  const handlePasteConfig = () => {
    if (!hfInput.trim()) {
      setMessage({ type: 'error', text: '请粘贴 config.json 内容' });
      return;
    }
    processHuggingFaceConfig(hfInput);
  };

  const handleImport = () => {
    if (!jsonInput.trim()) {
      setMessage({ type: 'error', text: '请输入 JSON 配置' });
      return;
    }

    const result = importCustomModels(jsonInput);
    if (result.success) {
      setMessage({ type: 'success', text: `成功导入 ${result.count} 个模型配置` });
      setJsonInput('');
      setCustomModels(getCustomModels());
      onModelImported();
    } else {
      setMessage({ type: 'error', text: result.error || '导入失败' });
    }
  };

  const handleCreateModel = () => {
    if (!editingModel.name.trim()) {
      setMessage({ type: 'error', text: '请输入模型名称' });
      return;
    }

    saveCustomModel(editingModel);
    setMessage({ type: 'success', text: `模型 "${editingModel.name}" 已保存` });
    setCustomModels(getCustomModels());
    setEditingModel(createModelTemplate());
    onModelImported();
  };

  const handleDeleteModel = (name: string) => {
    deleteCustomModel(name);
    setCustomModels(getCustomModels());
    onModelImported();
    setMessage({ type: 'success', text: `模型 "${name}" 已删除` });
  };

  const handleExportAll = () => {
    const models = getCustomModels();
    const json = JSON.stringify(models, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'custom-models.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="w-full max-w-2xl bg-gray-900 rounded-2xl border border-gray-700 shadow-2xl overflow-hidden"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
            <h2 className="text-lg font-semibold">自定义模型配置</h2>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-gray-700">
            {[
              { id: 'huggingface', label: 'HuggingFace', icon: Globe },
              { id: 'import', label: '导入 JSON', icon: Upload },
              { id: 'create', label: '手动创建', icon: Plus },
              { id: 'manage', label: '管理配置', icon: Trash2 },
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => { setActiveTab(tab.id as TabType); setMessage(null); setParsedConfig(null); }}
                  className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'text-indigo-400 border-b-2 border-indigo-400 bg-indigo-500/10'
                      : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Content */}
          <div className="p-6 max-h-[60vh] overflow-y-auto">
            {/* Message */}
            {message && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${
                  message.type === 'success'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : message.type === 'warning'
                    ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                }`}
              >
                {message.type === 'success' ? <Check className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                {message.text}
              </motion.div>
            )}

            {/* HuggingFace Tab */}
            {activeTab === 'huggingface' && (
              <div className="space-y-6">
                {/* File Drop Zone */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative p-8 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-indigo-500 bg-indigo-500/10'
                      : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/30'
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                    className="hidden"
                  />
                  <FileJson className={`w-12 h-12 mx-auto mb-3 ${isDragging ? 'text-indigo-400' : 'text-gray-500'}`} />
                  <p className="text-sm text-gray-400">
                    拖拽 <code className="px-1.5 py-0.5 bg-gray-800 rounded text-indigo-400">config.json</code> 到这里
                  </p>
                  <p className="text-xs text-gray-500 mt-1">或点击选择文件</p>
                </div>
                
                {/* Divider */}
                <div className="flex items-center gap-4">
                  <div className="flex-1 h-px bg-gray-700" />
                  <span className="text-xs text-gray-500">或</span>
                  <div className="flex-1 h-px bg-gray-700" />
                </div>
                
                {/* URL / Paste Input */}
                <div className="space-y-3">
                  <p className="text-sm text-gray-400">
                    输入 HuggingFace 模型 ID 或粘贴 config.json 内容
                  </p>
                  <textarea
                    value={hfInput}
                    onChange={e => setHfInput(e.target.value)}
                    placeholder={`示例 (模型 ID):
meta-llama/Llama-3.1-8B
Qwen/Qwen2.5-7B
mistralai/Mixtral-8x7B-v0.1

或直接粘贴 config.json 内容:
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  ...
}`}
                    className="w-full h-40 p-4 bg-gray-800 border border-gray-700 rounded-lg font-mono text-sm resize-none focus:outline-none focus:border-indigo-500"
                  />
                  
                  <div className="flex gap-3">
                    <button
                      onClick={handleHfUrlImport}
                      disabled={hfLoading}
                      className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      {hfLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Globe className="w-4 h-4" />
                      )}
                      从 HuggingFace 导入
                    </button>
                    <button
                      onClick={handlePasteConfig}
                      className="flex-1 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <FileJson className="w-4 h-4" />
                      解析 JSON
                    </button>
                  </div>
                </div>
                
                {/* Parsed Config Preview */}
                {parsedConfig && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-gray-800/50 rounded-lg border border-gray-700"
                  >
                    <h4 className="font-medium text-green-400 mb-3 flex items-center gap-2">
                      <Check className="w-4 h-4" />
                      已导入: {parsedConfig.name}
                    </h4>
                    <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Layers</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.numLayers}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Hidden Size</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.hiddenSize.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Attention</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.attentionType.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">FFN</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.ffnType.toUpperCase()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Heads (Q/KV)</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.numAttentionHeads}/{parsedConfig.numKVHeads}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Vocab Size</span>
                        <span className="text-gray-300 font-mono">{parsedConfig.vocabSize.toLocaleString()}</span>
                      </div>
                      {parsedConfig.ffnType === 'moe' && (
                        <>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Experts</span>
                            <span className="text-gray-300 font-mono">{parsedConfig.numExperts}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-500">Top-K</span>
                            <span className="text-gray-300 font-mono">{parsedConfig.numExpertsPerToken}</span>
                          </div>
                        </>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            )}

            {/* Import Tab */}
            {activeTab === 'import' && (
              <div className="space-y-4">
                <p className="text-sm text-gray-400">
                  粘贴模型配置的 JSON 格式，支持单个模型或模型数组。
                </p>
                <textarea
                  value={jsonInput}
                  onChange={e => setJsonInput(e.target.value)}
                  placeholder={`{
  "name": "My Custom Model",
  "hiddenSize": 4096,
  "numLayers": 32,
  "vocabSize": 32000,
  "numAttentionHeads": 32,
  "numKVHeads": 8,
  "headDim": 128,
  "attentionType": "gqa",
  "intermediateSize": 14336,
  "ffnType": "gated",
  "maxSeqLen": 8192
}`}
                  className="w-full h-64 p-4 bg-gray-800 border border-gray-700 rounded-lg font-mono text-sm resize-none focus:outline-none focus:border-indigo-500"
                />
                <button
                  onClick={handleImport}
                  className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <Upload className="w-4 h-4" />
                  导入配置
                </button>
              </div>
            )}

            {/* Create Tab */}
            {activeTab === 'create' && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="col-span-2">
                    <label className="block text-sm text-gray-400 mb-1">模型名称 *</label>
                    <input
                      type="text"
                      value={editingModel.name}
                      onChange={e => setEditingModel({ ...editingModel, name: e.target.value })}
                      className="w-full"
                      placeholder="My Custom Model"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Hidden Size</label>
                    <input
                      type="number"
                      value={editingModel.hiddenSize}
                      onChange={e => setEditingModel({ ...editingModel, hiddenSize: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Layers</label>
                    <input
                      type="number"
                      value={editingModel.numLayers}
                      onChange={e => setEditingModel({ ...editingModel, numLayers: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Vocab Size</label>
                    <input
                      type="number"
                      value={editingModel.vocabSize}
                      onChange={e => setEditingModel({ ...editingModel, vocabSize: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Attention Heads</label>
                    <input
                      type="number"
                      value={editingModel.numAttentionHeads}
                      onChange={e => setEditingModel({ ...editingModel, numAttentionHeads: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">KV Heads</label>
                    <input
                      type="number"
                      value={editingModel.numKVHeads}
                      onChange={e => setEditingModel({ ...editingModel, numKVHeads: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Head Dim</label>
                    <input
                      type="number"
                      value={editingModel.headDim}
                      onChange={e => setEditingModel({ ...editingModel, headDim: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Intermediate Size</label>
                    <input
                      type="number"
                      value={editingModel.intermediateSize}
                      onChange={e => setEditingModel({ ...editingModel, intermediateSize: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Max Seq Length</label>
                    <input
                      type="number"
                      value={editingModel.maxSeqLen}
                      onChange={e => setEditingModel({ ...editingModel, maxSeqLen: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Attention Type</label>
                    <select
                      value={editingModel.attentionType}
                      onChange={e => setEditingModel({ ...editingModel, attentionType: e.target.value as 'mha' | 'gqa' })}
                      className="w-full"
                    >
                      <option value="mha">MHA (Multi-Head)</option>
                      <option value="gqa">GQA (Grouped Query)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-1">FFN Type</label>
                    <select
                      value={editingModel.ffnType}
                      onChange={e => setEditingModel({ ...editingModel, ffnType: e.target.value as 'gpt' | 'gated' | 'moe' })}
                      className="w-full"
                    >
                      <option value="gpt">GPT (2 Linear)</option>
                      <option value="gated">Gated (3 Linear, LLaMA)</option>
                      <option value="moe">MoE</option>
                    </select>
                  </div>

                  {editingModel.ffnType === 'moe' && (
                    <>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Experts 数量</label>
                        <input
                          type="number"
                          value={editingModel.numExperts || 8}
                          onChange={e => setEditingModel({ ...editingModel, numExperts: Number(e.target.value) })}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-sm text-gray-400 mb-1">Top-K Experts</label>
                        <input
                          type="number"
                          value={editingModel.numExpertsPerToken || 2}
                          onChange={e => setEditingModel({ ...editingModel, numExpertsPerToken: Number(e.target.value) })}
                          className="w-full"
                        />
                      </div>
                    </>
                  )}
                </div>

                <button
                  onClick={handleCreateModel}
                  className="w-full py-3 bg-green-600 hover:bg-green-500 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  保存模型配置
                </button>
              </div>
            )}

            {/* Manage Tab */}
            {activeTab === 'manage' && (
              <div className="space-y-4">
                {customModels.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <p>暂无自定义模型配置</p>
                    <p className="text-sm mt-1">通过"导入 JSON"或"手动创建"添加模型</p>
                  </div>
                ) : (
                  <>
                    <div className="flex justify-end">
                      <button
                        onClick={handleExportAll}
                        className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        导出全部
                      </button>
                    </div>
                    
                    <div className="space-y-2">
                      {customModels.map(model => (
                        <div
                          key={model.name}
                          className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg border border-gray-700"
                        >
                          <div>
                            <div className="font-medium">{model.name}</div>
                            <div className="text-xs text-gray-500 mt-1">
                              {model.numLayers}L / {model.hiddenSize}d / {model.attentionType.toUpperCase()} / {model.ffnType.toUpperCase()}
                              {model.ffnType === 'moe' && ` / ${model.numExperts}E`}
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteModel(model.name)}
                            className="p-2 text-red-400 hover:bg-red-500/20 rounded-lg transition-colors"
                            title="删除"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
