/**
 * 本地存储工具 - 用于保存用户自定义模型配置
 */

import type { ModelConfig } from '../types/model';

const STORAGE_KEY = 'llm-ai-tracer-custom-models';

/**
 * 获取所有自定义模型配置
 */
export function getCustomModels(): ModelConfig[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    return JSON.parse(data);
  } catch {
    return [];
  }
}

/**
 * 保存自定义模型配置
 */
export function saveCustomModel(model: ModelConfig): void {
  const models = getCustomModels();
  // 检查是否已存在同名模型，存在则更新
  const existingIndex = models.findIndex(m => m.name === model.name);
  if (existingIndex >= 0) {
    models[existingIndex] = model;
  } else {
    models.push(model);
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(models));
}

/**
 * 删除自定义模型配置
 */
export function deleteCustomModel(name: string): void {
  const models = getCustomModels().filter(m => m.name !== name);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(models));
}

/**
 * 导出所有自定义模型配置为 JSON
 */
export function exportCustomModels(): string {
  const models = getCustomModels();
  return JSON.stringify(models, null, 2);
}

/**
 * 从 JSON 导入模型配置
 */
export function importCustomModels(jsonString: string): { success: boolean; count: number; error?: string } {
  try {
    const data = JSON.parse(jsonString);
    
    // 支持单个模型或模型数组
    const models: ModelConfig[] = Array.isArray(data) ? data : [data];
    
    // 验证每个模型的必要字段
    for (const model of models) {
      if (!validateModelConfig(model)) {
        return { success: false, count: 0, error: '模型配置格式不正确，缺少必要字段' };
      }
    }
    
    // 保存所有模型
    models.forEach(model => saveCustomModel(model));
    
    return { success: true, count: models.length };
  } catch {
    return { success: false, count: 0, error: 'JSON 解析失败，请检查格式' };
  }
}

/**
 * 验证模型配置是否有效
 */
export function validateModelConfig(config: unknown): config is ModelConfig {
  if (!config || typeof config !== 'object') return false;
  
  const c = config as Record<string, unknown>;
  
  // 检查必要字段
  const requiredFields = [
    'name',
    'hiddenSize',
    'numLayers',
    'vocabSize',
    'numAttentionHeads',
    'numKVHeads',
    'headDim',
    'attentionType',
    'intermediateSize',
    'ffnType',
    'maxSeqLen',
  ];
  
  for (const field of requiredFields) {
    if (!(field in c)) return false;
  }
  
  // 检查类型
  if (typeof c.name !== 'string') return false;
  if (typeof c.hiddenSize !== 'number') return false;
  if (typeof c.numLayers !== 'number') return false;
  if (!['mha', 'gqa'].includes(c.attentionType as string)) return false;
  if (!['gpt', 'gated', 'moe'].includes(c.ffnType as string)) return false;
  
  return true;
}

/**
 * 创建空白模型配置模板
 */
export function createModelTemplate(): ModelConfig {
  return {
    name: 'Custom Model',
    hiddenSize: 4096,
    numLayers: 32,
    vocabSize: 32000,
    numAttentionHeads: 32,
    numKVHeads: 8,
    headDim: 128,
    attentionType: 'gqa',
    intermediateSize: 14336,
    ffnType: 'gated',
    maxSeqLen: 8192,
  };
}

// ============ HuggingFace Config Support ============

/**
 * HuggingFace config.json 的常见字段接口
 */
export interface HuggingFaceConfig {
  // 基础字段
  model_type?: string;
  architectures?: string[];
  _name_or_path?: string;
  
  // 模型维度
  hidden_size?: number;
  num_hidden_layers?: number;
  vocab_size?: number;
  
  // Attention
  num_attention_heads?: number;
  num_key_value_heads?: number;  // GQA
  head_dim?: number;
  
  // FFN
  intermediate_size?: number;
  
  // 序列长度
  max_position_embeddings?: number;
  max_sequence_length?: number;
  
  // MoE 相关
  num_local_experts?: number;
  num_experts_per_tok?: number;
  num_experts?: number;  // Mixtral 格式
  num_selected_experts?: number;  // 某些格式
  
  // Qwen MoE 特有
  moe_intermediate_size?: number;
  shared_expert_intermediate_size?: number;
  num_shared_experts?: number;
  
  // DeepSeek MoE 特有
  n_routed_experts?: number;
  num_experts_per_token?: number;
  n_shared_experts?: number;
  moe_layer_freq?: number;
  first_k_dense_replace?: number;
  
  // 可能的别名字段
  n_embd?: number;  // GPT-2 style
  n_layer?: number;
  n_head?: number;
  n_head_kv?: number;  // 一些模型用这个
  d_model?: number;  // T5/BERT style
  d_ff?: number;
  n_positions?: number;
  
  // 模型名称相关
  name?: string;
  model_name?: string;
}

/**
 * 从 HuggingFace config.json 解析模型名称
 */
function extractModelName(config: HuggingFaceConfig): string {
  // 优先级：_name_or_path > name/model_name > architectures > model_type
  if (config._name_or_path) {
    // 提取路径最后一部分作为名称
    const parts = config._name_or_path.split('/').filter(Boolean);
    return parts[parts.length - 1] || 'HF Model';
  }

  if (config.name?.trim()) {
    return config.name.trim();
  }

  if (config.model_name?.trim()) {
    return config.model_name.trim();
  }
  
  if (config.architectures && config.architectures.length > 0) {
    // 从架构名称提取，如 "LlamaForCausalLM" -> "Llama"
    const arch = config.architectures[0];
    return arch.replace(/ForCausalLM|ForConditionalGeneration|Model$/g, '') || 'HF Model';
  }
  
  if (config.model_type) {
    // 首字母大写
    return config.model_type.charAt(0).toUpperCase() + config.model_type.slice(1);
  }
  
  return 'HuggingFace Model';
}

/**
 * 从用户输入中提取 HuggingFace model ID
 * 支持:
 * - org/model
 * - https://huggingface.co/org/model
 * - https://huggingface.co/org/model/blob/main/config.json
 * - https://huggingface.co/org/model/raw/main/config.json
 */
function normalizeHuggingFaceModelId(input: string): string | null {
  const trimmed = input.trim().replace(/\/+$/, '');
  if (!trimmed) return null;

  if (/^https?:\/\//i.test(trimmed)) {
    try {
      const url = new URL(trimmed);
      const host = url.hostname.toLowerCase();
      const isHuggingFaceHost =
        host === 'huggingface.co' ||
        host === 'www.huggingface.co' ||
        host === 'hf.co';

      if (!isHuggingFaceHost) {
        return null;
      }

      const [owner, repo] = url.pathname.split('/').filter(Boolean);
      if (!owner || !repo) {
        return null;
      }

      return `${owner}/${repo}`;
    } catch {
      return null;
    }
  }

  return /^[^/\s]+\/[^/\s]+$/.test(trimmed) ? trimmed : null;
}

/**
 * 检测是否为 MoE 模型
 */
function detectMoE(config: HuggingFaceConfig): { isMoE: boolean; numExperts?: number; numExpertsPerToken?: number; sharedExperts?: number } {
  // 检查各种 MoE 相关字段
  const numExperts = config.num_local_experts
    ?? config.num_experts
    ?? config.n_routed_experts;
  
  const numExpertsPerToken = config.num_experts_per_tok
    ?? config.num_selected_experts
    ?? config.num_experts_per_token
    ?? 2;  // 默认 top-2
  
  const sharedExperts = config.num_shared_experts ?? config.n_shared_experts;
  
  if (numExperts && numExperts > 1) {
    return {
      isMoE: true,
      numExperts,
      numExpertsPerToken,
      sharedExperts,
    };
  }
  
  return { isMoE: false };
}

/**
 * 检测 Attention 类型 (MHA vs GQA)
 */
function detectAttentionType(config: HuggingFaceConfig): { type: 'mha' | 'gqa'; numKVHeads: number } {
  const numHeads = config.num_attention_heads ?? config.n_head ?? 32;
  const numKVHeads = config.num_key_value_heads ?? config.n_head_kv;
  
  if (numKVHeads !== undefined && numKVHeads < numHeads) {
    return { type: 'gqa', numKVHeads };
  }
  
  return { type: 'mha', numKVHeads: numHeads };
}

/**
 * 检测 FFN 类型
 * - GPT style: 2 Linear layers (up, down)
 * - Gated (LLaMA/Qwen): 3 Linear layers (gate, up, down) with SiLU
 * - MoE: Mixture of Experts
 */
function detectFFNType(config: HuggingFaceConfig): 'gpt' | 'gated' | 'moe' {
  const moeInfo = detectMoE(config);
  if (moeInfo.isMoE) {
    return 'moe';
  }
  
  // 基于 model_type 或 architecture 推断
  const modelType = config.model_type?.toLowerCase() || '';
  const arch = config.architectures?.[0]?.toLowerCase() || '';
  
  // GPT-2/GPT-Neo 等使用 2 Linear FFN
  if (modelType.includes('gpt2') || modelType.includes('gpt_neo') || arch.includes('gpt2')) {
    return 'gpt';
  }
  
  // LLaMA, Mistral, Qwen, Phi 等使用 Gated FFN
  if (modelType.includes('llama') || modelType.includes('mistral') || 
      modelType.includes('qwen') || modelType.includes('phi') ||
      modelType.includes('gemma') || modelType.includes('deepseek')) {
    return 'gated';
  }
  
  // 默认为 gated (现代模型更常用)
  return 'gated';
}

/**
 * 解析 HuggingFace config.json 并转换为 ModelConfig
 */
export function parseHuggingFaceConfig(config: HuggingFaceConfig, customName?: string): ModelConfig {
  // 提取基础维度 (支持多种命名风格)
  // 检测 FFN 类型和 MoE
  const ffnType = detectFFNType(config);
  const moeInfo = detectMoE(config);

  const hiddenSize = config.hidden_size ?? config.n_embd ?? config.d_model ?? 4096;
  const numLayers = config.num_hidden_layers ?? config.n_layer ?? 32;
  const vocabSize = config.vocab_size ?? 32000;
  const numHeads = config.num_attention_heads ?? config.n_head ?? 32;
  const denseIntermediateSize = config.intermediate_size ?? config.d_ff ?? (hiddenSize * 4);
  const intermediateSize = ffnType === 'moe'
    ? (config.moe_intermediate_size ?? denseIntermediateSize)
    : denseIntermediateSize;
  const maxSeqLen = config.max_position_embeddings ?? config.max_sequence_length ?? config.n_positions ?? 4096;
  
  // 检测 Attention 类型
  const attention = detectAttentionType(config);
  
  // 计算 head_dim
  const headDim = config.head_dim ?? Math.floor(hiddenSize / numHeads);
  
  // 构建 ModelConfig
  const modelConfig: ModelConfig = {
    name: customName || extractModelName(config),
    hiddenSize,
    numLayers,
    vocabSize,
    numAttentionHeads: numHeads,
    numKVHeads: attention.numKVHeads,
    headDim,
    attentionType: attention.type,
    intermediateSize,
    ffnType,
    maxSeqLen,
  };
  
  // 添加 MoE 参数
  if (moeInfo.isMoE) {
    const densePrefixLayers = Math.max(0, config.first_k_dense_replace ?? 0);
    const moeLayerFreq = config.moe_layer_freq ?? 1;
    const remainingLayers = Math.max(0, numLayers - densePrefixLayers);
    const numMoeLayers = moeLayerFreq > 1 ? Math.ceil(remainingLayers / moeLayerFreq) : remainingLayers;
    const numDenseLayers = Math.max(0, numLayers - numMoeLayers);

    modelConfig.numExperts = moeInfo.numExperts;
    modelConfig.numExpertsPerToken = moeInfo.numExpertsPerToken;
    modelConfig.denseIntermediateSize = denseIntermediateSize;
    modelConfig.numDenseLayers = numDenseLayers;
    if (moeInfo.sharedExperts) {
      modelConfig.sharedExpertNum = moeInfo.sharedExperts;
    }
    const sharedIntermediateFromConfig = config.shared_expert_intermediate_size;
    if (sharedIntermediateFromConfig != null) {
      modelConfig.sharedExpertIntermediateSize = sharedIntermediateFromConfig;
    } else if (moeInfo.sharedExperts) {
      modelConfig.sharedExpertIntermediateSize = moeInfo.sharedExperts * intermediateSize;
    }
  }
  
  return modelConfig;
}

/**
 * 验证是否为有效的 HuggingFace config
 */
export function isHuggingFaceConfig(config: unknown): config is HuggingFaceConfig {
  if (!config || typeof config !== 'object') return false;
  
  const c = config as Record<string, unknown>;
  
  // HuggingFace config 通常有这些字段之一
  const hfIndicators = [
    'hidden_size',
    'num_hidden_layers', 
    'num_attention_heads',
    'model_type',
    'architectures',
    'n_embd',  // GPT-2 style
    'd_model', // T5/BERT style
  ];
  
  return hfIndicators.some(field => field in c);
}

/**
 * 从 HuggingFace config.json 导入模型
 */
export function importHuggingFaceConfig(configInput: string | HuggingFaceConfig, customName?: string): {
  success: boolean; 
  model?: ModelConfig; 
  error?: string;
  warnings?: string[];
} {
  try {
    const data = typeof configInput === 'string' ? JSON.parse(configInput) : configInput;
    
    if (!isHuggingFaceConfig(data)) {
      return { 
        success: false, 
        error: '无法识别为 HuggingFace config.json 格式，请检查文件内容' 
      };
    }
    
    const warnings: string[] = [];
    
    // 检查必要字段并警告缺失
    if (data.hidden_size == null && data.n_embd == null && data.d_model == null) {
      warnings.push('未找到 hidden_size，使用默认值 4096');
    }
    if (data.num_hidden_layers == null && data.n_layer == null) {
      warnings.push('未找到 num_hidden_layers，使用默认值 32');
    }
    if (detectMoE(data).isMoE && data.moe_intermediate_size == null && data.intermediate_size != null) {
      warnings.push('MoE config 未找到 moe_intermediate_size，已回退使用 intermediate_size');
    }
    
    const model = parseHuggingFaceConfig(data, customName);
    
    // 保存模型
    saveCustomModel(model);
    
    return { success: true, model, warnings: warnings.length > 0 ? warnings : undefined };
  } catch (e) {
    const errorPrefix = typeof configInput === 'string' ? 'JSON 解析失败' : '导入失败';
    return { 
      success: false, 
      error: `${errorPrefix}: ${e instanceof Error ? e.message : '未知错误'}`
    };
  }
}

/**
 * 从 HuggingFace Hub URL 获取 config.json
 */
export async function fetchHuggingFaceConfig(modelIdOrUrl: string): Promise<{
  success: boolean;
  config?: HuggingFaceConfig;
  modelId?: string;
  error?: string;
}> {
  try {
    const normalizedModelId = normalizeHuggingFaceModelId(modelIdOrUrl);
    const url = normalizedModelId
      ? `https://huggingface.co/${normalizedModelId}/raw/main/config.json`
      : modelIdOrUrl.trim();
    
    const response = await fetch(url);
    
    if (!response.ok) {
      if (response.status === 404) {
        return { success: false, error: '找不到该模型或 config.json 文件' };
      }
      return { success: false, error: `请求失败: ${response.status} ${response.statusText}` };
    }
    
    const config = await response.json();
    return { success: true, config, modelId: normalizedModelId ?? undefined };
  } catch (e) {
    return { 
      success: false, 
      error: `网络请求失败: ${e instanceof Error ? e.message : '未知错误'}` 
    };
  }
}
