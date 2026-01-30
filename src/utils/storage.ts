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
