# LLM AI Tracer

大模型算术密度可视化分析工具 - 基于 Roofline 模型的 LLM 性能瓶颈分析

## 功能特性

### 核心功能

- **FLOPs 计算**：精确估算模型各模块的浮点运算次数
- **内存访问分析**：分析每个操作的内存读写量
- **算术密度计算**：计算 Arithmetic Intensity (FLOP/Byte)
- **Roofline 模型**：基于硬件配置判断计算密集/内存密集

### 模型结构分析

- **Embedding 层**：Token Embedding 查表操作
- **Transformer Block**：
  - **Attention**：支持 MHA (Multi-Head Attention) 和 GQA (Grouped Query Attention)
  - **FFN**：支持 GPT (2 Linear) / Gated (3 Linear, LLaMA风格) / MoE (混合专家)
- **LM Head**：最终输出投影层

### 预设模型

- **Qwen3 系列**：0.6B, 1.7B, 4B, 8B, 14B, 32B
- **Qwen3 MoE**：30B-A3B, 235B-A22B
- **LLaMA 2**：7B, 13B, 70B
- **LLaMA 3**：8B, 70B, 405B
- **MiniCPM**：2B, 3B

### 预设硬件

- NVIDIA H100 SXM / PCIe
- NVIDIA A100 40G / 80G
- NVIDIA RTX 4090 / 3090
- NVIDIA L40S
- AMD MI300X

## 技术栈

- **前端框架**：React 18 + TypeScript
- **构建工具**：Vite 6
- **样式**：TailwindCSS 3
- **动画**：Framer Motion
- **图标**：Lucide React

## 快速开始

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

## 使用说明

1. **选择模型**：从左侧面板选择预设模型或查看模型详细参数
2. **配置硬件**：选择目标 GPU 或自定义计算能力和内存带宽
3. **调整推理参数**：设置 Batch Size、序列长度、数据类型
4. **查看分析结果**：
   - 上方统计面板显示总体 FLOPs、内存访问、算术密度
   - 性能瓶颈分析显示计算密集/内存密集操作比例
   - 展开各模块查看每个操作的详细分析

## Roofline 模型说明

Roofline 模型是一种直观的性能分析方法：

- **算术密度 (AI)** = FLOPs / Memory Bytes
- **Roofline 拐点** = 计算能力 / 内存带宽
- 当 AI ≥ 拐点时，操作为**计算密集** (绿色)
- 当 AI < 拐点时，操作为**内存密集** (红色)

## 优化建议

### 内存密集场景
- 增大 Batch Size 提高算术密度
- 使用量化 (INT8/INT4) 减少内存访问
- 使用 FlashAttention 等算子融合技术

### 计算密集场景
- 使用更高算力的 GPU
- 使用算子融合减少 kernel 启动开销
- 考虑张量并行分散计算

## 项目结构

```
src/
├── components/          # React 组件
│   ├── ConfigPanel.tsx     # 配置面板
│   ├── ModuleCard.tsx      # 模块卡片
│   ├── OperationList.tsx   # 操作列表
│   ├── RooflineChart.tsx   # Roofline 图表
│   └── StatsPanel.tsx      # 统计面板
├── config/
│   └── presets.ts       # 预设模型和硬件配置
├── types/
│   └── model.ts         # TypeScript 类型定义
├── utils/
│   └── calculator.ts    # FLOPs 和内存计算逻辑
├── App.tsx              # 主应用组件
├── main.tsx             # 入口文件
└── index.css            # 全局样式
```

## License

MIT
