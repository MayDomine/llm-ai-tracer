# LLM AI Tracer

大模型算术密度可视化分析工具 - 基于 Roofline 模型的 LLM 性能瓶颈分析

## 功能特性

### 核心功能

- **FLOPs 计算**：精确估算模型各模块的浮点运算次数（验证对标 HuggingFace calflops、vLLM、SGLang）
- **内存访问分析**：分析每个操作的内存读写量
- **算术密度计算**：计算 Arithmetic Intensity (FLOP/Byte)
- **Roofline 模型**：基于硬件配置判断计算密集/内存密集

### GPU 内存建模

- **模型权重**：支持 FP32/FP16/BF16/INT8/INT4 量化
- **KV Cache**：推理时的键值缓存计算
- **激活内存**：前向传播中间张量
- **梯度和优化器状态**：训练时的额外内存开销
- **框架开销**：CUDA/PyTorch 运行时内存

### 通信建模

- **数据并行 (DP)**：AllReduce 梯度同步
- **张量并行 (TP)**：层内 AllReduce/AllGather
- **流水线并行 (PP)**：P2P 激活/梯度传递 + Bubble 开销
- **NVLink vs 网络带宽**：自动选择节点内/跨节点带宽

### 策略优化器

- **自动推荐**：基于内存和吞吐量约束推荐最优并行配置
- **策略比较**：对比所有有效的 DP×TP×PP 组合
- **可视化指标**：吞吐量、延迟、内存使用、通信效率

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

- NVIDIA H100 SXM / PCIe (80GB, 989 TFLOPS)
- NVIDIA H200 SXM (141GB, 989 TFLOPS)
- NVIDIA B200 (192GB, 2250 TFLOPS)
- NVIDIA A100 40G / 80G (312 TFLOPS)
- NVIDIA RTX 4090 / 3090
- NVIDIA L40S (48GB)
- AMD MI300X (192GB, 1307 TFLOPS)

所有硬件配置包含：
- 计算能力 (TFLOPS FP16)
- 内存带宽 (TB/s)
- 显存容量 (GB)
- NVLink 带宽 (GB/s)
- 网络带宽 (GB/s)

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
├── components/                    # React 组件
│   ├── ConfigPanel.tsx               # 模型/硬件/推理配置面板
│   ├── ModuleCard.tsx                # 模块卡片
│   ├── OperationList.tsx             # 操作列表
│   ├── RooflineChart.tsx             # Roofline 图表
│   ├── StatsPanel.tsx                # 统计面板
│   ├── MemoryPanel.tsx               # GPU 内存分析面板
│   ├── ParallelConfigPanel.tsx       # 并行策略配置面板
│   └── StrategyComparison.tsx        # 策略比较模态框
├── config/
│   └── presets.ts                 # 预设模型和硬件配置
├── types/
│   └── model.ts                   # TypeScript 类型定义
├── utils/
│   ├── calculator.ts              # FLOPs 计算 + 参数统计
│   ├── memoryCalculator.ts        # GPU 内存建模
│   ├── communicationCalculator.ts # 通信开销计算
│   └── strategyOptimizer.ts       # 并行策略优化器
├── App.tsx                        # 主应用组件
├── main.tsx                       # 入口文件
└── index.css                      # 全局样式

docs/
└── TFLOPS_CALCULATION.md          # FLOPs 计算公式详细文档
```

## 文档

详细的 TFLOPS 计算公式和验证文档请参阅 [docs/TFLOPS_CALCULATION.md](docs/TFLOPS_CALCULATION.md)，包含：

- 核心公式和符号定义
- 各组件（Attention、FFN、Embedding）的详细计算
- Training vs Inference 模式差异
- 内存访问和 Roofline 分析
- 与 HuggingFace/vLLM/SGLang 的对比验证
- 示例计算（LLaMA2-7B）

## License

MIT
