import { useMemo } from 'react';
import { motion } from 'framer-motion';
import type { OperationAnalysis, HardwareConfig } from '../types/model';

interface RooflineChartProps {
  operations: OperationAnalysis[];
  hardwareConfig: HardwareConfig;
}

export function RooflineChart({ operations, hardwareConfig }: RooflineChartProps) {
  const { computeCapability, memoryBandwidth } = hardwareConfig;
  
  // 计算 Roofline 参数
  const rooflinePoint = computeCapability / memoryBandwidth; // FLOP/Byte
  
  // 图表范围 (对数刻度)
  const minAI = 0.01;
  const maxAI = 1000;
  const maxPerf = computeCapability * 2; // 留出一些空间
  
  // 将算术密度转换为 x 坐标 (对数刻度)
  const aiToX = (ai: number) => {
    const logMin = Math.log10(minAI);
    const logMax = Math.log10(maxAI);
    const logAI = Math.log10(Math.max(ai, minAI));
    return ((logAI - logMin) / (logMax - logMin)) * 100;
  };
  
  // 将性能转换为 y 坐标 (对数刻度)
  const perfToY = (perf: number) => {
    const logMin = Math.log10(0.01);
    const logMax = Math.log10(maxPerf);
    const logPerf = Math.log10(Math.max(perf, 0.01));
    return 100 - ((logPerf - logMin) / (logMax - logMin)) * 100;
  };
  
  // 计算每个操作的理论性能 (TFLOPS)
  const operationPoints = useMemo(() => {
    return operations.map(op => {
      // 理论性能 = min(算力, 算术密度 * 带宽)
      const theoreticalPerf = Math.min(
        computeCapability,
        op.arithmeticIntensity * memoryBandwidth
      );
      
      return {
        ...op,
        x: aiToX(op.arithmeticIntensity),
        y: perfToY(theoreticalPerf),
        perf: theoreticalPerf,
      };
    });
  }, [operations, computeCapability, memoryBandwidth]);
  
  // Roofline 路径点
  const rooflinePath = useMemo(() => {
    const points: { x: number; y: number }[] = [];
    
    // 内存带宽限制线 (斜线部分)
    for (let ai = minAI; ai <= rooflinePoint; ai *= 1.5) {
      const perf = ai * memoryBandwidth;
      points.push({ x: aiToX(ai), y: perfToY(perf) });
    }
    
    // 拐点
    points.push({ x: aiToX(rooflinePoint), y: perfToY(computeCapability) });
    
    // 计算能力限制线 (水平部分)
    points.push({ x: aiToX(maxAI), y: perfToY(computeCapability) });
    
    return points;
  }, [rooflinePoint, computeCapability, memoryBandwidth]);
  
  // 生成 SVG 路径
  const rooflinePathD = rooflinePath
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ');

  // X 轴刻度
  const xTicks = [0.01, 0.1, 1, 10, 100, 1000];
  // Y 轴刻度
  const yTicks = [0.1, 1, 10, 100, 1000];

  return (
    <motion.div
      className="glass rounded-xl p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h3 className="font-semibold mb-4">Roofline 模型分析</h3>
      
      <div className="relative" style={{ paddingBottom: '50%' }}>
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="-10 -5 120 110"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* 背景网格 */}
          <defs>
            <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
              <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(55, 65, 81, 0.3)" strokeWidth="0.2" />
            </pattern>
          </defs>
          <rect x="0" y="0" width="100" height="100" fill="url(#grid)" />
          
          {/* 拐点垂直虚线 */}
          <line
            x1={aiToX(rooflinePoint)}
            y1="0"
            x2={aiToX(rooflinePoint)}
            y2="100"
            stroke="rgba(99, 102, 241, 0.3)"
            strokeWidth="0.5"
            strokeDasharray="2,2"
          />
          
          {/* Roofline 曲线 */}
          <path
            d={rooflinePathD}
            fill="none"
            stroke="url(#rooflineGradient)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          <defs>
            <linearGradient id="rooflineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
          
          {/* 区域标签 */}
          <text x="15" y="85" className="text-[3px] fill-memory-bound" fontFamily="sans-serif">
            Memory Bound
          </text>
          <text x="70" y="25" className="text-[3px] fill-compute-bound" fontFamily="sans-serif">
            Compute Bound
          </text>
          
          {/* 拐点标记 */}
          <circle
            cx={aiToX(rooflinePoint)}
            cy={perfToY(computeCapability)}
            r="2"
            fill="#6366f1"
          />
          <text
            x={aiToX(rooflinePoint) + 3}
            y={perfToY(computeCapability) - 3}
            className="text-[2.5px] fill-indigo-400"
            fontFamily="monospace"
          >
            {rooflinePoint.toFixed(1)} FLOP/B
          </text>
          
          {/* 操作点 */}
          {operationPoints.map((op, idx) => (
            <g key={op.id}>
              <motion.circle
                cx={op.x}
                cy={op.y}
                r="1.5"
                className={op.isComputeBound ? 'fill-compute-bound' : 'fill-memory-bound'}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: idx * 0.02 }}
              />
              {/* Tooltip area - 鼠标悬停显示 */}
              <title>{`${op.name}\nAI: ${op.arithmeticIntensity.toFixed(2)} FLOP/B\nPerf: ${op.perf.toFixed(1)} TFLOPS`}</title>
            </g>
          ))}
          
          {/* X 轴 */}
          <line x1="0" y1="100" x2="100" y2="100" stroke="#374151" strokeWidth="0.5" />
          {xTicks.map(tick => (
            <g key={tick}>
              <line
                x1={aiToX(tick)}
                y1="100"
                x2={aiToX(tick)}
                y2="102"
                stroke="#374151"
                strokeWidth="0.3"
              />
              <text
                x={aiToX(tick)}
                y="106"
                textAnchor="middle"
                className="text-[2.5px] fill-gray-500"
                fontFamily="monospace"
              >
                {tick}
              </text>
            </g>
          ))}
          <text x="50" y="112" textAnchor="middle" className="text-[3px] fill-gray-400" fontFamily="sans-serif">
            Arithmetic Intensity (FLOP/Byte)
          </text>
          
          {/* Y 轴 */}
          <line x1="0" y1="0" x2="0" y2="100" stroke="#374151" strokeWidth="0.5" />
          {yTicks.filter(t => t < maxPerf).map(tick => (
            <g key={tick}>
              <line
                x1="-2"
                y1={perfToY(tick)}
                x2="0"
                y2={perfToY(tick)}
                stroke="#374151"
                strokeWidth="0.3"
              />
              <text
                x="-4"
                y={perfToY(tick) + 1}
                textAnchor="end"
                className="text-[2.5px] fill-gray-500"
                fontFamily="monospace"
              >
                {tick}
              </text>
            </g>
          ))}
          <text
            x="-8"
            y="50"
            textAnchor="middle"
            transform="rotate(-90, -8, 50)"
            className="text-[3px] fill-gray-400"
            fontFamily="sans-serif"
          >
            Performance (TFLOPS)
          </text>
        </svg>
      </div>
      
      {/* 图例 */}
      <div className="flex flex-wrap items-center gap-6 mt-4 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-8 h-1 bg-gradient-to-r from-red-500 via-amber-500 to-green-500 rounded" />
          <span className="text-gray-400">Roofline ({computeCapability} TFLOPS / {memoryBandwidth} TB/s)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-compute-bound" />
          <span className="text-gray-400">计算密集操作</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-memory-bound" />
          <span className="text-gray-400">内存密集操作</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-indigo-500" />
          <span className="text-gray-400">Roofline 拐点</span>
        </div>
      </div>
    </motion.div>
  );
}
