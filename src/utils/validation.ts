/**
 * Validation Utilities
 * 
 * Provides benchmark validation to compare calculator outputs against known values
 * from published benchmarks and real-world measurements.
 */

import { VALIDATION_BENCHMARKS, type ValidationBenchmark } from '../config/formulaConstants';
import type { TrainingStepAnalysis } from '../types/training';

export interface ValidationResult {
  benchmark: ValidationBenchmark;
  calculated: {
    memoryPerGPU_GB: number;
    mfu: number;
    tokensPerSecond?: number;
  };
  validation: {
    memoryInRange: boolean;
    mfuInRange: boolean;
    tokensInRange?: boolean;
    overallPass: boolean;
  };
  deltas: {
    memoryDeltaPercent: number;  // How far from expected range midpoint
    mfuDeltaPercent: number;
  };
}

/**
 * Validate calculated values against a benchmark
 */
export function validateAgainstBenchmark(
  benchmark: ValidationBenchmark,
  analysis: TrainingStepAnalysis
): ValidationResult {
  const memoryPerGPU_GB = analysis.memoryBreakdown.totalPerGPU / 1e9;
  const mfu = analysis.mfu;
  const tokensPerSecond = analysis.tokensPerSecond;
  
  // Check if values are in expected ranges
  const memoryInRange = 
    memoryPerGPU_GB >= benchmark.expected.memoryPerGPU_GB.min &&
    memoryPerGPU_GB <= benchmark.expected.memoryPerGPU_GB.max;
  
  const mfuInRange = 
    mfu >= benchmark.expected.mfu.min &&
    mfu <= benchmark.expected.mfu.max;
  
  let tokensInRange: boolean | undefined;
  if (benchmark.expected.tokensPerSecond && tokensPerSecond) {
    tokensInRange = 
      tokensPerSecond >= benchmark.expected.tokensPerSecond.min &&
      tokensPerSecond <= benchmark.expected.tokensPerSecond.max;
  }
  
  // Calculate deltas from midpoint
  const memoryMidpoint = (benchmark.expected.memoryPerGPU_GB.min + benchmark.expected.memoryPerGPU_GB.max) / 2;
  const mfuMidpoint = (benchmark.expected.mfu.min + benchmark.expected.mfu.max) / 2;
  
  const memoryDeltaPercent = ((memoryPerGPU_GB - memoryMidpoint) / memoryMidpoint) * 100;
  const mfuDeltaPercent = ((mfu - mfuMidpoint) / mfuMidpoint) * 100;
  
  const overallPass = memoryInRange && mfuInRange && (tokensInRange === undefined || tokensInRange);
  
  return {
    benchmark,
    calculated: {
      memoryPerGPU_GB,
      mfu,
      tokensPerSecond,
    },
    validation: {
      memoryInRange,
      mfuInRange,
      tokensInRange,
      overallPass,
    },
    deltas: {
      memoryDeltaPercent,
      mfuDeltaPercent,
    },
  };
}

/**
 * Run all validation benchmarks
 */
export function runAllValidations(
  analysisResults: Map<string, TrainingStepAnalysis>
): ValidationResult[] {
  const results: ValidationResult[] = [];
  
  for (const benchmark of VALIDATION_BENCHMARKS) {
    const analysis = analysisResults.get(benchmark.name);
    if (analysis) {
      results.push(validateAgainstBenchmark(benchmark, analysis));
    }
  }
  
  return results;
}

/**
 * Format validation result for display
 */
export function formatValidationResult(result: ValidationResult): string {
  const { benchmark, calculated, validation, deltas } = result;
  
  const memoryStatus = validation.memoryInRange ? '✓' : '✗';
  const mfuStatus = validation.mfuInRange ? '✓' : '✗';
  const overallStatus = validation.overallPass ? 'PASS' : 'FAIL';
  
  return `
${benchmark.name} [${overallStatus}]
  Memory: ${calculated.memoryPerGPU_GB.toFixed(1)} GB ${memoryStatus}
    Expected: ${benchmark.expected.memoryPerGPU_GB.min}-${benchmark.expected.memoryPerGPU_GB.max} GB
    Delta: ${deltas.memoryDeltaPercent > 0 ? '+' : ''}${deltas.memoryDeltaPercent.toFixed(1)}%
  MFU: ${(calculated.mfu * 100).toFixed(1)}% ${mfuStatus}
    Expected: ${(benchmark.expected.mfu.min * 100).toFixed(0)}-${(benchmark.expected.mfu.max * 100).toFixed(0)}%
    Delta: ${deltas.mfuDeltaPercent > 0 ? '+' : ''}${deltas.mfuDeltaPercent.toFixed(1)}%
  Source: ${benchmark.source}
`.trim();
}

/**
 * Get accuracy assessment based on validation results
 */
export function getAccuracyAssessment(results: ValidationResult[]): {
  totalBenchmarks: number;
  passed: number;
  failed: number;
  accuracyPercent: number;
  memoryAccuracyPercent: number;
  mfuAccuracyPercent: number;
  assessment: 'excellent' | 'good' | 'fair' | 'poor';
} {
  const total = results.length;
  const passed = results.filter(r => r.validation.overallPass).length;
  const memoryPassed = results.filter(r => r.validation.memoryInRange).length;
  const mfuPassed = results.filter(r => r.validation.mfuInRange).length;
  
  const accuracyPercent = (passed / total) * 100;
  const memoryAccuracyPercent = (memoryPassed / total) * 100;
  const mfuAccuracyPercent = (mfuPassed / total) * 100;
  
  let assessment: 'excellent' | 'good' | 'fair' | 'poor';
  if (accuracyPercent >= 90) {
    assessment = 'excellent';
  } else if (accuracyPercent >= 75) {
    assessment = 'good';
  } else if (accuracyPercent >= 50) {
    assessment = 'fair';
  } else {
    assessment = 'poor';
  }
  
  return {
    totalBenchmarks: total,
    passed,
    failed: total - passed,
    accuracyPercent,
    memoryAccuracyPercent,
    mfuAccuracyPercent,
    assessment,
  };
}

/**
 * User-provided profiling data for comparison
 */
export interface ProfilingData {
  name: string;
  source: 'nvidia-smi' | 'torch.profiler' | 'nsight' | 'manual';
  
  // Memory measurements
  peakMemoryGB?: number;
  allocatedMemoryGB?: number;
  
  // Performance measurements  
  timePerStepMs?: number;
  tokensPerSecond?: number;
  observedTFLOPS?: number;
  
  // Configuration
  gpuType: string;
  numGPUs: number;
  batchSize: number;
  seqLength: number;
}

/**
 * Compare calculator output with user-provided profiling data
 */
export function compareWithProfiling(
  analysis: TrainingStepAnalysis,
  profiling: ProfilingData
): {
  memoryDelta: { calculated: number; measured: number; deltaPercent: number } | null;
  timeDelta: { calculated: number; measured: number; deltaPercent: number } | null;
  throughputDelta: { calculated: number; measured: number; deltaPercent: number } | null;
  recommendations: string[];
} {
  const recommendations: string[] = [];
  
  let memoryDelta = null;
  if (profiling.peakMemoryGB) {
    const calculatedMemoryGB = analysis.memoryBreakdown.totalPerGPU / 1e9;
    const deltaPercent = ((calculatedMemoryGB - profiling.peakMemoryGB) / profiling.peakMemoryGB) * 100;
    memoryDelta = {
      calculated: calculatedMemoryGB,
      measured: profiling.peakMemoryGB,
      deltaPercent,
    };
    
    if (Math.abs(deltaPercent) > 20) {
      recommendations.push(
        `Memory estimate differs by ${deltaPercent.toFixed(0)}%. Consider adjusting frameworkOverheadRatio or activationFactor.`
      );
    }
  }
  
  let timeDelta = null;
  if (profiling.timePerStepMs) {
    const deltaPercent = ((analysis.timePerStep - profiling.timePerStepMs) / profiling.timePerStepMs) * 100;
    timeDelta = {
      calculated: analysis.timePerStep,
      measured: profiling.timePerStepMs,
      deltaPercent,
    };
    
    if (Math.abs(deltaPercent) > 20) {
      recommendations.push(
        `Time estimate differs by ${deltaPercent.toFixed(0)}%. Consider adjusting communicationOverlapFactor or kernelEfficiency.`
      );
    }
  }
  
  let throughputDelta = null;
  if (profiling.tokensPerSecond) {
    const deltaPercent = ((analysis.tokensPerSecond - profiling.tokensPerSecond) / profiling.tokensPerSecond) * 100;
    throughputDelta = {
      calculated: analysis.tokensPerSecond,
      measured: profiling.tokensPerSecond,
      deltaPercent,
    };
  }
  
  return {
    memoryDelta,
    timeDelta,
    throughputDelta,
    recommendations,
  };
}
