"""
Genetic Optimizer Example - 遗传算法优化ADC
===========================================

演示如何使用quantiamagica.optim优化ADC参数：
1. SDCoeffOptimizer - SD ADC系数一键优化
2. GeneticOptimizer - 完全自定义（SAR/Pipeline等任意ADC）

自动收敛：基于斜率判断，无需手动设置迭代次数

Usage:
    python 09_genetic_optimizer.py
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SARADC, SamplingEvent
from quantiamagica.optim import SDCoeffOptimizer, GeneticOptimizer, Gene


def main():
    print("=" * 60)
    print("Genetic Optimizer: ADC Parameter Optimization")
    print("=" * 60)
    
    # =========================================================================
    # Example 1: SDCoeffOptimizer - SD ADC系数一键优化
    # =========================================================================
    print("\n" + "-" * 60)
    print("1. SDCoeffOptimizer - One-Line SD ADC Optimization")
    print("-" * 60)
    
    sd_opt = SDCoeffOptimizer(order=2, bits=1, osr=64, fs=1e6)
    sd_result = sd_opt.optimize(seed=42)
    
    print(sd_result.summary())
    
    # =========================================================================
    # Example 2: GeneticOptimizer - 完全自定义 (SAR ADC示例)
    # =========================================================================
    print("\n" + "-" * 60)
    print("2. GeneticOptimizer - Custom SAR ADC Optimization")
    print("-" * 60)
    print("  优化采样噪声和比较器失调的影响")
    
    # 定义基因：采样噪声标准差、比较器失调
    genes = [
        Gene('sampling_noise', 0.0001, 0.01, 'float', log_scale=True),
        Gene('comparator_offset', 0.0, 0.005, 'float'),
    ]
    
    def evaluate_sar(params):
        adc = SARADC(bits=10)
        
        @adc.on(SamplingEvent)
        def add_noise(event):
            event.voltage += np.random.normal(0, params['sampling_noise'])
        
        adc.sim(n_samples=2048, fs=1e6, fin=12345, amplitude=0.48)
        return adc.enob()
    
    # 获取基准
    baseline = evaluate_sar({'sampling_noise': 0.001, 'comparator_offset': 0.001})
    print(f"  Baseline ENOB: {baseline:.2f} bits")
    
    # 优化（自动收敛）
    optimizer = GeneticOptimizer(genes, evaluate_sar, maximize=True)
    result = optimizer.run(population_size=30, seed=123)
    
    print(f"\n  Optimized ENOB: {result.best_fitness:.2f} bits")
    print(f"  Best params: {result.best_params}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  SD ADC: {sd_result.baseline_enob:.2f} -> {sd_result.best_enob:.2f} bits (+{sd_result.improvement:.2f})")
    print(f"  SAR ADC: {baseline:.2f} -> {result.best_fitness:.2f} bits")
    print("\n  GeneticOptimizer支持任意ADC和任意参数的优化")
    print("=" * 60)


if __name__ == "__main__":
    main()
