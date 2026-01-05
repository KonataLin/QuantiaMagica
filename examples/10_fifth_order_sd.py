"""
五阶Sigma-Delta ADC系数优化 - 高阶极致性能
==========================================

使用遗传算法优化5阶SD ADC的积分器增益和反馈系数，
目标是在保证稳定性的前提下极致最大化ENOB。

5阶SD ADC理论ENOB较高，但实际因稳定性限制，通常只能达到 16-20 bits

Usage:
    python 10_fifth_order_sd.py
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SigmaDeltaADC, QuantizerEvent
from quantiamagica.optim import GeneticOptimizer, Gene


def main():
    print("=" * 70)
    print("五阶 Sigma-Delta ADC 系数优化 - 极致性能")
    print("=" * 70)
    
    # ===================== 仿真参数 =====================
    OSR = 64   # 保持OSR=64
    BITS = 1   # 1位量化器
    FS = 1e6
    N_SAMPLES = OSR * 512  # 增加采样点提高FFT分辨率
    FIN = 13 * FS / N_SAMPLES  # 相干采样
    
    print(f"\n仿真参数:")
    print(f"  OSR = {OSR}")
    print(f"  量化器位数 = {BITS} bits")
    print(f"  采样频率 = {FS/1e6:.1f} MHz")
    print(f"  采样点数 = {N_SAMPLES}")
    print(f"  信号频率 = {FIN:.1f} Hz")
    print(f"  理论带宽 = {FS/(2*OSR)/1e3:.2f} kHz")
    
    # ===================== 定义5阶SD ADC基因 =====================
    # 5阶需要5个积分器增益和5个反馈系数
    # 扩大搜索范围以找到更优解
    genes = [
        # 积分器增益 - 适当放宽范围
        Gene('c1', 0.05, 0.6, 'float'),
        Gene('c2', 0.04, 0.5, 'float'),
        Gene('c3', 0.03, 0.4, 'float'),
        Gene('c4', 0.02, 0.3, 'float'),
        Gene('c5', 0.01, 0.2, 'float'),
        # 反馈系数 - 扩大范围
        Gene('a1', 0.5, 4.0, 'float'),
        Gene('a2', 0.3, 3.5, 'float'),
        Gene('a3', 0.2, 3.0, 'float'),
        Gene('a4', 0.1, 2.5, 'float'),
        Gene('a5', 0.05, 2.0, 'float'),
        # 输入幅度 - 高阶需要更小幅度保稳定
        Gene('amplitude', 0.03, 0.2, 'float'),
    ]
    
    print(f"\n优化参数: {len(genes)} 个基因")
    for g in genes:
        print(f"  {g.name}: [{g.low}, {g.high}]")
    
    # ===================== 评估函数 =====================
    def evaluate_5th_order_sd(params):
        """创建并评估5阶SD ADC"""
        c = [params[f'c{i+1}'] for i in range(5)]
        a = [params[f'a{i+1}'] for i in range(5)]
        amplitude = params['amplitude']
        
        sd = SigmaDeltaADC(order=1, bits=BITS, osr=OSR)  # 使用多位量化器
        state = [0.0] * 5
        
        @sd.on(QuantizerEvent)
        def fifth_order_cifb(event):
            x = event.input_signal
            y = event.prev_output
            
            # 5阶级联积分器 + 分布式反馈
            state[0] = state[0] + c[0] * (x - a[0] * y)
            for i in range(1, 5):
                state[i] = state[i] + c[i] * (state[i-1] - a[i] * y)
            
            event.quantizer_input = state[4]
        
        t = np.arange(N_SAMPLES) / FS
        signal = 0.5 + amplitude * np.sin(2 * np.pi * FIN * t)
        
        sd.sim(signal, fs=FS)
        sd._result.metadata['fin'] = FIN
        
        return sd.enob()
    
    # ===================== 适应度函数 =====================
    def fitness(params):
        try:
            enob = evaluate_5th_order_sd(params)
            if enob <= 0 or np.isnan(enob) or enob > 35:
                return -1000
            return enob
        except Exception:
            return -1000
    
    # ===================== 获取基准值 =====================
    print("\n" + "-" * 70)
    print("计算基准值...")
    baseline_params = {
        'c1': 0.25, 'c2': 0.18, 'c3': 0.12, 'c4': 0.08, 'c5': 0.05,
        'a1': 2.0, 'a2': 1.5, 'a3': 1.0, 'a4': 0.6, 'a5': 0.3,
        'amplitude': 0.1
    }
    baseline_enob = evaluate_5th_order_sd(baseline_params)
    print(f"  默认系数 ENOB = {baseline_enob:.2f} bits")
    
    # ===================== 运行优化 =====================
    print("\n" + "-" * 70)
    print("开始遗传算法优化...")
    print("  (11个参数空间较大，使用更大种群和更多迭代)")
    
    optimizer = GeneticOptimizer(genes, fitness, maximize=True)
    result = optimizer.run(
        population_size=300,       # 大种群充分探索11维空间
        max_generations=500,       # 足够迭代
        mutation_rate=0.18,        # 适中变异率
        mutation_strength=0.2,     # 适中变异强度
        crossover_rate=0.75,       # 高交叉率
        elite_ratio=0.10,          # 保留8%精英
        slope_threshold=0.0002,    # 严格收敛条件
        seed=2024                  # 固定种子
    )
    
    # ===================== 验证结果 =====================
    best_enob = evaluate_5th_order_sd(result.best_params)
    improvement = best_enob - baseline_enob
    
    print("\n" + "=" * 70)
    print("优化结果")
    print("=" * 70)
    
    print(f"\n  基准 ENOB:   {baseline_enob:.2f} bits")
    print(f"  优化后 ENOB: {best_enob:.2f} bits")
    if baseline_enob > 0:
        print(f"  提升:        +{improvement:.2f} bits ({improvement/baseline_enob*100:.1f}%)")
    else:
        print(f"  提升:        从振荡到 {best_enob:.2f} bits")
    
    print(f"\n  优化后的系数:")
    print(f"    积分器增益: c1={result.best_params['c1']:.4f}, c2={result.best_params['c2']:.4f}, "
          f"c3={result.best_params['c3']:.4f}, c4={result.best_params['c4']:.4f}, c5={result.best_params['c5']:.4f}")
    print(f"    反馈系数:   a1={result.best_params['a1']:.4f}, a2={result.best_params['a2']:.4f}, "
          f"a3={result.best_params['a3']:.4f}, a4={result.best_params['a4']:.4f}, a5={result.best_params['a5']:.4f}")
    print(f"    输入幅度:   amplitude={result.best_params['amplitude']:.4f}")
    
    print(f"\n  收敛信息: 迭代{result.generations}代, 收敛={result.converged}")
    print("=" * 70)


if __name__ == "__main__":
    main()
