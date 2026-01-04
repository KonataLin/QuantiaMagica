"""
ADC Optimizer - 专用于ADC参数优化的高级接口

提供开箱即用的ADC优化功能，同时保持高度可定制性。
"""

from typing import Callable, Dict, Any, Optional, List, Union
from dataclasses import dataclass
import numpy as np

from .gene import Gene
from .optimizer import GeneticOptimizer, OptimizationResult
from .constraints import CustomConstraint


@dataclass
class ADCOptimizeResult:
    """ADC优化结果"""
    best_params: Dict[str, Any]
    best_enob: float
    baseline_enob: float
    improvement: float
    generations: int
    history: List[float]
    
    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 50,
            "ADC Optimization Result",
            "=" * 50,
            f"  Baseline ENOB: {self.baseline_enob:.2f} bits",
            f"  Optimized ENOB: {self.best_enob:.2f} bits",
            f"  Improvement: +{self.improvement:.2f} bits ({100*self.improvement/max(0.01, self.baseline_enob):.1f}%)",
            "-" * 50,
            "  Optimized parameters:",
        ]
        for k, v in self.best_params.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4f}")
            else:
                lines.append(f"    {k}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


class SDCoeffOptimizer:
    """
    Sigma-Delta ADC 系数优化器
    
    一行代码优化SD ADC的积分器系数、反馈系数和输入幅度。
    
    Parameters
    ----------
    order : int
        调制器阶数 (1, 2, 3)
    bits : int
        量化器位数 (1 for 1-bit)
    osr : int
        过采样率
    fs : float
        采样频率
    n_samples : int, optional
        仿真样本数
    
    Examples
    --------
    >>> opt = SDCoeffOptimizer(order=2, bits=1, osr=64, fs=1e6)
    >>> result = opt.optimize(generations=30)
    >>> print(result.summary())
    """
    
    def __init__(self, order: int = 2, bits: int = 1, osr: int = 64,
                 fs: float = 1e6, n_samples: Optional[int] = None):
        self.order = order
        self.bits = bits
        self.osr = osr
        self.fs = fs
        self.n_samples = n_samples or osr * 128
        self.fin = 13 * fs / self.n_samples  # Coherent sampling
        
        # Default coefficient ranges based on order
        self._setup_default_genes()
    
    def _setup_default_genes(self):
        """设置默认基因范围"""
        self.genes = []
        
        # Integrator gains
        for i in range(self.order):
            self.genes.append(Gene(f'c{i+1}', 0.1, 0.8, 'float'))
        
        # Feedback coefficients
        for i in range(self.order):
            self.genes.append(Gene(f'a{i+1}', 0.5, 3.0, 'float'))
        
        # Input amplitude
        if self.bits == 1:
            self.genes.append(Gene('amplitude', 0.15, 0.4, 'float'))
        else:
            self.genes.append(Gene('amplitude', 0.25, 0.45, 'float'))
    
    def set_gene_range(self, name: str, low: float, high: float) -> 'SDCoeffOptimizer':
        """自定义基因范围"""
        for g in self.genes:
            if g.name == name:
                g.low = low
                g.high = high
                return self
        # Add new gene
        self.genes.append(Gene(name, low, high, 'float'))
        return self
    
    def _create_and_evaluate(self, params: Dict[str, Any]) -> float:
        """创建SD ADC并评估ENOB"""
        from ..adc.sigma_delta import SigmaDeltaADC, QuantizerEvent
        
        amplitude = params.get('amplitude', 0.3)
        
        # Create base SD ADC
        sd = SigmaDeltaADC(order=1, bits=self.bits, osr=self.osr)
        
        # Build state array for N-order
        state = [0.0] * self.order
        
        # Get coefficients
        c = [params.get(f'c{i+1}', 0.5) for i in range(self.order)]
        a = [params.get(f'a{i+1}', 1.0) for i in range(self.order)]
        order = self.order
        
        @sd.on(QuantizerEvent)
        def custom_topology(event):
            x = event.input_signal
            y = event.prev_output
            
            # N-order CIFB implementation
            for i in range(order):
                fb = a[i] * y
                if i == 0:
                    state[i] = state[i] + c[i] * (x - fb)
                else:
                    state[i] = state[i] + c[i] * (state[i-1] - fb)
            
            event.quantizer_input = state[order - 1]
        
        # Generate input signal
        t = np.arange(self.n_samples) / self.fs
        input_signal = 0.5 + amplitude * np.sin(2 * np.pi * self.fin * t)
        
        # Simulate
        sd.sim(input_signal, fs=self.fs)
        sd._result.metadata['fin'] = self.fin
        
        return sd.enob()
    
    def _fitness(self, params: Dict[str, Any]) -> float:
        """适应度函数"""
        try:
            enob = self._create_and_evaluate(params)
            
            if enob <= 0 or np.isnan(enob) or enob > 30:
                return -1000  # Oscillation or invalid
            
            # ENOB is primary, amplitude is secondary bonus
            amplitude = params.get('amplitude', 0.3)
            return enob * 10 + amplitude * 5
        except:
            return -1000
    
    def get_baseline(self, c: float = 0.5, a: float = 1.0, amplitude: float = 0.3) -> float:
        """获取基准ENOB（使用默认系数）"""
        params = {'amplitude': amplitude}
        for i in range(self.order):
            params[f'c{i+1}'] = c
            params[f'a{i+1}'] = a * (i + 1)  # a1=1, a2=2, ...
        return self._create_and_evaluate(params)
    
    def optimize(self,
                 population_size: int = 40,
                 verbose: bool = True,
                 seed: Optional[int] = None) -> ADCOptimizeResult:
        """
        运行优化（自动收敛）
        
        Parameters
        ----------
        population_size : int
            种群大小
        verbose : bool
            是否显示进度
        seed : int, optional
            随机种子
        
        Returns
        -------
        ADCOptimizeResult
            优化结果，包含对比信息
        """
        baseline = self.get_baseline()
        
        if verbose:
            print(f"\nBaseline ENOB: {baseline:.2f} bits (default coefficients)")
            print(f"Optimizing {self.order}-order {self.bits}-bit SD ADC...")
        
        optimizer = GeneticOptimizer(
            genes=self.genes,
            fitness_fn=self._fitness,
            maximize=True
        )
        
        result = optimizer.run(
            population_size=population_size,
            verbose=verbose,
            seed=seed
        )
        
        # Verify and get actual ENOB
        best_enob = self._create_and_evaluate(result.best_params)
        
        return ADCOptimizeResult(
            best_params=result.best_params,
            best_enob=best_enob,
            baseline_enob=baseline,
            improvement=best_enob - baseline,
            generations=result.generations,
            history=result.history
        )


