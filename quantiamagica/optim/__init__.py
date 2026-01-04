"""
QuantiaMagica Optimization Library
==================================

高度抽象的遗传算法优化库，支持CUDA/CPU并行计算。

核心类:
- GeneticOptimizer: 遗传算法优化器基类
- Gene: 基因定义（参数空间）
- Individual: 个体（一组参数）
- Population: 种群

使用示例:
---------
>>> from quantiamagica.optim import GeneticOptimizer, Gene
>>> 
>>> # 定义基因（参数空间）
>>> genes = [
...     Gene('amplitude', 0.1, 0.5, 'float'),
...     Gene('frequency', 1000, 10000, 'float'),
...     Gene('order', 1, 3, 'int'),
... ]
>>> 
>>> # 定义适应度函数
>>> def fitness(params):
...     adc = create_adc(params)
...     return adc.enob()
>>> 
>>> # 优化
>>> optimizer = GeneticOptimizer(genes, fitness, maximize=True)
>>> best = optimizer.run(generations=50, population_size=100)
>>> print(f"Best params: {best.params}, fitness: {best.fitness}")
"""

from .gene import Gene, GeneType
from .individual import Individual
from .population import Population
from .optimizer import GeneticOptimizer, OptimizationResult
from .gpu_utils import (
    GPUBatchSimulator,
    GPUSigmaDeltaSimulator,
    GPUSARSimulator, 
    GPUPipelineSimulator,
    create_batch_sd_evaluator,
    create_batch_sar_evaluator,
    create_batch_pipeline_evaluator,
    create_simple_batch_evaluator,
)
from .constraints import Constraint, RangeConstraint, CustomConstraint
from .adc_optimizer import SDCoeffOptimizer, ADCOptimizeResult

__all__ = [
    'Gene',
    'GeneType', 
    'Individual',
    'Population',
    'GeneticOptimizer',
    'OptimizationResult',
    'Constraint',
    'RangeConstraint',
    'CustomConstraint',
    'SDCoeffOptimizer',
    'ADCOptimizeResult',
    'GPUBatchSimulator',
    'GPUSigmaDeltaSimulator',
    'GPUSARSimulator',
    'GPUPipelineSimulator',
    'create_batch_sd_evaluator',
    'create_batch_sar_evaluator',
    'create_batch_pipeline_evaluator',
    'create_simple_batch_evaluator',
]
