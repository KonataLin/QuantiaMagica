"""
GeneticOptimizer - 遗传算法优化器

支持CUDA/CPU并行计算的高度抽象优化器。
"""

from typing import List, Callable, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

# GPU支持 (可选)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .gene import Gene
from .individual import Individual
from .population import Population
from .constraints import Constraint, ConstraintSet


# 全局线程池（避免重复创建）
_THREAD_POOL: Optional[ThreadPoolExecutor] = None


def _get_thread_pool(n_workers: int) -> ThreadPoolExecutor:
    """获取或创建全局线程池"""
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPoolExecutor(max_workers=n_workers)
    return _THREAD_POOL


@dataclass
class OptimizationResult:
    """优化结果"""
    best_individual: Individual
    best_params: Dict[str, Any]
    best_fitness: float
    history: List[float]
    generations: int
    converged: bool
    reason: str


class GeneticOptimizer:
    """
    遗传算法优化器
    
    高度抽象的优化器，支持:
    - 任意参数空间定义（连续、离散、布尔）
    - 自定义适应度函数
    - 约束条件
    - CUDA/CPU并行评估
    - 多种选择策略
    - 自适应参数
    
    Parameters
    ----------
    genes : List[Gene]
        基因定义列表
    fitness_fn : Callable
        适应度函数，接受params字典，返回float
    maximize : bool
        是否最大化适应度（默认True）
    constraints : List[Constraint], optional
        约束条件列表
    
    Examples
    --------
    >>> # 优化SD ADC系数
    >>> genes = [
    ...     Gene('c1', 0.1, 1.0, 'float'),
    ...     Gene('c2', 0.1, 1.0, 'float'),
    ...     Gene('a1', 0.5, 2.0, 'float'),
    ...     Gene('amplitude', 0.1, 0.5, 'float'),
    ... ]
    >>> 
    >>> def fitness(params):
    ...     adc = create_sd_adc(params)
    ...     enob = adc.enob()
    ...     if enob < 0:  # 振荡
    ...         return -1000
    ...     return enob + params['amplitude'] * 10  # ENOB + 输入摆幅奖励
    >>> 
    >>> optimizer = GeneticOptimizer(genes, fitness)
    >>> result = optimizer.run(generations=100, population_size=50)
    """
    
    def __init__(self,
                 genes: List[Gene],
                 fitness_fn: Callable[[Dict[str, Any]], float],
                 maximize: bool = True,
                 constraints: Optional[List[Constraint]] = None,
                 batch_fitness_fn: Optional[Callable[[List[Dict[str, Any]]], List[float]]] = None,
                 use_gpu: bool = True):
        """
        Parameters
        ----------
        genes : List[Gene]
            基因定义列表
        fitness_fn : Callable
            单个体适应度函数
        maximize : bool
            是否最大化
        constraints : List[Constraint], optional
            约束条件
        batch_fitness_fn : Callable, optional
            批量适应度函数（用于GPU加速）
        use_gpu : bool
            是否使用GPU加速（需要提供batch_fitness_fn）
        """
        self.genes = genes
        self.fitness_fn = fitness_fn
        self.batch_fitness_fn = batch_fitness_fn
        self.maximize = maximize
        self.use_gpu = use_gpu
        self.constraint_set = ConstraintSet()
        if constraints:
            for c in constraints:
                self.constraint_set.add(c)
        
        # Hardware detection - 多种方式检测CUDA
        self.has_cuda = False
        self.gpu_name = ""
        
        # 方法1: 尝试torch
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                self.gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
        
        # 方法2: 如果torch检测失败，尝试nvidia-smi
        if not self.has_cuda:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.has_cuda = True
                    self.gpu_name = result.stdout.strip().split('\n')[0]
            except:
                pass
        
        # State
        self.population: Optional[Population] = None
        self.best_ever: Optional[Individual] = None
        self.history: List[float] = []
        self.generation = 0
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """评估单个个体"""
        try:
            # 计算适应度
            raw_fitness = self.fitness_fn(individual.values)
            
            # 应用约束惩罚
            penalty = self.constraint_set.total_penalty(individual.values)
            
            fitness = raw_fitness - penalty if self.maximize else raw_fitness + penalty
            
            if not np.isfinite(fitness):
                fitness = float('-inf') if self.maximize else float('inf')
            
            return fitness
        except Exception as e:
            return float('-inf') if self.maximize else float('inf')
    
    def _evaluate_population(self, population: Population, n_workers: int) -> None:
        """
        并行评估种群 - 每个个体一个线程，最大化并行度
        
        优化策略:
        - GPU模式: 使用batch_fitness_fn批量评估
        - CPU模式: 每个个体独立线程，全局线程池管理
        """
        # GPU加速路径
        if self.has_cuda and self.use_gpu and self.batch_fitness_fn is not None:
            self._evaluate_population_gpu(population)
            return
        
        individuals = population.individuals
        n_individuals = len(individuals)
        
        # 小种群：直接串行评估（避免线程开销）
        if n_individuals <= 2:
            for ind in individuals:
                ind.fitness = self._evaluate_individual(ind)
            return
        
        # 每个个体一个线程，使用全局线程池
        # 线程池大小 = 种群大小，充分利用CPU
        pool = _get_thread_pool(min(n_individuals, n_workers * 4))
        
        # 提交所有个体评估任务
        futures = [pool.submit(self._evaluate_individual, ind) for ind in individuals]
        
        # 并行收集结果（as_completed可以更快获取已完成的结果）
        for ind, future in zip(individuals, futures):
            ind.fitness = future.result()
    
    def _evaluate_population_gpu(self, population: Population) -> None:
        """GPU批量评估种群"""
        # 收集所有个体的参数
        all_params = [ind.values for ind in population.individuals]
        
        try:
            # 调用批量适应度函数
            raw_fitnesses = self.batch_fitness_fn(all_params)
            
            # 应用约束惩罚
            for ind, raw_fit in zip(population.individuals, raw_fitnesses):
                penalty = self.constraint_set.total_penalty(ind.values)
                fitness = raw_fit - penalty if self.maximize else raw_fit + penalty
                
                if not np.isfinite(fitness):
                    fitness = float('-inf') if self.maximize else float('inf')
                
                ind.fitness = fitness
        except Exception as e:
            # GPU评估失败，回退到CPU
            for ind in population.individuals:
                ind.fitness = self._evaluate_individual(ind)
    
    def run(self,
            population_size: int = 50,
            max_generations: int = 200,
            elite_ratio: float = 0.1,
            mutation_rate: float = 0.15,
            mutation_strength: float = 0.2,
            crossover_rate: float = 0.7,
            selection: str = 'tournament',
            slope_window: int = 5,
            slope_threshold: float = 0.001,
            n_workers: Optional[int] = None,
            verbose: bool = True,
            callback: Optional[Callable[[int, Individual], None]] = None,
            seed: Optional[int] = None) -> OptimizationResult:
        """
        运行优化（自动斜率收敛）
        
        Parameters
        ----------
        population_size : int
            种群大小
        max_generations : int
            最大迭代代数（自动收敛时通常不会到达）
        slope_window : int
            斜率计算窗口
        slope_threshold : float
            收敛斜率阈值（适应度变化/代数 < 此值时收敛）
        verbose : bool
            是否打印进度
        seed : int, optional
            随机种子
        
        Returns
        -------
        OptimizationResult
            优化结果
        """
        rng = np.random.default_rng(seed)
        
        if n_workers is None:
            cpu_count = os.cpu_count() or 4
            n_workers = cpu_count * 2
        
        self.population = Population(self.genes, population_size, rng=rng)
        self.history = []
        self.generation = 0
        self.best_ever = None
        self._initial_slope = None  # 重置初始斜率
        converge_reason = ""
        
        # 判断是否使用GPU加速
        gpu_accel = self.has_cuda and self.use_gpu and self.batch_fitness_fn is not None
        
        if verbose:
            print("=" * 60)
            print(f"GeneticOptimizer: {len(self.genes)} genes, pop={population_size}")
            if gpu_accel:
                print(f"  Hardware: CUDA ({self.gpu_name}) - GPU加速已启用")
            elif self.has_cuda:
                print(f"  Hardware: CUDA ({self.gpu_name}) - 未提供batch_fitness_fn，使用CPU")
            else:
                print(f"  Hardware: CPU")
            print(f"  Auto-convergence: slope < {slope_threshold} over {slope_window} gens")
            print("=" * 60)
        
        # 主循环 - 自动收敛
        for gen in range(max_generations):
            self.generation = gen
            
            # 评估种群
            self._evaluate_population(self.population, n_workers)
            
            # 更新最佳
            current_best = self.population.best
            if self.best_ever is None or \
               (self.maximize and current_best.fitness > self.best_ever.fitness) or \
               (not self.maximize and current_best.fitness < self.best_ever.fitness):
                self.best_ever = current_best.copy()
            
            self.history.append(self.best_ever.fitness)
            
            # 计算斜率和进度
            slope = 1.0
            progress = 0.0
            if len(self.history) >= 2:
                # 使用最近几代的平均斜率（更平滑）
                window = min(len(self.history) - 1, slope_window)
                recent = self.history[-(window + 1):]
                slope = abs(recent[-1] - recent[0]) / max(window, 1)
                
                # 进度基于斜率下降（使用对数尺度）
                # 初始斜率估计为第一次计算的斜率
                if not hasattr(self, '_initial_slope') or self._initial_slope is None:
                    self._initial_slope = max(slope, 0.1)
                
                if slope > 0 and self._initial_slope > slope_threshold:
                    # 线性插值：从初始斜率到阈值
                    progress = 1.0 - (slope - slope_threshold) / (self._initial_slope - slope_threshold)
                    progress = np.clip(progress, 0, 1)
                elif slope <= slope_threshold:
                    progress = 1.0
            
            # 进度条输出
            if verbose:
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r  [{bar}] {progress*100:5.1f}% | "
                               f"Gen {gen+1} | Best={self.best_ever.fitness:.4f} | "
                               f"slope={slope:.2e}")
                sys.stdout.flush()
            
            if callback:
                callback(gen, self.best_ever)
            
            # 斜率收敛检查
            if len(self.history) >= slope_window + 1 and slope < slope_threshold:
                converge_reason = f"slope={slope:.6f} < {slope_threshold}"
                break
            
            # 进化
            self.population = self.population.evolve(
                elite_ratio=elite_ratio,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength,
                crossover_rate=crossover_rate,
                selection=selection
            )
        
        if verbose:
            print(f"\n  Converged at gen {self.generation+1}: {converge_reason}")
            print(f"  Best: {self.best_ever.fitness:.4f}")
            print("=" * 60)
        
        return OptimizationResult(
            best_individual=self.best_ever,
            best_params=self.best_ever.values.copy(),
            best_fitness=self.best_ever.fitness,
            history=self.history.copy(),
            generations=self.generation + 1,
            converged=len(converge_reason) > 0,
            reason=converge_reason if converge_reason else f"max_generations={max_generations}"
        )
    
    def add_constraint(self, constraint: Constraint) -> 'GeneticOptimizer':
        """添加约束"""
        self.constraint_set.add(constraint)
        return self
