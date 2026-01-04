"""
Population - 种群模块

管理个体集合，支持选择、进化操作。
"""

from typing import List, Optional, Callable
import numpy as np

from .gene import Gene
from .individual import Individual


class Population:
    """
    种群 - 个体集合
    
    Parameters
    ----------
    genes : List[Gene]
        基因定义列表
    size : int
        种群大小
    individuals : List[Individual], optional
        初始个体列表
    """
    
    def __init__(self, genes: List[Gene], size: int = 50,
                 individuals: Optional[List[Individual]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.genes = genes
        self.size = size
        self.rng = rng or np.random.default_rng()
        
        if individuals is not None:
            self.individuals = individuals
        else:
            self.individuals = [Individual.random(genes, self.rng) for _ in range(size)]
    
    @property
    def best(self) -> Individual:
        """获取最佳个体"""
        return max(self.individuals, key=lambda x: x.fitness)
    
    @property
    def worst(self) -> Individual:
        """获取最差个体"""
        return min(self.individuals, key=lambda x: x.fitness)
    
    @property
    def mean_fitness(self) -> float:
        """平均适应度"""
        return np.mean([ind.fitness for ind in self.individuals])
    
    @property
    def std_fitness(self) -> float:
        """适应度标准差"""
        return np.std([ind.fitness for ind in self.individuals])
    
    def sort(self, reverse: bool = True):
        """按适应度排序"""
        self.individuals.sort(key=lambda x: x.fitness, reverse=reverse)
    
    def select_tournament(self, k: int = 3) -> Individual:
        """锦标赛选择"""
        contestants = self.rng.choice(self.individuals, size=k, replace=False)
        return max(contestants, key=lambda x: x.fitness)
    
    def select_roulette(self) -> Individual:
        """轮盘赌选择"""
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        # 处理负适应度
        min_fit = fitnesses.min()
        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-10
        probs = fitnesses / fitnesses.sum()
        idx = self.rng.choice(len(self.individuals), p=probs)
        return self.individuals[idx]
    
    def evolve(self, 
               elite_ratio: float = 0.1,
               mutation_rate: float = 0.1,
               mutation_strength: float = 0.2,
               crossover_rate: float = 0.7,
               selection: str = 'tournament') -> 'Population':
        """
        进化一代
        
        Parameters
        ----------
        elite_ratio : float
            精英保留比例
        mutation_rate : float
            变异概率
        mutation_strength : float
            变异强度
        crossover_rate : float
            交叉概率
        selection : str
            选择策略: 'tournament' 或 'roulette'
        
        Returns
        -------
        Population
            新一代种群
        """
        self.sort()
        
        # 精英保留
        n_elite = max(1, int(self.size * elite_ratio))
        new_individuals = [ind.copy() for ind in self.individuals[:n_elite]]
        
        # 选择函数
        select_fn = self.select_tournament if selection == 'tournament' else self.select_roulette
        
        # 生成新个体
        while len(new_individuals) < self.size:
            parent1 = select_fn()
            
            if self.rng.random() < crossover_rate:
                parent2 = select_fn()
                child = parent1.crossover(parent2, self.rng)
            else:
                child = parent1.copy()
            
            child = child.mutate(mutation_rate, mutation_strength, self.rng)
            new_individuals.append(child)
        
        return Population(
            genes=self.genes,
            size=self.size,
            individuals=new_individuals[:self.size],
            rng=self.rng
        )
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __iter__(self):
        return iter(self.individuals)
    
    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]
