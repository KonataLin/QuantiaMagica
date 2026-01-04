"""
Individual - 个体模块

表示种群中的一个个体（一组参数值）。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from .gene import Gene


@dataclass
class Individual:
    """
    个体 - 一组基因值的集合
    
    Attributes
    ----------
    genes : List[Gene]
        基因定义列表
    values : Dict[str, Any]
        基因名称到值的映射
    fitness : float
        适应度值（由优化器设置）
    normalized : np.ndarray
        归一化后的基因值向量
    """
    genes: List[Gene]
    values: Dict[str, Any] = field(default_factory=dict)
    fitness: float = float('-inf')
    _normalized: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        if not self.values:
            self.values = {g.name: g.random() for g in self.genes}
        self._update_normalized()
    
    def _update_normalized(self):
        """更新归一化向量"""
        self._normalized = np.array([
            g.to_normalized(self.values[g.name]) for g in self.genes
        ])
    
    @property
    def normalized(self) -> np.ndarray:
        """获取归一化向量"""
        return self._normalized
    
    @normalized.setter
    def normalized(self, vec: np.ndarray):
        """从归一化向量设置值"""
        self._normalized = np.clip(vec, 0, 1)
        for i, g in enumerate(self.genes):
            self.values[g.name] = g.from_normalized(self._normalized[i])
    
    @classmethod
    def random(cls, genes: List[Gene], rng: Optional[np.random.Generator] = None) -> 'Individual':
        """创建随机个体"""
        if rng is None:
            rng = np.random.default_rng()
        values = {g.name: g.random(rng) for g in genes}
        return cls(genes=genes, values=values)
    
    @classmethod
    def from_normalized(cls, genes: List[Gene], normalized: np.ndarray) -> 'Individual':
        """从归一化向量创建个体"""
        ind = cls(genes=genes)
        ind.normalized = normalized
        return ind
    
    def copy(self) -> 'Individual':
        """创建副本"""
        return Individual(
            genes=self.genes,
            values=self.values.copy(),
            fitness=self.fitness,
            _normalized=self._normalized.copy() if self._normalized is not None else None
        )
    
    def mutate(self, rate: float = 0.1, strength: float = 0.2, 
               rng: Optional[np.random.Generator] = None) -> 'Individual':
        """
        变异操作
        
        Parameters
        ----------
        rate : float
            变异概率（每个基因）
        strength : float
            变异强度（归一化空间中的标准差）
        """
        if rng is None:
            rng = np.random.default_rng()
        
        new_normalized = self._normalized.copy()
        for i in range(len(self.genes)):
            if rng.random() < rate:
                new_normalized[i] += rng.normal(0, strength)
        
        child = self.copy()
        child.normalized = new_normalized
        child.fitness = float('-inf')
        return child
    
    def crossover(self, other: 'Individual', 
                  rng: Optional[np.random.Generator] = None) -> 'Individual':
        """
        交叉操作（均匀交叉）
        """
        if rng is None:
            rng = np.random.default_rng()
        
        mask = rng.random(len(self.genes)) < 0.5
        new_normalized = np.where(mask, self._normalized, other._normalized)
        
        child = Individual.from_normalized(self.genes, new_normalized)
        return child
    
    def __getitem__(self, key: str) -> Any:
        return self.values[key]
    
    def __repr__(self) -> str:
        params = ', '.join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" 
                          for k, v in self.values.items())
        return f"Individual({params}, fitness={self.fitness:.4f})"
