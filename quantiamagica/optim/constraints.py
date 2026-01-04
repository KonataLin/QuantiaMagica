"""
Constraints - 约束条件模块

定义优化过程中的约束条件。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass


class Constraint(ABC):
    """约束基类"""
    
    @abstractmethod
    def check(self, params: Dict[str, Any]) -> bool:
        """检查参数是否满足约束"""
        pass
    
    @abstractmethod
    def penalty(self, params: Dict[str, Any]) -> float:
        """计算违反约束的惩罚值"""
        pass


@dataclass
class RangeConstraint(Constraint):
    """
    范围约束
    
    Parameters
    ----------
    param_name : str
        参数名称
    min_value : float
        最小值
    max_value : float
        最大值
    penalty_weight : float
        惩罚权重
    """
    param_name: str
    min_value: float = float('-inf')
    max_value: float = float('inf')
    penalty_weight: float = 100.0
    
    def check(self, params: Dict[str, Any]) -> bool:
        value = params.get(self.param_name, 0)
        return self.min_value <= value <= self.max_value
    
    def penalty(self, params: Dict[str, Any]) -> float:
        value = params.get(self.param_name, 0)
        if value < self.min_value:
            return self.penalty_weight * (self.min_value - value) ** 2
        elif value > self.max_value:
            return self.penalty_weight * (value - self.max_value) ** 2
        return 0.0


@dataclass  
class CustomConstraint(Constraint):
    """
    自定义约束
    
    Parameters
    ----------
    check_fn : Callable
        检查函数，接受params字典，返回bool
    penalty_fn : Callable, optional
        惩罚函数，接受params字典，返回float
    penalty_weight : float
        默认惩罚权重（当penalty_fn未提供时使用）
    
    Examples
    --------
    >>> # 确保ENOB > 0（不振荡）
    >>> CustomConstraint(
    ...     check_fn=lambda p: p.get('enob', 0) > 0,
    ...     penalty_fn=lambda p: 1000 if p.get('enob', 0) <= 0 else 0
    ... )
    
    >>> # 确保参数组合有效
    >>> CustomConstraint(
    ...     check_fn=lambda p: p['a'] + p['b'] < 1.0,
    ...     penalty_weight=50.0
    ... )
    """
    check_fn: Callable[[Dict[str, Any]], bool]
    penalty_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    penalty_weight: float = 100.0
    
    def check(self, params: Dict[str, Any]) -> bool:
        return self.check_fn(params)
    
    def penalty(self, params: Dict[str, Any]) -> float:
        if not self.check(params):
            if self.penalty_fn is not None:
                return self.penalty_fn(params)
            return self.penalty_weight
        return 0.0


class ConstraintSet:
    """约束集合"""
    
    def __init__(self):
        self.constraints = []
    
    def add(self, constraint: Constraint) -> 'ConstraintSet':
        self.constraints.append(constraint)
        return self
    
    def check_all(self, params: Dict[str, Any]) -> bool:
        return all(c.check(params) for c in self.constraints)
    
    def total_penalty(self, params: Dict[str, Any]) -> float:
        return sum(c.penalty(params) for c in self.constraints)
    
    def __len__(self) -> int:
        return len(self.constraints)
