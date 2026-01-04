"""
Gene - 基因定义模块

定义参数空间的基本单元，支持多种数据类型。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, List, Union
import numpy as np


class GeneType(Enum):
    """基因类型枚举"""
    FLOAT = 'float'      # 连续浮点数
    INT = 'int'          # 整数
    CHOICE = 'choice'    # 离散选择
    BOOL = 'bool'        # 布尔值


@dataclass
class Gene:
    """
    基因定义 - 描述一个可优化参数的空间
    
    Parameters
    ----------
    name : str
        参数名称
    low : float
        下界（对于FLOAT/INT类型）
    high : float
        上界（对于FLOAT/INT类型）
    dtype : str or GeneType
        数据类型: 'float', 'int', 'choice', 'bool'
    choices : list, optional
        离散选择列表（仅用于CHOICE类型）
    log_scale : bool
        是否使用对数尺度（用于跨越多个数量级的参数）
    
    Examples
    --------
    >>> # 连续参数
    >>> Gene('amplitude', 0.1, 0.5, 'float')
    
    >>> # 整数参数
    >>> Gene('order', 1, 5, 'int')
    
    >>> # 离散选择
    >>> Gene('topology', choices=['CIFB', 'CIFF', 'CRFB'])
    
    >>> # 对数尺度（如电容值）
    >>> Gene('capacitance', 1e-15, 1e-12, 'float', log_scale=True)
    """
    name: str
    low: float = 0.0
    high: float = 1.0
    dtype: Union[str, GeneType] = 'float'
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    
    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = GeneType(self.dtype)
        
        if self.dtype == GeneType.CHOICE:
            if self.choices is None:
                raise ValueError("CHOICE type requires 'choices' list")
            self.low = 0
            self.high = len(self.choices) - 1
        elif self.dtype == GeneType.BOOL:
            self.low = 0
            self.high = 1
    
    def random(self, rng: Optional[np.random.Generator] = None) -> Any:
        """生成随机值"""
        if rng is None:
            rng = np.random.default_rng()
        
        if self.dtype == GeneType.FLOAT:
            if self.log_scale:
                log_val = rng.uniform(np.log(self.low), np.log(self.high))
                return np.exp(log_val)
            return rng.uniform(self.low, self.high)
        elif self.dtype == GeneType.INT:
            return rng.integers(int(self.low), int(self.high) + 1)
        elif self.dtype == GeneType.CHOICE:
            idx = rng.integers(0, len(self.choices))
            return self.choices[idx]
        elif self.dtype == GeneType.BOOL:
            return bool(rng.integers(0, 2))
        raise ValueError(f"Unknown dtype: {self.dtype}")
    
    def clip(self, value: Any) -> Any:
        """将值裁剪到有效范围"""
        if self.dtype == GeneType.FLOAT:
            return float(np.clip(value, self.low, self.high))
        elif self.dtype == GeneType.INT:
            return int(np.clip(int(round(value)), int(self.low), int(self.high)))
        elif self.dtype == GeneType.CHOICE:
            idx = int(np.clip(int(round(value)), 0, len(self.choices) - 1))
            return self.choices[idx]
        elif self.dtype == GeneType.BOOL:
            return bool(round(value))
        return value
    
    def to_normalized(self, value: Any) -> float:
        """将值转换为[0,1]归一化空间"""
        if self.dtype == GeneType.CHOICE:
            return self.choices.index(value) / max(1, len(self.choices) - 1)
        elif self.dtype == GeneType.BOOL:
            return 1.0 if value else 0.0
        elif self.log_scale:
            return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (value - self.low) / (self.high - self.low) if self.high > self.low else 0.5
    
    def from_normalized(self, norm_value: float) -> Any:
        """从[0,1]归一化空间转换回原始值"""
        norm_value = np.clip(norm_value, 0, 1)
        if self.dtype == GeneType.CHOICE:
            idx = int(round(norm_value * (len(self.choices) - 1)))
            return self.choices[idx]
        elif self.dtype == GeneType.BOOL:
            return norm_value >= 0.5
        elif self.dtype == GeneType.INT:
            return int(round(self.low + norm_value * (self.high - self.low)))
        elif self.log_scale:
            log_val = np.log(self.low) + norm_value * (np.log(self.high) - np.log(self.low))
            return np.exp(log_val)
        return self.low + norm_value * (self.high - self.low)
