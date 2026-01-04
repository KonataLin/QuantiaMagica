# QuantiaMagica 详细维护手册

> **适用对象**: 开发者、维护者、AI助手  
> **版本**: 1.0.0  
> **更新日期**: 2024

本文档提供完整的代码架构说明和扩展指南，确保人类开发者和AI助手都能理解并正确维护代码。

---

## 📁 完整项目结构

```
QuantiaMagica/
├── quantiamagica/                    # 主包
│   ├── __init__.py                   # ⭐ 公共API导出 (添加新功能必须更新)
│   ├── _analysis_impl.py             # 内部分析实现
│   │
│   ├── core/                         # 核心模块
│   │   ├── __init__.py               # 核心导出
│   │   ├── events.py                 # 事件系统 (Event, EventBus, 优先级)
│   │   └── base.py                   # ADConverter 抽象基类
│   │
│   ├── adc/                          # ⭐ ADC实现 (添加新ADC在这里)
│   │   ├── __init__.py               # ADC模块导出
│   │   ├── sar.py                    # SAR ADC + 所有SAR事件
│   │   └── pipeline.py               # Pipeline ADC + 所有Pipeline事件
│   │
│   ├── signals/                      # 信号生成
│   │   └── __init__.py               # Signal类和所有信号类型
│   │
│   ├── analysis/                     # 分析工具
│   │   └── __init__.py               # Analyzer类和独立函数
│   │
│   ├── plotting/                     # 绘图模块
│   │   └── __init__.py               # IEEE JSSC风格绘图
│   │
│   └── utils/                        # 工具函数
│       └── __init__.py               # 辅助函数
│
├── examples/                         # 示例代码
├── tests/                            # 单元测试
├── docs/                             # 文档
├── setup.py                          # 安装配置
├── pyproject.toml                    # 项目配置
└── requirements.txt                  # 依赖
```

---

## 🔧 添加新ADC类型 (详细步骤)

### 场景: 添加 Delta-Sigma ADC

#### 步骤 1: 创建ADC文件

**文件位置**: `quantiamagica/adc/delta_sigma.py`

```python
"""
Delta-Sigma ADC Implementation.
文件: quantiamagica/adc/delta_sigma.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# 必须导入的基类和事件系统
from ..core.events import Event, Cancellable, EventPriority
from ..core.base import ADConverter


# =========================================================================
# 第一部分: 定义事件 (每个关键操作一个事件)
# =========================================================================

@dataclass
class IntegratorEvent(Event, Cancellable):
    """
    积分器事件 - 每次积分器更新时触发。
    
    Attributes
    ----------
    stage : int
        积分器级数 (0, 1, 2, ...)
    input_voltage : float
        积分器输入电压 (可修改)
    output_voltage : float
        积分器输出电压 (可修改)
    leakage : float
        积分器泄漏系数 (可修改)
    """
    stage: int = 0
    input_voltage: float = 0.0
    output_voltage: float = 0.0
    leakage: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class QuantizerEvent(Event, Cancellable):
    """量化器事件。"""
    input_voltage: float = 0.0
    output_bit: int = 0
    threshold: float = 0.0
    hysteresis: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class FeedbackDACEvent(Event, Cancellable):
    """反馈DAC事件。"""
    bit_value: int = 0
    dac_output: float = 0.0
    mismatch: float = 1.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class DecimatorEvent(Event):
    """抽取滤波器事件。"""
    input_bitstream: List[int] = field(default_factory=list)
    output_code: int = 0
    osr: int = 64


# =========================================================================
# 第二部分: 实现ADC类
# =========================================================================

class DeltaSigmaADC(ADConverter):
    """
    Delta-Sigma ADC 实现。
    
    Parameters
    ----------
    bits : int
        输出分辨率。
    osr : int
        过采样率。
    order : int
        调制器阶数 (1, 2, 3)。
    vref : float
        参考电压。
    
    Example
    -------
    >>> adc = DeltaSigmaADC(bits=16, osr=128, order=2)
    >>> result = adc.sim()
    """
    
    def __init__(
        self,
        bits: int = 16,
        osr: int = 64,
        order: int = 2,
        vref: float = 1.0,
        vmin: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(bits, vref, vmin, name or "DeltaSigma-ADC")
        
        self.osr = osr
        self.order = order
        
        # 积分器状态
        self._integrators = [0.0] * order
        self._bitstream: List[int] = []
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        """
        执行单次Delta-Sigma转换。
        
        这是核心方法 - 必须实现。
        """
        self._bitstream = []
        self._integrators = [0.0] * self.order
        
        # 过采样循环
        for _ in range(self.osr):
            x = voltage
            
            # 积分器链
            for stage in range(self.order):
                # 触发积分器事件
                int_event = IntegratorEvent(
                    timestamp=timestamp,
                    stage=stage,
                    input_voltage=x,
                    output_voltage=self._integrators[stage],
                )
                self.fire(int_event)
                
                if not int_event.cancelled:
                    # 应用泄漏
                    self._integrators[stage] *= (1 - int_event.leakage)
                    self._integrators[stage] += int_event.input_voltage
                    x = self._integrators[stage]
            
            # 量化器
            quant_event = QuantizerEvent(
                timestamp=timestamp,
                input_voltage=x,
                threshold=0.0,
            )
            quant_event.output_bit = 1 if x >= quant_event.threshold else 0
            self.fire(quant_event)
            
            bit = quant_event.output_bit
            self._bitstream.append(bit)
            
            # 反馈DAC
            dac_event = FeedbackDACEvent(
                timestamp=timestamp,
                bit_value=bit,
            )
            dac_event.dac_output = (self.vref if bit == 1 else self.vmin) * dac_event.mismatch
            self.fire(dac_event)
            
            # 从第一个积分器减去DAC输出
            self._integrators[0] -= dac_event.dac_output
        
        # 抽取滤波
        dec_event = DecimatorEvent(
            timestamp=timestamp,
            input_bitstream=self._bitstream,
            osr=self.osr,
        )
        
        # 简单累加抽取
        dec_event.output_code = int(sum(self._bitstream) * (2**self.bits - 1) / self.osr)
        self.fire(dec_event)
        
        return dec_event.output_code
```

#### 步骤 2: 更新 adc/__init__.py

**文件**: `quantiamagica/adc/__init__.py`

```python
# 在文件开头添加导入
from .delta_sigma import (
    DeltaSigmaADC,
    IntegratorEvent,
    QuantizerEvent,
    FeedbackDACEvent,
    DecimatorEvent,
)

# 在 __all__ 列表中添加
__all__ = [
    # ... 现有导出 ...
    # Delta-Sigma ADC
    "DeltaSigmaADC",
    "IntegratorEvent",
    "QuantizerEvent",
    "FeedbackDACEvent",
    "DecimatorEvent",
]
```

#### 步骤 3: 更新主 __init__.py

**文件**: `quantiamagica/__init__.py`

```python
# 在导入部分添加
from .adc.delta_sigma import (
    DeltaSigmaADC,
    IntegratorEvent,
    QuantizerEvent,
    FeedbackDACEvent,
    DecimatorEvent,
)

# 在 __all__ 中添加
__all__ = [
    # ... 现有导出 ...
    # Delta-Sigma ADC
    "DeltaSigmaADC",
    "IntegratorEvent",
    "QuantizerEvent",
    "FeedbackDACEvent",
    "DecimatorEvent",
]
```

#### 步骤 4: 添加测试

**文件**: `tests/test_delta_sigma.py`

```python
import pytest
from quantiamagica import DeltaSigmaADC, IntegratorEvent

class TestDeltaSigmaADC:
    def test_creation(self):
        adc = DeltaSigmaADC(bits=16, osr=64, order=2)
        assert adc.bits == 16
        assert adc.osr == 64
    
    def test_simulation(self):
        adc = DeltaSigmaADC(bits=16, osr=64)
        result = adc.sim(n_samples=100)
        assert len(result.output_codes) == 100
    
    def test_event_handling(self):
        adc = DeltaSigmaADC(bits=16)
        
        @adc.on(IntegratorEvent)
        def add_leakage(event):
            event.leakage = 0.001
        
        result = adc.sim(n_samples=10)
        assert len(result.output_codes) == 10
```

#### 步骤 5: 添加示例

**文件**: `examples/07_delta_sigma.py`

```python
"""Delta-Sigma ADC 示例"""
from quantiamagica import DeltaSigmaADC, IntegratorEvent

adc = DeltaSigmaADC(bits=16, osr=128, order=2)

@adc.on(IntegratorEvent)
def model_leakage(event):
    event.leakage = 0.0001

adc.sim()
adc.plot()
print(f"ENOB: {adc.enob():.2f}")
```

#### 步骤 6: 更新文档

在 `docs/API_REFERENCE.md` 添加 Delta-Sigma ADC 章节。

---

## 🔗 Pipeline ADC 是否需要修改?

### 情况分析

| 场景 | 是否需要修改Pipeline | 说明 |
|------|---------------------|------|
| 添加独立新ADC (如Delta-Sigma) | ❌ 不需要 | 新ADC是独立的，不影响Pipeline |
| 新ADC可作为Pipeline级 | ✅ 可选 | 如果想让新ADC可用于Pipeline级，需确保兼容 |
| 修改Pipeline行为 | ✅ 需要 | 直接修改 `pipeline.py` |

### 让新ADC兼容Pipeline

如果希望新ADC可以作为Pipeline的一个级：

1. **确保继承自ADConverter**
2. **实现 `_convert_single` 方法**
3. **测试与 `PipelineADC.from_stages()` 的兼容性**

```python
# 测试兼容性
from quantiamagica import PipelineADC, SARADC, DeltaSigmaADC

# 混合架构Pipeline
stage1 = SARADC(bits=4)
stage2 = DeltaSigmaADC(bits=8, osr=16)  # 新ADC作为第二级
pipeline = PipelineADC.from_stages([stage1, stage2])
```

---

## 📝 添加新事件的检查清单

```
□ 1. 事件类继承自 Event
□ 2. 如果可取消，混入 Cancellable 并调用 __post_init__
□ 3. 使用 @dataclass 装饰器
□ 4. 所有属性有默认值
□ 5. 文档字符串说明每个属性
□ 6. 在ADC的 _convert_single 中调用 self.fire(event)
□ 7. 检查 event.cancelled 状态 (如果是Cancellable)
□ 8. 使用修改后的 event 属性值
□ 9. 导出到 adc/__init__.py
□ 10. 导出到主 __init__.py
□ 11. 添加测试用例
□ 12. 更新文档
```

---

## 📊 添加新信号类型

### 在 signals/__init__.py 中添加

```python
@classmethod
def my_new_signal(
    cls,
    n: int = 1024,
    fs: float = 1e6,
    # 其他参数...
) -> "Signal":
    """
    生成新信号类型。
    
    Parameters
    ----------
    n : int
        样本数。
    fs : float
        采样率。
    
    Returns
    -------
    Signal
        信号对象。
    """
    t = np.arange(n) / fs
    data = # 计算信号数据
    
    return cls(
        data=data,
        fs=fs,
        name="MyNewSignal",
        timestamps=t,
        metadata={"type": "my_new_signal"},
    )
```

---

## 🎨 添加新绘图函数

### 在 plotting/__init__.py 中添加

```python
def plot_xxx_jssc(
    data: NDArray,
    *,
    columns: int = 1,
    show: bool = True,
    save: Optional[str] = None,
) -> Any:
    """
    绘制符合 JSSC 规范的 XXX 图。
    """
    with jssc_style():
        fig, ax = create_figure(columns=columns)
        
        # 绑图逻辑
        ax.plot(...)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


# 添加到 __all__
__all__ = [
    # ... 现有 ...
    "plot_xxx_jssc",
]
```

然后更新主 `__init__.py` 导出。

---

## 🧪 测试规范

### 测试文件命名

- `test_sar.py` - SAR ADC测试
- `test_pipeline.py` - Pipeline ADC测试
- `test_delta_sigma.py` - Delta-Sigma ADC测试
- `test_events.py` - 事件系统测试
- `test_analysis.py` - 分析工具测试
- `test_signals.py` - 信号类测试

### 运行测试

```bash
# 所有测试
pytest tests/

# 特定文件
pytest tests/test_sar.py

# 详细输出
pytest -v tests/

# 覆盖率
pytest --cov=quantiamagica tests/
```

---

## 🔄 版本更新流程

1. **修改代码**
2. **运行测试**: `pytest tests/`
3. **更新版本号**: `quantiamagica/__init__.py` 中的 `__version__`
4. **更新文档**
5. **提交**: `git commit -m "feat: 添加XXX功能"`

---

## ⚠️ 常见错误和解决方案

### 1. 导入错误

**问题**: `ImportError: cannot import name 'XXX'`

**解决**: 检查是否在以下位置都添加了导出:
- `quantiamagica/adc/__init__.py`
- `quantiamagica/__init__.py`

### 2. 事件未触发

**问题**: 事件处理器没有被调用

**解决**: 
- 确保调用了 `self.fire(event)`
- 检查事件类型是否正确

### 3. Cancellable 错误

**问题**: `AttributeError: 'XXXEvent' object has no attribute 'cancelled'`

**解决**: 在事件类的 `__post_init__` 中调用 `Cancellable.__init__(self)`

---

## 📋 AI助手快速参考

当需要添加新ADC时，请按以下顺序修改文件:

1. `quantiamagica/adc/new_adc.py` - 创建新文件
2. `quantiamagica/adc/__init__.py` - 添加导入和__all__
3. `quantiamagica/__init__.py` - 添加导入和__all__
4. `tests/test_new_adc.py` - 创建测试
5. `examples/XX_new_adc.py` - 创建示例
6. `docs/API_REFERENCE.md` - 更新文档

需要检查的关键点:
- 新ADC必须继承 `ADConverter`
- 必须实现 `_convert_single(self, voltage, timestamp) -> int`
- 事件必须继承 `Event`，可选混入 `Cancellable`
- 所有公共类和函数必须在 `__all__` 中导出
