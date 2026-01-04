# QuantiaMagica 代码维护指南

本文档为开发者和维护者提供代码架构说明和扩展指南。

---

## 目录

1. [项目架构](#项目架构)
2. [核心设计模式](#核心设计模式)
3. [添加新ADC类型](#添加新adc类型)
4. [添加新事件](#添加新事件)
5. [扩展分析功能](#扩展分析功能)
6. [代码规范](#代码规范)
7. [测试指南](#测试指南)

---

## 项目架构

```
quantiamagica/
├── __init__.py              # 公共API导出
├── _analysis_impl.py        # 内部分析实现
├── core/
│   ├── __init__.py
│   ├── events.py            # 事件系统核心
│   └── base.py              # ADConverter基类
├── adc/
│   ├── __init__.py
│   ├── sar.py               # SAR ADC实现
│   ├── pipeline.py          # Pipeline ADC实现
│   └── sigma_delta.py       # Sigma-Delta ADC实现
├── analysis/
│   └── __init__.py          # 分析工具
└── utils/
    └── __init__.py          # 工具函数
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `core/events.py` | 事件基类、EventBus、优先级系统 |
| `core/base.py` | ADConverter抽象基类、SimulationResult |
| `adc/sar.py` | SAR ADC及其所有事件定义 |
| `adc/pipeline.py` | Pipeline ADC及其事件定义 |
| `adc/sigma_delta.py` | Sigma-Delta ADC，只有QuantizerEvent，通过事件实现任意拓扑 |
| `analysis/` | 独立分析函数和Analyzer类 |
| `utils/` | 信号生成、辅助函数 |

---

## 核心设计模式

### 1. 事件驱动架构 (Event-Driven Architecture)

灵感来自 Minecraft Bukkit API：

```python
# 事件定义
@dataclass
class MyEvent(Event, Cancellable):
    data: float = 0.0

# 事件触发
event = MyEvent(data=value)
self.fire(event)  # 所有注册的handler会被调用

# 使用修改后的值
result = event.data
```

**关键点：**
- 事件在ADC操作的每个关键点触发
- 用户通过handler修改事件属性来改变行为
- `Cancellable`混入类允许取消事件

### 2. 模板方法模式 (Template Method Pattern)

`ADConverter`基类定义了仿真流程，子类只需实现`_convert_single`：

```python
class ADConverter(ABC):
    def sim(self, ...):
        # 公共逻辑：生成信号、循环采样
        for i, v in enumerate(input_signal):
            output_codes[i] = self._convert_single(v, timestamp)
        # 公共逻辑：计算重建信号、打包结果
    
    @abstractmethod
    def _convert_single(self, voltage, timestamp) -> int:
        """子类必须实现"""
        pass
```

### 3. 插件系统 (Plugin System)

使用`@on_event`装饰器和`use()`方法：

```python
class MyPlugin:
    @on_event(SamplingEvent)
    def handle(self, event):
        event.voltage += self.offset

adc.use(MyPlugin())  # 自动注册所有handler
```

---

## 添加新ADC类型

### 步骤1：创建新文件

```python
# quantiamagica/adc/delta_sigma.py

from dataclasses import dataclass
from ..core.events import Event, Cancellable
from ..core.base import ADConverter

# 定义事件
@dataclass
class IntegratorEvent(Event, Cancellable):
    stage: int = 0
    input_voltage: float = 0.0
    integrator_output: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)

@dataclass 
class QuantizerEvent(Event):
    input_voltage: float = 0.0
    output_bit: int = 0

# 实现ADC
class DeltaSigmaADC(ADConverter):
    def __init__(
        self,
        bits: int,
        osr: int = 64,
        order: int = 2,
        **kwargs
    ):
        super().__init__(bits, **kwargs)
        self.osr = osr
        self.order = order
        self._integrators = [0.0] * order
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        # 实现Delta-Sigma调制
        # 在关键点触发事件
        for stage in range(self.order):
            event = IntegratorEvent(
                timestamp=timestamp,
                stage=stage,
                input_voltage=...,
            )
            self.fire(event)
            # 使用event修改后的值
        
        return final_code
```

### 步骤2：更新导出

```python
# quantiamagica/adc/__init__.py
from .delta_sigma import DeltaSigmaADC, IntegratorEvent, QuantizerEvent

# quantiamagica/__init__.py  
from .adc.delta_sigma import DeltaSigmaADC, IntegratorEvent, QuantizerEvent
```

### 步骤3：添加测试和示例

---

## 添加新事件

### 事件设计原则

1. **使用`@dataclass`**：简化定义
2. **继承`Event`**：基础事件功能
3. **可选`Cancellable`**：如果操作可以被跳过
4. **有意义的默认值**：减少用户代码
5. **文档字符串**：说明每个属性的用途

### 事件模板

```python
@dataclass
class NewEvent(Event, Cancellable):
    """
    简短描述。
    
    Attributes
    ----------
    property1 : type
        说明 (modifiable).
    property2 : type
        说明.
    
    Example
    -------
    >>> @adc.on(NewEvent)
    ... def handler(event):
    ...     event.property1 = new_value
    """
    property1: float = 0.0
    property2: int = 0
    
    def __post_init__(self):
        Cancellable.__init__(self)
```

### 事件触发位置

选择在ADC操作的**关键决策点**触发事件：

| 位置 | 示例事件 |
|------|----------|
| 采样开始 | `SamplingEvent` |
| DAC操作 | `CapacitorSwitchEvent` |
| 比较/量化 | `ComparatorEvent` |
| 位决策后 | `BitDecisionEvent` |
| 输出完成 | `OutputCodeEvent` |

---

## 扩展分析功能

### 添加新指标

```python
# quantiamagica/_analysis_impl.py

def compute_new_metric(codes, bits, fs):
    """
    计算新指标。
    
    Parameters
    ----------
    codes : NDArray
        输出码。
    bits : int
        分辨率。
    fs : float
        采样率。
    
    Returns
    -------
    float
        指标值。
    """
    # 实现
    return value
```

```python
# quantiamagica/analysis/__init__.py

from .._analysis_impl import compute_new_metric

def new_metric(codes, bits, fs=1e6):
    return compute_new_metric(codes, bits, fs)
```

```python
# quantiamagica/core/base.py - 添加到ADConverter类

def new_metric(self, result=None):
    """计算新指标。"""
    from .._analysis_impl import compute_new_metric
    if result is None:
        result = self._result
    return compute_new_metric(result.output_codes, self.bits, ...)
```

---

## 代码规范

### Python风格

- 遵循PEP 8
- 使用类型注解
- 文档字符串使用NumPy格式

### 文档字符串示例

```python
def function(param1: int, param2: float = 1.0) -> str:
    """
    简短描述（一行）。
    
    详细描述（可选，可多行）。
    
    Parameters
    ----------
    param1 : int
        参数1的描述。
    param2 : float, optional
        参数2的描述。默认值：1.0
    
    Returns
    -------
    str
        返回值描述。
    
    Raises
    ------
    ValueError
        何时抛出此异常。
    
    Example
    -------
    >>> result = function(42, 3.14)
    >>> print(result)
    'example output'
    
    See Also
    --------
    related_function : 相关函数描述。
    
    Notes
    -----
    额外说明（可选）。
    """
    pass
```

### 类型注解

```python
from typing import Optional, List, Dict, Tuple, Union, Callable, Any
from numpy.typing import NDArray
import numpy as np

def example(
    codes: NDArray[np.int64],
    callback: Callable[[Event], None],
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[NDArray, Dict[str, float]]:
    pass
```

### 导入顺序

```python
# 1. 标准库
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

# 2. 第三方库
import numpy as np
import matplotlib.pyplot as plt

# 3. 本地模块
from ..core.events import Event
from ..core.base import ADConverter
```

---

## 测试指南

### 测试文件结构

```
tests/
├── __init__.py
├── test_events.py
├── test_sar.py
├── test_pipeline.py
├── test_analysis.py
└── conftest.py
```

### 测试示例

```python
# tests/test_sar.py

import pytest
import numpy as np
from quantiamagica import SARADC, SamplingEvent

class TestSARADC:
    def test_basic_conversion(self):
        adc = SARADC(bits=10, vref=1.0)
        result = adc.sim(input_voltage=0.5)
        
        # 检查输出码在合理范围
        expected_code = 512  # 0.5V / 1.0V * 1024
        assert abs(result.output_codes[0] - expected_code) <= 1
    
    def test_event_modification(self):
        adc = SARADC(bits=10, vref=1.0)
        
        offset = 0.1
        @adc.on(SamplingEvent)
        def add_offset(event):
            event.voltage += offset
        
        result = adc.sim(input_voltage=0.4)
        # 应该转换 0.4 + 0.1 = 0.5V
        expected_code = 512
        assert abs(result.output_codes[0] - expected_code) <= 1
    
    def test_enob_ideal(self):
        adc = SARADC(bits=12, vref=1.0)
        adc.sim(n_samples=4096)
        
        # 理想ADC的ENOB应该接近标称位数
        assert adc.enob() > 11.5

@pytest.fixture
def sample_adc():
    return SARADC(bits=10, vref=1.0)
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_sar.py

# 显示详细输出
pytest -v tests/

# 生成覆盖率报告
pytest --cov=quantiamagica tests/
```

---

## 版本控制

### 语义版本

- **主版本** (1.x.x)：不兼容的API变更
- **次版本** (x.1.x)：向后兼容的功能添加
- **补丁版本** (x.x.1)：向后兼容的bug修复

### 更新版本

```python
# quantiamagica/__init__.py
__version__ = "1.0.0"
```

---

## 常见扩展场景

### 1. 添加新的非理想性

```python
@adc.on(CapacitorSwitchEvent)
def my_nonideality(event):
    # 修改事件属性
    event.capacitance_actual *= factor
```

### 2. 创建可复用插件

```python
class MyPlugin:
    def __init__(self, config):
        self.config = config
    
    def attach(self, adc):
        @adc.on(SamplingEvent)
        def handler(event):
            # 使用 self.config
            pass
```

### 3. 自定义绘图

```python
def custom_plot(result, adc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # 自定义绘图逻辑
    return fig
```

---

## 性能优化提示

1. **向量化操作**：尽可能使用NumPy向量操作
2. **避免频繁事件**：对于性能关键应用，考虑禁用事件日志
3. **预计算**：将不变的计算移到`__init__`
4. **JIT编译**：对于计算密集型代码，考虑使用Numba

```python
# 示例：使用Numba加速
from numba import jit

@jit(nopython=True)
def fast_computation(array):
    # 快速计算
    return result
```

---

## 已实现的高级功能

### 1. NS-SAR (噪声整形SAR) 与过采样

NS-SAR通过噪声整形将量化噪声推向高频，配合过采样可大幅提升带内ENOB。

**原理：**
- 噪声传递函数: NTF = (1 - z⁻¹)
- 理论提升: 0.5×log₂(OSR) + 1.5 bits (一阶整形)

**实现要点：**
```python
class NoiseShapingSAR:
    def __init__(self, feedback_coeff=1.0):
        self.feedback_coeff = feedback_coeff
        self.prev_residue = 0.0
        self.sampled_voltage = 0.0
    
    def attach(self, adc):
        @adc.on(SamplingEvent, priority=EventPriority.HIGH)
        def apply_feedback(event):
            event.voltage += self.feedback_coeff * self.prev_residue
            self.sampled_voltage = event.voltage  # 保存反馈后的电压
        
        @adc.on(OutputCodeEvent, priority=EventPriority.HIGH)
        def capture_residue(event):
            # 残差 = 采样电压(含反馈) - DAC输出
            reconstructed = adc.code_to_voltage(event.code)
            self.prev_residue = self.sampled_voltage - reconstructed
```

**关键：** 残差计算必须使用加了反馈后的采样电压，而不是原始输入电压。

**测试结果 (8-bit ADC, OSR=64)：**
| 指标 | Standard SAR | NS-SAR | 提升 |
|------|-------------|--------|------|
| 全带宽 ENOB | 7.67 bits | 7.19 bits | -0.48 |
| 带内 ENOB | 10.64 bits | 15.26 bits | **+4.62** |

> 注: 全带宽ENOB会因噪声整形而降低，但带内ENOB大幅提升。

### 4. SAR ADC 量化修正 (0.5 LSB偏移)

为获得正确的ENOB，SAR ADC使用居中量化 (round) 而非截断量化 (floor)。

**问题:** 原始SAR比较逻辑 `V >= threshold` 实现的是floor量化，导致ENOB偏低约0.4 bits。

**解决方案:** 在采样电压上加0.5 LSB偏移：
```python
# quantiamagica/adc/sar.py - _convert_single()
sampled_voltage_adjusted = sampled_voltage + 0.5 * self.lsb
```

**修复效果:**
| ADC | 修复前 | 修复后 | 理论值 |
|-----|--------|--------|--------|
| 8-bit | 7.50 | 7.89 | 8.0 |
| 10-bit | 9.36 | 9.95 | 10.0 |
| 12-bit | 11.30 | 12.00 | 12.0 |

### 2. IEEE JSSC 黑白绘图风格

所有绘图函数已更新为IEEE JSSC出版标准：

**样式参数：**
```python
JSSC_STYLE = {
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'lines.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
}
```

**尺寸标准：**
- 单栏: 3.5 英寸
- 双栏: 7.16 英寸
- DPI: 300 (出版质量)

**使用方式：**
```python
# 方法1：直接使用（默认已是JSSC风格）
adc.plot()
adc.spectrum(save="spectrum.pdf")

# 方法2：使用上下文管理器
from quantiamagica.plotting import jssc_style
with jssc_style():
    custom_plotting_code()

# 方法3：专用函数
from quantiamagica.plotting import plot_spectrum_jssc
plot_spectrum_jssc(freqs, spec_db, metrics, columns=1, save="fig.pdf")
```

### 3. 带内ENOB计算

用于过采样系统的带内性能评估：

```python
def compute_inband_snr(codes, bits, fs, signal_freq, bandwidth):
    """
    计算带内SNR (只考虑信号带宽内的噪声).
    
    关键步骤：
    1. FFT后只统计bandwidth范围内的噪声功率
    2. 排除信号bin及其泄漏
    3. 计算 SNR = 10*log10(signal_power / inband_noise_power)
    4. ENOB = (SNR - 1.76) / 6.02
    """
```

### 5. 统一绘图API - report()方法

新增`report()`方法提供简洁且灵活的分析报告：

```python
# 简洁用法
adc.sim().report()                    # 完整报告
adc.sim().report('spectrum')          # 仅频谱
adc.sim().report('metrics')           # 仅指标(无图)
adc.sim().report(save='fig.pdf')      # 保存PDF

# 参数说明
report(
    what='all',      # 'all'|'spectrum'|'time'|'static'|'metrics'
    save=None,       # 保存路径
    show=True,       # 显示图像
    columns=1,       # 1=单栏(3.5"), 2=双栏(7.16")
)
```

### 6. Sigma-Delta ADC使用SAR作为量化器

可以用SAR ADC实现Sigma-Delta调制器，达到接近理论值的ENOB：

```python
# 理论ENOB增益: 1.5 * log2(OSR) bits
# 例如 OSR=64 时，理论增益 = 9.0 bits

from quantiamagica import SARADC

def sigma_delta_1bit(input_signal):
    """1-bit一阶Sigma-Delta调制器 (标准差分方程)."""
    n = len(input_signal)
    output = np.zeros(n)
    u = 0.0      # 积分器状态
    y_prev = 0.0 # 上一个输出
    
    for i in range(n):
        u = u + input_signal[i] - y_prev  # 积分
        y = 1.0 if u >= 0 else -1.0       # 1-bit量化
        output[i] = y
        y_prev = y
    
    return output

# 参见 examples/08_sigma_delta.py
# 实测: OSR=64时 1-bit SD达到 8.88 bits (理论9.0)
```

### 7. 集成绘图函数

新增简化绘图API：

```python
from quantiamagica import plot_spectrum, plot_comparison

# 一键绘制频谱
plot_spectrum(codes, fs=1e6, title="My Spectrum")

# 对比多个ADC
plot_comparison(
    [codes1, codes2],
    fs=1e6,
    labels=['Standard', 'NS-SAR'],
    bandwidth=10e3,  # 高亮带内区域
    save='comparison.pdf'
)
```

---

## 联系与贡献

- 问题反馈：创建GitHub Issue
- 代码贡献：Fork + Pull Request
- 文档改进：欢迎提交文档PR
