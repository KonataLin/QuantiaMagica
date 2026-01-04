# QuantiaMagica

<p align="center">
  <strong>ADC 行为级事件驱动仿真器</strong><br>
  <em>事件系统的 Python ADC 建模框架</em>
</p>

<p align="center">
  <a href="#安装">安装</a> |
  <a href="#快速开始">快速开始</a> |
  <a href="#特性">特性</a> |
  <a href="#文档">文档</a> |
  <a href="#示例">示例</a>
</p>

---

## 特性

- **事件驱动架构**: 像 Minecraft Bukkit 插件一样，在 ADC 转换的每个阶段注入自定义行为
- **简洁 API**: 3 行代码完成仿真、绘图、分析
- **高度可扩展**: 通过事件处理器轻松建模任意非理想效应
- **内置分析工具**: ENOB, SNR, SFDR, THD, INL, DNL 等指标及 IEEE JSSC 风格绘图
- **多种架构支持**: SAR ADC, Pipeline ADC, Sigma-Delta ADC

## 安装

```bash
# 克隆仓库
git clone https://github.com/KonataLin/QuantiaMagica.git
cd QuantiaMagica

# 开发模式安装
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt
```

## 快速开始

### 3 行代码仿真

```python
from quantiamagica import SARADC

adc = SARADC(bits=12, vref=1.0)
adc.sim(fin=10e3)
adc.plot()
```

### 事件驱动非理想效应建模

```python
from quantiamagica import SARADC, SamplingEvent, CapacitorSwitchEvent
import numpy as np

adc = SARADC(bits=12, vref=1.0)

# 添加热噪声
@adc.on(SamplingEvent)
def add_noise(event):
    event.voltage += np.random.normal(0, 100e-6)

# 添加电容失配 (固定失配值，在初始化时生成)
cap_mismatch_values = {i: np.random.normal(0, 0.005) for i in range(12)}

@adc.on(CapacitorSwitchEvent)
def cap_mismatch(event):
    event.capacitance_actual *= 1 + cap_mismatch_values[event.bit_index]

adc.sim()
print(f"ENOB: {adc.enob():.2f} bits")
```

### 获取所有指标

```python
from quantiamagica import SARADC
from quantiamagica.analysis import Analyzer

adc = SARADC(bits=12)
result = adc.sim()
print(Analyzer(result).summary())
```

输出:
```
╔══════════════════════════════════════════╗
║       ADC Performance Summary            ║
╠══════════════════════════════════════════╣
║  Resolution:     12 bits                 ║
║  Sample Rate:    1.00 MHz                ║
╠══════════════════════════════════════════╣
║  Dynamic Performance:                    ║
║    ENOB:         11.98 bits              ║
║    SNR:          73.90 dB                ║
║    SFDR:         89.50 dB                ║
║    THD:         -85.20 dB                ║
╚══════════════════════════════════════════╝
```

## 文档

### 核心概念

#### 事件系统
事件在 ADC 转换的每个阶段触发，你可以监听并修改行为:

| 事件 | 说明 | 可修改属性 |
|-------|-------------|----------------------|
| `SamplingEvent` | 采样 | `voltage`, `sampling_capacitance` |
| `CapacitorSwitchEvent` | DAC电容切换 | `capacitance_actual`, `weight`, `charge_injection` |
| `ComparatorEvent` | 比较器判决 | `offset`, `noise_sigma`, `decision` |
| `BitDecisionEvent` | 位判决后 | `bit_value`, `residue` |
| `OutputCodeEvent` | 最终输出 | `code` |

#### 事件优先级
处理器按优先级顺序执行:

```python
from quantiamagica import EventPriority

@adc.on(SamplingEvent, priority=EventPriority.HIGHEST)
def runs_first(event):
    pass

@adc.on(SamplingEvent, priority=EventPriority.MONITOR)
def runs_last_readonly(event):
    pass
```

### ADC 类型

#### SAR ADC
```python
from quantiamagica import SARADC

adc = SARADC(
    bits=12,           # 分辨率
    vref=1.0,          # 参考电压
    vmin=0.0,          # 最小输入
    cap_unit=50.0,     # 单位电容 (fF)
    comparator_noise=0.1e-3,  # 比较器噪声 (V)
)
```

#### Pipeline ADC
```python
from quantiamagica import PipelineADC

pipeline = PipelineADC(
    bits=14,           # 总分辨率
    stages=4,          # 级数
    bits_per_stage=4,  # 每级位数
    redundancy=1,      # 冗余位
)
```

### 分析方法

```python
adc.sim()           # 运行仿真
adc.sim_auto(fs)    # 自动优化fin和幅度，最大化ENOB
adc.plot()          # 绘制时域图 (IEEE JSSC 黑白风格)
adc.spectrum()      # 绘制频谱 (IEEE JSSC 黑白风格)

# 指标
adc.enob()          # 有效位数
adc.snr()           # 信噪比 (dB)
adc.sfdr()          # 无杂散动态范围 (dB)
adc.thd()           # 总谐波失真 (dB)
adc.inl()           # 积分非线性 (LSB数组)
adc.dnl()           # 微分非线性 (LSB数组)
```

### 自动优化 (sim_auto)

使用**差分进化(DE)算法**自动搜索最佳测试参数，全自动收敛检测:

```python
from quantiamagica import SARADC

adc = SARADC(bits=12)

# 一行代码 - 自动优化！
result = adc.sim_auto(fs=1e6)

# 返回值
print(f"最佳fin: {result['best_fin']:.2f} Hz")
print(f"最佳幅度: {result['best_amplitude']:.4f} V") 
print(f"最佳ENOB: {result['best_enob']:.4f} bits")
print(f"收敛: {result['converged']}, 原因: {result['reason']}")

# 结果已保存，可直接画图
adc.report()
```

特点:
- **GPU加速**: 自动检测CUDA，GPU可用时种群128
- **极限并发**: 4x CPU核心数并行计算
- **快速收敛**: 通常2-4代达到理论ENOB极限
- **幅度优化**: 自动使用99.8%满量程获得最佳ENOB

### IEEE JSSC 风格绘图

所有图表采用 IEEE JSSC 期刊发表风格:
- **黑白配色**
- **衬线字体** (Times New Roman)
- **300 DPI** 发表质量
- **单栏/双栏** 尺寸 (3.5" / 7.16")

```python
from quantiamagica.plotting import plot_spectrum_jssc, jssc_style

# 使用 JSSC 风格上下文
with jssc_style():
    adc.plot()
    adc.spectrum(save="spectrum.pdf")

# 或使用专用 JSSC 绘图函数
freqs, spec_db, metrics = adc.spectrum(show=False)
plot_spectrum_jssc(freqs, spec_db, metrics, save="figure.pdf", columns=1)
```

### 数据导出

```python
result = adc.sim()
result.save("data.npz")              # NumPy 格式
result.save("data.csv", format="csv") # CSV 格式
result.save("data.json", format="json") # JSON 格式
```

## 示例

| 示例 | 说明 |
|---------|-------------|
| [01_basic_sar.py](examples/01_basic_sar.py) | 基础 SAR ADC 仿真 |
| [02_event_handling.py](examples/02_event_handling.py) | Bukkit 风格事件处理 |
| [03_ns_sar.py](examples/03_ns_sar.py) | **NS-SAR 过采样** - 带内 ENOB 提升 4.6+ bits |
| [04_pipeline_adc.py](examples/04_pipeline_adc.py) | 流水线 ADC |
| [05_advanced_nonidealities.py](examples/05_advanced_nonidealities.py) | 蒙特卡洛分析 |
| [06_quick_start.py](examples/06_quick_start.py) | 最简代码示例 |
| [07_signals.py](examples/07_signals.py) | 各种信号类型 |
| [08_sigma_delta.py](examples/08_sigma_delta.py) | **Sigma-Delta ADC** 自定义拓扑 |

## 项目结构

```
QuantiaMagica/
├── quantiamagica/
│   ├── __init__.py          # 主导出
│   ├── core/
│   │   ├── events.py        # 事件系统
│   │   └── base.py          # ADConverter 基类
│   ├── adc/
│   │   ├── sar.py           # SAR ADC 实现
│   │   ├── pipeline.py      # Pipeline ADC 实现
│   │   └── sigma_delta.py   # Sigma-Delta ADC 实现
│   ├── analysis/
│   │   └── __init__.py      # 分析工具
│   └── utils/
│       └── __init__.py      # 工具函数
├── examples/                 # 示例脚本
├── docs/                     # 文档
├── tests/                    # 单元测试
├── setup.py                  # 包配置
├── requirements.txt          # 依赖
└── README.md                 # 本文件
```

## 扩展 QuantiaMagica

### 创建自定义 ADC 类型

```python
from quantiamagica import ADConverter

class MyCustomADC(ADConverter):
    def __init__(self, bits, osr=64, **kwargs):
        super().__init__(bits, **kwargs)
        self.osr = osr
    
    def _convert_single(self, voltage, timestamp):
        # 你的实现
        # 根据需要触发事件
        return code
```

### Sigma-Delta ADC 自定义拓扑 (2阶/3阶)

```python
from quantiamagica import SigmaDeltaADC, QuantizerEvent

# ========== 2阶 (1-bit量化器) ==========
state = [0.0, 0.0]
sd2 = SigmaDeltaADC(order=1, bits=1, osr=64)

@sd2.on(QuantizerEvent)
def second_order(event):
    x, y = event.input_signal, event.prev_output
    state[0] = state[0] + x - y
    state[1] = state[1] + state[0] - 2*y
    event.quantizer_input = state[1]

# ========== 1-bit 3阶 (缩放积分器) ==========
u3 = [0.0, 0.0, 0.0]
c = 0.3  # 积分器缩放系数
sd3 = SigmaDeltaADC(order=1, bits=1, osr=64)

@sd3.on(QuantizerEvent)
def third_order(event):
    x, y = event.input_signal, event.prev_output
    u3[0] = u3[0] + c*(x - y)
    u3[1] = u3[1] + c*(u3[0] - 2*y)
    u3[2] = u3[2] + c*(u3[1] - y)
    event.quantizer_input = u3[2]
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 贡献

欢迎贡献代码！请阅读 [CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

<p align="center">
  Made by KonataLin | <a href="https://github.com/KonataLin/QuantiaMagica">GitHub</a>
</p>
