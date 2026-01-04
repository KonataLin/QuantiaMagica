"""
Example 07: Signal 类使用示例
============================

展示如何使用 Signal 类生成各种测试信号。

Usage:
    python 07_signals.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SARADC, Signal
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基本信号类型
# =============================================================================

print("=" * 60)
print("1. 基本信号类型")
print("=" * 60)

# 正弦波
sine = Signal.sine(n=1024, fs=1e6, freq=10e3, amplitude=0.4, offset=0.5)
print(f"正弦波: {sine}")

# 方波
square = Signal.square(n=1024, fs=1e6, freq=1e3, duty=0.5)
print(f"方波: {square}")

# 三角波
triangle = Signal.triangle(n=1024, fs=1e6, freq=1e3)
print(f"三角波: {triangle}")

# 锯齿波
sawtooth = Signal.sawtooth(n=1024, fs=1e6, freq=1e3, rising=True)
print(f"锯齿波: {sawtooth}")


# =============================================================================
# 2. 特殊测试信号
# =============================================================================

print("\n" + "=" * 60)
print("2. 特殊测试信号")
print("=" * 60)

# 斜坡（用于INL/DNL测试）
ramp = Signal.ramp(n=4096, vmin=0.0, vmax=1.0)
print(f"斜坡: {ramp}")

# 阶跃
step = Signal.step(n=1024, v_low=0.2, v_high=0.8, step_point=0.5)
print(f"阶跃: {step}")

# 脉冲
pulse = Signal.pulse(n=1024, v_base=0.2, v_pulse=0.8, pulse_start=0.3, pulse_width=0.2)
print(f"脉冲: {pulse}")


# =============================================================================
# 3. 多音信号（用于IMD测试）
# =============================================================================

print("\n" + "=" * 60)
print("3. 多音信号")
print("=" * 60)

# 双音
two_tone = Signal.two_tone(n=2048, f1=10e3, f2=11e3, amplitude=0.2)
print(f"双音: {two_tone}")

# 多音
multitone = Signal.multitone(
    n=2048,
    frequencies=[1e3, 2e3, 5e3, 10e3],
    amplitudes=[0.1, 0.1, 0.1, 0.1],
)
print(f"多音: {multitone}")


# =============================================================================
# 4. 调制信号
# =============================================================================

print("\n" + "=" * 60)
print("4. 调制信号")
print("=" * 60)

# 扫频（Chirp）
chirp = Signal.chirp(n=2048, f_start=1e3, f_end=100e3, method='linear')
print(f"扫频: {chirp}")

# 调幅（AM）
am = Signal.am(n=2048, carrier_freq=100e3, mod_freq=1e3, mod_depth=0.5)
print(f"调幅: {am}")

# 调频（FM）
fm = Signal.fm(n=2048, carrier_freq=100e3, mod_freq=1e3, freq_deviation=10e3)
print(f"调频: {fm}")


# =============================================================================
# 5. 噪声信号
# =============================================================================

print("\n" + "=" * 60)
print("5. 噪声信号")
print("=" * 60)

# 高斯噪声
gaussian = Signal.noise_gaussian(n=1024, mean=0.5, sigma=0.1)
print(f"高斯噪声: {gaussian}")

# 均匀噪声
uniform = Signal.noise_uniform(n=1024, vmin=0.3, vmax=0.7)
print(f"均匀噪声: {uniform}")


# =============================================================================
# 6. 信号处理
# =============================================================================

print("\n" + "=" * 60)
print("6. 信号处理")
print("=" * 60)

sig = Signal.sine(n=1024, freq=10e3, amplitude=0.3, offset=0.5)

# 添加噪声
noisy = sig.add_noise(sigma=0.01)
print(f"添加噪声: {noisy}")

# 添加偏置
offset_sig = sig.add_offset(0.1)
print(f"添加偏置: {offset_sig}")

# 缩放
scaled = sig.scale(2.0)
print(f"缩放: {scaled}")

# 裁剪
clipped = sig.clip(vmin=0.3, vmax=0.7)
print(f"裁剪: {clipped}")

# 归一化
normalized = sig.normalize(vmin=0.0, vmax=1.0)
print(f"归一化: {normalized}")


# =============================================================================
# 7. 与ADC一起使用
# =============================================================================

print("\n" + "=" * 60)
print("7. 与ADC一起使用")
print("=" * 60)

adc = SARADC(bits=12, vref=1.0)

# 使用正弦波
result = adc.sim(Signal.sine(n=2048, freq=10e3))
print(f"正弦波 ENOB: {adc.enob():.2f} bits")

# 使用双音
result = adc.sim(Signal.two_tone(f1=10e3, f2=11e3))
print(f"双音 ENOB: {adc.enob():.2f} bits")

# 使用斜坡（INL/DNL测试）
result = adc.sim(Signal.ramp(n=4096))
print(f"INL max: {max(abs(adc.inl())):.3f} LSB")
print(f"DNL max: {max(abs(adc.dnl())):.3f} LSB")


# =============================================================================
# 8. 可视化
# =============================================================================

print("\n" + "=" * 60)
print("8. 可视化")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=150)
fig.suptitle('Signal 类信号类型展示', fontsize=14, fontweight='bold')

signals = [
    (Signal.sine(n=500, freq=5e3), "正弦波"),
    (Signal.square(n=500, freq=2e3), "方波"),
    (Signal.triangle(n=500, freq=2e3), "三角波"),
    (Signal.sawtooth(n=500, freq=2e3), "锯齿波"),
    (Signal.ramp(n=500), "斜坡"),
    (Signal.step(n=500), "阶跃"),
    (Signal.chirp(n=500, f_start=1e3, f_end=50e3), "扫频"),
    (Signal.am(n=500, carrier_freq=20e3, mod_freq=2e3), "调幅"),
    (Signal.noise_gaussian(n=500, sigma=0.1), "高斯噪声"),
]

for ax, (sig, name) in zip(axes.flat, signals):
    ax.plot(sig.timestamps * 1e6, sig.data, 'b-', linewidth=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('Time (μs)', fontsize=8)
    ax.set_ylabel('Voltage (V)', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('signals_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图片已保存至 signals_demo.png")
