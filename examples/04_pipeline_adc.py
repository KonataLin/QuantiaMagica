"""
Example 04: Pipeline ADC - 3个8-bit SAR串联成24-bit
=====================================================

本示例展示新的Pipeline类（不继承ADConverter）的使用方法：
- 使用3个8-bit SAR ADC串联成24-bit高精度Pipeline
- 级间增益放大事件（InterstageGainEvent）用于注入非理想效应
- 验证Pipeline功能是否正确达到24位精度

Usage:
    python 04_pipeline_adc.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import (
    Pipeline,
    SARADC,
    InterstageGainEvent,
    StageEvent,
    ResidueEvent,
    EventPriority,
)


# =============================================================================
# Example 1: 3个8-bit SAR ADC串联成24-bit Pipeline（理想情况）
# =============================================================================

print("=" * 70)
print("Example 1: 3×8-bit SAR → 24-bit Pipeline (理想)")
print("=" * 70)

# 创建3个独立的8-bit SAR ADC
stage1 = SARADC(bits=8, vref=1.0, name="SAR-Stage1")
stage2 = SARADC(bits=8, vref=1.0, name="SAR-Stage2")
stage3 = SARADC(bits=8, vref=1.0, name="SAR-Stage3")

# 级间增益 = 2^8 = 256（每级8位）
# 这样第一级的残差放大256倍后，刚好覆盖第二级的满量程
pipeline_ideal = Pipeline(
    stages=[stage1, stage2, stage3],
    gains=[256.0, 256.0],  # 2个级间增益（3级需要2个）
    name="24-bit-SAR-Pipeline"
)

print(f"\nPipeline配置:")
print(f"  总位数: {pipeline_ideal.bits} bits")
print(f"  级数: {pipeline_ideal.num_stages}")
for info in pipeline_ideal.get_stage_info():
    print(f"    Stage {info['index']}: {info['adc_name']} ({info['bits']} bits), gain={info['gain']:.0f}")

# 运行仿真（65536样本点以达到24-bit精度验证）
result = pipeline_ideal.sim(n_samples=65536, fs=1e6, fin=10e3)

print(f"\n理想Pipeline性能:")
print(f"  ENOB: {pipeline_ideal.enob():.2f} bits")
print(f"  SNR:  {pipeline_ideal.snr():.2f} dB")
print(f"  SFDR: {pipeline_ideal.sfdr():.2f} dB")
print(f"  THD:  {pipeline_ideal.thd():.2f} dB")


# =============================================================================
# Example 2: 添加级间增益误差
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Pipeline with Inter-stage Gain Error")
print("=" * 70)

# 重新创建ADC（每个ADC实例只能用于一个Pipeline）
stage1_err = SARADC(bits=8, vref=1.0, name="SAR-Stage1")
stage2_err = SARADC(bits=8, vref=1.0, name="SAR-Stage2")
stage3_err = SARADC(bits=8, vref=1.0, name="SAR-Stage3")

pipeline_error = Pipeline(
    stages=[stage1_err, stage2_err, stage3_err],
    gains=[256.0, 256.0],
    name="24-bit-Pipeline-GainError"
)

# 监听InterstageGainEvent添加非理想效应
@pipeline_error.on(InterstageGainEvent)
def add_gain_error(event):
    """模拟有限运放增益导致的增益误差"""
    # 运放增益=1000，导致约0.1%增益误差
    opamp_gain = 1000
    gain_error = 1 - 1/opamp_gain
    event.actual_gain = event.ideal_gain * gain_error
    
    # 添加0.5mV失调
    event.offset = 0.5e-3
    
    # 添加0.1mV噪声
    event.noise_sigma = 0.1e-3

result_error = pipeline_error.sim(n_samples=65536, fs=1e6, fin=10e3)

print(f"带增益误差的Pipeline性能 (opamp gain=1000):")
print(f"  ENOB: {pipeline_error.enob():.2f} bits")
print(f"  SNR:  {pipeline_error.snr():.2f} dB")
print(f"  ENOB损失: {pipeline_ideal.enob() - pipeline_error.enob():.2f} bits")


# =============================================================================
# Example 3: 级间监控
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Stage-by-Stage Monitoring")
print("=" * 70)

stage1_mon = SARADC(bits=8, vref=1.0, name="SAR-Stage1")
stage2_mon = SARADC(bits=8, vref=1.0, name="SAR-Stage2")
stage3_mon = SARADC(bits=8, vref=1.0, name="SAR-Stage3")

pipeline_monitor = Pipeline(
    stages=[stage1_mon, stage2_mon, stage3_mon],
    gains=[256.0, 256.0],
    name="Monitored-Pipeline"
)

# 使用MONITOR优先级监控每一级的转换（只读，不修改）
@pipeline_monitor.on(StageEvent, priority=EventPriority.MONITOR)
def log_stage(event):
    """记录每级输入电压"""
    if event.sample_index == 0:  # 只打印第一个样本
        print(f"  Stage {event.stage_index}: input={event.input_voltage:.6f}V")

@pipeline_monitor.on(ResidueEvent, priority=EventPriority.MONITOR)
def log_residue(event):
    """记录每级残差"""
    if event.source._sample_index == 0:
        print(f"    → residue={event.residue:.6f}V")

@pipeline_monitor.on(InterstageGainEvent, priority=EventPriority.MONITOR)
def log_gain(event):
    """记录级间放大"""
    if event.source._sample_index == 0:
        amplified = event.input_voltage * event.actual_gain
        print(f"    → amplified={amplified:.6f}V (gain={event.actual_gain:.0f})")

print("\n第一个采样点的转换过程:")
result_monitor = pipeline_monitor.sim(n_samples=100, fs=1e6, fin=10e3)


# =============================================================================
# Example 4: 完整报告
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: Full Pipeline Report")
print("=" * 70)

pipeline_ideal.report()


# =============================================================================
# Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Generating Plots...")
print("=" * 70)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
fig.suptitle('24-bit Pipeline ADC (3×8-bit SAR) Analysis', fontsize=14, fontweight='bold')

# 理想Pipeline频谱
ax1 = axes[0, 0]
freqs, spec, _ = pipeline_ideal.spectrum(show=False)
ax1.plot(freqs/1e3, spec, 'b-', linewidth=0.8)
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('Power (dB)')
ax1.set_title(f'Ideal Pipeline (ENOB={pipeline_ideal.enob():.2f} bits)')
ax1.grid(True, alpha=0.3)

# 带增益误差的频谱
ax2 = axes[0, 1]
freqs2, spec2, _ = pipeline_error.spectrum(show=False)
ax2.plot(freqs2/1e3, spec2, 'r-', linewidth=0.8)
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Power (dB)')
ax2.set_title(f'With Gain Error (ENOB={pipeline_error.enob():.2f} bits)')
ax2.grid(True, alpha=0.3)

# 时域
ax3 = axes[1, 0]
result = pipeline_ideal._result
t = result.timestamps * 1e6
ax3.plot(t[:100], result.input_signal[:100], 'b-', label='Input', linewidth=1)
ax3.plot(t[:100], result.reconstructed[:100], 'r--', label='Reconstructed', linewidth=1)
ax3.set_xlabel('Time (μs)')
ax3.set_ylabel('Voltage (V)')
ax3.set_title('Time Domain (first 100 samples)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 级配置
ax4 = axes[1, 1]
stage_info = pipeline_ideal.get_stage_info()
stage_names = [f'Stage {i["index"]}\n({i["adc_name"]})' for i in stage_info]
stage_bits = [i['bits'] for i in stage_info]
bars = ax4.bar(stage_names, stage_bits, color='steelblue', alpha=0.8)
ax4.set_ylabel('Bits')
ax4.set_title('Bits per Stage (Total = 24 bits)')
ax4.grid(True, alpha=0.3, axis='y')
for bar, bits in zip(bars, stage_bits):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bits} bits', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('pipeline_24bit_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved to pipeline_24bit_analysis.png")
print("\n" + "=" * 70)
print("Pipeline重构验证完成！")
print(f"3个8-bit SAR ADC成功串联成 {pipeline_ideal.bits}-bit Pipeline")
print("=" * 70)
