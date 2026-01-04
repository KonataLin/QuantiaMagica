"""
Example 03: Noise-Shaping SAR (NS-SAR) ADC with Oversampling
============================================================

NS-SAR通过噪声整形将量化噪声推向高频，配合过采样和抽取滤波可以
在信号带宽内获得更高的有效分辨率。

关键概念:
- OSR (过采样率): fs / (2 * f_signal_bandwidth)
- 噪声整形: 将量化噪声从低频推向高频
- 带内ENOB: 只计算信号带宽内的噪声功率

Usage:
    python 03_ns_sar.py
"""

import sys
from pathlib import Path
import numpy as np
from scipy import signal as scipy_signal

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SARADC, SamplingEvent, OutputCodeEvent, EventPriority


def compute_inband_snr(codes, bits, fs, signal_freq, bandwidth):
    """
    计算带内SNR (只考虑信号带宽内的噪声).
    
    Parameters
    ----------
    codes : array
        ADC输出码
    bits : int
        ADC位数
    fs : float
        采样率
    signal_freq : float
        信号频率
    bandwidth : float
        信号带宽 (Hz)
    
    Returns
    -------
    snr : float
        带内SNR (dB)
    enob : float
        带内ENOB (bits)
    """
    n = len(codes)
    
    # 归一化到[-0.5, 0.5]
    codes_norm = codes / (2**bits) - 0.5
    
    # 加窗FFT (Hanning窗)
    win = np.hanning(n)
    coherent_gain = np.sum(win) / n
    windowed = codes_norm * win / coherent_gain
    
    # 功率谱
    spectrum = np.abs(np.fft.rfft(windowed))**2 / n
    freqs = np.fft.rfftfreq(n, 1/fs)
    freq_res = fs / n
    
    # 找信号bin (在带宽内的最大值)
    signal_bin = np.argmax(spectrum[1:]) + 1
    
    # 信号功率 (包含信号bin及相邻几个bin，因为窗函数会导致泄漏)
    signal_bins = max(1, int(3 * freq_res / freq_res))  # 信号占3个bin
    signal_power = np.sum(spectrum[max(1, signal_bin-1):signal_bin+2])
    
    # 带内噪声功率 (从DC到bandwidth，排除信号)
    bw_bin = int(bandwidth / freq_res)
    noise_power = 0
    for i in range(1, min(bw_bin + 1, len(spectrum))):
        if abs(i - signal_bin) > 2:  # 排除信号及其泄漏
            noise_power += spectrum[i]
    
    # 处理边界情况
    if noise_power < 1e-20:
        noise_power = 1e-20
    
    snr = 10 * np.log10(signal_power / noise_power)
    enob = (snr - 1.76) / 6.02
    
    return snr, enob


def simulate_ns_sar_ideal(input_signal, bits, vref=1.0, feedback_coeff=1.0):
    """
    理想NS-SAR数学模型 (用于验证).
    
    一阶噪声整形: NTF = (1 - z^-1)
    """
    n = len(input_signal)
    codes = np.zeros(n, dtype=np.int64)
    lsb = vref / (2**bits)
    
    residue = 0.0
    
    for i in range(n):
        # 加入上一次的残差反馈
        v_sample = input_signal[i] + feedback_coeff * residue
        
        # 量化 (使用round实现居中量化)
        code = int(np.clip(np.round(v_sample / lsb), 0, 2**bits - 1))
        codes[i] = code
        
        # 计算新的残差 (量化误差)
        v_dac = code * lsb
        residue = v_sample - v_dac
    
    return codes


def simulate_standard_adc(input_signal, bits, vref=1.0):
    """标准ADC理想量化模型."""
    lsb = vref / (2**bits)
    codes = np.clip(np.round(input_signal / lsb), 0, 2**bits - 1).astype(np.int64)
    return codes


class NoiseShapingSAR:
    """
    一阶噪声整形SAR ADC插件.
    
    原理: y[n] = x[n] + e[n] - e[n-1]
    其中 e[n] 是量化误差。噪声传递函数为 (1 - z^-1)，
    使量化噪声功率谱密度在低频降低，高频增加。
    
    实现方法:
    1. v_sample = v_in + residue[n-1]
    2. code = quantize(v_sample)  
    3. residue[n] = v_sample - DAC(code) = 量化误差
    """
    
    def __init__(self, feedback_coeff: float = 1.0):
        """
        Parameters
        ----------
        feedback_coeff : float
            反馈系数，1.0为标准一阶噪声整形
        """
        self.feedback_coeff = feedback_coeff
        self.prev_residue = 0.0
        self.sampled_voltage = 0.0  # 保存加了反馈后的采样电压
        self.adc = None
    
    def attach(self, adc: SARADC) -> None:
        self.adc = adc
        self.prev_residue = 0.0
        
        @adc.on(SamplingEvent, priority=EventPriority.HIGH)
        def apply_feedback(event):
            # 将上一次的量化残差加到当前采样
            event.voltage += self.feedback_coeff * self.prev_residue
            # 保存加了反馈后的电压，用于后面计算残差
            self.sampled_voltage = event.voltage
        
        @adc.on(OutputCodeEvent, priority=EventPriority.HIGH)
        def capture_residue(event):
            # 残差 = 采样电压(含反馈) - DAC输出
            # 这正是量化误差 e[n]
            reconstructed = adc.code_to_voltage(event.code)
            self.prev_residue = self.sampled_voltage - reconstructed
    
    def reset(self):
        self.prev_residue = 0.0
        self.sampled_voltage = 0.0


# =============================================================================
# 参数设置
# =============================================================================

BITS = 8                     # ADC位数 (用较低位数更容易看出效果)
FS = 1e6                     # 采样率 1MHz
OSR = 64                     # 过采样率
SIGNAL_BW = FS / (2 * OSR)   # 信号带宽 = fs/(2*OSR) ≈ 7.8kHz
FIN = 1e3                    # 输入信号频率 (在带宽内)
N_SAMPLES = 16384            # 采样点数

print("=" * 60)
print("Noise-Shaping SAR ADC with Oversampling")
print("=" * 60)
print(f"\n参数:")
print(f"  ADC位数:     {BITS} bits")
print(f"  采样率:      {FS/1e6:.1f} MHz")
print(f"  过采样率:    OSR = {OSR}")
print(f"  信号带宽:    {SIGNAL_BW/1e3:.2f} kHz")
print(f"  输入频率:    {FIN/1e3:.1f} kHz")
print(f"  采样点数:    {N_SAMPLES}")


# =============================================================================
# 生成输入信号
# =============================================================================

# 相干采样 - 确保整数周期
n_periods = int(np.round(FIN * N_SAMPLES / FS))
fin_coherent = n_periods * FS / N_SAMPLES
t = np.arange(N_SAMPLES) / FS
amplitude = 0.4
offset = 0.5
input_signal = offset + amplitude * np.sin(2 * np.pi * fin_coherent * t)


# =============================================================================
# 1. 标准SAR ADC (使用原生SARADC)
# =============================================================================

print("\n" + "-" * 60)
print("1. Standard SAR ADC (原生SARADC)")
print("-" * 60)

# 使用原生SARADC
std_adc = SARADC(bits=BITS, vref=1.0)
std_adc.sim(input_signal, fs=FS)
std_codes = std_adc._result.output_codes

# 全带宽和带内SNR
std_fullband_snr, std_fullband_enob = compute_inband_snr(
    std_codes, BITS, FS, fin_coherent, FS/2
)
std_inband_snr, std_inband_enob = compute_inband_snr(
    std_codes, BITS, FS, fin_coherent, SIGNAL_BW
)

print(f"  全带宽 SNR:  {std_fullband_snr:.1f} dB  (ENOB = {std_fullband_enob:.2f})")
print(f"  带内 SNR:    {std_inband_snr:.1f} dB  (ENOB = {std_inband_enob:.2f})")


# =============================================================================
# 2. NS-SAR ADC (理想数学模型 - 噪声整形)
# =============================================================================

print("\n" + "-" * 60)
print("2. Noise-Shaping SAR ADC (1st order)")
print("-" * 60)

# 使用理想NS-SAR模型 (数学等效，更精确的噪声整形)
ns_codes = simulate_ns_sar_ideal(input_signal, BITS, vref=1.0, feedback_coeff=1.0)

# 全带宽和带内SNR
ns_fullband_snr, ns_fullband_enob = compute_inband_snr(
    ns_codes, BITS, FS, fin_coherent, FS/2
)
ns_inband_snr, ns_inband_enob = compute_inband_snr(
    ns_codes, BITS, FS, fin_coherent, SIGNAL_BW
)

print(f"  全带宽 SNR:  {ns_fullband_snr:.1f} dB  (ENOB = {ns_fullband_enob:.2f})")
print(f"  带内 SNR:    {ns_inband_snr:.1f} dB  (ENOB = {ns_inband_enob:.2f})")


# =============================================================================
# 3. 对比总结
# =============================================================================

print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"\n{'指标':<25} {'Standard SAR':<15} {'NS-SAR':<15} {'提升':<10}")
print("-" * 65)
print(f"{'全带宽 ENOB':<25} {std_fullband_enob:<15.2f} {ns_fullband_enob:<15.2f} {ns_fullband_enob - std_fullband_enob:+.2f}")
print(f"{'带内 ENOB (OSR={OSR})':<25} {std_inband_enob:<15.2f} {ns_inband_enob:<15.2f} {ns_inband_enob - std_inband_enob:+.2f}")

improvement = ns_inband_enob - std_inband_enob
print(f"\n结论: NS-SAR在带内获得 {improvement:.1f} bits 的ENOB提升!")
print(f"       理论提升: 一阶整形 + OSR={OSR} ≈ {0.5 * np.log2(OSR) + 0.5 * 3:.1f} bits")


# =============================================================================
# 4. 频谱对比图 (IEEE JSSC风格)
# =============================================================================

import matplotlib.pyplot as plt

# 使用Core集成的对比绘图函数
from quantiamagica import plot_comparison

plot_comparison(
    [std_codes, ns_codes],
    fs=FS,
    labels=[f'Standard SAR\nENOB={std_inband_enob:.2f}', f'NS-SAR\nENOB={ns_inband_enob:.2f}'],
    bandwidth=SIGNAL_BW,
    title='Standard vs NS-SAR Comparison',
    save='ns_sar_comparison.png'
)

print("\n频谱图已保存至 ns_sar_comparison.png")
print("\n观察频谱图可以看到:")
print("- Standard SAR: 量化噪声在全频带平坦分布")
print("- NS-SAR: 量化噪声被推向高频，带内(灰色区域)噪声降低")
