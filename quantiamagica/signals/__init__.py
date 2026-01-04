"""
Signal Module - 信号生成与管理
==============================

提供丰富的测试信号生成功能，支持多种波形和自定义导入。

Example
-------
>>> from quantiamagica.signals import Signal
>>> 
>>> # 正弦波
>>> sig = Signal.sine(n=1024, fs=1e6, freq=10e3, amplitude=0.4)
>>> 
>>> # 从文件导入
>>> sig = Signal.from_file("my_signal.csv")
>>> 
>>> # 直接用于ADC仿真
>>> adc.sim(sig)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Union, Callable, Any
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@dataclass
class Signal:
    """
    信号类 - 封装测试信号及其元数据。
    
    Attributes
    ----------
    data : NDArray[np.float64]
        信号电压数据。
    fs : float
        采样率 (Hz)。
    name : str
        信号名称。
    timestamps : NDArray[np.float64]
        时间戳数组。
    metadata : dict
        额外元数据。
    
    Example
    -------
    >>> sig = Signal.sine(n=1024, freq=10e3)
    >>> print(f"信号长度: {len(sig)}, 峰峰值: {sig.vpp:.3f}V")
    """
    
    data: NDArray[np.float64]
    fs: float = 1e6
    name: str = "Signal"
    timestamps: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.timestamps) == 0:
            self.timestamps = np.arange(len(self.data)) / self.fs
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __array__(self, dtype=None) -> NDArray:
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data
    
    def __getitem__(self, idx) -> Union[float, NDArray]:
        return self.data[idx]
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def n_samples(self) -> int:
        """样本数量。"""
        return len(self.data)
    
    @property
    def duration(self) -> float:
        """信号持续时间 (秒)。"""
        return len(self.data) / self.fs
    
    @property
    def vmax(self) -> float:
        """最大电压。"""
        return float(np.max(self.data))
    
    @property
    def vmin(self) -> float:
        """最小电压。"""
        return float(np.min(self.data))
    
    @property
    def vpp(self) -> float:
        """峰峰值电压。"""
        return self.vmax - self.vmin
    
    @property
    def vrms(self) -> float:
        """RMS电压。"""
        return float(np.sqrt(np.mean(self.data ** 2)))
    
    @property
    def mean(self) -> float:
        """平均值。"""
        return float(np.mean(self.data))
    
    # =========================================================================
    # 正弦波系列
    # =========================================================================
    
    @classmethod
    def sine(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        freq: float = 1e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
        phase: float = 0.0,
        coherent: bool = True,
    ) -> "Signal":
        """
        生成正弦波信号。
        
        Parameters
        ----------
        n : int
            样本数量。
        fs : float
            采样率 (Hz)。
        freq : float
            信号频率 (Hz)。
        amplitude : float
            振幅 (V)。
        offset : float
            直流偏置 (V)。
        phase : float
            初始相位 (弧度)。
        coherent : bool
            是否进行相干采样调整。
        
        Returns
        -------
        Signal
            正弦波信号对象。
        """
        t = np.arange(n) / fs
        
        if coherent:
            n_periods = max(1, int(np.round(freq * n / fs)))
            freq = n_periods * fs / n
        
        data = offset + amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Sine_{freq/1e3:.1f}kHz",
            timestamps=t,
            metadata={"type": "sine", "freq": freq, "amplitude": amplitude, "offset": offset},
        )
    
    @classmethod
    def cosine(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        freq: float = 1e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
        coherent: bool = True,
    ) -> "Signal":
        """生成余弦波信号。"""
        return cls.sine(n, fs, freq, amplitude, offset, phase=np.pi/2, coherent=coherent)
    
    # =========================================================================
    # 方波、三角波、锯齿波
    # =========================================================================
    
    @classmethod
    def square(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        freq: float = 1e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
        duty: float = 0.5,
    ) -> "Signal":
        """
        生成方波信号。
        
        Parameters
        ----------
        duty : float
            占空比 (0-1)。
        """
        t = np.arange(n) / fs
        period = 1.0 / freq
        phase = (t % period) / period
        data = offset + amplitude * (2 * (phase < duty).astype(float) - 1)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Square_{freq/1e3:.1f}kHz",
            timestamps=t,
            metadata={"type": "square", "freq": freq, "duty": duty},
        )
    
    @classmethod
    def triangle(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        freq: float = 1e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
    ) -> "Signal":
        """生成三角波信号。"""
        t = np.arange(n) / fs
        period = 1.0 / freq
        phase = (t % period) / period
        data = offset + amplitude * (4 * np.abs(phase - 0.5) - 1)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Triangle_{freq/1e3:.1f}kHz",
            timestamps=t,
            metadata={"type": "triangle", "freq": freq},
        )
    
    @classmethod
    def sawtooth(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        freq: float = 1e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
        rising: bool = True,
    ) -> "Signal":
        """
        生成锯齿波信号。
        
        Parameters
        ----------
        rising : bool
            True为上升锯齿，False为下降锯齿。
        """
        t = np.arange(n) / fs
        period = 1.0 / freq
        phase = (t % period) / period
        
        if rising:
            data = offset + amplitude * (2 * phase - 1)
        else:
            data = offset + amplitude * (1 - 2 * phase)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Sawtooth_{freq/1e3:.1f}kHz",
            timestamps=t,
            metadata={"type": "sawtooth", "freq": freq, "rising": rising},
        )
    
    # =========================================================================
    # 特殊测试信号
    # =========================================================================
    
    @classmethod
    def ramp(
        cls,
        n: int = 4096,
        fs: float = 1e6,
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> "Signal":
        """生成线性斜坡信号（用于INL/DNL测试）。"""
        t = np.arange(n) / fs
        data = np.linspace(vmin, vmax, n)
        
        return cls(
            data=data,
            fs=fs,
            name="Ramp",
            timestamps=t,
            metadata={"type": "ramp", "vmin": vmin, "vmax": vmax},
        )
    
    @classmethod
    def dc(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        voltage: float = 0.5,
    ) -> "Signal":
        """生成直流信号。"""
        t = np.arange(n) / fs
        data = np.full(n, voltage)
        
        return cls(
            data=data,
            fs=fs,
            name=f"DC_{voltage:.2f}V",
            timestamps=t,
            metadata={"type": "dc", "voltage": voltage},
        )
    
    @classmethod
    def step(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        v_low: float = 0.2,
        v_high: float = 0.8,
        step_point: float = 0.5,
    ) -> "Signal":
        """
        生成阶跃信号。
        
        Parameters
        ----------
        step_point : float
            阶跃位置 (0-1)。
        """
        t = np.arange(n) / fs
        step_idx = int(n * step_point)
        data = np.concatenate([
            np.full(step_idx, v_low),
            np.full(n - step_idx, v_high)
        ])
        
        return cls(
            data=data,
            fs=fs,
            name="Step",
            timestamps=t,
            metadata={"type": "step", "v_low": v_low, "v_high": v_high},
        )
    
    @classmethod
    def pulse(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        v_base: float = 0.2,
        v_pulse: float = 0.8,
        pulse_start: float = 0.3,
        pulse_width: float = 0.2,
    ) -> "Signal":
        """生成脉冲信号。"""
        t = np.arange(n) / fs
        start_idx = int(n * pulse_start)
        end_idx = int(n * (pulse_start + pulse_width))
        
        data = np.full(n, v_base)
        data[start_idx:end_idx] = v_pulse
        
        return cls(
            data=data,
            fs=fs,
            name="Pulse",
            timestamps=t,
            metadata={"type": "pulse"},
        )
    
    # =========================================================================
    # 多音信号
    # =========================================================================
    
    @classmethod
    def multitone(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        frequencies: List[float] = None,
        amplitudes: List[float] = None,
        offset: float = 0.5,
        coherent: bool = True,
    ) -> "Signal":
        """
        生成多音信号（用于IMD测试）。
        
        Parameters
        ----------
        frequencies : List[float]
            频率列表 (Hz)。
        amplitudes : List[float]
            各频率对应的振幅。
        """
        if frequencies is None:
            frequencies = [1e3, 1.1e3]
        if amplitudes is None:
            amplitudes = [0.2] * len(frequencies)
        
        t = np.arange(n) / fs
        data = np.full(n, offset)
        
        actual_freqs = []
        for freq, amp in zip(frequencies, amplitudes):
            if coherent:
                n_periods = max(1, int(np.round(freq * n / fs)))
                freq = n_periods * fs / n
            actual_freqs.append(freq)
            data += amp * np.sin(2 * np.pi * freq * t)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Multitone_{len(frequencies)}",
            timestamps=t,
            metadata={"type": "multitone", "frequencies": actual_freqs, "amplitudes": amplitudes},
        )
    
    @classmethod
    def two_tone(
        cls,
        n: int = 2048,
        fs: float = 1e6,
        f1: float = 10e3,
        f2: float = 11e3,
        amplitude: float = 0.2,
        offset: float = 0.5,
    ) -> "Signal":
        """生成双音信号（用于IMD3测试）。"""
        return cls.multitone(
            n=n,
            fs=fs,
            frequencies=[f1, f2],
            amplitudes=[amplitude, amplitude],
            offset=offset,
        )
    
    # =========================================================================
    # 噪声信号
    # =========================================================================
    
    @classmethod
    def noise_gaussian(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        mean: float = 0.5,
        sigma: float = 0.1,
    ) -> "Signal":
        """生成高斯白噪声信号。"""
        t = np.arange(n) / fs
        data = mean + np.random.normal(0, sigma, n)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Gaussian_σ={sigma:.3f}",
            timestamps=t,
            metadata={"type": "gaussian_noise", "mean": mean, "sigma": sigma},
        )
    
    @classmethod
    def noise_uniform(
        cls,
        n: int = 1024,
        fs: float = 1e6,
        vmin: float = 0.3,
        vmax: float = 0.7,
    ) -> "Signal":
        """生成均匀分布噪声信号。"""
        t = np.arange(n) / fs
        data = np.random.uniform(vmin, vmax, n)
        
        return cls(
            data=data,
            fs=fs,
            name="Uniform_Noise",
            timestamps=t,
            metadata={"type": "uniform_noise", "vmin": vmin, "vmax": vmax},
        )
    
    # =========================================================================
    # 调制信号
    # =========================================================================
    
    @classmethod
    def am(
        cls,
        n: int = 2048,
        fs: float = 1e6,
        carrier_freq: float = 100e3,
        mod_freq: float = 1e3,
        mod_depth: float = 0.5,
        amplitude: float = 0.4,
        offset: float = 0.5,
    ) -> "Signal":
        """生成调幅(AM)信号。"""
        t = np.arange(n) / fs
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        data = offset + amplitude * carrier * modulation
        
        return cls(
            data=data,
            fs=fs,
            name=f"AM_{carrier_freq/1e3:.0f}kHz",
            timestamps=t,
            metadata={"type": "am", "carrier_freq": carrier_freq, "mod_freq": mod_freq},
        )
    
    @classmethod
    def fm(
        cls,
        n: int = 2048,
        fs: float = 1e6,
        carrier_freq: float = 100e3,
        mod_freq: float = 1e3,
        freq_deviation: float = 10e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
    ) -> "Signal":
        """生成调频(FM)信号。"""
        t = np.arange(n) / fs
        phase = 2 * np.pi * carrier_freq * t + (freq_deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
        data = offset + amplitude * np.sin(phase)
        
        return cls(
            data=data,
            fs=fs,
            name=f"FM_{carrier_freq/1e3:.0f}kHz",
            timestamps=t,
            metadata={"type": "fm", "carrier_freq": carrier_freq, "mod_freq": mod_freq},
        )
    
    @classmethod
    def chirp(
        cls,
        n: int = 2048,
        fs: float = 1e6,
        f_start: float = 1e3,
        f_end: float = 100e3,
        amplitude: float = 0.4,
        offset: float = 0.5,
        method: str = "linear",
    ) -> "Signal":
        """
        生成扫频(Chirp)信号。
        
        Parameters
        ----------
        method : str
            扫频方式: 'linear', 'quadratic', 'logarithmic'
        """
        t = np.arange(n) / fs
        T = n / fs
        
        if method == "linear":
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * T))
        elif method == "quadratic":
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**3 / (3 * T**2))
        elif method == "logarithmic":
            k = (f_end / f_start) ** (1 / T)
            phase = 2 * np.pi * f_start * (k**t - 1) / np.log(k)
        else:
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * T))
        
        data = offset + amplitude * np.sin(phase)
        
        return cls(
            data=data,
            fs=fs,
            name=f"Chirp_{f_start/1e3:.0f}-{f_end/1e3:.0f}kHz",
            timestamps=t,
            metadata={"type": "chirp", "f_start": f_start, "f_end": f_end, "method": method},
        )
    
    # =========================================================================
    # 导入/导出
    # =========================================================================
    
    @classmethod
    def from_array(
        cls,
        data: NDArray,
        fs: float = 1e6,
        name: str = "Custom",
    ) -> "Signal":
        """从NumPy数组创建信号。"""
        return cls(
            data=np.asarray(data, dtype=np.float64),
            fs=fs,
            name=name,
            metadata={"type": "custom"},
        )
    
    @classmethod
    def from_file(
        cls,
        path: str,
        fs: float = 1e6,
        column: int = 0,
        delimiter: str = ",",
        skip_header: int = 1,
    ) -> "Signal":
        """
        从文件导入信号。
        
        支持格式: CSV, TXT, NPY, NPZ
        
        Parameters
        ----------
        path : str
            文件路径。
        fs : float
            采样率。
        column : int
            数据列索引 (用于CSV/TXT)。
        delimiter : str
            分隔符。
        skip_header : int
            跳过的头部行数。
        """
        filepath = Path(path)
        ext = filepath.suffix.lower()
        
        if ext == ".npy":
            data = np.load(path)
        elif ext == ".npz":
            npz = np.load(path)
            keys = list(npz.keys())
            data = npz[keys[0]]
        elif ext in [".csv", ".txt", ".dat"]:
            data = np.loadtxt(
                path,
                delimiter=delimiter,
                skiprows=skip_header,
                usecols=column,
            )
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        return cls(
            data=np.asarray(data, dtype=np.float64),
            fs=fs,
            name=filepath.stem,
            metadata={"type": "imported", "source": str(path)},
        )
    
    def save(self, path: str, format: str = "auto") -> None:
        """
        保存信号到文件。
        
        Parameters
        ----------
        path : str
            输出路径。
        format : str
            格式: 'auto', 'npy', 'npz', 'csv'
        """
        filepath = Path(path)
        
        if format == "auto":
            format = filepath.suffix.lower().lstrip(".")
        
        if format == "npy":
            np.save(path, self.data)
        elif format == "npz":
            np.savez(path, data=self.data, timestamps=self.timestamps)
        elif format == "csv":
            np.savetxt(
                path,
                np.column_stack([self.timestamps, self.data]),
                delimiter=",",
                header="timestamp,voltage",
                comments="",
            )
        else:
            np.save(path, self.data)
    
    # =========================================================================
    # 信号处理
    # =========================================================================
    
    def add_noise(self, sigma: float) -> "Signal":
        """添加高斯噪声。"""
        noisy_data = self.data + np.random.normal(0, sigma, len(self.data))
        return Signal(
            data=noisy_data,
            fs=self.fs,
            name=f"{self.name}_noisy",
            timestamps=self.timestamps.copy(),
            metadata={**self.metadata, "noise_sigma": sigma},
        )
    
    def add_offset(self, offset: float) -> "Signal":
        """添加直流偏置。"""
        return Signal(
            data=self.data + offset,
            fs=self.fs,
            name=self.name,
            timestamps=self.timestamps.copy(),
            metadata=self.metadata,
        )
    
    def scale(self, factor: float) -> "Signal":
        """缩放信号。"""
        return Signal(
            data=self.data * factor,
            fs=self.fs,
            name=self.name,
            timestamps=self.timestamps.copy(),
            metadata=self.metadata,
        )
    
    def clip(self, vmin: float, vmax: float) -> "Signal":
        """裁剪信号到指定范围。"""
        return Signal(
            data=np.clip(self.data, vmin, vmax),
            fs=self.fs,
            name=f"{self.name}_clipped",
            timestamps=self.timestamps.copy(),
            metadata=self.metadata,
        )
    
    def normalize(self, vmin: float = 0.0, vmax: float = 1.0) -> "Signal":
        """归一化信号到指定范围。"""
        data_min, data_max = self.data.min(), self.data.max()
        if data_max == data_min:
            normalized = np.full_like(self.data, (vmin + vmax) / 2)
        else:
            normalized = (self.data - data_min) / (data_max - data_min)
            normalized = normalized * (vmax - vmin) + vmin
        
        return Signal(
            data=normalized,
            fs=self.fs,
            name=f"{self.name}_normalized",
            timestamps=self.timestamps.copy(),
            metadata=self.metadata,
        )
    
    def resample(self, new_fs: float) -> "Signal":
        """重采样到新采样率。"""
        from scipy import signal as sp_signal
        
        ratio = new_fs / self.fs
        new_n = int(len(self.data) * ratio)
        resampled = sp_signal.resample(self.data, new_n)
        
        return Signal(
            data=resampled,
            fs=new_fs,
            name=f"{self.name}_resampled",
            metadata=self.metadata,
        )
    
    def __add__(self, other: Union["Signal", float]) -> "Signal":
        """信号相加。"""
        if isinstance(other, Signal):
            return Signal(
                data=self.data + other.data,
                fs=self.fs,
                name=f"{self.name}+{other.name}",
                timestamps=self.timestamps.copy(),
            )
        return self.add_offset(float(other))
    
    def __mul__(self, factor: float) -> "Signal":
        """信号缩放。"""
        return self.scale(factor)
    
    def __repr__(self) -> str:
        return f"Signal('{self.name}', n={len(self)}, fs={self.fs/1e6:.2f}MHz, vpp={self.vpp:.3f}V)"
    
    # =========================================================================
    # 绘图
    # =========================================================================
    
    def plot(
        self,
        show: bool = True,
        save: Optional[str] = None,
        figsize: tuple = (10, 4),
    ):
        """绘制信号波形。"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.plot(self.timestamps * 1e6, self.data, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(self.name)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


# 便捷函数
def sine(n=1024, fs=1e6, freq=1e3, **kwargs) -> Signal:
    """生成正弦波（快捷函数）。"""
    return Signal.sine(n=n, fs=fs, freq=freq, **kwargs)

def ramp(n=4096, vmin=0.0, vmax=1.0, **kwargs) -> Signal:
    """生成斜坡（快捷函数）。"""
    return Signal.ramp(n=n, vmin=vmin, vmax=vmax, **kwargs)

def multitone(frequencies, **kwargs) -> Signal:
    """生成多音信号（快捷函数）。"""
    return Signal.multitone(frequencies=frequencies, **kwargs)


__all__ = [
    "Signal",
    "sine",
    "ramp", 
    "multitone",
]
