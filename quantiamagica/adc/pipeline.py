"""
Pipeline: ADC级联链类（不继承ADConverter）

Pipeline是一个独立的类，用于将多个ADC（SAR/Sigma-Delta等）串联成流水线结构。
每一级ADC转换后，误差（残差）经过级间增益放大传递给下一级。

核心功能：
- 接收ADC实例列表，按流水线时序逐级调度转换
- 级间InterstageGainEvent支持用户注入增益误差、失调、噪声等非理想效应
- 时序对齐输出，合成最终高精度数字码
- 实现spectrum/enob/snr/sfdr/thd/dnl/inl等分析方法

Example
-------
>>> from quantiamagica import Pipeline, SARADC, InterstageGainEvent
>>> 
>>> # 3个8-bit SAR ADC串联成24-bit Pipeline
>>> stages = [SARADC(bits=8) for _ in range(3)]
>>> pipeline = Pipeline(stages, gains=[256, 256])  # 级间增益=2^8
>>> 
>>> # 监听级间增益事件添加非理想效应
>>> @pipeline.on(InterstageGainEvent)
>>> def add_gain_error(event):
>>>     event.actual_gain *= 0.999  # 0.1%增益误差
>>>     event.offset = 0.5e-3       # 0.5mV失调
>>> 
>>> result = pipeline.sim(n_samples=4096)
>>> print(f"ENOB: {pipeline.enob():.2f} bits")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from numpy.typing import NDArray

from ..core.events import Event, Cancellable, EventBus, EventPriority
from ..core.base import ADConverter, SimulationResult


# =============================================================================
# Pipeline Events
# =============================================================================

@dataclass
class StageEvent(Event):
    """
    Fired when processing moves to a new pipeline stage.
    
    Attributes
    ----------
    stage_index : int
        Index of the current stage (0-indexed).
    stage_adc : ADConverter
        The ADC being used for this stage.
    input_voltage : float
        Input voltage to this stage.
    sample_index : int
        Global sample index.
    """
    stage_index: int = 0
    stage_adc: Any = None
    input_voltage: float = 0.0
    sample_index: int = 0


@dataclass
class ResidueEvent(Event, Cancellable):
    """
    Fired when computing residue voltage for next stage.
    
    Attributes
    ----------
    stage_index : int
        Current stage index.
    input_voltage : float
        Stage input voltage.
    dac_voltage : float
        DAC output from flash code.
    residue : float
        Computed residue (modifiable).
    ideal_residue : float
        Ideal residue without errors.
    """
    stage_index: int = 0
    input_voltage: float = 0.0
    dac_voltage: float = 0.0
    residue: float = 0.0
    ideal_residue: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class InterstageGainEvent(Event, Cancellable):
    """
    Fired during inter-stage amplification.
    
    Modify to add amplifier non-idealities:
    - Finite gain
    - Gain error
    - Offset
    - Bandwidth limitations
    - Noise
    
    Attributes
    ----------
    stage_index : int
        Current stage index.
    input_voltage : float
        Amplifier input (residue).
    ideal_gain : float
        Ideal gain (typically 2^stage_bits).
    actual_gain : float
        Actual gain with errors (modifiable).
    offset : float
        Amplifier offset voltage (modifiable).
    noise_sigma : float
        Amplifier noise sigma (modifiable).
    output_voltage : float
        Amplified output (computed after event).
    bandwidth : float
        Amplifier bandwidth in Hz.
    settling_error : float
        Incomplete settling error.
    
    Example
    -------
    >>> @pipeline.on(InterstageGainEvent)
    ... def finite_gain(event):
    ...     # Model finite opamp gain of 1000
    ...     event.actual_gain = event.ideal_gain * (1 - 1/1000)
    """
    stage_index: int = 0
    input_voltage: float = 0.0
    ideal_gain: float = 2.0
    actual_gain: float = 2.0
    offset: float = 0.0
    noise_sigma: float = 0.0
    output_voltage: float = 0.0
    bandwidth: float = 100e6
    settling_error: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


# =============================================================================
# Pipeline: ADC级联链类（独立于ADConverter）
# =============================================================================

class Pipeline:
    """
    Pipeline ADC级联链 - 将多个ADC串联成流水线结构。
    
    Pipeline是一个独立的类（不继承ADConverter），用于将多个ADC实例
    （SAR/Sigma-Delta等）串联成高精度流水线结构。
    
    工作原理：
    1. 第一级ADC转换输入信号，产生粗量化码
    2. 计算残差（输入 - DAC重建）
    3. 残差经过级间增益放大（InterstageGainEvent）
    4. 放大后的残差传递给下一级ADC
    5. 重复步骤2-4直到最后一级
    6. 合并各级数字码，输出高精度结果
    
    Parameters
    ----------
    stages : List[ADConverter]
        ADC实例列表，按流水线顺序排列。
    gains : List[float], optional
        级间增益列表，长度为len(stages)-1。
        默认为每级ADC的2^bits。
    vref : float, optional
        参考电压，默认使用第一级ADC的vref。
    name : str, optional
        Pipeline名称。
    
    Example
    -------
    >>> # 3个8-bit SAR串联成24-bit Pipeline
    >>> stages = [SARADC(bits=8) for _ in range(3)]
    >>> pipeline = Pipeline(stages, gains=[256, 256])
    >>> 
    >>> @pipeline.on(InterstageGainEvent)
    >>> def add_gain_error(event):
    >>>     event.actual_gain *= 0.999  # 0.1%增益误差
    >>> 
    >>> pipeline.sim(n_samples=4096)
    >>> print(f"ENOB: {pipeline.enob():.2f} bits")
    """
    
    def __init__(
        self,
        stages: List[ADConverter],
        gains: Optional[List[float]] = None,
        vref: Optional[float] = None,
        name: Optional[str] = None,
    ):
        if len(stages) < 1:
            raise ValueError("Pipeline requires at least 1 stage")
        
        self._stages = stages
        self.name = name or "Pipeline"
        
        # 参考电压
        self.vref = vref if vref is not None else stages[0].vref
        self.vmin = stages[0].vmin
        
        # 总位数 = 各级位数之和
        self.bits = sum(adc.bits for adc in stages)
        
        # 级间增益：默认为2^(当前级位数)
        if gains is None:
            self._gains = [float(2 ** adc.bits) for adc in stages[:-1]]
        else:
            if len(gains) != len(stages) - 1:
                raise ValueError(f"gains length must be {len(stages)-1}, got {len(gains)}")
            self._gains = list(gains)
        
        # 事件总线
        self._event_bus = EventBus()
        
        # 仿真结果
        self._result: Optional[SimulationResult] = None
        self._time: float = 0.0
        self._sample_index: int = 0
        
        # 各级转换结果
        self._stage_codes: List[List[int]] = []
        self._stage_residues: List[List[float]] = []
    
    @property
    def lsb(self) -> float:
        """最小分辨电压"""
        return (self.vref - self.vmin) / (2 ** self.bits)
    
    @property
    def max_code(self) -> int:
        """最大输出码"""
        return 2 ** self.bits - 1
    
    @property
    def num_stages(self) -> int:
        """级数"""
        return len(self._stages)
    
    @property
    def result(self) -> Optional[SimulationResult]:
        """仿真结果"""
        return self._result
    
    def on(
        self,
        event_type: Type[Event],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
    ) -> Callable:
        """注册事件处理器的装饰器"""
        return self._event_bus.on(event_type, priority, ignore_cancelled)
    
    def register(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
    ) -> None:
        """程序化注册事件处理器"""
        self._event_bus.register(event_type, callback, priority, ignore_cancelled)
    
    def fire(self, event: Event) -> Event:
        """触发事件"""
        event.source = self
        return self._event_bus.fire(event)
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        """
        执行一次流水线转换。
        
        Pipeline ADC直接计算理想的高精度输出码，同时触发事件允许注入非理想效应。
        理想情况下，3个8-bit ADC串联可达到接近24-bit ENOB。
        
        Parameters
        ----------
        voltage : float
            输入电压
        timestamp : float
            时间戳
        
        Returns
        -------
        int
            合并后的数字码
        """
        vrange = self.vref - self.vmin
        
        # 添加0.5 LSB偏移用于正确舍入（与SAR ADC一致）
        voltage_adjusted = voltage + 0.5 * self.lsb
        
        # 计算理想的完整输出码（使用全精度）
        normalized = (voltage_adjusted - self.vmin) / vrange
        normalized = np.clip(normalized, 0.0, 1.0 - 1e-15)
        ideal_full_code = int(normalized * (2 ** self.bits))
        ideal_full_code = max(0, min(self.max_code, ideal_full_code))
        
        # 从理想码中分解各级码字（用于事件触发）
        stage_codes = []
        remaining_code = ideal_full_code
        
        # 计算各级的位移量
        shifts = []
        shift = 0
        for i in range(len(self._stages) - 1, -1, -1):
            shifts.insert(0, shift)
            shift += self._stages[i].bits
        
        # 累积非理想效应的误差
        total_error = 0.0
        
        for stage_idx, stage_adc in enumerate(self._stages):
            stage_bits = stage_adc.bits
            stage_levels = 2 ** stage_bits
            shift = shifts[stage_idx]
            
            # 从理想码中提取该级的码字
            stage_code = (ideal_full_code >> shift) & (stage_levels - 1)
            stage_codes.append(stage_code)
            
            # 计算该级的输入电压（用于事件）
            if stage_idx == 0:
                current_voltage = voltage
            else:
                # 后续级的输入是前一级残差放大后的结果
                current_voltage = self.vmin + (remaining_code / (2 ** (self.bits - shift))) * vrange
            
            # 触发StageEvent
            stage_event = StageEvent(
                timestamp=timestamp,
                stage_index=stage_idx,
                stage_adc=stage_adc,
                input_voltage=current_voltage,
                sample_index=self._sample_index,
            )
            self.fire(stage_event)
            
            # 计算残差
            dac_voltage = self.vmin + (stage_code / stage_levels) * vrange
            residue = current_voltage - dac_voltage
            
            # 触发ResidueEvent
            residue_event = ResidueEvent(
                timestamp=timestamp,
                stage_index=stage_idx,
                input_voltage=current_voltage,
                dac_voltage=dac_voltage,
                residue=residue,
                ideal_residue=residue,
            )
            self.fire(residue_event)
            if not residue_event.cancelled:
                residue = residue_event.residue
            
            # 非最后一级：触发级间增益事件
            if stage_idx < len(self._stages) - 1:
                ideal_gain = self._gains[stage_idx]
                
                # 触发InterstageGainEvent
                gain_event = InterstageGainEvent(
                    timestamp=timestamp,
                    stage_index=stage_idx,
                    input_voltage=residue,
                    ideal_gain=ideal_gain,
                    actual_gain=ideal_gain,
                )
                self.fire(gain_event)
                
                # 累积非理想效应
                # 误差影响后续级，后续级的码字占据低位
                # 后续级的总位数 = shift (Stage i+1到最后一级的位数和)
                if not gain_event.cancelled:
                    # 增益误差导致的电压误差
                    # 残差被放大后送入下一级，增益误差导致放大不足/过度
                    gain_error_ratio = (gain_event.actual_gain - ideal_gain) / ideal_gain
                    # 误差电压 = residue * ideal_gain * error_ratio
                    error_voltage = residue * ideal_gain * gain_error_ratio
                    # 转换为后续级的LSB单位（后续级的量程仍是vrange）
                    next_stage_lsb = vrange / (2 ** shift) if shift > 0 else self.lsb
                    total_error += error_voltage / next_stage_lsb
                    
                    # 失调（直接加到放大器输出）
                    if gain_event.offset != 0:
                        total_error += gain_event.offset / next_stage_lsb
                    
                    # 噪声
                    if gain_event.noise_sigma > 0:
                        noise = np.random.normal(0, gain_event.noise_sigma)
                        total_error += noise / next_stage_lsb
            
            # 更新剩余码
            remaining_code = remaining_code & ((1 << shift) - 1) if shift > 0 else 0
        
        # 应用累积误差
        final_code = ideal_full_code + int(round(total_error))
        final_code = max(0, min(self.max_code, final_code))
        
        return final_code
    
    def _combine_codes(self, stage_codes: List[int]) -> int:
        """
        合并各级数字码。
        
        简单拼接：code = stage0 << (bits1+bits2+...) | stage1 << (bits2+...) | ...
        """
        code = 0
        shift = 0
        
        # 从最后一级开始
        for i in range(len(self._stages) - 1, -1, -1):
            stage_code = stage_codes[i]
            code |= (stage_code << shift)
            shift += self._stages[i].bits
        
        return min(max(0, code), self.max_code)
    
    def code_to_voltage(self, code: int) -> float:
        """数字码转电压"""
        return self.vmin + (code + 0.5) * self.lsb
    
    def sim(
        self,
        input_voltage: Union[float, NDArray, None] = None,
        *,
        n_samples: int = 1024,
        fs: float = 1e6,
        fin: float = 10e3,
        amplitude: Optional[float] = None,
        offset: Optional[float] = None,
        signal: str = "sine",
    ) -> SimulationResult:
        """
        运行Pipeline仿真。
        
        Parameters
        ----------
        input_voltage : float, NDArray, or None
            输入信号，None则生成测试信号
        n_samples : int
            采样点数
        fs : float
            采样率 (Hz)
        fin : float
            信号频率 (Hz)
        amplitude : float, optional
            信号幅度
        offset : float, optional
            直流偏置
        signal : str
            信号类型: 'sine', 'ramp', 'dc'
        
        Returns
        -------
        SimulationResult
            仿真结果
        """
        self._time = 0.0
        self._sample_index = 0
        
        if amplitude is None:
            amplitude = 0.498 * (self.vref - self.vmin)
        if offset is None:
            offset = (self.vref + self.vmin) / 2
        
        # 生成输入信号
        if input_voltage is None:
            timestamps = np.arange(n_samples) / fs
            if signal == "sine":
                n_periods = int(np.round(fin * n_samples / fs))
                n_periods = max(1, n_periods)
                fin_coherent = n_periods * fs / n_samples
                input_signal = offset + amplitude * np.sin(
                    2 * np.pi * fin_coherent * timestamps
                )
            elif signal == "ramp":
                input_signal = np.linspace(self.vmin, self.vref, n_samples)
            elif signal == "dc":
                input_signal = np.full(n_samples, offset)
            else:
                raise ValueError(f"Unknown signal type: {signal}")
        elif isinstance(input_voltage, (int, float)):
            input_signal = np.array([float(input_voltage)])
            timestamps = np.array([0.0])
            fin_coherent = fin
        else:
            input_signal = np.asarray(input_voltage, dtype=np.float64)
            timestamps = np.arange(len(input_signal)) / fs
            fin_coherent = fin
        
        # 转换
        output_codes = np.zeros(len(input_signal), dtype=np.int64)
        for i, v in enumerate(input_signal):
            self._time = timestamps[i]
            self._sample_index = i
            output_codes[i] = self._convert_single(float(v), self._time)
        
        # 重建
        reconstructed = np.array([self.code_to_voltage(c) for c in output_codes])
        
        self._result = SimulationResult(
            input_signal=input_signal,
            output_codes=output_codes,
            timestamps=timestamps,
            reconstructed=reconstructed,
            events=[],
            metadata={
                "adc_name": self.name,
                "bits": self.bits,
                "vref": self.vref,
                "vmin": self.vmin,
                "lsb": self.lsb,
                "fs": fs,
                "fin": fin_coherent if signal == "sine" and input_voltage is None else fin,
                "n_samples": len(input_signal),
                "num_stages": self.num_stages,
                "stage_bits": [adc.bits for adc in self._stages],
                "gains": self._gains,
            },
        )
        
        return self._result
    
    def sim_auto(
        self,
        fs: float = 1e6,
        n_samples: int = 4096,
    ) -> Dict[str, Any]:
        """自动优化仿真参数"""
        best_enob = -np.inf
        best_params = {}
        
        for fin_ratio in [7, 11, 13, 17, 19, 23]:
            fin = fin_ratio * fs / n_samples
            for amp_ratio in [0.45, 0.48, 0.49, 0.498]:
                amplitude = amp_ratio * (self.vref - self.vmin)
                self.sim(n_samples=n_samples, fs=fs, fin=fin, amplitude=amplitude)
                enob = self.enob()
                if enob > best_enob:
                    best_enob = enob
                    best_params = {
                        'best_fin': fin,
                        'best_amplitude': amplitude,
                        'best_enob': enob,
                        'generations': 1,
                    }
        
        # 用最佳参数再仿真一次
        self.sim(n_samples=n_samples, fs=fs, 
                 fin=best_params['best_fin'], 
                 amplitude=best_params['best_amplitude'])
        return best_params
    
    # =========================================================================
    # 分析方法（与ADConverter保持一致的接口）
    # =========================================================================
    
    def spectrum(
        self,
        result: Optional[SimulationResult] = None,
        *,
        window: str = "hann",
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[float, float] = (3.5, 2.5),
    ) -> Tuple[NDArray, NDArray, Dict[str, float]]:
        """
        计算并绘制频谱。
        
        Returns
        -------
        freqs : NDArray
            频率 (Hz)
        spectrum_db : NDArray
            功率谱 (dB)
        metrics : Dict[str, float]
            指标 (SNR, SFDR, ENOB, THD)
        """
        from .._analysis_impl import compute_spectrum, auto_freq_unit
        
        if result is None:
            if self._result is None:
                raise ValueError("No simulation result. Run sim() first.")
            result = self._result
        
        freqs, spectrum_db, metrics = compute_spectrum(
            result.output_codes,
            self.bits,
            result.metadata.get('fs', 1e6),
            window=window,
        )
        
        if show or save:
            import matplotlib.pyplot as plt
            
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 9,
                'axes.linewidth': 0.6,
                'lines.linewidth': 0.8,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
            })
            
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            f_scaled, f_unit = auto_freq_unit(freqs)
            
            ax.plot(f_scaled, spectrum_db, 'k-', linewidth=0.6)
            ax.set_xlabel(f'Frequency ({f_unit})', fontsize=8)
            ax.set_ylabel('Magnitude (dB)', fontsize=8)
            ax.set_title(f'{self.name}', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
            ax.set_xlim([0, f_scaled[-1]])
            ax.set_ylim([max(-120, np.min(spectrum_db) - 10), 10])
            
            textstr = '\n'.join([
                f"SNR = {metrics['snr']:.1f} dB",
                f"SFDR = {metrics['sfdr']:.1f} dB", 
                f"ENOB = {metrics['enob']:.2f} bits",
            ])
            props = dict(boxstyle='square,pad=0.3', facecolor='white', 
                        edgecolor='black', linewidth=0.5)
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right', 
                    bbox=props, family='monospace')
            
            plt.tight_layout()
            
            if save:
                plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            if show:
                plt.show()
        
        return freqs, spectrum_db, metrics
    
    def enob(self, result: Optional[SimulationResult] = None) -> float:
        """计算ENOB"""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['enob']
    
    def snr(self, result: Optional[SimulationResult] = None) -> float:
        """计算SNR (dB)"""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['snr']
    
    def sfdr(self, result: Optional[SimulationResult] = None) -> float:
        """计算SFDR (dB)"""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['sfdr']
    
    def thd(self, result: Optional[SimulationResult] = None) -> float:
        """计算THD (dB)"""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['thd']
    
    def inl(
        self,
        result: Optional[SimulationResult] = None,
        *,
        plot: bool = False,
    ) -> NDArray:
        """计算INL"""
        from .._analysis_impl import compute_inl_dnl
        
        if result is None:
            if self._result is None:
                self.sim(signal='ramp', n_samples=4096)
            result = self._result
        
        inl_vals, _ = compute_inl_dnl(result.output_codes, self.bits)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            ax.plot(inl_vals, 'b-', linewidth=0.8)
            ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
            ax.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Code')
            ax.set_ylabel('INL (LSB)')
            ax.set_title(f'{self.name} INL')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return inl_vals
    
    def dnl(
        self,
        result: Optional[SimulationResult] = None,
        *,
        plot: bool = False,
    ) -> NDArray:
        """计算DNL"""
        from .._analysis_impl import compute_inl_dnl
        
        if result is None:
            if self._result is None:
                self.sim(signal='ramp', n_samples=4096)
            result = self._result
        
        _, dnl_vals = compute_inl_dnl(result.output_codes, self.bits)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            ax.plot(dnl_vals, 'b-', linewidth=0.8)
            ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
            ax.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Code')
            ax.set_ylabel('DNL (LSB)')
            ax.set_title(f'{self.name} DNL')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return dnl_vals
    
    def plot(
        self,
        result: Optional[SimulationResult] = None,
        *,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[float, float] = (7.16, 5.0),
    ) -> Any:
        """
        绘制完整仿真结果（时域+频域+数字输出+误差）。
        """
        import matplotlib.pyplot as plt
        from .._analysis_impl import auto_time_unit
        
        if result is None:
            if self._result is None:
                raise ValueError("No simulation result. Run sim() first.")
            result = self._result
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 9,
            'axes.linewidth': 0.6,
            'lines.linewidth': 0.8,
            'grid.linewidth': 0.4,
            'grid.alpha': 0.4,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
        })
        
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        fig.suptitle(f"{self.name}", fontsize=10, fontweight='bold')
        
        t_scaled, t_unit = auto_time_unit(result.timestamps)
        
        # 时域
        ax1 = axes[0, 0]
        ax1.plot(t_scaled, result.input_signal, 'k-', linewidth=0.8, label='Input')
        ax1.plot(t_scaled, result.reconstructed, 'k--', linewidth=0.6, label='Reconstructed')
        ax1.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax1.set_ylabel('Voltage (V)', fontsize=8)
        ax1.set_title('Time Domain', fontsize=9)
        ax1.legend(loc='upper right', fontsize=7, frameon=True, edgecolor='black')
        ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # 数字输出
        ax2 = axes[0, 1]
        ax2.step(t_scaled, result.output_codes, 'k-', linewidth=0.6, where='mid')
        ax2.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax2.set_ylabel('Code', fontsize=8)
        ax2.set_title('Digital Output', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # 量化误差
        ax3 = axes[1, 0]
        error = result.input_signal - result.reconstructed
        ax3.plot(t_scaled, error / self.lsb, 'k-', linewidth=0.6)
        ax3.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax3.set_ylabel('Error (LSB)', fontsize=8)
        ax3.set_title('Quantization Error', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # 频谱
        ax4 = axes[1, 1]
        freqs, spectrum_db, metrics = self.spectrum(result, show=False)
        from .._analysis_impl import auto_freq_unit
        f_scaled, f_unit = auto_freq_unit(freqs)
        ax4.plot(f_scaled, spectrum_db, 'k-', linewidth=0.6)
        ax4.set_xlabel(f'Frequency ({f_unit})', fontsize=8)
        ax4.set_ylabel('Power (dB)', fontsize=8)
        ax4.set_title('Spectrum', fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        ax4.set_xlim([0, f_scaled[-1]])
        
        textstr = f"ENOB={metrics['enob']:.2f}\nSNR={metrics['snr']:.1f}dB"
        props = dict(boxstyle='square,pad=0.3', facecolor='white', 
                    edgecolor='black', linewidth=0.5)
        ax4.text(0.97, 0.97, textstr, transform=ax4.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right', 
                bbox=props, family='monospace')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
    
    def report(self) -> None:
        """打印完整报告"""
        if self._result is None:
            raise ValueError("No simulation result. Run sim() first.")
        
        print(f"\n{'='*60}")
        print(f"Pipeline Report: {self.name}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Total bits: {self.bits}")
        print(f"  Stages: {self.num_stages}")
        for i, adc in enumerate(self._stages):
            gain_str = f", gain={self._gains[i]:.0f}" if i < len(self._gains) else ""
            print(f"    Stage {i}: {adc.name} ({adc.bits} bits){gain_str}")
        print(f"\nPerformance:")
        print(f"  ENOB: {self.enob():.2f} bits")
        print(f"  SNR:  {self.snr():.2f} dB")
        print(f"  SFDR: {self.sfdr():.2f} dB")
        print(f"  THD:  {self.thd():.2f} dB")
        print(f"{'='*60}\n")
    
    def get_stage_info(self) -> List[Dict[str, Any]]:
        """获取各级信息"""
        info = []
        for i, adc in enumerate(self._stages):
            info.append({
                'index': i,
                'bits': adc.bits,
                'adc_name': adc.name,
                'gain': self._gains[i] if i < len(self._gains) else 1.0,
            })
        return info


# =============================================================================
# 保留PipelineADC作为别名（向后兼容）
# =============================================================================

# 为了向后兼容，PipelineADC现在指向Pipeline
# 如果需要旧的继承ADConverter的行为，可以自行包装
PipelineADC = Pipeline
