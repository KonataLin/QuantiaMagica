"""
Sigma-Delta ADC - 简洁事件驱动架构

核心设计：只有一个QuantizerEvent
- 事件提供：当前输入x[n]、前一输出y[n-1]、内部积分器状态
- 用户可修改：量化器输入值（实现任意积分器/反馈拓扑）
- 内置默认1阶/2阶CIFB实现

Example
-------
>>> from quantiamagica import SigmaDeltaADC, QuantizerEvent
>>> 
>>> sd = SigmaDeltaADC(order=1, osr=64)
>>> sd.sim(n_samples=8192, fs=1e6, fin=1e3, amplitude=0.4)
>>> print(f"ENOB: {sd.enob():.2f}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..core.events import Event, Cancellable
from ..core.base import ADConverter


@dataclass
class QuantizerEvent(Event, Cancellable):
    """
    量化器事件 - Sigma-Delta调制器的核心事件
    
    Attributes
    ----------
    input_signal : float
        当前输入信号 x[n] (归一化到 [-1, 1])
    prev_output : float
        前一时刻量化器输出 y[n-1]
    integrator_states : List[float]
        各级积分器当前状态（只读参考）
    quantizer_input : float
        量化器输入值 (可修改！修改此值来实现自定义拓扑)
    output : float
        量化器输出，可修改
    output_code : int
        数字码，可修改
    bits : int
        量化器位数
    offset : float
        比较器偏移，可修改
    noise_sigma : float
        比较器噪声，可修改
    """
    input_signal: float = 0.0
    prev_output: float = 0.0
    integrator_states: List[float] = field(default_factory=list)
    quantizer_input: float = 0.0
    output: float = 0.0
    output_code: int = 0
    bits: int = 1
    offset: float = 0.0
    noise_sigma: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


class SigmaDeltaADC(ADConverter):
    """
    Sigma-Delta ADC - 简洁灵活的事件驱动架构
    
    Parameters
    ----------
    order : int
        调制器阶数 (1 或 2)，default 1
    bits : int
        量化器位数，default 1
    osr : int
        过采样率，default 64
    vref : float
        参考电压，default 1.0
    """
    
    def __init__(
        self,
        order: int = 1,
        bits: int = 1,
        osr: int = 64,
        vref: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(bits, vref, 0.0, name or f"SD{order}-{bits}b-OSR{osr}")
        
        self.order = order
        self.osr = osr
        self._integrator_states = [0.0] * order
        self._prev_output = 0.0
        self._bitstream: List[float] = []
        self._raw_codes: List[int] = []
    
    @property
    def integrator_states(self) -> List[float]:
        return list(self._integrator_states)
    
    @property
    def theoretical_enob_gain(self) -> float:
        if self.order == 1:
            return 1.5 * np.log2(self.osr) - 0.86
        elif self.order == 2:
            return 2.5 * np.log2(self.osr) - 2.14
        return (self.order + 0.5) * np.log2(self.osr)
    
    def reset(self) -> None:
        super().reset()
        self._integrator_states = [0.0] * self.order
        self._prev_output = 0.0
        self._bitstream = []
        self._raw_codes = []
    
    def _quantize(self, value: float) -> Tuple[int, float]:
        if self.bits == 1:
            code = 1 if value >= 0 else 0
            output = 1.0 if code == 1 else -1.0
        else:
            n_levels = 2 ** self.bits
            normalized = (value + 1) / 2
            code = int(np.clip(np.round(normalized * (n_levels - 1)), 0, n_levels - 1))
            output = code / (n_levels - 1) * 2 - 1
        return code, output
    
    def _default_integrator(self, x: float, y_prev: float) -> float:
        if self.order == 1:
            self._integrator_states[0] += x - y_prev
            return self._integrator_states[0]
        elif self.order == 2:
            u1 = self._integrator_states[0]
            u2 = self._integrator_states[1]
            u1_new = u1 + x - y_prev
            u2_new = u2 + u1_new - 2 * y_prev
            self._integrator_states[0] = u1_new
            self._integrator_states[1] = u2_new
            return u2_new
        else:
            for i in range(self.order):
                if i == 0:
                    self._integrator_states[i] += x - y_prev
                else:
                    self._integrator_states[i] += self._integrator_states[i-1] - y_prev
            return self._integrator_states[-1]
    
    def _modulator_step(self, x: float, timestamp: float) -> int:
        y_prev = self._prev_output
        quant_input = self._default_integrator(x, y_prev)
        code, output = self._quantize(quant_input)
        
        event = QuantizerEvent(
            timestamp=timestamp,
            input_signal=x,
            prev_output=y_prev,
            integrator_states=list(self._integrator_states),
            quantizer_input=quant_input,
            output=output,
            output_code=code,
            bits=self.bits,
        )
        self.fire(event)
        
        if not event.cancelled:
            # 检查用户是否修改了quantizer_input
            user_modified_input = (event.quantizer_input != quant_input)
            
            if user_modified_input:
                effective_input = event.quantizer_input
            else:
                effective_input = quant_input
            
            # 应用偏移和噪声
            effective_input -= event.offset
            if event.noise_sigma > 0:
                effective_input += np.random.normal(0, event.noise_sigma)
            
            # 如果输入被修改，重新量化（忽略原始的output/output_code）
            if user_modified_input or event.offset != 0 or event.noise_sigma > 0:
                code, output = self._quantize(effective_input)
            
            # 只有当用户直接修改output/output_code时才覆盖
            # （且用户没有修改quantizer_input）
            if not user_modified_input:
                if event.output_code != code:
                    code = event.output_code
                if event.output != output:
                    output = event.output
        
        self._prev_output = output
        self._bitstream.append(output)
        self._raw_codes.append(code)
        return code
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        x = (voltage - self.vmin) / (self.vref - self.vmin) * 2 - 1
        return self._modulator_step(x, timestamp)
    
    def get_bitstream(self) -> NDArray[np.float64]:
        return np.array(self._bitstream, dtype=np.float64)
    
    def get_raw_codes(self) -> NDArray[np.int64]:
        return np.array(self._raw_codes, dtype=np.int64)
    
    def decimate(self, method: str = 'average') -> NDArray[np.float64]:
        data = np.array(self._bitstream, dtype=np.float64)
        n_out = len(data) // self.osr
        if method == 'average':
            return np.mean(data[:n_out * self.osr].reshape(n_out, self.osr), axis=1)
        raise ValueError(f"Unknown method: {method}")
    
    def enob(self, bandwidth: Optional[float] = None) -> float:
        if self._result is None or len(self._bitstream) == 0:
            return 0.0
        
        bitstream = np.array(self._bitstream, dtype=np.float64)
        fs = self._result.metadata.get('fs', 1.0)
        fin = self._result.metadata.get('fin', fs / 100)
        
        if bandwidth is None:
            bandwidth = fs / (2 * self.osr)
        
        n = len(bitstream)
        win = np.hanning(n)
        S1 = np.sum(win)
        windowed = (bitstream - np.mean(bitstream)) * win
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2 / (S1 ** 2)
        spectrum[1:-1] *= 2
        
        sig_bin = int(round(fin * n / fs))
        sig_bin = max(1, min(sig_bin, len(spectrum) - 3))
        sig_power = np.sum(spectrum[max(1, sig_bin-2):sig_bin+3])
        
        bw_bin = int(bandwidth * n / fs)
        bw_bin = min(bw_bin, len(spectrum))
        noise_power = np.sum(spectrum[1:bw_bin]) - sig_power
        noise_power = max(noise_power, 1e-20)
        
        snr_db = 10 * np.log10(sig_power / noise_power)
        return (snr_db - 1.76) / 6.02
    
    def snr(self, bandwidth: Optional[float] = None) -> float:
        return self.enob(bandwidth) * 6.02 + 1.76
