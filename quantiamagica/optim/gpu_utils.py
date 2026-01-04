"""
GPU批量仿真工具 - 用于遗传算法的GPU加速

提供与现有ADC类兼容的GPU批量仿真功能。
支持 SigmaDeltaADC, SARADC, PipelineADC 的GPU加速优化。
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class GPUBatchSimulator:
    """
    GPU批量ADC仿真器基类
    
    用于在GPU上并行仿真多个ADC配置，大幅加速遗传算法优化。
    """
    
    def __init__(self, device: Optional[str] = None):
        if not HAS_TORCH:
            raise ImportError("需要安装PyTorch: pip install torch")
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._cache = {}
    
    def _get_cached_signal(self, key: str, n_samples: int, fs: float, 
                           fin: float, amplitude: float, offset: float = 0.5) -> torch.Tensor:
        """获取或创建缓存的信号"""
        cache_key = (key, n_samples, fs, fin)
        if cache_key not in self._cache:
            t = torch.arange(n_samples, device=self.device, dtype=torch.float32) / fs
            # 相干采样
            n_periods = max(1, int(round(fin * n_samples / fs)))
            fin_coherent = n_periods * fs / n_samples
            base_signal = torch.sin(2 * np.pi * fin_coherent * t)
            self._cache[cache_key] = base_signal
        return self._cache[cache_key]


class GPUSigmaDeltaSimulator(GPUBatchSimulator):
    """
    GPU批量Sigma-Delta ADC仿真器
    
    在GPU上并行仿真多个SD ADC配置，使用与CPU版本相同的算法。
    """
    
    def __init__(self, order: int, bits: int, osr: int, fs: float, 
                 n_samples: int, fin: float, device: Optional[str] = None):
        super().__init__(device)
        self.order = order
        self.bits = bits
        self.osr = osr
        self.fs = fs
        self.n_samples = n_samples
        self.fin = fin
        
        # 相干采样
        n_periods = max(1, int(round(fin * n_samples / fs)))
        self.fin_coherent = n_periods * fs / n_samples
    
    def simulate_batch(self, params_list: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量仿真SD ADC
        
        Parameters
        ----------
        params_list : List[Dict]
            每个dict包含: c1..cN, a1..aN, amplitude
        
        Returns
        -------
        bitstreams : torch.Tensor [batch, n_samples]
            比特流输出
        enobs : torch.Tensor [batch]
            ENOB值
        """
        batch_size = len(params_list)
        
        # 提取参数
        c_keys = [f'c{i+1}' for i in range(self.order)]
        a_keys = [f'a{i+1}' for i in range(self.order)]
        
        c_vals = torch.tensor(
            [[p.get(k, 0.5) for k in c_keys] for p in params_list],
            device=self.device, dtype=torch.float32
        )
        a_vals = torch.tensor(
            [[p.get(k, 1.0) for k in a_keys] for p in params_list],
            device=self.device, dtype=torch.float32
        )
        amplitudes = torch.tensor(
            [p.get('amplitude', 0.3) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        
        # 生成输入信号
        t = torch.arange(self.n_samples, device=self.device, dtype=torch.float32) / self.fs
        omega = 2 * np.pi * self.fin_coherent
        # 归一化到 [-1, 1]
        signals = amplitudes.unsqueeze(1) * torch.sin(omega * t.unsqueeze(0))
        
        # 初始化
        states = torch.zeros(batch_size, self.order, device=self.device, dtype=torch.float32)
        bitstream = torch.zeros(batch_size, self.n_samples, device=self.device, dtype=torch.float32)
        y_prev = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        
        # 仿真循环
        for i in range(self.n_samples):
            x = signals[:, i]
            
            # CIFB拓扑积分器
            for j in range(self.order):
                if j == 0:
                    delta = c_vals[:, j] * (x - a_vals[:, j] * y_prev)
                else:
                    delta = c_vals[:, j] * (states[:, j-1] - a_vals[:, j] * y_prev)
                states[:, j] = states[:, j] + delta
            
            # 量化器
            quant_in = states[:, -1]
            if self.bits == 1:
                y = torch.sign(quant_in)
                y = torch.where(y == 0, torch.ones_like(y), y)
            else:
                n_levels = 2 ** self.bits
                normalized = (quant_in + 1) / 2
                code = torch.clamp(torch.round(normalized * (n_levels - 1)), 0, n_levels - 1)
                y = code / (n_levels - 1) * 2 - 1
            
            bitstream[:, i] = y
            y_prev = y
        
        # 计算ENOB
        enobs = self._compute_enob_batch(bitstream)
        
        return bitstream, enobs
    
    def _compute_enob_batch(self, bitstream: torch.Tensor) -> torch.Tensor:
        """
        批量计算ENOB，使用与CPU版本相同的算法
        
        关键：必须在带宽内计算噪声，并正确处理信号功率
        """
        batch_size, n_samples = bitstream.shape
        bandwidth = self.fs / (2 * self.osr)
        
        # 汉宁窗
        win = torch.hann_window(n_samples, device=self.device, dtype=torch.float32)
        S1 = win.sum()
        
        # 去均值 + 加窗
        mean_val = bitstream.mean(dim=1, keepdim=True)
        windowed = (bitstream - mean_val) * win.unsqueeze(0)
        
        # FFT
        spectrum = torch.fft.rfft(windowed, dim=1)
        power = (torch.abs(spectrum) ** 2) / (S1 ** 2)
        power[:, 1:-1] *= 2  # 单边谱校正
        
        # 信号bin (相干采样应该是精确的整数bin)
        sig_bin = int(round(self.fin_coherent * n_samples / self.fs))
        sig_bin = max(1, min(sig_bin, power.shape[1] - 3))
        
        # 信号功率 (信号bin周围5个bin)
        sig_start = max(1, sig_bin - 2)
        sig_end = min(power.shape[1], sig_bin + 3)
        sig_power = power[:, sig_start:sig_end].sum(dim=1)
        
        # 带宽内噪声功率 (排除DC和信号)
        bw_bin = int(bandwidth * n_samples / self.fs)
        bw_bin = max(bw_bin, sig_end + 1)  # 确保至少包含信号之后的一些bin
        bw_bin = min(bw_bin, power.shape[1])
        
        # 计算带宽内总功率，然后减去信号功率
        total_power_in_bw = power[:, 1:bw_bin].sum(dim=1)
        noise_power = total_power_in_bw - sig_power
        noise_power = torch.clamp(noise_power, min=1e-20)
        
        # SNR和ENOB
        snr_db = 10 * torch.log10(sig_power / noise_power)
        enob = (snr_db - 1.76) / 6.02
        
        # 处理异常值 - 限制在合理范围内
        # 2阶SD ADC理论上限约 2.5*log2(OSR) + N ≈ 2.5*6 + 1 = 16 bits
        max_theoretical = (self.order + 0.5) * np.log2(self.osr) + self.bits + 2
        enob = torch.where(
            torch.isfinite(enob) & (enob > -50) & (enob < max_theoretical), 
            enob, 
            torch.full_like(enob, -100.0)
        )
        
        return enob
    
    def evaluate_batch(self, params_list: List[Dict[str, Any]]) -> List[float]:
        """
        批量评估SD ADC，返回ENOB列表
        """
        _, enobs = self.simulate_batch(params_list)
        return enobs.cpu().numpy().tolist()


class GPUSARSimulator(GPUBatchSimulator):
    """
    GPU批量SAR ADC仿真器
    """
    
    def __init__(self, bits: int, fs: float, n_samples: int, fin: float,
                 vref: float = 1.0, device: Optional[str] = None):
        super().__init__(device)
        self.bits = bits
        self.fs = fs
        self.n_samples = n_samples
        self.fin = fin
        self.vref = vref
        self.lsb = vref / (2 ** bits)
        
        # 相干采样
        n_periods = max(1, int(round(fin * n_samples / fs)))
        self.fin_coherent = n_periods * fs / n_samples
    
    def simulate_batch(self, params_list: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量仿真SAR ADC
        
        Parameters
        ----------
        params_list : List[Dict]
            每个dict可包含: amplitude, sampling_noise, comparator_offset, cap_mismatch
        
        Returns
        -------
        codes : torch.Tensor [batch, n_samples]
            输出码
        enobs : torch.Tensor [batch]
            ENOB值
        """
        batch_size = len(params_list)
        
        # 提取参数
        amplitudes = torch.tensor(
            [p.get('amplitude', 0.4) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        sampling_noise = torch.tensor(
            [p.get('sampling_noise', 0.0) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        comp_offset = torch.tensor(
            [p.get('comparator_offset', 0.0) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        
        # 生成输入信号
        t = torch.arange(self.n_samples, device=self.device, dtype=torch.float32) / self.fs
        omega = 2 * np.pi * self.fin_coherent
        offset = self.vref / 2
        signals = offset + amplitudes.unsqueeze(1) * torch.sin(omega * t.unsqueeze(0))
        
        # 添加采样噪声
        if sampling_noise.max() > 0:
            noise = torch.randn_like(signals) * sampling_noise.unsqueeze(1)
            signals = signals + noise
        
        # SAR转换 (理想情况)
        signals_clipped = torch.clamp(signals, 0, self.vref - self.lsb)
        codes = torch.floor(signals_clipped / self.lsb).long()
        
        # 添加比较器偏移效应
        if comp_offset.max() > 0:
            offset_codes = torch.floor(comp_offset.unsqueeze(1) / self.lsb).long()
            codes = codes + offset_codes
            codes = torch.clamp(codes, 0, 2**self.bits - 1)
        
        # 重建电压
        reconstructed = codes.float() * self.lsb + self.lsb / 2
        
        # 计算ENOB
        enobs = self._compute_enob_batch(signals, reconstructed)
        
        return codes, enobs
    
    def _compute_enob_batch(self, input_signal: torch.Tensor, 
                            reconstructed: torch.Tensor) -> torch.Tensor:
        """批量计算ENOB"""
        batch_size, n_samples = input_signal.shape
        
        # 汉宁窗
        win = torch.hann_window(n_samples, device=self.device, dtype=torch.float32)
        S1 = win.sum()
        
        # 误差信号
        error = reconstructed - input_signal
        error_windowed = (error - error.mean(dim=1, keepdim=True)) * win.unsqueeze(0)
        
        # 输入信号功率
        input_windowed = (input_signal - input_signal.mean(dim=1, keepdim=True)) * win.unsqueeze(0)
        input_spectrum = torch.fft.rfft(input_windowed, dim=1)
        input_power = (torch.abs(input_spectrum) ** 2).sum(dim=1)
        
        # 噪声功率
        error_spectrum = torch.fft.rfft(error_windowed, dim=1)
        noise_power = (torch.abs(error_spectrum) ** 2).sum(dim=1)
        noise_power = torch.clamp(noise_power, min=1e-20)
        
        # SNDR和ENOB
        sndr_db = 10 * torch.log10(input_power / noise_power)
        enob = (sndr_db - 1.76) / 6.02
        
        enob = torch.where(torch.isfinite(enob) & (enob > -50) & (enob < 50),
                          enob, torch.full_like(enob, -100.0))
        
        return enob
    
    def evaluate_batch(self, params_list: List[Dict[str, Any]]) -> List[float]:
        """批量评估SAR ADC"""
        _, enobs = self.simulate_batch(params_list)
        return enobs.cpu().numpy().tolist()


class GPUPipelineSimulator(GPUBatchSimulator):
    """
    GPU批量Pipeline ADC仿真器
    """
    
    def __init__(self, bits: int, stages: int, bits_per_stage: int,
                 fs: float, n_samples: int, fin: float,
                 vref: float = 1.0, gain: float = 2.0,
                 device: Optional[str] = None):
        super().__init__(device)
        self.bits = bits
        self.stages = stages
        self.bits_per_stage = bits_per_stage
        self.fs = fs
        self.n_samples = n_samples
        self.fin = fin
        self.vref = vref
        self.gain = gain
        
        # 相干采样
        n_periods = max(1, int(round(fin * n_samples / fs)))
        self.fin_coherent = n_periods * fs / n_samples
    
    def simulate_batch(self, params_list: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量仿真Pipeline ADC
        
        Parameters
        ----------
        params_list : List[Dict]
            每个dict可包含: amplitude, gain_error, comparator_offset
        """
        batch_size = len(params_list)
        
        # 提取参数
        amplitudes = torch.tensor(
            [p.get('amplitude', 0.4) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        gain_errors = torch.tensor(
            [p.get('gain_error', 0.0) for p in params_list],
            device=self.device, dtype=torch.float32
        )
        
        # 生成输入信号
        t = torch.arange(self.n_samples, device=self.device, dtype=torch.float32) / self.fs
        omega = 2 * np.pi * self.fin_coherent
        offset = self.vref / 2
        signals = offset + amplitudes.unsqueeze(1) * torch.sin(omega * t.unsqueeze(0))
        
        # Pipeline转换
        codes = torch.zeros(batch_size, self.n_samples, device=self.device, dtype=torch.long)
        residue = signals.clone()
        
        for stage in range(self.stages):
            stage_lsb = self.vref / (2 ** self.bits_per_stage)
            stage_codes = torch.clamp(torch.floor(residue / stage_lsb), 0, 2**self.bits_per_stage - 1).long()
            
            # DAC输出
            dac_out = stage_codes.float() * stage_lsb + stage_lsb / 2
            
            # 残差放大 (带增益误差)
            effective_gain = self.gain * (1 + gain_errors.unsqueeze(1))
            residue = (residue - dac_out) * effective_gain
            residue = torch.clamp(residue, 0, self.vref)
            
            # 累加码
            shift = self.bits_per_stage * (self.stages - 1 - stage)
            codes = codes + (stage_codes << shift)
        
        codes = torch.clamp(codes, 0, 2**self.bits - 1)
        
        # 重建电压
        lsb = self.vref / (2 ** self.bits)
        reconstructed = codes.float() * lsb + lsb / 2
        
        # 计算ENOB
        enobs = self._compute_enob_batch(signals, reconstructed)
        
        return codes, enobs
    
    def _compute_enob_batch(self, input_signal: torch.Tensor,
                            reconstructed: torch.Tensor) -> torch.Tensor:
        """批量计算ENOB"""
        batch_size, n_samples = input_signal.shape
        
        win = torch.hann_window(n_samples, device=self.device, dtype=torch.float32)
        S1 = win.sum()
        
        error = reconstructed - input_signal
        error_windowed = (error - error.mean(dim=1, keepdim=True)) * win.unsqueeze(0)
        input_windowed = (input_signal - input_signal.mean(dim=1, keepdim=True)) * win.unsqueeze(0)
        
        input_spectrum = torch.fft.rfft(input_windowed, dim=1)
        input_power = (torch.abs(input_spectrum) ** 2).sum(dim=1)
        
        error_spectrum = torch.fft.rfft(error_windowed, dim=1)
        noise_power = (torch.abs(error_spectrum) ** 2).sum(dim=1)
        noise_power = torch.clamp(noise_power, min=1e-20)
        
        sndr_db = 10 * torch.log10(input_power / noise_power)
        enob = (sndr_db - 1.76) / 6.02
        
        enob = torch.where(torch.isfinite(enob) & (enob > -50) & (enob < 50),
                          enob, torch.full_like(enob, -100.0))
        
        return enob
    
    def evaluate_batch(self, params_list: List[Dict[str, Any]]) -> List[float]:
        """批量评估Pipeline ADC"""
        _, enobs = self.simulate_batch(params_list)
        return enobs.cpu().numpy().tolist()


# =============================================================================
# 便捷函数
# =============================================================================

def create_batch_sd_evaluator(order: int, osr: int, fs: float, 
                              n_samples: int, fin: float, bits: int = 1) -> Callable:
    """创建SD ADC批量评估器"""
    sim = GPUSigmaDeltaSimulator(order, bits, osr, fs, n_samples, fin)
    return sim.evaluate_batch


def create_batch_sar_evaluator(bits: int, fs: float, n_samples: int, 
                               fin: float, vref: float = 1.0) -> Callable:
    """创建SAR ADC批量评估器"""
    sim = GPUSARSimulator(bits, fs, n_samples, fin, vref)
    return sim.evaluate_batch


def create_batch_pipeline_evaluator(bits: int, stages: int, bits_per_stage: int,
                                    fs: float, n_samples: int, fin: float,
                                    vref: float = 1.0, gain: float = 2.0) -> Callable:
    """创建Pipeline ADC批量评估器"""
    sim = GPUPipelineSimulator(bits, stages, bits_per_stage, fs, n_samples, fin, vref, gain)
    return sim.evaluate_batch


def create_simple_batch_evaluator(fitness_fn: Callable[[Dict[str, Any]], float]) -> Callable:
    """将单个适应度函数包装为批量版本（CPU多线程）"""
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    n_workers = (os.cpu_count() or 4) * 2
    
    def batch_evaluate(params_list: List[Dict[str, Any]]) -> List[float]:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(fitness_fn, params_list))
        return results
    
    return batch_evaluate
