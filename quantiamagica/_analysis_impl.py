"""
Internal analysis implementation for ADC metrics.

This module contains the core computation functions used by ADConverter
and the analysis module.
"""

from typing import Dict, Tuple, Any
import numpy as np
from numpy.typing import NDArray


def auto_time_unit(time_array: NDArray) -> Tuple[NDArray, str]:
    """
    Auto-scale time array and return appropriate unit label.
    
    Parameters
    ----------
    time_array : NDArray
        Time values in seconds.
    
    Returns
    -------
    scaled_time : NDArray
        Scaled time values.
    unit_label : str
        Unit string (e.g., 'ns', 'μs', 'ms', 's').
    """
    max_time = np.max(np.abs(time_array))
    
    if max_time == 0:
        return time_array, 's'
    elif max_time < 1e-6:
        return time_array * 1e9, 'ns'
    elif max_time < 1e-3:
        return time_array * 1e6, 'μs'
    elif max_time < 1:
        return time_array * 1e3, 'ms'
    else:
        return time_array, 's'


def auto_freq_unit(freq_array: NDArray) -> Tuple[NDArray, str]:
    """
    Auto-scale frequency array and return appropriate unit label.
    
    Parameters
    ----------
    freq_array : NDArray
        Frequency values in Hz.
    
    Returns
    -------
    scaled_freq : NDArray
        Scaled frequency values.
    unit_label : str
        Unit string (e.g., 'Hz', 'kHz', 'MHz', 'GHz').
    """
    max_freq = np.max(np.abs(freq_array))
    
    if max_freq == 0:
        return freq_array, 'Hz'
    elif max_freq < 1e3:
        return freq_array, 'Hz'
    elif max_freq < 1e6:
        return freq_array / 1e3, 'kHz'
    elif max_freq < 1e9:
        return freq_array / 1e6, 'MHz'
    else:
        return freq_array / 1e9, 'GHz'


def apply_jssc_style():
    """Apply IEEE JSSC publication style to matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.6,
        'lines.linewidth': 0.8,
        'grid.linewidth': 0.4,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fontsize': 7,
    })


def plot_spectrum(
    codes: NDArray,
    fs: float,
    bits: int = None,
    bandwidth: float = None,
    title: str = "Output Spectrum",
    show_metrics: bool = True,
    save: str = None,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot ADC output spectrum with automatic unit scaling.
    
    集成的频谱绘图函数，自动处理单位和标注。
    
    Parameters
    ----------
    codes : NDArray
        ADC output codes or sigma-delta bitstream.
    fs : float
        Sampling frequency (Hz).
    bits : int, optional
        ADC resolution (for normalization). If None, auto-detect.
    bandwidth : float, optional
        Signal bandwidth for in-band analysis (Hz). If None, use fs/2.
    title : str
        Plot title.
    show_metrics : bool
        Show SNR/ENOB annotation box.
    save : str, optional
        Save figure to file.
    show : bool
        Display figure.
    
    Returns
    -------
    Dict with 'snr', 'enob', 'freqs', 'power_db'
    """
    import matplotlib.pyplot as plt
    apply_jssc_style()
    
    n = len(codes)
    codes_float = codes.astype(np.float64)
    
    # Auto-detect bits if not provided
    if bits is None:
        max_code = np.max(codes)
        bits = int(np.ceil(np.log2(max_code + 1))) if max_code > 1 else 1
    
    # Normalize and remove DC
    if bits == 1 or np.max(codes) <= 1:
        codes_norm = codes_float - np.mean(codes_float)
    else:
        codes_norm = codes_float / (2**bits) - 0.5
        codes_norm = codes_norm - np.mean(codes_norm)
    
    # FFT with Hanning window
    win = np.hanning(n)
    S1 = np.sum(win)
    fft_result = np.fft.rfft(codes_norm * win)
    power = (np.abs(fft_result) ** 2) / (S1 ** 2)
    power[1:-1] *= 2
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    power_db = 10 * np.log10(np.maximum(power, 1e-20))
    
    # Find signal
    sig_bin = np.argmax(power[1:]) + 1
    sig_power = np.sum(power[max(1, sig_bin-3):sig_bin+4])
    
    # Compute SNR (in-band if bandwidth specified)
    if bandwidth is None:
        bandwidth = fs / 2
    bw_bin = min(int(bandwidth * n / fs), len(power))
    
    noise_power = 0.0
    for i in range(1, bw_bin):
        if abs(i - sig_bin) > 3:
            noise_power += power[i]
    noise_power = max(noise_power, 1e-20)
    
    snr = 10 * np.log10(sig_power / noise_power)
    enob = (snr - 1.76) / 6.02
    
    # Plot
    f_scaled, f_unit = auto_freq_unit(freqs)
    
    fig, ax = plt.subplots(figsize=(4, 2.8), dpi=150)
    ax.plot(f_scaled, power_db, 'k-', linewidth=0.6)
    
    # Mark bandwidth if specified
    if bandwidth < fs / 2:
        bw_scaled = bandwidth / (fs / f_scaled[-1] * 2) if f_scaled[-1] > 0 else bandwidth
        ax.axvline(x=bw_scaled, color='gray', linestyle='--', linewidth=0.5)
        ax.axvspan(0, bw_scaled, alpha=0.1, color='gray')
    
    ax.set_xlabel(f'Frequency ({f_unit})', fontsize=8)
    ax.set_ylabel('Magnitude (dB)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim([0, f_scaled[-1]])
    ax.set_ylim([max(-120, np.min(power_db) - 10), np.max(power_db) + 10])
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    
    if show_metrics:
        textstr = f"SNR={snr:.1f}dB\nENOB={enob:.2f}"
        props = dict(boxstyle='square,pad=0.2', facecolor='white', 
                    edgecolor='black', linewidth=0.4)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right', 
                bbox=props, family='monospace')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return {'snr': snr, 'enob': enob, 'freqs': freqs, 'power_db': power_db}


def plot_comparison(
    data_list: list,
    fs: float,
    labels: list = None,
    bandwidth: float = None,
    title: str = "Spectrum Comparison",
    save: str = None,
    show: bool = True,
):
    """
    Plot multiple spectrums for comparison (e.g., Standard vs NS-SAR).
    
    对比绘图函数，用于比较不同ADC架构的频谱。
    
    Parameters
    ----------
    data_list : list of NDArray
        List of ADC output codes to compare.
    fs : float
        Sampling frequency.
    labels : list of str
        Labels for each dataset.
    bandwidth : float, optional
        Signal bandwidth to highlight.
    title : str
        Overall title.
    save : str
        Save path.
    show : bool
        Display figure.
    """
    import matplotlib.pyplot as plt
    apply_jssc_style()
    
    n_plots = len(data_list)
    if labels is None:
        labels = [f'ADC {i+1}' for i in range(n_plots)]
    
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5*n_plots, 2.8), dpi=150)
    if n_plots == 1:
        axes = [axes]
    
    for idx, (codes, label) in enumerate(zip(data_list, labels)):
        ax = axes[idx]
        n = len(codes)
        codes_float = codes.astype(np.float64)
        codes_norm = codes_float - np.mean(codes_float)
        
        win = np.hanning(n)
        S1 = np.sum(win)
        fft_result = np.fft.rfft(codes_norm * win)
        power = (np.abs(fft_result) ** 2) / (S1 ** 2)
        power[1:-1] *= 2
        freqs = np.fft.rfftfreq(n, 1/fs)
        power_db = 10 * np.log10(np.maximum(power, 1e-20))
        
        f_scaled, f_unit = auto_freq_unit(freqs)
        ax.plot(f_scaled, power_db, 'k-', linewidth=0.5)
        
        if bandwidth and bandwidth < fs / 2:
            bw_scaled = bandwidth * f_scaled[-1] / (fs / 2)
            ax.axvline(x=bw_scaled, color='gray', linestyle='--', linewidth=0.5)
            ax.axvspan(0, bw_scaled, alpha=0.1, color='gray')
        
        # Compute in-band SNR
        sig_bin = np.argmax(power[1:]) + 1
        sig_power = np.sum(power[max(1,sig_bin-3):sig_bin+4])
        bw_bin = int((bandwidth or fs/2) * n / fs)
        noise_power = sum(power[i] for i in range(1, min(bw_bin, len(power))) if abs(i-sig_bin) > 3)
        snr = 10 * np.log10(sig_power / max(noise_power, 1e-20))
        enob = (snr - 1.76) / 6.02
        
        ax.set_xlabel(f'Frequency ({f_unit})', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', fontsize=8)
        ax.set_title(f'{label}\nENOB={enob:.2f}', fontsize=9)
        ax.set_xlim([0, f_scaled[-1]])
        ax.set_ylim([-100, 10])
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    
    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_spectrum(
    codes: NDArray[np.int64],
    bits: int,
    fs: float,
    window: str = "hann",
) -> Tuple[NDArray, NDArray, Dict[str, float]]:
    """
    Compute power spectrum and ADC metrics using IEEE standard approach.
    
    Parameters
    ----------
    codes : NDArray
        Digital output codes.
    bits : int
        ADC resolution.
    fs : float
        Sampling frequency.
    window : str
        Window function.
    
    Returns
    -------
    freqs : NDArray
        Frequency bins.
    spectrum_db : NDArray
        Power spectrum in dB.
    metrics : Dict
        SNR, SFDR, ENOB, THD.
    """
    n = len(codes)
    
    # Normalize codes to [-0.5, 0.5] range
    codes_float = codes.astype(np.float64)
    codes_normalized = codes_float / (2 ** bits) - 0.5
    
    # Apply window
    if window == "hann":
        win = np.hanning(n)
    elif window == "hamming":
        win = np.hamming(n)
    elif window == "blackman":
        win = np.blackman(n)
    elif window == "rectangular" or window == "none":
        win = np.ones(n)
    else:
        win = np.hanning(n)
    
    # Window correction factors
    S1 = np.sum(win)       # Coherent gain
    S2 = np.sum(win ** 2)  # Power sum for noise bandwidth
    ENBW = n * S2 / (S1 ** 2)  # Equivalent Noise Bandwidth (1.5 for Hanning)
    
    # Apply window (no pre-amplification)
    windowed = codes_normalized * win
    
    # FFT
    fft_result = np.fft.rfft(windowed)
    
    # Power spectrum with proper normalization
    # Divide by S1^2 for coherent gain, multiply by 2 for single-sided
    power = (np.abs(fft_result) ** 2) / (S1 ** 2)
    power[1:-1] *= 2  # Single-sided spectrum correction
    
    # Zero out DC
    power[0] = 0
    
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    # Find signal bin (largest peak excluding DC)
    signal_bin = np.argmax(power[1:]) + 1
    
    # Sum signal power over a few bins around peak (to handle spectral leakage)
    half_width = max(2, n // 512)  # Adaptive width
    sig_start = max(1, signal_bin - half_width)
    sig_end = min(len(power), signal_bin + half_width + 1)
    signal_power = np.sum(power[sig_start:sig_end])
    
    # Create mask for signal bins
    signal_mask = np.zeros(len(power), dtype=bool)
    signal_mask[sig_start:sig_end] = True
    
    # Find harmonics and their power
    num_harmonics = 5
    harmonic_power = 0
    for h in range(2, num_harmonics + 2):
        harm_bin = signal_bin * h
        if harm_bin < len(power) - half_width:
            h_start = max(1, harm_bin - half_width)
            h_end = min(len(power), harm_bin + half_width + 1)
            harmonic_power += np.sum(power[h_start:h_end])
            signal_mask[h_start:h_end] = True
    
    # Noise power is everything except signal and harmonics
    noise_power = np.sum(power[~signal_mask])
    
    noise_power = max(noise_power, 1e-20)
    signal_power = max(signal_power, 1e-20)
    
    # Compute metrics
    snr = 10 * np.log10(signal_power / noise_power)
    
    # SFDR: signal to largest spur
    power_copy = power.copy()
    power_copy[signal_mask] = 0
    max_spur_power = np.max(power_copy)
    max_spur_power = max(max_spur_power, 1e-20)
    sfdr = 10 * np.log10(signal_power / max_spur_power)
    
    # THD
    if harmonic_power > 0:
        thd = 10 * np.log10(harmonic_power / signal_power)
    else:
        thd = -100
    
    # SINAD and ENOB
    sinad = 10 * np.log10(signal_power / (noise_power + harmonic_power))
    enob = (sinad - 1.76) / 6.02
    
    # Spectrum in dB for plotting
    power_safe = np.maximum(power, 1e-20)
    spectrum_db = 10 * np.log10(power_safe)
    
    metrics = {
        'snr': snr,
        'sfdr': sfdr,
        'thd': thd,
        'enob': enob,
        'sinad': sinad,
        'signal_bin': signal_bin,
        'signal_freq': freqs[signal_bin],
    }
    
    return freqs, spectrum_db, metrics


def compute_inband_snr(
    codes: NDArray[np.int64],
    bits: int,
    fs: float,
    signal_freq: float,
    bandwidth: float,
) -> Tuple[float, float]:
    """
    Compute in-band SNR for oversampled ADC (useful for NS-SAR, Sigma-Delta).
    
    Parameters
    ----------
    codes : NDArray
        ADC output codes.
    bits : int
        ADC resolution.
    fs : float
        Sampling frequency (Hz).
    signal_freq : float
        Input signal frequency (Hz).
    bandwidth : float
        Signal bandwidth for in-band calculation (Hz).
    
    Returns
    -------
    snr : float
        In-band SNR (dB).
    enob : float
        In-band ENOB (bits).
    """
    n = len(codes)
    
    # Normalize codes and remove DC
    codes_norm = codes.astype(np.float64) / (2**bits) - 0.5
    codes_norm = codes_norm - np.mean(codes_norm)
    
    # FFT with Hanning window
    win = np.hanning(n)
    S1 = np.sum(win)
    windowed = codes_norm * win
    fft_result = np.fft.rfft(windowed)
    power = (np.abs(fft_result) ** 2) / (S1 ** 2)
    power[1:-1] *= 2  # Single-sided correction
    
    freqs = np.fft.rfftfreq(n, 1/fs)
    freq_res = fs / n
    
    # Find signal bin (search around expected frequency)
    expected_bin = int(signal_freq * n / fs)
    search_range = max(5, n // 500)
    start = max(1, expected_bin - search_range)
    end = min(len(power), expected_bin + search_range)
    signal_bin = start + np.argmax(power[start:end])
    
    # Signal power (sum over several bins for spectral leakage)
    sig_width = max(3, n // 1000)
    sig_start = max(1, signal_bin - sig_width)
    sig_end = min(len(power), signal_bin + sig_width + 1)
    signal_power = np.sum(power[sig_start:sig_end])
    
    # In-band noise (only up to bandwidth, excluding signal)
    bw_bin = max(int(bandwidth * n / fs), sig_end + 1)
    noise_power = 0.0
    for i in range(1, min(bw_bin, len(power))):
        if i < sig_start or i >= sig_end:
            noise_power += power[i]
    
    noise_power = max(noise_power, 1e-20)
    signal_power = max(signal_power, 1e-20)
    
    snr = 10 * np.log10(signal_power / noise_power)
    enob = (snr - 1.76) / 6.02
    
    return snr, enob


def compute_inl_dnl(
    codes: NDArray[np.int64],
    bits: int,
) -> Tuple[NDArray, NDArray]:
    """
    Compute INL and DNL from histogram.
    
    Parameters
    ----------
    codes : NDArray
        Output codes from ramp test.
    bits : int
        ADC resolution.
    
    Returns
    -------
    inl : NDArray
        INL values in LSB.
    dnl : NDArray
        DNL values in LSB.
    """
    n_codes = 2 ** bits
    
    hist, _ = np.histogram(codes, bins=np.arange(n_codes + 1) - 0.5)
    
    total = np.sum(hist)
    ideal_count = total / n_codes
    
    if ideal_count == 0:
        return np.zeros(n_codes), np.zeros(n_codes)
    
    dnl = hist / ideal_count - 1.0
    
    inl = np.cumsum(dnl)
    inl = inl - np.mean(inl)
    
    return inl, dnl
