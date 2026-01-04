"""
Analysis Module - ADC Performance Metrics and Analysis Tools.

Provides standalone functions and the Analyzer class for comprehensive
ADC characterization.

Example
-------
>>> from quantiamagica import SARADC
>>> from quantiamagica.analysis import Analyzer
>>> 
>>> adc = SARADC(bits=12)
>>> result = adc.sim()
>>> 
>>> analyzer = Analyzer(result)
>>> print(analyzer.summary())
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..core.base import SimulationResult
from .._analysis_impl import compute_spectrum, compute_inl_dnl


def spectrum(
    codes: NDArray[np.int64],
    bits: int,
    fs: float = 1e6,
    window: str = "hann",
) -> Tuple[NDArray, NDArray, Dict[str, float]]:
    """
    Compute power spectrum and metrics.
    
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
    power_db : NDArray
        Power spectrum in dB.
    metrics : Dict
        SNR, SFDR, ENOB, THD.
    """
    return compute_spectrum(codes, bits, fs, window)


def enob(codes: NDArray[np.int64], bits: int, fs: float = 1e6) -> float:
    """Compute Effective Number of Bits."""
    _, _, metrics = compute_spectrum(codes, bits, fs)
    return metrics['enob']


def snr(codes: NDArray[np.int64], bits: int, fs: float = 1e6) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    _, _, metrics = compute_spectrum(codes, bits, fs)
    return metrics['snr']


def sfdr(codes: NDArray[np.int64], bits: int, fs: float = 1e6) -> float:
    """Compute Spurious-Free Dynamic Range in dB."""
    _, _, metrics = compute_spectrum(codes, bits, fs)
    return metrics['sfdr']


def thd(codes: NDArray[np.int64], bits: int, fs: float = 1e6) -> float:
    """Compute Total Harmonic Distortion in dB."""
    _, _, metrics = compute_spectrum(codes, bits, fs)
    return metrics['thd']


def inl(codes: NDArray[np.int64], bits: int) -> NDArray:
    """Compute Integral Non-Linearity in LSB."""
    inl_vals, _ = compute_inl_dnl(codes, bits)
    return inl_vals


def dnl(codes: NDArray[np.int64], bits: int) -> NDArray:
    """Compute Differential Non-Linearity in LSB."""
    _, dnl_vals = compute_inl_dnl(codes, bits)
    return dnl_vals


class Analyzer:
    """
    Comprehensive ADC performance analyzer.
    
    Provides a unified interface for all ADC metrics and visualizations.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation result to analyze.
    bits : int, optional
        ADC resolution (inferred from result if not provided).
    fs : float, optional
        Sampling frequency.
    
    Example
    -------
    >>> analyzer = Analyzer(result, bits=12)
    >>> print(f"ENOB: {analyzer.enob:.2f}")
    >>> print(f"SNR: {analyzer.snr:.2f} dB")
    >>> analyzer.plot_all()
    """
    
    def __init__(
        self,
        result: SimulationResult,
        bits: Optional[int] = None,
        fs: Optional[float] = None,
    ):
        self.result = result
        self.bits = bits or result.metadata.get('bits', 12)
        self.fs = fs or result.metadata.get('fs', 1e6)
        
        self._metrics: Optional[Dict[str, float]] = None
        self._spectrum: Optional[Tuple[NDArray, NDArray]] = None
    
    def _compute_metrics(self) -> None:
        """Compute all spectral metrics."""
        if self._metrics is None:
            freqs, power_db, metrics = compute_spectrum(
                self.result.output_codes,
                self.bits,
                self.fs,
            )
            self._spectrum = (freqs, power_db)
            self._metrics = metrics
    
    @property
    def enob(self) -> float:
        """Effective Number of Bits."""
        self._compute_metrics()
        return self._metrics['enob']
    
    @property
    def snr(self) -> float:
        """Signal-to-Noise Ratio in dB."""
        self._compute_metrics()
        return self._metrics['snr']
    
    @property
    def sfdr(self) -> float:
        """Spurious-Free Dynamic Range in dB."""
        self._compute_metrics()
        return self._metrics['sfdr']
    
    @property
    def thd(self) -> float:
        """Total Harmonic Distortion in dB."""
        self._compute_metrics()
        return self._metrics['thd']
    
    @property
    def sinad(self) -> float:
        """Signal-to-Noise and Distortion in dB."""
        self._compute_metrics()
        return self._metrics['sinad']
    
    @property
    def inl(self) -> NDArray:
        """Integral Non-Linearity in LSB."""
        inl_vals, _ = compute_inl_dnl(self.result.output_codes, self.bits)
        return inl_vals
    
    @property
    def dnl(self) -> NDArray:
        """Differential Non-Linearity in LSB."""
        _, dnl_vals = compute_inl_dnl(self.result.output_codes, self.bits)
        return dnl_vals
    
    @property
    def inl_max(self) -> float:
        """Maximum INL in LSB."""
        return float(np.max(np.abs(self.inl)))
    
    @property
    def dnl_max(self) -> float:
        """Maximum DNL in LSB."""
        return float(np.max(np.abs(self.dnl)))
    
    def summary(self) -> str:
        """
        Generate summary string of all metrics.
        
        Returns
        -------
        str
            Formatted summary.
        """
        self._compute_metrics()
        return f"""
╔══════════════════════════════════════════╗
║       ADC Performance Summary            ║
╠══════════════════════════════════════════╣
║  Resolution:     {self.bits:2d} bits                 ║
║  Sample Rate:    {self.fs/1e6:.2f} MHz               ║
╠══════════════════════════════════════════╣
║  Dynamic Performance:                    ║
║    ENOB:         {self.enob:6.2f} bits              ║
║    SNR:          {self.snr:6.2f} dB                ║
║    SFDR:         {self.sfdr:6.2f} dB                ║
║    THD:          {self.thd:6.2f} dB                ║
║    SINAD:        {self.sinad:6.2f} dB                ║
╠══════════════════════════════════════════╣
║  Static Performance:                     ║
║    INL (max):    {self.inl_max:6.3f} LSB              ║
║    DNL (max):    {self.dnl_max:6.3f} LSB              ║
╚══════════════════════════════════════════╝
"""
    
    def to_dict(self) -> Dict[str, float]:
        """Export all metrics as dictionary."""
        self._compute_metrics()
        return {
            'bits': self.bits,
            'fs': self.fs,
            'enob': self.enob,
            'snr': self.snr,
            'sfdr': self.sfdr,
            'thd': self.thd,
            'sinad': self.sinad,
            'inl_max': self.inl_max,
            'dnl_max': self.dnl_max,
        }
    
    def plot_all(
        self,
        *,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        Generate comprehensive analysis plots.
        
        Parameters
        ----------
        show : bool
            Display plots.
        save : str, optional
            Save path prefix.
        dpi : int
            Figure DPI.
        """
        import matplotlib.pyplot as plt
        
        self._compute_metrics()
        
        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(self.result.timestamps * 1e6, self.result.input_signal, 
                 'b-', linewidth=0.8, label='Input')
        ax1.plot(self.result.timestamps * 1e6, self.result.reconstructed,
                 'r--', linewidth=0.8, label='Reconstructed', alpha=0.8)
        ax1.set_xlabel('Time (μs)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Time Domain')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(2, 3, 2)
        freqs, power_db = self._spectrum
        ax2.plot(freqs / 1e3, power_db, 'b-', linewidth=0.8)
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Power (dB)')
        ax2.set_title('Spectrum')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(2, 3, 3)
        error = (self.result.input_signal - self.result.reconstructed)
        lsb = (max(self.result.input_signal) - min(self.result.input_signal)) / (2**self.bits)
        ax3.plot(self.result.timestamps * 1e6, error / lsb, 'purple', linewidth=0.8)
        ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
        ax3.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
        ax3.set_xlabel('Time (μs)')
        ax3.set_ylabel('Error (LSB)')
        ax3.set_title('Quantization Error')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(self.inl, 'b-', linewidth=0.8)
        ax4.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
        ax4.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
        ax4.set_xlabel('Code')
        ax4.set_ylabel('INL (LSB)')
        ax4.set_title(f'INL (max: {self.inl_max:.3f} LSB)')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(self.dnl, 'g-', linewidth=0.8)
        ax5.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
        ax5.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
        ax5.set_xlabel('Code')
        ax5.set_ylabel('DNL (LSB)')
        ax5.set_title(f'DNL (max: {self.dnl_max:.3f} LSB)')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
        Performance Metrics
        ──────────────────
        ENOB:   {self.enob:.2f} bits
        SNR:    {self.snr:.2f} dB
        SFDR:   {self.sfdr:.2f} dB
        THD:    {self.thd:.2f} dB
        SINAD:  {self.sinad:.2f} dB
        """
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig


__all__ = [
    "Analyzer",
    "spectrum",
    "enob",
    "snr",
    "sfdr",
    "thd",
    "inl",
    "dnl",
]
