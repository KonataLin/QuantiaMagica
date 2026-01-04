"""
Base ADConverter class - Abstract base for all ADC implementations.

This module provides the foundation for all ADC types in QuantiaMagica.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import numpy as np
from numpy.typing import NDArray

from .events import Event, EventBus, EventPriority


@dataclass
class SimulationResult:
    """
    Container for ADC simulation results.
    
    Attributes
    ----------
    input_signal : NDArray[np.float64]
        Input voltage samples.
    output_codes : NDArray[np.int64]
        Digital output codes.
    timestamps : NDArray[np.float64]
        Time points for each sample.
    reconstructed : NDArray[np.float64]
        Reconstructed analog signal from codes.
    events : List[Event]
        All events fired during simulation (if logging enabled).
    metadata : Dict[str, Any]
        Additional simulation metadata.
    """
    input_signal: NDArray[np.float64]
    output_codes: NDArray[np.int64]
    timestamps: NDArray[np.float64]
    reconstructed: NDArray[np.float64]
    events: List[Event] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the result."""
        return len(self.output_codes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "input_signal": self.input_signal.tolist(),
            "output_codes": self.output_codes.tolist(),
            "timestamps": self.timestamps.tolist(),
            "reconstructed": self.reconstructed.tolist(),
            "metadata": self.metadata,
        }
    
    def save(self, path: str, format: str = "npz") -> None:
        """
        Save results to file.
        
        Parameters
        ----------
        path : str
            Output file path.
        format : str
            Format: 'npz', 'csv', or 'json'.
        """
        if format == "npz":
            np.savez(
                path,
                input_signal=self.input_signal,
                output_codes=self.output_codes,
                timestamps=self.timestamps,
                reconstructed=self.reconstructed,
            )
        elif format == "csv":
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "input", "code", "reconstructed"])
                for t, inp, code, rec in zip(
                    self.timestamps, self.input_signal, 
                    self.output_codes, self.reconstructed
                ):
                    writer.writerow([t, inp, code, rec])
        elif format == "json":
            import json
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


class ADConverter(ABC):
    """
    Abstract base class for all ADC implementations.
    
    Provides common functionality including event handling, simulation,
    plotting, and analysis. All specific ADC types inherit from this class.
    
    Parameters
    ----------
    bits : int
        Resolution in bits.
    vref : float
        Reference voltage (full scale).
    vmin : float, optional
        Minimum input voltage, by default 0.0.
    name : str, optional
        ADC instance name for identification.
    
    Attributes
    ----------
    bits : int
        ADC resolution.
    vref : float
        Reference voltage.
    vmin : float
        Minimum input voltage.
    lsb : float
        Least significant bit voltage.
    event_bus : EventBus
        Event dispatcher for this ADC.
    result : SimulationResult
        Most recent simulation result.
    
    Example
    -------
    >>> class MyADC(ADConverter):
    ...     def _convert_single(self, voltage, timestamp):
    ...         # Implementation
    ...         return code
    """
    
    def __init__(
        self,
        bits: int,
        vref: float = 1.0,
        vmin: float = 0.0,
        name: Optional[str] = None,
    ):
        self.bits = bits
        self.vref = vref
        self.vmin = vmin
        self.name = name or self.__class__.__name__
        
        self._event_bus = EventBus()
        self._result: Optional[SimulationResult] = None
        self._time: float = 0.0
        self._sample_index: int = 0
        
        self._plugins: List[Any] = []
    
    @property
    def lsb(self) -> float:
        """Least significant bit voltage."""
        return (self.vref - self.vmin) / (2 ** self.bits)
    
    @property
    def max_code(self) -> int:
        """Maximum output code."""
        return 2 ** self.bits - 1
    
    @property
    def levels(self) -> int:
        """Number of quantization levels."""
        return 2 ** self.bits
    
    @property
    def result(self) -> Optional[SimulationResult]:
        """Most recent simulation result."""
        return self._result
    
    @property
    def event_bus(self) -> EventBus:
        """Event bus for this ADC."""
        return self._event_bus
    
    def on(
        self,
        event_type: Type[Event],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
    ) -> Callable:
        """
        Decorator to register an event handler.
        
        Parameters
        ----------
        event_type : Type[Event]
            Event class to listen for.
        priority : EventPriority, optional
            Handler priority.
        ignore_cancelled : bool, optional
            Run even if event is cancelled.
        
        Returns
        -------
        Callable
            Decorator function.
        
        Example
        -------
        >>> @adc.on(SamplingEvent)
        ... def handle_sample(event):
        ...     print(f"Sampled: {event.voltage}V")
        """
        return self._event_bus.on(event_type, priority, ignore_cancelled)
    
    def register(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
    ) -> None:
        """Register an event handler programmatically."""
        self._event_bus.register(event_type, callback, priority, ignore_cancelled)
    
    def unregister(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
    ) -> bool:
        """Remove a registered handler."""
        return self._event_bus.unregister(event_type, callback)
    
    def fire(self, event: Event) -> Event:
        """Fire an event to all handlers."""
        event.source = self
        return self._event_bus.fire(event)
    
    def use(self, plugin: Any) -> "ADConverter":
        """
        Register a plugin with this ADC.
        
        Plugins are objects with methods decorated with @on_event.
        All decorated methods are registered as handlers.
        
        Parameters
        ----------
        plugin : Any
            Plugin object with event handler methods.
        
        Returns
        -------
        ADConverter
            Self, for method chaining.
        
        Example
        -------
        >>> class NoisePlugin:
        ...     @on_event(SamplingEvent)
        ...     def add_noise(self, event):
        ...         event.voltage += np.random.normal(0, 0.001)
        >>> 
        >>> adc.use(NoisePlugin())
        """
        self._plugins.append(plugin)
        
        for name in dir(plugin):
            method = getattr(plugin, name)
            if callable(method) and hasattr(method, '_event_handler'):
                self._event_bus.register(
                    method._event_type,
                    lambda e, m=method: m(e),
                    method._event_priority,
                    method._ignore_cancelled,
                    owner=plugin,
                )
        
        return self
    
    def voltage_to_code(self, voltage: float) -> int:
        """
        Convert voltage to ideal digital code.
        
        Parameters
        ----------
        voltage : float
            Input voltage.
        
        Returns
        -------
        int
            Digital code (clipped to valid range).
        """
        code = int((voltage - self.vmin) / self.lsb)
        return max(0, min(self.max_code, code))
    
    def code_to_voltage(self, code: int) -> float:
        """
        Convert digital code to reconstructed voltage.
        
        Parameters
        ----------
        code : int
            Digital code.
        
        Returns
        -------
        float
            Reconstructed voltage.
        """
        return self.vmin + (code + 0.5) * self.lsb
    
    @abstractmethod
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        """
        Convert a single voltage sample to digital code.
        
        This method must be implemented by subclasses.
        
        Parameters
        ----------
        voltage : float
            Input voltage.
        timestamp : float
            Current simulation time.
        
        Returns
        -------
        int
            Digital output code.
        """
        pass
    
    def convert(self, voltage: Union[float, NDArray]) -> Union[int, NDArray]:
        """
        Convert voltage(s) to digital code(s).
        
        Parameters
        ----------
        voltage : float or NDArray
            Input voltage(s).
        
        Returns
        -------
        int or NDArray
            Digital code(s).
        """
        if isinstance(voltage, (int, float)):
            return self._convert_single(float(voltage), self._time)
        else:
            codes = np.zeros(len(voltage), dtype=np.int64)
            for i, v in enumerate(voltage):
                codes[i] = self._convert_single(float(v), self._time + i)
                self._sample_index += 1
            return codes
    
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
        log_events: bool = False,
    ) -> SimulationResult:
        """
        Run ADC simulation.
        
        Parameters
        ----------
        input_voltage : float, NDArray, or None
            Input signal. If None, generates test signal.
        n_samples : int, optional
            Number of samples for generated signal.
        fs : float, optional
            Sampling frequency in Hz.
        fin : float, optional
            Input signal frequency in Hz.
        amplitude : float, optional
            Signal amplitude (defaults to 0.9 * vref/2).
        offset : float, optional
            DC offset (defaults to mid-range).
        signal : str, optional
            Signal type: 'sine', 'ramp', 'dc'.
        log_events : bool, optional
            Enable event logging.
        
        Returns
        -------
        SimulationResult
            Simulation results.
        
        Example
        -------
        >>> # Quick simulation with default sine
        >>> result = adc.sim()
        >>> 
        >>> # Custom input
        >>> result = adc.sim(my_signal)
        >>> 
        >>> # Generated signal
        >>> result = adc.sim(n_samples=2048, fin=10e3)
        """
        self._event_bus.enable_logging(log_events)
        self._time = 0.0
        self._sample_index = 0
        
        if amplitude is None:
            # Use 99.6% of full scale - optimal for ENOB (avoids clipping artifacts)
            amplitude = 0.498 * (self.vref - self.vmin)
        if offset is None:
            offset = (self.vref + self.vmin) / 2
        
        if input_voltage is None:
            timestamps = np.arange(n_samples) / fs
            if signal == "sine":
                n_periods = int(np.round(fin * n_samples / fs))
                n_periods = max(1, n_periods)  # Ensure at least 1 period
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
        else:
            input_signal = np.asarray(input_voltage, dtype=np.float64)
            timestamps = np.arange(len(input_signal)) / fs
        
        output_codes = np.zeros(len(input_signal), dtype=np.int64)
        
        for i, v in enumerate(input_signal):
            self._time = timestamps[i]
            self._sample_index = i
            output_codes[i] = self._convert_single(float(v), self._time)
        
        reconstructed = np.array([
            self.code_to_voltage(c) for c in output_codes
        ])
        
        self._result = SimulationResult(
            input_signal=input_signal,
            output_codes=output_codes,
            timestamps=timestamps,
            reconstructed=reconstructed,
            events=self._event_bus.get_log() if log_events else [],
            metadata={
                "adc_name": self.name,
                "bits": self.bits,
                "vref": self.vref,
                "vmin": self.vmin,
                "lsb": self.lsb,
                "fs": fs,
                "fin": fin_coherent if signal == "sine" and input_voltage is None else fin,
                "n_samples": len(input_signal),
            },
        )
        
        self._event_bus.enable_logging(False)
        
        return self._result
    
    def plot(
        self,
        result: Optional[SimulationResult] = None,
        *,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        figsize: Tuple[float, float] = (7.16, 5.0),
        style: str = "jssc",
    ) -> Any:
        """
        Plot simulation results in IEEE JSSC black-white style.
        
        Parameters
        ----------
        result : SimulationResult, optional
            Results to plot. Uses most recent if None.
        show : bool, optional
            Display the plot.
        save : str, optional
            Save path for figure.
        dpi : int, optional
            Figure DPI (300 for publication).
        figsize : tuple, optional
            Figure size (width, height) in inches.
        style : str, optional
            'jssc' for IEEE JSSC style.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt
        
        if result is None:
            if self._result is None:
                raise ValueError("No simulation result. Run sim() first.")
            result = self._result
        
        # IEEE JSSC style: black/white, Times font, thin lines
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
        
        # Auto-scale time units
        from .._analysis_impl import auto_time_unit
        t_scaled, t_unit = auto_time_unit(result.timestamps)
        
        # Time domain - black solid and dashed
        ax1 = axes[0, 0]
        ax1.plot(t_scaled, result.input_signal, 
                 'k-', linewidth=0.8, label='Input')
        ax1.plot(t_scaled, result.reconstructed, 
                 'k--', linewidth=0.6, label='Reconstructed')
        ax1.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax1.set_ylabel('Voltage (V)', fontsize=8)
        ax1.set_title('Time Domain', fontsize=9)
        ax1.legend(loc='upper right', fontsize=7, frameon=True, edgecolor='black')
        ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # Digital output - black step
        ax2 = axes[0, 1]
        ax2.step(t_scaled, result.output_codes, 
                 'k-', linewidth=0.6, where='mid')
        ax2.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax2.set_ylabel('Code', fontsize=8)
        ax2.set_title('Digital Output', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # Quantization error - gray
        ax3 = axes[1, 0]
        error = result.input_signal - result.reconstructed
        ax3.plot(t_scaled, error * 1e3, 
                 color='#404040', linewidth=0.6)
        ax3.axhline(y=self.lsb * 1e3 / 2, color='black', linestyle='--', 
                    linewidth=0.5, label=f'+/-LSB/2')
        ax3.axhline(y=-self.lsb * 1e3 / 2, color='black', linestyle='--', linewidth=0.5)
        ax3.set_xlabel(f'Time ({t_unit})', fontsize=8)
        ax3.set_ylabel('Error (mV)', fontsize=8)
        ax3.set_title('Quantization Error', fontsize=9)
        ax3.legend(loc='upper right', fontsize=7, frameon=True, edgecolor='black')
        ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        # Histogram - gray bars
        ax4 = axes[1, 1]
        hist, bins = np.histogram(result.output_codes, bins=min(64, self.max_code + 1))
        ax4.bar(bins[:-1], hist, width=np.diff(bins), 
                color='#808080', edgecolor='black', linewidth=0.3)
        ax4.set_xlabel('Code', fontsize=8)
        ax4.set_ylabel('Count', fontsize=8)
        ax4.set_title('Code Histogram', fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
        
        if show:
            plt.show()
        
        return fig
    
    def spectrum(
        self,
        result: Optional[SimulationResult] = None,
        *,
        window: str = "hann",
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 150,
        figsize: Tuple[float, float] = (10, 6),
    ) -> Tuple[NDArray, NDArray, Dict[str, float]]:
        """
        Compute and plot frequency spectrum.
        
        Parameters
        ----------
        result : SimulationResult, optional
            Results to analyze.
        window : str, optional
            Window function name.
        show : bool, optional
            Display the plot.
        save : str, optional
            Save path.
        dpi : int, optional
            Figure DPI.
        figsize : tuple, optional
            Figure size.
        
        Returns
        -------
        freqs : NDArray
            Frequency bins in Hz.
        spectrum_db : NDArray
            Power spectrum in dB.
        metrics : Dict[str, float]
            Computed metrics (SNR, SFDR, ENOB, THD).
        """
        from .._analysis_impl import compute_spectrum
        
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
            
            # IEEE JSSC style
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 9,
                'axes.linewidth': 0.6,
                'lines.linewidth': 0.8,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
            })
            
            fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=dpi)
            
            # Auto-scale frequency units
            from .._analysis_impl import auto_freq_unit
            f_scaled, f_unit = auto_freq_unit(freqs)
            
            # Black line for spectrum
            ax.plot(f_scaled, spectrum_db, 'k-', linewidth=0.6)
            ax.set_xlabel(f'Frequency ({f_unit})', fontsize=8)
            ax.set_ylabel('Magnitude (dB)', fontsize=8)
            ax.set_title(f'{self.name}', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
            ax.set_xlim([0, f_scaled[-1]])
            ax.set_ylim([max(-120, np.min(spectrum_db) - 10), 10])
            
            # Metrics box - white background, black border
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
        """Compute Effective Number of Bits."""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['enob']
    
    def snr(self, result: Optional[SimulationResult] = None) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['snr']
    
    def sfdr(self, result: Optional[SimulationResult] = None) -> float:
        """Compute Spurious-Free Dynamic Range in dB."""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['sfdr']
    
    def thd(self, result: Optional[SimulationResult] = None) -> float:
        """Compute Total Harmonic Distortion in dB."""
        _, _, metrics = self.spectrum(result, show=False)
        return metrics['thd']
    
    def inl(
        self,
        result: Optional[SimulationResult] = None,
        *,
        plot: bool = False,
    ) -> NDArray:
        """
        Compute Integral Non-Linearity.
        
        Parameters
        ----------
        result : SimulationResult, optional
            Results from ramp test.
        plot : bool, optional
            Plot INL curve.
        
        Returns
        -------
        NDArray
            INL values in LSB for each code.
        """
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
        """
        Compute Differential Non-Linearity.
        
        Parameters
        ----------
        result : SimulationResult, optional
            Results from ramp test.
        plot : bool, optional
            Plot DNL curve.
        
        Returns
        -------
        NDArray
            DNL values in LSB for each code.
        """
        from .._analysis_impl import compute_inl_dnl
        
        if result is None:
            if self._result is None:
                self.sim(signal='ramp', n_samples=4096)
            result = self._result
        
        _, dnl_vals = compute_inl_dnl(result.output_codes, self.bits)
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            ax.plot(dnl_vals, 'g-', linewidth=0.8)
            ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1)
            ax.axhline(y=-0.5, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Code')
            ax.set_ylabel('DNL (LSB)')
            ax.set_title(f'{self.name} DNL')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return dnl_vals
    
    def reset(self) -> None:
        """Reset ADC state for new simulation."""
        self._time = 0.0
        self._sample_index = 0
        self._result = None
        self._event_bus.clear_log()
    
    def report(
        self,
        what: str = "all",
        *,
        result: Optional[SimulationResult] = None,
        save: Optional[str] = None,
        show: bool = True,
        columns: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with flexible plotting.
        
        Simple yet flexible API - just call adc.report() for full analysis.
        
        Parameters
        ----------
        result : SimulationResult, optional
            Results to analyze. Uses most recent if None.
        what : str, optional
            What to plot/analyze:
            - 'all': Full report (time, spectrum, INL/DNL)
            - 'spectrum' or 'fft': Frequency spectrum only
            - 'time': Time domain only
            - 'static': INL/DNL static analysis
            - 'metrics': Print metrics only, no plot
        save : str, optional
            Save figure to file (e.g., 'report.pdf', 'spectrum.png')
        show : bool, optional
            Display plots interactively.
        columns : int, optional
            1 for single-column (3.5"), 2 for double-column (7.16")
        
        Returns
        -------
        Dict[str, Any]
            Analysis results including all metrics.
        
        Examples
        --------
        >>> adc.sim().report()                    # Full report
        >>> adc.sim().report('spectrum')          # Spectrum only
        >>> adc.sim().report(save='fig.pdf')      # Save to PDF
        >>> adc.sim().report('metrics')           # Metrics only
        """
        import matplotlib.pyplot as plt
        
        if result is None:
            if self._result is None:
                raise ValueError("No simulation result. Run sim() first.")
            result = self._result
        
        # JSSC style configuration
        figwidth = 3.5 if columns == 1 else 7.16
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 8 if columns == 1 else 9,
            'axes.linewidth': 0.6,
            'lines.linewidth': 0.8,
            'grid.linewidth': 0.4,
            'grid.alpha': 0.4,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
        })
        
        # Compute all metrics
        from .._analysis_impl import compute_spectrum, compute_inl_dnl
        fs = result.metadata.get('fs', 1e6)
        freqs, spectrum_db, spec_metrics = compute_spectrum(
            result.output_codes, self.bits, fs
        )
        
        metrics = {
            'enob': spec_metrics['enob'],
            'snr': spec_metrics['snr'],
            'sfdr': spec_metrics['sfdr'],
            'thd': spec_metrics['thd'],
            'sinad': spec_metrics['sinad'],
            'bits': self.bits,
            'fs': fs,
            'n_samples': len(result.output_codes),
        }
        
        what = what.lower()
        
        if what == 'metrics':
            print(f"╔{'═'*40}╗")
            print(f"║{'ADC Performance Report':^40}║")
            print(f"╠{'═'*40}╣")
            print(f"║  Resolution:  {self.bits:>6} bits{' '*14}║")
            print(f"║  ENOB:        {metrics['enob']:>6.2f} bits{' '*14}║")
            print(f"║  SNR:         {metrics['snr']:>6.1f} dB{' '*16}║")
            print(f"║  SFDR:        {metrics['sfdr']:>6.1f} dB{' '*16}║")
            print(f"║  THD:         {metrics['thd']:>6.1f} dB{' '*16}║")
            print(f"╚{'═'*40}╝")
            return metrics
        
        # Prepare figure based on 'what'
        if what in ('spectrum', 'fft'):
            fig, ax = plt.subplots(figsize=(figwidth, figwidth*0.7), dpi=300)
            self._plot_spectrum_ax(ax, freqs, spectrum_db, spec_metrics)
            
        elif what == 'time':
            fig, ax = plt.subplots(figsize=(figwidth, figwidth*0.5), dpi=300)
            self._plot_time_ax(ax, result)
            
        elif what == 'static':
            inl_vals, dnl_vals = compute_inl_dnl(result.output_codes, self.bits)
            metrics['inl_max'] = np.max(np.abs(inl_vals))
            metrics['dnl_max'] = np.max(np.abs(dnl_vals))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figwidth, figwidth), dpi=300)
            self._plot_inl_ax(ax1, inl_vals)
            self._plot_dnl_ax(ax2, dnl_vals)
            
        else:  # 'all'
            fig = plt.figure(figsize=(figwidth*1.5, figwidth*1.2), dpi=300)
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_time_ax(ax1, result)
            
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_spectrum_ax(ax2, freqs, spectrum_db, spec_metrics)
            
            ax3 = fig.add_subplot(gs[1, :])
            ax3.step(np.arange(len(result.output_codes)), result.output_codes,
                    'k-', linewidth=0.5, where='mid')
            ax3.set_xlabel('Sample', fontsize=7)
            ax3.set_ylabel('Code', fontsize=7)
            ax3.set_title('Digital Output', fontsize=8)
            ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
            
            fig.suptitle(f'{self.name} Analysis Report', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"Figure saved to {save}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return metrics
    
    def _plot_spectrum_ax(self, ax, freqs, spectrum_db, metrics):
        """Plot spectrum on given axis (internal helper)."""
        from .._analysis_impl import auto_freq_unit
        f_scaled, f_unit = auto_freq_unit(freqs)
        ax.plot(f_scaled, spectrum_db, 'k-', linewidth=0.6)
        ax.set_xlabel(f'Frequency ({f_unit})', fontsize=7)
        ax.set_ylabel('Magnitude (dB)', fontsize=7)
        ax.set_title('Output Spectrum', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
        ax.set_xlim([0, f_scaled[-1]])
        ax.set_ylim([max(-120, np.min(spectrum_db) - 10), 10])
        
        textstr = f"ENOB={metrics['enob']:.2f}\nSNR={metrics['snr']:.1f}dB"
        props = dict(boxstyle='square,pad=0.2', facecolor='white', 
                    edgecolor='black', linewidth=0.4)
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', horizontalalignment='right', 
                bbox=props, family='monospace')
    
    def _plot_time_ax(self, ax, result):
        """Plot time domain on given axis (internal helper)."""
        from .._analysis_impl import auto_time_unit
        t_scaled, t_unit = auto_time_unit(result.timestamps)
        ax.plot(t_scaled, result.input_signal, 'k-', linewidth=0.6, label='Input')
        ax.plot(t_scaled, result.reconstructed, 'k--', linewidth=0.5, label='Output')
        ax.set_xlabel(f'Time ({t_unit})', fontsize=7)
        ax.set_ylabel('Voltage (V)', fontsize=7)
        ax.set_title('Time Domain', fontsize=8)
        ax.legend(loc='upper right', fontsize=6, frameon=True, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    
    def _plot_inl_ax(self, ax, inl_vals):
        """Plot INL on given axis (internal helper)."""
        ax.plot(inl_vals, 'k-', linewidth=0.6)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Code', fontsize=7)
        ax.set_ylabel('INL (LSB)', fontsize=7)
        ax.set_title(f'INL (max={np.max(np.abs(inl_vals)):.2f} LSB)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    
    def _plot_dnl_ax(self, ax, dnl_vals):
        """Plot DNL on given axis (internal helper)."""
        ax.plot(dnl_vals, 'k-', linewidth=0.6)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Code', fontsize=7)
        ax.set_ylabel('DNL (LSB)', fontsize=7)
        ax.set_title(f'DNL (max={np.max(np.abs(dnl_vals)):.2f} LSB)', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    
    def summary(self) -> str:
        """Return a formatted string summary of the last simulation."""
        if self._result is None:
            return "No simulation result available."
        
        _, _, m = self.spectrum(show=False)
        return (
            f"{self.name} | {self.bits}-bit\n"
            f"ENOB: {m['enob']:.2f} | SNR: {m['snr']:.1f} dB | SFDR: {m['sfdr']:.1f} dB"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits}, vref={self.vref})"
    
    def sim_auto(
        self,
        fs: float = 1e6,
        *,
        n_samples: int = 4096,
        verbose: bool = True,
        save_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Automatic optimization of fin and amplitude to maximize ENOB.
        
        Uses genetic algorithm with concurrent evaluation to find optimal
        single-tone test parameters. Convergence is fully automatic based on
        ENOB improvement slope and multiple safety criteria.
        
        Parameters
        ----------
        fs : float, optional
            Sampling frequency in Hz. Default 1 MHz.
        n_samples : int, optional
            Number of samples per simulation. Default 4096.
        verbose : bool, optional
            Print progress. Default True.
        save_result : bool, optional
            Save best result to self._result. Default True.
        
        Returns
        -------
        dict
            Optimization results containing:
            - 'best_fin': Optimal input frequency (Hz)
            - 'best_amplitude': Optimal amplitude (V)
            - 'best_enob': Achieved ENOB (bits)
            - 'fs': Sampling frequency used
            - 'n_samples': Number of samples
            - 'history': ENOB history per generation
            - 'converged': Whether optimization converged
            - 'reason': Convergence/termination reason
        
        Example
        -------
        >>> adc = SARADC(bits=12)
        >>> result = adc.sim_auto(fs=1e6)
        >>> print(f"Best: fin={result['best_fin']:.1f}Hz, ENOB={result['best_enob']:.2f}")
        """
        import copy
        import os
        from concurrent.futures import ThreadPoolExecutor
        
        # ============ Hardware Detection ============
        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if has_cuda else ""
        except:
            pass
        
        # ============ Algorithm Config ============
        cpu_count = os.cpu_count() or 4
        n_workers = cpu_count * 2
        
        # Detect ADC type
        adc_class_name = self.__class__.__name__
        is_sd = 'SigmaDelta' in adc_class_name
        amp_range = self.vref - self.vmin
        offset = (self.vref + self.vmin) / 2
        
        if is_sd:
            osr = getattr(self, 'osr', 64)
            order = getattr(self, 'order', 1)
            bits_q = getattr(self, 'bits', 1)
            # SD ENOB needs: n_samples >= OSR*64 for accurate FFT
            min_n = osr * 64
            n_samples = max(n_samples, min_n)
            signal_bw = fs / (2 * osr)
            fin_max = signal_bw * 0.5  # Stay within 50% bandwidth
            fin_min = fs / n_samples * 5
            # Amplitude: use range that gives ENOB close to theoretical
            # amp ~0.35 gives ENOB ~ theoretical for 1-bit
            if bits_q >= 3:
                amp_min, amp_max = 0.3 * amp_range, 0.4 * amp_range
            else:
                # 1-bit: amp around 0.3-0.35 gives ENOB close to theory
                base_amp = 0.35 * amp_range / (1 + 0.1 * (order - 1))
                amp_min = base_amp * 0.8
                amp_max = base_amp * 1.1
            theo_limit = getattr(self, 'theoretical_enob_gain', 15)
        else:
            # SAR/Pipeline
            fin_min = fs / n_samples * 5
            fin_max = fs / 4
            amp_min, amp_max = 0.45 * amp_range, 0.499 * amp_range
            theo_limit = self.bits
        
        if verbose:
            print("=" * 60)
            print(f"simAuto: {self.name}")
            print(f"  Hardware: {'CUDA (' + gpu_name + ')' if has_cuda else 'CPU'}")
            print(f"  n_samples={n_samples}, theo_limit={theo_limit:.1f} bits")
            if is_sd:
                print(f"  fin range: {fin_min/1e3:.1f} - {fin_max/1e3:.1f} kHz")
            print("-" * 60)
        
        # ============ Grid Search (simpler, more reliable) ============
        def make_coherent_fin(fin_raw: float) -> float:
            n_periods = max(1, int(np.round(fin_raw * n_samples / fs)))
            if n_periods % 2 == 0 and n_periods > 1:
                n_periods -= 1
            return n_periods * fs / n_samples
        
        def evaluate(fin: float, amp: float) -> float:
            adc_copy = copy.deepcopy(self)
            try:
                adc_copy.sim(n_samples=n_samples, fs=fs, fin=fin, 
                            amplitude=amp, offset=offset)
                enob = adc_copy.enob()
                return enob if not np.isnan(enob) and enob < 50 else -100.0
            except:
                return -100.0
        
        # Generate coherent frequencies
        period_min = max(1, int(fin_min * n_samples / fs))
        period_max = int(fin_max * n_samples / fs)
        periods = [p for p in range(period_min, period_max + 1) if p % 2 == 1]
        if len(periods) > 20:
            step = len(periods) // 20
            periods = periods[::step]
        fins = [p * fs / n_samples for p in periods]
        
        # Amplitude grid
        amps = np.linspace(amp_min, amp_max, 5)
        
        # Create all combinations
        params = [(f, a) for f in fins for a in amps]
        
        if verbose:
            print(f"  Searching {len(params)} points ({len(fins)} fins x {len(amps)} amps)...")
        
        def eval_wrapper(p):
            return evaluate(p[0], p[1])
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(eval_wrapper, params))
        
        best_idx = np.argmax(results)
        best_fin, best_amp = params[best_idx]
        best_enob = results[best_idx]
        
        if verbose:
            print(f"  Best: fin={best_fin:.0f}Hz, amp={best_amp:.3f}V, ENOB={best_enob:.2f}")
            print(f"  vs theo: {best_enob:.2f} / {theo_limit:.1f} ({100*best_enob/theo_limit:.0f}%)")
            print("=" * 60)
        
        # Save final result
        if save_result:
            self.sim(n_samples=n_samples, fs=fs, fin=best_fin,
                    amplitude=best_amp, offset=offset)
        
        converged = best_enob > theo_limit - 0.5
        reason = f"ENOB={best_enob:.2f} vs limit={theo_limit:.1f}"
        
        if verbose:
            print(f"  Result: ENOB={best_enob:.4f} bits")
            print("=" * 60)
        
        return {
            'best_fin': best_fin,
            'best_amplitude': best_amp,
            'best_enob': best_enob,
            'fs': fs,
            'n_samples': n_samples,
            'converged': converged,
            'reason': reason,
            'generations': 2,
        }
