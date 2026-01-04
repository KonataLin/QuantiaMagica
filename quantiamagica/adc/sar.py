"""
SAR ADC Implementation with Bukkit-style Event System.

This module provides a fully event-driven SAR (Successive Approximation Register)
ADC implementation that allows users to hook into every stage of conversion.

Example
-------
>>> from quantiamagica import SARADC, CapacitorSwitchEvent
>>> 
>>> adc = SARADC(bits=10, vref=1.0)
>>> 
>>> @adc.on(CapacitorSwitchEvent)
... def add_cap_mismatch(event):
...     # Add 1% mismatch to bit 5
...     if event.bit_index == 5:
...         event.capacitance *= 1.01
>>> 
>>> result = adc.sim()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from numpy.typing import NDArray

from ..core.events import Event, Cancellable, EventPriority
from ..core.base import ADConverter


# =============================================================================
# SAR ADC Events
# =============================================================================

@dataclass
class ConversionStartEvent(Event):
    """
    Fired at the beginning of each conversion cycle.
    
    Attributes
    ----------
    sample_index : int
        Index of current sample being converted.
    input_voltage : float
        Raw input voltage before any modification.
    
    Example
    -------
    >>> @adc.on(ConversionStartEvent)
    ... def on_start(event):
    ...     print(f"Starting conversion {event.sample_index}")
    """
    sample_index: int = 0
    input_voltage: float = 0.0


@dataclass
class SamplingEvent(Event, Cancellable):
    """
    Fired during the sampling phase.
    
    Modify voltage to simulate sampling effects like noise, offset,
    bandwidth limitations, etc.
    
    Attributes
    ----------
    voltage : float
        Sampled voltage (modifiable).
    sample_index : int
        Current sample index.
    sampling_capacitance : float
        Total sampling capacitance in fF.
    sampling_time : float
        Sampling time in seconds.
    bandwidth : float
        Input bandwidth in Hz.
    
    Example
    -------
    >>> @adc.on(SamplingEvent)
    ... def add_thermal_noise(event):
    ...     kT = 1.38e-23 * 300
    ...     noise_var = kT / (event.sampling_capacitance * 1e-15)
    ...     event.voltage += np.random.normal(0, np.sqrt(noise_var))
    """
    voltage: float = 0.0
    sample_index: int = 0
    sampling_capacitance: float = 1000.0  # fF
    sampling_time: float = 1e-9  # seconds
    bandwidth: float = 100e6  # Hz
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class CapacitorSwitchEvent(Event, Cancellable):
    """
    Fired for each capacitor switching during SAR conversion.
    
    This is the key event for implementing non-idealities like:
    - Capacitor mismatch
    - Parasitic capacitance
    - Charge injection
    - Incomplete settling
    
    Attributes
    ----------
    bit_index : int
        Bit position (0 = LSB, N-1 = MSB).
    capacitance : float
        Nominal capacitance in fF (modifiable).
    capacitance_actual : float
        Actual capacitance after mismatch (modifiable).
    weight : float
        Bit weight (modifiable for NS-SAR etc.).
    switch_voltage : float
        Voltage being switched to (modifiable).
    charge_injection : float
        Charge injection in fC (modifiable).
    settling_error : float
        Settling error as fraction (modifiable).
    parasitic : float
        Parasitic capacitance in fF.
    dac_voltage : float
        Current DAC output voltage.
    comparator_input : float
        Current comparator input voltage.
    
    Example
    -------
    >>> @adc.on(CapacitorSwitchEvent)
    ... def cap_mismatch(event):
    ...     # Random 0.5% mismatch
    ...     event.capacitance_actual = event.capacitance * (1 + np.random.normal(0, 0.005))
    """
    bit_index: int = 0
    capacitance: float = 0.0  # fF, nominal
    capacitance_actual: float = 0.0  # fF, with mismatch
    weight: float = 0.0  # bit weight
    switch_voltage: float = 0.0
    charge_injection: float = 0.0  # fC
    settling_error: float = 0.0  # fraction
    parasitic: float = 0.0  # fF
    dac_voltage: float = 0.0
    comparator_input: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)
        if self.capacitance_actual == 0.0:
            self.capacitance_actual = self.capacitance


@dataclass
class ComparatorEvent(Event, Cancellable):
    """
    Fired when the comparator makes a decision.
    
    Modify to add comparator non-idealities like:
    - Offset
    - Noise
    - Metastability
    - Kickback
    
    Attributes
    ----------
    bit_index : int
        Current bit being decided.
    input_voltage : float
        Comparator input voltage.
    threshold : float
        Comparison threshold (modifiable).
    decision : bool
        Comparator output (True = input > threshold).
    offset : float
        Comparator offset voltage (modifiable).
    noise_sigma : float
        Comparator noise standard deviation (modifiable).
    delay : float
        Comparator delay in seconds.
    metastable : bool
        Whether comparator is in metastable region.
    kickback : float
        Kickback voltage in mV.
    
    Example
    -------
    >>> @adc.on(ComparatorEvent)
    ... def add_comparator_offset(event):
    ...     event.offset = 0.5e-3  # 0.5mV offset
    """
    bit_index: int = 0
    input_voltage: float = 0.0
    threshold: float = 0.0
    decision: bool = False
    offset: float = 0.0
    noise_sigma: float = 0.0
    delay: float = 1e-9
    metastable: bool = False
    kickback: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class BitDecisionEvent(Event, Cancellable):
    """
    Fired after each bit decision is made.
    
    Allows modification of the bit decision before it's finalized.
    
    Attributes
    ----------
    bit_index : int
        Bit position.
    bit_value : int
        Decided bit value (0 or 1, modifiable).
    residue : float
        Remaining voltage to be resolved.
    accumulated_code : int
        Code accumulated so far.
    
    Example
    -------
    >>> @adc.on(BitDecisionEvent)
    ... def flip_stuck_bit(event):
    ...     if event.bit_index == 3:
    ...         event.bit_value = 1  # Stuck-at-1 fault
    """
    bit_index: int = 0
    bit_value: int = 0
    residue: float = 0.0
    accumulated_code: int = 0
    
    def __post_init__(self):
        Cancellable.__init__(self)


@dataclass
class OutputCodeEvent(Event):
    """
    Fired when the final output code is ready.
    
    Attributes
    ----------
    code : int
        Final digital output code (modifiable).
    input_voltage : float
        Original input voltage.
    conversion_time : float
        Total conversion time in seconds.
    bit_decisions : List[int]
        All bit decisions made.
    
    Example
    -------
    >>> @adc.on(OutputCodeEvent)
    ... def log_output(event):
    ...     print(f"V={event.input_voltage:.4f} -> Code={event.code}")
    """
    code: int = 0
    input_voltage: float = 0.0
    conversion_time: float = 0.0
    bit_decisions: List[int] = field(default_factory=list)


@dataclass
class ConversionEndEvent(Event):
    """
    Fired at the end of each conversion cycle.
    
    Attributes
    ----------
    sample_index : int
        Index of completed sample.
    final_code : int
        Final output code.
    total_time : float
        Total conversion time.
    
    Example
    -------
    >>> @adc.on(ConversionEndEvent)
    ... def on_complete(event):
    ...     print(f"Conversion {event.sample_index} complete: {event.final_code}")
    """
    sample_index: int = 0
    final_code: int = 0
    total_time: float = 0.0


# =============================================================================
# SAR ADC Implementation
# =============================================================================

class SARADC(ADConverter):
    """
    Successive Approximation Register ADC with event-driven architecture.
    
    Implements a charge-redistribution SAR ADC with complete event hooks
    for every stage of conversion. Perfect for behavioral modeling of
    non-idealities and advanced architectures like NS-SAR.
    
    Parameters
    ----------
    bits : int
        Resolution in bits.
    vref : float, optional
        Reference voltage, by default 1.0.
    vmin : float, optional
        Minimum input voltage, by default 0.0.
    cap_unit : float, optional
        Unit capacitance in fF, by default 50.0.
    comparator_noise : float, optional
        Comparator noise sigma in V, by default 0.0.
    comparator_offset : float, optional
        Comparator offset in V, by default 0.0.
    name : str, optional
        Instance name.
    
    Attributes
    ----------
    capacitances : NDArray
        Capacitor values for each bit.
    weights : NDArray
        Bit weights (for radix modification).
    cap_mismatches : NDArray
        Capacitor mismatch ratios.
    
    Example
    -------
    >>> # Basic 10-bit SAR
    >>> adc = SARADC(bits=10, vref=1.0)
    >>> result = adc.sim(fin=10e3)
    >>> adc.plot()
    >>> print(f"ENOB: {adc.enob():.2f}")
    
    >>> # NS-SAR with noise shaping
    >>> adc = SARADC(bits=12, vref=1.0)
    >>> @adc.on(CapacitorSwitchEvent)
    ... def ns_sar_logic(event):
    ...     # Implement noise shaping by modifying weights
    ...     pass
    """
    
    def __init__(
        self,
        bits: int,
        vref: float = 1.0,
        vmin: float = 0.0,
        cap_unit: float = 50.0,  # fF
        comparator_noise: float = 0.0,
        comparator_offset: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(bits, vref, vmin, name or "SAR-ADC")
        
        self.cap_unit = cap_unit
        self.comparator_noise = comparator_noise
        self.comparator_offset = comparator_offset
        
        self.capacitances = np.array([
            cap_unit * (2 ** i) for i in range(bits)
        ], dtype=np.float64)
        
        self.weights = np.array([
            2 ** i for i in range(bits)
        ], dtype=np.float64)
        self.weights = self.weights / (2 ** bits)  # Normalize to full scale
        
        self.cap_mismatches = np.ones(bits, dtype=np.float64)
        
        self._conversion_time_per_bit = 1e-9
        
        self._current_residue = 0.0
        self._bit_decisions: List[int] = []
    
    @property
    def total_capacitance(self) -> float:
        """Total capacitance in fF."""
        return float(np.sum(self.capacitances) + self.cap_unit)
    
    def set_capacitor_mismatch(
        self,
        mismatches: Optional[NDArray] = None,
        sigma: float = 0.0,
    ) -> None:
        """
        Set capacitor mismatches.
        
        Parameters
        ----------
        mismatches : NDArray, optional
            Direct mismatch values (ratio, 1.0 = no mismatch).
        sigma : float, optional
            If mismatches not provided, generate random with this sigma.
        """
        if mismatches is not None:
            self.cap_mismatches = np.asarray(mismatches, dtype=np.float64)
        else:
            self.cap_mismatches = 1 + np.random.normal(0, sigma, self.bits)
    
    def set_radix(self, radix: float) -> None:
        """
        Set non-binary radix for redundancy.
        
        Parameters
        ----------
        radix : float
            Radix value (e.g., 1.85 for redundant SAR).
        """
        self.weights = np.array([
            radix ** i for i in range(self.bits)
        ], dtype=np.float64)
        # For non-binary radix, normalize so full scale maps correctly
        self.weights = self.weights / np.sum(self.weights) * (1 - 1/(2**self.bits))
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        """Perform SAR conversion for a single sample."""
        
        start_event = ConversionStartEvent(
            timestamp=timestamp,
            sample_index=self._sample_index,
            input_voltage=voltage,
        )
        self.fire(start_event)
        
        sample_event = SamplingEvent(
            timestamp=timestamp,
            voltage=voltage,
            sample_index=self._sample_index,
            sampling_capacitance=self.total_capacitance,
        )
        self.fire(sample_event)
        
        if sample_event.cancelled:
            return 0
        
        sampled_voltage = sample_event.voltage
        
        # Add 0.5 LSB offset for proper rounding behavior (centers quantization noise)
        sampled_voltage_adjusted = sampled_voltage + 0.5 * self.lsb
        
        self._current_residue = sampled_voltage_adjusted
        self._bit_decisions = []
        
        dac_voltage = 0.0
        code = 0
        
        for bit in range(self.bits - 1, -1, -1):
            bit_time = timestamp + (self.bits - bit) * self._conversion_time_per_bit
            
            cap_event = CapacitorSwitchEvent(
                timestamp=bit_time,
                bit_index=bit,
                capacitance=self.capacitances[bit],
                capacitance_actual=self.capacitances[bit] * self.cap_mismatches[bit],
                weight=self.weights[bit],
                switch_voltage=self.vref,
                dac_voltage=dac_voltage,
                comparator_input=self._current_residue - dac_voltage,
            )
            self.fire(cap_event)
            
            if cap_event.cancelled:
                self._bit_decisions.append(0)
                continue
            
            trial_dac = dac_voltage + cap_event.weight * (self.vref - self.vmin)
            
            comp_input = sampled_voltage_adjusted - trial_dac
            
            comp_event = ComparatorEvent(
                timestamp=bit_time,
                bit_index=bit,
                input_voltage=comp_input,
                threshold=0.0,
                offset=self.comparator_offset,
                noise_sigma=self.comparator_noise,
            )
            
            effective_input = comp_input - comp_event.offset
            if comp_event.noise_sigma > 0:
                effective_input += np.random.normal(0, comp_event.noise_sigma)
            
            comp_event.decision = effective_input >= comp_event.threshold
            
            self.fire(comp_event)
            
            if comp_event.cancelled:
                bit_value = 0
            else:
                bit_value = 1 if comp_event.decision else 0
            
            bit_event = BitDecisionEvent(
                timestamp=bit_time,
                bit_index=bit,
                bit_value=bit_value,
                residue=sampled_voltage - dac_voltage,
                accumulated_code=code,
            )
            self.fire(bit_event)
            
            if not bit_event.cancelled:
                bit_value = bit_event.bit_value
            
            self._bit_decisions.append(bit_value)
            
            if bit_value == 1:
                dac_voltage += cap_event.weight * (self.vref - self.vmin)
                code += (1 << bit)
        
        self._bit_decisions.reverse()
        
        conversion_time = self.bits * self._conversion_time_per_bit
        
        output_event = OutputCodeEvent(
            timestamp=timestamp + conversion_time,
            code=code,
            input_voltage=voltage,
            conversion_time=conversion_time,
            bit_decisions=list(self._bit_decisions),
        )
        self.fire(output_event)
        
        final_code = output_event.code
        
        end_event = ConversionEndEvent(
            timestamp=timestamp + conversion_time,
            sample_index=self._sample_index,
            final_code=final_code,
            total_time=conversion_time,
        )
        self.fire(end_event)
        
        return final_code
    
    def get_dac_levels(self) -> NDArray:
        """Get all DAC voltage levels."""
        levels = np.zeros(2 ** self.bits)
        for code in range(2 ** self.bits):
            voltage = 0.0
            for bit in range(self.bits):
                if code & (1 << bit):
                    voltage += self.weights[bit] * (self.vref - self.vmin)
            levels[code] = voltage + self.vmin
        return levels
    
    def plot_dac(
        self,
        *,
        show: bool = True,
        save: Optional[str] = None,
    ) -> Any:
        """Plot DAC transfer characteristic."""
        import matplotlib.pyplot as plt
        
        levels = self.get_dac_levels()
        codes = np.arange(len(levels))
        ideal = self.vmin + codes * self.lsb
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
        
        ax1.step(codes, levels, 'b-', where='mid', label='Actual', linewidth=1)
        ax1.plot(codes, ideal, 'r--', label='Ideal', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Code')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'{self.name} DAC Transfer Characteristic')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        error = (levels - ideal) / self.lsb
        ax2.step(codes, error, 'g-', where='mid', linewidth=1)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Code')
        ax2.set_ylabel('Error (LSB)')
        ax2.set_title('DAC Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
