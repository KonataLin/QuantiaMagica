"""
Pipeline ADC Implementation with Event-Driven Architecture.

Supports chaining multiple ADConverter stages with inter-stage amplification.

Example
-------
>>> from quantiamagica import PipelineADC, SARADC, InterstageGainEvent
>>> 
>>> # Create 12-bit pipeline with 4 stages
>>> pipeline = PipelineADC(bits=12, stages=4)
>>> 
>>> # Or chain custom ADCs
>>> stage1 = SARADC(bits=4, vref=1.0)
>>> stage2 = SARADC(bits=4, vref=1.0)
>>> pipeline = PipelineADC.from_stages([stage1, stage2], gain=2.0)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray

from ..core.events import Event, Cancellable, EventPriority
from ..core.base import ADConverter, SimulationResult


# =============================================================================
# Pipeline ADC Events
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
class FlashEvent(Event, Cancellable):
    """
    Fired during flash conversion in pipeline stage.
    
    Attributes
    ----------
    stage_index : int
        Current stage index.
    input_voltage : float
        Flash ADC input.
    code : int
        Flash output code (modifiable).
    reference_levels : List[float]
        Flash comparator thresholds.
    comparator_offsets : List[float]
        Comparator offset voltages (modifiable).
    """
    stage_index: int = 0
    input_voltage: float = 0.0
    code: int = 0
    reference_levels: List[float] = field(default_factory=list)
    comparator_offsets: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        Cancellable.__init__(self)


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
# Pipeline Stage
# =============================================================================

@dataclass
class PipelineStage:
    """
    Represents a single pipeline stage.
    
    Attributes
    ----------
    bits : int
        Resolution of this stage.
    gain : float
        Inter-stage gain.
    adc : ADConverter, optional
        Custom ADC for this stage.
    redundancy : int
        Redundancy bits for digital correction.
    """
    bits: int
    gain: float = 2.0
    adc: Optional[ADConverter] = None
    redundancy: int = 0
    
    @property
    def effective_bits(self) -> int:
        """Effective bits after redundancy."""
        return self.bits - self.redundancy


# =============================================================================
# Pipeline ADC Implementation
# =============================================================================

class PipelineADC(ADConverter):
    """
    Pipeline ADC with configurable stages and digital error correction.
    
    Implements a multi-stage pipeline architecture where each stage
    performs coarse conversion, computes residue, and amplifies for
    the next stage.
    
    Parameters
    ----------
    bits : int
        Total resolution in bits.
    vref : float, optional
        Reference voltage.
    vmin : float, optional
        Minimum input voltage.
    stages : int, optional
        Number of pipeline stages.
    bits_per_stage : int, optional
        Bits resolved per stage.
    redundancy : int, optional
        Redundancy bits per stage for digital correction.
    interstage_gain : float, optional
        Inter-stage amplifier gain.
    name : str, optional
        Instance name.
    
    Attributes
    ----------
    stages : List[PipelineStage]
        Pipeline stage configurations.
    digital_correction : bool
        Enable digital error correction.
    
    Example
    -------
    >>> # Standard 12-bit pipeline
    >>> pipeline = PipelineADC(bits=12, stages=4)
    >>> result = pipeline.sim()
    >>> 
    >>> # With gain error modeling
    >>> @pipeline.on(InterstageGainEvent)
    ... def gain_error(event):
    ...     event.actual_gain *= 0.995  # 0.5% gain error
    """
    
    def __init__(
        self,
        bits: int = 12,
        vref: float = 1.0,
        vmin: float = 0.0,
        stages: int = 4,
        bits_per_stage: int = 3,
        redundancy: int = 1,
        interstage_gain: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(bits, vref, vmin, name or "Pipeline-ADC")
        
        self.digital_correction = True
        
        self._stages: List[PipelineStage] = []
        
        remaining_bits = bits
        for i in range(stages):
            if i == stages - 1:
                stage_bits = remaining_bits
                gain = 1.0
            else:
                stage_bits = bits_per_stage
                gain = interstage_gain if interstage_gain else 2 ** (stage_bits - redundancy)
            
            self._stages.append(PipelineStage(
                bits=stage_bits,
                gain=gain,
                redundancy=redundancy if i < stages - 1 else 0,
            ))
            
            remaining_bits -= (stage_bits - redundancy)
            if remaining_bits <= 0:
                break
        
        self._stage_codes: List[int] = []
        self._stage_residues: List[float] = []
    
    @property
    def num_stages(self) -> int:
        """Number of pipeline stages."""
        return len(self._stages)
    
    @property
    def stage_configs(self) -> List[PipelineStage]:
        """Stage configurations."""
        return self._stages
    
    @classmethod
    def from_stages(
        cls,
        adcs: List[ADConverter],
        gain: float = 2.0,
        name: Optional[str] = None,
    ) -> "PipelineADC":
        """
        Create pipeline from custom ADC stages.
        
        Parameters
        ----------
        adcs : List[ADConverter]
            ADC instances to use as stages.
        gain : float
            Inter-stage gain.
        name : str, optional
            Instance name.
        
        Returns
        -------
        PipelineADC
            Configured pipeline.
        
        Example
        -------
        >>> stage1 = SARADC(bits=4)
        >>> stage2 = SARADC(bits=4)
        >>> stage3 = SARADC(bits=6)
        >>> pipeline = PipelineADC.from_stages([stage1, stage2, stage3])
        """
        total_bits = sum(adc.bits for adc in adcs)
        
        instance = cls.__new__(cls)
        ADConverter.__init__(
            instance, 
            total_bits, 
            adcs[0].vref, 
            adcs[0].vmin,
            name or "Custom-Pipeline"
        )
        
        instance.digital_correction = True
        instance._stages = []
        instance._stage_codes = []
        instance._stage_residues = []
        
        for i, adc in enumerate(adcs):
            is_last = (i == len(adcs) - 1)
            instance._stages.append(PipelineStage(
                bits=adc.bits,
                gain=1.0 if is_last else gain,
                adc=adc,
                redundancy=0,
            ))
        
        return instance
    
    def _flash_convert(
        self,
        voltage: float,
        stage_index: int,
        bits: int,
    ) -> int:
        """Simple flash ADC for pipeline stage."""
        levels = 2 ** bits
        lsb = (self.vref - self.vmin) / levels
        
        ref_levels = [self.vmin + (i + 0.5) * lsb for i in range(levels - 1)]
        
        flash_event = FlashEvent(
            timestamp=self._time,
            stage_index=stage_index,
            input_voltage=voltage,
            reference_levels=ref_levels,
            comparator_offsets=[0.0] * (levels - 1),
        )
        
        code = 0
        for i, ref in enumerate(ref_levels):
            offset = flash_event.comparator_offsets[i] if i < len(flash_event.comparator_offsets) else 0
            if voltage >= ref + offset:
                code = i + 1
        
        flash_event.code = code
        self.fire(flash_event)
        
        if flash_event.cancelled:
            return 0
        
        return flash_event.code
    
    def _convert_single(self, voltage: float, timestamp: float) -> int:
        """
        Perform pipeline conversion.
        
        Uses ideal quantization with events fired at each stage for 
        non-ideality modeling via event handlers.
        """
        self._stage_codes = []
        self._stage_residues = []
        
        # Compute ideal output code first
        ideal_code = int((voltage - self.vmin) / self.lsb)
        ideal_code = max(0, min(self.max_code, ideal_code))
        
        # Process through stages for event firing
        current_voltage = voltage
        code_accumulator = 0
        bit_position = self.bits
        
        for stage_idx, stage in enumerate(self._stages):
            # Fire stage event
            stage_event = StageEvent(
                timestamp=timestamp,
                stage_index=stage_idx,
                stage_adc=stage.adc,
                input_voltage=current_voltage,
                sample_index=self._sample_index,
            )
            self.fire(stage_event)
            
            # Compute stage code (what this stage "sees")
            effective_bits = stage.bits - stage.redundancy if stage_idx < len(self._stages) - 1 else stage.bits
            stage_levels = 2 ** stage.bits
            stage_lsb = (self.vref - self.vmin) / stage_levels
            
            stage_code = int((current_voltage - self.vmin) / stage_lsb)
            stage_code = max(0, min(stage_levels - 1, stage_code))
            self._stage_codes.append(stage_code)
            
            # Compute residue
            dac_voltage = self.vmin + stage_code * stage_lsb
            residue = current_voltage - dac_voltage
            
            # Fire residue event
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
            self._stage_residues.append(residue)
            
            # Process for next stage
            if stage_idx < len(self._stages) - 1:
                # Fire gain event
                gain_event = InterstageGainEvent(
                    timestamp=timestamp,
                    stage_index=stage_idx,
                    input_voltage=residue,
                    ideal_gain=stage.gain,
                    actual_gain=stage.gain,
                )
                self.fire(gain_event)
                
                # Apply gain with potential non-idealities from event
                vmid = (self.vref + self.vmin) / 2
                if not gain_event.cancelled:
                    current_voltage = residue * gain_event.actual_gain + vmid + gain_event.offset
                    if gain_event.noise_sigma > 0:
                        current_voltage += np.random.normal(0, gain_event.noise_sigma)
                else:
                    current_voltage = residue * stage.gain + vmid
                
                current_voltage = np.clip(current_voltage, self.vmin, self.vref)
        
        return ideal_code
    
    def _simple_combine(self) -> int:
        """Simple code combination without correction."""
        code = 0
        shift = 0
        
        for i in range(len(self._stages) - 1, -1, -1):
            stage = self._stages[i]
            stage_code = self._stage_codes[i]
            
            if i == len(self._stages) - 1:
                code = stage_code
                shift = stage.bits
            else:
                effective_bits = stage.bits - stage.redundancy
                code = (stage_code << shift) + code
                shift += effective_bits
        
        max_code = 2 ** self.bits - 1
        return min(max(0, code), max_code)
    
    def _digital_correction(self) -> int:
        """Digital error correction for redundancy."""
        # For proper pipeline digital correction:
        # Each stage contributes bits, with redundancy allowing overlap
        
        # Compute bit positions for each stage
        bit_position = self.bits  # Start from MSB
        code = 0
        
        for i, stage in enumerate(self._stages):
            stage_code = self._stage_codes[i]
            effective_bits = stage.bits - stage.redundancy
            
            # Move to this stage's bit position
            bit_position -= effective_bits
            
            # Add stage contribution (can be negative relative to expected)
            # For last stage, use all bits
            if i == len(self._stages) - 1:
                code += stage_code
            else:
                # Weight is based on position
                weight = 2 ** bit_position
                # Subtract mid-code to center the contribution
                mid_code = 2 ** (stage.bits - 1)
                code += (stage_code - mid_code) * weight // (2 ** (stage.bits - effective_bits - 1))
        
        # Add offset to center the result
        code += 2 ** (self.bits - 1)
        
        max_code = 2 ** self.bits - 1
        return min(max(0, code), max_code)
    
    def get_stage_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all stages."""
        info = []
        for i, stage in enumerate(self._stages):
            info.append({
                'index': i,
                'bits': stage.bits,
                'effective_bits': stage.effective_bits,
                'gain': stage.gain,
                'redundancy': stage.redundancy,
                'has_custom_adc': stage.adc is not None,
            })
        return info
    
    def plot_stages(
        self,
        result: Optional[SimulationResult] = None,
        *,
        show: bool = True,
        save: Optional[str] = None,
    ) -> Any:
        """Plot per-stage analysis."""
        import matplotlib.pyplot as plt
        
        if result is None:
            if self._result is None:
                raise ValueError("No simulation result. Run sim() first.")
            result = self._result
        
        n_stages = len(self._stages)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        fig.suptitle(f'{self.name} Stage Analysis', fontsize=14, fontweight='bold')
        
        ax1 = axes[0, 0]
        stage_bits = [s.bits for s in self._stages]
        stage_names = [f'Stage {i}' for i in range(n_stages)]
        bars = ax1.bar(stage_names, stage_bits, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Bits')
        ax1.set_title('Bits per Stage')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, bits in zip(bars, stage_bits):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(bits), ha='center', va='bottom', fontsize=10)
        
        ax2 = axes[0, 1]
        gains = [s.gain for s in self._stages]
        ax2.bar(stage_names, gains, color='coral', alpha=0.8)
        ax2.set_ylabel('Gain')
        ax2.set_title('Inter-stage Gain')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = axes[1, 0]
        ax3.plot(result.timestamps * 1e6, result.input_signal, 'b-', 
                 linewidth=1, label='Input')
        ax3.plot(result.timestamps * 1e6, result.reconstructed, 'r--',
                 linewidth=1, label='Output', alpha=0.8)
        ax3.set_xlabel('Time (μs)')
        ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Signal Reconstruction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        redundancy = [s.redundancy for s in self._stages]
        ax4.bar(stage_names, redundancy, color='seagreen', alpha=0.8)
        ax4.set_ylabel('Redundancy Bits')
        ax4.set_title('Redundancy per Stage')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig
