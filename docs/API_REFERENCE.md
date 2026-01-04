# QuantiaMagica API Reference

Complete API documentation for QuantiaMagica ADC simulator.

---

## Table of Contents

1. [Core Module](#core-module)
2. [SAR ADC](#sar-adc)
3. [Pipeline ADC](#pipeline-adc)
4. [Events](#events)
5. [Analysis](#analysis)
6. [Utilities](#utilities)

---

## Core Module

### `quantiamagica.ADConverter`

Abstract base class for all ADC implementations.

```python
class ADConverter(ABC):
    """
    Abstract base class for all ADC implementations.
    
    Parameters
    ----------
    bits : int
        Resolution in bits.
    vref : float, optional
        Reference voltage (full scale). Default: 1.0
    vmin : float, optional
        Minimum input voltage. Default: 0.0
    name : str, optional
        ADC instance name for identification.
    
    Attributes
    ----------
    lsb : float
        Least significant bit voltage = (vref - vmin) / 2^bits
    max_code : int
        Maximum output code = 2^bits - 1
    levels : int
        Number of quantization levels = 2^bits
    result : SimulationResult
        Most recent simulation result.
    event_bus : EventBus
        Event dispatcher for this ADC.
    """
```

#### Methods

##### `sim(**kwargs) -> SimulationResult`

Run ADC simulation.

```python
def sim(
    self,
    input_voltage: Union[float, NDArray, None] = None,
    *,
    n_samples: int = 1024,
    fs: float = 1e6,
    fin: float = 1e3,
    amplitude: Optional[float] = None,
    offset: Optional[float] = None,
    signal: str = "sine",
    log_events: bool = False,
) -> SimulationResult:
    """
    Parameters
    ----------
    input_voltage : float, NDArray, or None
        Input signal. If None, generates test signal.
    n_samples : int
        Number of samples for generated signal.
    fs : float
        Sampling frequency in Hz.
    fin : float
        Input signal frequency in Hz.
    amplitude : float, optional
        Signal amplitude (defaults to 0.45 * (vref-vmin)).
    offset : float, optional
        DC offset (defaults to mid-range).
    signal : str
        Signal type: 'sine', 'ramp', 'dc'.
    log_events : bool
        Enable event logging for debugging.
    
    Returns
    -------
    SimulationResult
        Container with input, output codes, timestamps, etc.
    """
```

##### `plot(**kwargs)`

Plot simulation results.

```python
def plot(
    self,
    result: Optional[SimulationResult] = None,
    *,
    show: bool = True,
    save: Optional[str] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (12, 8),
) -> matplotlib.figure.Figure
```

##### `spectrum(**kwargs)`

Compute and plot frequency spectrum.

```python
def spectrum(
    self,
    result: Optional[SimulationResult] = None,
    *,
    window: str = "hann",
    show: bool = True,
    save: Optional[str] = None,
) -> Tuple[NDArray, NDArray, Dict[str, float]]:
    """
    Returns
    -------
    freqs : NDArray
        Frequency bins in Hz.
    spectrum_db : NDArray
        Power spectrum in dB.
    metrics : Dict[str, float]
        Contains 'snr', 'sfdr', 'enob', 'thd', 'sinad'.
    """
```

##### `on(event_type, **kwargs)`

Decorator to register event handler.

```python
def on(
    self,
    event_type: Type[Event],
    priority: EventPriority = EventPriority.NORMAL,
    ignore_cancelled: bool = False,
) -> Callable:
    """
    Example
    -------
    >>> @adc.on(SamplingEvent)
    ... def handler(event):
    ...     event.voltage += noise
    """
```

##### Metric Methods

```python
def enob(self) -> float      # Effective Number of Bits
def snr(self) -> float       # Signal-to-Noise Ratio (dB)
def sfdr(self) -> float      # Spurious-Free Dynamic Range (dB)
def thd(self) -> float       # Total Harmonic Distortion (dB)
def inl(self, plot=False) -> NDArray  # INL in LSB
def dnl(self, plot=False) -> NDArray  # DNL in LSB
```

---

## SAR ADC

### `quantiamagica.SARADC`

Successive Approximation Register ADC.

```python
class SARADC(ADConverter):
    """
    Parameters
    ----------
    bits : int
        Resolution in bits.
    vref : float
        Reference voltage. Default: 1.0
    vmin : float
        Minimum input voltage. Default: 0.0
    cap_unit : float
        Unit capacitance in fF. Default: 50.0
    comparator_noise : float
        Comparator noise sigma in V. Default: 0.0
    comparator_offset : float
        Comparator offset in V. Default: 0.0
    
    Attributes
    ----------
    capacitances : NDArray
        Capacitor values for each bit (fF).
    weights : NDArray
        Bit weights (normalized).
    cap_mismatches : NDArray
        Capacitor mismatch ratios.
    total_capacitance : float
        Total capacitance in fF.
    """
```

#### Methods

##### `set_capacitor_mismatch(mismatches=None, sigma=0.0)`

Set capacitor mismatches for non-ideality modeling.

```python
# Direct values
adc.set_capacitor_mismatch(mismatches=np.array([1.01, 0.99, 1.02, ...]))

# Random with sigma
adc.set_capacitor_mismatch(sigma=0.005)  # 0.5% mismatch
```

##### `set_radix(radix)`

Set non-binary radix for redundant SAR.

```python
adc.set_radix(1.85)  # Radix-1.85 redundant SAR
```

##### `plot_dac(show=True, save=None)`

Plot DAC transfer characteristic and error.

---

## Pipeline ADC

### `quantiamagica.PipelineADC`

Multi-stage pipeline ADC with digital error correction.

```python
class PipelineADC(ADConverter):
    """
    Parameters
    ----------
    bits : int
        Total resolution in bits.
    vref : float
        Reference voltage.
    stages : int
        Number of pipeline stages.
    bits_per_stage : int
        Bits resolved per stage.
    redundancy : int
        Redundancy bits for digital correction.
    interstage_gain : float, optional
        Inter-stage amplifier gain.
    
    Attributes
    ----------
    num_stages : int
        Number of pipeline stages.
    stage_configs : List[PipelineStage]
        Stage configuration objects.
    digital_correction : bool
        Enable digital error correction.
    """
```

#### Class Methods

##### `PipelineADC.from_stages(adcs, gain, name=None)`

Create pipeline from custom ADC stages.

```python
stage1 = SARADC(bits=4)
stage2 = SARADC(bits=4)
stage3 = SARADC(bits=6)
pipeline = PipelineADC.from_stages([stage1, stage2, stage3], gain=4.0)
```

#### Methods

##### `get_stage_info() -> List[Dict]`

Get detailed information about all stages.

##### `plot_stages(result=None, show=True, save=None)`

Plot per-stage analysis.

---

## Events

### Event Priority

```python
class EventPriority(IntEnum):
    LOWEST = 0    # Executes first
    LOW = 1
    NORMAL = 2    # Default
    HIGH = 3
    HIGHEST = 4   # Executes last before MONITOR
    MONITOR = 5   # Read-only observation
```

### SAR ADC Events

#### `SamplingEvent`

```python
@dataclass
class SamplingEvent(Event, Cancellable):
    voltage: float           # Sampled voltage (modifiable)
    sample_index: int        # Current sample index
    sampling_capacitance: float  # Total capacitance (fF)
    sampling_time: float     # Sampling time (s)
    bandwidth: float         # Input bandwidth (Hz)
```

#### `CapacitorSwitchEvent`

```python
@dataclass
class CapacitorSwitchEvent(Event, Cancellable):
    bit_index: int           # Bit position (0=LSB)
    capacitance: float       # Nominal capacitance (fF)
    capacitance_actual: float # With mismatch (modifiable)
    weight: float            # Bit weight (modifiable)
    switch_voltage: float    # Voltage switched to
    charge_injection: float  # Charge injection (fC)
    settling_error: float    # Settling error fraction
    parasitic: float         # Parasitic capacitance (fF)
    dac_voltage: float       # Current DAC output
    comparator_input: float  # Comparator input voltage
```

#### `ComparatorEvent`

```python
@dataclass
class ComparatorEvent(Event, Cancellable):
    bit_index: int           # Current bit
    input_voltage: float     # Comparator input
    threshold: float         # Threshold (modifiable)
    decision: bool           # Output (modifiable)
    offset: float            # Offset voltage (modifiable)
    noise_sigma: float       # Noise sigma (modifiable)
    delay: float             # Comparator delay (s)
    metastable: bool         # Metastability flag
    kickback: float          # Kickback voltage (mV)
```

#### `BitDecisionEvent`

```python
@dataclass
class BitDecisionEvent(Event, Cancellable):
    bit_index: int           # Bit position
    bit_value: int           # 0 or 1 (modifiable)
    residue: float           # Remaining voltage
    accumulated_code: int    # Code so far
```

#### `OutputCodeEvent`

```python
@dataclass
class OutputCodeEvent(Event):
    code: int                # Final code (modifiable)
    input_voltage: float     # Original input
    conversion_time: float   # Total time (s)
    bit_decisions: List[int] # All bit decisions
```

### Pipeline ADC Events

#### `InterstageGainEvent`

```python
@dataclass
class InterstageGainEvent(Event, Cancellable):
    stage_index: int         # Current stage
    input_voltage: float     # Amplifier input
    ideal_gain: float        # Ideal gain
    actual_gain: float       # With errors (modifiable)
    offset: float            # Offset voltage (modifiable)
    noise_sigma: float       # Noise sigma (modifiable)
    output_voltage: float    # Amplified output
    bandwidth: float         # Bandwidth (Hz)
    settling_error: float    # Settling error
```

---

## Analysis

### `quantiamagica.analysis.Analyzer`

Comprehensive ADC analyzer class.

```python
class Analyzer:
    """
    Parameters
    ----------
    result : SimulationResult
        Simulation result to analyze.
    bits : int, optional
        ADC resolution.
    fs : float, optional
        Sampling frequency.
    
    Properties
    ----------
    enob : float      # Effective Number of Bits
    snr : float       # Signal-to-Noise Ratio (dB)
    sfdr : float      # Spurious-Free Dynamic Range (dB)
    thd : float       # Total Harmonic Distortion (dB)
    sinad : float     # SINAD (dB)
    inl : NDArray     # INL array (LSB)
    dnl : NDArray     # DNL array (LSB)
    inl_max : float   # Maximum |INL| (LSB)
    dnl_max : float   # Maximum |DNL| (LSB)
    
    Methods
    -------
    summary() -> str
        Formatted summary string.
    to_dict() -> Dict
        All metrics as dictionary.
    plot_all(show=True, save=None)
        Comprehensive analysis plots.
    """
```

### Standalone Functions

```python
from quantiamagica.analysis import spectrum, enob, snr, sfdr, thd, inl, dnl

# All functions take: codes, bits, fs (optional)
freqs, power_db, metrics = spectrum(codes, bits=12, fs=1e6)
enob_val = enob(codes, bits=12)
snr_val = snr(codes, bits=12)
```

---

## Utilities

### Signal Generation

```python
from quantiamagica.utils import generate_sine, generate_ramp, generate_multitone

# Sine wave (coherent sampling by default)
t, signal = generate_sine(
    n_samples=1024,
    fs=1e6,
    fin=10e3,
    amplitude=0.45,
    offset=0.5,
)

# Ramp
t, signal = generate_ramp(n_samples=4096, vmin=0, vmax=1)

# Multi-tone
t, signal = generate_multitone(
    n_samples=2048,
    frequencies=[1e3, 2e3, 5e3],
    amplitudes=[0.2, 0.15, 0.1],
)
```

### Helper Functions

```python
from quantiamagica.utils import (
    ideal_code,
    codes_to_voltage,
    add_noise,
    thermal_noise_voltage,
)

# Ideal quantization
code = ideal_code(voltage=0.5, bits=12, vref=1.0)

# Reconstruct voltage
voltages = codes_to_voltage(codes, bits=12, vref=1.0)

# Add Gaussian noise
noisy_signal = add_noise(signal, sigma=1e-3)

# kT/C noise voltage
noise_v = thermal_noise_voltage(capacitance_fF=500, temperature_K=300)
```

---

## SimulationResult

Container for simulation results.

```python
@dataclass
class SimulationResult:
    input_signal: NDArray[np.float64]    # Input voltages
    output_codes: NDArray[np.int64]      # Digital codes
    timestamps: NDArray[np.float64]      # Time points
    reconstructed: NDArray[np.float64]   # Reconstructed voltages
    events: List[Event]                  # Event log (if enabled)
    metadata: Dict[str, Any]             # ADC info, fs, etc.
    
    @property
    def n_samples(self) -> int
    
    def to_dict(self) -> Dict
    def save(self, path: str, format: str = "npz")
```

---

## Type Annotations

QuantiaMagica uses standard Python type hints throughout:

```python
from typing import Optional, List, Dict, Tuple, Union, Callable
from numpy.typing import NDArray
import numpy as np
```
