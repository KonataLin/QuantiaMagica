"""
QuantiaMagica - ADC Behavioral Event-Driven Simulator
======================================================

A Bukkit-style event-driven ADC simulation framework for Python.

Quick Start
-----------
>>> from quantiamagica import SARADC, on_event, SamplingEvent
>>> 
>>> adc = SARADC(bits=10, vref=1.0)
>>> 
>>> @adc.on(SamplingEvent)
... def handle_sampling(event):
...     print(f"Sampled: {event.voltage}V")
>>> 
>>> adc.sim(input_voltage=0.5)
>>> adc.plot()

Author: QuantiaMagica Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "QuantiaMagica Team"

from .core.events import (
    Event,
    EventBus,
    EventPriority,
    Cancellable,
    on_event,
)

from .core.base import ADConverter, SimulationResult

from ._analysis_impl import (
    compute_spectrum,
    compute_inband_snr,
    compute_inl_dnl,
    auto_time_unit,
    auto_freq_unit,
    apply_jssc_style,
    plot_spectrum,
    plot_comparison,
)

from .adc.sar import (
    SARADC,
    SamplingEvent,
    CapacitorSwitchEvent,
    ComparatorEvent,
    OutputCodeEvent,
    ConversionStartEvent,
    ConversionEndEvent,
    BitDecisionEvent,
)

from .adc.pipeline import (
    PipelineADC,
    StageEvent,
    ResidueEvent,
    FlashEvent,
    InterstageGainEvent,
)

from .adc.sigma_delta import (
    SigmaDeltaADC,
    QuantizerEvent as SDQuantizerEvent,  # 别名避免与其他模块冲突
)
# 为了兼容性，也直接导出
QuantizerEvent = SDQuantizerEvent

from .analysis import (
    Analyzer,
    spectrum,
    enob,
    snr,
    sfdr,
    thd,
    inl,
    dnl,
)

from .utils import (
    generate_sine,
    generate_ramp,
    ideal_code,
)

from .signals import Signal

from .plotting import (
    jssc_style,
    apply_jssc_style,
    plot_spectrum_jssc,
    plot_inl_dnl_jssc,
    plot_time_domain_jssc,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Event",
    "EventBus", 
    "EventPriority",
    "Cancellable",
    "on_event",
    "ADConverter",
    "SimulationResult",
    # SAR ADC
    "SARADC",
    "SamplingEvent",
    "CapacitorSwitchEvent", 
    "ComparatorEvent",
    "OutputCodeEvent",
    "ConversionStartEvent",
    "ConversionEndEvent",
    "BitDecisionEvent",
    # Pipeline ADC
    "PipelineADC",
    "StageEvent",
    "ResidueEvent",
    "FlashEvent",
    "InterstageGainEvent",
    # Sigma-Delta ADC
    "SigmaDeltaADC",
    "SDQuantizerEvent",
    "QuantizerEvent",
    # Analysis functions
    "Analyzer",
    "spectrum",
    "enob",
    "snr",
    "sfdr",
    "thd",
    "inl",
    "dnl",
    "compute_spectrum",
    "compute_inband_snr",
    "compute_inl_dnl",
    "plot_spectrum",
    "plot_comparison",
    # Utils
    "generate_sine",
    "generate_ramp",
    "ideal_code",
    "auto_time_unit",
    "auto_freq_unit",
    # Signal
    "Signal",
    # Plotting
    "jssc_style",
    "apply_jssc_style",
    "plot_spectrum_jssc",
    "plot_inl_dnl_jssc",
    "plot_time_domain_jssc",
]
