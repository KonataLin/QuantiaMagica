"""ADC implementations module."""

from .sar import (
    SARADC,
    SamplingEvent,
    CapacitorSwitchEvent,
    ComparatorEvent,
    OutputCodeEvent,
    ConversionStartEvent,
    ConversionEndEvent,
    BitDecisionEvent,
)

from .pipeline import (
    PipelineADC,
    StageEvent,
    ResidueEvent,
    InterstageGainEvent,
)

from .sigma_delta import (
    SigmaDeltaADC,
    QuantizerEvent,
)

__all__ = [
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
    "InterstageGainEvent",
    # Sigma-Delta ADC
    "SigmaDeltaADC",
    "QuantizerEvent",
]
