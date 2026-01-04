"""
Pytest configuration and fixtures for QuantiaMagica tests.
"""

import pytest
import numpy as np
from quantiamagica import SARADC, PipelineADC


@pytest.fixture
def sar_10bit():
    """10-bit SAR ADC fixture."""
    return SARADC(bits=10, vref=1.0)


@pytest.fixture
def sar_12bit():
    """12-bit SAR ADC fixture."""
    return SARADC(bits=12, vref=1.0)


@pytest.fixture
def pipeline_12bit():
    """12-bit Pipeline ADC fixture."""
    return PipelineADC(bits=12, stages=4)


@pytest.fixture
def sine_signal():
    """Generate a test sine signal."""
    n = 1024
    fs = 1e6
    fin = 10e3
    t = np.arange(n) / fs
    signal = 0.5 + 0.4 * np.sin(2 * np.pi * fin * t)
    return signal


@pytest.fixture
def ramp_signal():
    """Generate a test ramp signal."""
    return np.linspace(0.0, 1.0, 4096)
