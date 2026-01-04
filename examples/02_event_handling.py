"""
Example 02: Event Handling (Bukkit-style API)
=============================================

This example shows how to use the event system to hook into
ADC operations and modify behavior - similar to Minecraft Bukkit plugins.

Usage:
    python 02_event_handling.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import (
    SARADC,
    SamplingEvent,
    CapacitorSwitchEvent,
    ComparatorEvent,
    OutputCodeEvent,
    ConversionStartEvent,
    EventPriority,
)

# Create ADC
adc = SARADC(bits=10, vref=1.0, name="Event-Demo-ADC")

# =============================================================================
# Example 1: Simple event listener with decorator
# =============================================================================

@adc.on(SamplingEvent)
def on_sampling(event):
    """Called every time a sample is taken."""
    # Add thermal noise based on capacitance
    kT = 1.38e-23 * 300  # Boltzmann * Temperature
    C = event.sampling_capacitance * 1e-15  # Convert fF to F
    noise_sigma = np.sqrt(kT / C)
    event.voltage += np.random.normal(0, noise_sigma)


# =============================================================================
# Example 2: Capacitor mismatch modeling
# =============================================================================

@adc.on(CapacitorSwitchEvent)
def add_capacitor_mismatch(event):
    """Model random capacitor mismatch."""
    # Add 0.5% random mismatch
    mismatch = 1 + np.random.normal(0, 0.005)
    event.capacitance_actual = event.capacitance * mismatch
    
    # You can also modify the weight directly
    # event.weight *= mismatch


# =============================================================================
# Example 3: Comparator non-idealities
# =============================================================================

@adc.on(ComparatorEvent)
def comparator_effects(event):
    """Add comparator offset and noise."""
    # Static offset
    event.offset = 0.5e-3  # 0.5mV offset
    
    # Random noise
    event.noise_sigma = 0.1e-3  # 0.1mV noise


# =============================================================================
# Example 4: Monitor output (read-only with MONITOR priority)
# =============================================================================

@adc.on(OutputCodeEvent, priority=EventPriority.MONITOR)
def log_output(event):
    """Log conversion results (monitor priority = read-only)."""
    # Only print first 5 samples to avoid spam
    if event.source._sample_index < 5:
        print(f"Sample {event.source._sample_index}: "
              f"V={event.input_voltage:.4f}V -> Code={event.code}")


# =============================================================================
# Example 5: Priority-based handler ordering
# =============================================================================

@adc.on(ConversionStartEvent, priority=EventPriority.HIGHEST)
def high_priority_handler(event):
    """Runs first due to HIGHEST priority."""
    pass  # Could set up state here


@adc.on(ConversionStartEvent, priority=EventPriority.LOWEST)
def low_priority_handler(event):
    """Runs last before MONITOR handlers."""
    pass  # Could clean up here


# =============================================================================
# Run simulation
# =============================================================================

print("Running simulation with event handlers...")
print("-" * 50)

result = adc.sim(n_samples=1024, fs=1e6, fin=10e3)

print("-" * 50)
print(f"\nWith non-idealities:")
print(f"  ENOB: {adc.enob():.2f} bits")
print(f"  SNR:  {adc.snr():.2f} dB")

# Compare with ideal ADC
ideal_adc = SARADC(bits=10, vref=1.0, name="Ideal-ADC")
ideal_adc.sim(n_samples=1024, fs=1e6, fin=10e3)

print(f"\nIdeal (no handlers):")
print(f"  ENOB: {ideal_adc.enob():.2f} bits")
print(f"  SNR:  {ideal_adc.snr():.2f} dB")

# Plot both
adc.plot()
