"""
Example 01: Basic SAR ADC Simulation
====================================

This example demonstrates the simplest use case of QuantiaMagica:
creating a SAR ADC, running a simulation, and viewing results.

首次使用前，请先安装模块:
    cd QuantiaMagica
    pip install -e .

Usage:
    python 01_basic_sar.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SARADC

# Create a 10-bit SAR ADC with 1V reference
adc = SARADC(bits=10, vref=1.0)

# Run simulation with default sine wave input
# This generates a coherent sine wave and converts all samples

result = adc.sim(n_samples=1024, fs=1e6, fin=10e3)

# Print basic info
print(f"ADC: {adc.name}")
print(f"Resolution: {adc.bits} bits")
print(f"LSB: {adc.lsb * 1e3:.3f} mV")
print(f"Samples: {result.n_samples}")

# Quick metrics
print(f"\nPerformance:")
print(f"  ENOB: {adc.enob():.2f} bits")
print(f"  SNR:  {adc.snr():.2f} dB")
print(f"  SFDR: {adc.sfdr():.2f} dB")

# Plot results
adc.plot()

# Plot spectrum with metrics
adc.spectrum()

# Save data if needed
# result.save("output.npz")
# result.save("output.csv", format="csv")
