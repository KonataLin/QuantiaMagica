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

# Auto-optimize: find best fin and amplitude using Differential Evolution
# This automatically searches for optimal test parameters
opt_result = adc.sim_auto(fs=1e6)

# Print basic info
print(f"\nADC: {adc.name}")
print(f"Resolution: {adc.bits} bits")
print(f"LSB: {adc.lsb * 1e3:.3f} mV")

# Optimal parameters found
print(f"\nOptimal Parameters:")
print(f"  fin: {opt_result['best_fin']:.2f} Hz")
print(f"  amplitude: {opt_result['best_amplitude']:.4f} V")

# Quick metrics (using optimized result)
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
