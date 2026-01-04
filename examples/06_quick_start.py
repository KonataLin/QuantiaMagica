"""
Example 06: Quick Start - Minimal Code Examples
================================================

This file shows the simplest possible API calls for common tasks.
Copy these snippets directly into your code!

Usage:
    python 06_quick_start.py
"""

# =============================================================================
# 1. One-liner: Simulate and get ENOB
# =============================================================================
import sys
from pathlib import Path

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import SARADC
# One-liner approach - create, simulate, then get ENOB

# Proper minimal way:
adc = SARADC(10)
adc.sim()
print(f"1. Quick ENOB: {adc.enob():.2f} bits")


# =============================================================================
# 2. Three lines: Full simulation and plot
# =============================================================================

from quantiamagica import SARADC
adc = SARADC(bits=12, vref=1.0)
adc.sim(fin=10e3)
adc.plot()


# =============================================================================
# 3. Add noise in 2 lines
# =============================================================================

from quantiamagica import SARADC, SamplingEvent
import numpy as np

adc = SARADC(12)

@adc.on(SamplingEvent)
def noise(e): e.voltage += np.random.normal(0, 1e-3)

adc.sim()
print(f"3. With noise ENOB: {adc.enob():.2f} bits")


# =============================================================================
# 4. Get all metrics at once
# =============================================================================

from quantiamagica import SARADC
from quantiamagica.analysis import Analyzer

adc = SARADC(12)
result = adc.sim()
print(Analyzer(result).summary())


# =============================================================================
# 5. Pipeline ADC in 3 lines
# =============================================================================

from quantiamagica import PipelineADC
pipe = PipelineADC(bits=14, stages=4)
pipe.sim()
print(f"5. Pipeline ENOB: {pipe.enob():.2f} bits")


# =============================================================================
# 6. Save results
# =============================================================================

from quantiamagica import SARADC
adc = SARADC(10)
result = adc.sim()
# result.save("data.npz")        # NumPy format
# result.save("data.csv", format="csv")  # CSV format
# result.save("data.json", format="json")  # JSON format
print("6. Data export: npz, csv, json formats available")


# =============================================================================
# 7. Spectrum analysis
# =============================================================================

from quantiamagica import SARADC
adc = SARADC(12)
adc.sim(n_samples=4096, fin=50e3)
freqs, power_db, metrics = adc.spectrum(show=False)
print(f"7. Metrics: SNR={metrics['snr']:.1f}dB, SFDR={metrics['sfdr']:.1f}dB")


# =============================================================================
# 8. Custom input signal
# =============================================================================

from quantiamagica import SARADC
from quantiamagica.utils import generate_sine
import numpy as np

adc = SARADC(10, vref=1.0)

# Use built-in generator
t, signal = generate_sine(n_samples=1024, amplitude=0.4, offset=0.5)
result = adc.sim(signal)

# Or use your own numpy array
my_signal = 0.5 + 0.4 * np.sin(2 * np.pi * 10e3 * np.arange(1024) / 1e6)
result = adc.sim(my_signal)
print(f"8. Custom signal ENOB: {adc.enob():.2f} bits")


# =============================================================================
# 9. Compare ideal vs non-ideal
# =============================================================================

from quantiamagica import SARADC, ComparatorEvent

# Ideal
ideal = SARADC(12)
ideal.sim()

# Non-ideal  
noisy = SARADC(12)

@noisy.on(ComparatorEvent)
def add_offset(e): e.offset = 1e-3  # 1mV offset

noisy.sim()

print(f"9. Ideal: {ideal.enob():.2f}, Non-ideal: {noisy.enob():.2f} bits")


# =============================================================================
# 10. INL/DNL with one call
# =============================================================================

from quantiamagica import SARADC
adc = SARADC(10)
adc.sim(signal='ramp', n_samples=4096)
print(f"10. INL max: {max(abs(adc.inl())):.3f} LSB")
print(f"    DNL max: {max(abs(adc.dnl())):.3f} LSB")


# =============================================================================
# 11. Auto-optimize: find best fin and amplitude (sim_auto)
# =============================================================================

from quantiamagica import SARADC

adc = SARADC(12)

# One call - automatically finds optimal parameters using Differential Evolution
result = adc.sim_auto(fs=1e6)

print(f"11. sim_auto results:")
print(f"    Best fin: {result['best_fin']:.2f} Hz")
print(f"    Best amplitude: {result['best_amplitude']:.4f} V")
print(f"    Best ENOB: {result['best_enob']:.4f} bits")
print(f"    Converged in {result['generations']} generations")

# Result is saved, can directly plot
adc.report('spectrum')
