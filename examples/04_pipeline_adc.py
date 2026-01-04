"""
Example 04: Pipeline ADC Simulation
====================================

This example demonstrates Pipeline ADC usage including:
- Basic pipeline configuration
- Custom stage configurations
- Inter-stage gain error modeling
- Digital error correction

Usage:
    python 04_pipeline_adc.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import (
    PipelineADC,
    SARADC,
    InterstageGainEvent,
    StageEvent,
    ResidueEvent,
    FlashEvent,
    EventPriority,
)

# =============================================================================
# Example 1: Basic Pipeline ADC
# =============================================================================

print("=" * 60)
print("Example 1: Basic 12-bit Pipeline ADC")
print("=" * 60)

# Create a 12-bit pipeline with 4 stages
pipeline = PipelineADC(
    bits=12,
    vref=1.0,
    stages=4,
    bits_per_stage=3,
    redundancy=1,
    name="12-bit-Pipeline"
)

# Print stage configuration
print("\nStage Configuration:")
for info in pipeline.get_stage_info():
    print(f"  Stage {info['index']}: {info['bits']} bits, "
          f"gain={info['gain']:.1f}, redundancy={info['redundancy']}")

# Run simulation
result = pipeline.sim(n_samples=1024, fs=10e6, fin=100e3)

print(f"\nPerformance:")
print(f"  ENOB: {pipeline.enob():.2f} bits")
print(f"  SNR:  {pipeline.snr():.2f} dB")
print(f"  SFDR: {pipeline.sfdr():.2f} dB")


# =============================================================================
# Example 2: Pipeline with Gain Error
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Pipeline with Inter-stage Gain Error")
print("=" * 60)

pipeline_error = PipelineADC(bits=12, vref=1.0, stages=4, name="Pipeline-GainError")

@pipeline_error.on(InterstageGainEvent)
def add_gain_error(event):
    """Model finite opamp gain causing gain error."""
    # Finite opamp gain of 1000 causes ~0.1% gain error
    opamp_gain = 1000
    gain_error = 1 - 1/opamp_gain
    event.actual_gain = event.ideal_gain * gain_error
    
    # Also add some offset
    event.offset = 0.5e-3  # 0.5mV offset
    
    # And amplifier noise
    event.noise_sigma = 0.1e-3  # 0.1mV noise

result_error = pipeline_error.sim(n_samples=1024, fs=10e6, fin=100e3)

print(f"With gain error (opamp gain=1000):")
print(f"  ENOB: {pipeline_error.enob():.2f} bits")
print(f"  SNR:  {pipeline_error.snr():.2f} dB")


# =============================================================================
# Example 3: Custom Pipeline from SAR Stages
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Pipeline with Custom SAR Stages")
print("=" * 60)

# Create individual SAR stages
stage1 = SARADC(bits=4, vref=1.0, name="Stage1-SAR")
stage2 = SARADC(bits=4, vref=1.0, name="Stage2-SAR")
stage3 = SARADC(bits=6, vref=1.0, name="Stage3-SAR")

# Add mismatch to stage 1 (using the CapacitorSwitchEvent from stage1's module)
from quantiamagica import CapacitorSwitchEvent

@stage1.on(CapacitorSwitchEvent)
def stage1_mismatch(event):
    event.capacitance_actual *= 1 + np.random.normal(0, 0.01)

# Build pipeline from stages
custom_pipeline = PipelineADC.from_stages(
    [stage1, stage2, stage3],
    gain=4.0,  # 2^(4-2) for 1-bit redundancy
    name="Custom-SAR-Pipeline"
)

result_custom = custom_pipeline.sim(n_samples=1024, fs=10e6, fin=100e3)

print(f"Custom SAR Pipeline (4+4+6 bits):")
print(f"  Total bits: {custom_pipeline.bits}")
print(f"  ENOB: {custom_pipeline.enob():.2f} bits")


# =============================================================================
# Example 4: Stage-by-stage monitoring
# =============================================================================

print("\n" + "=" * 60)
print("Example 4: Stage Monitoring")
print("=" * 60)

pipeline_monitor = PipelineADC(bits=10, vref=1.0, stages=3, name="Monitored-Pipeline")

stage_data = {"voltages": [], "codes": [], "residues": []}

@pipeline_monitor.on(StageEvent, priority=EventPriority.MONITOR)
def monitor_stage(event):
    """Log data at each stage."""
    if event.sample_index == 0:  # Only first sample
        print(f"  Stage {event.stage_index}: input={event.input_voltage:.4f}V")

@pipeline_monitor.on(ResidueEvent, priority=EventPriority.MONITOR)
def monitor_residue(event):
    """Log residue computation."""
    if event.source._sample_index == 0:
        print(f"    -> residue={event.residue:.4f}V")

print("\nFirst sample conversion trace:")
result_monitor = pipeline_monitor.sim(n_samples=100, fs=10e6, fin=100e3)


# =============================================================================
# Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Generating Plots...")
print("=" * 60)

# Plot comparison
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
fig.suptitle('Pipeline ADC Analysis', fontsize=14, fontweight='bold')

# Ideal pipeline spectrum
ax1 = axes[0, 0]
freqs, spec, _ = pipeline.spectrum(show=False)
ax1.plot(freqs/1e3, spec, 'b-', linewidth=0.8)
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('Power (dB)')
ax1.set_title(f'Ideal Pipeline (ENOB={pipeline.enob():.2f})')
ax1.grid(True, alpha=0.3)

# With gain error spectrum
ax2 = axes[0, 1]
freqs2, spec2, _ = pipeline_error.spectrum(show=False)
ax2.plot(freqs2/1e3, spec2, 'r-', linewidth=0.8)
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Power (dB)')
ax2.set_title(f'With Gain Error (ENOB={pipeline_error.enob():.2f})')
ax2.grid(True, alpha=0.3)

# Time domain
ax3 = axes[1, 0]
t = result.timestamps * 1e6
ax3.plot(t[:100], result.input_signal[:100], 'b-', label='Input', linewidth=1)
ax3.plot(t[:100], result.reconstructed[:100], 'r--', label='Output', linewidth=1)
ax3.set_xlabel('Time (μs)')
ax3.set_ylabel('Voltage (V)')
ax3.set_title('Time Domain (first 100 samples)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Stage configuration bar chart
ax4 = axes[1, 1]
stage_info = pipeline.get_stage_info()
stage_names = [f'S{i["index"]}' for i in stage_info]
stage_bits = [i['bits'] for i in stage_info]
bars = ax4.bar(stage_names, stage_bits, color='steelblue', alpha=0.8)
ax4.set_ylabel('Bits')
ax4.set_title('Bits per Stage')
ax4.grid(True, alpha=0.3, axis='y')
for bar, bits in zip(bars, stage_bits):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(bits), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('pipeline_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved to pipeline_analysis.png")
