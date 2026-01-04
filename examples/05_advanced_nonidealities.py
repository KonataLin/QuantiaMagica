"""
Example 05: Advanced Non-Idealities Modeling
=============================================

This example shows comprehensive non-ideality modeling including:
- Capacitor mismatch with Monte Carlo analysis
- Thermal noise
- Comparator metastability
- Charge injection
- Incomplete settling

Usage:
    python 05_advanced_nonidealities.py
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径（如果未安装模块）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from quantiamagica import (
    SARADC,
    SamplingEvent,
    CapacitorSwitchEvent,
    ComparatorEvent,
    BitDecisionEvent,
    EventPriority,
)
from quantiamagica.utils import thermal_noise_voltage


# =============================================================================
# Comprehensive Non-Ideality Plugin
# =============================================================================

class RealisticSARPlugin:
    """
    Plugin that models realistic SAR ADC non-idealities.
    
    Parameters
    ----------
    cap_mismatch_sigma : float
        Capacitor mismatch standard deviation (relative).
    comparator_offset : float
        Comparator offset in Volts.
    comparator_noise : float
        Comparator noise sigma in Volts.
    sampling_cap_fF : float
        Sampling capacitance in fF (for kT/C noise).
    charge_injection_fC : float
        Charge injection per switch in fC.
    settling_tau : float
        Settling time constant in seconds.
    settling_time : float
        Available settling time in seconds.
    """
    
    def __init__(
        self,
        cap_mismatch_sigma: float = 0.005,
        comparator_offset: float = 0.5e-3,
        comparator_noise: float = 0.1e-3,
        sampling_cap_fF: float = 1000.0,
        charge_injection_fC: float = 0.1,
        settling_tau: float = 1e-9,
        settling_time: float = 5e-9,
    ):
        self.cap_mismatch_sigma = cap_mismatch_sigma
        self.comparator_offset = comparator_offset
        self.comparator_noise = comparator_noise
        self.sampling_cap_fF = sampling_cap_fF
        self.charge_injection_fC = charge_injection_fC
        self.settling_tau = settling_tau
        self.settling_time = settling_time
        
        # Pre-generate fixed mismatches (same for all conversions)
        self.cap_mismatches = None
        
    def attach(self, adc: SARADC) -> None:
        """Attach all handlers to the ADC."""
        
        # Generate fixed capacitor mismatches
        self.cap_mismatches = 1 + np.random.normal(
            0, self.cap_mismatch_sigma, adc.bits
        )
        
        # Thermal noise voltage
        self.thermal_noise_v = thermal_noise_voltage(self.sampling_cap_fF)
        
        @adc.on(SamplingEvent, priority=EventPriority.HIGH)
        def add_sampling_noise(event):
            """Add kT/C thermal noise during sampling."""
            event.voltage += np.random.normal(0, self.thermal_noise_v)
        
        @adc.on(CapacitorSwitchEvent)
        def add_cap_effects(event):
            """Add capacitor mismatch and switching effects."""
            # Capacitor mismatch
            event.capacitance_actual = (
                event.capacitance * self.cap_mismatches[event.bit_index]
            )
            
            # Charge injection (converted to voltage)
            charge_v = (self.charge_injection_fC * 1e-15) / (
                self.sampling_cap_fF * 1e-15
            )
            event.charge_injection = charge_v
            
            # Incomplete settling
            settling_ratio = 1 - np.exp(-self.settling_time / self.settling_tau)
            event.settling_error = 1 - settling_ratio
        
        @adc.on(ComparatorEvent)
        def add_comparator_effects(event):
            """Add comparator offset and noise."""
            event.offset = self.comparator_offset
            event.noise_sigma = self.comparator_noise
            
            # Check for metastability region
            effective_input = abs(event.input_voltage - event.threshold)
            metastability_region = self.comparator_noise * 3
            event.metastable = effective_input < metastability_region


# =============================================================================
# Monte Carlo Analysis
# =============================================================================

def monte_carlo_analysis(
    bits: int = 10,
    n_runs: int = 100,
    cap_mismatch_sigma: float = 0.005,
) -> dict:
    """
    Run Monte Carlo analysis varying capacitor mismatches.
    
    Returns dictionary with ENOB statistics.
    """
    enobs = []
    snrs = []
    
    for run in range(n_runs):
        # Create ADC with random mismatches
        adc = SARADC(bits=bits, vref=1.0)
        
        # Set random mismatches
        adc.set_capacitor_mismatch(sigma=cap_mismatch_sigma)
        
        # Run simulation
        adc.sim(n_samples=1024, fs=1e6, fin=10e3)
        
        enobs.append(adc.enob())
        snrs.append(adc.snr())
    
    return {
        'enob_mean': np.mean(enobs),
        'enob_std': np.std(enobs),
        'enob_min': np.min(enobs),
        'enob_max': np.max(enobs),
        'snr_mean': np.mean(snrs),
        'snr_std': np.std(snrs),
        'enobs': enobs,
        'snrs': snrs,
    }


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Non-Idealities Analysis")
    print("=" * 60)
    
    # 1. Create ideal ADC for reference
    print("\n1. Ideal ADC (reference)")
    ideal_adc = SARADC(bits=12, vref=1.0, name="Ideal")
    ideal_adc.sim(n_samples=2048, fs=1e6, fin=10e3)
    print(f"   ENOB: {ideal_adc.enob():.2f} bits")
    print(f"   SNR:  {ideal_adc.snr():.2f} dB")
    
    # 2. Create ADC with realistic non-idealities
    print("\n2. Realistic ADC (with plugin)")
    real_adc = SARADC(bits=12, vref=1.0, name="Realistic")
    
    plugin = RealisticSARPlugin(
        cap_mismatch_sigma=0.005,   # 0.5% mismatch
        comparator_offset=0.5e-3,   # 0.5mV offset
        comparator_noise=0.2e-3,    # 0.2mV noise
        sampling_cap_fF=500,        # 500fF sampling cap
    )
    plugin.attach(real_adc)
    
    real_adc.sim(n_samples=2048, fs=1e6, fin=10e3)
    print(f"   ENOB: {real_adc.enob():.2f} bits")
    print(f"   SNR:  {real_adc.snr():.2f} dB")
    print(f"   Thermal noise: {plugin.thermal_noise_v*1e6:.2f} μV rms")
    
    # 3. Monte Carlo analysis
    print("\n3. Monte Carlo Analysis (50 runs)")
    print("   Running...")
    mc_results = monte_carlo_analysis(
        bits=12, 
        n_runs=50, 
        cap_mismatch_sigma=0.005
    )
    print(f"   ENOB: {mc_results['enob_mean']:.2f} ± {mc_results['enob_std']:.2f} bits")
    print(f"   Range: [{mc_results['enob_min']:.2f}, {mc_results['enob_max']:.2f}]")
    
    # 4. Visualization
    print("\n4. Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    fig.suptitle('Non-Ideality Analysis', fontsize=14, fontweight='bold')
    
    # Spectrum comparison
    ax1 = axes[0, 0]
    f1, s1, _ = ideal_adc.spectrum(show=False)
    f2, s2, _ = real_adc.spectrum(show=False)
    ax1.plot(f1/1e3, s1, 'b-', linewidth=0.8, label='Ideal', alpha=0.7)
    ax1.plot(f2/1e3, s2, 'r-', linewidth=0.8, label='Realistic', alpha=0.7)
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('Spectrum Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Monte Carlo ENOB histogram
    ax2 = axes[0, 1]
    ax2.hist(mc_results['enobs'], bins=20, color='steelblue', 
             alpha=0.7, edgecolor='black')
    ax2.axvline(mc_results['enob_mean'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {mc_results['enob_mean']:.2f}")
    ax2.set_xlabel('ENOB (bits)')
    ax2.set_ylabel('Count')
    ax2.set_title('Monte Carlo ENOB Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Capacitor mismatch visualization
    ax3 = axes[1, 0]
    bit_indices = np.arange(real_adc.bits)
    mismatches_pct = (plugin.cap_mismatches - 1) * 100
    colors = ['green' if m > 0 else 'red' for m in mismatches_pct]
    ax3.bar(bit_indices, mismatches_pct, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xlabel('Bit Index')
    ax3.set_ylabel('Mismatch (%)')
    ax3.set_title('Capacitor Mismatches')
    ax3.grid(True, alpha=0.3)
    
    # INL/DNL comparison
    ax4 = axes[1, 1]
    # Need ramp test for proper INL
    ideal_adc.sim(signal='ramp', n_samples=4096)
    real_adc.sim(signal='ramp', n_samples=4096)
    ax4.plot(ideal_adc.inl(), 'b-', linewidth=0.8, label='Ideal', alpha=0.7)
    ax4.plot(real_adc.inl(), 'r-', linewidth=0.8, label='Realistic', alpha=0.7)
    ax4.set_xlabel('Code')
    ax4.set_ylabel('INL (LSB)')
    ax4.set_title('INL Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonideality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved to nonideality_analysis.png")
