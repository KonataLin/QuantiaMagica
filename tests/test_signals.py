"""
Tests for Signal class.
"""

import pytest
import numpy as np
from quantiamagica.signals import Signal


class TestSignalCreation:
    """Signal creation tests."""
    
    def test_sine_creation(self):
        sig = Signal.sine(n=1024, fs=1e6, freq=10e3)
        assert len(sig) == 1024
        assert sig.fs == 1e6
    
    def test_square_creation(self):
        sig = Signal.square(n=1024, freq=1e3)
        assert len(sig) == 1024
    
    def test_triangle_creation(self):
        sig = Signal.triangle(n=1024, freq=1e3)
        assert len(sig) == 1024
    
    def test_sawtooth_creation(self):
        sig = Signal.sawtooth(n=1024, freq=1e3)
        assert len(sig) == 1024
    
    def test_ramp_creation(self):
        sig = Signal.ramp(n=4096, vmin=0.0, vmax=1.0)
        assert len(sig) == 4096
        assert sig.vmin == pytest.approx(0.0, abs=0.01)
        assert sig.vmax == pytest.approx(1.0, abs=0.01)
    
    def test_dc_creation(self):
        sig = Signal.dc(n=100, voltage=0.5)
        assert len(sig) == 100
        assert np.allclose(sig.data, 0.5)
    
    def test_step_creation(self):
        sig = Signal.step(n=1024, v_low=0.2, v_high=0.8)
        assert len(sig) == 1024
    
    def test_pulse_creation(self):
        sig = Signal.pulse(n=1024)
        assert len(sig) == 1024


class TestSignalMultitone:
    """Multitone signal tests."""
    
    def test_two_tone(self):
        sig = Signal.two_tone(n=2048, f1=10e3, f2=11e3)
        assert len(sig) == 2048
    
    def test_multitone(self):
        sig = Signal.multitone(
            n=2048,
            frequencies=[1e3, 2e3, 5e3],
            amplitudes=[0.1, 0.1, 0.1],
        )
        assert len(sig) == 2048


class TestSignalModulation:
    """Modulation signal tests."""
    
    def test_chirp(self):
        sig = Signal.chirp(n=2048, f_start=1e3, f_end=100e3)
        assert len(sig) == 2048
    
    def test_am(self):
        sig = Signal.am(n=2048, carrier_freq=100e3, mod_freq=1e3)
        assert len(sig) == 2048
    
    def test_fm(self):
        sig = Signal.fm(n=2048, carrier_freq=100e3, mod_freq=1e3)
        assert len(sig) == 2048


class TestSignalNoise:
    """Noise signal tests."""
    
    def test_gaussian_noise(self):
        sig = Signal.noise_gaussian(n=10000, mean=0.5, sigma=0.1)
        assert len(sig) == 10000
        assert sig.mean == pytest.approx(0.5, abs=0.02)
    
    def test_uniform_noise(self):
        sig = Signal.noise_uniform(n=1000, vmin=0.3, vmax=0.7)
        assert len(sig) == 1000
        assert sig.vmin >= 0.3
        assert sig.vmax <= 0.7


class TestSignalProperties:
    """Signal property tests."""
    
    def test_vpp(self):
        sig = Signal.sine(n=1024, amplitude=0.4, offset=0.5)
        assert sig.vpp == pytest.approx(0.8, abs=0.01)
    
    def test_duration(self):
        sig = Signal.sine(n=1000, fs=1e6)
        assert sig.duration == pytest.approx(1e-3, abs=1e-6)
    
    def test_n_samples(self):
        sig = Signal.sine(n=512)
        assert sig.n_samples == 512


class TestSignalProcessing:
    """Signal processing tests."""
    
    def test_add_noise(self):
        sig = Signal.sine(n=1024, amplitude=0.4)
        noisy = sig.add_noise(sigma=0.01)
        assert len(noisy) == 1024
        # Should be different from original
        assert not np.allclose(sig.data, noisy.data)
    
    def test_add_offset(self):
        sig = Signal.sine(n=1024, offset=0.5)
        shifted = sig.add_offset(0.1)
        assert shifted.mean == pytest.approx(sig.mean + 0.1, abs=0.01)
    
    def test_scale(self):
        sig = Signal.sine(n=1024, amplitude=0.4)
        scaled = sig.scale(2.0)
        assert scaled.vpp == pytest.approx(sig.vpp * 2, abs=0.01)
    
    def test_clip(self):
        sig = Signal.sine(n=1024, amplitude=0.5, offset=0.5)
        clipped = sig.clip(vmin=0.3, vmax=0.7)
        assert clipped.vmin >= 0.3
        assert clipped.vmax <= 0.7
    
    def test_normalize(self):
        sig = Signal.sine(n=1024)
        normalized = sig.normalize(vmin=0.0, vmax=1.0)
        assert normalized.vmin == pytest.approx(0.0, abs=0.01)
        assert normalized.vmax == pytest.approx(1.0, abs=0.01)


class TestSignalImportExport:
    """Import/export tests."""
    
    def test_from_array(self):
        data = np.linspace(0, 1, 100)
        sig = Signal.from_array(data, fs=1e6)
        assert len(sig) == 100
        np.testing.assert_array_almost_equal(sig.data, data)
    
    def test_array_interface(self):
        sig = Signal.sine(n=100)
        arr = np.array(sig)
        assert len(arr) == 100


class TestSignalOperators:
    """Operator tests."""
    
    def test_add_float(self):
        sig = Signal.sine(n=100, offset=0.5)
        result = sig + 0.1
        assert result.mean == pytest.approx(sig.mean + 0.1, abs=0.01)
    
    def test_mul_float(self):
        sig = Signal.sine(n=100, amplitude=0.4)
        result = sig * 2.0
        assert result.vpp == pytest.approx(sig.vpp * 2, abs=0.01)
