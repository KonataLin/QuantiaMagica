"""
Tests for SAR ADC implementation.
"""

import pytest
import numpy as np
from quantiamagica import (
    SARADC,
    SamplingEvent,
    CapacitorSwitchEvent,
    ComparatorEvent,
    BitDecisionEvent,
    OutputCodeEvent,
    EventPriority,
)


class TestSARADCBasic:
    """Basic functionality tests."""
    
    def test_creation(self):
        """Test ADC creation with default parameters."""
        adc = SARADC(bits=10)
        assert adc.bits == 10
        assert adc.vref == 1.0
        assert adc.vmin == 0.0
        assert adc.max_code == 1023
    
    def test_lsb_calculation(self):
        """Test LSB voltage calculation."""
        adc = SARADC(bits=10, vref=1.0, vmin=0.0)
        expected_lsb = 1.0 / 1024
        assert abs(adc.lsb - expected_lsb) < 1e-10
    
    def test_single_conversion(self):
        """Test single voltage conversion."""
        adc = SARADC(bits=10, vref=1.0)
        result = adc.sim(input_voltage=0.5)
        
        # Expected code for 0.5V with 10-bit, 1V reference
        expected = 512
        assert abs(result.output_codes[0] - expected) <= 1
    
    def test_full_scale(self):
        """Test conversion at full scale."""
        adc = SARADC(bits=10, vref=1.0)
        
        # Near zero
        result = adc.sim(input_voltage=0.001)
        assert result.output_codes[0] >= 0
        assert result.output_codes[0] <= 10
        
        # Near full scale
        result = adc.sim(input_voltage=0.999)
        assert result.output_codes[0] >= 1013
        assert result.output_codes[0] <= 1023


class TestSARADCSimulation:
    """Simulation tests."""
    
    def test_sine_simulation(self):
        """Test simulation with sine wave."""
        adc = SARADC(bits=12, vref=1.0)
        result = adc.sim(n_samples=1024, fs=1e6, fin=10e3)
        
        assert len(result.output_codes) == 1024
        assert len(result.input_signal) == 1024
        assert len(result.timestamps) == 1024
        assert len(result.reconstructed) == 1024
    
    def test_ramp_simulation(self):
        """Test simulation with ramp signal."""
        adc = SARADC(bits=10, vref=1.0)
        result = adc.sim(signal='ramp', n_samples=1024)
        
        # Output should be monotonically increasing (mostly)
        diffs = np.diff(result.output_codes)
        assert np.sum(diffs >= 0) > 0.9 * len(diffs)
    
    def test_custom_input(self):
        """Test simulation with custom input array."""
        adc = SARADC(bits=10, vref=1.0)
        custom_signal = np.linspace(0.1, 0.9, 100)
        result = adc.sim(custom_signal)
        
        assert len(result.output_codes) == 100


class TestSARADCEvents:
    """Event system tests."""
    
    def test_sampling_event(self):
        """Test sampling event is fired and modifiable."""
        adc = SARADC(bits=10, vref=1.0)
        
        event_fired = [False]
        
        @adc.on(SamplingEvent)
        def handler(event):
            event_fired[0] = True
            event.voltage += 0.1  # Add offset
        
        result = adc.sim(input_voltage=0.4)
        
        assert event_fired[0], "SamplingEvent was not fired"
        # Should convert 0.4 + 0.1 = 0.5V
        expected = 512
        assert abs(result.output_codes[0] - expected) <= 2
    
    def test_capacitor_switch_event(self):
        """Test capacitor switch event."""
        adc = SARADC(bits=10, vref=1.0)
        
        switch_count = [0]
        
        @adc.on(CapacitorSwitchEvent)
        def handler(event):
            switch_count[0] += 1
        
        adc.sim(input_voltage=0.5)
        
        # Should fire once per bit
        assert switch_count[0] == 10
    
    def test_comparator_event(self):
        """Test comparator event."""
        adc = SARADC(bits=10, vref=1.0)
        
        @adc.on(ComparatorEvent)
        def handler(event):
            event.offset = 0.01  # Add 10mV offset
        
        # This should shift the result
        result = adc.sim(input_voltage=0.5)
        # With positive offset, output should be slightly lower
        assert result.output_codes[0] < 520
    
    def test_output_code_event(self):
        """Test output code event."""
        adc = SARADC(bits=10, vref=1.0)
        
        @adc.on(OutputCodeEvent)
        def handler(event):
            event.code = 999  # Force specific code
        
        result = adc.sim(input_voltage=0.5)
        assert result.output_codes[0] == 999
    
    def test_event_priority(self):
        """Test event handler priority ordering."""
        adc = SARADC(bits=10, vref=1.0)
        
        order = []
        
        @adc.on(SamplingEvent, priority=EventPriority.LOW)
        def low_handler(event):
            order.append('low')
        
        @adc.on(SamplingEvent, priority=EventPriority.HIGH)
        def high_handler(event):
            order.append('high')
        
        @adc.on(SamplingEvent, priority=EventPriority.NORMAL)
        def normal_handler(event):
            order.append('normal')
        
        adc.sim(input_voltage=0.5)
        
        # First sample should have handlers in priority order
        assert order[:3] == ['low', 'normal', 'high']
    
    def test_cancellable_event(self):
        """Test event cancellation."""
        adc = SARADC(bits=10, vref=1.0)
        
        @adc.on(CapacitorSwitchEvent, priority=EventPriority.HIGHEST)
        def cancel_handler(event):
            if event.bit_index == 5:
                event.cancel()
        
        # Should still work but bit 5 behavior may differ
        result = adc.sim(input_voltage=0.5)
        assert result.output_codes[0] >= 0


class TestSARADCMetrics:
    """Metric calculation tests."""
    
    def test_enob_ideal(self):
        """Test ENOB for ideal ADC."""
        adc = SARADC(bits=12, vref=1.0)
        adc.sim(n_samples=4096, fs=1e6, fin=10e3)
        
        enob = adc.enob()
        # Ideal should be close to nominal
        assert enob > 11.0
    
    def test_snr_positive(self):
        """Test SNR is positive for valid signal."""
        adc = SARADC(bits=10, vref=1.0)
        adc.sim(n_samples=1024)
        
        snr = adc.snr()
        assert snr > 0
    
    def test_inl_dnl(self):
        """Test INL/DNL calculation."""
        adc = SARADC(bits=8, vref=1.0)
        adc.sim(signal='ramp', n_samples=4096)
        
        inl = adc.inl()
        dnl = adc.dnl()
        
        assert len(inl) == 256
        assert len(dnl) == 256
        # For ideal ADC, should be small
        assert np.max(np.abs(inl)) < 2.0
        assert np.max(np.abs(dnl)) < 2.0


class TestSARADCCapMismatch:
    """Capacitor mismatch tests."""
    
    def test_set_mismatch_direct(self):
        """Test setting mismatch directly."""
        adc = SARADC(bits=4, vref=1.0)
        mismatches = np.array([1.01, 0.99, 1.02, 0.98])
        adc.set_capacitor_mismatch(mismatches=mismatches)
        
        np.testing.assert_array_almost_equal(adc.cap_mismatches, mismatches)
    
    def test_set_mismatch_random(self):
        """Test setting random mismatch with sigma."""
        adc = SARADC(bits=10, vref=1.0)
        adc.set_capacitor_mismatch(sigma=0.01)
        
        # Should have some variation
        assert not np.allclose(adc.cap_mismatches, 1.0)
    
    def test_radix_setting(self):
        """Test non-binary radix setting."""
        adc = SARADC(bits=10, vref=1.0)
        adc.set_radix(1.85)
        
        # Weights should sum to 1
        assert abs(np.sum(adc.weights) - 1.0) < 1e-10


class TestSARADCDataExport:
    """Data export tests."""
    
    def test_result_to_dict(self):
        """Test result to dictionary conversion."""
        adc = SARADC(bits=10, vref=1.0)
        result = adc.sim(n_samples=100)
        
        data = result.to_dict()
        
        assert 'input_signal' in data
        assert 'output_codes' in data
        assert 'timestamps' in data
        assert 'reconstructed' in data
        assert len(data['output_codes']) == 100
