"""
Tests for analysis module.
"""

import pytest
import numpy as np
from quantiamagica import SARADC
from quantiamagica.analysis import (
    Analyzer,
    spectrum,
    enob,
    snr,
    sfdr,
    thd,
    inl,
    dnl,
)


class TestAnalyzer:
    """Analyzer class tests."""
    
    def test_analyzer_creation(self):
        """Test analyzer creation from result."""
        adc = SARADC(bits=10)
        result = adc.sim(n_samples=1024)
        
        analyzer = Analyzer(result)
        assert analyzer.bits == 10
    
    def test_analyzer_metrics(self):
        """Test analyzer metric properties."""
        adc = SARADC(bits=12)
        result = adc.sim(n_samples=2048)
        
        analyzer = Analyzer(result)
        
        # All metrics should be accessible
        assert isinstance(analyzer.enob, float)
        assert isinstance(analyzer.snr, float)
        assert isinstance(analyzer.sfdr, float)
        assert isinstance(analyzer.thd, float)
        assert isinstance(analyzer.sinad, float)
    
    def test_analyzer_summary(self):
        """Test summary generation."""
        adc = SARADC(bits=10)
        result = adc.sim()
        
        analyzer = Analyzer(result)
        summary = analyzer.summary()
        
        assert 'ENOB' in summary
        assert 'SNR' in summary
    
    def test_analyzer_to_dict(self):
        """Test dictionary export."""
        adc = SARADC(bits=10)
        result = adc.sim()
        
        analyzer = Analyzer(result)
        data = analyzer.to_dict()
        
        assert 'enob' in data
        assert 'snr' in data
        assert 'sfdr' in data


class TestStandaloneFunctions:
    """Standalone analysis function tests."""
    
    def test_spectrum_function(self):
        """Test standalone spectrum function."""
        adc = SARADC(bits=10)
        result = adc.sim(n_samples=1024)
        
        freqs, power_db, metrics = spectrum(
            result.output_codes,
            bits=10,
            fs=1e6
        )
        
        assert len(freqs) > 0
        assert len(power_db) == len(freqs)
        assert 'snr' in metrics
        assert 'enob' in metrics
    
    def test_enob_function(self):
        """Test standalone ENOB function."""
        adc = SARADC(bits=12)
        result = adc.sim(n_samples=2048)
        
        enob_val = enob(result.output_codes, bits=12)
        
        assert enob_val > 10.0
        assert enob_val <= 12.0
    
    def test_snr_function(self):
        """Test standalone SNR function."""
        adc = SARADC(bits=10)
        result = adc.sim(n_samples=1024)
        
        snr_val = snr(result.output_codes, bits=10)
        
        assert snr_val > 0
    
    def test_inl_dnl_functions(self):
        """Test INL and DNL functions."""
        adc = SARADC(bits=8)
        result = adc.sim(signal='ramp', n_samples=2048)
        
        inl_vals = inl(result.output_codes, bits=8)
        dnl_vals = dnl(result.output_codes, bits=8)
        
        assert len(inl_vals) == 256
        assert len(dnl_vals) == 256
