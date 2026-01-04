"""
Tests for Pipeline ADC implementation.
"""

import pytest
import numpy as np
from quantiamagica import (
    PipelineADC,
    SARADC,
    InterstageGainEvent,
    StageEvent,
    ResidueEvent,
    EventPriority,
)


class TestPipelineADCBasic:
    """Basic functionality tests."""
    
    def test_creation(self):
        """Test pipeline creation."""
        pipeline = PipelineADC(bits=12, stages=4)
        assert pipeline.bits == 12
        assert pipeline.num_stages == 4
    
    def test_stage_info(self):
        """Test stage information retrieval."""
        pipeline = PipelineADC(bits=12, stages=4, bits_per_stage=3)
        info = pipeline.get_stage_info()
        
        assert len(info) == 4
        assert all('bits' in s for s in info)
        assert all('gain' in s for s in info)
    
    def test_single_conversion(self):
        """Test single voltage conversion."""
        pipeline = PipelineADC(bits=10, stages=3)
        result = pipeline.sim(input_voltage=0.5)
        
        expected = 512
        assert abs(result.output_codes[0] - expected) <= 5


class TestPipelineADCSimulation:
    """Simulation tests."""
    
    def test_sine_simulation(self):
        """Test simulation with sine wave."""
        pipeline = PipelineADC(bits=12, stages=4)
        result = pipeline.sim(n_samples=1024, fs=10e6, fin=100e3)
        
        assert len(result.output_codes) == 1024
    
    def test_enob(self):
        """Test ENOB calculation."""
        pipeline = PipelineADC(bits=12, stages=4)
        pipeline.sim(n_samples=2048)
        
        enob = pipeline.enob()
        assert enob > 10.0


class TestPipelineADCEvents:
    """Event system tests."""
    
    def test_stage_event(self):
        """Test stage event firing."""
        pipeline = PipelineADC(bits=10, stages=3)
        
        stage_count = [0]
        
        @pipeline.on(StageEvent)
        def handler(event):
            stage_count[0] += 1
        
        pipeline.sim(input_voltage=0.5)
        
        assert stage_count[0] == 3
    
    def test_interstage_gain_event(self):
        """Test interstage gain event."""
        pipeline = PipelineADC(bits=12, stages=4)
        
        @pipeline.on(InterstageGainEvent)
        def handler(event):
            event.actual_gain = event.ideal_gain * 0.99  # 1% error
        
        result = pipeline.sim(n_samples=1024)
        # Should still work with gain error
        assert len(result.output_codes) == 1024


class TestPipelineFromStages:
    """Tests for custom stage creation."""
    
    def test_from_sar_stages(self):
        """Test creating pipeline from SAR stages."""
        stage1 = SARADC(bits=4, vref=1.0)
        stage2 = SARADC(bits=4, vref=1.0)
        stage3 = SARADC(bits=4, vref=1.0)
        
        pipeline = PipelineADC.from_stages([stage1, stage2, stage3], gain=4.0)
        
        assert pipeline.num_stages == 3
        assert pipeline.bits == 12
    
    def test_custom_pipeline_simulation(self):
        """Test custom pipeline simulation."""
        stage1 = SARADC(bits=4, vref=1.0)
        stage2 = SARADC(bits=6, vref=1.0)
        
        pipeline = PipelineADC.from_stages([stage1, stage2], gain=4.0)
        result = pipeline.sim(n_samples=512)
        
        assert len(result.output_codes) == 512
