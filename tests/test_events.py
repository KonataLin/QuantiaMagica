"""
Tests for the event system.
"""

import pytest
from quantiamagica.core.events import (
    Event,
    EventBus,
    EventPriority,
    Cancellable,
    on_event,
)
from dataclasses import dataclass


@dataclass
class TestEvent(Event):
    """Test event for unit tests."""
    value: float = 0.0


@dataclass
class CancellableTestEvent(Event, Cancellable):
    """Cancellable test event."""
    value: float = 0.0
    
    def __post_init__(self):
        Cancellable.__init__(self)


class TestEventBus:
    """EventBus tests."""
    
    def test_register_and_fire(self):
        """Test basic event registration and firing."""
        bus = EventBus()
        received = []
        
        @bus.on(TestEvent)
        def handler(event):
            received.append(event.value)
        
        bus.fire(TestEvent(value=42.0))
        
        assert received == [42.0]
    
    def test_multiple_handlers(self):
        """Test multiple handlers for same event."""
        bus = EventBus()
        results = []
        
        @bus.on(TestEvent)
        def handler1(event):
            results.append('h1')
        
        @bus.on(TestEvent)
        def handler2(event):
            results.append('h2')
        
        bus.fire(TestEvent())
        
        assert 'h1' in results
        assert 'h2' in results
    
    def test_priority_ordering(self):
        """Test handlers execute in priority order."""
        bus = EventBus()
        order = []
        
        @bus.on(TestEvent, priority=EventPriority.LOW)
        def low(event):
            order.append('low')
        
        @bus.on(TestEvent, priority=EventPriority.HIGH)
        def high(event):
            order.append('high')
        
        @bus.on(TestEvent, priority=EventPriority.NORMAL)
        def normal(event):
            order.append('normal')
        
        bus.fire(TestEvent())
        
        assert order == ['low', 'normal', 'high']
    
    def test_unregister(self):
        """Test handler unregistration."""
        bus = EventBus()
        count = [0]
        
        def handler(event):
            count[0] += 1
        
        bus.register(TestEvent, handler)
        bus.fire(TestEvent())
        assert count[0] == 1
        
        bus.unregister(TestEvent, handler)
        bus.fire(TestEvent())
        assert count[0] == 1  # Should not increase


class TestCancellable:
    """Cancellable event tests."""
    
    def test_cancel_event(self):
        """Test event cancellation."""
        event = CancellableTestEvent()
        assert not event.cancelled
        
        event.cancel()
        assert event.cancelled
        
        event.uncancel()
        assert not event.cancelled
    
    def test_cancelled_handlers_skipped(self):
        """Test that cancelled events skip non-ignoring handlers."""
        bus = EventBus()
        results = []
        
        @bus.on(CancellableTestEvent, priority=EventPriority.LOW)
        def first(event):
            results.append('first')
            event.cancel()
        
        @bus.on(CancellableTestEvent, priority=EventPriority.NORMAL)
        def second(event):
            results.append('second')
        
        @bus.on(CancellableTestEvent, priority=EventPriority.HIGH, ignore_cancelled=True)
        def third(event):
            results.append('third')
        
        bus.fire(CancellableTestEvent())
        
        assert 'first' in results
        assert 'second' not in results  # Skipped due to cancellation
        assert 'third' in results  # Runs because ignore_cancelled=True


class TestEventModification:
    """Event modification tests."""
    
    def test_modify_event_value(self):
        """Test that handlers can modify event values."""
        bus = EventBus()
        
        @bus.on(TestEvent)
        def handler(event):
            event.value *= 2
        
        event = TestEvent(value=10.0)
        bus.fire(event)
        
        assert event.value == 20.0
    
    def test_chained_modifications(self):
        """Test chained modifications through priority."""
        bus = EventBus()
        
        @bus.on(TestEvent, priority=EventPriority.LOW)
        def add(event):
            event.value += 10
        
        @bus.on(TestEvent, priority=EventPriority.HIGH)
        def multiply(event):
            event.value *= 2
        
        event = TestEvent(value=5.0)
        bus.fire(event)
        
        # Order: add first (LOW), then multiply (HIGH)
        # (5 + 10) * 2 = 30
        assert event.value == 30.0


class TestOnEventDecorator:
    """Tests for @on_event decorator."""
    
    def test_decorator_marks_handler(self):
        """Test that decorator adds metadata."""
        
        @on_event(TestEvent, priority=EventPriority.HIGH)
        def handler(event):
            pass
        
        assert hasattr(handler, '_event_handler')
        assert handler._event_handler is True
        assert handler._event_type == TestEvent
        assert handler._event_priority == EventPriority.HIGH


class TestEventLogging:
    """Event logging tests."""
    
    def test_logging_enabled(self):
        """Test event logging when enabled."""
        bus = EventBus()
        bus.enable_logging(True)
        
        bus.fire(TestEvent(value=1.0))
        bus.fire(TestEvent(value=2.0))
        
        log = bus.get_log()
        assert len(log) == 2
        assert log[0].value == 1.0
        assert log[1].value == 2.0
    
    def test_logging_disabled(self):
        """Test no logging when disabled."""
        bus = EventBus()
        bus.enable_logging(False)
        
        bus.fire(TestEvent(value=1.0))
        
        log = bus.get_log()
        assert len(log) == 0
    
    def test_clear_log(self):
        """Test log clearing."""
        bus = EventBus()
        bus.enable_logging(True)
        
        bus.fire(TestEvent())
        bus.clear_log()
        
        assert len(bus.get_log()) == 0
