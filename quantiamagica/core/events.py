"""
Event System - Bukkit-style event-driven architecture for ADC simulation.

This module provides a flexible event system that allows users to listen to
and modify ADC behavior through event handlers.

Example
-------
>>> from quantiamagica import SARADC, SamplingEvent
>>> 
>>> adc = SARADC(bits=10)
>>> 
>>> @adc.on(SamplingEvent)
... def on_sample(event):
...     event.voltage += 0.001  # Add offset
"""

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Generic,
    Union,
    Set,
)
from functools import wraps
import weakref
import bisect
import warnings


class EventPriority(IntEnum):
    """
    Priority levels for event handlers.
    
    Lower values execute first. Use MONITOR for read-only observation.
    
    Attributes
    ----------
    LOWEST : int
        Executes first, can be overridden by later handlers.
    LOW : int
        Low priority handler.
    NORMAL : int
        Default priority for most handlers.
    HIGH : int
        High priority, executes before NORMAL.
    HIGHEST : int
        Executes last before MONITOR, has final say on modifications.
    MONITOR : int
        Read-only observation, should not modify event state.
    """
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    MONITOR = 5


class Cancellable:
    """
    Mixin for events that can be cancelled.
    
    When an event is cancelled, subsequent handlers may choose to skip
    processing, and the ADC may skip the associated action.
    
    Attributes
    ----------
    cancelled : bool
        Whether the event has been cancelled.
    
    Example
    -------
    >>> @adc.on(CapacitorSwitchEvent)
    ... def skip_switch(event):
    ...     if event.bit_index == 5:
    ...         event.cancel()  # Skip switching bit 5
    """
    
    def __init__(self):
        self._cancelled: bool = False
    
    @property
    def cancelled(self) -> bool:
        """Check if event is cancelled."""
        return self._cancelled
    
    def cancel(self) -> None:
        """Cancel this event."""
        self._cancelled = True
    
    def uncancel(self) -> None:
        """Uncancel this event."""
        self._cancelled = False


@dataclass
class Event(ABC):
    """
    Base class for all events in the QuantiaMagica framework.
    
    Events carry information about ADC operations and allow handlers
    to observe or modify the simulation behavior.
    
    Attributes
    ----------
    timestamp : float
        Simulation time when the event occurred.
    source : Any
        The ADC instance that fired this event.
    
    Example
    -------
    >>> @dataclass
    ... class MyCustomEvent(Event):
    ...     my_data: float = 0.0
    """
    timestamp: float = 0.0
    source: Any = None
    
    def __post_init__(self):
        pass


E = TypeVar('E', bound=Event)


@dataclass
class HandlerInfo:
    """Information about a registered event handler."""
    callback: Callable[[Event], None]
    priority: EventPriority
    ignore_cancelled: bool
    owner: Optional[weakref.ref] = None


class EventBus:
    """
    Central event dispatcher for ADC simulation.
    
    Manages event handler registration and dispatching. Handlers are
    called in priority order when events are fired.
    
    Optimizations:
    - 缓存事件类型层次结构，避免重复isinstance检查
    - 使用bisect进行O(log n)插入
    - 预计算处理器列表
    
    Attributes
    ----------
    handlers : Dict[Type[Event], List[HandlerInfo]]
        Registered handlers organized by event type.
    
    Example
    -------
    >>> bus = EventBus()
    >>> 
    >>> @bus.on(SamplingEvent)
    ... def handler(event):
    ...     print(f"Sampled: {event.voltage}")
    >>> 
    >>> bus.fire(SamplingEvent(voltage=0.5))
    """
    
    __slots__ = ('_handlers', '_event_log', '_logging_enabled', '_type_cache', '_handler_cache_valid')
    
    def __init__(self):
        self._handlers: Dict[Type[Event], List[HandlerInfo]] = {}
        self._event_log: List[Event] = []
        self._logging_enabled: bool = False
        self._type_cache: Dict[Type[Event], List[Type[Event]]] = {}
        self._handler_cache_valid: bool = True
    
    def on(
        self,
        event_type: Type[E],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
    ) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
        """
        Decorator to register an event handler.
        
        Parameters
        ----------
        event_type : Type[Event]
            The event class to listen for.
        priority : EventPriority, optional
            Handler priority, by default NORMAL.
        ignore_cancelled : bool, optional
            If True, handler runs even for cancelled events.
        
        Returns
        -------
        Callable
            Decorator function.
        
        Example
        -------
        >>> @bus.on(SamplingEvent, priority=EventPriority.HIGH)
        ... def high_priority_handler(event):
        ...     event.voltage *= 1.001  # Apply gain error
        """
        def decorator(func: Callable[[E], None]) -> Callable[[E], None]:
            self.register(event_type, func, priority, ignore_cancelled)
            return func
        return decorator
    
    def register(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
        priority: EventPriority = EventPriority.NORMAL,
        ignore_cancelled: bool = False,
        owner: Any = None,
    ) -> None:
        """
        Register an event handler programmatically.
        
        Parameters
        ----------
        event_type : Type[Event]
            The event class to listen for.
        callback : Callable[[Event], None]
            The handler function.
        priority : EventPriority, optional
            Handler priority.
        ignore_cancelled : bool, optional
            If True, handler runs even for cancelled events.
        owner : Any, optional
            Owner object for the handler (used for cleanup).
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        owner_ref = weakref.ref(owner) if owner is not None else None
        
        info = HandlerInfo(
            callback=callback,
            priority=priority,
            ignore_cancelled=ignore_cancelled,
            owner=owner_ref,
        )
        
        # 使用bisect进行O(log n)插入，保持排序
        handlers = self._handlers[event_type]
        priorities = [h.priority for h in handlers]
        insert_pos = bisect.bisect_right(priorities, priority)
        handlers.insert(insert_pos, info)
        
        # 清除类型缓存
        self._type_cache.clear()
    
    def unregister(
        self,
        event_type: Type[Event],
        callback: Callable[[Event], None],
    ) -> bool:
        """
        Remove a registered handler.
        
        Parameters
        ----------
        event_type : Type[Event]
            The event class.
        callback : Callable
            The handler to remove.
        
        Returns
        -------
        bool
            True if handler was found and removed.
        """
        if event_type not in self._handlers:
            return False
        
        original_len = len(self._handlers[event_type])
        self._handlers[event_type] = [
            h for h in self._handlers[event_type] if h.callback != callback
        ]
        if len(self._handlers[event_type]) < original_len:
            self._type_cache.clear()
            return True
        return False
    
    def unregister_all(self, owner: Any = None) -> int:
        """
        Remove all handlers, optionally filtering by owner.
        
        Parameters
        ----------
        owner : Any, optional
            If provided, only remove handlers owned by this object.
        
        Returns
        -------
        int
            Number of handlers removed.
        """
        count = 0
        for event_type in self._handlers:
            if owner is None:
                count += len(self._handlers[event_type])
                self._handlers[event_type] = []
            else:
                original_len = len(self._handlers[event_type])
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type]
                    if h.owner is None or h.owner() != owner
                ]
                count += original_len - len(self._handlers[event_type])
        if count > 0:
            self._type_cache.clear()
        return count
    
    def _get_matching_types(self, event_type: Type[Event]) -> List[Type[Event]]:
        """
        获取所有匹配的事件类型（使用缓存）
        
        缓存事件类型的层次结构，避免重复isinstance检查。
        """
        if event_type in self._type_cache:
            return self._type_cache[event_type]
        
        # 计算所有匹配的注册事件类型
        matching = [etype for etype in self._handlers if issubclass(event_type, etype)]
        self._type_cache[event_type] = matching
        return matching
    
    def fire(self, event: Event) -> Event:
        """
        Dispatch an event to all registered handlers.
        
        Handlers are called in priority order. If the event is Cancellable
        and gets cancelled, handlers with ignore_cancelled=False will be skipped.
        
        Parameters
        ----------
        event : Event
            The event to dispatch.
        
        Returns
        -------
        Event
            The event after all handlers have processed it.
        """
        if self._logging_enabled:
            self._event_log.append(event)
        
        event_type = type(event)
        
        # 快速路径：没有处理器时直接返回
        if not self._handlers:
            return event
        
        # 使用缓存获取匹配的事件类型
        matching_types = self._get_matching_types(event_type)
        
        if not matching_types:
            return event
        
        # 预计算是否为Cancellable（避免重复isinstance检查）
        is_cancellable = isinstance(event, Cancellable)
        
        for etype in matching_types:
            handlers = self._handlers[etype]
            for handler_info in handlers:
                # 检查owner是否存活
                owner_ref = handler_info.owner
                if owner_ref is not None and owner_ref() is None:
                    continue
                
                # 检查cancelled状态
                if is_cancellable and event._cancelled and not handler_info.ignore_cancelled:
                    continue
                
                try:
                    handler_info.callback(event)
                except Exception as e:
                    warnings.warn(
                        f"Handler {handler_info.callback.__name__} raised: {e}"
                    )
        
        return event
    
    def enable_logging(self, enabled: bool = True) -> None:
        """Enable or disable event logging."""
        self._logging_enabled = enabled
        if not enabled:
            self._event_log.clear()
    
    def get_log(self) -> List[Event]:
        """Get the event log (copy)."""
        return list(self._event_log)
    
    def clear_log(self) -> None:
        """Clear the event log."""
        self._event_log.clear()


def on_event(
    event_type: Type[E],
    priority: EventPriority = EventPriority.NORMAL,
    ignore_cancelled: bool = False,
) -> Callable[[Callable[[E], None]], Callable[[E], None]]:
    """
    Decorator to mark a method as an event handler.
    
    Use this in class definitions to create handler methods that will
    be automatically registered when the class is used with an ADC.
    
    Parameters
    ----------
    event_type : Type[Event]
        The event class to listen for.
    priority : EventPriority, optional
        Handler priority, by default NORMAL.
    ignore_cancelled : bool, optional
        If True, handler runs even for cancelled events.
    
    Returns
    -------
    Callable
        Decorated method with handler metadata.
    
    Example
    -------
    >>> class MyPlugin:
    ...     @on_event(SamplingEvent, priority=EventPriority.HIGH)
    ...     def on_sample(self, event):
    ...         event.voltage += self.offset
    """
    def decorator(func: Callable[[E], None]) -> Callable[[E], None]:
        func._event_handler = True
        func._event_type = event_type
        func._event_priority = priority
        func._ignore_cancelled = ignore_cancelled
        return func
    return decorator
