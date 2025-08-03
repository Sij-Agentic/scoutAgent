"""
Logging module for ScoutAgent.

Provides unified logging and tracing capabilities for the ScoutAgent system.
"""

from .logger import (
    ScoutLogger,
    get_logger,
    setup_logger,
    debug,
    info,
    warning,
    error,
    critical,
    exception
)

from .tracer import (
    DataTracer,
    get_tracer,
    start_trace,
    log_event,
    log_data_flow,
    end_trace,
    trace_context
)

__all__ = [
    # Logger exports
    "ScoutLogger",
    "get_logger",
    "setup_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    
    # Tracer exports
    "DataTracer",
    "get_tracer",
    "start_trace",
    "log_event",
    "log_data_flow",
    "end_trace",
    "trace_context"
]
