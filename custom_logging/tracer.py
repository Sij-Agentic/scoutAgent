"""
Tracer for data flow tracking and debugging in ScoutAgent.
Provides structured tracing capabilities for agent execution and data flow.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

from .logger import get_logger


class DataTracer:
    """Tracks data flow and execution for debugging purposes."""
    
    def __init__(self, 
                 trace_dir: Optional[str] = None,
                 enable_tracing: bool = True):
        """
        Initialize the data tracer.
        
        Args:
            trace_dir: Directory for trace files (defaults to ./logs/traces)
            enable_tracing: Whether to enable tracing
        """
        self.enable_tracing = enable_tracing
        self.trace_dir = Path(trace_dir) if trace_dir else Path("./logs/traces")
        self.logger = get_logger("tracer")
        
        if self.enable_tracing:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self._trace_stack: List[Dict[str, Any]] = []
        self._current_trace_id: Optional[str] = None
    
    def start_trace(self, 
                   trace_id: str,
                   operation: str,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new trace.
        
        Args:
            trace_id: Unique identifier for the trace
            operation: Name of the operation being traced
            metadata: Additional metadata for the trace
            
        Returns:
            The trace ID
        """
        if not self.enable_tracing:
            return trace_id
            
        trace_data = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": datetime.utcnow().isoformat(),
            "timestamp": time.time(),
            "metadata": metadata or {},
            "events": [],
            "data_flow": []
        }
        
        self._trace_stack.append(trace_data)
        self._current_trace_id = trace_id
        
        self.logger.info(f"Started trace: {trace_id} - {operation}")
        return trace_id
    
    def log_event(self, 
                  event_type: str,
                  data: Any,
                  level: str = "info",
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Log an event within the current trace.
        
        Args:
            event_type: Type of event (e.g., 'agent_start', 'data_processed')
            data: Data associated with the event
            level: Log level (debug, info, warning, error)
            metadata: Additional metadata
        """
        if not self.enable_tracing or not self._trace_stack:
            return
            
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "level": level,
            "data": self._serialize_data(data),
            "metadata": metadata or {}
        }
        
        current_trace = self._trace_stack[-1]
        current_trace["events"].append(event)
        
        # Also log to regular logger
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"Trace {current_trace['trace_id']}: {event_type} - {data}")
    
    def log_data_flow(self, 
                     source: str,
                     destination: str,
                     data_type: str,
                     data_size: int,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Log data flow between components.
        
        Args:
            source: Source component
            destination: Destination component
            data_type: Type of data being transferred
            data_size: Size of data in bytes
            metadata: Additional metadata
        """
        if not self.enable_tracing or not self._trace_stack:
            return
            
        flow_entry = {
            "timestamp": time.time(),
            "source": source,
            "destination": destination,
            "data_type": data_type,
            "data_size": data_size,
            "metadata": metadata or {}
        }
        
        current_trace = self._trace_stack[-1]
        current_trace["data_flow"].append(flow_entry)
        
        self.logger.debug(f"Data flow: {source} -> {destination} ({data_type}, {data_size} bytes)")
    
    def end_trace(self, 
                  result: Any = None,
                  error: Optional[str] = None):
        """
        End the current trace and save to file.
        
        Args:
            result: Final result of the traced operation
            error: Error message if operation failed
        """
        if not self.enable_tracing or not self._trace_stack:
            return
            
        current_trace = self._trace_stack.pop()
        current_trace.update({
            "end_time": datetime.utcnow().isoformat(),
            "duration": time.time() - current_trace["timestamp"],
            "result": self._serialize_data(result),
            "error": error,
            "status": "error" if error else "success"
        })
        
        # Save trace to file
        trace_file = self.trace_dir / f"trace_{current_trace['trace_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trace_file, 'w') as f:
            json.dump(current_trace, f, indent=2, default=str)
        
        self.logger.info(f"Completed trace: {current_trace['trace_id']} ({current_trace['duration']:.2f}s)")
        
        # Update current trace ID
        if self._trace_stack:
            self._current_trace_id = self._trace_stack[-1]["trace_id"]
        else:
            self._current_trace_id = None
    
    @contextmanager
    def trace_context(self, 
                     trace_id: str,
                     operation: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing operations.
        
        Args:
            trace_id: Unique identifier for the trace
            operation: Name of the operation being traced
            metadata: Additional metadata
        """
        self.start_trace(trace_id, operation, metadata)
        try:
            yield self
        except Exception as e:
            self.end_trace(error=str(e))
            raise
        else:
            self.end_trace()
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if data is None:
            return None
            
        try:
            # Handle common data types
            if isinstance(data, (str, int, float, bool)):
                return data
            elif isinstance(data, dict):
                return {k: self._serialize_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._serialize_data(item) for item in data]
            elif isinstance(data, bytes):
                return f"<bytes: {len(data)}>"
            else:
                return str(data)
        except Exception as e:
            return f"<serialization_error: {str(e)}>"
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of a specific trace.
        
        Args:
            trace_id: The trace ID to summarize
            
        Returns:
            Trace summary or None if not found
        """
        trace_files = list(self.trace_dir.glob(f"trace_{trace_id}_*.json"))
        if not trace_files:
            return None
            
        latest_trace = max(trace_files, key=lambda x: x.stat().st_mtime)
        with open(latest_trace, 'r') as f:
            return json.load(f)
    
    def list_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent traces.
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace summaries
        """
        if not self.trace_dir.exists():
            return []
            
        trace_files = list(self.trace_dir.glob("trace_*.json"))
        trace_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        traces = []
        for trace_file in trace_files[:limit]:
            try:
                with open(trace_file, 'r') as f:
                    traces.append(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load trace file {trace_file}: {e}")
        
        return traces


# Global tracer instance
_global_tracer = None


def get_tracer(enable_tracing: bool = True) -> DataTracer:
    """
    Get the global tracer instance.
    
    Args:
        enable_tracing: Whether to enable tracing
        
    Returns:
        Configured tracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DataTracer(enable_tracing=enable_tracing)
    return _global_tracer


# Convenience functions for direct usage
def start_trace(trace_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Start a new trace."""
    return get_tracer().start_trace(trace_id, operation, metadata)


def log_event(event_type: str, data: Any, level: str = "info", metadata: Optional[Dict[str, Any]] = None):
    """Log an event within the current trace."""
    get_tracer().log_event(event_type, data, level, metadata)


def log_data_flow(source: str, destination: str, data_type: str, data_size: int, metadata: Optional[Dict[str, Any]] = None):
    """Log data flow between components."""
    get_tracer().log_data_flow(source, destination, data_type, data_size, metadata)


def end_trace(result: Any = None, error: Optional[str] = None):
    """End the current trace."""
    get_tracer().end_trace(result, error)


@contextmanager
def trace_context(trace_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations."""
    tracer = get_tracer()
    with tracer.trace_context(trace_id, operation, metadata):
        yield


if __name__ == "__main__":
    # Test the tracer
    tracer = get_tracer(enable_tracing=True)
    
    with tracer.trace_context("test_trace", "test_operation", {"test": True}):
        tracer.log_event("test_start", {"message": "Starting test"})
        tracer.log_data_flow("input", "processor", "test_data", 1024)
        tracer.log_event("processing", {"step": 1})
        tracer.log_event("processing", {"step": 2})
        tracer.log_data_flow("processor", "output", "result", 512)
    
    print("Tracer test completed. Check the logs/traces directory for trace files.")
