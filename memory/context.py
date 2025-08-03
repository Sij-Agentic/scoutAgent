"""
Context sharing system for agent communication.

Provides shared context between agents in workflows, enabling data flow
and state synchronization across the DAG execution.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import contextmanager

from .persistence import get_memory_manager
from ..custom_logging.logger import get_logger


@dataclass
class ContextItem:
    """A single item in the shared context."""
    key: str
    value: Any
    source: str  # Which agent/node created this
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextItem":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class SharedContext:
    """Thread-safe shared context for agent communication."""
    
    def __init__(self, context_id: str):
        self.context_id = context_id
        self._data: Dict[str, ContextItem] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[callable]] = {}
        self.logger = get_logger(f"SharedContext.{context_id}")
        self.memory_manager = get_memory_manager()
    
    def set(self, key: str, value: Any, source: str, metadata: Dict[str, Any] = None) -> None:
        """Set a value in the shared context."""
        with self._lock:
            item = ContextItem(
                key=key,
                value=value,
                source=source,
                metadata=metadata or {}
            )
            
            self._data[key] = item
            self.logger.debug(f"Set context: {key} = {value} (from {source})")
            
            # Notify subscribers
            self._notify_subscribers(key, value, source)
            
            # Persist to memory
            self._persist_context()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context."""
        with self._lock:
            item = self._data.get(key)
            return item.value if item else default
    
    def get_with_metadata(self, key: str) -> Optional[ContextItem]:
        """Get a context item with metadata."""
        with self._lock:
            return self._data.get(key)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the context."""
        with self._lock:
            return key in self._data
    
    def delete(self, key: str) -> bool:
        """Delete a key from the context."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self.logger.debug(f"Deleted context: {key}")
                self._persist_context()
                return True
            return False
    
    def keys(self) -> List[str]:
        """Get all keys in the context."""
        with self._lock:
            return list(self._data.keys())
    
    def items(self) -> List[ContextItem]:
        """Get all context items."""
        with self._lock:
            return list(self._data.values())
    
    def get_by_source(self, source: str) -> Dict[str, Any]:
        """Get all values from a specific source."""
        with self._lock:
            return {
                item.key: item.value
                for item in self._data.values()
                if item.source == source
            }
    
    def subscribe(self, key: str, callback: callable) -> None:
        """Subscribe to changes for a specific key."""
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append(callback)
    
    def unsubscribe(self, key: str, callback: callable) -> None:
        """Unsubscribe from changes for a specific key."""
        with self._lock:
            if key in self._subscribers:
                try:
                    self._subscribers[key].remove(callback)
                except ValueError:
                    pass
    
    def _notify_subscribers(self, key: str, value: Any, source: str) -> None:
        """Notify subscribers of a context change."""
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                try:
                    callback(key, value, source)
                except Exception as e:
                    self.logger.error(f"Error in context subscriber: {e}")
    
    def _persist_context(self) -> None:
        """Persist context to memory store."""
        try:
            context_data = {
                item.key: item.to_dict()
                for item in self._data.values()
            }
            self.memory_manager.save_context(self.context_id, context_data)
        except Exception as e:
            self.logger.error(f"Failed to persist context: {e}")
    
    def load_from_memory(self) -> None:
        """Load context from memory store."""
        try:
            context_data = self.memory_manager.load_context(self.context_id)
            if context_data:
                with self._lock:
                    self._data.clear()
                    for key, item_data in context_data.items():
                        self._data[key] = ContextItem.from_dict(item_data)
                self.logger.info(f"Loaded context with {len(self._data)} items")
        except Exception as e:
            self.logger.error(f"Failed to load context: {e}")
    
    def clear(self) -> None:
        """Clear all context data."""
        with self._lock:
            self._data.clear()
            self._persist_context()
            self.logger.info("Cleared all context data")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire context to dictionary."""
        with self._lock:
            return {
                "context_id": self.context_id,
                "items": {
                    key: item.to_dict()
                    for key, item in self._data.items()
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load context from dictionary."""
        with self._lock:
            self._data.clear()
            for key, item_data in data.get("items", {}).items():
                self._data[key] = ContextItem.from_dict(item_data)
            self.logger.info(f"Loaded context from dict with {len(self._data)} items")


class ContextManager:
    """Manages multiple shared contexts across workflows."""
    
    def __init__(self):
        self._contexts: Dict[str, SharedContext] = {}
        self._lock = threading.RLock()
        self.logger = get_logger("ContextManager")
    
    def create_context(self, context_id: str) -> SharedContext:
        """Create a new shared context."""
        with self._lock:
            if context_id not in self._contexts:
                self._contexts[context_id] = SharedContext(context_id)
                self.logger.info(f"Created context: {context_id}")
            return self._contexts[context_id]
    
    def get_context(self, context_id: str) -> Optional[SharedContext]:
        """Get an existing shared context."""
        with self._lock:
            return self._contexts.get(context_id)
    
    def delete_context(self, context_id: str) -> bool:
        """Delete a shared context."""
        with self._lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
                self.logger.info(f"Deleted context: {context_id}")
                return True
            return False
    
    def list_contexts(self) -> List[str]:
        """List all context IDs."""
        with self._lock:
            return list(self._contexts.keys())
    
    def get_or_create_context(self, context_id: str) -> SharedContext:
        """Get existing context or create new one."""
        context = self.get_context(context_id)
        if not context:
            context = self.create_context(context_id)
        return context
    
    def clear_all_contexts(self) -> None:
        """Clear all contexts."""
        with self._lock:
            self._contexts.clear()
            self.logger.info("Cleared all contexts")
    
    @contextmanager
    def context_scope(self, context_id: str):
        """Context manager for temporary context usage."""
        context = self.get_or_create_context(context_id)
        try:
            yield context
        finally:
            # Optionally clean up if it's a temporary context
            pass


# Global context manager
_global_context_manager = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = ContextManager()
    return _global_context_manager


class AgentContext:
    """Convenience class for agent-specific context access."""
    
    def __init__(self, agent_name: str, context_id: str):
        self.agent_name = agent_name
        self.context_id = context_id
        self.context = get_context_manager().get_or_create_context(context_id)
    
    def set(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """Set a value in the agent's context."""
        self.context.set(key, value, self.agent_name, metadata)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.context.get(key, default)
    
    def get_from_agent(self, agent_name: str, key: str, default: Any = None) -> Any:
        """Get a value from a specific agent's contributions."""
        agent_data = self.context.get_by_source(agent_name)
        return agent_data.get(key, default)
    
    def share(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """Share a value with all agents in the workflow."""
        self.set(key, value, metadata)
    
    def subscribe(self, key: str, callback: callable) -> None:
        """Subscribe to changes in a specific key."""
        self.context.subscribe(key, callback)
    
    def get_all_shared(self) -> Dict[str, Any]:
        """Get all shared context data."""
        return {
            item.key: {
                "value": item.value,
                "source": item.source,
                "timestamp": item.timestamp,
                "metadata": item.metadata
            }
            for item in self.context.items()
        }
