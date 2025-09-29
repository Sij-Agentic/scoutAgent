"""
Agent communication system for ScoutAgent.

Provides message passing protocols, event-driven workflows, and pub/sub patterns
for agent-to-agent communication within DAG workflows.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum

from ..custom_logging.logger import get_logger
from ..memory.context import get_context_manager, AgentContext


class MessageType(Enum):
    """Types of messages agents can send."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    PROGRESS = "progress"
    CHECKPOINT = "checkpoint"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """A message between agents."""
    
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    recipient: str = ""
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For request/response patterns
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to")
        )


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.logger = get_logger(f"MessageBus.{context_id}")
        self.context = get_context_manager().get_or_create_context(context_id)
        
        # Message storage
        self._messages: Dict[str, AgentMessage] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0
        }
    
    async def start(self) -> None:
        """Start the message bus processing loop."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Message bus started")
        
        # Start message processing task
        asyncio.create_task(self._process_messages())
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        self.logger.info("Message bus stopped")
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message through the bus."""
        if not self._running:
            raise RuntimeError("Message bus not running")
        
        self._messages[message.message_id] = message
        await self._message_queue.put(message)
        
        # Store in context for persistence
        self.context.set(
            f"message_{message.message_id}",
            message.to_dict(),
            message.sender,
            {"type": "message_log"}
        )
        
        self._stats["messages_sent"] += 1
        self.logger.debug(f"Message sent: {message.message_id} from {message.sender}")
    
    async def send_request(self, sender: str, recipient: str, 
                          content: Any, priority: MessagePriority = MessagePriority.NORMAL,
                          metadata: Dict[str, Any] = None) -> str:
        """Send a request message."""
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.REQUEST,
            priority=priority,
            content=content,
            metadata=metadata or {}
        )
        
        await self.send_message(message)
        return message.message_id
    
    async def send_response(self, sender: str, recipient: str,
                           content: Any, correlation_id: str,
                           metadata: Dict[str, Any] = None) -> str:
        """Send a response message."""
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.RESPONSE,
            content=content,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        await self.send_message(message)
        return message.message_id
    
    async def broadcast(self, sender: str, content: Any,
                       message_type: MessageType = MessageType.NOTIFICATION,
                       metadata: Dict[str, Any] = None) -> str:
        """Broadcast a message to all agents."""
        message = AgentMessage(
            sender=sender,
            recipient="*",  # Broadcast indicator
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        await self.send_message(message)
        return message.message_id
    
    def subscribe(self, agent_name: str, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to messages for a specific agent."""
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = []
        self._subscribers[agent_name].append(callback)
        self.logger.debug(f"Agent {agent_name} subscribed to messages")
    
    def unsubscribe(self, agent_name: str, callback: Callable[[AgentMessage], None]) -> None:
        """Unsubscribe from messages."""
        if agent_name in self._subscribers:
            try:
                self._subscribers[agent_name].remove(callback)
            except ValueError:
                pass
    
    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_message(message)
                self._stats["messages_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self._stats["errors"] += 1
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle a single message."""
        self._stats["messages_received"] += 1
        
        # Handle broadcast messages
        if message.recipient == "*":
            for agent_name, callbacks in self._subscribers.items():
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in broadcast callback for {agent_name}: {e}")
            return
        
        # Handle directed messages
        if message.recipient in self._subscribers:
            for callback in self._subscribers[message.recipient]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    self.logger.error(f"Error in callback for {message.recipient}: {e}")
    
    def get_message(self, message_id: str) -> Optional[AgentMessage]:
        """Get a specific message by ID."""
        return self._messages.get(message_id)
    
    def get_messages_for_agent(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages for a specific agent."""
        return [
            msg for msg in self._messages.values()
            if msg.recipient == agent_name or msg.recipient == "*"
        ]
    
    def get_messages_from_agent(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages from a specific agent."""
        return [
            msg for msg in self._messages.values()
            if msg.sender == agent_name
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """Get message bus statistics."""
        return self._stats.copy()
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self.logger.info("Cleared all messages")


class AgentMessenger:
    """Convenience class for agent messaging."""
    
    def __init__(self, agent_name: str, context_id: str):
        self.agent_name = agent_name
        self.context_id = context_id
        self.message_bus = None
        self.context = AgentContext(agent_name, context_id)
        self.logger = get_logger(f"AgentMessenger.{agent_name}")
    
    async def initialize(self) -> None:
        """Initialize the messenger."""
        from .base import AgentRegistry
        registry = AgentRegistry()
        
        # Get or create message bus for this context
        if self.context_id not in registry._message_buses:
            registry._message_buses[self.context_id] = MessageBus(self.context_id)
        
        self.message_bus = registry._message_buses[self.context_id]
        await self.message_bus.start()
    
    async def send_request(self, recipient: str, content: Any,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          metadata: Dict[str, Any] = None) -> str:
        """Send a request to another agent."""
        if not self.message_bus:
            raise RuntimeError("Messenger not initialized")
        
        return await self.message_bus.send_request(
            self.agent_name, recipient, content, priority, metadata
        )
    
    async def send_response(self, recipient: str, content: Any,
                           correlation_id: str,
                           metadata: Dict[str, Any] = None) -> str:
        """Send a response to another agent."""
        if not self.message_bus:
            raise RuntimeError("Messenger not initialized")
        
        return await self.message_bus.send_response(
            self.agent_name, recipient, content, correlation_id, metadata
        )
    
    async def broadcast(self, content: Any,
                       message_type: MessageType = MessageType.NOTIFICATION,
                       metadata: Dict[str, Any] = None) -> str:
        """Broadcast a message to all agents."""
        if not self.message_bus:
            raise RuntimeError("Messenger not initialized")
        
        return await self.message_bus.broadcast(
            self.agent_name, content, message_type, metadata
        )
    
    def subscribe(self, callback: Callable[[AgentMessage], None]) -> None:
        """Subscribe to incoming messages."""
        if not self.message_bus:
            raise RuntimeError("Messenger not initialized")
        
        self.message_bus.subscribe(self.agent_name, callback)
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """Get shared context data."""
        return self.context.get(key, default)
    
    def set_context_data(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """Set shared context data."""
        self.context.set(key, value, metadata)
    
    async def shutdown(self) -> None:
        """Shutdown the messenger."""
        if self.message_bus:
            await self.message_bus.stop()


class EventBus:
    """Event-driven communication system."""
    
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.logger = get_logger(f"EventBus.{context_id}")
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._running = False
    
    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event."""
        if event_type not in self._event_handlers:
            return
        
        for handler in self._event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")


# Global event bus registry
_global_event_buses = {}


def get_event_bus(context_id: str) -> EventBus:
    """Get or create an event bus for a context."""
    if context_id not in _global_event_buses:
        _global_event_buses[context_id] = EventBus(context_id)
    return _global_event_buses[context_id]
