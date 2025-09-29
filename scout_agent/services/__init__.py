"""
ScoutAgent Services Module

This module provides centralized service registration and initialization
for all ScoutAgent services including execution, logging, config, etc.
"""

# Import execution services
from .execution.prelude_service import PreludeService
from .execution.message_service import StageMessageService
from .execution.code_executor import AgentCodeExecutor

# Import existing services
from .agents.code.service import CodeExecutionService
from .agents.memory.service import MemoryService
from .config.service import ConfigService
from .logging.service import LoggingService

# Service registry imports
from scout_agent.service_registry import get_registry

def register_all_services():
    """Register all services with the service registry."""
    registry = get_registry()
    
    # Register execution services (new)
    registry.register_service_class("prelude", PreludeService)
    registry.register_service_class("stage_message", StageMessageService)
    registry.register_service_class("agent_code_executor", AgentCodeExecutor)
    
    # Register existing services
    registry.register_service_class("code_execution", CodeExecutionService)
    registry.register_service_class("memory", MemoryService)
    registry.register_service_class("config", ConfigService)
    registry.register_service_class("logging", LoggingService)

__all__ = [
    "PreludeService",
    "StageMessageService", 
    "AgentCodeExecutor",
    "CodeExecutionService",
    "MemoryService",
    "ConfigService",
    "LoggingService",
    "register_all_services"
]