"""
Service Initializer - Ensures all services are properly registered with the service registry.

This module imports and registers all service classes to ensure they're available
in the service registry before they're requested.
"""

from scout_agent.service_registry import get_registry, ServiceRegistry
from scout_agent.services.execution.message_service import StageMessageService
from scout_agent.services.execution.code_executor import AgentCodeExecutor
from scout_agent.services.execution.prelude_service import PreludeService


def initialize_services() -> ServiceRegistry:
    """
    Initialize and register all services with the service registry.
    
    Returns:
        ServiceRegistry: The initialized service registry
    """
    registry = get_registry()
    
    # Create service instances
    message_service = StageMessageService()
    code_executor = AgentCodeExecutor()
    prelude_service = PreludeService()
    
    # Register services manually if needed
    if not registry.has_service_instance("stage_message"):
        registry.services["stage_message"] = {message_service.service_id: message_service}
    
    if not registry.has_service_instance("agent_code_executor"):
        registry.services["agent_code_executor"] = {code_executor.service_id: code_executor}
    
    if not registry.has_service_instance("prelude"):
        registry.services["prelude"] = {prelude_service.service_id: prelude_service}
    
    return registry
