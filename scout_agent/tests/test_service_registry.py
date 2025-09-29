"""
Test script for service registry functionality.

This script demonstrates how to use the service registry and
tests the config and logging services.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from service_registry import (
    get_registry, 
    get_lifecycle_manager,
    ServiceBase,
    service,
    requires,
    inject
)

from services.config import ConfigService, get_config_service
from services.logging import LoggingService, get_logging_service


@service(name="test_service")
@requires("config")
@requires("logging")
class TestService(ServiceBase):
    """Test service that depends on config and logging."""
    
    def __init__(self):
        super().__init__(name="test_service", version="1.0.0")
        # Use get_original_logger for initialization before service dependencies are ready
        from custom_logging import get_logger as get_original_logger
        self.logger = get_original_logger("test_service")
        self.config = None
    
    async def _initialize(self, registry) -> bool:
        """Initialize the test service."""
        # Get dependencies (using explicit registry)
        self.config = registry.get_service("config")
        self.logger = registry.get_service("logging").get_logger("test_service")
        
        self.logger.info("Test service initialized")
        return True
    
    async def _start(self) -> bool:
        """Start the test service."""
        self.logger.info("Test service started")
        
        # Test config access
        api_key = self.config.get_config_value("api.openai_api_key", "default_key")
        self.logger.info(f"API Key from config: {api_key}")
        
        return True
    
    async def _stop(self) -> bool:
        """Stop the test service."""
        self.logger.info("Test service stopped")
        return True
    
    @inject
    def test_injection(self, config: ConfigService = None, logging: LoggingService = None):
        """Test dependency injection."""
        injected_logger = logging.get_logger("injected")
        injected_logger.info("This log comes from an injected dependency")
        
        config_value = config.get_config_value("agent.default_backend", {})
        injected_logger.info(f"Config value from injected dependency: {config_value}")
        return True


async def main():
    """Run the service registry test."""
    print("Testing Service Registry")
    print("=======================\n")
    
    # Get registry and lifecycle manager
    registry = get_registry()
    lifecycle = get_lifecycle_manager()
    
    # Create config service with specific config file
    config_service = ConfigService(config_file="config.json")
    
    # Create logging service
    logging_service = LoggingService()
    
    # Create test service
    test_service = TestService()
    
    # Initialize services
    print("Initializing services...")
    await lifecycle.initialize_all()
    
    # Start services
    print("Starting services...")
    await lifecycle.start_all()
    
    # Get a logger from the logging service
    logger = logging_service.get_logger("test_script")
    logger.info("Test script is running with initialized services")
    
    # Test injected dependencies
    test_service.test_injection()
    
    # Test dependency resolution order
    print("\nService dependency resolution order:")
    for service_id in lifecycle.dependency_graph.get_initialization_order():
        node = lifecycle.dependency_graph.graph.nodes[service_id]
        service = node['service']
        print(f"  {service.name} ({service_id})")
    
    # Test shutdown in reverse order
    print("\nShutting down services...")
    await lifecycle.shutdown_all()
    
    print("\nTest completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
