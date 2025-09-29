"""
Test script for agent services integration.

This script demonstrates the initialization and usage of the CodeExecutionService
and MemoryService within the service registry architecture.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from service_registry import (
    ServiceRegistry, get_registry, service, requires, ServiceBase
)
from services.config import ConfigService, get_config_service
from services.logging import LoggingService, get_logging_service
from services.agents.code import CodeExecutionService, get_code_execution_service
from services.agents.memory import MemoryService, get_memory_service
from services.agents.code.service import ExecutionEnvironment


@service(name="test_agent_service")
class TestAgentService(ServiceBase):
    """Test service to demonstrate agent services integration."""
    
    def __init__(self):
        """Initialize the test service."""
        super().__init__(name="test_agent_service", version="1.0.0")
        # Initialize logger first to avoid NoneType errors
        self.logger = logging.getLogger("test_agent_service")
        self.config = None
        self.code_service = None
        self.memory_service = None
        
    def declare_dependencies(self):
        """Declare service dependencies manually."""
        self.declare_dependency("config", required=True)
        self.declare_dependency("logging", required=True)
        self.declare_dependency("code_execution", required=True)
        self.declare_dependency("memory", required=True)
        
    async def _initialize(self, registry) -> bool:
        """Initialize with dependencies."""
        # Get dependencies
        self.config = registry.get_service("config")
        self.logger = registry.get_service("logging").get_logger("test_agent_service")
        self.code_service = registry.get_service("code_execution")
        self.memory_service = registry.get_service("memory")
        
        self.logger.info("Test agent service initialized")
        return True
    
    async def _start(self) -> bool:
        """Start the test service."""
        self.logger.info("Test agent service starting")
        return True
        
    async def _stop(self) -> bool:
        """Stop the test service."""
        self.logger.info("Test agent service stopping")
        return True
    
    async def run_test_sequence(self):
        """Run a test sequence to demonstrate agent services."""
        self.logger.info("Starting agent services test sequence")
        
        # Test Memory Service
        await self._test_memory_service()
        
        # Test Code Execution Service
        await self._test_code_execution()
        
        # Test integration between services
        await self._test_service_integration()
        
        self.logger.info("Agent services test sequence completed")
    
    async def _test_memory_service(self):
        """Test memory service operations."""
        self.logger.info("Testing memory service...")
        
        # Create a test memory
        memory_id = await self.memory_service.create_memory(
            content="This is a test memory for the agent service",
            memory_type="text",
            metadata={"source": "test_script"},
            tags=["test", "agent", "memory"]
        )
        
        self.logger.info(f"Created memory with ID: {memory_id}")
        
        # Retrieve the memory
        memory = await self.memory_service.get_memory(memory_id)
        self.logger.info(f"Retrieved memory: {memory['content']}")
        
        # Update the memory
        await self.memory_service.update_memory(
            memory_id,
            {"content": "Updated test memory content"}
        )
        
        # Search for memories
        results = await self.memory_service.search_memories("test memory")
        self.logger.info(f"Search found {len(results)} results")
        
        # List memories with tags
        memories = await self.memory_service.list_memories(
            tags=["test"]
        )
        self.logger.info(f"Found {len(memories)} memories with 'test' tag")
        
        # Test DAG task creation
        task = await self.memory_service.create_dag_task(
            operation="search",
            params={"query": "test", "limit": 5}
        )
        self.logger.info(f"Created memory DAG task: {task['id']}")
    
    async def _test_code_execution(self):
        """Test code execution service operations."""
        self.logger.info("Testing code execution service...")
        
        # Test Python code execution
        python_code = """
print("Hello from Python test code!")
result = 42
print(f"The answer is {result}")
"""
        
        self.logger.info("Executing Python code...")
        result = await self.code_service.execute_code(
            code=python_code,
            language="python",
            env=ExecutionEnvironment.LOCAL,
            timeout=5
        )
        
        self.logger.info(f"Execution success: {result.success}")
        self.logger.info(f"Execution output: {result.output}")
        
        # Test code generation
        self.logger.info("Testing code generation...")
        success, generated_code = await self.code_service.generate_code(
            prompt="Write a function to calculate factorial",
            language="python"
        )
        
        if success:
            self.logger.info("Generated code sample:")
            self.logger.info(generated_code[:100] + "..." if len(generated_code) > 100 else generated_code)
            
            # Validate the generated code
            valid, issues = self.code_service.validate_code(
                code=generated_code,
                language="python"
            )
            self.logger.info(f"Code validation result: {valid}")
            
        # Test DAG task creation
        task = await self.code_service.create_dag_task(
            code="print('Hello from DAG task')",
            language="python"
        )
        self.logger.info(f"Created code execution DAG task: {task['id']}")
    
    async def _test_service_integration(self):
        """Test integration between code and memory services."""
        self.logger.info("Testing service integration...")
        
        # Generate code and store in memory
        success, generated_code = await self.code_service.generate_code(
            prompt="Write a function to check if a number is prime",
            language="python"
        )
        
        if success:
            # Store the generated code in memory
            memory_id = await self.memory_service.create_memory(
                content=generated_code,
                memory_type="code",
                metadata={
                    "language": "python",
                    "description": "Prime number checker function",
                    "generated": True
                },
                tags=["code", "python", "prime", "generated"]
            )
            
            self.logger.info(f"Stored generated code in memory with ID: {memory_id}")
            
            # Execute the stored code
            memory = await self.memory_service.get_memory(memory_id)
            if memory:
                result = await self.code_service.execute_code(
                    code=memory["content"],
                    language="python"
                )
                
                self.logger.info(f"Executed code from memory, success: {result.success}")
                
                # Store execution result back in memory
                await self.memory_service.update_memory(
                    memory_id,
                    {"metadata": {"execution_result": result.to_dict()}}
                )
                
                self.logger.info("Updated memory with execution results")


async def main():
    """Run the agent services test."""
    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("Starting agent services test...")
    
    try:
        # Set up registry
        registry = get_registry()
        
        # Get lifecycle manager
        from service_registry.lifecycle import get_lifecycle_manager
        lifecycle = get_lifecycle_manager()
        
        # Register and create all services first
        config_service = ConfigService()
        logging_service = LoggingService()
        code_service = CodeExecutionService()
        memory_service = MemoryService()
        
        # Manually initialize these services first
        print("Manually initializing core services...")
        await config_service.initialize(registry)
        await logging_service.initialize(registry)
        await code_service.initialize(registry)
        await memory_service.initialize(registry)
        
        # Start the core services
        print("Starting core services...")
        await config_service.start()
        await logging_service.start()
        await code_service.start()
        await memory_service.start()
        
        # Now create and initialize test service last
        test_service = TestAgentService()
        
        # Manually set dependencies since we're not using the lifecycle manager
        print("Setting up test service dependencies...")
        test_service.config = registry.get_service("config")
        test_service.logger = registry.get_service("logging").get_logger("test_agent_service")
        test_service.code_service = registry.get_service("code_execution")
        test_service.memory_service = registry.get_service("memory")
        
        print("Running test sequence...")
        await test_service.run_test_sequence()
        
        # Shutdown
        print("Test complete, shutting down services...")
        await lifecycle.shutdown_all()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
