"""
Direct test for agent services without relying on service registry.

This script tests the CodeExecutionService and MemoryService directly,
without depending on the service registry for dependency injection.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.config import ConfigService
from services.logging import LoggingService
from services.agents.code import CodeExecutionService
from services.agents.memory import MemoryService
from services.agents.code.service import ExecutionEnvironment


async def test_memory_service(memory_service, logger):
    """Test memory service operations directly."""
    logger.info("Testing memory service...")
    
    # Create a test memory
    memory_id = await memory_service.create_memory(
        content="This is a test memory for direct testing",
        memory_type="text",
        metadata={"source": "direct_test"},
        tags=["test", "direct", "memory"]
    )
    
    logger.info(f"Created memory with ID: {memory_id}")
    
    # Retrieve the memory
    memory = await memory_service.get_memory(memory_id)
    logger.info(f"Retrieved memory: {memory['content']}")
    
    # Update the memory
    await memory_service.update_memory(
        memory_id,
        {"content": "Updated direct test memory content"}
    )
    
    # Search for memories
    results = await memory_service.search_memories("test memory")
    logger.info(f"Search found {len(results)} results")
    
    # List memories with tags
    memories = await memory_service.list_memories(
        tags=["test"]
    )
    logger.info(f"Found {len(memories)} memories with 'test' tag")
    
    # Test DAG task creation
    task = await memory_service.create_dag_task(
        operation="search",
        params={"query": "test", "limit": 5}
    )
    logger.info(f"Created memory DAG task: {task['id'] if 'id' in task else 'task created'}")
    
    return True

async def test_code_execution(code_service, logger):
    """Test code execution service operations directly."""
    logger.info("Testing code execution service...")
    
    # Test Python code execution
    python_code = """
print("Hello from Python direct test!")
result = 42
print(f"The answer is {result}")
"""
    
    logger.info("Executing Python code...")
    result = await code_service.execute_code(
        code=python_code,
        language="python",
        env=ExecutionEnvironment.LOCAL,
        timeout=5
    )
    
    logger.info(f"Execution success: {result.success}")
    logger.info(f"Execution output: {result.output}")
    
    # Test code generation
    logger.info("Testing code generation...")
    success, generated_code = await code_service.generate_code(
        prompt="Write a function to calculate factorial",
        language="python"
    )
    
    if success:
        logger.info("Generated code sample:")
        logger.info(generated_code[:100] + "..." if len(generated_code) > 100 else generated_code)
        
        # Validate the generated code
        valid, issues = code_service.validate_code(
            code=generated_code,
            language="python"
        )
        logger.info(f"Code validation result: {valid}")
    
    # Test DAG task creation
    task = await code_service.create_dag_task(
        code="print('Hello from DAG task')",
        language="python"
    )
    logger.info(f"Created code execution DAG task: {task['id'] if 'id' in task else 'task created'}")
    
    return True

async def test_service_integration(code_service, memory_service, logger):
    """Test integration between code and memory services."""
    logger.info("Testing service integration...")
    
    # Generate code and store in memory
    success, generated_code = await code_service.generate_code(
        prompt="Write a function to check if a number is prime",
        language="python"
    )
    
    if success:
        # Store the generated code in memory
        memory_id = await memory_service.create_memory(
            content=generated_code,
            memory_type="code",
            metadata={
                "language": "python",
                "description": "Prime number checker function",
                "generated": True
            },
            tags=["code", "python", "prime", "generated"]
        )
        
        logger.info(f"Stored generated code in memory with ID: {memory_id}")
        
        # Execute the stored code
        memory = await memory_service.get_memory(memory_id)
        if memory:
            result = await code_service.execute_code(
                code=memory["content"],
                language="python"
            )
            
            logger.info(f"Executed code from memory, success: {result.success}")
            
            # Store execution result back in memory
            await memory_service.update_memory(
                memory_id,
                {"metadata": {"execution_result": result.to_dict()}}
            )
            
            logger.info("Updated memory with execution results")
    
    return True

async def main():
    """Run the direct agent services test."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("direct_test")
    
    print("Starting direct agent services test...")
    
    try:
        # Create services directly
        print("Creating services...")
        config_service = ConfigService()
        logging_service = LoggingService()
        
        # Initialize services manually with minimal dependencies
        print("Initializing config service...")
        await config_service._initialize(None)
        await config_service._start()
        
        print("Initializing logging service...")
        logging_service._config = config_service
        await logging_service._initialize(None)
        await logging_service._start()
        logger = logging_service.get_logger("direct_test")
        
        print("Initializing code execution service...")
        code_service = CodeExecutionService()
        code_service._config = config_service
        code_service._logger = logging_service.get_logger("code_execution")
        await code_service._initialize(None)
        await code_service._start()
        
        print("Initializing memory service...")
        memory_service = MemoryService()
        memory_service._config = config_service
        memory_service._logger = logging_service.get_logger("memory_service")
        await memory_service._initialize(None)
        await memory_service._start()
        
        # Run tests
        print("Running memory service tests...")
        await test_memory_service(memory_service, logger)
        
        print("Running code execution tests...")
        await test_code_execution(code_service, logger)
        
        print("Running service integration tests...")
        await test_service_integration(code_service, memory_service, logger)
        
        # Shutdown services
        print("Shutting down services...")
        await memory_service._stop()
        await code_service._stop()
        await logging_service._stop()
        await config_service._stop()
        
        print("Direct tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
