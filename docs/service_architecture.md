# ScoutAgent Service Architecture Documentation

## Overview

ScoutAgent has been refactored to use a robust service-oriented architecture that provides better dependency management, lifecycle control, and testability. This document explains the architecture of the central service registry, memory service, and code execution service.

## Service Registry Architecture

The service registry implements a hybrid Service Registry/Dependency Injection pattern with the following key features:

### Core Concepts

1. **Service Registration**: Services are registered with the global registry using the `@service` decorator, which specifies:
   - Service name
   - Whether it's a singleton or factory-created instance
   - Version information

2. **Dependency Management**: Dependencies between services are declared using the `@requires` decorator with:
   - Required service name
   - Optional flag to indicate if the dependency is optional
   - Proper initialization order based on dependency graph

3. **Lifecycle Management**:
   - Services follow a structured lifecycle: initialize → start → stop → shutdown
   - Initialization order respects dependencies (dependencies initialize first)
   - Shutdown order is reversed to prevent dangling references

4. **Service Discovery**: Services can be looked up at runtime using:
   - `get_service(name)` - Get a service instance by name
   - `has_service(name)` - Check if a service is registered
   - `get_registry()` - Get the global registry instance

### Service Base Class

All services inherit from `ServiceBase`, which provides:

- Standard lifecycle methods (`_initialize`, `_start`, `_stop`, `_shutdown`)
- Dependency declaration and resolution
- Registry integration
- Version and state management

## Memory Service Architecture

The MemoryService replaces the MemoryAgent with a more robust service-oriented design.

### Key Features

1. **Storage Backend Integration**:
   - Abstracted storage backends (FileSystem, VectorDB)
   - Pluggable backend architecture
   - Transparent switching between backends

2. **Memory Operations**:
   - CRUD operations for memory entries
   - Advanced search capabilities with relevance scoring
   - Tagging and metadata support
   - Efficient caching of search results

3. **DAG Integration**:
   - Memory operations exposed as task nodes for workflows
   - Asynchronous memory operations within DAGs

4. **Robustness Improvements**:
   - Optional dependencies on config and logging services
   - Direct initialization support for testing
   - Graceful handling of missing dependencies
   - Default fallback backends
   - Proper registration in the global registry

### Usage Patterns

```python
# Via service registry (preferred)
from service_registry import get_registry
memory_service = get_registry().get_service("memory")
await memory_service.create_memory("content", "text", tags=["important"])

# Via direct factory method
from services.agents.memory import get_memory_service
memory_service = get_memory_service()
await memory_service.search_memories("query")

# For testing
memory_service = MemoryService()
memory_service.setup_direct(config=test_config, logger=test_logger)
await memory_service._initialize(None)  # Direct initialization
```

## Code Execution Service Architecture

The CodeExecutionService replaces the CodeAgent with a service that manages code generation, validation, and execution in various environments.

### Key Features

1. **Code Execution Environments**:
   - Local execution in the host environment
   - Sandboxed execution via Docker containers
   - WebAssembly execution (planned)
   - Resource limits and timeouts

2. **Language Support**:
   - Multi-language execution (Python, JavaScript, Bash, etc.)
   - Language-specific linting and validation
   - Cross-language execution pipelines

3. **Security Features**:
   - Resource limits (CPU, memory, processes)
   - Network access controls
   - Execution timeouts
   - Safe code validation

4. **DAG Integration**:
   - Code execution tasks for workflows
   - Asynchronous generation and execution
   - Result capture and analysis

5. **Robustness Improvements**:
   - Optional dependencies on config and logging services
   - Direct initialization support for testing
   - Graceful handling of missing dependencies
   - Working directory management
   - Proper registration in the global registry

### Usage Patterns

```python
# Via service registry (preferred)
from service_registry import get_registry
code_service = get_registry().get_service("code_execution")
result = await code_service.execute_code("print('hello')", "python")

# Via direct factory method
from services.agents.code import get_code_execution_service
code_service = get_code_execution_service()
success, code = await code_service.generate_code("Write a factorial function", "python")

# For testing
code_service = CodeExecutionService()
code_service.setup_direct(working_dir="/tmp/test_code")
await code_service._initialize(None)  # Direct initialization
```

## Integration Benefits

1. **Elimination of Agent-to-Agent Messaging Overhead**:
   - Direct service method calls replace complex agent messaging
   - Simplified debugging and testing
   - Lower latency for operations

2. **Centralized Configuration**:
   - Services access configuration via the ConfigService
   - Agent-specific LLM backend preferences
   - Environment-specific overrides

3. **Consistent Logging**:
   - Services access logging via the LoggingService
   - Consistent log format and levels
   - Context-aware logging

4. **Improved Error Handling**:
   - Structured error propagation
   - Recovery mechanisms
   - Circuit breakers for service resilience

5. **Testing Support**:
   - Direct instantiation without registry
   - Mock dependencies
   - Isolated testing

## Best Practices

1. **Service Design**:
   - Make core dependencies optional when possible
   - Implement `setup_direct` for testing
   - Handle missing dependencies gracefully
   - Register instances in the global registry

2. **Dependency Management**:
   - Declare dependencies explicitly with `@requires`
   - Use optional dependencies when appropriate
   - Follow initialization/shutdown lifecycle correctly

3. **Error Handling**:
   - Catch exceptions in lifecycle methods
   - Provide meaningful error messages
   - Implement fallback mechanisms

4. **State Management**:
   - Initialize service state in `_initialize`
   - Start active components in `_start`
   - Stop active components in `_stop`
   - Clean up resources in `_shutdown`
