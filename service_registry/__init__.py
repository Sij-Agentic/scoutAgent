"""
Service Registry Package for ScoutAgent.

This package provides a hybrid Registry/Dependency Injection approach for
service management throughout the ScoutAgent system.

Core components:
- ServiceBase: Base class for all services
- ServiceRegistry: Central registry for managing services
- Decorators for service registration and dependency injection
- Service lifecycle management
"""

from .base import (
    ServiceBase, 
    ServiceRegistry,
    ServiceState,
    ServiceDependency,
    get_registry,
    register_service,
    create_service,
    get_service
)

from .decorators import (
    service,
    requires,
    factory,
    inject
)

from .lifecycle import (
    LifecycleManager,
    DependencyGraph,
    get_lifecycle_manager
)

from .exceptions import (
    ServiceRegistryError,
    ServiceNotFoundError,
    ServiceInitializationError,
    ServiceStartupError,
    ServiceShutdownError,
    DependencyError,
    CircularDependencyError
)

__all__ = [
    # Base classes and functions
    'ServiceBase',
    'ServiceRegistry',
    'ServiceState',
    'ServiceDependency',
    'get_registry',
    'register_service',
    'create_service',
    'get_service',
    
    # Decorators
    'service',
    'requires',
    'factory',
    'inject',
    
    # Lifecycle management
    'LifecycleManager',
    'DependencyGraph',
    'get_lifecycle_manager',
    
    # Exceptions
    'ServiceRegistryError',
    'ServiceNotFoundError',
    'ServiceInitializationError',
    'ServiceStartupError',
    'ServiceShutdownError',
    'DependencyError',
    'CircularDependencyError'
]
