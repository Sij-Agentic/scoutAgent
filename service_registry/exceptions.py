"""
Service Registry Exceptions for ScoutAgent.

This module defines custom exceptions for the service registry system.
"""

class ServiceRegistryError(Exception):
    """Base exception for all service registry errors."""
    pass


class ServiceNotFoundError(ServiceRegistryError):
    """Raised when a requested service is not registered."""
    def __init__(self, service_name):
        super().__init__(f"Service not found: {service_name}")
        self.service_name = service_name


class ServiceInitializationError(ServiceRegistryError):
    """Raised when a service fails to initialize."""
    def __init__(self, service_name, reason=None):
        message = f"Service initialization failed: {service_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
        self.service_name = service_name
        self.reason = reason


class ServiceStartupError(ServiceRegistryError):
    """Raised when a service fails to start."""
    def __init__(self, service_name, reason=None):
        message = f"Service startup failed: {service_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
        self.service_name = service_name
        self.reason = reason


class ServiceShutdownError(ServiceRegistryError):
    """Raised when a service fails to stop."""
    def __init__(self, service_name, reason=None):
        message = f"Service shutdown failed: {service_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
        self.service_name = service_name
        self.reason = reason


class DependencyError(ServiceRegistryError):
    """Raised when there's an issue with service dependencies."""
    def __init__(self, service_name, dependency_name, message=None):
        if not message:
            message = f"Dependency error for {service_name} -> {dependency_name}"
        super().__init__(message)
        self.service_name = service_name
        self.dependency_name = dependency_name


class CircularDependencyError(DependencyError):
    """Raised when circular dependencies are detected."""
    def __init__(self, dependency_chain):
        chain_str = " -> ".join(dependency_chain)
        super().__init__(
            dependency_chain[0], 
            dependency_chain[-1], 
            f"Circular dependency detected: {chain_str}"
        )
        self.dependency_chain = dependency_chain
