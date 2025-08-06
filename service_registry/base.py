"""
Base Service Classes for ScoutAgent's Service Registry.

This module provides the core service architecture including:
- ServiceBase: Abstract base class for all services
- ServiceRegistry: Central registry for managing services
- Service lifecycle management
"""

from abc import ABC, abstractmethod
import asyncio
import inspect
import uuid
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Type, Set, Callable, Tuple, Union
import logging
from collections import defaultdict

from custom_logging import get_logger


class ServiceState(Enum):
    """Possible states of a service."""
    UNINITIALIZED = "uninitialized"  # Service instance created but not initialized
    INITIALIZING = "initializing"    # Service initialization in progress
    INITIALIZED = "initialized"      # Service initialized but not started
    STARTING = "starting"            # Service startup in progress
    RUNNING = "running"              # Service is running
    STOPPING = "stopping"            # Service shutdown in progress
    STOPPED = "stopped"              # Service has been stopped
    FAILED = "failed"                # Service failed during initialization/startup/runtime


class ServiceDependency:
    """
    Represents a dependency between services.
    
    Attributes:
        service_name: Name of the required service
        required: If True, service cannot start without this dependency
        version: Optional version requirement
    """
    def __init__(self, service_name: str, required: bool = True, version: Optional[str] = None):
        self.service_name = service_name
        self.required = required
        self.version = version
    
    def __repr__(self):
        return f"ServiceDependency({self.service_name}, required={self.required})"


class ServiceBase(ABC):
    """
    Abstract base class for all services.
    
    All services in the system should inherit from this class and implement
    the required lifecycle methods.
    """
    
    def __init__(self, name: Optional[str] = None, version: str = "1.0.0"):
        """
        Initialize a service.
        
        Args:
            name: Service name (defaults to class name if None)
            version: Service version
        """
        self.name = name or self.__class__.__name__
        self.version = version
        self.service_id = str(uuid.uuid4())
        self.state = ServiceState.UNINITIALIZED
        self.logger = get_logger(f"service.{self.name}")
        self.dependencies: List[ServiceDependency] = []
        self.start_time = None
        self.stop_time = None
        self._is_singleton = getattr(self.__class__, "_is_singleton", True)
        self._initialized = False
        self._context = {}  # Service context for storing runtime data
    
    def __str__(self):
        return f"{self.name} ({self.service_id[:8]}, state={self.state.value})"
    
    def __repr__(self):
        return f"Service<{self.name}, {self.state.value}>"
    
    def declare_dependency(self, service_name: str, required: bool = True, version: Optional[str] = None):
        """
        Declare a dependency on another service.
        
        Args:
            service_name: Name of the required service
            required: If True, service cannot start without this dependency
            version: Optional version requirement
        """
        dependency = ServiceDependency(service_name, required, version)
        self.dependencies.append(dependency)
        self.logger.debug(f"Declared dependency on {service_name} (required={required})")
    
    def get_dependencies(self) -> List[ServiceDependency]:
        """Get list of declared service dependencies."""
        return self.dependencies
    
    async def initialize(self, registry: 'ServiceRegistry') -> bool:
        """
        Initialize the service with configuration and dependencies.
        
        This is called before the service is started to set up any resources
        or configuration needed by the service. Should be overridden by subclasses.
        
        Args:
            registry: Service registry instance for accessing dependencies
            
        Returns:
            True if initialization was successful, False otherwise
        """
        self.state = ServiceState.INITIALIZING
        self.logger.info(f"Initializing service: {self.name}")
        
        try:
            result = await self._initialize(registry)
            if result:
                self.state = ServiceState.INITIALIZED
                self._initialized = True
                self.logger.info(f"Service {self.name} initialized successfully")
            else:
                self.state = ServiceState.FAILED
                self.logger.error(f"Service {self.name} initialization failed")
            return result
        except Exception as e:
            self.state = ServiceState.FAILED
            self.logger.exception(f"Error initializing service {self.name}: {e}")
            return False
    
    async def _initialize(self, registry: 'ServiceRegistry') -> bool:
        """
        Internal initialization implementation.
        
        Should be overridden by subclasses for custom initialization logic.
        
        Args:
            registry: Service registry for accessing dependencies
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        return True
    
    async def start(self) -> bool:
        """
        Start the service.
        
        This is called after initialization to start the service running.
        Should be overridden by subclasses for custom startup logic.
        
        Returns:
            True if startup was successful, False otherwise
        """
        if not self._initialized:
            self.logger.error(f"Cannot start uninitialized service: {self.name}")
            return False
        
        self.state = ServiceState.STARTING
        self.start_time = time.time()
        self.logger.info(f"Starting service: {self.name}")
        
        try:
            result = await self._start()
            if result:
                self.state = ServiceState.RUNNING
                self.logger.info(f"Service {self.name} started successfully")
            else:
                self.state = ServiceState.FAILED
                self.logger.error(f"Service {self.name} startup failed")
            return result
        except Exception as e:
            self.state = ServiceState.FAILED
            self.logger.exception(f"Error starting service {self.name}: {e}")
            return False
    
    async def _start(self) -> bool:
        """
        Internal startup implementation.
        
        Should be overridden by subclasses for custom startup logic.
        
        Returns:
            True if startup succeeded, False otherwise
        """
        return True
    
    async def stop(self) -> bool:
        """
        Stop the service.
        
        This is called when the service needs to be stopped, either during
        system shutdown or when a service is being restarted.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        if self.state not in (ServiceState.RUNNING, ServiceState.FAILED):
            self.logger.warning(f"Service {self.name} is not running, current state: {self.state}")
            return True
            
        self.state = ServiceState.STOPPING
        self.logger.info(f"Stopping service: {self.name}")
        
        try:
            result = await self._stop()
            if result:
                self.state = ServiceState.STOPPED
                self.stop_time = time.time()
                self.logger.info(f"Service {self.name} stopped successfully")
            else:
                self.state = ServiceState.FAILED
                self.logger.error(f"Service {self.name} shutdown failed")
            return result
        except Exception as e:
            self.state = ServiceState.FAILED
            self.logger.exception(f"Error stopping service {self.name}: {e}")
            return False
    
    async def _stop(self) -> bool:
        """
        Internal shutdown implementation.
        
        Should be overridden by subclasses for custom shutdown logic.
        
        Returns:
            True if shutdown succeeded, False otherwise
        """
        return True
    
    def is_singleton(self) -> bool:
        """
        Return whether this service is a singleton.
        
        If True, only one instance of this service can exist in the registry.
        If False, multiple instances can be created (e.g. for different configurations).
        """
        return self._is_singleton
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current service status information.
        
        Returns:
            Dictionary with service status information
        """
        status = {
            "service_id": self.service_id,
            "name": self.name,
            "version": self.version,
            "state": self.state.value,
            "uptime": None,
            "singleton": self._is_singleton,
            "dependencies": [dep.service_name for dep in self.dependencies]
        }
        
        if self.start_time:
            if self.stop_time:
                status["uptime"] = self.stop_time - self.start_time
            else:
                status["uptime"] = time.time() - self.start_time
                
        return status


class ServiceRegistry:
    """
    Central registry for all services.
    
    This class manages service registration, lifecycle, and dependency resolution.
    """
    
    def __init__(self):
        """Initialize the service registry."""
        self.logger = get_logger("service_registry")
        self.services: Dict[str, Dict[str, ServiceBase]] = defaultdict(dict)
        self.service_classes: Dict[str, Type[ServiceBase]] = {}
        self.factory_functions: Dict[str, Callable[..., ServiceBase]] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
    
    def register_service(self, service_class: Type[ServiceBase]) -> None:
        """
        Register a service class with the registry.
        
        Args:
            service_class: Service class to register (must inherit from ServiceBase)
        """
        if not issubclass(service_class, ServiceBase):
            raise TypeError(f"Service class must inherit from ServiceBase: {service_class.__name__}")
        
        # Use class name as service name if not specified
        service_name = getattr(service_class, "_service_name", service_class.__name__)
        
        # Check if service already registered
        if service_name in self.service_classes:
            self.logger.warning(f"Service '{service_name}' already registered, overriding")
        
        self.service_classes[service_name] = service_class
        self.logger.info(f"Registered service: {service_name}")
    
    def register_factory(self, 
                       service_name: str, 
                       factory_func: Callable[..., ServiceBase]) -> None:
        """
        Register a factory function for creating service instances.
        
        Args:
            service_name: Name of the service
            factory_func: Function that returns a ServiceBase instance
        """
        if service_name in self.factory_functions:
            self.logger.warning(f"Factory for '{service_name}' already registered, overriding")
            
        self.factory_functions[service_name] = factory_func
        self.logger.info(f"Registered factory for service: {service_name}")
    
    def create_service(self, service_name: str, **kwargs) -> ServiceBase:
        """
        Create a new instance of a registered service.
        
        Args:
            service_name: Name of the registered service
            **kwargs: Arguments to pass to service constructor or factory function
            
        Returns:
            ServiceBase instance
        """
        # Check if we have a factory function
        if service_name in self.factory_functions:
            self.logger.debug(f"Creating service '{service_name}' using factory function")
            service = self.factory_functions[service_name](**kwargs)
            
        # Check if we have a registered service class
        elif service_name in self.service_classes:
            self.logger.debug(f"Creating service '{service_name}' using registered class")
            service_class = self.service_classes[service_name]
            service = service_class(**kwargs)
            
        else:
            raise ValueError(f"Service not registered: {service_name}")
        
        # Validate service instance
        if not isinstance(service, ServiceBase):
            raise TypeError(f"Service must inherit from ServiceBase: {type(service).__name__}")
            
        # For singleton services, check if we already have an instance
        if service.is_singleton() and service_name in self.services:
            self.logger.warning(
                f"Singleton service '{service_name}' already exists, returning existing instance"
            )
            return next(iter(self.services[service_name].values()))
            
        # Store the service instance
        self.services[service_name][service.service_id] = service
        self.logger.info(f"Created service: {service_name} ({service.service_id})")
            
        return service
    
    def get_service(self, service_name: str, service_id: Optional[str] = None) -> ServiceBase:
        """
        Get a service instance by name and optional ID.
        
        Args:
            service_name: Service name
            service_id: Optional service instance ID (for non-singleton services)
            
        Returns:
            ServiceBase instance
        """
        if service_name not in self.services:
            raise ValueError(f"No instances of service '{service_name}' found")
            
        if service_id:
            # Get specific service instance by ID
            if service_id not in self.services[service_name]:
                raise ValueError(f"Service '{service_name}' with ID '{service_id}' not found")
            return self.services[service_name][service_id]
            
        # If no ID specified, return the first instance
        # (for singleton services this is the only instance)
        return next(iter(self.services[service_name].values()))
    
    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self.services or service_name in self.service_classes
    
    def has_service_instance(self, service_name: str) -> bool:
        """Check if a service has active instances."""
        return service_name in self.services and len(self.services[service_name]) > 0
    
    def list_services(self) -> List[str]:
        """List all registered service types."""
        return list(self.service_classes.keys())
    
    def list_service_instances(self) -> Dict[str, List[str]]:
        """List all service instances by name and ID."""
        return {
            name: list(instances.keys())
            for name, instances in self.services.items()
        }
    
    async def initialize_service(self, service: ServiceBase) -> bool:
        """
        Initialize a service and its dependencies.
        
        Args:
            service: Service instance to initialize
            
        Returns:
            True if initialization was successful, False otherwise
        """
        # Skip if already initialized
        if service.state != ServiceState.UNINITIALIZED:
            return service.state == ServiceState.INITIALIZED
        
        self.logger.info(f"Initializing service and dependencies: {service.name}")
        
        # Initialize dependencies first
        dependencies_ok = True
        for dependency in service.get_dependencies():
            # Skip non-required dependencies
            if not dependency.required:
                continue
                
            if not self.has_service_instance(dependency.service_name):
                # If the dependency exists but hasn't been instantiated yet
                if dependency.service_name in self.service_classes:
                    dep_service = self.create_service(dependency.service_name)
                    success = await self.initialize_service(dep_service)
                    if not success and dependency.required:
                        self.logger.error(
                            f"Failed to initialize required dependency {dependency.service_name} "
                            f"for service {service.name}"
                        )
                        dependencies_ok = False
                else:
                    if dependency.required:
                        self.logger.error(
                            f"Required dependency {dependency.service_name} not found "
                            f"for service {service.name}"
                        )
                        dependencies_ok = False
            else:
                # Get the dependency service
                dep_service = self.get_service(dependency.service_name)
                
                # Initialize it if needed
                if dep_service.state == ServiceState.UNINITIALIZED:
                    success = await self.initialize_service(dep_service)
                    if not success and dependency.required:
                        dependencies_ok = False
        
        # Don't initialize if dependencies failed
        if not dependencies_ok:
            self.logger.error(f"Not initializing {service.name} due to dependency failure")
            return False
        
        # Initialize the service
        return await service.initialize(self)
    
    async def start_service(self, service: ServiceBase) -> bool:
        """
        Start a service and its dependencies.
        
        Args:
            service: Service instance to start
            
        Returns:
            True if startup was successful, False otherwise
        """
        # Skip if already running
        if service.state == ServiceState.RUNNING:
            return True
            
        # Can't start a service that isn't initialized
        if service.state != ServiceState.INITIALIZED:
            self.logger.error(
                f"Cannot start service {service.name} in state {service.state}"
            )
            return False
        
        self.logger.info(f"Starting service and dependencies: {service.name}")
        
        # Start dependencies first
        dependencies_ok = True
        for dependency in service.get_dependencies():
            # Skip non-required dependencies
            if not dependency.required:
                continue
                
            if self.has_service_instance(dependency.service_name):
                dep_service = self.get_service(dependency.service_name)
                
                # Ensure dependency is initialized
                if dep_service.state == ServiceState.UNINITIALIZED:
                    success = await self.initialize_service(dep_service)
                    if not success:
                        dependencies_ok = False
                        continue
                
                # Start the dependency if needed
                if dep_service.state == ServiceState.INITIALIZED:
                    success = await self.start_service(dep_service)
                    if not success:
                        dependencies_ok = False
            else:
                if dependency.required:
                    self.logger.error(
                        f"Required dependency {dependency.service_name} not found "
                        f"for service {service.name}"
                    )
                    dependencies_ok = False
        
        # Don't start if dependencies failed
        if not dependencies_ok:
            self.logger.error(f"Not starting {service.name} due to dependency failure")
            return False
        
        # Add to startup order
        if service.name not in self.startup_order:
            self.startup_order.append(service.name)
            
        # Start the service
        return await service.start()
    
    async def stop_service(self, service: ServiceBase) -> bool:
        """
        Stop a service.
        
        Args:
            service: Service instance to stop
            
        Returns:
            True if shutdown was successful, False otherwise
        """
        # Skip if not running
        if service.state != ServiceState.RUNNING:
            return True
        
        self.logger.info(f"Stopping service: {service.name}")
        
        # Add to shutdown order (in reverse of startup)
        if service.name not in self.shutdown_order:
            self.shutdown_order.append(service.name)
            
        # Stop the service
        return await service.stop()
    
    async def initialize_all(self) -> bool:
        """
        Initialize all registered services.
        
        Returns:
            True if all services were initialized successfully, False otherwise
        """
        self.logger.info("Initializing all registered services")
        success = True
        
        # Create instances of all registered services that don't have instances
        for service_name, service_class in self.service_classes.items():
            if not self.has_service_instance(service_name):
                self.create_service(service_name)
        
        # Initialize all service instances
        for service_name, instances in self.services.items():
            for service_id, service in instances.items():
                if service.state == ServiceState.UNINITIALIZED:
                    if not await self.initialize_service(service):
                        success = False
        
        if success:
            self.logger.info("All services initialized successfully")
        else:
            self.logger.error("Some services failed to initialize")
        
        return success
    
    async def start_all(self) -> bool:
        """
        Start all initialized services.
        
        Returns:
            True if all services were started successfully, False otherwise
        """
        self.logger.info("Starting all initialized services")
        success = True
        
        # Start all initialized services
        for service_name, instances in self.services.items():
            for service_id, service in instances.items():
                if service.state == ServiceState.INITIALIZED:
                    if not await self.start_service(service):
                        success = False
        
        if success:
            self.logger.info("All services started successfully")
        else:
            self.logger.error("Some services failed to start")
        
        return success
    
    async def stop_all(self) -> bool:
        """
        Stop all running services in reverse dependency order.
        
        Returns:
            True if all services were stopped successfully, False otherwise
        """
        self.logger.info("Stopping all running services")
        success = True
        
        # Stop services in reverse of startup order
        for service_name in reversed(self.startup_order):
            if self.has_service_instance(service_name):
                service = self.get_service(service_name)
                if service.state == ServiceState.RUNNING:
                    if not await self.stop_service(service):
                        success = False
        
        # Stop any services not in startup order but still running
        for service_name, instances in self.services.items():
            if service_name not in self.startup_order:
                for service_id, service in instances.items():
                    if service.state == ServiceState.RUNNING:
                        if not await self.stop_service(service):
                            success = False
        
        if success:
            self.logger.info("All services stopped successfully")
        else:
            self.logger.error("Some services failed to stop")
        
        return success
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for all services.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "registered_services": list(self.service_classes.keys()),
            "active_instances": {
                service_name: {
                    service_id: service.get_status()
                    for service_id, service in instances.items()
                }
                for service_name, instances in self.services.items()
            },
            "startup_order": self.startup_order,
            "shutdown_order": self.shutdown_order
        }


# Global registry instance
_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


def register_service(service_class: Type[ServiceBase]) -> Type[ServiceBase]:
    """
    Register a service class with the global registry.
    
    This can be used as a decorator:
    
    @register_service
    class MyService(ServiceBase):
        ...
    
    Args:
        service_class: Service class to register
        
    Returns:
        The original service class (for decorator use)
    """
    get_registry().register_service(service_class)
    return service_class


def create_service(service_name: str, **kwargs) -> ServiceBase:
    """Create a service instance using the global registry."""
    return get_registry().create_service(service_name, **kwargs)


def get_service(service_name: str) -> ServiceBase:
    """Get a service instance from the global registry."""
    return get_registry().get_service(service_name)
