"""
Service Registry Decorators for ScoutAgent.

This module provides decorators for registering and configuring services.
"""

import inspect
from functools import wraps
from typing import Optional, List, Type, Dict, Any, Callable, Union, TypeVar

from service_registry.base import ServiceBase, get_registry, register_service


T = TypeVar('T', bound=ServiceBase)


def service(name: Optional[str] = None, singleton: bool = True):
    """
    Decorator for registering a service class.
    
    This decorator both registers the service with the registry and configures
    basic service properties like name and singleton status.
    
    Args:
        name: Custom service name (defaults to class name)
        singleton: Whether the service is a singleton (only one instance allowed)
        
    Returns:
        Decorator function
    
    Example:
        @service(name="config", singleton=True)
        class ConfigService(ServiceBase):
            ...
    """
    def decorator(cls: Type[ServiceBase]) -> Type[ServiceBase]:
        # Set service properties
        if name is not None:
            cls._service_name = name
        cls._is_singleton = singleton
        
        # Register the service
        return register_service(cls)
    
    return decorator


def requires(*service_names: str, optional: bool = False):
    """
    Decorator for declaring service dependencies.
    
    Args:
        *service_names: Names of required services
        optional: Whether dependencies are optional
        
    Returns:
        Decorator function
    
    Example:
        @service()
        @requires("config", "logging")
        class MyService(ServiceBase):
            ...
    """
    def decorator(cls: Type[ServiceBase]) -> Type[ServiceBase]:
        # Store original __init__ method
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Call the original __init__
            original_init(self, *args, **kwargs)
            
            # Declare dependencies
            for service_name in service_names:
                self.declare_dependency(service_name, required=not optional)
                
        # Replace __init__ method
        cls.__init__ = new_init
        
        return cls
    
    return decorator


def factory(service_name: str):
    """
    Decorator for registering a factory function for a service.
    
    A factory function creates service instances with custom configuration.
    
    Args:
        service_name: Name of the service this factory creates
        
    Returns:
        Decorator function
    
    Example:
        @factory("database")
        def create_database_service(connection_string: str) -> DatabaseService:
            return DatabaseService(connection_string)
    """
    def decorator(factory_func: Callable[..., ServiceBase]) -> Callable[..., ServiceBase]:
        # Register the factory function
        get_registry().register_factory(service_name, factory_func)
        
        # Return the original function unchanged
        return factory_func
    
    return decorator


def inject(*service_names: str):
    """
    Decorator for injecting services into a function or method.
    
    This decorator injects the specified services as keyword arguments.
    
    Args:
        *service_names: Names of services to inject
        
    Returns:
        Decorator function
    
    Example:
        @inject("config", "logging")
        def setup_application(config, logging):
            # config and logging are injected service instances
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the registry
            registry = get_registry()
            
            # Inject services as kwargs
            for service_name in service_names:
                if service_name not in kwargs:
                    if registry.has_service_instance(service_name):
                        kwargs[service_name] = registry.get_service(service_name)
            
            # Call the original function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
