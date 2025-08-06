"""
Service Lifecycle Management for ScoutAgent.

This module provides tools for managing service lifecycles including:
- Initialization and shutdown ordering based on dependencies
- Service startup and shutdown coordination
- Service health monitoring
"""

import asyncio
import time
import networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple

from custom_logging import get_logger
from service_registry.base import ServiceBase, ServiceRegistry, ServiceState
from service_registry.exceptions import CircularDependencyError, DependencyError

logger = get_logger("service_lifecycle")


class DependencyGraph:
    """
    Builds and analyzes the dependency graph for services.
    
    This class helps determine initialization and shutdown order
    based on declared service dependencies.
    """
    
    def __init__(self, registry: ServiceRegistry):
        """
        Initialize the dependency graph.
        
        Args:
            registry: Service registry instance
        """
        self.registry = registry
        self.logger = get_logger("dependency_graph")
        self.graph = nx.DiGraph()
        
    def build_graph(self):
        """
        Build the dependency graph from current service instances.
        
        Raises:
            CircularDependencyError: If circular dependencies are detected
        """
        self.logger.debug("Building service dependency graph")
        self.graph.clear()
        
        # Add all services as nodes
        for service_name, instances in self.registry.services.items():
            for service_id, service in instances.items():
                # Use service_id as node identifier for uniqueness
                self.graph.add_node(service_id, 
                                   service=service, 
                                   name=service_name)
        
        # Add dependency edges
        for service_name, instances in self.registry.services.items():
            for service_id, service in instances.items():
                for dependency in service.get_dependencies():
                    dep_name = dependency.service_name
                    
                    # Find all instances of the dependency
                    if dep_name in self.registry.services:
                        dep_instances = self.registry.services[dep_name]
                        
                        # For singletons, we only need one edge
                        if len(dep_instances) == 1 or service._is_singleton:
                            dep_id = next(iter(dep_instances.keys()))
                            self.graph.add_edge(
                                service_id, 
                                dep_id, 
                                required=dependency.required
                            )
                        else:
                            # For non-singletons, add edges to all instances
                            for dep_id in dep_instances:
                                self.graph.add_edge(
                                    service_id, 
                                    dep_id, 
                                    required=dependency.required
                                )
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                # Convert service IDs in cycle to service names for better error messages
                named_cycles = []
                for cycle in cycles:
                    named_cycle = []
                    for service_id in cycle:
                        for node_id, node_data in self.graph.nodes(data=True):
                            if node_id == service_id:
                                named_cycle.append(node_data.get('name', service_id))
                    named_cycles.append(named_cycle)
                    
                self.logger.error(f"Circular dependencies detected: {named_cycles}")
                raise CircularDependencyError(named_cycles[0])
                
        except nx.NetworkXNoCycle:
            pass
            
        self.logger.debug(f"Dependency graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
                
    def get_initialization_order(self) -> List[str]:
        """
        Get the initialization order for services.
        
        Returns:
            List of service IDs in dependency order
        """
        # Build the graph if it's empty
        if not self.graph.nodes():
            self.build_graph()
            
        # Get topological sort (dependencies first)
        try:
            order = list(nx.topological_sort(self.graph))
            order.reverse()  # Reverse so that dependencies come first
            return order
        except nx.NetworkXUnfeasible:
            self.logger.error("Cannot determine initialization order: graph has cycles")
            # Return alphabetical order as fallback
            return sorted(list(self.graph.nodes()))
            
    def get_shutdown_order(self) -> List[str]:
        """
        Get the shutdown order for services (reverse of initialization).
        
        Returns:
            List of service IDs in reverse dependency order
        """
        init_order = self.get_initialization_order()
        return list(reversed(init_order))
        
    def get_required_services(self, service_id: str) -> Set[str]:
        """
        Get all required services for a given service.
        
        Args:
            service_id: Service ID to get dependencies for
            
        Returns:
            Set of service IDs that the given service depends on
        """
        if not self.graph.nodes():
            self.build_graph()
            
        if service_id not in self.graph.nodes:
            return set()
            
        required = set()
        for _, dep_id, data in self.graph.out_edges(service_id, data=True):
            if data.get('required', True):
                required.add(dep_id)
                # Recursively add dependencies of dependencies
                required.update(self.get_required_services(dep_id))
                
        return required


class LifecycleManager:
    """
    Manages the lifecycle of all services.
    
    This class coordinates initialization, startup, and shutdown
    of services based on their dependencies.
    """
    
    def __init__(self, registry: ServiceRegistry):
        """
        Initialize the lifecycle manager.
        
        Args:
            registry: Service registry instance
        """
        self.registry = registry
        self.logger = get_logger("lifecycle_manager")
        self.dependency_graph = DependencyGraph(registry)
        self.initialized = False
        
    async def initialize_all(self) -> bool:
        """
        Initialize all services in dependency order.
        
        Returns:
            True if all services were initialized successfully
        """
        if self.initialized:
            self.logger.info("Services already initialized")
            return True
            
        self.logger.info("Initializing all services in dependency order")
        
        # Build dependency graph
        try:
            self.dependency_graph.build_graph()
        except CircularDependencyError as e:
            self.logger.error(f"Cannot initialize services: {e}")
            return False
            
        # Get initialization order
        init_order = self.dependency_graph.get_initialization_order()
        self.logger.debug(f"Initialization order: {init_order}")
        
        # Initialize services in order
        success = True
        for service_id in init_order:
            node_data = self.dependency_graph.graph.nodes[service_id]
            service = node_data['service']
            
            self.logger.info(f"Initializing service: {service.name} ({service_id})")
            
            # Initialize the service
            if not await service.initialize(self.registry):
                self.logger.error(f"Failed to initialize service: {service.name} ({service_id})")
                success = False
                
                # Check if service has dependents
                dependent_services = []
                for s_id in self.dependency_graph.graph.nodes:
                    deps = self.dependency_graph.get_required_services(s_id)
                    if service_id in deps:
                        node = self.dependency_graph.graph.nodes[s_id]
                        dependent_services.append(node['service'].name)
                
                if dependent_services:
                    self.logger.warning(
                        f"Service {service.name} failed to initialize, "
                        f"these services will be affected: {dependent_services}"
                    )
        
        self.initialized = success
        return success
        
    async def start_all(self) -> bool:
        """
        Start all initialized services in dependency order.
        
        Returns:
            True if all services were started successfully
        """
        if not self.initialized:
            self.logger.error("Cannot start services: not initialized")
            return False
            
        self.logger.info("Starting all initialized services in dependency order")
        
        # Get initialization order (same as startup order)
        startup_order = self.dependency_graph.get_initialization_order()
        self.logger.debug(f"Startup order: {startup_order}")
        
        # Start services in order
        success = True
        for service_id in startup_order:
            node_data = self.dependency_graph.graph.nodes[service_id]
            service = node_data['service']
            
            # Skip services that aren't initialized
            if service.state != ServiceState.INITIALIZED:
                continue
                
            self.logger.info(f"Starting service: {service.name} ({service_id})")
            
            # Start the service
            if not await service.start():
                self.logger.error(f"Failed to start service: {service.name} ({service_id})")
                success = False
        
        return success
        
    async def shutdown_all(self) -> bool:
        """
        Shutdown all running services in reverse dependency order.
        
        Returns:
            True if all services were shutdown successfully
        """
        self.logger.info("Shutting down all services in reverse dependency order")
        
        # Get shutdown order (reverse of initialization)
        shutdown_order = self.dependency_graph.get_shutdown_order()
        self.logger.debug(f"Shutdown order: {shutdown_order}")
        
        # Stop services in reverse order
        success = True
        for service_id in shutdown_order:
            node_data = self.dependency_graph.graph.nodes[service_id]
            service = node_data['service']
            
            # Skip services that aren't running
            if service.state != ServiceState.RUNNING:
                continue
                
            self.logger.info(f"Stopping service: {service.name} ({service_id})")
            
            # Stop the service
            if not await service.stop():
                self.logger.error(f"Failed to stop service: {service.name} ({service_id})")
                success = False
        
        return success
        
    async def restart_service(self, service_name: str, service_id: Optional[str] = None) -> bool:
        """
        Restart a specific service and its dependents.
        
        Args:
            service_name: Name of the service to restart
            service_id: Optional service instance ID
            
        Returns:
            True if the service was restarted successfully
        """
        self.logger.info(f"Restarting service: {service_name}")
        
        # Get the service instance
        try:
            service = self.registry.get_service(service_name, service_id)
        except ValueError as e:
            self.logger.error(f"Cannot restart service: {e}")
            return False
        
        # Find services that depend on this one
        dependent_services = []
        self.dependency_graph.build_graph()
        
        # Stop dependents first
        for node_id, node_data in self.dependency_graph.graph.nodes(data=True):
            node_service = node_data['service']
            deps = self.dependency_graph.get_required_services(node_id)
            if service.service_id in deps and node_service.state == ServiceState.RUNNING:
                self.logger.info(f"Stopping dependent service: {node_service.name}")
                await node_service.stop()
                dependent_services.append(node_service)
        
        # Restart the target service
        success = True
        if service.state == ServiceState.RUNNING:
            success = await service.stop()
            
        if success:
            success = await service.initialize(self.registry)
            if success:
                success = await service.start()
                
        # Restart dependent services
        if success:
            for dep_service in reversed(dependent_services):
                self.logger.info(f"Restarting dependent service: {dep_service.name}")
                if await dep_service.initialize(self.registry):
                    await dep_service.start()
                    
        return success


# Create a lifecycle manager for the global registry
def get_lifecycle_manager() -> LifecycleManager:
    """Get a lifecycle manager for the global registry."""
    from service_registry.base import get_registry
    return LifecycleManager(get_registry())
