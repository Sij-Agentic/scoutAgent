"""
Configuration Service Implementation.

This module provides a service-based wrapper around the existing configuration system,
enabling dependency injection and lifecycle management.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import asdict

from service_registry import ServiceBase, service
from config import (
    ScoutConfig, 
    ConfigManager,
    get_config as get_global_config,
    init_config
)
from custom_logging import get_logger


@service(name="config", singleton=True)
class ConfigService(ServiceBase):
    """
    Configuration service for ScoutAgent.
    
    This service manages application configuration, providing access to settings
    from environment variables and configuration files. It leverages the existing
    configuration system but makes it available as a service.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration service.
        
        Args:
            config_file: Optional path to configuration file
        """
        super().__init__(name="config", version="1.0.0")
        self.logger = get_logger("service.config")
        self.config_file = config_file
        self.config_manager = ConfigManager()
        self._config = None
    
    async def _initialize(self, registry) -> bool:
        """
        Initialize configuration from file and environment.
        
        Args:
            registry: Service registry
            
        Returns:
            True if initialization was successful
        """
        self.logger.info("Initializing configuration service")
        
        try:
            # Load configuration
            self._config = self.config_manager.load_config(self.config_file)
            
            # Set as global config
            init_config(self.config_file)
            
            self.logger.info("Configuration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration: {e}")
            return False
    
    async def _start(self) -> bool:
        """
        Start the configuration service.
        
        For the config service, start simply validates the configuration.
        
        Returns:
            True if startup was successful
        """
        self.logger.info("Starting configuration service")
        
        # Configuration is already loaded during initialization
        # Here we could add any additional runtime validation
        
        return True
    
    async def _stop(self) -> bool:
        """
        Stop the configuration service.
        
        For the config service, this handles saving any pending changes.
        
        Returns:
            True if shutdown was successful
        """
        self.logger.info("Stopping configuration service")
        return True
    
    def get_config(self) -> ScoutConfig:
        """
        Get the current configuration.
        
        Returns:
            ScoutConfig instance
        """
        if not self._config:
            raise RuntimeError("Configuration service not initialized")
        
        return self._config
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-notation path.
        
        Args:
            path: Dot-notation path to config value (e.g., "api.openai_api_key")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        if not self._config:
            raise RuntimeError("Configuration service not initialized")
        
        # Navigate the path
        value = self._config
        for key in path.split('.'):
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return default
                
        return value
    
    def update_config_value(self, path: str, value: Any) -> bool:
        """
        Update a configuration value using a dot-notation path.
        
        Args:
            path: Dot-notation path to config value (e.g., "api.openai_api_key")
            value: New value
            
        Returns:
            True if value was updated successfully
        """
        if not self._config:
            raise RuntimeError("Configuration service not initialized")
        
        try:
            # Navigate the path except last key
            keys = path.split('.')
            target = self._config
            
            for i, key in enumerate(keys[:-1]):
                if hasattr(target, key):
                    target = getattr(target, key)
                else:
                    self.logger.error(f"Invalid config path: {path}")
                    return False
            
            # Set the value
            setattr(target, keys[-1], value)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update config value: {e}")
            return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_file: Optional file path (uses current path if None)
            
        Returns:
            True if configuration was saved successfully
        """
        if not self._config:
            raise RuntimeError("Configuration service not initialized")
        
        file_path = config_file or self.config_file or "config.json"
        
        try:
            self.config_manager.save_config(self._config, file_path)
            self.logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file and environment.
        
        Returns:
            True if configuration was reloaded successfully
        """
        try:
            self._config = self.config_manager.load_config(self.config_file)
            self.logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_llm_backend_for_agent(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the preferred LLM backend configuration for a specific agent type.
        
        This method enables agent-specific LLM backends, allowing different agents
        to use different LLM providers based on their specific tasks.
        
        Args:
            agent_type: Type of agent (e.g., "code", "research", "analysis")
            
        Returns:
            Dictionary with backend configuration or None if not configured
        """
        # First check if there's a specific override for this agent type
        agent_config_path = f"agent.backend_preferences.{agent_type}"
        agent_backend = self.get_config_value(agent_config_path)
        
        if agent_backend:
            return agent_backend
        
        # Fall back to default LLM backend
        return self.get_config_value("agent.default_backend")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        if not self._config:
            raise RuntimeError("Configuration service not initialized")
            
        return asdict(self._config)


# Global config service instance for convenience
_config_service_instance: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """
    Get the global config service instance.
    
    Returns:
        ConfigService instance
    """
    global _config_service_instance
    
    if _config_service_instance is None:
        from service_registry import get_registry
        
        # Check if registered in service registry
        registry = get_registry()
        
        if registry.has_service_instance("config"):
            _config_service_instance = registry.get_service("config")
        else:
            # Create and register a new instance
            _config_service_instance = ConfigService()
            
    return _config_service_instance
