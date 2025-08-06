"""
Logging Service Implementation.

This module provides a service-based wrapper around the existing logging system,
enabling dependency injection and lifecycle management for logging.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

from service_registry import ServiceBase, service, requires
from custom_logging import get_logger as get_original_logger, setup_logger
import logging
from contextlib import contextmanager
from custom_logging import logger as custom_logger_module

@service(name="logging", singleton=True)
@requires("config")
class LoggingService(ServiceBase):
    """
    Logging service for ScoutAgent.
    
    This service manages application logging, providing standardized logging
    capabilities throughout the application. It leverages the existing
    logging system but makes it available as a service with enhanced features.
    """
    
    def __init__(self):
        """Initialize the logging service."""
        super().__init__(name="logging", version="1.0.0")
        self.logger = get_original_logger("service.logging")
        self._loggers = {}
        self._log_dir = None
        self._log_level = None
        self._trace_dir = None
        self._enable_tracing = None
        self._file_handlers = {}
    
    async def _initialize(self, registry) -> bool:
        """
        Initialize logging with configuration.
        
        Args:
            registry: Service registry for accessing dependencies
            
        Returns:
            True if initialization was successful
        """
        self.logger.info("Initializing logging service")
        
        try:
            # Get config service
            config_service = registry.get_service("config")
            config = config_service.get_config()
            
            # Extract logging configuration
            self._log_dir = Path(config.logging.log_dir).resolve()
            self._log_level = config.logging.level
            self._trace_dir = Path(config.logging.trace_dir).resolve()
            self._enable_tracing = config.logging.enable_tracing
            self._max_file_size = config.logging.max_file_size
            self._backup_count = config.logging.backup_count
            
            # Ensure log directories exist
            self._log_dir.mkdir(parents=True, exist_ok=True)
            if self._enable_tracing:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                
            self.logger.info(f"Logging initialized with level={self._log_level}, dir={self._log_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize logging: {e}")
            return False
    
    async def _start(self) -> bool:
        """
        Start the logging service.
        
        Sets up global logging configuration and file handlers.
        
        Returns:
            True if startup was successful
        """
        self.logger.info("Starting logging service")
        
        try:
            # Set up global logging
            log_file = self._log_dir / "scout_agent.log"
            setup_logger(self._log_level, str(log_file))
            
            self.logger.info("Logging service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start logging service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """
        Stop the logging service.
        
        Closes all file handlers and performs cleanup.
        
        Returns:
            True if shutdown was successful
        """
        self.logger.info("Stopping logging service")
        
        # Close all file handlers
        for name, handlers in self._file_handlers.items():
            for handler in handlers:
                try:
                    handler.close()
                    logger = logging.getLogger(name)
                    logger.removeHandler(handler)
                except Exception as e:
                    self.logger.error(f"Error closing handler for {name}: {e}")
        
        self._file_handlers.clear()
        return True
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name: Logger name (usually module name)
            level: Optional log level (overrides default)
            
        Returns:
            Configured logger instance
        """
        # Use the existing get_logger function from custom_logging
        logger = get_original_logger(name)
        
        # Set custom level if specified
        if level:
            try:
                numeric_level = getattr(logging, level.upper())
                logger.setLevel(numeric_level)
            except (AttributeError, TypeError):
                logger.warning(f"Invalid log level: {level}, using default")
        
        # Keep track of loggers we've created
        self._loggers[name] = logger
        
        return logger
    
    def set_log_level(self, level: str, logger_name: Optional[str] = None) -> bool:
        """
        Set the log level for a logger or all loggers.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            logger_name: Optional logger name (affects all loggers if None)
            
        Returns:
            True if successful
        """
        try:
            numeric_level = getattr(logging, level.upper())
            
            if logger_name:
                # Set for specific logger
                if logger_name in self._loggers:
                    self._loggers[logger_name].setLevel(numeric_level)
                else:
                    # Create it if it doesn't exist
                    logger = logging.getLogger(logger_name)
                    logger.setLevel(numeric_level)
                    self._loggers[logger_name] = logger
            else:
                # Set for root logger and all existing loggers
                logging.getLogger().setLevel(numeric_level)
                for name, logger in self._loggers.items():
                    logger.setLevel(numeric_level)
                
                # Update default level
                self._log_level = level.upper()
                
            return True
            
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Invalid log level: {level} - {e}")
            return False
    
    def add_file_handler(self, 
                        logger_name: str, 
                        file_name: str, 
                        level: Optional[str] = None) -> bool:
        """
        Add a file handler to a logger.
        
        Args:
            logger_name: Logger name
            file_name: Log file name (will be placed in log directory)
            level: Optional log level for this handler
            
        Returns:
            True if successful
        """
        try:
            # Get or create logger
            if logger_name in self._loggers:
                logger = self._loggers[logger_name]
            else:
                logger = self.get_logger(logger_name)
            
            # Create file path
            log_path = self._log_dir / file_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create handler
            handler = logging.FileHandler(str(log_path))
            
            # Set level
            if level:
                handler.setLevel(getattr(logging, level.upper()))
            else:
                handler.setLevel(getattr(logging, self._log_level))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(handler)
            
            # Store handler for cleanup
            if logger_name not in self._file_handlers:
                self._file_handlers[logger_name] = []
            self._file_handlers[logger_name].append(handler)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add file handler: {e}")
            return False
    
    @contextmanager
    def create_context(self, logger_name: str, level: str):
        """
        Create a logging context manager for temporary logging level.
        
        Args:
            logger_name: Logger name
            level: Log level to use within context
            
        Returns:
            Context manager for temporarily changing log level
        """
        # Get the logger
        logger = self.get_logger(logger_name)
        
        # Store the current level
        old_level = logger.level
        
        # Set the new level
        try:
            new_level = getattr(logging, level.upper())
            logger.setLevel(new_level)
            yield logger
        finally:
            # Restore the old level
            logger.setLevel(old_level)
    
    def get_log_dir(self) -> Path:
        """
        Get the current log directory.
        
        Returns:
            Path to log directory
        """
        return self._log_dir
    
    def get_log_files(self) -> List[Path]:
        """
        Get all log files in the log directory.
        
        Returns:
            List of log file paths
        """
        return list(self._log_dir.glob("*.log"))
    
    def get_log_content(self, log_file: str, max_lines: int = 100) -> List[str]:
        """
        Get the content of a log file.
        
        Args:
            log_file: Name of log file (not full path)
            max_lines: Maximum number of lines to return
            
        Returns:
            List of log lines
        """
        log_path = self._log_dir / log_file
        
        if not log_path.exists():
            self.logger.error(f"Log file not found: {log_file}")
            return []
            
        try:
            # Read the last max_lines lines
            with open(log_path, "r") as f:
                lines = f.readlines()
                return lines[-max_lines:] if len(lines) > max_lines else lines
                
        except Exception as e:
            self.logger.error(f"Failed to read log file {log_file}: {e}")
            return []


# Global logging service instance for convenience
_logging_service_instance: Optional[LoggingService] = None


def get_logging_service() -> LoggingService:
    """
    Get the global logging service instance.
    
    Returns:
        LoggingService instance
    """
    global _logging_service_instance
    
    if _logging_service_instance is None:
        from service_registry import get_registry
        
        # Check if registered in service registry
        registry = get_registry()
        
        if registry.has_service_instance("logging"):
            _logging_service_instance = registry.get_service("logging")
        else:
            # Create and register a new instance
            _logging_service_instance = LoggingService()
            
    return _logging_service_instance


def get_service_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger from the logging service.
    
    This is a convenience function for code that doesn't want to directly
    interact with the service registry.
    
    Args:
        name: Logger name
        level: Optional log level
        
    Returns:
        Configured logger instance
    """
    try:
        service = get_logging_service()
        return service.get_logger(name, level)
    except Exception:
        # Fall back to original logger if service not available
        return get_original_logger(name, level)
