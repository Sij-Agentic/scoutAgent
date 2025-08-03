"""
Unified structured logger setup for ScoutAgent.
Provides file + stdout logging with configurable levels and formatting.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ScoutLogger:
    """Centralized logging configuration for ScoutAgent."""
    
    def __init__(self, 
                 name: str = "scout_agent",
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize the ScoutAgent logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (defaults to ./logs)
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
            
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up file and console handlers."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler for errors only
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self.logger
    
    def set_level(self, level: str):
        """Change logging level at runtime."""
        new_level = getattr(logging, level.upper())
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            handler.setLevel(new_level)


# Global logger instance
_global_logger = None


def get_logger(name: str = "scout_agent") -> logging.Logger:
    """
    Get the global logger instance.
    
    Args:
        name: Logger name to retrieve
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = ScoutLogger(name).get_logger()
    return _global_logger


def setup_logger(config: dict = None) -> logging.Logger:
    """
    Setup logger with configuration.
    
    Args:
        config: Configuration dictionary with keys:
               - log_level: str (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               - log_dir: str (directory path)
               - max_file_size: int (bytes)
               - backup_count: int
               
    Returns:
        Configured logger instance
    """
    if config is None:
        config = {}
    
    logger_config = ScoutLogger(
        name=config.get("name", "scout_agent"),
        log_level=config.get("log_level", "INFO"),
        log_dir=config.get("log_dir"),
        max_file_size=config.get("max_file_size", 10 * 1024 * 1024),
        backup_count=config.get("backup_count", 5)
    )
    
    return logger_config.get_logger()


# Convenience functions for direct usage
def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """Log exception with traceback."""
    get_logger().exception(msg, *args, **kwargs)


if __name__ == "__main__":
    # Test the logging system
    logger = get_logger()
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    try:
        1/0
    except ZeroDivisionError:
        logger.exception("An exception occurred")
    
    print("Logging test completed. Check the logs directory for output.")
