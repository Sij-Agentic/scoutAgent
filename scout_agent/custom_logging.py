"""
Custom logging configuration for ScoutAgent.
Provides structured logging with different levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
    
    # Set level
    if level is None:
        level = "INFO"  # Default level
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup global logging configuration.
    
    Args:
        log_level: Default log level
        log_file: Optional log file path
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class LogContext:
    """Context manager for temporary logging context."""
    
    def __init__(self, logger_name: str, level: str):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = getattr(logging, level.upper())
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
