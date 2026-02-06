"""Logging configuration for YOLO Training Pipeline"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class PipelineLogger:
    """Centralized logging for the YOLO training pipeline"""
    
    def __init__(self, log_dir: Optional[Path] = None, verbose: bool = False):
        """
        Initialize pipeline logger.
        
        Args:
            log_dir: Directory for log files (defaults to ./logs)
            verbose: Enable verbose logging
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Create loggers for different components
        self.loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self) -> None:
        """Set up loggers for each component"""
        components = [
            'dataset_manager',
            'annotation_processor',
            'configuration_manager',
            'training_engine',
            'evaluation_module',
            'pipeline_orchestrator'
        ]
        
        for component in components:
            logger = self._create_logger(component)
            self.loggers[component] = logger
    
    def _create_logger(self, component_name: str) -> logging.Logger:
        """
        Create a logger for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"yolo_pipeline.{component_name}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.handlers.clear()  # Clear existing handlers
        
        # File handler - component-specific log file
        log_file = self.log_dir / f"{component_name}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        if self.verbose:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, component_name: str) -> logging.Logger:
        """
        Get logger for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Logger instance
        """
        if component_name not in self.loggers:
            self.loggers[component_name] = self._create_logger(component_name)
        return self.loggers[component_name]
    
    def log_error(self, component: str, operation: str, error: Exception, context: Optional[Dict] = None) -> None:
        """
        Log an error with context.
        
        Args:
            component: Component name
            operation: Operation being performed
            error: Exception that occurred
            context: Additional context information
        """
        logger = self.get_logger(component)
        error_msg = f"Error in {operation}: {str(error)}"
        
        if context:
            error_msg += f" | Context: {context}"
        
        logger.error(error_msg, exc_info=self.verbose)
    
    def log_success(self, component: str, operation: str, details: str = "") -> None:
        """
        Log a successful operation.
        
        Args:
            component: Component name
            operation: Operation that succeeded
            details: Additional details
        """
        logger = self.get_logger(component)
        success_msg = f"✓ {operation} completed successfully"
        
        if details:
            success_msg += f" | {details}"
        
        logger.info(success_msg)
    
    def log_warning(self, component: str, message: str) -> None:
        """
        Log a warning.
        
        Args:
            component: Component name
            message: Warning message
        """
        logger = self.get_logger(component)
        logger.warning(message)
    
    def log_info(self, component: str, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            component: Component name
            message: Info message
        """
        logger = self.get_logger(component)
        logger.info(message)
    
    def log_debug(self, component: str, message: str) -> None:
        """
        Log a debug message (only in verbose mode).
        
        Args:
            component: Component name
            message: Debug message
        """
        logger = self.get_logger(component)
        logger.debug(message)


# Global logger instance
_global_logger: Optional[PipelineLogger] = None


def get_pipeline_logger(log_dir: Optional[Path] = None, verbose: bool = False) -> PipelineLogger:
    """
    Get or create the global pipeline logger.
    
    Args:
        log_dir: Directory for log files
        verbose: Enable verbose logging
        
    Returns:
        PipelineLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PipelineLogger(log_dir, verbose)
    
    return _global_logger


def reset_logger() -> None:
    """Reset the global logger (useful for testing)"""
    global _global_logger
    _global_logger = None
