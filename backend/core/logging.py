"""
RoboData Logging System
======================

A comprehensive logging system that provides:
- Colored terminal output for better readability
- File logging to experiments directory
- Multiple log levels and formatters
- Clear separation between turns and components
"""

import logging
import os
import sys
import pprint
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Dict
from enum import Enum


class Colors:
    """ANSI color codes for terminal output."""
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LogLevel(Enum):
    """Custom log levels for RoboData components."""
    PROMPT = 35
    TOOL_RESULT = 25
    TOOL_ERROR = 45
    MEMORY = 15
    RESPONSE = 30
    DEBUG = 10
    TURN_SEPARATOR = 50


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to terminal output."""
    
    COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
        LogLevel.PROMPT.value: Colors.YELLOW + Colors.BOLD,
        LogLevel.TOOL_RESULT.value: Colors.GREEN,
        LogLevel.TOOL_ERROR.value: Colors.RED,
        LogLevel.MEMORY.value: Colors.PURPLE,
        LogLevel.RESPONSE.value: Colors.BLUE,
        LogLevel.DEBUG.value: Colors.GRAY,
        LogLevel.TURN_SEPARATOR.value: Colors.CYAN + Colors.BOLD,
    }
    
    def format(self, record):
        # Get color for this log level
        color = self.COLORS.get(record.levelno, Colors.WHITE)
        
        # Format the message
        formatted = super().format(record)
        
        # Add color if this is a terminal output
        if hasattr(record, 'add_color') and getattr(record, 'add_color', False):
            formatted = f"{color}{formatted}{Colors.ENDC}"
            
        return formatted


class FileFormatter(logging.Formatter):
    """Custom formatter for file output without colors but with better structure."""
    
    def format(self, record):
        # Add extra spacing and structure for file logs
        formatted = super().format(record)
        
        # Add extra formatting for special log types
        if record.levelno == LogLevel.TURN_SEPARATOR.value:
            separator = "=" * 80
            formatted = f"\n{separator}\n{formatted}\n{separator}\n"
        elif record.levelno == LogLevel.PROMPT.value:
            formatted = f"\n{'─' * 40} PROMPT {'─' * 40}\n{formatted}\n{'─' * 87}\n"
        elif record.levelno == LogLevel.TOOL_RESULT.value:
            formatted = f"\n🔧 {formatted}\n"
        elif record.levelno == LogLevel.TOOL_ERROR.value:
            formatted = f"\n❌ {formatted}\n"
        elif record.levelno == LogLevel.MEMORY.value:
            formatted = f"\n🧠 {formatted}\n"
        elif record.levelno == LogLevel.RESPONSE.value:
            formatted = f"\n🤖 {formatted}\n"
            
        return formatted


class RoboDataLogger:
    """Main logger class for RoboData system."""
    
    def __init__(self, experiment_id: Optional[str] = None, log_dir: Optional[Path] = None):
        """
        Initialize the RoboData logger.
        
        Args:
            experiment_id: Unique identifier for the experiment (defaults to timestamp)
            log_dir: Directory to store log files (defaults to RoboData/experiments/{experiment_id})
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_id = experiment_id
        
        # Set up log directory
        if log_dir is None:
            # Default to RoboData/experiments/{experiment_id}
            # Path calculation: backend/core/logging.py -> backend/core -> backend -> RoboData
            robodata_root = Path(__file__).parent.parent.parent
            self.log_dir = robodata_root / "experiments" / experiment_id
        else:
            self.log_dir = Path(log_dir)
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('robodata')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Add custom log levels
        for level in LogLevel:
            logging.addLevelName(level.value, level.name)
        
        # Set up handlers
        self._setup_handlers()
        
        # Log initialization
        self.logger.info(f"RoboData Logger initialized - Experiment ID: {experiment_id}")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def _setup_handlers(self):
        """Set up console and file handlers with appropriate formatters."""
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Console formatter with colors and compact format
        console_formatter = ColoredFormatter(
            fmt='%(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for detailed logs
        detailed_log_file = self.log_dir / f"{self.experiment_id}_detailed.log"
        file_handler = logging.FileHandler(detailed_log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # File formatter with timestamps and structure
        file_formatter = FileFormatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Mark console records for coloring
        def add_color_filter(record):
            record.add_color = True
            return True
        console_handler.addFilter(add_color_filter)
    
    def turn_separator(self, turn_number: int, context: str = ""):
        """Log a turn separator for better readability."""
        prefix = f"[{context.upper()}] " if context else ""
        message = f"{prefix}TURN {turn_number}"
        self.logger.log(LogLevel.TURN_SEPARATOR.value, message)
    
    def prompt(self, system: str, user: Optional[str] = None, assistant: Optional[str] = None, 
               context: str = "", available_tools: Optional[List] = None):
        """Log prompt components with clear formatting."""
        prefix = f"[{context.upper()}] " if context else ""
        
        # Build prompt message
        message_parts = [f"{prefix}PROMPT:"]
        
        # System prompt
        message_parts.append("├── SYSTEM:")
        for line in system.strip().split('\n'):
            message_parts.append(f"│   {line}")
        
        # User prompt
        if user:
            message_parts.append("├── USER:")
            for line in user.strip().split('\n'):
                message_parts.append(f"│   {line}")
        
        # Assistant prompt
        if assistant:
            message_parts.append("├── ASSISTANT:")
            for line in assistant.strip().split('\n'):
                message_parts.append(f"│   {line}")
        
        # Available tools
        if available_tools:
            message_parts.append(f"└── AVAILABLE TOOLS ({len(available_tools)} tools):")
            
            # Extract tool names
            tool_names = []
            for tool in available_tools:
                if isinstance(tool, dict):
                    if 'function' in tool and 'name' in tool['function']:
                        tool_names.append(tool['function']['name'])
                    elif 'name' in tool:
                        tool_names.append(tool['name'])
                    else:
                        tool_names.append(str(tool))
                else:
                    tool_names.append(str(tool))
            
            # Format tool names compactly
            tools_str = pprint.pformat(tool_names, width=60, compact=True)
            for line in tools_str.split('\n'):
                message_parts.append(f"    {line}")
        
        message = '\n'.join(message_parts)
        self.logger.log(LogLevel.PROMPT.value, message)
    
    def memory_entry(self, message: str, role: str = "System"):
        """Log memory entries."""
        formatted_message = f"MEMORY [{role}]: {message}"
        self.logger.log(LogLevel.MEMORY.value, formatted_message)
    
    def tool_result(self, tool_name: str, arguments: dict, result: Any, context: str = ""):
        """Log tool execution results."""
        prefix = f"[{context.upper()}] " if context else ""
        
        message_parts = [f"{prefix}TOOL RESULT - {tool_name}:"]
        message_parts.append("Arguments:")
        
        # Pretty print arguments
        args_str = pprint.pformat(arguments, width=80, depth=None)
        for line in args_str.split('\n'):
            message_parts.append(f"    {line}")
        
        message_parts.append("Result:")
        
        # Pretty print result
        result_str = pprint.pformat(result, width=80, depth=None)
        for line in result_str.split('\n'):
            message_parts.append(f"    {line}")
        
        message = '\n'.join(message_parts)
        self.logger.log(LogLevel.TOOL_RESULT.value, message)
    
    def tool_error(self, tool_name: str, arguments: dict, error: str, context: str = ""):
        """Log tool execution errors."""
        prefix = f"[{context.upper()}] " if context else ""
        message = f"{prefix}TOOL ERROR - {tool_name}:\nArguments: {arguments}\nError: {error}"
        self.logger.log(LogLevel.TOOL_ERROR.value, message)
    
    def response(self, response_content: str, context: str = ""):
        """Log LLM responses."""
        prefix = f"[{context.upper()}] " if context else ""
        message_parts = [f"{prefix}LLM RESPONSE:"]
        
        for line in str(response_content).strip().split('\n'):
            message_parts.append(f"    {line}")
        
        message = '\n'.join(message_parts)
        self.logger.log(LogLevel.RESPONSE.value, message)
    
    def debug(self, message: str, context: str = "DEBUG"):
        """Log debug messages."""
        formatted_message = f"[{context}] {message}"
        self.logger.log(LogLevel.DEBUG.value, formatted_message)
    
    def info(self, message: str):
        """Log info messages."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning messages."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error messages."""
        self.logger.error(message)
    
    def tool_results_summary(self, results: list, context: str = ""):
        """Log a summary of tool call results."""
        prefix = f"[{context.upper()}] " if context else ""
        
        # For debug mode, only show last 2 results
        debug_results = results[-2:] if len(results) > 2 else results
        
        message_parts = [f"{prefix}TOOL RESULTS SUMMARY ({len(results)} tools executed, showing last {len(debug_results)}):"]
        
        # Pretty print results
        results_str = pprint.pformat(debug_results, width=80, depth=None)
        for line in results_str.split('\n'):
            message_parts.append(f"    {line}")
        
        message = '\n'.join(message_parts)
        self.logger.log(LogLevel.DEBUG.value, message)


# Global logger instance
_global_logger: Optional[RoboDataLogger] = None


def get_logger() -> RoboDataLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = RoboDataLogger()
    return _global_logger


def setup_logger(experiment_id: Optional[str] = None, log_dir: Optional[str] = None) -> RoboDataLogger:
    """Set up and return the global logger instance."""
    global _global_logger
    log_dir_path = Path(log_dir) if log_dir else None
    _global_logger = RoboDataLogger(experiment_id=experiment_id, log_dir=log_dir_path)
    return _global_logger


# Convenience functions for backward compatibility and easy use
def log_prompt(system: str, user: Optional[str] = None, assistant: Optional[str] = None, 
               context: str = "", available_tools: Optional[List] = None):
    """Log prompt components."""
    get_logger().prompt(system, user, assistant, context, available_tools)


def log_memory_entry(message: str, role: str = "System"):
    """Log memory entries."""
    get_logger().memory_entry(message, role)


def log_tool_result(tool_name: str, arguments: dict, result: Any, context: str = ""):
    """Log tool results."""
    get_logger().tool_result(tool_name, arguments, result, context)


def log_tool_error(tool_name: str, arguments: dict, error: str, context: str = ""):
    """Log tool errors."""
    get_logger().tool_error(tool_name, arguments, error, context)


def log_response(response_content: str, context: str = ""):
    """Log LLM responses."""
    get_logger().response(response_content, context)


def log_debug(message: str, context: str = "DEBUG"):
    """Log debug messages."""
    get_logger().debug(message, context)


def log_tool_results_summary(results: list, context: str = ""):
    """Log tool results summary."""
    get_logger().tool_results_summary(results, context)


def log_turn_separator(turn_number: int, context: str = ""):
    """Log turn separator."""
    get_logger().turn_separator(turn_number, context)