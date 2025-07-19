"""
DEPRECATED: Formatting utilities for improved readability of debug output.

⚠️  WARNING: This module is deprecated!
⚠️  Please use backend.core.logging instead.
⚠️  The new logging system provides better functionality including:
    - File logging to experiments directory
    - Proper log levels and formatters  
    - Better separation between turns
    - Timestamps and structured output
    
    Migration guide:
    - Replace print_prompt() with log_prompt()
    - Replace print_tool_result() with log_tool_result() 
    - Replace print_tool_error() with log_tool_error()
    - Replace print_debug() with log_debug()
    - Replace print_response() with log_response()
    - Replace print_memory_entry() with log_memory_entry()
    - Replace print_tool_results_summary() with log_tool_results_summary()
    
    Set up logging with: setup_logger(experiment_id="your_experiment_id")
"""

import pprint
from typing import Any, Optional, List


class Colors:
    """ANSI color codes for terminal output."""
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'


def print_prompt(system: str, user: Optional[str] = None, assistant: Optional[str] = None, context: str = "", available_tools: Optional[List] = None):
    """Print prompt components in yellow with clear separation and pretty printing.
    
    Args:
        system (str): System prompt component (mandatory)
        user (str): User prompt component (mandatory)
        assistant (str, optional): Assistant prompt component
        context (str, optional): Context label for the prompt
        available_tools (list, optional): List of available tools to display compactly
    """
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.YELLOW}{Colors.BOLD}\t\t{prefix}PROMPT:{Colors.ENDC}")
    
    # Print system prompt
    print(f"{Colors.YELLOW}{Colors.BOLD}\t\t├── SYSTEM:{Colors.ENDC}")
    system_lines = system.strip().split('\n')
    for line in system_lines:
        print(f"{Colors.YELLOW}\t\t│   {line}{Colors.ENDC}")
    
    # Print user prompt
    if user:
        print(f"{Colors.YELLOW}{Colors.BOLD}\t\t├── USER:{Colors.ENDC}")
        user_lines = user.strip().split('\n')
        for line in user_lines:
            print(f"{Colors.YELLOW}\t\t│   {line}{Colors.ENDC}")
    
    # Print assistant prompt if provided
    if assistant:
        print(f"{Colors.YELLOW}{Colors.BOLD}\t\t├── ASSISTANT:{Colors.ENDC}")
        assistant_lines = assistant.strip().split('\n')
        for line in assistant_lines:
            print(f"{Colors.YELLOW}\t\t│   {line}{Colors.ENDC}")
    
    # Print available tools if provided
    if available_tools:
        print(f"{Colors.YELLOW}{Colors.BOLD}\t\t└── AVAILABLE TOOLS ({len(available_tools)} tools):{Colors.ENDC}")
        
        # Extract tool names for compact display
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
        
        # Pretty print tool names in a compact format
        tools_str = pprint.pformat(tool_names, width=60, compact=True)
        for line in tools_str.split('\n'):
            print(f"{Colors.YELLOW}\t\t    {line}{Colors.ENDC}")
    
    print()  # Add blank line after prompt


def print_memory_entry(message: str, role: str = "System"):
    """Print memory entries in purple with single indentation, role highlighting, and newline."""
    print(f"{Colors.PURPLE}\tMEMORY [{role}]: {message}{Colors.ENDC}\n")


def print_tool_result(tool_name: str, arguments: dict, result: Any, context: str = ""):
    """Print tool results in green with pretty printing and single indentation."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.GREEN}\t{prefix}TOOL RESULT - {tool_name}:{Colors.ENDC}")
    print(f"{Colors.GREEN}\tArguments:{Colors.ENDC}")
    
    # Pretty print arguments with indentation
    args_str = pprint.pformat(arguments, width=80, depth=None)
    for line in args_str.split('\n'):
        print(f"{Colors.GREEN}\t\t{line}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}\tResult:{Colors.ENDC}")
    
    # Pretty print result with indentation
    result_str = pprint.pformat(result, width=80, depth=None)
    for line in result_str.split('\n'):
        print(f"{Colors.GREEN}\t\t{line}{Colors.ENDC}")
    print()  # Add blank line after result


def print_tool_error(tool_name: str, arguments: dict, error: str, context: str = ""):
    """Print tool errors in red with single indentation."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.RED}\t{prefix}TOOL ERROR - {tool_name}:{Colors.ENDC}")
    print(f"{Colors.RED}\tArguments: {arguments}{Colors.ENDC}")
    print(f"{Colors.RED}\tError: {error}{Colors.ENDC}")
    print()


def print_debug(message: str, context: str = "DEBUG"):
    """Print debug messages in blue."""
    print(f"[{context}] {message}")


def print_response(response_content: str, context: str = ""):
    """Print LLM responses with proper formatting."""
    prefix = f"\t[{context.upper()}] " if context else ""
    print(f"{Colors.BLUE}\t\t{prefix}LLM RESPONSE:{Colors.ENDC}")
    lines = str(response_content).strip().split('\n')
    for line in lines:
        print(f"{Colors.BLUE}\t\t\t{line}{Colors.ENDC}")
    print()


def print_tool_results_summary(results: list, context: str = ""):
    """Print a summary of tool call results with pretty printing."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.GREEN}\t{prefix}TOOL RESULTS SUMMARY ({len(results)} tools executed):{Colors.ENDC}")
    
    # Pretty print the entire results list with indentation
    results_str = pprint.pformat(results, width=80, depth=None)
    for line in results_str.split('\n'):
        print(f"{Colors.GREEN}\t\t{line}{Colors.ENDC}")
    print()

