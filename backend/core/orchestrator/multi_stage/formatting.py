"""Formatting utilities for improved readability of debug output."""

import pprint
from typing import Any


class Colors:
    """ANSI color codes for terminal output."""
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    ENDC = '\033[0m'  # End color
    BOLD = '\033[1m'


def print_prompt(prompt: str, context: str = ""):
    """Print prompts in yellow with double indentation."""
    prefix = f"[{context.upper()}] " if context else ""
    lines = prompt.strip().split('\n')
    print(f"{Colors.YELLOW}{Colors.BOLD}    {prefix}PROMPT:{Colors.ENDC}")
    for line in lines:
        print(f"{Colors.YELLOW}        {line}{Colors.ENDC}")
    print()  # Add blank line after prompt


def print_memory_entry(message: str):
    """Print memory entries in purple with single indentation and newline."""
    print(f"\n{Colors.PURPLE}    MEMORY: {message}{Colors.ENDC}")


def print_tool_result(tool_name: str, arguments: dict, result: Any, context: str = ""):
    """Print tool results in green with pretty printing and single indentation."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.GREEN}    {prefix}TOOL RESULT - {tool_name}:{Colors.ENDC}")
    print(f"{Colors.GREEN}    Arguments:{Colors.ENDC}")
    
    # Pretty print arguments with indentation
    args_str = pprint.pformat(arguments, width=80, depth=None)
    for line in args_str.split('\n'):
        print(f"{Colors.GREEN}        {line}{Colors.ENDC}")
    
    print(f"{Colors.GREEN}    Result:{Colors.ENDC}")
    
    # Pretty print result with indentation
    result_str = pprint.pformat(result, width=80, depth=None)
    for line in result_str.split('\n'):
        print(f"{Colors.GREEN}        {line}{Colors.ENDC}")
    print()  # Add blank line after result


def print_tool_error(tool_name: str, arguments: dict, error: str, context: str = ""):
    """Print tool errors in red with single indentation."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.RED}    {prefix}TOOL ERROR - {tool_name}:{Colors.ENDC}")
    print(f"{Colors.RED}    Arguments: {arguments}{Colors.ENDC}")
    print(f"{Colors.RED}    Error: {error}{Colors.ENDC}")
    print()


def print_debug(message: str, context: str = "DEBUG"):
    """Print debug messages in blue."""
    print(f"{Colors.BLUE}[{context}] {message}{Colors.ENDC}")


def print_response(response_content: str, context: str = ""):
    """Print LLM responses with proper formatting."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.BLUE}    {prefix}LLM RESPONSE:{Colors.ENDC}")
    lines = str(response_content).strip().split('\n')
    for line in lines:
        print(f"{Colors.BLUE}        {line}{Colors.ENDC}")
    print()


def print_tool_results_summary(results: list, context: str = ""):
    """Print a summary of tool call results with pretty printing."""
    prefix = f"[{context.upper()}] " if context else ""
    print(f"{Colors.GREEN}    {prefix}TOOL RESULTS SUMMARY ({len(results)} tools executed):{Colors.ENDC}")
    
    # Pretty print the entire results list with indentation
    results_str = pprint.pformat(results, width=80, depth=None)
    for line in results_str.split('\n'):
        print(f"{Colors.GREEN}        {line}{Colors.ENDC}")
    print()
