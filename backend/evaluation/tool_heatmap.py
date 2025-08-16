"""
Tool Usage Heatmap Generator

This script generates a heatmap visualization showing tool usage statistics
across different experimental queries. It analyzes experiment result files
to extract tool call counts and creates an informative heatmap visualization.

The script automatically searches through multiple experiment directories:
- experiments/ (including batch subdirectories)
- experiments_GPT_5/
- experiments_old/

Usage:
    From the RoboData root directory:
    . .venv/bin/activate && python -m backend.evaluation.tool_heatmap
    
    To specify custom directories:
    . .venv/bin/activate && python -m backend.evaluation.tool_heatmap --dirs experiments experiments_GPT_5
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional, List


def load_experiment_data(base_dirs: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
    """
    Load tool usage data from experiment result files across multiple directories.
    
    Args:
        base_dirs: List of directories containing experiment results.
                   If None, defaults to common experiment directories.
        
    Returns:
        Dictionary mapping tool names to query IDs and their call counts
    """
    if base_dirs is None:
        base_dirs = ["experiments", "experiments_GPT_5", "experiments_old"]
    
    tool_usage = {}
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Base directory '{base_dir}' does not exist")
            continue
            
        print(f"Processing base directory: {base_dir}")
        _process_directory_recursive(base_dir, tool_usage)
    
    return tool_usage


def _process_directory_recursive(directory: str, tool_usage: Dict[str, Dict[str, int]]) -> None:
    """
    Recursively process a directory to find and extract tool usage statistics.
    
    Args:
        directory: Directory path to process
        tool_usage: Dictionary to update with tool usage data
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isdir(item_path):
            # Check if this directory contains statistics files directly
            has_statistics = any(
                f.startswith("statistics_") and f.endswith(".json") 
                for f in os.listdir(item_path) 
                if os.path.isfile(os.path.join(item_path, f))
            )
            
            if has_statistics:
                print(f"    Processing experiment directory: {item_path}")
                _extract_statistics_from_directory(item_path, tool_usage)
            else:
                # Recursively process subdirectories (e.g., batch folders)
                _process_directory_recursive(item_path, tool_usage)


def _extract_statistics_from_directory(exp_path: str, tool_usage: Dict[str, Dict[str, int]]) -> None:
    """
    Extract statistics from all JSON files in a given experiment directory.
    
    Args:
        exp_path: Path to experiment directory
        tool_usage: Dictionary to update with tool usage data
    """
    exp_dir = os.path.basename(exp_path)
    
    for file in os.listdir(exp_path):
        if file.startswith("statistics_") and file.endswith(".json"):
            try:
                with open(os.path.join(exp_path, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                query_id = data.get("experiment_id", exp_dir)
                tools = data.get("tool_statistics", {})
                
                for tool, stats in tools.items():
                    if isinstance(stats, dict) and "call_count" in stats:
                        tool_usage.setdefault(tool, {})[query_id] = stats["call_count"]
                    else:
                        print(f"        Warning: Invalid tool stats format for {tool} in {file}")
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"        Error reading file {file}: {e}")


def create_heatmap(tool_usage: Dict[str, Dict[str, int]], output_file: Optional[str] = None) -> None:
    """
    Create and display a heatmap of tool usage statistics.
    
    Args:
        tool_usage: Dictionary mapping tools to query usage counts
        output_file: Optional path to save the heatmap image
    """
    if not tool_usage:
        print("No tool usage data available to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(tool_usage).fillna(0).astype(int).T
    
    if df.empty:
        print("No valid tool usage data found")
        return
    
    print(f"Creating heatmap with {len(df)} tools and {len(df.columns)} queries")
    
    # Create the heatmap
    plt.figure(figsize=(max(14, len(df.columns) * 0.8), max(6, len(df) * 0.4)))
    
    # Use a color map that's accessible and informative
    sns.heatmap(
        df, 
        annot=True, 
        cmap="YlGnBu", 
        cbar=True, 
        linewidths=0.5,
        fmt='d'  # Integer format for annotations
    )
    
    plt.title("Tool Usage per Query", fontsize=16, fontweight='bold')
    plt.xlabel("Query ID", fontsize=12)
    plt.ylabel("Tool", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")
    
    plt.show()


def print_summary_statistics(tool_usage: Dict[str, Dict[str, int]]) -> None:
    """
    Print summary statistics about tool usage.
    
    Args:
        tool_usage: Dictionary mapping tools to query usage counts
    """
    if not tool_usage:
        print("No tool usage data available for summary")
        return
    
    print("\n" + "="*50)
    print("TOOL USAGE SUMMARY STATISTICS")
    print("="*50)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(tool_usage).fillna(0).astype(int).T
    
    print(f"Total number of tools analyzed: {len(df)}")
    print(f"Total number of queries analyzed: {len(df.columns)}")
    print(f"Total tool calls across all experiments: {df.sum().sum()}")
    
    print("\nMost used tools (by total calls):")
    tool_totals = df.sum(axis=1).sort_values(ascending=False)
    for i, (tool, count) in enumerate(tool_totals.head(5).items()):
        print(f"    {i+1}. {tool}: {count} calls")
    
    print("\nQueries with most tool usage:")
    query_totals = df.sum(axis=0).sort_values(ascending=False)
    for i, (query, count) in enumerate(query_totals.head(5).items()):
        print(f"    {i+1}. {query}: {count} total calls")


def main():
    """Main function to generate tool usage heatmap."""
    parser = argparse.ArgumentParser(
        description="Generate tool usage heatmap from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default directories (experiments/, experiments_GPT_5/, experiments_old/)
    python -m backend.evaluation.tool_heatmap
    
    # Specify custom directories
    python -m backend.evaluation.tool_heatmap --dirs experiments experiments_GPT_5
    
    # Analyze only one directory
    python -m backend.evaluation.tool_heatmap --dirs experiments_old
        """
    )
    
    parser.add_argument(
        '--dirs', 
        nargs='*', 
        default=None,
        help='Experiment directories to analyze (default: experiments, experiments_GPT_5, experiments_old)'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        default=None,
        help='Optional path to save the heatmap image'
    )
    
    args = parser.parse_args()
    
    print("Generating Tool Usage Heatmap...")
    
    if args.dirs:
        print(f"Loading experiment data from specified directories: {args.dirs}")
        tool_usage = load_experiment_data(args.dirs)
    else:
        print("Loading experiment data from default directories...")
        tool_usage = load_experiment_data()
    
    if not tool_usage:
        print("No tool usage data found. Make sure experiment results exist in the specified directories.")
        if not args.dirs:
            print("Default directories checked: experiments/, experiments_GPT_5/, experiments_old/")
        return
    
    # Print summary statistics
    print_summary_statistics(tool_usage)
    
    # Create and display heatmap
    print("\nGenerating heatmap visualization...")
    create_heatmap(tool_usage, args.output)
    
    print("Heatmap generation complete!")


if __name__ == "__main__":
    main()
