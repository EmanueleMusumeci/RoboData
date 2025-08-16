"""
Tool Usage Distribution Analyzer

This script analyzes and visualizes the distribution of tool usage across different
experimental queries. It creates stacked bar charts showing how different tools
are used in each query and provides insights into tool usage patterns.

Usage:
    From the RoboData root directory:
    . .venv/bin/activate && python -m backend.evaluation.tool_distribution
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional


def load_tool_distribution_data(base_dir: str = "experiments") -> Dict[str, Dict[str, int]]:
    """
    Load tool usage distribution data from experiment result files.
    
    Args:
        base_dir: Directory containing experiment results
        
    Returns:
        Dictionary mapping query IDs to their tool call counts
    """
    tool_calls_per_query = {}
    
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory '{base_dir}' does not exist")
        return tool_calls_per_query
    
    print(f"Loading tool distribution data from: {base_dir}")
    
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        print(f"    Processing experiment directory: {exp_dir}")
        
        for file in os.listdir(exp_path):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(exp_path, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    query_id = data.get("experiment_id", exp_dir)
                    tools = data.get("tool_statistics", {})
                    
                    for tool, stats in tools.items():
                        if isinstance(stats, dict) and "call_count" in stats:
                            tool_calls_per_query.setdefault(query_id, {})[tool] = stats["call_count"]
                        else:
                            print(f"        Warning: Invalid tool stats format for {tool} in {file}")
                            
                except (json.JSONDecodeError, IOError) as e:
                    print(f"        Error reading file {file}: {e}")
    
    print(f"Successfully loaded data for {len(tool_calls_per_query)} queries")
    return tool_calls_per_query


def create_distribution_plot(tool_calls_per_query: Dict[str, Dict[str, int]], 
                           output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create a stacked bar chart showing tool usage distribution per query.
    
    Args:
        tool_calls_per_query: Dictionary mapping query IDs to tool call counts
        output_file: Optional path to save the plot image
        
    Returns:
        DataFrame containing the tool distribution data
    """
    if not tool_calls_per_query:
        print("No tool distribution data available to plot")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(tool_calls_per_query, orient="index").fillna(0).astype(int)
    
    if df.empty:
        print("No valid tool distribution data found")
        return df
    
    print(f"Creating distribution plot with {len(df)} queries and {len(df.columns)} tools")
    
    # Create the stacked bar chart
    plt.figure(figsize=(max(14, len(df) * 0.8), 6))
    
    df.plot(kind="bar", stacked=True, figsize=(max(14, len(df) * 0.8), 6), colormap="tab20")
    plt.ylabel("Tool Calls", fontsize=12)
    plt.title("Tool Usage Distribution per Query", fontsize=14, fontweight='bold')
    plt.xlabel("Query ID", fontsize=12)
    plt.legend(title="Tool", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {output_file}")
    
    plt.show()
    
    return df


def analyze_distribution_statistics(df: pd.DataFrame) -> None:
    """
    Analyze and print distribution statistics for tool usage.
    
    Args:
        df: DataFrame containing tool usage distribution data
    """
    if df.empty:
        print("No data available for distribution analysis")
        return
    
    print("\n" + "="*60)
    print("TOOL USAGE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"Number of queries analyzed: {len(df)}")
    print(f"Number of tools analyzed: {len(df.columns)}")
    print(f"Total tool calls across all queries: {df.sum().sum()}")
    
    # Calculate total calls per tool across all queries
    tool_totals = df.sum(axis=0).sort_values(ascending=False)
    print("\nTotal tool usage across all queries:")
    for tool, total in tool_totals.items():
        percentage = (total / df.sum().sum()) * 100
        print(f"    {tool}: {total} calls ({percentage:.1f}%)")
    
    # Find queries with highest and lowest tool diversity
    tool_diversity = (df > 0).sum(axis=1)
    most_diverse_query = tool_diversity.idxmax()
    least_diverse_query = tool_diversity.idxmin()
    
    print(f"\nTool diversity per query:")
    print(f"    Most diverse: {most_diverse_query} ({tool_diversity[most_diverse_query]} different tools)")
    print(f"    Least diverse: {least_diverse_query} ({tool_diversity[least_diverse_query]} different tools)")
    print(f"    Average diversity: {tool_diversity.mean():.1f} tools per query")
    
    # Find queries with highest and lowest total tool usage
    query_totals = df.sum(axis=1).sort_values(ascending=False)
    print(f"\nQueries with highest tool usage:")
    for i, (query, total) in enumerate(query_totals.head(3).items()):
        print(f"    {i+1}. {query}: {total} total calls")
    
    print(f"\nQueries with lowest tool usage:")
    for i, (query, total) in enumerate(query_totals.tail(3).items()):
        print(f"    {len(query_totals)-2+i}. {query}: {total} total calls")


def main():
    """Main function to analyze tool usage distribution."""
    print("Analyzing Tool Usage Distribution...")
    
    # Load tool distribution data
    tool_calls_per_query = load_tool_distribution_data()
    
    if not tool_calls_per_query:
        print("No tool distribution data found. Make sure experiment results exist in the 'experiments' directory.")
        return
    
    # Create distribution plot
    print("\nGenerating distribution visualization...")
    df = create_distribution_plot(tool_calls_per_query)
    
    # Analyze distribution statistics
    analyze_distribution_statistics(df)
    
    print("\nDistribution analysis complete!")


if __name__ == "__main__":
    main()
