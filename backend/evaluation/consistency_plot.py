"""
Tool Usage Consistency Analyzer

This script analyzes tool usage consistency across multiple runs of the same query.
It generates visualizations showing how tool call counts vary between runs and
provides statistical summaries of the consistency patterns.

Usage:
    From the RoboData root directory:
    . .venv/bin/activate && python -m backend.evaluation.consistency_plot
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional


def load_consistency_data(consistency_dir: str) -> List[Dict[str, int]]:
    """
    Load tool usage data from consistency experiment files.
    
    Args:
        consistency_dir: Directory containing consistency experiment results
        
    Returns:
        List of dictionaries containing tool call counts for each run
    """
    runs = []
    
    if not os.path.exists(consistency_dir):
        print(f"Warning: Consistency directory '{consistency_dir}' does not exist")
        return runs
    
    print(f"Loading consistency data from: {consistency_dir}")
    
    for file in os.listdir(consistency_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(consistency_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                tool_stats = data.get("tool_statistics", {})
                run_data = {
                    tool: stats["call_count"] 
                    for tool, stats in tool_stats.items() 
                    if isinstance(stats, dict) and "call_count" in stats
                }
                
                if run_data:  # Only add non-empty runs
                    runs.append(run_data)
                    print(f"    Loaded run from: {file}")
                else:
                    print(f"    Warning: No valid tool statistics found in {file}")
                    
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"    Error reading file {file}: {e}")
    
    print(f"Successfully loaded {len(runs)} consistency runs")
    return runs


def create_consistency_plot(runs: List[Dict[str, int]], 
                          title: str = "Tool Calls Across Consistency Runs",
                          output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create a grouped bar chart showing tool usage consistency across runs.
    
    Args:
        runs: List of dictionaries containing tool call counts for each run
        title: Title for the plot
        output_file: Optional path to save the plot image
        
    Returns:
        DataFrame containing the tool usage data
    """
    if not runs:
        print("No consistency data available to plot")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(runs).fillna(0).astype(int)
    df.index.name = "Run"
    
    if df.empty:
        print("No valid tool usage data found")
        return df
    
    print(f"Creating consistency plot with {len(df)} runs and {len(df.columns)} tools")
    
    # Create the grouped bar chart
    plt.figure(figsize=(max(10, len(df) * 0.8), 6))
    
    df.plot(kind="bar", figsize=(max(10, len(df) * 0.8), 6))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Number of Calls", fontsize=12)
    plt.xlabel("Run", fontsize=12)
    plt.legend(title="Tool", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Consistency plot saved to: {output_file}")
    
    plt.show()
    
    return df


def analyze_consistency_statistics(df: pd.DataFrame) -> None:
    """
    Analyze and print consistency statistics for tool usage.
    
    Args:
        df: DataFrame containing tool usage data across runs
    """
    if df.empty:
        print("No data available for consistency analysis")
        return
    
    print("\n" + "="*60)
    print("TOOL USAGE CONSISTENCY ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    avg_calls = df.mean()
    std_calls = df.std()
    min_calls = df.min()
    max_calls = df.max()
    
    print(f"Number of runs analyzed: {len(df)}")
    print(f"Number of tools analyzed: {len(df.columns)}")
    
    print("\nAverage tool calls per run:")
    for tool, avg in avg_calls.items():
        print(f"    {tool}: {avg:.2f} calls")
    
    print("\nTool usage consistency (standard deviation):")
    for tool, std in std_calls.items():
        consistency_score = "High" if std < 1 else "Medium" if std < 2 else "Low"
        print(f"    {tool}: σ={std:.2f} ({consistency_score} consistency)")
    
    print("\nTool usage range (min-max):")
    for tool in df.columns:
        print(f"    {tool}: {min_calls[tool]}-{max_calls[tool]} calls")
    
    # Identify most and least consistent tools
    most_consistent = std_calls.idxmin()
    least_consistent = std_calls.idxmax()
    
    print(f"\nMost consistent tool: {most_consistent} (σ={std_calls[most_consistent]:.2f})")
    print(f"Least consistent tool: {least_consistent} (σ={std_calls[least_consistent]:.2f})")


def main():
    """Main function to analyze tool usage consistency."""
    consistency_dir = "experiments/QA10_consistency"
    
    print("Analyzing Tool Usage Consistency...")
    
    # Load consistency data
    runs = load_consistency_data(consistency_dir)
    
    if not runs:
        print(f"No consistency data found in '{consistency_dir}'. "
              "Make sure consistency experiment results exist.")
        return
    
    # Create consistency plot
    print("\nGenerating consistency visualization...")
    df = create_consistency_plot(
        runs, 
        title="Tool Calls Across Consistency Runs (QA10)"
    )
    
    # Analyze consistency statistics
    analyze_consistency_statistics(df)
    
    print("\nConsistency analysis complete!")


if __name__ == "__main__":
    main()
