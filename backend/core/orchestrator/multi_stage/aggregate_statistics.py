#!/usr/bin/env python3
"""
Statistics aggregation script for RoboData multi-stage orchestrator.

This script aggregates multiple statistics files from different query experiments
and generates comprehensive reports for experimental evaluation.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics as py_stats


class StatisticsAggregator:
    """Aggregates statistics from multiple experiment runs."""
    
    def __init__(self):
        self.stats_files: List[Dict[str, Any]] = []
        self.aggregated_data: Dict[str, Any] = {}
        
    def load_statistics_files(self, pattern: str, directory: Optional[str] = None) -> int:
        """Load statistics files matching the pattern."""
        if directory:
            search_path = Path(directory)
        else:
            search_path = Path("experiments")
            
        if not search_path.exists():
            print(f"Directory {search_path} not found")
            return 0
            
        # Find all matching files
        files_found = list(search_path.rglob(pattern))
        
        for file_path in files_found:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = str(file_path)
                    self.stats_files.append(data)
                    print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        print(f"\nLoaded {len(self.stats_files)} statistics files")
        return len(self.stats_files)
        
    def aggregate_basic_metrics(self) -> Dict[str, Any]:
        """Aggregate basic performance metrics."""
        if not self.stats_files:
            return {}
            
        total_times = [s['total_time'] for s in self.stats_files if s.get('total_time')]
        success_count = sum(1 for s in self.stats_files if s.get('success', False))
        metacognition_enabled = sum(1 for s in self.stats_files if s.get('enable_metacognition', False))
        
        # Attempt history aggregation
        remote_explorations = [s['attempt_history']['remote_explorations'] 
                              for s in self.stats_files 
                              if s.get('attempt_history', {}).get('remote_explorations') is not None]
        local_explorations = [s['attempt_history']['local_explorations'] 
                             for s in self.stats_files 
                             if s.get('attempt_history', {}).get('local_explorations') is not None]
        
        return {
            "total_experiments": len(self.stats_files),
            "successful_experiments": success_count,
            "success_rate": success_count / len(self.stats_files),
            "metacognition_enabled_count": metacognition_enabled,
            "total_time": {
                "min": min(total_times) if total_times else 0,
                "max": max(total_times) if total_times else 0,
                "mean": py_stats.mean(total_times) if total_times else 0,
                "median": py_stats.median(total_times) if total_times else 0,
                "stdev": py_stats.stdev(total_times) if len(total_times) > 1 else 0
            },
            "remote_explorations": {
                "min": min(remote_explorations) if remote_explorations else 0,
                "max": max(remote_explorations) if remote_explorations else 0,
                "mean": py_stats.mean(remote_explorations) if remote_explorations else 0,
                "median": py_stats.median(remote_explorations) if remote_explorations else 0,
                "total": sum(remote_explorations)
            },
            "local_explorations": {
                "min": min(local_explorations) if local_explorations else 0,
                "max": max(local_explorations) if local_explorations else 0,
                "mean": py_stats.mean(local_explorations) if local_explorations else 0,
                "median": py_stats.median(local_explorations) if local_explorations else 0,
                "total": sum(local_explorations)
            }
        }
        
    def aggregate_state_statistics(self) -> Dict[str, Any]:
        """Aggregate state-level statistics across all experiments."""
        state_data = {}
        
        for stats_file in self.stats_files:
            state_stats = stats_file.get('state_statistics', {})
            
            for state_name, state_info in state_stats.items():
                if state_name not in state_data:
                    state_data[state_name] = {
                        "visits": [],
                        "total_times": [],
                        "average_times": [],
                        "tool_calls": [],
                        "total_inference_times": [],
                        "average_inference_times": []
                    }
                
                state_data[state_name]["visits"].append(state_info.get("visits", 0))
                state_data[state_name]["total_times"].append(state_info.get("total_time", 0))
                state_data[state_name]["average_times"].append(state_info.get("average_time", 0))
                state_data[state_name]["tool_calls"].append(state_info.get("tool_calls", 0))
                state_data[state_name]["total_inference_times"].append(state_info.get("total_inference_time", 0))
                state_data[state_name]["average_inference_times"].append(state_info.get("average_inference_time", 0))
        
        # Calculate aggregated statistics for each state
        aggregated_states = {}
        for state_name, data in state_data.items():
            def safe_stats(values):
                if not values or all(v == 0 for v in values):
                    return {"min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0, "total": 0}
                return {
                    "min": min(values),
                    "max": max(values), 
                    "mean": py_stats.mean(values),
                    "median": py_stats.median(values),
                    "stdev": py_stats.stdev(values) if len(values) > 1 else 0,
                    "total": sum(values)
                }
                
            aggregated_states[state_name] = {
                "experiments_visited": len([v for v in data["visits"] if v > 0]),
                "visits": safe_stats(data["visits"]),
                "total_time": safe_stats(data["total_times"]),
                "average_time": safe_stats(data["average_times"]),
                "tool_calls": safe_stats(data["tool_calls"]),
                "total_inference_time": safe_stats(data["total_inference_times"]),
                "average_inference_time": safe_stats(data["average_inference_times"])
            }
            
        return aggregated_states
        
    def aggregate_tool_statistics(self) -> Dict[str, Any]:
        """Aggregate tool-level statistics across all experiments."""
        tool_data = {}
        
        for stats_file in self.stats_files:
            tool_stats = stats_file.get('tool_statistics', {})
            
            for tool_name, tool_info in tool_stats.items():
                if tool_name not in tool_data:
                    tool_data[tool_name] = {
                        "call_counts": [],
                        "success_counts": [],
                        "failure_counts": [],
                        "total_times": [],
                        "average_times": [],
                        "success_rates": [],
                        "all_contexts": set()
                    }
                
                tool_data[tool_name]["call_counts"].append(tool_info.get("call_count", 0))
                tool_data[tool_name]["success_counts"].append(tool_info.get("success_count", 0))
                tool_data[tool_name]["failure_counts"].append(tool_info.get("failure_count", 0))
                tool_data[tool_name]["total_times"].append(tool_info.get("total_time", 0))
                tool_data[tool_name]["average_times"].append(tool_info.get("average_time", 0))
                tool_data[tool_name]["success_rates"].append(tool_info.get("success_rate", 0))
                tool_data[tool_name]["all_contexts"].update(tool_info.get("contexts", []))
                
        # Calculate aggregated statistics for each tool
        aggregated_tools = {}
        for tool_name, data in tool_data.items():
            def safe_stats(values):
                if not values or all(v == 0 for v in values):
                    return {"min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0, "total": 0}
                return {
                    "min": min(values),
                    "max": max(values),
                    "mean": py_stats.mean(values),
                    "median": py_stats.median(values),
                    "stdev": py_stats.stdev(values) if len(values) > 1 else 0,
                    "total": sum(values)
                }
                
            aggregated_tools[tool_name] = {
                "experiments_used": len([v for v in data["call_counts"] if v > 0]),
                "call_count": safe_stats(data["call_counts"]),
                "success_count": safe_stats(data["success_counts"]),
                "failure_count": safe_stats(data["failure_counts"]),
                "total_time": safe_stats(data["total_times"]),
                "average_time": safe_stats(data["average_times"]),
                "success_rate": safe_stats(data["success_rates"]),
                "contexts": list(data["all_contexts"])
            }
            
        return aggregated_tools
        
    def aggregate_inference_statistics(self) -> Dict[str, Any]:
        """Aggregate inference-level statistics."""
        orchestrator_data = {"counts": [], "total_times": [], "average_times": [], "total_tokens": [], "average_tokens": []}
        metacognition_data = {"counts": [], "total_times": [], "average_times": [], "total_tokens": [], "average_tokens": []}
        
        for stats_file in self.stats_files:
            inference_stats = stats_file.get('inference_statistics', {})
            
            # Orchestrator inference
            orch_stats = inference_stats.get('orchestrator', {})
            if orch_stats:
                orchestrator_data["counts"].append(orch_stats.get("count", 0))
                orchestrator_data["total_times"].append(orch_stats.get("total_time", 0))
                orchestrator_data["average_times"].append(orch_stats.get("average_time", 0))
                orchestrator_data["total_tokens"].append(orch_stats.get("total_tokens", 0))
                orchestrator_data["average_tokens"].append(orch_stats.get("average_tokens", 0))
            
            # Metacognition inference
            meta_stats = inference_stats.get('metacognition')
            if meta_stats:
                metacognition_data["counts"].append(meta_stats.get("count", 0))
                metacognition_data["total_times"].append(meta_stats.get("total_time", 0))
                metacognition_data["average_times"].append(meta_stats.get("average_time", 0))
                metacognition_data["total_tokens"].append(meta_stats.get("total_tokens", 0))
                metacognition_data["average_tokens"].append(meta_stats.get("average_tokens", 0))
                
        def safe_stats(values):
            if not values:
                return {"min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0, "total": 0}
            return {
                "min": min(values),
                "max": max(values),
                "mean": py_stats.mean(values),
                "median": py_stats.median(values),
                "stdev": py_stats.stdev(values) if len(values) > 1 else 0,
                "total": sum(values)
            }
            
        return {
            "orchestrator": {
                "count": safe_stats(orchestrator_data["counts"]),
                "total_time": safe_stats(orchestrator_data["total_times"]),
                "average_time": safe_stats(orchestrator_data["average_times"]),
                "total_tokens": safe_stats(orchestrator_data["total_tokens"]),
                "average_tokens": safe_stats(orchestrator_data["average_tokens"])
            },
            "metacognition": {
                "count": safe_stats(metacognition_data["counts"]),
                "total_time": safe_stats(metacognition_data["total_times"]),
                "average_time": safe_stats(metacognition_data["average_times"]),
                "total_tokens": safe_stats(metacognition_data["total_tokens"]),
                "average_tokens": safe_stats(metacognition_data["average_tokens"])
            } if any(metacognition_data["counts"]) else None
        }
        
    def aggregate_metacognition_statistics(self) -> Optional[Dict[str, Any]]:
        """Aggregate metacognition-specific statistics."""
        metacog_data = {"cycle_counts": [], "total_times": [], "average_times": [], 
                       "total_tokens": [], "average_tokens": [], "suggestion_counts": [], "suggestion_rates": []}
        
        has_metacognition = False
        for stats_file in self.stats_files:
            meta_stats = stats_file.get('metacognition_statistics')
            if meta_stats:
                has_metacognition = True
                metacog_data["cycle_counts"].append(meta_stats.get("cycle_count", 0))
                metacog_data["total_times"].append(meta_stats.get("total_time", 0))
                metacog_data["average_times"].append(meta_stats.get("average_time", 0))
                metacog_data["total_tokens"].append(meta_stats.get("total_tokens", 0))
                metacog_data["average_tokens"].append(meta_stats.get("average_tokens", 0))
                metacog_data["suggestion_counts"].append(meta_stats.get("suggestion_count", 0))
                metacog_data["suggestion_rates"].append(meta_stats.get("suggestion_rate", 0))
                
        if not has_metacognition:
            return None
            
        def safe_stats(values):
            if not values:
                return {"min": 0, "max": 0, "mean": 0, "median": 0, "stdev": 0, "total": 0}
            return {
                "min": min(values),
                "max": max(values),
                "mean": py_stats.mean(values),
                "median": py_stats.median(values),
                "stdev": py_stats.stdev(values) if len(values) > 1 else 0,
                "total": sum(values)
            }
            
        return {
            "cycle_count": safe_stats(metacog_data["cycle_counts"]),
            "total_time": safe_stats(metacog_data["total_times"]),
            "average_time": safe_stats(metacog_data["average_times"]),
            "total_tokens": safe_stats(metacog_data["total_tokens"]),
            "average_tokens": safe_stats(metacog_data["average_tokens"]),
            "suggestion_count": safe_stats(metacog_data["suggestion_counts"]),
            "suggestion_rate": safe_stats(metacog_data["suggestion_rates"])
        }
        
    def generate_aggregated_report(self) -> Dict[str, Any]:
        """Generate the complete aggregated report."""
        if not self.stats_files:
            return {"error": "No statistics files loaded"}
            
        self.aggregated_data = {
            "aggregation_info": {
                "timestamp": datetime.now().isoformat(),
                "source_files": [s.get('_source_file') for s in self.stats_files],
                "total_files": len(self.stats_files)
            },
            "basic_metrics": self.aggregate_basic_metrics(),
            "state_statistics": self.aggregate_state_statistics(),
            "tool_statistics": self.aggregate_tool_statistics(),
            "inference_statistics": self.aggregate_inference_statistics(),
            "metacognition_statistics": self.aggregate_metacognition_statistics()
        }
        
        return self.aggregated_data
        
    def save_report(self, output_file: str):
        """Save the aggregated report to a JSON file."""
        if not self.aggregated_data:
            self.generate_aggregated_report()
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregated_data, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"\nAggregated report saved to: {output_file}")
        
    def print_summary(self):
        """Print a summary of the aggregated statistics."""
        if not self.aggregated_data:
            self.generate_aggregated_report()
            
        basic = self.aggregated_data.get('basic_metrics', {})
        
        print("\n" + "="*60)
        print("ROBODATA MULTI-STAGE ORCHESTRATOR - AGGREGATED STATISTICS")
        print("="*60)
        
        print(f"\nBASIC METRICS:")
        print(f"  Total experiments: {basic.get('total_experiments', 0)}")
        print(f"  Successful experiments: {basic.get('successful_experiments', 0)}")
        print(f"  Success rate: {basic.get('success_rate', 0):.2%}")
        print(f"  Metacognition enabled: {basic.get('metacognition_enabled_count', 0)} experiments")
        
        total_time = basic.get('total_time', {})
        print(f"\nTOTAL TIME STATISTICS:")
        print(f"  Mean: {total_time.get('mean', 0):.2f}s")
        print(f"  Median: {total_time.get('median', 0):.2f}s")
        print(f"  Min: {total_time.get('min', 0):.2f}s")
        print(f"  Max: {total_time.get('max', 0):.2f}s")
        print(f"  Std Dev: {total_time.get('stdev', 0):.2f}s")
        
        remote_exp = basic.get('remote_explorations', {})
        local_exp = basic.get('local_explorations', {})
        print(f"\nEXPLORATION STATISTICS:")
        print(f"  Remote explorations - Mean: {remote_exp.get('mean', 0):.1f}, Total: {remote_exp.get('total', 0)}")
        print(f"  Local explorations - Mean: {local_exp.get('mean', 0):.1f}, Total: {local_exp.get('total', 0)}")
        
        # Show most used tools
        tool_stats = self.aggregated_data.get('tool_statistics', {})
        if tool_stats:
            print(f"\nTOP 5 MOST USED TOOLS:")
            sorted_tools = sorted(tool_stats.items(), 
                                key=lambda x: x[1]['call_count']['total'], 
                                reverse=True)[:5]
            for tool_name, stats in sorted_tools:
                print(f"  {tool_name}: {stats['call_count']['total']} calls "
                      f"(avg {stats['call_count']['mean']:.1f} per experiment)")
        
        # Show most visited states
        state_stats = self.aggregated_data.get('state_statistics', {})
        if state_stats:
            print(f"\nTOP 5 MOST VISITED STATES:")
            sorted_states = sorted(state_stats.items(),
                                 key=lambda x: x[1]['visits']['total'],
                                 reverse=True)[:5]
            for state_name, stats in sorted_states:
                print(f"  {state_name}: {stats['visits']['total']} visits "
                      f"(avg {stats['visits']['mean']:.1f} per experiment)")


def main():
    """Main entry point for the aggregation script."""
    parser = argparse.ArgumentParser(
        description="Aggregate RoboData multi-stage orchestrator statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all statistics files in experiments directory
  python aggregate_statistics.py

  # Aggregate specific pattern
  python aggregate_statistics.py -p "statistics_*.json"
  
  # Aggregate from specific directory
  python aggregate_statistics.py -d /path/to/experiments
  
  # Save to specific output file
  python aggregate_statistics.py -o results/aggregated_stats.json
  
  # Print summary only (no file output)
  python aggregate_statistics.py --summary-only
        """
    )
    
    parser.add_argument(
        "-p", "--pattern",
        default="statistics_*.json",
        help="Pattern to match statistics files (default: statistics_*.json)"
    )
    
    parser.add_argument(
        "-d", "--directory",
        help="Directory to search for statistics files (default: experiments/)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="aggregated_statistics.json",
        help="Output file for aggregated statistics (default: aggregated_statistics.json)"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, don't save to file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = StatisticsAggregator()
    
    # Load files
    if not args.quiet:
        print(f"Searching for files matching '{args.pattern}'...")
    
    files_loaded = aggregator.load_statistics_files(args.pattern, args.directory)
    
    if files_loaded == 0:
        print("No statistics files found. Make sure you have run some experiments first.")
        return 1
        
    # Generate report
    if not args.quiet:
        print("\nGenerating aggregated report...")
    
    aggregator.generate_aggregated_report()
    
    # Save report unless summary-only
    if not args.summary_only:
        aggregator.save_report(args.output)
        
    # Print summary
    aggregator.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
