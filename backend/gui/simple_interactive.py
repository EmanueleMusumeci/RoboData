"""
Simple Interactive Interface for RoboData

This module provides a simple console-based interface for the MultiStage orchestrator
that uses standard input/output without curses, allowing users to scroll back through
the terminal history. Users can enter queries one at a time and see the full
orchestrator processing output.
"""

import asyncio
import sys
from typing import Dict, Any
from datetime import datetime

from core.logging import log_debug


async def run_simple_interactive_session(orchestrator, config: Dict[str, Any]) -> None:
    """Run a simple interactive session with the MultiStage orchestrator.
    
    Args:
        orchestrator: Configured MultiStage orchestrator instance
        config: Configuration dictionary
    """
    print("=== RoboData Simple Interactive Mode ===")
    print("Enter your queries to explore knowledge with AI assistance.")
    print("Commands:")
    print("  - Type your questions to process them with the orchestrator")
    print("  - Type 'exit', 'quit', or 'stop' to end the session")
    print("  - Press Ctrl+C to interrupt")
    print("=" * 55)
    print()
    
    query_count = 0
    
    try:
        while True:
            # Get user input
            try:
                print(f"\n[Query #{query_count + 1}]")
                user_query = input("You: ").strip()
                
                # Check for exit commands
                if user_query.lower() in ['exit', 'quit', 'stop', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Skip empty queries
                if not user_query:
                    print("Please enter a query or 'exit' to quit.")
                    continue
                
                query_count += 1
                
                # Process the query
                print(f"\n🤖 Processing query: {user_query}")
                print("=" * 60)
                
                start_time = datetime.now()
                result = await orchestrator.process_user_query(user_query)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Display results
                print("\n" + "=" * 60)
                print("🎯 RESULT")
                print("=" * 60)
                print(f"Query: {user_query}")
                print(f"Answer: {result['answer']}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Turns taken: {result['turns_taken']}")
                if result.get('max_turns', 0) > 0:
                    print(f"Max turns: {result['max_turns']}")
                
                # Show attempt statistics
                attempts = result.get('attempts', {})
                if attempts:
                    print("\nExploration Summary:")
                    print(f"  Remote explorations: {attempts.get('remote_explorations', 0)}")
                    print(f"  Local explorations: {attempts.get('local_explorations', 0)}")
                    if attempts.get('failures'):
                        print(f"  Failures: {len(attempts['failures'])}")
                
                print("=" * 60)
                
                # Log the result for debugging
                log_debug(f"Interactive query #{query_count} completed: {result}")
                
                # Save results if configured
                if config.get("output", {}).get("save_results", True):
                    from utils import save_orchestrator_results, generate_experiment_id
                    experiment_id = f"simple_interactive_{query_count:03d}_{generate_experiment_id(user_query)}"
                    await save_orchestrator_results(result, experiment_id, user_query, orchestrator.knowledge_graph)
                    print(f"💾 Results saved with ID: {experiment_id}")
                
                print("\nReady for next query...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted! Type 'exit' to quit properly or continue with another query.")
                continue
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("You can try another query or type 'exit' to quit.")
                import traceback
                log_debug(f"Error in interactive session: {traceback.format_exc()}")
                continue
    
    except Exception as e:
        print(f"\n❌ Fatal error in interactive session: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n📊 Session Summary:")
        print(f"Total queries processed: {query_count}")
        print("Session ended.")


if __name__ == "__main__":
    """Test the simple interactive interface in standalone mode."""
    print("This module should be run as part of the main application.")
    print("Use: python -m backend.main --interactive")
