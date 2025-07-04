import asyncio
import sys
from typing import Dict, Any
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from settings import settings_manager
from core.agents.gemini import GeminiAgent, Query
from core.toolbox.toolbox import Toolbox
from core.toolbox.wikidata import (
    SPARQLQueryTool, SubclassQueryTool, SuperclassQueryTool, InstanceQueryTool,
    EntityExplorationTool, PathFindingTool, LocalGraphTool
)

class RoboDataApp:
    """Main RoboData application with interactive terminal."""
    
    def __init__(self):
        self.settings = settings_manager.get_settings()
        self.agent = None
        self.toolbox = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all application components."""
        print("🚀 Initializing RoboData...")
        
        # Initialize toolbox first
        self.toolbox = Toolbox()
        
        # Register default Wikidata tools if enabled
        if self.settings.toolbox.auto_register_wikidata_tools:
            self._register_wikidata_tools()
        
        # Initialize LLM agent with the toolbox
        if self.settings.llm.provider == "gemini":
            self.agent = GeminiAgent(toolbox=self.toolbox)
            print(f"✅ Initialized Gemini agent with model: {self.settings.llm.model}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.settings.llm.provider}")
        
        print(f"✅ Registered {len(self.toolbox.list_tools())} tools")
        print("✅ RoboData initialized successfully!")
    
    def _register_wikidata_tools(self):
        """Register all Wikidata tools in the main toolbox."""
        print("🔧 Registering Wikidata tools...")
        
        tools = [
            SPARQLQueryTool(),
            SubclassQueryTool(),
            SuperclassQueryTool(),
            InstanceQueryTool(),
            EntityExplorationTool(),
            PathFindingTool(),
            LocalGraphTool()
        ]
        
        for tool in tools:
            self.toolbox.register_tool(tool)
            print(f"  ✓ Registered: {tool.name}")
    
    async def process_user_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query with detailed logging."""
        if self.settings.interactive.show_intermediate_steps:
            print(f"\n🔍 Processing query: '{user_input}'")
        
        # Extract entity ID if present in the input
        entity_id = self._extract_entity_id(user_input)
        
        # Create query object
        query = Query(text=user_input, entity_id=entity_id)
        
        if self.settings.interactive.show_intermediate_steps:
            if entity_id:
                print(f"🎯 Detected entity ID: {entity_id}")
            else:
                print("🔎 No entity ID detected in query")
        
        try:
            # Process query through agent
            result = await self.agent.process_query(query)
            
            if self.settings.interactive.show_tool_calls:
                print(f"⚡ Tool execution completed")
            
            return result
        
        except Exception as e:
            error_result = {"error": str(e), "type": "processing_error"}
            print(f"❌ Error processing query: {e}")
            return error_result
    
    def _extract_entity_id(self, text: str) -> str:
        """Extract Wikidata entity ID from user input."""
        import re
        # Look for Wikidata entity IDs (Q followed by numbers)
        match = re.search(r'\bQ\d+\b', text)
        return match.group(0) if match else None
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format result for display."""
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        if "message" in result:
            return f"💬 {result['message']}"
        
        if isinstance(result, list):
            if len(result) == 0:
                return "📭 No results found"
            
            formatted = "📊 Results:\n"
            for i, item in enumerate(result[:10], 1):  # Limit to 10 items
                if isinstance(item, dict):
                    if 'label' in item and 'id' in item:
                        formatted += f"  {i}. {item['label']} ({item['id']})\n"
                    else:
                        formatted += f"  {i}. {item}\n"
                else:
                    formatted += f"  {i}. {item}\n"
            
            if len(result) > 10:
                formatted += f"  ... and {len(result) - 10} more results\n"
            
            return formatted
        
        if isinstance(result, dict):
            if 'nodes' in result and 'edges' in result:
                return f"🕸️  Graph: {result.get('total_nodes', 0)} nodes, {result.get('total_edges', 0)} edges"
            elif 'entity' in result:
                entity = result['entity']
                return f"🏷️  Entity: {entity.get('label', 'Unknown')} ({entity.get('id', 'Unknown')})\n   📝 {entity.get('description', 'No description')}"
        
        return f"📄 {result}"
    
    def show_help(self):
        """Show help information."""
        help_text = """
🤖 RoboData Interactive Terminal

Available commands:
  help                    - Show this help message
  tools                   - List all available tools
  settings                - Show current settings
  clear                   - Clear conversation history
  exit/quit               - Exit the application

Query examples:
  "What are the subclasses of Q35120?"
  "Explore entity Q5"
  "Find instances of Q16521"
  "Build a local graph around Q1"
  "Find path between Q5 and Q35120"

Tips:
  - Include Wikidata entity IDs (like Q35120) in your queries
  - Use natural language to describe what you want to explore
  - The system will show intermediate steps if enabled in settings
        """
        print(help_text)
    
    def show_tools(self):
        """Show available tools."""
        tools = self.toolbox.list_tools()
        print(f"\n🔧 Available tools ({len(tools)}):")
        for tool_name in tools:
            tool_def = self.toolbox.get_tool_definition(tool_name)
            print(f"  • {tool_name}: {tool_def.description}")
    
    def show_settings(self):
        """Show current settings."""
        print(f"\n⚙️  Current settings:")
        print(f"  LLM Provider: {self.settings.llm.provider}")
        print(f"  LLM Model: {self.settings.llm.model}")
        print(f"  Show tool calls: {self.settings.interactive.show_tool_calls}")
        print(f"  Show steps: {self.settings.interactive.show_intermediate_steps}")
        print(f"  Max results: {self.settings.wikidata.max_results}")
    
    async def run_interactive(self):
        """Run the interactive terminal session."""
        print("\n" + "="*60)
        print("🤖 Welcome to RoboData Interactive Terminal!")
        print("="*60)
        print("Type 'help' for commands or start asking questions about Wikidata!")
        print("Example: 'What are the subclasses of Q35120?'")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🎤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'tools':
                    self.show_tools()
                    continue
                elif user_input.lower() == 'settings':
                    self.show_settings()
                    continue
                elif user_input.lower() == 'clear':
                    self.agent.clear_history()
                    print("🧹 Conversation history cleared!")
                    continue
                
                # Process the query
                result = await self.process_user_query(user_input)
                
                # Format and display result
                formatted_result = self._format_result(result)
                print(f"\n🤖 RoboData: {formatted_result}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

async def main():
    """Main entry point."""
    try:
        app = RoboDataApp()
        await app.run_interactive()
    except Exception as e:
        print(f"❌ Failed to start RoboData: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Comment out API and webapp components for now
    # print("Starting FastAPI server...")
    # print("Starting WebSocket server...")
    
    asyncio.run(main())

