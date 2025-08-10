#!/usr/bin/env python3
"""
Simple test script to verify the knowledge graph visualization functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

async def test_visualization():
    """Test the visualization functionality with a simple query."""
    
    print("🧪 Testing Knowledge Graph Visualization")
    print("=" * 50)
    
    try:
        # Import required modules
        from backend.main import run_multi_stage_orchestrator, get_default_config
        
        # Get default configuration
        config = get_default_config()
        
        # Override some settings for testing
        config["max_turns"] = 5
        config["output"]["save_results"] = True
        
        # Simple test query
        test_query = "Who was Albert Einstein?"
        
        print(f"🔍 Testing query: {test_query}")
        print(f"📊 Max turns: {config['max_turns']}")
        
        # Run the orchestrator
        result = await run_multi_stage_orchestrator(
            config=config,
            query=test_query,
            enable_question_decomposition=False,
            enable_metacognition=False
        )
        
        print("\n✅ Query completed successfully!")
        print(f"🔄 Turns taken: {result['turns_taken']}")
        print(f"📝 Answer: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
        
        # Check visualization results
        if 'visualizations' in result:
            vis_info = result['visualizations']
            print(f"\n🎨 Visualization Results:")
            print(f"📁 Output directory: {vis_info['output_dir']}")
            
            if vis_info.get('index_html'):
                print(f"📄 Index HTML: {vis_info['index_html']}")
            
            if vis_info.get('animation_path'):
                print(f"🎬 Animation: {vis_info['animation_path']}")
            
            if vis_info.get('final_image'):
                print(f"🖼️  Final graph: {vis_info['final_image']}")
            
            # Check if files actually exist
            from pathlib import Path
            output_dir = Path(vis_info['output_dir'])
            if output_dir.exists():
                png_files = list(output_dir.glob("*.png"))
                gif_files = list(output_dir.glob("*.gif"))
                html_files = list(output_dir.glob("*.html"))
                
                print(f"\n📊 Files generated:")
                print(f"  🖼️  PNG images: {len(png_files)}")
                print(f"  🎬 GIF animations: {len(gif_files)}")
                print(f"  📄 HTML files: {len(html_files)}")
                
                if png_files:
                    print(f"  📋 Image files:")
                    for png_file in sorted(png_files):
                        print(f"    - {png_file.name}")
            else:
                print(f"⚠️  Output directory does not exist: {output_dir}")
        else:
            print("⚠️  No visualization information in result")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_visualization())
    if success:
        print("\n✅ Visualization test completed successfully!")
    else:
        print("\n❌ Visualization test failed!")
        sys.exit(1)
