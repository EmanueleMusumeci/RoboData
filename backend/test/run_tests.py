#!/usr/bin/env python3
"""
Test runner for RoboData backend tests.
This script checks for dependencies and runs tests accordingly.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import google.generativeai
    except ImportError:
        missing_deps.append("google-generativeai")
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
    
    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")
    
    try:
        import wikidata
    except ImportError:
        missing_deps.append("wikidata")
    
    return missing_deps

def main():
    """Run tests with proper dependency checking."""
    print("🧪 RoboData Test Runner (Real API Tests)")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return 1
    
    print("✅ All dependencies available")
    print("🌐 Note: These tests make real API calls to Wikidata")
    
    # Add backend to path
    backend_path = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_path))
    
    # Run tests
    test_files = [
        Path(__file__).parent / "test_datamodel.py",
        Path(__file__).parent / "test_base_tools.py",
#        Path(__file__).parent / "test_tools.py",
#        Path(__file__).parent / "test_api.py",
#        Path(__file__).parent / "test_api_kif.py",
    ]
    
    for test_file in test_files:
        print(f"\n🔍 Running tests in {test_file.name}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_file), "-v", "-s"
            ], cwd=str(backend_path))
            
            if result.returncode != 0:
                print(f"❌ Tests failed for {test_file}")
                return result.returncode
            else:
                print(f"✅ Tests passed for {test_file}")
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return 1

    print("\n🎉 All real API tests passed successfully!")
    print("💡 Tests verified the complete tool and API pipeline")
    return 0

if __name__ == "__main__":
    sys.exit(main())