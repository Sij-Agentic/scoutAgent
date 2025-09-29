#!/usr/bin/env python3
"""
Test script to verify ScoutAgent container functionality
"""

import subprocess
import sys
import os

def test_scout_agent_help():
    """Test that ScoutAgent can show help without errors"""
    try:
        result = subprocess.run([
            "python", "-m", "scout_agent.main", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ ScoutAgent help command works")
            print("Help output:")
            print(result.stdout)
            return True
        else:
            print("❌ ScoutAgent help command failed")
            print("Error output:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ ScoutAgent help command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running ScoutAgent: {e}")
        return False

def test_imports():
    """Test that key modules can be imported"""
    try:
        import scout_agent.main
        import scout_agent.agents.scout
        import scout_agent.orchestration
        print("✅ Key ScoutAgent modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment variables and paths"""
    print(f"Python path: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"SCOUT_LLM_DEFAULT_BACKEND: {os.environ.get('SCOUT_LLM_DEFAULT_BACKEND', 'Not set')}")
    print(f"SCOUT_LLM_DEFAULT_MODEL: {os.environ.get('SCOUT_LLM_DEFAULT_MODEL', 'Not set')}")

if __name__ == "__main__":
    print("🧪 Testing ScoutAgent Container...")
    print("=" * 50)
    
    test_environment()
    print()
    
    success = True
    success &= test_imports()
    success &= test_scout_agent_help()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Container is ready.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Check the output above.")
        sys.exit(1)
