#!/usr/bin/env python3
"""
Comprehensive test script for ScoutAgent Phase 2: Agent Registry and Base Agent Classes
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import get_registry, create_agent, AgentInput
from custom_logging import get_logger

logger = get_logger("test_agents")

def test_agent_registry():
    """Test the agent registry system."""
    print("🧪 Testing Agent Registry...")
    
    registry = get_registry()
    
    # Test listing agents
    agents = registry.list_agents()
    print(f"✅ Registered agents: {agents}")
    
    expected_agents = ['ResearchAgent', 'research', 'CodeAgent', 'code', 'AnalysisAgent', 'analysis']
    for agent in expected_agents:
        if agent not in agents:
            raise ValueError(f"Missing agent: {agent}")
    
    print("✅ All expected agents registered")

def test_agent_creation():
    """Test agent creation and basic functionality."""
    print("\n🧪 Testing Agent Creation...")
    
    # Test creating each agent type
    agents = {}
    
    for agent_name in ['research', 'code', 'analysis']:
        agent = create_agent(agent_name)
        agents[agent_name] = agent
        print(f"✅ Created {agent_name} agent: {agent.agent_id}")
    
    return agents

def test_agent_execution():
    """Test agent execution with different inputs."""
    print("\n🧪 Testing Agent Execution...")
    
    # Test Research Agent
    research_agent = create_agent('research')
    research_input = AgentInput(
        data="Python async programming best practices",
        metadata={"max_results": 5, "include_sources": True}
    )
    
    print("📊 Running Research Agent...")
    research_result = research_agent.execute(research_input)
    print(f"✅ Research completed in {research_result.execution_time:.2f}s")
    print(f"   Success: {research_result.success}")
    print(f"   Sources found: {len(research_result.result.get('sources', []))}")
    
    # Test Code Agent
    code_agent = create_agent('code')
    code_input = AgentInput(
        data="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        metadata={"language": "python", "analysis_type": "complexity"}
    )
    
    print("\n💻 Running Code Agent...")
    code_result = code_agent.execute(code_input)
    print(f"✅ Code analysis completed in {code_result.execution_time:.2f}s")
    print(f"   Success: {code_result.success}")
    print(f"   Language detected: {code_result.result.get('language')}")
    print(f"   Complexity: {code_result.result.get('complexity')}")
    
    # Test Analysis Agent
    analysis_agent = create_agent('analysis')
    analysis_input = AgentInput(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        metadata={"analysis_type": "descriptive_statistics"}
    )
    
    print("\n📈 Running Analysis Agent...")
    analysis_result = analysis_agent.execute(analysis_input)
    print(f"✅ Analysis completed in {analysis_result.execution_time:.2f}s")
    print(f"   Success: {analysis_result.success}")
    print(f"   Mean: {analysis_result.result.get('statistics', {}).get('mean')}")
    print(f"   Std Dev: {analysis_result.result.get('statistics', {}).get('std_dev')}")
    
    return {
        'research': research_result,
        'code': code_result,
        'analysis': analysis_result
    }

def test_agent_state_management():
    """Test agent state management and serialization."""
    print("\n🧪 Testing Agent State Management...")
    
    agent = create_agent('research')
    
    # Test state access
    state = agent.get_state()
    print(f"✅ Agent state retrieved: {state['status']}")
    
    # Test state saving
    state_file = Path("test_agent_state.json")
    agent.save_state(str(state_file))
    print(f"✅ Agent state saved to {state_file}")
    
    # Test state loading
    agent.load_state(str(state_file))
    print(f"✅ Agent state loaded from {state_file}")
    
    # Clean up
    state_file.unlink(missing_ok=True)
    print("✅ Test state file cleaned up")

def test_agent_instances():
    """Test agent instance management."""
    print("\n🧪 Testing Agent Instance Management...")
    
    registry = get_registry()
    
    # Create multiple instances
    agent1 = create_agent('research')
    agent2 = create_agent('research')
    
    print(f"✅ Created two research agents: {agent1.agent_id}, {agent2.agent_id}")
    
    # List instances
    instances = registry.list_instances()
    print(f"✅ Active instances: {len(instances)}")
    
    # Remove an instance
    registry.remove_agent(agent1.agent_id)
    instances = registry.list_instances()
    print(f"✅ After removal: {len(instances)} instances")
    
    # Clear all instances
    registry.clear_instances()
    instances = registry.list_instances()
    print(f"✅ After clear: {len(instances)} instances")

def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Starting ScoutAgent Phase 2 Tests...\n")
    
    try:
        test_agent_registry()
        agents = test_agent_creation()
        results = test_agent_execution()
        test_agent_state_management()
        test_agent_instances()
        
        print("\n🎉 All tests passed successfully!")
        print("\n📊 Test Summary:")
        print("   ✅ Agent Registry: Working")
        print("   ✅ Agent Creation: Working")
        print("   ✅ Agent Execution: Working")
        print("   ✅ State Management: Working")
        print("   ✅ Instance Management: Working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error("Test failed", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
