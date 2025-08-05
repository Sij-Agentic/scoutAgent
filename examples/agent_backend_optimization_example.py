#!/usr/bin/env python3
"""
Agent-Specific Backend Optimization Example

This example demonstrates how different agents can be optimized for specific tasks
by using the most suitable LLM backend for their workload.

Usage:
    python agent_backend_optimization_example.py
"""

import asyncio
import time
from typing import Dict, Any, List

from agents.base import BaseAgent, AgentInput, AgentOutput
from agents.gap_finder_enhanced import EnhancedGapFinderAgent
from agents.gap_finder import GapFinderInput, MarketGap
from llm.utils import LLMAgentMixin
from llm import get_llm_manager
from config import get_config


class CodeOptimizedAgent(BaseAgent, LLMAgentMixin):
    """Agent optimized for coding tasks using DeepSeek."""
    
    def __init__(self):
        super().__init__(name="CodeOptimizedAgent")
        self.initialize_llm()
        # DeepSeek is automatically preferred for this agent
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        return {"task": "code_analysis", "backend": "deepseek"}
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        # Use task-specific backend selection
        code_analysis = await self.llm_generate(
            "Analyze this code for bugs and improvements: " + str(agent_input.data),
            task_type="code_review"  # Automatically uses DeepSeek
        )
        return {"analysis": code_analysis}
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Any:
        # Generate optimized code
        optimized_code = await self.llm_generate(
            f"Optimize this code based on analysis: {thoughts['analysis']}",
            task_type="code_generation"  # Uses DeepSeek for coding
        )
        return {"optimized_code": optimized_code, "analysis": thoughts["analysis"]}


class CreativeWriterAgent(BaseAgent, LLMAgentMixin):
    """Agent optimized for creative writing using Claude."""
    
    def __init__(self):
        super().__init__(name="CreativeWriterAgent")
        self.initialize_llm()
        # Claude is automatically preferred for creative tasks
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        return {"task": "creative_writing", "backend": "claude"}
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        # Use Claude for creative brainstorming
        ideas = await self.llm_generate(
            f"Brainstorm creative ideas for: {agent_input.data}",
            task_type="creative_writing"  # Automatically uses Claude
        )
        return {"ideas": ideas}
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Any:
        # Generate creative content
        content = await self.llm_generate(
            f"Write engaging content based on these ideas: {thoughts['ideas']}",
            task_type="storytelling"  # Uses Claude for storytelling
        )
        return {"content": content, "ideas": thoughts["ideas"]}


class HybridIntelligentAgent(BaseAgent, LLMAgentMixin):
    """Agent that intelligently switches backends based on task complexity."""
    
    def __init__(self):
        super().__init__(name="HybridIntelligentAgent")
        self.initialize_llm()
        # Set up custom task preferences
        self.set_task_backend("simple_query", "gemini")
        self.set_task_backend("complex_analysis", "openai")
        self.set_task_backend("creative_content", "claude")
        self.set_task_backend("code_task", "deepseek")
        self.set_task_backend("private_data", "ollama")
    
    def determine_task_complexity(self, query: str) -> str:
        """Determine the best backend based on query complexity."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["code", "programming", "debug", "function"]):
            return "code_task"
        elif any(word in query_lower for word in ["creative", "story", "write", "content"]):
            return "creative_content"
        elif any(word in query_lower for word in ["analyze", "complex", "research", "detailed"]):
            return "complex_analysis"
        elif any(word in query_lower for word in ["private", "confidential", "local", "sensitive"]):
            return "private_data"
        else:
            return "simple_query"
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        task_type = self.determine_task_complexity(str(agent_input.data))
        optimal_backend = self.get_optimal_backend(task_type)
        return {"task_type": task_type, "optimal_backend": optimal_backend}
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        # Use the optimal backend for thinking
        thoughts = await self.llm_generate(
            f"Think about this request: {agent_input.data}",
            task_type=plan["task_type"]
        )
        return {"thoughts": thoughts, "backend_used": plan["optimal_backend"]}
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Any:
        # Execute with the same optimal backend
        result = await self.llm_generate(
            f"Execute this request: {agent_input.data}\nBased on thoughts: {thoughts['thoughts']}",
            task_type=plan["task_type"]
        )
        return {
            "result": result,
            "backend_used": thoughts["backend_used"],
            "task_type": plan["task_type"]
        }


async def demonstrate_backend_optimization():
    """Demonstrate how different agents use optimal backends for their tasks."""
    
    print("🚀 Agent-Specific Backend Optimization Demo")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Code Review Task",
            "data": "def calculate_sum(numbers): return sum(numbers)",
            "expected_backend": "deepseek"
        },
        {
            "name": "Creative Writing Task", 
            "data": "Write a story about AI and humans working together",
            "expected_backend": "claude"
        },
        {
            "name": "Quick Question",
            "data": "What is machine learning?",
            "expected_backend": "gemini"
        },
        {
            "name": "Complex Research",
            "data": "Analyze the impact of AI on job markets with detailed research",
            "expected_backend": "openai"
        }
    ]
    
    # Test the hybrid agent's backend selection
    hybrid_agent = HybridIntelligentAgent()
    
    for scenario in test_scenarios:
        print(f"\n📋 Testing Scenario: {scenario['name']}")
        print(f"Data: {scenario['data']}")
        print(f"Expected optimal backend: {scenario['expected_backend']}")
        print("-" * 40)
        
        # Determine optimal backend
        task_type = hybrid_agent.determine_task_complexity(scenario['data'])
        optimal_backend = hybrid_agent.get_optimal_backend(task_type)
        
        print(f"  Task Type: {task_type}")
        print(f"  Optimal Backend: {optimal_backend}")
        print(f"  ✅ Correct Selection: {optimal_backend == scenario['expected_backend']}")


async def demonstrate_dynamic_backend_switching():
    """Show how an agent can dynamically switch backends mid-conversation."""
    
    print("\n🔄 Dynamic Backend Switching Demo")
    print("=" * 50)
    
    agent = HybridIntelligentAgent()
    
    conversation_flow = [
        ("Quick question: What is Python?", "simple_query"),
        ("Now write a creative story about Python the snake", "creative_content"),
        ("Debug this Python code: def broken_func(): return x + y", "code_task"),
        ("Analyze this private customer data: [CONFIDENTIAL]", "private_data"),
        ("Provide detailed research on Python's impact on software development", "complex_analysis")
    ]
    
    for i, (query, expected_task_type) in enumerate(conversation_flow, 1):
        print(f"\n{i}. Query: {query}")
        
        # Determine optimal backend
        task_type = agent.determine_task_complexity(query)
        optimal_backend = agent.get_optimal_backend(task_type)
        
        print(f"   Task Type: {task_type}")
        print(f"   Optimal Backend: {optimal_backend}")
        print(f"   Expected Task Type: {expected_task_type}")
        print(f"   ✅ Correct Classification: {task_type == expected_task_type}")


async def demonstrate_agent_preferences():
    """Show how different agent types have different default preferences."""
    
    print("\n🎯 Agent Type Preferences Demo")
    print("=" * 50)
    
    # Create different agent types
    agents = [
        ("CodeOptimizedAgent", CodeOptimizedAgent()),
        ("CreativeWriterAgent", CreativeWriterAgent()),
        ("EnhancedGapFinderAgent", EnhancedGapFinderAgent()),
        ("HybridIntelligentAgent", HybridIntelligentAgent())
    ]
    
    for agent_name, agent in agents:
        print(f"\n{agent_name}:")
        print(f"  Preferred Backend: {getattr(agent, 'preferred_backend', 'None')}")
        print(f"  Task Preferences: {getattr(agent, 'task_backend_preferences', {})}")


if __name__ == "__main__":
    async def main():
        try:
            # Run the demonstrations
            await demonstrate_backend_optimization()
            await demonstrate_dynamic_backend_switching()
            await demonstrate_agent_preferences()
            
            print("\n🎉 Demo completed successfully!")
            print("\nTo use agent-specific backends in your own agents:")
            print("1. Inherit from LLMAgentMixin")
            print("2. Call self.initialize_llm() in __init__")
            print("3. Use task_type parameter in llm_generate() calls")
            print("4. Use set_preferred_backend() or set_task_backend() for custom preferences")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())
