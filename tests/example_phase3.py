#!/usr/bin/env python3
"""
Example usage of ScoutAgent Phase 3 DAG Engine & Orchestration.

This script demonstrates how to use the new orchestration system
for pain point discovery and market research workflows.
"""

import asyncio
import json
from datetime import datetime

from orchestrator import get_orchestrator, run_pain_point_discovery, run_market_research
from dag.templates.pain_point_discovery import create_pain_point_discovery_workflow
from dag.templates.market_research import create_market_research_workflow
from dag.engine import DAGEngine
from agents.base import BaseAgent, AgentInput, AgentOutput


# Mock agents for demonstration
class MockResearchAgent(BaseAgent):
    """Mock research agent for demonstration."""
    
    def plan(self, agent_input):
        return {
            "task": "research",
            "query": agent_input.data.get("query", "default query"),
            "sources": ["web", "social", "forums"]
        }
    
    def think(self, agent_input, plan):
        return {
            "analysis": f"Researching {plan['query']}",
            "confidence": 0.85,
            "expected_sources": len(plan["sources"])
        }
    
    def act(self, agent_input, plan, thoughts):
        # Simulate research work
        return {
            "pain_points": [
                {
                    "description": "Manual data entry is time-consuming",
                    "severity": "high",
                    "frequency": "daily",
                    "impact": "productivity"
                },
                {
                    "description": "Lack of integration between tools",
                    "severity": "medium", 
                    "frequency": "weekly",
                    "impact": "workflow"
                }
            ],
            "market_size": "$2.3B",
            "key_players": ["Company A", "Company B"],
            "research_summary": f"Found 2 key pain points in {agent_input.data.get('target_market', 'unknown market')}"
        }


class MockAnalysisAgent(BaseAgent):
    """Mock analysis agent for demonstration."""
    
    def plan(self, agent_input):
        return {
            "task": "analyze",
            "data": agent_input.data.get("pain_points", []),
            "analysis_type": "validation"
        }
    
    def think(self, agent_input, plan):
        return {
            "analysis": "Validating pain points",
            "method": "market_validation",
            "criteria": ["severity", "frequency", "market_size"]
        }
    
    def act(self, agent_input, plan, thoughts):
        # Simulate analysis work
        pain_points = agent_input.data.get("pain_points", [])
        
        return {
            "validated_pain_points": [
                {**pp, "validation_score": 0.9, "market_opportunity": "high"}
                for pp in pain_points
            ],
            "recommendations": [
                "Focus on automation solutions",
                "Target integration pain points"
            ],
            "priority_ranking": ["integration", "automation"]
        }


class MockReportAgent(BaseAgent):
    """Mock report generation agent."""
    
    def plan(self, agent_input):
        return {
            "task": "generate_report",
            "validated_pain_points": agent_input.data.get("validated_pain_points", []),
            "format": "executive_summary"
        }
    
    def think(self, agent_input, plan):
        return {
            "structure": ["executive_summary", "pain_points", "recommendations"],
            "target_audience": "executives"
        }
    
    def act(self, agent_input, plan, thoughts):
        validated_points = agent_input.data.get("validated_pain_points", [])
        
        return {
            "report": {
                "title": "Market Research Report",
                "date": datetime.now().isoformat(),
                "executive_summary": f"Found {len(validated_points)} validated pain points",
                "pain_points": validated_points,
                "recommendations": [
                    "Develop automation tools",
                    "Create integration platform"
                ],
                "next_steps": [
                    "Conduct user interviews",
                    "Build MVP for top pain point"
                ]
            }
        }


async def demonstrate_simple_dag():
    """Demonstrate a simple DAG workflow."""
    print("🎯 Demonstrating Simple DAG Workflow")
    print("=" * 40)
    
    # Create a simple 3-node DAG
    engine = DAGEngine(max_concurrent=2)
    
    # Define nodes
    root = DAGNode(
        name="start",
        description="Start the workflow",
        node_type="root"
    )
    
    research = DAGNode(
        name="research",
        description="Research pain points",
        node_type="agent",
        agent_name="research",
        dependencies=[root.node_id],
        inputs={"target_market": "SaaS project management", "query": "pain points"}
    )
    
    analyze = DAGNode(
        name="analyze",
        description="Analyze findings",
        node_type="agent", 
        agent_name="analysis",
        dependencies=[research.node_id],
        inputs={"analysis_type": "validation"}
    )
    
    report = DAGNode(
        name="report",
        description="Generate final report",
        node_type="agent",
        agent_name="report",
        dependencies=[analyze.node_id],
        inputs={"format": "executive_summary"}
    )
    
    # Build and validate DAG
    engine.build_from_nodes([root, research, analyze, report])
    
    print(f"✅ DAG created with {len(engine.nodes)} nodes")
    print(f"✅ DAG is valid: {engine.is_valid()}")
    print(f"✅ Ready nodes: {engine.get_ready_nodes()}")
    
    return engine


async def demonstrate_workflow_templates():
    """Demonstrate workflow templates."""
    print("\n🏗️ Demonstrating Workflow Templates")
    print("=" * 40)
    
    # Pain point discovery workflow
    print("\n📋 Pain Point Discovery Workflow:")
    pp_workflow = create_pain_point_discovery_workflow(
        target_market="SaaS project management tools",
        research_focus="comprehensive",
        max_pain_points=10,
        validation_depth="moderate"
    )
    
    print(f"   ✅ Nodes: {len(pp_workflow.nodes)}")
    print(f"   ✅ Valid: {pp_workflow.is_valid()}")
    
    # Market research workflow
    print("\n📊 Market Research Workflow:")
    mr_workflow = create_market_research_workflow(
        market="e-commerce analytics platforms",
        research_scope="comprehensive",
        include_competitors=True,
        include_trends=True,
        include_customer_segments=True
    )
    
    print(f"   ✅ Nodes: {len(mr_workflow.nodes)}")
    print(f"   ✅ Valid: {mr_workflow.is_valid()}")
    
    return pp_workflow, mr_workflow


async def demonstrate_orchestrator():
    """Demonstrate the high-level orchestrator."""
    print("\n🎛️ Demonstrating Orchestrator")
    print("=" * 40)
    
    # Get orchestrator instance
    orchestrator = get_orchestrator()
    
    # Register our mock agents
    registry = orchestrator.registry
    registry.register_agent(MockResearchAgent, "research")
    registry.register_agent(MockAnalysisAgent, "analysis") 
    registry.register_agent(MockReportAgent, "report")
    
    print("✅ Registered mock agents")
    
    # Create a workflow
    execution_id = await orchestrator.create_workflow(
        "pain_point_discovery",
        {
            "target_market": "SaaS project management",
            "research_focus": "quick",
            "max_pain_points": 5,
            "validation_depth": "light"
        }
    )
    
    print(f"✅ Created workflow: {execution_id}")
    
    # List workflows
    workflows = orchestrator.list_workflows()
    print(f"✅ Active workflows: {len(workflows)}")
    
    # Get workflow status
    status = orchestrator.get_workflow_status(execution_id)
    print(f"✅ Workflow status: {status['status']}")
    
    return orchestrator, execution_id


async def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n⚡ Demonstrating Convenience Functions")
    print("=" * 40)
    
    # Run pain point discovery using convenience function
    print("\n🔍 Running pain point discovery...")
    try:
        results = await run_pain_point_discovery(
            target_market="remote work tools",
            research_focus="focused",
            max_pain_points=3
        )
        
        print("✅ Pain point discovery completed")
        print(f"   📊 Found {len(results.get('final_report', {}).get('pain_points', []))} pain points")
        
    except Exception as e:
        print(f"⚠️  Pain point discovery demo: {e}")
    
    # Run market research using convenience function
    print("\n📈 Running market research...")
    try:
        results = await run_market_research(
            market="AI writing assistants",
            research_scope="quick"
        )
        
        print("✅ Market research completed")
        print(f"   📊 Research scope: {results.get('market_research_report', {}).get('report_type', 'unknown')}")
        
    except Exception as e:
        print(f"⚠️  Market research demo: {e}")


async def demonstrate_checkpoint_system():
    """Demonstrate checkpoint and resume functionality."""
    print("\n💾 Demonstrating Checkpoint System")
    print("=" * 40)
    
    from memory.checkpoints import CheckpointManager
    
    checkpoint_manager = CheckpointManager()
    
    # Create a checkpoint
    checkpoint_id = await checkpoint_manager.create_checkpoint(
        "demo_execution",
        {
            "workflow_type": "pain_point_discovery",
            "progress": 50,
            "current_node": "analysis",
            "data": {"pain_points_found": 3}
        }
    )
    
    print(f"✅ Created checkpoint: {checkpoint_id}")
    
    # List checkpoints
    checkpoints = await checkpoint_manager.list_checkpoints("demo_execution")
    print(f"✅ Found {len(checkpoints)} checkpoints")
    
    # Restore checkpoint
    restored = await checkpoint_manager.restore_checkpoint("demo_execution")
    if restored:
        print(f"✅ Restored checkpoint data: {restored.state['workflow_type']}")
    
    return checkpoint_manager


async def demonstrate_context_sharing():
    """Demonstrate context sharing between agents."""
    print("\n🔄 Demonstrating Context Sharing")
    print("=" * 40)
    
    from memory.context import get_context_manager
    
    context_manager = get_context_manager()
    context = context_manager.get_or_create_context("demo_context")
    
    # Share data between agents
    context.set("market_data", {"size": "$2.3B", "growth": "15%"}, "research_agent")
    context.set("pain_points", ["integration", "automation"], "analysis_agent")
    
    # Retrieve shared data
    market_data = context.get("market_data")
    pain_points = context.get("pain_points")
    
    print(f"✅ Shared market data: {market_data}")
    print(f"✅ Shared pain points: {pain_points}")
    
    # Demonstrate agent-specific context
    from memory.context import AgentContext
    agent_context = AgentContext("demo_agent", "demo_context")
    
    agent_context.set("my_data", "agent_specific_value")
    retrieved = agent_context.get("my_data")
    
    print(f"✅ Agent-specific context: {retrieved}")
    
    return context


async def main():
    """Run all demonstrations."""
    print("🚀 ScoutAgent Phase 3 Demonstration")
    print("=" * 50)
    
    try:
        # Register mock agents
        from agents.base import get_registry
        registry = get_registry()
        registry.register_agent(MockResearchAgent, "research")
        registry.register_agent(MockAnalysisAgent, "analysis")
        registry.register_agent(MockReportAgent, "report")
        
        print("✅ Mock agents registered")
        
        # Run demonstrations
        await demonstrate_simple_dag()
        await demonstrate_workflow_templates()
        await demonstrate_orchestrator()
        await demonstrate_convenience_functions()
        await demonstrate_checkpoint_system()
        await demonstrate_context_sharing()
        
        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Run the test suite: python test_phase3.py")
        print("2. Explore the DAG engine interactively")
        print("3. Create custom workflow templates")
        print("4. Implement real agent classes")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
