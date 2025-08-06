#!/usr/bin/env python3
"""
Comprehensive validation script for Phase 4: Workflow-Specific Agents

This script validates all Phase 4 agents (GapFinderAgent, BuilderAgent, WriterAgent, MemoryAgent)
and their integration with the existing DAG engine and orchestration system.
"""

import asyncio
import sys
from pathlib import Path
from agents.gap_finder import GapFinderAgent, GapFinderInput
from agents.builder import BuilderAgent, BuilderInput
from agents.writer import WriterAgent, WriterInput
from services.agents.memory import MemoryService, get_memory_service


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_test_result(test_name, success, details=""):
    """Print test result with formatting."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status} {test_name}")
    if details:
        print(f"    Details: {details}")


async def test_gap_finder_agent():
    """Test the GapFinderAgent."""
    print_section("Testing GapFinderAgent")
    
    try:
        agent = GapFinderAgent()
        
        # Mock validated pain points
        validated_pain_points = [
            {
                "id": "pp_001",
                "description": "Complex onboarding process for new users",
                "severity": "high",
                "frequency": "common",
                "impact": "high",
                "tags": ["onboarding", "user_experience", "complexity"],
                "market": "SaaS",
                "confidence": 0.9
            },
            {
                "id": "pp_002", 
                "description": "Integration challenges with existing tools",
                "severity": "medium",
                "frequency": "frequent",
                "impact": "medium",
                "tags": ["integration", "compatibility", "tools"],
                "market": "SaaS",
                "confidence": 0.85
            }
        ]
        
        # Create input
        input_data = GapFinderInput(
            validated_pain_points=validated_pain_points,
            market_context="SaaS onboarding and integration solutions",
            analysis_scope="comprehensive"
        )
        
        # Test planning
        plan = await agent.plan(input_data)
        print_test_result("Planning", True, f"Generated {len(plan['phases'])} phases")
        
        # Test thinking
        strategy = await agent.think(input_data)
        print_test_result("Strategy", True, f"Cluster count: {strategy['cluster_count']}")
        
        # Test execution
        output = await agent.act(input_data)
        print_test_result("Execution", True, f"Found {len(output.market_gaps)} market gaps")
        
        # Validate output structure
        assert len(output.market_gaps) > 0, "Expected market gaps"
        assert len(output.prioritized_opportunities) > 0, "Expected prioritized opportunities"
        assert output.market_analysis, "Expected market analysis"
        
        print_test_result("GapFinderAgent Validation", True)
        return True
        
    except Exception as e:
        print_test_result("GapFinderAgent Validation", False, str(e))
        return False


async def test_builder_agent():
    """Test the BuilderAgent."""
    print_section("Testing BuilderAgent")
    
    try:
        agent = BuilderAgent()
        
        # Mock market gaps
        market_gaps = [
            {
                "gap_description": "Simplified onboarding for SaaS tools",
                "market_size": 25000000,
                "competition_level": "medium",
                "opportunity_score": 85.5,
                "target_segments": ["SMBs", "Startups"],
                "solution_ideas": ["Guided setup wizard", "Template library"],
                "barriers_to_entry": ["Technical complexity", "User adoption"],
                "estimated_tam": 50000000,
                "estimated_sam": 25000000,
                "estimated_som": 5000000,
                "risk_factors": ["Competition", "Market timing"],
                "timeline_to_market": "6-12 months"
            }
        ]
        
        # Create input
        input_data = BuilderInput(
            market_gaps=market_gaps,
            target_market="SMB SaaS market",
            solution_type="software",
            budget_range="moderate",
            timeline="3-6 months"
        )
        
        # Test planning
        plan = await agent.plan(input_data)
        print_test_result("Planning", True, f"Phases: {plan['phases']}")
        
        # Test thinking
        strategy = await agent.think(input_data)
        print_test_result("Strategy", True, f"Feasibility score: {strategy['feasibility_score']}")
        
        # Test execution
        output = await agent.act(input_data)
        print_test_result("Execution", True, f"Created {len(output.solution_prototypes)} prototypes")
        
        # Validate output
        assert len(output.solution_prototypes) > 0, "Expected solution prototypes"
        assert output.recommended_solution, "Expected recommended solution"
        assert output.implementation_roadmap, "Expected implementation roadmap"
        
        print_test_result("BuilderAgent Validation", True)
        return True
        
    except Exception as e:
        print_test_result("BuilderAgent Validation", False, str(e))
        return False


async def test_writer_agent():
    """Test the WriterAgent."""
    print_section("Testing WriterAgent")
    
    try:
        agent = WriterAgent()
        
        # Mock workflow data
        workflow_data = {
            "scout": {
                "pain_points": [
                    {"id": "pp_001", "description": "Complex onboarding", "severity": "high"},
                    {"id": "pp_002", "description": "Integration issues", "severity": "medium"}
                ]
            },
            "gap_finder": {
                "market_gaps": [
                    {"gap_description": "SaaS onboarding gap", "market_size": 25000000, "opportunity_score": 85.5}
                ]
            },
            "builder": {
                "solution_prototypes": [
                    {"solution_name": "EasyOnboard", "estimated_cost": 150000, "key_features": ["wizard", "templates"]}
                ]
            }
        }
        
        # Create input
        input_data = WriterInput(
            workflow_data=workflow_data,
            report_type="comprehensive",
            target_audience="stakeholders"
        )
        
        # Test planning
        plan = await agent.plan(input_data)
        print_test_result("Planning", True, f"Sections: {plan['phases']}")
        
        # Test thinking
        strategy = await agent.think(input_data)
        print_test_result("Strategy", True, f"Data completeness: {strategy['data_completeness']}")
        
        # Test execution
        output = await agent.act(input_data)
        print_test_result("Execution", True, f"Report length: {len(output.report)} chars")
        
        # Validate output
        assert output.report, "Expected report content"
        assert output.executive_summary, "Expected executive summary"
        assert len(output.report_sections) > 0, "Expected report sections"
        
        print_test_result("WriterAgent Validation", True)
        return True
        
    except Exception as e:
        print_test_result("WriterAgent Validation", False, str(e))
        return False


async def test_memory_agent():
    """Test the MemoryService."""
    print_section("Testing MemoryService")
    
    try:
        # Get memory service instance
        memory_service = get_memory_service()
        await memory_service._initialize(None)  # Initialize directly for testing
        await memory_service._start()
        
        # Mock workflow data
        workflow_data = {
            "scout": {
                "pain_points": [
                    {"id": "pp_001", "description": "Complex onboarding", "severity": "high", "tags": ["onboarding"]}
                ]
            },
            "gap_finder": {
                "market_gaps": [
                    {"gap_description": "SaaS onboarding gap", "market_size": 25000000, "opportunity_score": 85.5}
                ]
            }
        }
        
        # Test create memory operation
        memory_id = await memory_service.create_memory(
            content=workflow_data,
            memory_type="workflow",
            metadata={"source": "validation_test"},
            tags=["test", "workflow"]
        )
        print_test_result("Create Memory", memory_id is not None, f"Created memory with ID: {memory_id}")
        
        # Test retrieve memory
        memory = await memory_service.get_memory(memory_id)
        print_test_result("Retrieve Memory", memory is not None, "Memory retrieved successfully")
        
        # Test search memories
        results = await memory_service.search_memories("onboarding")
        print_test_result("Search Memories", len(results) > 0, f"Found {len(results)} results")
        
        # Test DAG task creation
        task = await memory_service.create_dag_task(
            operation="search",
            params={"query": "onboarding", "limit": 5}
        )
        print_test_result("DAG Task Creation", task is not None, "Created memory DAG task")
        
        await memory_service._stop()
        print_test_result("MemoryService Validation", True)
        return True
        
    except Exception as e:
        print_test_result("MemoryService Validation", False, str(e))
        return False


async def test_agent_registration():
    """Test that all agents are properly registered."""
    print_section("Testing Agent Registration")
    
    try:
        from agents.base import get_agent_class
        from service_registry import get_registry
        
        # Test traditional agents
        agents_to_test = ["gapfinder", "builder", "writer"]
        
        for agent_name in agents_to_test:
            agent_class = get_agent_class(agent_name)
            if agent_class:
                print_test_result(f"{agent_name} registration", True)
            else:
                print_test_result(f"{agent_name} registration", False, "Agent not found")
                return False
        
        # Test services
        registry = get_registry()
        services_to_test = ["memory", "code_execution"]
        
        for service_name in services_to_test:
            if registry.has_service(service_name):
                print_test_result(f"{service_name} service registration", True)
            else:
                try:
                    if service_name == "memory":
                        from services.agents.memory import get_memory_service
                        service = get_memory_service()
                        print_test_result(f"{service_name} service factory", True)
                    elif service_name == "code_execution":
                        from services.agents.code import get_code_execution_service
                        service = get_code_execution_service()
                        print_test_result(f"{service_name} service factory", True)
                except Exception as e:
                    print_test_result(f"{service_name} service access", False, str(e))
                    return False
        
        return True
        
    except Exception as e:
        print_test_result("Registration Tests", False, str(e))
        return False


async def main():
    """Main validation function."""
    print("🚀 Starting Phase 4 Validation")
    print("=" * 60)
    
    test_results = []
    
    # Test agent registration
    test_results.append(await test_agent_registration())
    
    # Test individual agents
    test_results.append(await test_gap_finder_agent())
    test_results.append(await test_builder_agent())
    test_results.append(await test_writer_agent())
    test_results.append(await test_memory_agent())
    
    # Summary
    print_section("Validation Summary")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All Phase 4 validations passed!")
        print("\nNext Steps:")
        print("1. Test agent integration with DAG engine")
        print("2. Create end-to-end workflow demonstrations")
        print("3. Implement real-time monitoring dashboard")
        print("4. Prepare for Phase 5 production features")
        return 0
    else:
        print("❌ Some validations failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
