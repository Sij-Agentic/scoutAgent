"""
Pain Point Discovery Workflow Template.

This template implements the complete pain point scouting workflow:
scout → screener → validator → gap_finder
"""

from typing import List, Dict, Any
from datetime import datetime

from dag.node import DAGNode, NodeType, NodeConfig
from dag.engine import DAGEngine
from memory.context import get_context_manager
from memory.checkpoints import get_checkpoint_manager


def create_pain_point_discovery_workflow(
    target_market: str,
    research_focus: str,
    max_pain_points: int = 10,
    validation_depth: str = "medium"
) -> DAGEngine:
    """
    Create a complete pain point discovery workflow.
    
    Args:
        target_market: The market/industry to research
        research_focus: Specific area of focus (e.g., "SaaS onboarding", "e-commerce checkout")
        max_pain_points: Maximum number of pain points to discover
        validation_depth: "shallow", "medium", or "deep" validation
    
    Returns:
        Configured DAGEngine ready for execution
    """
    
    dag_engine = DAGEngine(max_concurrent=3)
    
    # Create root node
    root_node = DAGNode(
        name="workflow_root",
        description="Initialize pain point discovery workflow",
        node_type=NodeType.ROOT,
        inputs={
            "target_market": target_market,
            "research_focus": research_focus,
            "max_pain_points": max_pain_points,
            "validation_depth": validation_depth,
            "start_time": datetime.now().isoformat()
        }
    )
    
    # Scout Agent - Discover initial pain points
    scout_node = DAGNode(
        name="pain_point_scout",
        description="Discover potential pain points through web research",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[root_node.node_id],
        inputs={
            "task": "discover_pain_points",
            "market": target_market,
            "focus": research_focus,
            "max_results": max_pain_points * 3,  # Get more than needed for screening
            "sources": ["reddit", "twitter", "forums", "reviews", "competitor_analysis"]
        },
        config=NodeConfig(
            timeout_seconds=600,
            retry_count=2,
            metadata={"phase": "discovery"}
        )
    )
    
    # Screener Agent - Initial filtering and categorization
    screener_node = DAGNode(
        name="pain_point_screener",
        description="Screen and categorize discovered pain points",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[scout_node.node_id],
        inputs={
            "task": "screen_pain_points",
            "criteria": {
                "severity": ["high", "critical"],
                "frequency": ["common", "very_common"],
                "market_size": ["large", "growing"],
                "solution_feasibility": ["high", "medium"]
            },
            "max_output": max_pain_points
        },
        config=NodeConfig(
            timeout_seconds=300,
            retry_count=2,
            metadata={"phase": "screening"}
        )
    )
    
    # Validator Agent - Deep validation of screened pain points
    validator_node = DAGNode(
        name="pain_point_validator",
        description="Deep validation of screened pain points",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[screener_node.node_id],
        inputs={
            "task": "validate_pain_points",
            "validation_depth": validation_depth,
            "validation_criteria": [
                "market_size_verification",
                "problem_frequency_analysis",
                "competitive_landscape",
                "solution_viability",
                "monetization_potential"
            ],
            "data_sources": [
                "market_research",
                "user_interviews",
                "competitor_analysis",
                "industry_reports"
            ]
        },
        config=NodeConfig(
            timeout_seconds=900 if validation_depth == "deep" else 600,
            retry_count=2,
            metadata={"phase": "validation"}
        )
    )
    
    # Gap Finder Agent - Market gap analysis
    gap_finder_node = DAGNode(
        name="market_gap_analyzer",
        description="Analyze market gaps and opportunities",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[validator_node.node_id],
        inputs={
            "task": "find_market_gaps",
            "analysis_type": "comprehensive",
            "include": [
                "competitive_gaps",
                "underserved_segments",
                "pricing_opportunities",
                "feature_gaps",
                "distribution_gaps"
            ],
            "output_format": "structured_report"
        },
        config=NodeConfig(
            timeout_seconds=600,
            retry_count=2,
            metadata={"phase": "gap_analysis"}
        )
    )
    
    # Report Generator - Compile final report
    report_node = DAGNode(
        name="discovery_report_generator",
        description="Generate comprehensive pain point discovery report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[gap_finder_node.node_id],
        inputs={
            "task": "generate_discovery_report",
            "report_sections": [
                "executive_summary",
                "pain_point_analysis",
                "market_validation",
                "gap_analysis",
                "opportunity_ranking",
                "next_steps"
            ],
            "format": "markdown"
        },
        config=NodeConfig(
            timeout_seconds=300,
            retry_count=2,
            metadata={"phase": "reporting"}
        )
    )
    
    # Add all nodes to DAG
    nodes = [
        root_node,
        scout_node,
        screener_node,
        validator_node,
        gap_finder_node,
        report_node
    ]
    
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine


def create_focused_discovery_workflow(
    target_market: str,
    specific_problem: str,
    research_sources: List[str] = None
) -> DAGEngine:
    """
    Create a focused discovery workflow for a specific problem area.
    
    Args:
        target_market: The market/industry
        specific_problem: Specific problem to investigate
        research_sources: Custom research sources
    
    Returns:
        Configured DAGEngine for focused discovery
    """
    
    if research_sources is None:
        research_sources = ["forums", "reviews", "social_media", "support_tickets"]
    
    dag_engine = DAGEngine(max_concurrent=2)
    
    # Root node
    root_node = DAGNode(
        name="focused_discovery_root",
        description=f"Focused discovery for: {specific_problem}",
        node_type=NodeType.ROOT,
        inputs={
            "target_market": target_market,
            "specific_problem": specific_problem,
            "research_sources": research_sources
        }
    )
    
    # Targeted research
    research_node = DAGNode(
        name="targeted_research",
        description=f"Research specific problem: {specific_problem}",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[root_node.node_id],
        inputs={
            "task": "targeted_research",
            "query": specific_problem,
            "market": target_market,
            "sources": research_sources,
            "depth": "comprehensive"
        },
        config=NodeConfig(timeout_seconds=450, retry_count=2)
    )
    
    # Problem analysis
    analysis_node = DAGNode(
        name="problem_analysis",
        description="Analyze the specific problem in detail",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[research_node.node_id],
        inputs={
            "task": "analyze_problem",
            "analysis_type": "problem_deep_dive",
            "include": [
                "problem_scope",
                "affected_users",
                "current_solutions",
                "gaps_analysis",
                "opportunity_size"
            ]
        },
        config=NodeConfig(timeout_seconds=400, retry_count=2)
    )
    
    # Validation
    validation_node = DAGNode(
        name="problem_validation",
        description="Validate the problem and opportunity",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[analysis_node.node_id],
        inputs={
            "task": "validate_problem",
            "validation_type": "market_validation",
            "include": [
                "market_size",
                "willingness_to_pay",
                "competitive_analysis",
                "user_validation"
            ]
        },
        config=NodeConfig(timeout_seconds=500, retry_count=2)
    )
    
    # Final report
    report_node = DAGNode(
        name="focused_report",
        description="Generate focused discovery report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[validation_node.node_id],
        inputs={
            "task": "generate_focused_report",
            "report_type": "problem_analysis",
            "format": "detailed"
        },
        config=NodeConfig(timeout_seconds=300, retry_count=2)
    )
    
    nodes = [root_node, research_node, analysis_node, validation_node, report_node]
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine


def create_competitive_analysis_workflow(
    competitors: List[str],
    analysis_focus: str = "pain_points"
) -> DAGEngine:
    """
    Create a competitive analysis workflow focused on pain points.
    
    Args:
        competitors: List of competitor names to analyze
        analysis_focus: Focus of analysis ("pain_points", "features", "gaps")
    
    Returns:
        Configured DAGEngine for competitive analysis
    """
    
    dag_engine = DAGEngine(max_concurrent=4)
    
    # Root node
    root_node = DAGNode(
        name="competitive_analysis_root",
        description=f"Competitive analysis of: {', '.join(competitors[:3])}...",
        node_type=NodeType.ROOT,
        inputs={
            "competitors": competitors,
            "analysis_focus": analysis_focus,
            "analysis_date": datetime.now().isoformat()
        }
    )
    
    # Research each competitor
    research_nodes = []
    for i, competitor in enumerate(competitors):
        research_node = DAGNode(
            name=f"research_{competitor.lower().replace(' ', '_')}",
            description=f"Research {competitor}",
            node_type=NodeType.AGENT,
            agent_name="research",
            dependencies=[root_node.node_id],
            inputs={
                "task": "competitor_research",
                "competitor": competitor,
                "focus": analysis_focus,
                "sources": ["reviews", "forums", "social_media", "app_stores"]
            },
            config=NodeConfig(timeout_seconds=300, retry_count=2)
        )
        research_nodes.append(research_node)
    
    # Comparative analysis
    analysis_node = DAGNode(
        name="comparative_analysis",
        description="Comprehensive competitive analysis",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[node.node_id for node in research_nodes],
        inputs={
            "task": "comparative_analysis",
            "analysis_type": "competitive_comparison",
            "include": [
                "pain_point_comparison",
                "feature_gap_analysis",
                "user_satisfaction_comparison",
                "opportunity_identification"
            ]
        },
        config=NodeConfig(timeout_seconds=600, retry_count=2)
    )
    
    # Final report
    report_node = DAGNode(
        name="competitive_report",
        description="Generate competitive analysis report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[analysis_node.node_id],
        inputs={
            "task": "generate_competitive_report",
            "report_type": "competitive_analysis",
            "format": "comprehensive"
        },
        config=NodeConfig(timeout_seconds=400, retry_count=2)
    )
    
    nodes = [root_node] + research_nodes + [analysis_node, report_node]
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine
