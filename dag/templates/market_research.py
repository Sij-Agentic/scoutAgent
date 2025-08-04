"""
Market Research Workflow Template.

This template implements a comprehensive market research workflow:
research → analysis → validation → report
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from dag.node import DAGNode, NodeType, NodeConfig
from dag.engine import DAGEngine


def create_market_research_workflow(
    market: str,
    research_scope: str = "comprehensive",
    geographic_focus: Optional[str] = None,
    include_competitors: bool = True,
    include_trends: bool = True,
    include_customer_segments: bool = True
) -> DAGEngine:
    """
    Create a comprehensive market research workflow.
    
    Args:
        market: The market/industry to research (e.g., "SaaS project management", "e-commerce")
        research_scope: "comprehensive", "focused", or "quick"
        geographic_focus: Specific geographic region to focus on
        include_competitors: Whether to include competitive analysis
        include_trends: Whether to include market trends analysis
        include_customer_segments: Whether to include customer segmentation
    
    Returns:
        Configured DAGEngine ready for execution
    """
    
    dag_engine = DAGEngine(max_concurrent=4)
    
    # Determine timeout based on scope
    timeout_map = {
        "comprehensive": 900,
        "focused": 600,
        "quick": 300
    }
    base_timeout = timeout_map.get(research_scope, 600)
    
    # Root node
    root_node = DAGNode(
        name="market_research_root",
        description=f"Market research for: {market}",
        node_type=NodeType.ROOT,
        inputs={
            "market": market,
            "research_scope": research_scope,
            "geographic_focus": geographic_focus,
            "include_competitors": include_competitors,
            "include_trends": include_trends,
            "include_customer_segments": include_customer_segments,
            "start_time": datetime.now().isoformat()
        }
    )
    
    # Market Overview Research
    overview_node = DAGNode(
        name="market_overview",
        description="Research market overview and size",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[root_node.node_id],
        inputs={
            "task": "market_overview",
            "market": market,
            "geographic_focus": geographic_focus,
            "research_depth": research_scope,
            "include": [
                "market_size",
                "growth_rate",
                "key_segments",
                "major_players",
                "market_maturity"
            ]
        },
        config=NodeConfig(
            timeout_seconds=base_timeout,
            retry_count=2,
            metadata={"phase": "overview"}
        )
    )
    
    # Customer Research
    customer_node = DAGNode(
        name="customer_research",
        description="Research customer needs and segments",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[root_node.node_id],
        inputs={
            "task": "customer_research",
            "market": market,
            "include_segments": include_customer_segments,
            "research_depth": research_scope,
            "include": [
                "customer_segments",
                "pain_points",
                "buying_criteria",
                "decision_makers",
                "budget_ranges"
            ]
        },
        config=NodeConfig(
            timeout_seconds=base_timeout,
            retry_count=2,
            metadata={"phase": "customer"}
        )
    )
    
    # Competitive Analysis (if enabled)
    competitor_nodes = []
    if include_competitors:
        competitor_research_node = DAGNode(
            name="competitor_research",
            description="Research key competitors",
            node_type=NodeType.AGENT,
            agent_name="research",
            dependencies=[root_node.node_id],
            inputs={
                "task": "competitor_analysis",
                "market": market,
                "research_depth": research_scope,
                "include": [
                    "market_leaders",
                    "emerging_players",
                    "feature_comparison",
                    "pricing_analysis",
                    "market_share"
                ]
            },
            config=NodeConfig(
                timeout_seconds=base_timeout,
                retry_count=2,
                metadata={"phase": "competitive"}
            )
        )
        competitor_nodes.append(competitor_research_node)
    
    # Market Trends Analysis (if enabled)
    trend_nodes = []
    if include_trends:
        trends_node = DAGNode(
            name="market_trends",
            description="Analyze market trends and future outlook",
            node_type=NodeType.AGENT,
            agent_name="research",
            dependencies=[root_node.node_id],
            inputs={
                "task": "trend_analysis",
                "market": market,
                "research_depth": research_scope,
                "include": [
                    "technology_trends",
                    "regulatory_changes",
                    "consumer_behavior",
                    "market_disruptions",
                    "future_projections"
                ]
            },
            config=NodeConfig(
                timeout_seconds=base_timeout,
                retry_count=2,
                metadata={"phase": "trends"}
            )
        )
        trend_nodes.append(trends_node)
    
    # Wait for all research nodes to complete
    research_dependencies = [overview_node.node_id, customer_node.node_id]
    research_dependencies.extend([n.node_id for n in competitor_nodes])
    research_dependencies.extend([n.node_id for n in trend_nodes])
    
    # Market Analysis
    analysis_node = DAGNode(
        name="market_analysis",
        description="Comprehensive market analysis",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=research_dependencies,
        inputs={
            "task": "market_analysis",
            "analysis_type": "comprehensive",
            "include": [
                "market_attractiveness",
                "competitive_intensity",
                "customer_opportunities",
                "barriers_to_entry",
                "market_gaps"
            ]
        },
        config=NodeConfig(
            timeout_seconds=base_timeout + 300,
            retry_count=2,
            metadata={"phase": "analysis"}
        )
    )
    
    # Validation
    validation_node = DAGNode(
        name="market_validation",
        description="Validate market insights and findings",
        node_type=NodeType.AGENT,
        agent_name="research",
        dependencies=[analysis_node.node_id],
        inputs={
            "task": "validate_market_insights",
            "validation_type": "cross_verification",
            "include": [
                "data_verification",
                "source_credibility",
                "finding_consistency",
                "market_expert_validation"
            ]
        },
        config=NodeConfig(
            timeout_seconds=base_timeout,
            retry_count=2,
            metadata={"phase": "validation"}
        )
    )
    
    # Final Report
    report_node = DAGNode(
        name="market_research_report",
        description="Generate comprehensive market research report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[validation_node.node_id],
        inputs={
            "task": "generate_market_report",
            "report_type": "comprehensive_market_research",
            "format": "executive_summary",
            "include_sections": [
                "executive_summary",
                "market_overview",
                "customer_analysis",
                "competitive_landscape",
                "trend_analysis",
                "opportunities",
                "risks",
                "recommendations"
            ]
        },
        config=NodeConfig(
            timeout_seconds=base_timeout + 200,
            retry_count=2,
            metadata={"phase": "reporting"}
        )
    )
    
    # Build nodes list
    nodes = [root_node, overview_node, customer_node]
    nodes.extend(competitor_nodes)
    nodes.extend(trend_nodes)
    nodes.extend([analysis_node, validation_node, report_node])
    
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine


def create_competitor_analysis_workflow(
    competitors: List[str],
    analysis_depth: str = "detailed"
) -> DAGEngine:
    """
    Create a focused competitor analysis workflow.
    
    Args:
        competitors: List of competitor names/companies to analyze
        analysis_depth: "high_level", "detailed", or "deep_dive"
    
    Returns:
        Configured DAGEngine for competitor analysis
    """
    
    dag_engine = DAGEngine(max_concurrent=len(competitors))
    
    # Root node
    root_node = DAGNode(
        name="competitor_analysis_root",
        description=f"Competitive analysis: {', '.join(competitors[:3])}...",
        node_type=NodeType.ROOT,
        inputs={
            "competitors": competitors,
            "analysis_depth": analysis_depth,
            "analysis_date": datetime.now().isoformat()
        }
    )
    
    # Research each competitor
    research_nodes = []
    for competitor in competitors:
        research_node = DAGNode(
            name=f"research_{competitor.lower().replace(' ', '_')}",
            description=f"Research {competitor}",
            node_type=NodeType.AGENT,
            agent_name="research",
            dependencies=[root_node.node_id],
            inputs={
                "task": "competitor_deep_dive",
                "competitor": competitor,
                "analysis_depth": analysis_depth,
                "include": [
                    "company_overview",
                    "products_services",
                    "pricing_strategy",
                    "target_market",
                    "key_features",
                    "customer_reviews",
                    "market_position",
                    "strengths_weaknesses"
                ]
            },
            config=NodeConfig(
                timeout_seconds=600 if analysis_depth == "deep_dive" else 400,
                retry_count=2,
                metadata={"competitor": competitor}
            )
        )
        research_nodes.append(research_node)
    
    # Comparative analysis
    comparison_node = DAGNode(
        name="competitive_comparison",
        description="Compare all competitors",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[node.node_id for node in research_nodes],
        inputs={
            "task": "competitive_comparison",
            "analysis_type": "side_by_side",
            "include": [
                "feature_comparison",
                "pricing_comparison",
                "market_positioning",
                "competitive_advantages",
                "market_gaps",
                "opportunity_areas"
            ]
        },
        config=NodeConfig(
            timeout_seconds=800,
            retry_count=2,
            metadata={"phase": "comparison"}
        )
    )
    
    # Strategic insights
    insights_node = DAGNode(
        name="strategic_insights",
        description="Generate strategic insights and recommendations",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[comparison_node.node_id],
        inputs={
            "task": "generate_strategic_insights",
            "insight_type": "competitive_strategy",
            "include": [
                "market_positioning",
                "differentiation_opportunities",
                "pricing_strategy",
                "feature_gaps",
                "go_to_market_strategy"
            ]
        },
        config=NodeConfig(
            timeout_seconds=600,
            retry_count=2,
            metadata={"phase": "insights"}
        )
    )
    
    # Final report
    report_node = DAGNode(
        name="competitor_report",
        description="Generate comprehensive competitor analysis report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[insights_node.node_id],
        inputs={
            "task": "generate_competitor_report",
            "report_type": "competitive_analysis",
            "format": "executive_summary",
            "include_sections": [
                "executive_summary",
                "competitor_profiles",
                "competitive_comparison",
                "market_positioning",
                "strategic_recommendations",
                "opportunity_analysis"
            ]
        },
        config=NodeConfig(
            timeout_seconds=500,
            retry_count=2,
            metadata={"phase": "reporting"}
        )
    )
    
    # Build nodes list
    nodes = [root_node] + research_nodes + [comparison_node, insights_node, report_node]
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine


def create_customer_research_workflow(
    target_market: str,
    customer_segments: List[str],
    research_methods: List[str] = None
) -> DAGEngine:
    """
    Create a customer research workflow.
    
    Args:
        target_market: The market/industry
        customer_segments: List of customer segments to research
        research_methods: Research methods to use
    
    Returns:
        Configured DAGEngine for customer research
    """
    
    if research_methods is None:
        research_methods = ["surveys", "interviews", "social_media", "reviews"]
    
    dag_engine = DAGEngine(max_concurrent=3)
    
    # Root node
    root_node = DAGNode(
        name="customer_research_root",
        description=f"Customer research for {target_market}",
        node_type=NodeType.ROOT,
        inputs={
            "target_market": target_market,
            "customer_segments": customer_segments,
            "research_methods": research_methods
        }
    )
    
    # Research each segment
    segment_nodes = []
    for segment in customer_segments:
        segment_node = DAGNode(
            name=f"research_{segment.lower().replace(' ', '_')}",
            description=f"Research {segment} segment",
            node_type=NodeType.AGENT,
            agent_name="research",
            dependencies=[root_node.node_id],
            inputs={
                "task": "customer_segment_research",
                "segment": segment,
                "market": target_market,
                "methods": research_methods,
                "include": [
                    "demographics",
                    "pain_points",
                    "buying_behavior",
                    "preferences",
                    "budget_ranges",
                    "decision_factors"
                ]
            },
            config=NodeConfig(timeout_seconds=500, retry_count=2)
        )
        segment_nodes.append(segment_node)
    
    # Cross-segment analysis
    cross_segment_node = DAGNode(
        name="cross_segment_analysis",
        description="Analyze patterns across customer segments",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[node.node_id for node in segment_nodes],
        inputs={
            "task": "cross_segment_analysis",
            "analysis_type": "segment_comparison",
            "include": [
                "common_pain_points",
                "segment_differences",
                "market_sizing",
                "prioritization",
                "targeting_strategy"
            ]
        },
        config=NodeConfig(timeout_seconds=600, retry_count=2)
    )
    
    # Final report
    report_node = DAGNode(
        name="customer_research_report",
        description="Generate customer research report",
        node_type=NodeType.AGENT,
        agent_name="analysis",
        dependencies=[cross_segment_node.node_id],
        inputs={
            "task": "generate_customer_report",
            "report_type": "customer_research",
            "format": "comprehensive"
        },
        config=NodeConfig(timeout_seconds=400, retry_count=2)
    )
    
    # Build nodes list
    nodes = [root_node] + segment_nodes + [cross_segment_node, report_node]
    dag_engine.build_from_nodes(nodes)
    
    return dag_engine
