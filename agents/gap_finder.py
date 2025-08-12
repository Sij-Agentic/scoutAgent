"""
GapFinderAgent - Market Gap Analysis Agent

This agent specializes in analyzing market gaps and opportunities
based on validated pain points and market research.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from llm.utils import LLMAgentMixin
from .analysis_agent import AnalysisAgent
from .research_agent import ResearchAgent
from config import get_config


@dataclass
class MarketGap:
    """Represents a discovered market gap."""
    gap_description: str
    market_size: float  # in USD
    competition_level: str  # low, medium, high
    opportunity_score: float  # 0-100
    target_segments: List[str]
    solution_ideas: List[str]
    barriers_to_entry: List[str]
    estimated_tam: float  # Total Addressable Market
    estimated_sam: float  # Serviceable Available Market
    estimated_som: float  # Serviceable Obtainable Market
    risk_factors: List[str]
    timeline_to_market: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GapFinderInput:
    """Input for GapFinderAgent."""
    validated_pain_points: List[Dict[str, Any]]  # From ValidatorAgent
    market_context: str
    analysis_scope: str = "comprehensive"  # quick, focused, comprehensive
    include_competitive_analysis: bool = True
    include_market_sizing: bool = True
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.validated_pain_points:
            raise ValueError("Must provide validated pain points for gap analysis")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GapFinderOutput:
    """Output from GapFinderAgent."""
    market_gaps: List[MarketGap]
    prioritized_opportunities: List[Dict[str, Any]]
    market_analysis: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    result: Any = None
    metadata: Dict[str, Any] = None
    logs: List[str] = None
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.logs is None:
            self.logs = []


class GapFinderAgent(BaseAgent, LLMAgentMixin):
    """
    GapFinderAgent for analyzing market gaps and opportunities.
    
    Uses comprehensive market analysis, competitive research,
    and financial modeling to identify and prioritize market gaps.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="gap_finder", agent_id=agent_id)
        LLMAgentMixin.__init__(self)
        self.analysis_agent = AnalysisAgent()
        self.research_agent = ResearchAgent()
        self.config = get_config()
        self.name = "gap_finder"
    
    async def plan(self, input_data: GapFinderInput) -> Dict[str, Any]:
        """Plan the market gap analysis process."""
        self.logger.info(f"Planning market gap analysis for {len(input_data.validated_pain_points)} pain points")
        
        plan = {
            "phases": [
                "pain_point_clustering",
                "market_research",
                "competitive_analysis",
                "market_sizing",
                "opportunity_scoring",
                "risk_assessment",
                "prioritization"
            ],
            "analysis_scope": input_data.analysis_scope,
            "include_competitive_analysis": input_data.include_competitive_analysis,
            "include_market_sizing": input_data.include_market_sizing,
            "expected_duration": 900,  # 15 minutes
            "pain_point_count": len(input_data.validated_pain_points)
        }
        
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: GapFinderInput) -> Dict[str, Any]:
        """Analyze pain points to identify market gaps."""
        self.logger.info("Analyzing market gaps and opportunities...")
        
        # Cluster similar pain points
        clusters = await self._cluster_pain_points(input_data.validated_pain_points)
        
        # Analyze market context
        market_context = await self._analyze_market_context(
            input_data.market_context,
            clusters
        )
        
        # Prepare gap analysis strategy
        analysis_strategy = {
            "cluster_count": len(clusters),
            "market_segments": list(set(p.get("market", "") for p in input_data.validated_pain_points)),
            "analysis_approach": self._determine_analysis_approach(input_data.analysis_scope),
            "competitive_intensity": "medium",  # Will be updated
            "market_maturity": "growing",  # Will be analyzed
            "expected_gaps": min(len(clusters), 5)
        }
        
        return analysis_strategy
    
    async def act(self, input_data: GapFinderInput) -> GapFinderOutput:
        """Execute market gap analysis and return results."""
        self.logger.info("Executing market gap analysis...")
        
        start_time = datetime.now()
        
        # Cluster pain points into market opportunities
        clusters = await self._cluster_pain_points(input_data.validated_pain_points)
        
        # Analyze each cluster for market gaps
        market_gaps = []
        for cluster in clusters:
            gap = await self._analyze_market_gap(
                cluster,
                input_data.market_context,
                input_data.include_competitive_analysis,
                input_data.include_market_sizing
            )
            market_gaps.append(gap)
        
        # Analyze competitive landscape
        competitive_landscape = await self._analyze_competitive_landscape(
            market_gaps,
            input_data.market_context
        )
        
        # Prioritize opportunities
        prioritized_opportunities = self._prioritize_opportunities(market_gaps)
        
        # Generate market insights
        market_analysis = await self._generate_market_analysis(
            market_gaps,
            competitive_landscape
        )
        
        # Risk assessment
        risk_assessment = self._assess_risks(market_gaps, competitive_landscape)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prioritized_opportunities,
            risk_assessment
        )
        
        return GapFinderOutput(
            market_gaps=market_gaps,
            prioritized_opportunities=prioritized_opportunities,
            market_analysis=market_analysis,
            competitive_landscape=competitive_landscape,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )

    async def _analyze_market_context(self, market_context: str, *args, **kwargs) -> Dict[str, Any]:
        """Analyze the overall market context and trends."""
        # Mock implementation for validation
        return {
            "market_size": "large",
            "growth_rate": "high",
            "maturity": "growing",
            "key_trends": ["digital_transformation", "automation", "ai_adoption"],
            "barriers": ["technical_complexity", "regulatory_compliance", "market_competition"]
        }
    
    async def _cluster_pain_points(self, pain_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar pain points into market opportunities."""
        clusters = []
        
        # Mock clustering - in real implementation, use NLP
        cluster_themes = [
            {"theme": "onboarding_complexity", "pain_points": []},
            {"theme": "integration_challenges", "pain_points": []},
            {"theme": "pricing_barriers", "pain_points": []},
            {"theme": "performance_issues", "pain_points": []},
            {"theme": "usability_problems", "pain_points": []}
        ]
        
        for point in pain_points:
            description = point.get("description", "").lower()
            tags = point.get("tags", [])
            
            # Simple clustering based on keywords
            if any(word in description for word in ["setup", "onboarding", "complex"]):
                cluster_themes[0]["pain_points"].append(point)
            elif any(word in description for word in ["integration", "connect", "sync"]):
                cluster_themes[1]["pain_points"].append(point)
            elif any(word in description for word in ["price", "cost", "expensive"]):
                cluster_themes[2]["pain_points"].append(point)
            elif any(word in description for word in ["slow", "performance", "lag"]):
                cluster_themes[3]["pain_points"].append(point)
            else:
                cluster_themes[4]["pain_points"].append(point)
        
        # Filter clusters with actual pain points
        return [cluster for cluster in cluster_themes if cluster["pain_points"]]
    
    async def _analyze_market_gap(self, cluster: Dict[str, Any], market_context: str,
                                include_competitive: bool, include_sizing: bool) -> MarketGap:
        """Analyze a specific market gap."""
        theme = cluster["theme"]
        pain_points = cluster["pain_points"]
        
        # Market research
        market_data = await self._research_market_for_theme(theme, market_context)
        
        # Competitive analysis
        competitive_data = {}
        if include_competitive:
            competitive_data = await self._analyze_competition_for_theme(theme, market_context)
        
        # Market sizing
        market_sizing = {}
        if include_sizing:
            market_sizing = await self._size_market_for_theme(theme, market_context, pain_points)
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(
            market_data,
            competitive_data,
            len(pain_points)
        )
        
        # Generate solution ideas
        solution_ideas = self._generate_solution_ideas(theme, pain_points)
        
        # Assess barriers
        barriers = self._assess_barriers_to_entry(theme, market_context)
        
        # Risk factors
        risk_factors = self._identify_risk_factors(theme, market_context)
        
        return MarketGap(
            gap_description=f"Market gap in {theme} for {market_context}",
            market_size=market_sizing.get("total_market", 25000000),
            competition_level=competitive_data.get("competition_level", "medium"),
            opportunity_score=opportunity_score,
            target_segments=self._identify_target_segments(theme, pain_points),
            solution_ideas=solution_ideas,
            barriers_to_entry=barriers,
            estimated_tam=market_sizing.get("tam", 50000000),
            estimated_sam=market_sizing.get("sam", 25000000),
            estimated_som=market_sizing.get("som", 5000000),
            risk_factors=risk_factors,
            timeline_to_market="6-12 months"
        )
    
    async def _research_market_for_theme(self, theme: str, market_context: str) -> Dict[str, Any]:
        """Research market data for a specific theme."""
        # Mock research
        theme_markets = {
            "onboarding_complexity": {"size": 30000000, "growth": 0.2},
            "integration_challenges": {"size": 45000000, "growth": 0.15},
            "pricing_barriers": {"size": 25000000, "growth": 0.18},
            "performance_issues": {"size": 35000000, "growth": 0.12},
            "usability_problems": {"size": 40000000, "growth": 0.22}
        }
        
        return theme_markets.get(theme, {"size": 20000000, "growth": 0.15})
    
    async def _analyze_competition_for_theme(self, theme: str, market_context: str) -> Dict[str, Any]:
        """Analyze competition for a specific theme."""
        # Mock competitive analysis
        theme_competition = {
            "onboarding_complexity": {"level": "medium", "players": 8},
            "integration_challenges": {"level": "high", "players": 12},
            "pricing_barriers": {"level": "low", "players": 5},
            "performance_issues": {"level": "medium", "players": 7},
            "usability_problems": {"level": "high", "players": 15}
        }
        
        data = theme_competition.get(theme, {"level": "medium", "players": 10})
        return {
            "competition_level": data["level"],
            "direct_competitors": data["players"],
            "market_saturation": data["players"] * 0.05,
            "competitive_intensity": "high" if data["players"] > 10 else "medium"
        }
    
    async def _size_market_for_theme(self, theme: str, market_context: str, 
                                   pain_points: List[Dict]) -> Dict[str, float]:
        """Size the market for a specific theme."""
        # Mock market sizing
        base_size = 50000000  # $50M base
        
        # Adjust based on pain point count and severity
        pain_point_factor = min(len(pain_points) * 0.1, 1.5)
        
        tam = base_size * pain_point_factor
        sam = tam * 0.5  # Serviceable Available Market
        som = sam * 0.2  # Serviceable Obtainable Market
        
        return {
            "tam": tam,
            "sam": sam,
            "som": som,
            "total_market": tam
        }
    
    def _calculate_opportunity_score(self, market_data: Dict, competitive_data: Dict, 
                                   pain_point_count: int) -> float:
        """Calculate opportunity score for a market gap."""
        # Market size score (0-40 points)
        market_size = market_data.get("size", 0)
        size_score = min(market_size / 1000000, 40)  # $1M = 40 points
        
        # Competition score (0-30 points, reverse scoring)
        competition_level = competitive_data.get("level", "medium")
        competition_scores = {"low": 30, "medium": 20, "high": 10}
        competition_score = competition_scores.get(competition_level, 20)
        
        # Pain point intensity score (0-20 points)
        intensity_score = min(pain_point_count * 4, 20)
        
        # Growth potential score (0-10 points)
        growth_rate = market_data.get("growth", 0.15)
        growth_score = min(growth_rate * 50, 10)
        
        total_score = size_score + competition_score + intensity_score + growth_score
        return min(total_score, 100)
    
    def _generate_solution_ideas(self, theme: str, pain_points: List[Dict]) -> List[str]:
        """Generate solution ideas for a market gap."""
        solutions = {
            "onboarding_complexity": [
                "Interactive guided tours",
                "Simplified setup wizard",
                "Template-based configuration",
                "AI-powered onboarding assistant"
            ],
            "integration_challenges": [
                "Universal API connector",
                "Pre-built integrations",
                "Integration marketplace",
                "No-code integration builder"
            ],
            "pricing_barriers": [
                "Freemium model",
                "Usage-based pricing",
                "Small business tier",
                "Open-source alternative"
            ],
            "performance_issues": [
                "Performance optimization tools",
                "Caching solutions",
                "CDN integration",
                "Monitoring dashboard"
            ],
            "usability_problems": [
                "UI/UX redesign",
                "User testing platform",
                "Accessibility improvements",
                "Mobile-first design"
            ]
        }
        
        return solutions.get(theme, ["Custom solution development"])
    
    def _assess_barriers_to_entry(self, theme: str, market_context: str) -> List[str]:
        """Assess barriers to entry for a market gap."""
        common_barriers = [
            "Technical complexity",
            "Customer acquisition costs",
            "Integration requirements",
            "Regulatory compliance",
            "Brand recognition"
        ]
        
        theme_barriers = {
            "onboarding_complexity": ["User behavior change", "Integration complexity"],
            "integration_challenges": ["API limitations", "Legacy system compatibility"],
            "pricing_barriers": ["Competitive pricing pressure", "Cost structure optimization"],
            "performance_issues": ["Technical expertise required", "Infrastructure costs"],
            "usability_problems": ["Design expertise", "User research requirements"]
        }
        
        return common_barriers + theme_barriers.get(theme, [])
    
    def _identify_risk_factors(self, theme: str, market_context: str) -> List[str]:
        """Identify risk factors for a market gap."""
        common_risks = [
            "Market saturation",
            "Competitive response",
            "Technology changes",
            "Customer acquisition",
            "Scaling challenges"
        ]
        
        return common_risks
    
    def _identify_target_segments(self, theme: str, pain_points: List[Dict]) -> List[str]:
        """Identify target market segments."""
        segments = {
            "onboarding_complexity": ["SMBs", "Startups", "Non-technical users"],
            "integration_challenges": ["Mid-market companies", "Enterprises", "Tech teams"],
            "pricing_barriers": ["SMBs", "Startups", "Individual users"],
            "performance_issues": ["Enterprises", "High-volume users", "Global companies"],
            "usability_problems": ["All business sizes", "Non-technical users", "Mobile users"]
        }
        
        return segments.get(theme, ["General business users"])
    
    async def _analyze_competitive_landscape(self, market_gaps: List[MarketGap], 
                                           market_context: str) -> Dict[str, Any]:
        """Analyze the competitive landscape."""
        return {
            "total_gaps": len(market_gaps),
            "competitive_intensity": "medium",
            "market_maturity": "growing",
            "key_players": ["Established players", "Startups", "Tech giants"],
            "market_dynamics": ["Consolidation", "Innovation", "Disruption"],
            "opportunity_areas": [g.gap_description for g in market_gaps]
        }
    
    def _prioritize_opportunities(self, market_gaps: List[MarketGap]) -> List[Dict[str, Any]]:
        """Prioritize market opportunities by score."""
        opportunities = []
        
        for gap in market_gaps:
            opportunities.append({
                "gap_description": gap.gap_description,
                "opportunity_score": gap.opportunity_score,
                "market_size": gap.market_size,
                "competition_level": gap.competition_level,
                "priority": "high" if gap.opportunity_score > 70 else "medium" if gap.opportunity_score > 50 else "low"
            })
        
        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x["opportunity_score"], reverse=True)
    
    async def _generate_market_analysis(self, market_gaps: List[MarketGap], 
                                      competitive_landscape: Dict) -> Dict[str, Any]:
        """Generate comprehensive market analysis."""
        total_market_size = sum(gap.market_size for gap in market_gaps)
        avg_opportunity_score = sum(gap.opportunity_score for gap in market_gaps) / len(market_gaps)
        
        return {
            "total_addressable_market": total_market_size,
            "average_opportunity_score": avg_opportunity_score,
            "market_gaps_count": len(market_gaps),
            "market_segments": list(set(segment for gap in market_gaps for segment in gap.target_segments)),
            "key_trends": ["Digital transformation", "User experience focus", "Cost optimization"],
            "growth_drivers": ["Remote work", "SaaS adoption", "SMB digitization"]
        }
    
    def _assess_risks(self, market_gaps: List[MarketGap], 
                     competitive_landscape: Dict) -> Dict[str, Any]:
        """Assess risks for market opportunities."""
        return {
            "overall_risk_level": "medium",
            "market_risks": ["Competition", "Market saturation", "Economic downturn"],
            "technical_risks": ["Technology changes", "Scalability challenges"],
            "business_risks": ["Customer acquisition", "Pricing pressure", "Regulatory changes"],
            "mitigation_strategies": ["Focus on niche", "Build moat", "Customer validation"]
        }
    
    def _generate_recommendations(self, opportunities: List[Dict], 
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if opportunities:
            top_opportunity = opportunities[0]
            recommendations.append(f"Focus on top opportunity: {top_opportunity['gap_description']}")
            recommendations.append(f"Target market size: ${top_opportunity['market_size']:,.0f}")
            
            if len(opportunities) > 3:
                recommendations.append("Consider building a platform addressing multiple gaps")
            
            recommendations.append("Validate with potential customers before full development")
            recommendations.append("Start with MVP targeting highest-scoring opportunity")
        
        return recommendations
    
    def _determine_analysis_approach(self, scope: str) -> str:
        """Determine analysis approach based on scope."""
        approaches = {
            "quick": "High-level analysis with key metrics",
            "focused": "Detailed analysis of top opportunities",
            "comprehensive": "Full market analysis with detailed modeling"
        }
        return approaches.get(scope, "focused")


# Register the agent - moved to agent_registry.py
# from .base import register_agent
# register_agent("gap_finder", GapFinderAgent)
