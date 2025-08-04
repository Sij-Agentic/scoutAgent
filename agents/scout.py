"""
ScoutAgent - Pain Point Discovery Agent

This agent specializes in discovering pain points in target markets using
web research, social media analysis, and user feedback collection.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .research_agent import ResearchAgent
from ..config import get_config


@dataclass
class PainPoint:
    """Represents a discovered pain point."""
    description: str
    severity: str  # low, medium, high, critical
    market: str
    source: str
    evidence: List[str]
    frequency: int
    impact_score: float
    tags: List[str]
    discovered_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoutInput(AgentInput):
    """Input for ScoutAgent."""
    target_market: str
    research_scope: str = "comprehensive"  # quick, focused, comprehensive
    max_pain_points: int = 10
    sources: List[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = ["reddit", "twitter", "forums", "reviews", "blogs"]
        if self.keywords is None:
            self.keywords = ["pain point", "problem", "frustration", "issue"]


@dataclass
class ScoutOutput(AgentOutput):
    """Output from ScoutAgent."""
    pain_points: List[PainPoint]
    total_discovered: int
    market_summary: str
    confidence_score: float
    sources_used: List[str]
    research_duration: float


class ScoutAgent(BaseAgent):
    """
    ScoutAgent for discovering pain points in target markets.
    
    Uses web research, social media analysis, and user feedback
    to identify and categorize pain points with evidence.
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)
        self.research_agent = ResearchAgent()
        self.config = get_config()
    
    async def plan(self, input_data: ScoutInput) -> Dict[str, Any]:
        """Plan the pain point discovery process."""
        self.logger.info(f"Planning pain point discovery for market: {input_data.target_market}")
        
        plan = {
            "phases": [
                "market_research",
                "pain_point_extraction", 
                "evidence_collection",
                "validation",
                "categorization"
            ],
            "sources": input_data.sources,
            "keywords": input_data.keywords,
            "expected_duration": 300,  # 5 minutes
            "max_pain_points": input_data.max_pain_points
        }
        
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: ScoutInput) -> Dict[str, Any]:
        """Analyze research data to identify pain points."""
        self.logger.info("Thinking about discovered pain points...")
        
        # Use research agent to gather market data
        research_input = {
            "query": f"pain points {input_data.target_market}",
            "sources": input_data.sources,
            "max_results": 50,
            "include_sentiment": True
        }
        
        research_results = await self.research_agent.execute(research_input)
        
        # Extract pain points from research
        pain_points = self._extract_pain_points(research_results, input_data.target_market)
        
        analysis = {
            "total_sources_analyzed": len(research_results.get("sources", [])),
            "pain_points_found": len(pain_points),
            "confidence_factors": [
                "multiple_sources",
                "consistent_patterns",
                "high_severity_indicators"
            ],
            "next_steps": ["validate", "categorize", "prioritize"]
        }
        
        return analysis
    
    async def act(self, input_data: ScoutInput) -> ScoutOutput:
        """Execute pain point discovery and return results."""
        self.logger.info("Executing pain point discovery...")
        
        start_time = datetime.now()
        
        # Execute research
        research_input = {
            "query": f"common problems {input_data.target_market}",
            "sources": input_data.sources,
            "max_results": 100,
            "include_sentiment": True
        }
        
        research_results = await self.research_agent.execute(research_input)
        
        # Extract and validate pain points
        pain_points = self._extract_pain_points(research_results, input_data.target_market)
        validated_pain_points = self._validate_pain_points(pain_points)
        
        # Generate market summary
        market_summary = self._generate_market_summary(
            validated_pain_points, 
            input_data.target_market
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            validated_pain_points,
            research_results
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ScoutOutput(
            pain_points=validated_pain_points,
            total_discovered=len(validated_pain_points),
            market_summary=market_summary,
            confidence_score=confidence_score,
            sources_used=input_data.sources,
            research_duration=duration
        )
    
    def _extract_pain_points(self, research_results: Dict[str, Any], market: str) -> List[PainPoint]:
        """Extract pain points from research data."""
        pain_points = []
        
        # Mock extraction - in real implementation, use NLP
        mock_pain_points = [
            {
                "description": "Complex setup process takes too long",
                "severity": "high",
                "source": "user_reviews",
                "evidence": ["Review 1", "Forum post 3"],
                "frequency": 15,
                "impact_score": 8.5,
                "tags": ["onboarding", "complexity"]
            },
            {
                "description": "Integration with existing tools is limited",
                "severity": "medium",
                "source": "reddit",
                "evidence": ["Reddit thread", "Twitter mention"],
                "frequency": 12,
                "impact_score": 7.2,
                "tags": ["integration", "compatibility"]
            },
            {
                "description": "Pricing is too high for small teams",
                "severity": "high",
                "source": "forums",
                "evidence": ["Forum discussion", "Blog comment"],
                "frequency": 20,
                "impact_score": 9.1,
                "tags": ["pricing", "accessibility"]
            }
        ]
        
        for pp_data in mock_pain_points:
            pain_point = PainPoint(
                description=pp_data["description"],
                severity=pp_data["severity"],
                market=market,
                source=pp_data["source"],
                evidence=pp_data["evidence"],
                frequency=pp_data["frequency"],
                impact_score=pp_data["impact_score"],
                tags=pp_data["tags"],
                discovered_at=datetime.now().isoformat()
            )
            pain_points.append(pain_point)
        
        return pain_points[:10]  # Limit to max_pain_points
    
    def _validate_pain_points(self, pain_points: List[PainPoint]) -> List[PainPoint]:
        """Validate and filter pain points."""
        validated = []
        
        for pp in pain_points:
            # Simple validation criteria
            if pp.frequency >= 3 and pp.impact_score >= 5.0:
                validated.append(pp)
        
        return validated
    
    def _generate_market_summary(self, pain_points: List[PainPoint], market: str) -> str:
        """Generate a summary of the pain point discovery."""
        if not pain_points:
            return f"No significant pain points discovered in {market}"
        
        high_severity = [pp for pp in pain_points if pp.severity == "high"]
        medium_severity = [pp for pp in pain_points if pp.severity == "medium"]
        
        summary = f"""
Pain Point Discovery Summary for {market}:
- Total pain points discovered: {len(pain_points)}
- High severity: {len(high_severity)}
- Medium severity: {len(medium_severity)}
- Key themes: {', '.join(set(tag for pp in pain_points for tag in pp.tags))}
- Average impact score: {sum(pp.impact_score for pp in pain_points) / len(pain_points):.1f}
        """.strip()
        
        return summary
    
    def _calculate_confidence_score(self, pain_points: List[PainPoint], research_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and consistency."""
        if not pain_points:
            return 0.0
        
        # Base score from number of sources
        sources_count = len(research_results.get("sources", []))
        base_score = min(sources_count * 0.1, 0.4)
        
        # Score from pain point quality
        avg_impact = sum(pp.impact_score for pp in pain_points) / len(pain_points)
        quality_score = min(avg_impact / 10.0, 0.3)
        
        # Score from evidence
        evidence_score = min(len(pain_points) * 0.05, 0.3)
        
        return min(base_score + quality_score + evidence_score, 1.0)


# Register the agent
from .base import register_agent
register_agent("scout", ScoutAgent)
