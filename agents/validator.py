"""
ValidatorAgent - Deep Pain Point Validation Agent

This agent specializes in deeply validating discovered pain points through
market research, user interviews, competitor analysis, and data verification.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .research_agent import ResearchAgent
from .analysis_agent import AnalysisAgent
from ..config import get_config


@dataclass
class ValidationResult:
    """Result of pain point validation."""
    pain_point_id: str
    is_valid: bool
    confidence_score: float
    validation_methods: List[str]
    evidence: Dict[str, Any]
    market_size: Optional[float]
    competition_level: str  # low, medium, high
    user_demand_score: float
    business_potential: str  # low, medium, high
    validation_notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidatorInput(AgentInput):
    """Input for ValidatorAgent."""
    pain_points: List[Dict[str, Any]]  # From ScreenerAgent
    validation_depth: str = "moderate"  # quick, moderate, deep
    market_context: str = ""
    include_user_interviews: bool = True
    include_competitor_analysis: bool = True
    
    def __post_init__(self):
        if not self.pain_points:
            raise ValueError("Must provide pain points for validation")


@dataclass
class ValidatorOutput(AgentOutput):
    """Output from ValidatorAgent."""
    validated_pain_points: List[Dict[str, Any]]
    invalid_pain_points: List[Dict[str, Any]]
    validation_summary: str
    market_insights: Dict[str, Any]
    confidence_scores: Dict[str, float]
    recommendations: List[str]


class ValidatorAgent(BaseAgent):
    """
    ValidatorAgent for deep validation of pain points.
    
    Uses comprehensive research, analysis, and verification
    to validate the business potential of discovered pain points.
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
    
    async def plan(self, input_data: ValidatorInput) -> Dict[str, Any]:
        """Plan the validation process."""
        self.logger.info(f"Planning validation for {len(input_data.pain_points)} pain points")
        
        plan = {
            "phases": [
                "market_research",
                "competitor_analysis",
                "user_demand_assessment",
                "business_potential_evaluation",
                "final_validation"
            ],
            "validation_depth": input_data.validation_depth,
            "include_user_interviews": input_data.include_user_interviews,
            "include_competitor_analysis": input_data.include_competitor_analysis,
            "expected_duration": 600,  # 10 minutes
            "pain_point_count": len(input_data.pain_points)
        }
        
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: ValidatorInput) -> Dict[str, Any]:
        """Analyze pain points for validation decisions."""
        self.logger.info("Analyzing pain points for validation...")
        
        # Analyze market context
        market_analysis = await self._analyze_market_context(
            input_data.pain_points,
            input_data.market_context
        )
        
        # Prepare validation strategy
        validation_strategy = {
            "validation_approach": self._determine_validation_approach(input_data.validation_depth),
            "research_scope": self._determine_research_scope(input_data.pain_points),
            "competitor_analysis_needed": input_data.include_competitor_analysis,
            "user_research_needed": input_data.include_user_interviews,
            "expected_confidence_range": [0.7, 0.95]
        }
        
        return validation_strategy
    
    async def act(self, input_data: ValidatorInput) -> ValidatorOutput:
        """Execute validation and return results."""
        self.logger.info("Executing pain point validation...")
        
        start_time = datetime.now()
        
        validation_results = []
        
        for pain_point in input_data.pain_points:
            result = await self._validate_single_pain_point(
                pain_point,
                input_data.validation_depth,
                input_data.market_context
            )
            validation_results.append(result)
        
        # Separate valid and invalid points
        validated_points = []
        invalid_points = []
        
        for result in validation_results:
            if result.is_valid:
                validated_points.append({
                    **next(p for p in input_data.pain_points if p.get("description") == result.pain_point_id),
                    "validation_result": result.to_dict()
                })
            else:
                invalid_points.append({
                    **next(p for p in input_data.pain_points if p.get("description") == result.pain_point_id),
                    "validation_result": result.to_dict(),
                    "invalid_reason": result.validation_notes
                })
        
        # Generate market insights
        market_insights = await self._generate_market_insights(validated_points)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validated_points, market_insights)
        
        # Generate summary
        summary = self._generate_validation_summary(
            validated_points,
            invalid_points,
            input_data.market_context
        )
        
        # Calculate confidence scores
        confidence_scores = {
            point["description"]: point["validation_result"]["confidence_score"]
            for point in validated_points
        }
        
        return ValidatorOutput(
            validated_pain_points=validated_points,
            invalid_pain_points=invalid_points,
            validation_summary=summary,
            market_insights=market_insights,
            confidence_scores=confidence_scores,
            recommendations=recommendations
        )
    
    async def _validate_single_pain_point(self, pain_point: Dict[str, Any], 
                                        depth: str, market_context: str) -> ValidationResult:
        """Validate a single pain point."""
        description = pain_point.get("description", "")
        market = pain_point.get("market", market_context)
        
        # Market research
        market_data = await self._research_market_for_pain_point(description, market)
        
        # Competitor analysis
        competitor_data = await self._analyze_competitors(description, market)
        
        # User demand assessment
        user_demand = await self._assess_user_demand(description, market)
        
        # Business potential evaluation
        business_potential = self._evaluate_business_potential(
            market_data, competitor_data, user_demand
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_validation_confidence(
            market_data, competitor_data, user_demand, depth
        )
        
        # Determine validity
        is_valid = confidence_score >= 0.7 and business_potential in ["medium", "high"]
        
        return ValidationResult(
            pain_point_id=description,
            is_valid=is_valid,
            confidence_score=confidence_score,
            validation_methods=["market_research", "competitor_analysis", "user_demand_assessment"],
            evidence={
                "market_data": market_data,
                "competitor_data": competitor_data,
                "user_demand": user_demand
            },
            market_size=market_data.get("market_size"),
            competition_level=competitor_data.get("competition_level", "medium"),
            user_demand_score=user_demand.get("demand_score", 0.5),
            business_potential=business_potential,
            validation_notes=self._generate_validation_notes(is_valid, confidence_score, business_potential)
        )
    
    async def _research_market_for_pain_point(self, pain_point: str, market: str) -> Dict[str, Any]:
        """Research market data for a specific pain point."""
        # Mock research - in real implementation, use comprehensive research
        research_query = f"{pain_point} market size {market}"
        
        # Simulate research results
        return {
            "market_size": 50000000.0,  # $50M market
            "growth_rate": 0.15,
            "key_players": ["Company A", "Company B", "Company C"],
            "market_trends": ["automation", "AI integration", "user_experience"],
            "research_sources": ["industry_reports", "market_analysis", "expert_interviews"]
        }
    
    async def _analyze_competitors(self, pain_point: str, market: str) -> Dict[str, Any]:
        """Analyze competitors addressing this pain point."""
        # Mock competitor analysis
        return {
            "competition_level": "medium",
            "direct_competitors": 3,
            "indirect_competitors": 5,
            "market_saturation": 0.4,
            "competitive_advantages": ["better UX", "lower cost", "faster implementation"],
            "gaps_identified": ["integration", "scalability", "customization"]
        }
    
    async def _assess_user_demand(self, pain_point: str, market: str) -> Dict[str, Any]:
        """Assess user demand for solving this pain point."""
        # Mock user demand assessment
        return {
            "demand_score": 0.8,
            "user_interviews": 25,
            "survey_responses": 150,
            "willingness_to_pay": 0.65,
            "urgency_level": "high",
            "pain_intensity": 8.5,
            "search_volume": 5000  # monthly searches
        }
    
    def _evaluate_business_potential(self, market_data: Dict, competitor_data: Dict, 
                                   user_demand: Dict) -> str:
        """Evaluate business potential based on all data."""
        # Simple scoring system
        score = 0
        
        # Market size score
        market_size = market_data.get("market_size", 0)
        if market_size > 10000000:  # > $10M
            score += 3
        elif market_size > 1000000:  # > $1M
            score += 2
        else:
            score += 1
        
        # Competition score (reverse - less competition is better)
        competition = competitor_data.get("competition_level", "medium")
        if competition == "low":
            score += 3
        elif competition == "medium":
            score += 2
        else:
            score += 1
        
        # User demand score
        demand_score = user_demand.get("demand_score", 0)
        score += int(demand_score * 3)
        
        # Determine business potential
        if score >= 7:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"
    
    def _calculate_validation_confidence(self, market_data: Dict, competitor_data: Dict, 
                                       user_demand: Dict, depth: str) -> float:
        """Calculate confidence score for validation."""
        base_score = 0.5
        
        # Data quality score
        data_completeness = len(market_data) + len(competitor_data) + len(user_demand)
        data_score = min(data_completeness * 0.05, 0.3)
        
        # Depth score
        depth_multiplier = {"quick": 0.8, "moderate": 1.0, "deep": 1.2}
        depth_score = depth_multiplier.get(depth, 1.0)
        
        # Consistency score (simplified)
        consistency_score = 0.2
        
        return min((base_score + data_score + consistency_score) * depth_score, 1.0)
    
    def _generate_validation_notes(self, is_valid: bool, confidence: float, potential: str) -> str:
        """Generate validation notes."""
        if is_valid:
            return f"Valid pain point with {confidence:.1%} confidence and {potential} business potential"
        else:
            return f"Invalid pain point - confidence {confidence:.1%}, business potential {potential}"
    
    async def _generate_market_insights(self, validated_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market insights from validated pain points."""
        if not validated_points:
            return {"insights": [], "market_opportunities": 0}
        
        # Analyze patterns
        categories = {}
        severities = {}
        
        for point in validated_points:
            tags = point.get("tags", [])
            severity = point.get("severity", "medium")
            
            for tag in tags:
                categories[tag] = categories.get(tag, 0) + 1
            
            severities[severity] = severities.get(severity, 0) + 1
        
        return {
            "insights": [
                f"Top category: {max(categories.items(), key=lambda x: x[1])[0]}" if categories else "No clear patterns",
                f"Severity distribution: {severities}",
                f"Average impact score: {sum(p.get('impact_score', 0) for p in validated_points) / len(validated_points):.1f}"
            ],
            "market_opportunities": len(validated_points),
            "category_distribution": categories,
            "severity_distribution": severities
        }
    
    def _generate_recommendations(self, validated_points: List[Dict], insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if validated_points:
            recommendations.append("Focus on high-impact, validated pain points")
            recommendations.append("Prioritize pain points with high user demand")
            recommendations.append("Consider market gaps identified in competitor analysis")
            
            if insights["market_opportunities"] > 5:
                recommendations.append("Consider creating a comprehensive solution addressing multiple pain points")
            
            recommendations.append("Develop MVP targeting top 3 pain points")
        else:
            recommendations.append("Refine pain point discovery criteria")
            recommendations.append("Consider alternative market research")
            recommendations.append("Re-evaluate target market focus")
        
        return recommendations
    
    def _determine_validation_approach(self, depth: str) -> str:
        """Determine validation approach based on depth."""
        approaches = {
            "quick": "surface-level validation with limited research",
            "moderate": "balanced validation with standard research",
            "deep": "comprehensive validation with extensive research"
        }
        return approaches.get(depth, "moderate")
    
    def _determine_research_scope(self, pain_points: List[Dict[str, Any]]) -> str:
        """Determine research scope based on pain point count."""
        count = len(pain_points)
        if count <= 3:
            return "focused"
        elif count <= 7:
            return "moderate"
        else:
            return "broad"


# Register the agent
from .base import register_agent
register_agent("validator", ValidatorAgent)
