"""
ScreenerAgent - Initial Pain Point Filtering Agent

This agent specializes in filtering and categorizing discovered pain points
based on severity, market relevance, and business potential.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .analysis_agent import AnalysisAgent
from ..config import get_config


@dataclass
class ScreeningCriteria:
    """Criteria for screening pain points."""
    min_severity: str = "medium"  # low, medium, high, critical
    min_impact_score: float = 5.0
    min_frequency: int = 3
    max_age_days: int = 365
    required_tags: List[str] = None
    excluded_tags: List[str] = None
    
    def __post_init__(self):
        if self.required_tags is None:
            self.required_tags = []
        if self.excluded_tags is None:
            self.excluded_tags = []


@dataclass
class ScreenerInput(AgentInput):
    """Input for ScreenerAgent."""
    pain_points: List[Dict[str, Any]]  # From ScoutAgent
    criteria: ScreeningCriteria = None
    market_focus: str = ""
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = ScreeningCriteria()


@dataclass
class ScreenerOutput(AgentOutput):
    """Output from ScreenerAgent."""
    filtered_pain_points: List[Dict[str, Any]]
    rejected_pain_points: List[Dict[str, Any]]
    categorization: Dict[str, List[str]]
    summary: str
    filtering_stats: Dict[str, int]
    confidence_score: float


class ScreenerAgent(BaseAgent):
    """
    ScreenerAgent for filtering and categorizing pain points.
    
    Uses criteria-based filtering, categorization, and initial
    business potential assessment to prioritize pain points.
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
    
    async def plan(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Plan the screening process."""
        self.logger.info(f"Planning screening for {len(input_data.pain_points)} pain points")
        
        plan = {
            "phases": [
                "initial_filtering",
                "severity_assessment",
                "categorization",
                "business_potential_check",
                "final_ranking"
            ],
            "criteria": asdict(input_data.criteria),
            "expected_duration": 120,  # 2 minutes
            "input_count": len(input_data.pain_points)
        }
        
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Analyze pain points for screening decisions."""
        self.logger.info("Analyzing pain points for screening...")
        
        # Analyze patterns
        analysis_input = {
            "data": input_data.pain_points,
            "analysis_type": "categorization",
            "dimensions": ["severity", "impact", "frequency", "business_potential"]
        }
        
        analysis_results = await self.analysis_agent.execute(analysis_input)
        
        # Calculate screening metrics
        screening_analysis = {
            "total_points": len(input_data.pain_points),
            "severity_distribution": self._get_severity_distribution(input_data.pain_points),
            "impact_score_range": self._get_impact_range(input_data.pain_points),
            "frequency_distribution": self._get_frequency_distribution(input_data.pain_points),
            "categorization_potential": len(analysis_results.get("categories", [])),
            "filtering_strategy": self._determine_filtering_strategy(input_data.criteria)
        }
        
        return screening_analysis
    
    async def act(self, input_data: ScreenerInput) -> ScreenerOutput:
        """Execute screening and return filtered results."""
        self.logger.info("Executing pain point screening...")
        
        # Apply filtering criteria
        filtered_points, rejected_points = self._apply_filtering(
            input_data.pain_points,
            input_data.criteria
        )
        
        # Categorize filtered points
        categorization = self._categorize_pain_points(filtered_points)
        
        # Generate summary
        summary = self._generate_screening_summary(
            filtered_points,
            rejected_points,
            input_data.market_focus
        )
        
        # Calculate filtering stats
        filtering_stats = {
            "total_input": len(input_data.pain_points),
            "accepted": len(filtered_points),
            "rejected": len(rejected_points),
            "acceptance_rate": len(filtered_points) / len(input_data.pain_points) if input_data.pain_points else 0
        }
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            filtered_points,
            rejected_points,
            input_data.criteria
        )
        
        return ScreenerOutput(
            filtered_pain_points=filtered_points,
            rejected_pain_points=rejected_points,
            categorization=categorization,
            summary=summary,
            filtering_stats=filtering_stats,
            confidence_score=confidence_score
        )
    
    def _apply_filtering(self, pain_points: List[Dict[str, Any]], criteria: ScreeningCriteria) -> tuple:
        """Apply filtering criteria to pain points."""
        filtered = []
        rejected = []
        
        for point in pain_points:
            is_valid, reason = self._evaluate_point(point, criteria)
            
            if is_valid:
                filtered.append(point)
            else:
                point["rejection_reason"] = reason
                rejected.append(point)
        
        return filtered, rejected
    
    def _evaluate_point(self, point: Dict[str, Any], criteria: ScreeningCriteria) -> tuple:
        """Evaluate a single pain point against criteria."""
        reasons = []
        
        # Check severity
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        point_severity = severity_map.get(point.get("severity", "low"), 1)
        criteria_severity = severity_map.get(criteria.min_severity, 2)
        
        if point_severity < criteria_severity:
            reasons.append(f"severity too low ({point.get('severity')})")
        
        # Check impact score
        impact_score = point.get("impact_score", 0)
        if impact_score < criteria.min_impact_score:
            reasons.append(f"impact score too low ({impact_score})")
        
        # Check frequency
        frequency = point.get("frequency", 0)
        if frequency < criteria.min_frequency:
            reasons.append(f"frequency too low ({frequency})")
        
        # Check required tags
        point_tags = point.get("tags", [])
        for required_tag in criteria.required_tags:
            if required_tag not in point_tags:
                reasons.append(f"missing required tag: {required_tag}")
        
        # Check excluded tags
        for excluded_tag in criteria.excluded_tags:
            if excluded_tag in point_tags:
                reasons.append(f"contains excluded tag: {excluded_tag}")
        
        # Check age (mock implementation)
        discovered_at = point.get("discovered_at", datetime.now().isoformat())
        # In real implementation, check against max_age_days
        
        if reasons:
            return False, "; ".join(reasons)
        return True, ""
    
    def _categorize_pain_points(self, pain_points: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize pain points by themes."""
        categories = {
            "onboarding": [],
            "integration": [],
            "pricing": [],
            "performance": [],
            "usability": [],
            "features": [],
            "support": [],
            "security": []
        }
        
        for point in pain_points:
            tags = point.get("tags", [])
            description = point.get("description", "").lower()
            
            # Categorize based on tags and description
            for category in categories:
                if category in tags or category in description:
                    categories[category].append(point.get("description", ""))
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _generate_screening_summary(self, accepted: List[Dict], rejected: List[Dict], market: str) -> str:
        """Generate a summary of the screening process."""
        if not accepted:
            return f"No pain points met screening criteria for {market}"
        
        high_severity = [p for p in accepted if p.get("severity") == "high"]
        medium_severity = [p for p in accepted if p.get("severity") == "medium"]
        
        summary = f"""
Pain Point Screening Summary for {market}:
- Total screened: {len(accepted) + len(rejected)}
- Accepted: {len(accepted)} ({len(accepted)/(len(accepted)+len(rejected))*100:.1f}%)
- Rejected: {len(rejected)}
- High severity: {len(high_severity)}
- Medium severity: {len(medium_severity)}
- Key categories: {', '.join(self._categorize_pain_points(accepted).keys())}
        """.strip()
        
        return summary
    
    def _calculate_confidence_score(self, accepted: List[Dict], rejected: List[Dict], criteria: ScreeningCriteria) -> float:
        """Calculate confidence score based on screening quality."""
        total_points = len(accepted) + len(rejected)
        if total_points == 0:
            return 0.0
        
        # Base score from acceptance rate
        acceptance_rate = len(accepted) / total_points
        base_score = min(acceptance_rate * 0.5, 0.5)
        
        # Score from criteria consistency
        criteria_score = 0.3
        
        # Score from categorization
        categories = self._categorize_pain_points(accepted)
        categorization_score = min(len(categories) * 0.05, 0.2)
        
        return min(base_score + criteria_score + categorization_score, 1.0)
    
    def _get_severity_distribution(self, pain_points: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of pain point severities."""
        distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for point in pain_points:
            severity = point.get("severity", "low")
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution
    
    def _get_impact_range(self, pain_points: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get min and max impact scores."""
        if not pain_points:
            return {"min": 0.0, "max": 0.0}
        
        scores = [p.get("impact_score", 0) for p in pain_points]
        return {"min": min(scores), "max": max(scores)}
    
    def _get_frequency_distribution(self, pain_points: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of pain point frequencies."""
        distribution = {"low": 0, "medium": 0, "high": 0}
        for point in pain_points:
            freq = point.get("frequency", 0)
            if freq < 5:
                distribution["low"] += 1
            elif freq < 15:
                distribution["medium"] += 1
            else:
                distribution["high"] += 1
        return distribution
    
    def _determine_filtering_strategy(self, criteria: ScreeningCriteria) -> str:
        """Determine the filtering strategy based on criteria."""
        if criteria.min_severity == "high" and criteria.min_impact_score >= 7.0:
            return "strict"
        elif criteria.min_severity == "medium" and criteria.min_impact_score >= 5.0:
            return "moderate"
        else:
            return "lenient"


# Register the agent
from .base import register_agent
register_agent("screener", ScreenerAgent)
