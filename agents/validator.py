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
from ..llm.utils import LLMAgentMixin, load_prompt_template


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


class ValidatorAgent(BaseAgent, LLMAgentMixin):
    """
    ValidatorAgent for deep validation of pain points.
    
    Uses comprehensive research, analysis, and verification
    to validate the business potential of discovered pain points.
    Uses LLM-driven prompt templates for plan, think, and act phases.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, agent_id)
        LLMAgentMixin.__init__(self)
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
        self.name = "validator_agent"  # Used for prompt template loading
        
        # Set preferred backend to Ollama for all tasks
        self.preferred_backend = "ollama"
        self.task_backend_preferences = {
            "validation": "ollama",
            "research": "ollama",
            "analysis": "ollama"
        }
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Adapter: accept AgentInput, coerce to ValidatorInput, and run.

        Keeps a uniform DAG->Agent interface while using ValidatorInput internally.
        """
        try:
            ctx = agent_input.context or {}
            data = agent_input.data
            # Determine pain_points shape: allow passing list directly or under key
            pain_points = None
            if isinstance(data, dict):
                pain_points = data.get("pain_points") or data.get("filtered_pain_points") or data.get("items")
            if pain_points is None:
                pain_points = data if isinstance(data, list) else []
            
            v_input = ValidatorInput(
                data=agent_input.data,
                metadata=agent_input.metadata,
                context=agent_input.context,
                pain_points=pain_points,
                validation_depth=ctx.get("validation_depth", "moderate"),
                market_context=ctx.get("market_context", ""),
                include_user_interviews=bool(ctx.get("include_user_interviews", True)),
                include_competitor_analysis=bool(ctx.get("include_competitor_analysis", True)),
            )
            
            self._update_status('planning')
            plan = await self.plan(v_input)
            self._update_status('thinking')
            thoughts = await self.think(v_input)
            self._update_status('acting')
            output = await self.act(v_input)  # Expected ValidatorOutput (subclass of AgentOutput)
            self._update_status('completed')
            # Ensure metadata captures plan/thoughts for observability
            if isinstance(output, AgentOutput):
                output.metadata = {**(output.metadata or {}), 'plan': plan, 'thoughts': thoughts, 'agent_name': self.name, 'agent_id': self.agent_id}
            return output
        except Exception as e:
            self._update_status('failed')
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=0.0,
                success=False,
                error=str(e),
            )
    
    async def plan(self, input_data: ValidatorInput) -> Dict[str, Any]:
        """Plan the validation process using LLM prompt."""
        self.logger.info(f"Planning validation for {len(input_data.pain_points)} pain points")
        start_time = datetime.now()
        
        try:
            # Prepare substitutions for the prompt template
            substitutions = {
                "pain_points": json.dumps(input_data.pain_points, indent=2),
                "pain_point_count": len(input_data.pain_points),
                "validation_depth": input_data.validation_depth,
                "market_context": input_data.market_context,
                "include_user_interviews": str(input_data.include_user_interviews).lower(),
                "include_competitor_analysis": str(input_data.include_competitor_analysis).lower()
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("plan.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate plan using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="validation"
            )
            
            # Extract JSON from LLM response
            plan = self._extract_json(llm_response)
            
            # Set the expected_duration from the plan or default to 600 seconds (10 minutes)
            if "expected_duration" not in plan:
                plan["expected_duration"] = 600
                
            # Add execution time
            plan["execution_time"] = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Plan generation completed in {plan['execution_time']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error generating validation plan: {str(e)}")
            # Create fallback plan
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
                "pain_point_count": len(input_data.pain_points),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
        # Store plan in state
        self.state.plan = plan
        return plan
        
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        try:
            # First attempt: Try to parse the entire response as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Second attempt: Look for JSON code block
                import re
                json_match = re.search(r'```(?:json)?\n([\s\S]*?)\n```', text)
                if json_match:
                    json_str = json_match.group(1).strip()
                    # Handle trailing commas which are not valid in JSON
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*\]', ']', json_str)
                    return json.loads(json_str)
            except Exception:
                pass
                
            try:
                # Third attempt: Try to find any JSON-like structure
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx+1]
                    # Handle trailing commas
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*\]', ']', json_str)
                    return json.loads(json_str)
            except Exception:
                pass
                
            # If all parsing attempts fail, log warning and return empty dict
            self.logger.warning(f"Could not parse LLM response as JSON: {text[:50]}...")
            return {}
    
    async def think(self, input_data: ValidatorInput) -> Dict[str, Any]:
        """Analyze pain points for validation decisions using LLM prompt."""
        self.logger.info("Analyzing pain points for validation...")
        start_time = datetime.now()
        
        # Get plan or run if not available
        plan = getattr(self.state, 'plan', None) or await self.plan(input_data)
        
        try:
            # Prepare substitutions for the prompt template
            substitutions = {
                "pain_points": json.dumps(input_data.pain_points, indent=2),
                "pain_point_count": len(input_data.pain_points),
                "market_context": input_data.market_context,
                "validation_plan": json.dumps(plan, indent=2),
                "validation_depth": input_data.validation_depth,
                "include_user_interviews": str(input_data.include_user_interviews).lower(),
                "include_competitor_analysis": str(input_data.include_competitor_analysis).lower()
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("think.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate validation strategy using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="analysis"
            )
            
            # Extract JSON from LLM response
            validation_strategy = self._extract_json(llm_response)
            
            # Add execution time
            validation_strategy["execution_time"] = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Validation strategy analysis completed in {validation_strategy['execution_time']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error generating validation strategy: {str(e)}")
            # Create fallback strategy
            validation_strategy = {
                "validation_approach": self._determine_validation_approach(input_data.validation_depth),
                "research_scope": self._determine_research_scope(input_data.pain_points),
                "competitor_analysis_needed": input_data.include_competitor_analysis,
                "user_research_needed": input_data.include_user_interviews,
                "expected_confidence_range": [0.7, 0.95],
                "market_analysis": {
                    "key_segments": ["small businesses", "mid-market companies"],
                    "market_size_estimate": "Unknown",
                    "growth_trend": "Unknown"
                },
                "validation_criteria": {
                    "user_demand_threshold": 7.5,
                    "market_size_threshold": 1000000,
                    "competition_level_acceptable": "medium"
                },
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
        
        # Store validation strategy in state
        self.state.validation_strategy = validation_strategy
        return validation_strategy
    
    async def act(self, input_data: ValidatorInput) -> ValidatorOutput:
        """Execute validation and return results using LLM prompt."""
        self.logger.info(f"Validating {len(input_data.pain_points)} pain points")
        start_time = datetime.now()
        
        # Get plan and validation strategy or run if not available
        plan = getattr(self.state, 'plan', None) or await self.plan(input_data)
        validation_strategy = getattr(self.state, 'validation_strategy', None) or await self.think(input_data)
        
        try:
            # Prepare substitutions for the prompt template
            substitutions = {
                "pain_points": json.dumps(input_data.pain_points, indent=2),
                "pain_point_count": len(input_data.pain_points),
                "market_context": input_data.market_context,
                "validation_plan": json.dumps(plan, indent=2),
                "validation_strategy": json.dumps(validation_strategy, indent=2),
                "validation_depth": input_data.validation_depth,
                "include_user_interviews": str(input_data.include_user_interviews).lower(),
                "include_competitor_analysis": str(input_data.include_competitor_analysis).lower()
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("act.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate validation results using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="validation"
            )
            
            # Extract JSON from LLM response
            validation_results = self._extract_json(llm_response)
            
            # Process validation results
            validated_pain_points = validation_results.get("validated_pain_points", [])
            invalid_pain_points = validation_results.get("invalid_pain_points", [])
            validation_summary = validation_results.get("validation_summary", 
                                                        f"Validated {len(validated_pain_points)} of {len(input_data.pain_points)} pain points")
            market_insights = validation_results.get("market_insights", {"insights": [], "market_opportunities": 0})
            confidence_scores = validation_results.get("confidence_scores", {})
            recommendations = validation_results.get("recommendations", [])
            
            # Create and return output
            output = ValidatorOutput(
                validated_pain_points=validated_pain_points,
                invalid_pain_points=invalid_pain_points,
                validation_summary=validation_summary,
                market_insights=market_insights,
                confidence_scores=confidence_scores,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.logger.info(f"Validation completed in {output.execution_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error executing validation: {str(e)}")
            # Create fallback validation using internal methods
            validation_results = []
            for pain_point in input_data.pain_points:
                result = await self._validate_single_pain_point(
                    pain_point,
                    input_data.validation_depth,
                    input_data.market_context
                )
                validation_results.append(result)
            
            # Separate valid and invalid pain points
            validated_points = [p.to_dict() for p in validation_results if p.is_valid]
            invalid_points = [p.to_dict() for p in validation_results if not p.is_valid]
            
            # Generate confidence scores
            confidence_scores = {r.pain_point_id: r.confidence_score for r in validation_results}
            
            # Generate market insights
            market_insights = await self._generate_market_insights(validated_points)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validated_points, market_insights)
            
            # Create validation summary
            validation_summary = f"Validated {len(validated_points)} of {len(input_data.pain_points)} pain points"
            
            # Create and return output
            output = ValidatorOutput(
                validated_pain_points=validated_points,
                invalid_pain_points=invalid_points,
                validation_summary=validation_summary,
                market_insights=market_insights,
                confidence_scores=confidence_scores,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.logger.info(f"Validation completed with fallback in {output.execution_time:.2f} seconds")
        
        return output
    
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
