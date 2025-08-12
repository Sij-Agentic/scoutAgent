"""
ScreenerAgent - Initial Pain Point Filtering Agent

This agent specializes in filtering and categorizing discovered pain points
based on severity, market relevance, and business potential.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .analysis_agent import AnalysisAgent
from config import get_config
from llm.utils import LLMAgentMixin, load_prompt_template
from llm.base import LLMBackendType


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
    # AgentInput already has data, metadata, and context fields
    # Custom fields for ScreenerInput
    market_focus: str = ""
    criteria: ScreeningCriteria = None
    
    def __post_init__(self):
        super().__post_init__() if hasattr(AgentInput, "__post_init__") else None
        if self.criteria is None:
            self.criteria = ScreeningCriteria()
        # Extract pain points from data field
        self.pain_points = self.data


@dataclass
class ScreenerOutput:
    """Output from ScreenerAgent."""
    # Using composition instead of inheritance to avoid dataclass field order issues
    result: Any
    metadata: Dict[str, Any]
    logs: List[str]
    execution_time: float
    filtered_pain_points: List[Dict[str, Any]]
    rejected_pain_points: List[Dict[str, Any]]
    categorization: Dict[str, List[str]]
    summary: str
    filtering_stats: Dict[str, int]
    confidence_score: float
    success: bool = True
    error: Optional[str] = None
    
    def to_agent_output(self) -> AgentOutput:
        """Convert to standard AgentOutput for compatibility."""
        return AgentOutput(
            result=self.result,
            metadata=self.metadata,
            logs=self.logs,
            execution_time=self.execution_time,
            success=self.success,
            error=self.error
        )


class ScreenerAgent(BaseAgent, LLMAgentMixin):
    """
    ScreenerAgent for filtering and categorizing pain points.
    
    Uses criteria-based filtering, categorization, and initial
    business potential assessment to prioritize pain points.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="screener", agent_id=agent_id)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
        self.name = "screener"
        self.start_time = time.time()
        
        # Set backend preferences for this agent
        self.preferred_backend = 'ollama'
        self.task_backend_preferences = {
            "default": LLMBackendType.OLLAMA
        }
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Adapter: accept AgentInput, coerce to ScreenerInput, and run.

        This keeps ScreenerAgent compatible with DAG executor passing a
        standard AgentInput while allowing ScreenerAgent's methods to use
        ScreenerInput-specific fields.
        """
        self.start_time = time.time()
        try:
            # Build criteria from context or defaults
            ctx = agent_input.context or {}
            crit = ctx.get("criteria") or {}
            criteria = ScreeningCriteria(
                min_severity=crit.get("min_severity", "medium"),
                min_impact_score=crit.get("min_impact_score", 5.0),
                min_frequency=crit.get("min_frequency", 3),
                max_age_days=crit.get("max_age_days", 365),
                required_tags=crit.get("required_tags"),
                excluded_tags=crit.get("excluded_tags"),
            )
            market_focus = ctx.get("market_focus", "")
            s_input = ScreenerInput(
                data=agent_input.data,
                metadata=agent_input.metadata,
                context=agent_input.context,
                market_focus=market_focus,
                criteria=criteria,
            )
            # plan-think-act
            self._update_status('planning')
            plan = await self.plan(s_input)
            self._update_status('thinking')
            thoughts = await self.think(s_input)
            self._update_status('acting')
            result = await self.act(s_input)
            exec_time = time.time() - self.start_time
            # Build ScreenerOutput enriched object, then flatten to AgentOutput
            filtered = result.get("filtered", []) if isinstance(result, dict) else []
            rejected = result.get("rejected", []) if isinstance(result, dict) else []
            categorization = result.get("categorization", {}) if isinstance(result, dict) else {}
            summary = result.get("summary", "") if isinstance(result, dict) else ""
            stats = result.get("stats", {}) if isinstance(result, dict) else {}
            confidence = result.get("confidence", 0.0) if isinstance(result, dict) else 0.0
            s_output = ScreenerOutput(
                result=result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'plan': plan,
                    'thoughts': thoughts,
                },
                logs=self.execution_logs,
                execution_time=exec_time,
                filtered_pain_points=filtered,
                rejected_pain_points=rejected,
                categorization=categorization,
                summary=summary,
                filtering_stats=stats,
                confidence_score=confidence,
                success=True,
            )
            self._update_status('completed')
            return s_output.to_agent_output()
        except Exception as e:
            exec_time = time.time() - self.start_time
            self._update_status('failed')
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=exec_time,
                success=False,
                error=str(e),
            )
    
    def extract_json_from_markdown(self, text):
        """Extract JSON from markdown code blocks or plain text with robust error handling."""
        import re
        import json

        # Default fallback result if parsing fails
        default_result = {
            "status": "fallback",
            "message": "Failed to parse JSON from response"
        }

        if not text or not text.strip():
            self.logger.warning("Empty text provided to extract_json_from_markdown")
            return json.dumps(default_result)

        # First try to find JSON in markdown code blocks
        code_block_patterns = [
            r'```(?:json)?\s*(.+?)\s*```',  # Standard code block
            r'`(.+?)`',                       # Inline code
            r'\{\s*"(.+?)"\s*\}',         # Just look for JSON-like content
        ]
        
        json_str = text
        for pattern in code_block_patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                json_str = matches.group(1)
                break
            
        # Clean the JSON string
        # Remove trailing commas before closing brackets or braces
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
        
        # Attempt to fix common JSON formatting issues
        json_str = json_str.replace("'", '"')   # Replace single quotes with double quotes
        json_str = re.sub(r'([\{\[,:]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)  # Add quotes to unquoted keys
        
        # Try to parse the JSON to validate it
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode error: {str(e)}. Attempting further cleanup...")
            
            # Additional cleanup attempts
            # Remove all non-ASCII characters
            json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)
            
            # Try to extract valid JSON subset if there's extra text
            json_subset_pattern = r'\{.*\}'  # Find the first full JSON object
            subset_match = re.search(json_subset_pattern, json_str, re.DOTALL)
            if subset_match:
                json_str = subset_match.group(0)
            
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # Last resort: return the default fallback
                self.logger.error(f"Failed to extract valid JSON from LLM response")
                return json.dumps(default_result)
    
    async def plan(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Plan the screening process using LLM."""
        self.logger.info(f"Planning screening for {len(input_data.pain_points)} pain points")
        
        try:
            # Load prompt template
            prompt = load_prompt_template(
                template_name="plan.prompt",
                agent_name=self.name,
                substitutions={
                    "pain_points_count": len(input_data.pain_points),
                    "criteria_min_severity": input_data.criteria.min_severity,
                    "criteria_min_impact_score": input_data.criteria.min_impact_score,
                    "criteria_min_frequency": input_data.criteria.min_frequency,
                    "criteria_max_age_days": input_data.criteria.max_age_days,
                    "criteria_required_tags": input_data.criteria.required_tags,
                    "criteria_excluded_tags": input_data.criteria.excluded_tags,
                    "market_focus": input_data.market_focus
                }
            )
            
            # Generate plan using LLM with Ollama backend
            response = await self.llm_generate(prompt, backend_type="ollama")
            
            # Extract JSON from response
            plan = json.loads(self.extract_json_from_markdown(response))
            
            self.logger.info(f"Generated plan with {len(plan.get('phases', []))} phases")
            self.state.plan = plan
            return plan
            
        except Exception as e:
            self.logger.error(f"Error in ScreenerAgent plan: {str(e)}")
            # Fallback plan
            fallback_plan = {
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
            
            self.state.plan = fallback_plan
            return fallback_plan
    
    async def think(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Analyze pain points for screening decisions using LLM."""
        self.logger.info("Analyzing pain points for screening...")
        
        try:
            # Ensure we have a plan
            plan = self.state.plan
            if not plan:
                plan = await self.plan(input_data)
            
            # Load prompt template
            prompt = load_prompt_template(
                template_name="think.prompt",
                agent_name=self.name,
                substitutions={
                    "pain_points": json.dumps(input_data.pain_points, indent=2),
                    "criteria": json.dumps(asdict(input_data.criteria), indent=2),
                    "plan": json.dumps(plan, indent=2),
                    "market_focus": input_data.market_focus
                }
            )
            
            # Generate analysis using LLM with Ollama backend
            response = await self.llm_generate(prompt, backend_type="ollama")
            
            # Extract JSON from response
            analysis = json.loads(self.extract_json_from_markdown(response))
            
            self.logger.info(f"Generated screening analysis with strategy: {analysis.get('filtering_strategy', 'unknown')}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in ScreenerAgent think: {str(e)}")
            # Fallback analysis using helper methods
            analysis_input = {
                "data": input_data.pain_points,
                "analysis_type": "categorization",
                "dimensions": ["severity", "impact", "frequency", "business_potential"]
            }
            
            try:
                # Ensure AnalysisAgent receives an AgentInput, not a raw dict
                from .base import AgentInput as _AgentInput
                analysis_results_output = await self.analysis_agent.execute(
                    _AgentInput(
                        data=analysis_input,
                        metadata={"source": "screener_fallback", "stage": "analysis"}
                    )
                )
                # BaseAgent.execute returns AgentOutput; extract result safely
                analysis_results = analysis_results_output.result if hasattr(analysis_results_output, "result") else {}
            except Exception:
                analysis_results = {"categories": []}
                
            # Calculate screening metrics as fallback
            fallback_analysis = {
                "total_points": len(input_data.pain_points),
                "severity_distribution": self._get_severity_distribution(input_data.pain_points),
                "impact_score_range": self._get_impact_range(input_data.pain_points),
                "frequency_distribution": self._get_frequency_distribution(input_data.pain_points),
                "categorization_potential": len(analysis_results.get("categories", [])),
                "filtering_strategy": self._determine_filtering_strategy(input_data.criteria)
            }
            
            return fallback_analysis
    
    async def act(self, input_data: ScreenerInput) -> ScreenerOutput:
        """Execute screening and return filtered results using LLM."""
        self.logger.info("Executing pain point screening...")
        
        try:
            # Get the analysis results from think phase
            analysis = getattr(self.state, "think_result", None)
            if not analysis:
                analysis = await self.think(input_data)
                
            # Load prompt template
            prompt = load_prompt_template(
                template_name="act.prompt",
                agent_name=self.name,
                substitutions={
                    "pain_points": json.dumps(input_data.pain_points, indent=2),
                    "criteria": json.dumps(asdict(input_data.criteria), indent=2),
                    "analysis": json.dumps(analysis, indent=2),
                    "market_focus": input_data.market_focus
                }
            )
            
            # Generate screening results using LLM with Ollama backend
            response = await self.llm_generate(prompt, backend_type="ollama")
            
            # Extract JSON from response
            results = json.loads(self.extract_json_from_markdown(response))
            
            self.logger.info(f"Generated screening results with {len(results.get('filtered_pain_points', []))} accepted pain points")
            
            # Create ScreenerOutput with the results
            return ScreenerOutput(
                result=results,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "plan": self.state.plan,
                    "thoughts": analysis
                },
                logs=self.execution_logs,
                execution_time=time.time() - self.start_time,
                filtered_pain_points=results.get("filtered_pain_points", []),
                rejected_pain_points=results.get("rejected_pain_points", []),
                categorization=results.get("categorization", {}),
                summary=results.get("summary", ""),
                filtering_stats=results.get("filtering_stats", {"total": 0, "accepted": 0, "rejected": 0}),
                confidence_score=results.get("confidence_score", 0.0),
                success=True,
                error=None
            )
        except Exception as e:
            self.logger.error(f"Error in ScreenerAgent act: {str(e)}")
            # First attempt simple JSON handling fallback
            try:
                error_message = f"Error in screening execution: {str(e)}"
                return ScreenerOutput(
                    result={},
                    metadata={
                        "agent_id": self.agent_id,
                        "agent_name": self.name,
                        "error": error_message
                    },
                    logs=self.execution_logs,
                    execution_time=time.time() - self.start_time,
                    filtered_pain_points=[],
                    rejected_pain_points=[],
                    categorization={},
                    summary="Error in screening execution",
                    filtering_stats={"total": 0, "accepted": 0, "rejected": 0},
                    confidence_score=0.0,
                    success=False,
                    error=error_message
                )
            except:
                self.logger.error(f"Error in ScreenerAgent fallback handling, using helper methods")
                # Final fallback using helper methods
                # Apply filtering
                accepted_points = self._apply_filtering(input_data.pain_points, input_data.criteria)
                rejected_points = [p for p in input_data.pain_points if p not in accepted_points]
                
                # Categorize filtered points
                categories = self._categorize_pain_points(accepted_points)
                
                # Generate summary
                summary = self._generate_screening_summary(
                    accepted_points, 
                    rejected_points,
                input_data.market_focus
            )
            
            # Calculate filtering stats
            filtering_stats = {
                "total": len(input_data.pain_points),
                "accepted": len(accepted_points),
                "rejected": len(rejected_points),
                "by_severity": self._get_severity_distribution(accepted_points)
            }
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                accepted_points,
                rejected_points,
                input_data.criteria
            )
            
            return ScreenerOutput(
                filtered_pain_points=accepted_points,
                rejected_pain_points=rejected_points,
                categorization=categories,
                summary=summary,
                filtering_stats=filtering_stats,
                confidence_score=confidence
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
            # Handle both dict and list format pain points
            if isinstance(point, dict):
                tags = point.get("tags", [])
                description = point.get("description", "").lower()
            else:
                # Handle case where point might be a list or other type
                tags = []
                description = str(point).lower()
            
            # Categorize based on tags and description
            for category in categories:
                if category in tags or category in description:
                    # Handle different point types when appending to categories
                    if isinstance(point, dict):
                        categories[category].append(point.get("description", ""))
                    else:
                        # If point is not a dict, append the string representation
                        categories[category].append(str(point))
        
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
register_agent(ScreenerAgent)
