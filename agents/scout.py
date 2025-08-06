"""
ScoutAgent - Pain Point Discovery Agent

This agent specializes in discovering pain points in target markets using
web research, social media analysis, and user feedback collection.
"""

import asyncio
import json
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .research_agent import ResearchAgent
from ..config import get_config
from ..llm.utils import LLMAgentMixin, load_prompt_template
from ..llm.base import LLMBackendType


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


class ScoutAgent(BaseAgent, LLMAgentMixin):
    """
    ScoutAgent for discovering pain points in target markets.
    
    Uses web research, social media analysis, and user feedback
    to identify and categorize pain points with evidence.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, agent_id)
        LLMAgentMixin.__init__(self)
        self.name = "scout_agent"  # Used for prompt template loading
        self.research_agent = ResearchAgent()
        self.config = get_config()
        
        # Set default backend preferences
        self.task_backend_preferences = {
            "default": [LLMBackendType.OLLAMA, LLMBackendType.OPENAI, LLMBackendType.CLAUDE],
            "plan": [LLMBackendType.OLLAMA, LLMBackendType.OPENAI],
            "think": [LLMBackendType.OLLAMA, LLMBackendType.OPENAI, LLMBackendType.CLAUDE],
            "act": [LLMBackendType.OLLAMA, LLMBackendType.OPENAI, LLMBackendType.CLAUDE]
        }
    
    async def plan(self, input_data: ScoutInput) -> Dict[str, Any]:
        """Plan the pain point discovery process."""
        self.logger.info(f"Planning pain point discovery for market: {input_data.target_market}")
        
        try:
            # Load the planning prompt template
            prompt_template = load_prompt_template(agent_name=self.name, prompt_name="plan")
            
            # Prepare prompt substitutions
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "max_pain_points": input_data.max_pain_points,
                "sources": json.dumps(input_data.sources),
                "keywords": json.dumps(input_data.keywords)
            }
            
            # Generate plan using LLM
            llm_response = await self.llm_generate(
                prompt_template=prompt_template,
                substitutions=substitutions,
                task="plan"
            )
            
            if llm_response and llm_response.success:
                # Parse the JSON response
                plan = self._extract_json(llm_response.content)
                self.logger.info(f"Generated pain point discovery plan with {len(plan.get('phases', []))} phases")
            else:
                # Fallback if LLM fails
                self.logger.warning("LLM plan generation failed, using fallback plan")
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
        except Exception as e:
            self.logger.error(f"Error in plan phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback plan
            plan = {
                "phases": [
                    "market_research",
                    "pain_point_extraction", 
                    "validation"
                ],
                "sources": input_data.sources,
                "keywords": input_data.keywords,
                "expected_duration": 300,
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
        
        try:
            # Use research agent to gather market data
            research_results = await self.research_agent.execute(research_input)
            
            # Load the thinking prompt template
            prompt_template = load_prompt_template(agent_name=self.name, prompt_name="think")
            
            # Prepare prompt substitutions
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "plan": json.dumps(self.state.plan),
                "research_results": json.dumps(research_results)
            }
            
            # Generate analysis using LLM
            llm_response = await self.llm_generate(
                prompt_template=prompt_template,
                substitutions=substitutions,
                task="think"
            )
            
            if llm_response and llm_response.success:
                # Parse the JSON response
                analysis = self._extract_json(llm_response.content)
                self.logger.info(f"Generated analysis found {analysis.get('pain_points_found', 0)} pain points")
            else:
                # Fallback if LLM fails
                self.logger.warning("LLM analysis generation failed, using fallback analysis")
                analysis = {
                    "total_sources_analyzed": len(research_results.get("sources", [])),
                    "pain_points_found": len(self._extract_pain_points(research_results, input_data.target_market)),
                    "confidence_factors": [
                        "multiple_sources",
                        "consistent_patterns",
                        "high_severity_indicators"
                    ],
                    "next_steps": ["validate", "categorize", "prioritize"]
                }
                
            # Save research results for the act phase
            self.state.research_results = research_results
            self.state.analysis = analysis
            
        except Exception as e:
            self.logger.error(f"Error in think phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback analysis
            analysis = {
                "total_sources_analyzed": 0,
                "pain_points_found": 0,
                "confidence_factors": ["limited_data"],
                "next_steps": ["gather_more_data"]
            }
        
        return analysis
    
    async def act(self, input_data: ScoutInput) -> ScoutOutput:
        """Execute pain point discovery and return results."""
        self.logger.info("Executing pain point discovery...")
        
        start_time = datetime.now()
        
        try:
            # Load the action prompt template
            prompt_template = load_prompt_template(agent_name=self.name, prompt_name="act")
            
            # Get research results and analysis from state
            research_results = getattr(self.state, 'research_results', {})
            analysis = getattr(self.state, 'analysis', {})
            
            # If we don't have research results, use mock data
            if not research_results:
                self.logger.warning("No research results in state, using mock data")
                research_results = {
                    "sources": ["reddit", "twitter", "product_reviews", "forums"],
                    "results": [
                        {"source": "reddit", "sentiment": "negative", "content": "This tool is so hard to use..."},
                        {"source": "twitter", "sentiment": "negative", "content": "Can't believe how complicated the setup is"},
                        {"source": "product_reviews", "sentiment": "mixed", "content": "Great features but steep learning curve"}
                    ]
                }
            
            # Prepare prompt substitutions
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "max_pain_points": input_data.max_pain_points,
                "plan": json.dumps(self.state.plan),
                "analysis": json.dumps(analysis),
                "research_results": json.dumps(research_results)
            }
            
            # Generate action result using LLM
            llm_response = await self.llm_generate(
                prompt_template=prompt_template,
                substitutions=substitutions,
                task="act"
            )
            
            if llm_response and llm_response.success:
                # Parse the JSON response
                act_result = self._extract_json(llm_response.content)
                
                # Create pain point objects from the response
                pain_points = []
                for pp_data in act_result.get("pain_points", []):
                    pain_point = PainPoint(
                        description=pp_data.get("description", ""),
                        severity=pp_data.get("severity", "medium"),
                        market=input_data.target_market,
                        source=pp_data.get("source", "unknown"),
                        evidence=pp_data.get("evidence", []),
                        frequency=pp_data.get("frequency", 0),
                        impact_score=pp_data.get("impact_score", 5.0),
                        tags=pp_data.get("tags", []),
                        discovered_at=datetime.now().isoformat()
                    )
                    pain_points.append(pain_point)
                
                self.logger.info(f"Generated {len(pain_points)} pain points using LLM")
                
                # Prepare output using the LLM-generated result
                output = ScoutOutput(
                    pain_points=pain_points[:input_data.max_pain_points],
                    total_discovered=act_result.get("total_discovered", len(pain_points)),
                    market_summary=act_result.get("market_summary", ""),
                    confidence_score=act_result.get("confidence_score", 0.0),
                    sources_used=act_result.get("sources_used", []),
                    research_duration=(datetime.now() - start_time).total_seconds()
                )
            else:
                # Fallback if LLM fails
                self.logger.warning("LLM action generation failed, using fallback method")
                
                # Extract pain points using fallback method
                pain_points = self._extract_pain_points(research_results, input_data.target_market)
                
                # Validate pain points
                validated_pain_points = self._validate_pain_points(pain_points)
                
                # Generate market summary
                market_summary = self._generate_market_summary(validated_pain_points, input_data.target_market)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(validated_pain_points, research_results)
                
                # Prepare output using fallback method
                output = ScoutOutput(
                    pain_points=validated_pain_points,
                    total_discovered=len(pain_points),
                    market_summary=market_summary,
                    confidence_score=confidence_score,
                    sources_used=research_results.get("sources", []),
                    research_duration=(datetime.now() - start_time).total_seconds()
                )
                
        except Exception as e:
            self.logger.error(f"Error in act phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback output with minimal data
            output = ScoutOutput(
                pain_points=[],
                total_discovered=0,
                market_summary=f"Error processing pain points for {input_data.target_market}",
                confidence_score=0.0,
                sources_used=[],
                research_duration=(datetime.now() - start_time).total_seconds()
            )
        
        self.logger.info(f"Found {len(output.pain_points)} validated pain points")
        return output
    
    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response content, handling markdown and common issues."""
        try:
            # Check if the content contains a code block
            if "```json" in content:
                # Extract content between ```json and ```
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Extract content between ``` and ```
                content = content.split("```")[1].split("```")[0].strip()
                
            # Remove any trailing commas before closing braces/brackets (common JSON error)
            content = content.replace(",}", "}").replace(",]", "]")
            
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {str(e)}\n{traceback.format_exc()}\nContent: {content}")
            return {}
    
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
