"""
Enhanced GapFinderAgent with LLM integration.

This agent demonstrates how to integrate the Multi-Backend LLM Abstraction Layer
into the existing ScoutAgent architecture.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from agents.gap_finder import MarketGap, GapFinderInput, GapFinderOutput
from llm.utils import LLMAgentMixin, AgentPrompt, AgentPromptTemplates, load_prompt_template
from llm.utils import LLMAgentMixin, AgentPrompt, AgentPromptTemplates
from config import get_config


class EnhancedGapFinderAgent(BaseAgent, LLMAgentMixin):
    """
    Enhanced GapFinderAgent with LLM integration for intelligent market gap analysis.
    
    This agent uses LLMs to:
    - Analyze pain points and identify market themes
    - Generate market insights and competitive analysis
    - Create solution ideas and recommendations
    - Assess market opportunities with AI-powered analysis
    """
    
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Async execute method to handle async plan/think/act methods."""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting execution of agent {self.name}")
        
        try:
            # Planning phase
            self._update_status('planning')
            plan = await self.plan(agent_input)
            self.logger.debug(f"Plan created: {plan}")
            
            # Thinking phase
            self._update_status('thinking')
            thoughts = await self.think(agent_input, plan)
            self.logger.debug(f"Thoughts: {thoughts}")
            
            # Action phase
            self._update_status('acting')
            result = await self.act(agent_input, plan, thoughts)
            self.logger.debug(f"Action result: {result}")
            
            # Create output
            execution_time = time.time() - self.start_time
            output = AgentOutput(
                result=result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'plan': plan,
                    'thoughts': thoughts
                },
                logs=self.execution_logs,
                execution_time=execution_time,
                success=True
            )
            
            self._update_status('completed')
            self.logger.info(f"Agent {self.name} completed successfully in {execution_time:.2f}s")
            return output
            
        except Exception as e:
            execution_time = time.time() - self.start_time
            error_msg = f"Agent {self.name} failed: {str(e)}"
            self.logger.error(error_msg)
            self._update_status('failed')
            
            output = AgentOutput(
                result=None,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'error': str(e)
                },
                logs=self.execution_logs,
                execution_time=execution_time,
                success=False
            )
            return output
    
    def __init__(self, config: Optional[Any] = None, agent_id: str = None):
        """Initialize the enhanced gap finder agent."""
        # Initialize BaseAgent
        BaseAgent.__init__(self, agent_id=agent_id, name="gap_finder_agent", config=config)
        self.description = "AI-powered market gap analysis with LLM insights"
        
        # Initialize LLM capabilities
        from llm.utils import LLMAgentMixin
        LLMAgentMixin.__init__(self, preferred_backend="claude")
        
        # Explicitly initialize LLM manager
        if not hasattr(self, '_llm_manager') or self._llm_manager is None:
            self.initialize_llm()
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan the enhanced market gap analysis process."""
        if not isinstance(agent_input.data, GapFinderInput):
            raise ValueError("Expected GapFinderInput data")
        
        input_data = agent_input.data
        pain_points_count = len(input_data.validated_pain_points)
        
        self.logger.info(f"Planning enhanced gap analysis for {pain_points_count} pain points")
        
        # Load the plan prompt template
        prompt_template = load_prompt_template(
            'plan.prompt',
            agent_name=self.name,
            substitutions={
                'pain_points_count': pain_points_count,
                'market_context': input_data.market_context,
                'analysis_scope': input_data.analysis_scope,
                'include_competitive_analysis': input_data.include_competitive_analysis,
                'include_market_sizing': input_data.include_market_sizing
            }
        )
        
        # Generate plan using LLM
        response = await self.llm_generate(prompt_template)
        
        # Extract plan from LLM response
        plan = self._extract_json(response)
        
        if not plan:
            self.logger.warning("Failed to parse market gap analysis plan from LLM response, using default")
            plan = {
                "operation": "market_gap_analysis",
                "phases": [
                    "pain_point_clustering",
                    "market_research",
                    "competitive_analysis",
                    "gap_identification",
                    "opportunity_scoring",
                    "risk_assessment",
                    "recommendation_generation"
                ],
                "expected_duration": 600,  # 10 minutes
                "data_sources": ["pain_points", "market_context"],
                "special_considerations": [input_data.analysis_scope]
            }
        
        self.logger.info(f"Market gap analysis plan created with {len(plan.get('phases', []))} phases")
        self.state.plan = plan
        return plan
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from LLM output text with enhanced parsing."""
        # Log the full response for debugging
        self.logger.debug(f"Raw LLM response: {text}")
        
        try:
            # Try to load as-is first
            return json.loads(text)
        except json.JSONDecodeError:
            # Look for JSON block in markdown format (common LLM output pattern)
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
            if json_match:
                json_content = json_match.group(1).strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error in markdown block: {e}")
                    # Try to clean up common issues with LLM-generated JSON
                    # Remove trailing commas before closing brackets or braces
                    cleaned_json = re.sub(r',\s*([\]\}])', r'\1', json_content)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse JSON even after cleanup")
            
            # Look for { ... } pattern as a fallback
            brace_match = re.search(r"\{[\s\S]*?\}", text)
            if brace_match:
                try:
                    json_content = brace_match.group(0)
                    # Try cleaning this content too
                    cleaned_json = re.sub(r',\s*([\]\}])', r'\1', json_content)
                    return json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error in brace match: {e}")
        
        self.logger.error(f"Failed to decode JSON: {text[:100]}...")
        return {}
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced thinking phase using LLM analysis."""
        input_data = agent_input.data
        pain_points = input_data.validated_pain_points
        pain_points_count = len(pain_points)
        
        self.logger.info("Analyzing market gaps and opportunities...")
        
        # Get a sample of pain points for the prompt template
        pain_points_sample = json.dumps(pain_points[:3], indent=2) if pain_points else "[]"
        
        # Load the think prompt template
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'think.prompt',
            agent_name=self.name,
            substitutions={
                'pain_points_count': pain_points_count,
                'market_context': input_data.market_context,
                'analysis_scope': input_data.analysis_scope,
                'pain_points_sample': pain_points_sample
            }
        )
        
        # Generate analysis using LLM
        response = await self.llm_generate(prompt_template)
        
        # Extract analysis from LLM response
        thoughts = self._extract_json(response)
        
        if not thoughts:
            self.logger.warning("Failed to parse market gap analysis from LLM response, using traditional methods")
            # Use LLM to analyze pain points
            pain_point_analysis = await self._llm_analyze_pain_points(
                pain_points, 
                input_data.market_context
            )
            
            # Use LLM to identify market themes
            market_themes = await self._llm_identify_market_themes(
                pain_point_analysis, 
                input_data.market_context
            )
            
            # Extract key insights
            insights = await self._extract_key_insights(
                pain_point_analysis,
                market_themes,
                {}
            )
            
            # Calculate analysis confidence
            confidence = self._calculate_analysis_confidence(
                pain_point_analysis,
                market_themes,
                {}
            )
            
            thoughts = {
                "market_themes": market_themes,
                "data_complexity": "medium",
                "storage_requirements": {
                    "estimated_size_kb": 100,
                    "entries_count": len(market_themes),
                    "complexity": "medium"
                },
                "analysis_scope": input_data.analysis_scope,
                "expected_insights": len(insights)
            }
        
        self.logger.info("Market gap analysis completed.")
        self.state.thoughts = thoughts
        return thoughts
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> GapFinderOutput:
        """Execute enhanced market gap analysis with LLM-generated insights."""
        input_data = agent_input.data
        market_themes = thoughts.get("market_themes", [])
        market_themes_count = len(market_themes)
        pain_points_count = len(input_data.validated_pain_points)
        
        self.logger.info(f"Executing market gap analysis with {market_themes_count} market themes")
        start_time = datetime.now()
        
        # Prepare market themes sample for the prompt
        market_themes_sample = json.dumps(market_themes[:2], indent=2) if market_themes else "[]"
        
        # Load the act prompt template
        from llm.utils import load_prompt_template
        prompt_template = load_prompt_template(
            'act.prompt',
            agent_name=self.name,
            substitutions={
                'market_themes_count': market_themes_count,
                'pain_points_count': pain_points_count,
                'market_context': input_data.market_context,
                'analysis_scope': input_data.analysis_scope,
                'include_competitive_analysis': input_data.include_competitive_analysis,
                'include_market_sizing': input_data.include_market_sizing,
                'market_themes_sample': market_themes_sample
            }
        )
        
        # Generate execution results using LLM
        response = await self.llm_generate(prompt_template)
        
        # Extract execution results from LLM response
        results = self._extract_json(response)
        
        if not results:
            self.logger.warning("Failed to parse market gap results from LLM response, falling back to traditional methods")
            # Generate market gaps using traditional method
            market_gaps = await self._generate_enhanced_market_gaps(
                market_themes, 
                {},  # empty competitive analysis
                input_data
            )
            
            # Prioritize opportunities using traditional method
            prioritized_opportunities = await self._llm_prioritize_opportunities(market_gaps)
            
            # Generate recommendations using traditional method
            recommendations = await self._generate_llm_recommendations(
                market_gaps,
                prioritized_opportunities,
                []
            )
            
            # Risk assessment using traditional method
            risk_assessment = await self._llm_assess_risks(
                market_gaps,
                input_data.market_context
            )
            
            # Create result structure to match expected output
            results = {
                "market_gaps": [gap.to_dict() for gap in market_gaps],
                "prioritized_opportunities": prioritized_opportunities,
                "market_analysis": await self._generate_market_analysis(market_gaps, []),
                "competitive_landscape": await self._generate_competitive_landscape(market_gaps, {}),
                "recommendations": recommendations,
                "risk_assessment": risk_assessment
            }
        
        # Convert market_gaps from dict to MarketGap objects if needed
        market_gaps = []
        for gap_dict in results.get("market_gaps", []):
            market_gap = MarketGap(
                gap_description=gap_dict.get("gap_description", "Unknown gap"),
                market_size=float(gap_dict.get("market_size", 0.0)),
                competition_level=gap_dict.get("competition_level", "medium"),
                opportunity_score=float(gap_dict.get("opportunity_score", 0.0)),
                target_segments=gap_dict.get("target_segments", []),
                solution_ideas=gap_dict.get("solution_ideas", []),
                barriers_to_entry=gap_dict.get("barriers_to_entry", []),
                estimated_tam=float(gap_dict.get("estimated_tam", 0.0)),
                estimated_sam=float(gap_dict.get("estimated_sam", 0.0)),
                estimated_som=float(gap_dict.get("estimated_som", 0.0)),
                risk_factors=gap_dict.get("risk_factors", []),
                timeline_to_market=gap_dict.get("timeline_to_market", "12-18 months")
            )
            market_gaps.append(market_gap)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create and return output
        output = GapFinderOutput(
            market_gaps=market_gaps,
            prioritized_opportunities=results.get("prioritized_opportunities", []),
            market_analysis=results.get("market_analysis", {}),
            competitive_landscape=results.get("competitive_landscape", {}),
            recommendations=results.get("recommendations", []),
            risk_assessment=results.get("risk_assessment", {}),
            execution_time=execution_time,
            metadata={
                "llm_enhanced": True,
                "prompt_driven": True,
                "market_themes_count": market_themes_count,
                "pain_points_count": pain_points_count
            }
        )
        
        return output
    
    async def _llm_analyze_pain_points(self, pain_points: List[Dict[str, Any]], 
                                     market_context: str) -> Dict[str, Any]:
        """Use LLM to analyze pain points and extract insights."""
        pain_points_text = "\n".join([
            f"- {pp.get('description', str(pp))}" for pp in pain_points
        ])
        
        prompt = AgentPrompt(
            system_prompt="""You are an expert market researcher analyzing customer pain points.
            
            Analyze the provided pain points and provide insights in JSON format:
            {
                "pain_point_clusters": [
                    {
                        "theme": "theme_name",
                        "pain_points": ["pain1", "pain2"],
                        "severity": "high|medium|low",
                        "frequency": "common|occasional|rare",
                        "market_impact": "description"
                    }
                ],
                "key_insights": ["insight1", "insight2"],
                "market_implications": ["implication1", "implication2"],
                "urgency_assessment": "high|medium|low"
            }""",
            user_prompt=f"""Market Context: {market_context}

Pain Points to Analyze:
{pain_points_text}

Please analyze these pain points and provide structured insights."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON, using fallback analysis")
            return {
                "pain_point_clusters": [{"theme": "general", "pain_points": pain_points_text.split('\n')}],
                "key_insights": ["Analysis requires manual review"],
                "raw_response": response
            }
    
    async def _llm_identify_market_themes(self, pain_point_analysis: Dict[str, Any],
                                        market_context: str) -> List[Dict[str, Any]]:
        """Use LLM to identify market themes from pain point analysis."""
        prompt = AgentPrompt(
            system_prompt="""You are a market strategist identifying key market themes.
            
            Based on pain point analysis, identify market themes in JSON format:
            [
                {
                    "theme": "theme_name",
                    "description": "detailed description",
                    "market_size_estimate": "small|medium|large",
                    "growth_potential": "low|medium|high", 
                    "competition_level": "low|medium|high",
                    "related_pain_points": ["pain1", "pain2"],
                    "opportunity_score": 0-100
                }
            ]""",
            user_prompt=f"""Market Context: {market_context}

Pain Point Analysis:
{json.dumps(pain_point_analysis, indent=2)}

Please identify key market themes and opportunities."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.4)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse market themes, using fallback")
            return [{"theme": "general_market", "description": "Requires manual analysis"}]
    
    async def _llm_analyze_competitive_landscape(self, market_themes: List[Dict[str, Any]],
                                               market_context: str) -> Dict[str, Any]:
        """Use LLM to analyze competitive landscape."""
        themes_text = "\n".join([f"- {theme['theme']}: {theme.get('description', '')}" 
                                for theme in market_themes])
        
        prompt = AgentPrompt(
            system_prompt="""You are a competitive intelligence analyst.
            
            Analyze the competitive landscape for the given market themes in JSON format:
            {
                "overall_competition": "low|medium|high",
                "key_competitors": [
                    {
                        "name": "competitor_name",
                        "market_position": "leader|challenger|follower|niche",
                        "strengths": ["strength1", "strength2"],
                        "weaknesses": ["weakness1", "weakness2"],
                        "market_share_estimate": "percentage or description"
                    }
                ],
                "competitive_gaps": ["gap1", "gap2"],
                "barriers_to_entry": ["barrier1", "barrier2"],
                "opportunities": ["opportunity1", "opportunity2"]
            }""",
            user_prompt=f"""Market Context: {market_context}

Market Themes:
{themes_text}

Please analyze the competitive landscape for these themes."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"overall_competition": "medium", "analysis_note": "Requires manual review"}
    
    async def _generate_enhanced_market_gaps(self, market_themes: List[Dict[str, Any]],
                                           competitive_analysis: Dict[str, Any],
                                           input_data: GapFinderInput) -> List[MarketGap]:
        """Generate market gaps using LLM analysis."""
        market_gaps = []
        
        for theme in market_themes:
            # Use LLM to generate detailed gap analysis
            gap_analysis = await self._llm_analyze_specific_gap(theme, competitive_analysis, input_data)
            
            # Create MarketGap object
            market_gap = MarketGap(
                gap_description=gap_analysis.get("gap_description", theme["theme"]),
                market_size=self._estimate_market_size(gap_analysis),
                competition_level=gap_analysis.get("competition_level", "medium"),
                opportunity_score=gap_analysis.get("opportunity_score", theme.get("opportunity_score", 50)),
                target_segments=gap_analysis.get("target_segments", ["general"]),
                solution_ideas=gap_analysis.get("solution_ideas", []),
                barriers_to_entry=gap_analysis.get("barriers_to_entry", []),
                estimated_tam=gap_analysis.get("estimated_tam", 1000000),
                estimated_sam=gap_analysis.get("estimated_sam", 100000),
                estimated_som=gap_analysis.get("estimated_som", 10000),
                risk_factors=gap_analysis.get("risk_factors", []),
                timeline_to_market=gap_analysis.get("timeline_to_market", "12-18 months")
            )
            
            market_gaps.append(market_gap)
        
        return market_gaps
    
    async def _llm_analyze_specific_gap(self, theme: Dict[str, Any],
                                      competitive_analysis: Dict[str, Any],
                                      input_data: GapFinderInput) -> Dict[str, Any]:
        """Use LLM to analyze a specific market gap in detail."""
        prompt = AgentPrompt(
            system_prompt="""You are a market opportunity analyst providing detailed gap analysis.
            
            Provide comprehensive analysis in JSON format:
            {
                "gap_description": "detailed description of the market gap",
                "opportunity_score": 0-100,
                "competition_level": "low|medium|high",
                "target_segments": ["segment1", "segment2"],
                "solution_ideas": ["idea1", "idea2", "idea3"],
                "barriers_to_entry": ["barrier1", "barrier2"],
                "estimated_tam": 1000000,
                "estimated_sam": 100000,
                "estimated_som": 10000,
                "risk_factors": ["risk1", "risk2"],
                "timeline_to_market": "6-12 months",
                "key_success_factors": ["factor1", "factor2"]
            }""",
            user_prompt=f"""Market Theme: {theme['theme']}
Theme Description: {theme.get('description', '')}

Market Context: {input_data.market_context}

Competitive Analysis:
{json.dumps(competitive_analysis, indent=2)}

Please provide detailed analysis for this market gap."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.4)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "gap_description": theme["theme"],
                "opportunity_score": theme.get("opportunity_score", 50),
                "competition_level": "medium"
            }
    
    async def _llm_prioritize_opportunities(self, market_gaps: List[MarketGap]) -> List[Dict[str, Any]]:
        """Use LLM to prioritize market opportunities."""
        gaps_summary = []
        for gap in market_gaps:
            gaps_summary.append({
                "description": gap.gap_description,
                "opportunity_score": gap.opportunity_score,
                "market_size": gap.market_size,
                "competition_level": gap.competition_level
            })
        
        prompt = AgentPrompt(
            system_prompt="""You are a strategic advisor prioritizing market opportunities.
            
            Rank opportunities and provide analysis in JSON format:
            [
                {
                    "gap_description": "description",
                    "priority_rank": 1,
                    "priority_level": "high|medium|low",
                    "rationale": "explanation for ranking",
                    "recommended_action": "immediate|short-term|long-term",
                    "investment_level": "low|medium|high",
                    "expected_roi": "description"
                }
            ]""",
            user_prompt=f"""Market Opportunities to Prioritize:
{json.dumps(gaps_summary, indent=2)}

Please rank these opportunities by priority and provide strategic recommendations."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback prioritization
            return [
                {
                    "gap_description": gap.gap_description,
                    "priority_rank": i + 1,
                    "priority_level": "high" if gap.opportunity_score > 70 else "medium"
                }
                for i, gap in enumerate(sorted(market_gaps, key=lambda x: x.opportunity_score, reverse=True))
            ]
    
    async def _generate_llm_recommendations(self, market_gaps: List[MarketGap],
                                          prioritized_opportunities: List[Dict[str, Any]],
                                          insights: List[str]) -> List[str]:
        """Generate strategic recommendations using LLM."""
        prompt = AgentPrompt(
            system_prompt="""You are a strategic business consultant providing actionable recommendations.
            
            Based on market gap analysis, provide 5-7 specific, actionable recommendations.
            Focus on:
            - Immediate next steps
            - Strategic priorities
            - Risk mitigation
            - Resource allocation
            - Timeline considerations""",
            user_prompt=f"""Market Gaps Analysis:
{len(market_gaps)} gaps identified

Top Opportunities:
{json.dumps(prioritized_opportunities[:3], indent=2)}

Key Insights:
{chr(10).join([f'- {insight}' for insight in insights])}

Please provide strategic recommendations for pursuing these market opportunities."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.4)
        
        # Extract recommendations from response
        recommendations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                recommendations.append(line.lstrip('-•0123456789. '))
        
        return recommendations if recommendations else [response]
    
    async def _llm_assess_risks(self, market_gaps: List[MarketGap],
                              market_context: str) -> Dict[str, Any]:
        """Use LLM to assess risks for market opportunities."""
        gaps_summary = [gap.gap_description for gap in market_gaps]
        
        prompt = AgentPrompt(
            system_prompt="""You are a risk assessment specialist analyzing market opportunities.
            
            Provide risk analysis in JSON format:
            {
                "overall_risk_level": "low|medium|high",
                "market_risks": ["risk1", "risk2"],
                "competitive_risks": ["risk1", "risk2"],
                "execution_risks": ["risk1", "risk2"],
                "financial_risks": ["risk1", "risk2"],
                "mitigation_strategies": [
                    {
                        "risk": "risk_description",
                        "mitigation": "mitigation_strategy",
                        "priority": "high|medium|low"
                    }
                ],
                "risk_score": 0-100
            }""",
            user_prompt=f"""Market Context: {market_context}

Market Opportunities:
{chr(10).join([f'- {gap}' for gap in gaps_summary])}

Please assess the risks associated with pursuing these market opportunities."""
        )
        
        response = await self.llm_generate(prompt, temperature=0.3)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "overall_risk_level": "medium",
                "market_risks": ["Market uncertainty", "Competition"],
                "risk_score": 50
            }
    
    async def _extract_key_insights(self, pain_point_analysis: Dict[str, Any],
                                  market_themes: List[Dict[str, Any]],
                                  competitive_analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights using LLM analysis."""
        analysis_data = {
            "pain_points": pain_point_analysis,
            "themes": market_themes,
            "competition": competitive_analysis
        }
        
        return await self.llm_extract_insights(
            analysis_data,
            insight_type="market_opportunity"
        )
    
    async def _generate_competitive_landscape(self, market_gaps: List[MarketGap],
                                            competitive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive competitive landscape analysis."""
        return {
            "total_gaps": len(market_gaps),
            "competitive_intensity": competitive_analysis.get("overall_competition", "medium"),
            "key_competitors": competitive_analysis.get("key_competitors", []),
            "competitive_gaps": competitive_analysis.get("competitive_gaps", []),
            "market_opportunities": [gap.gap_description for gap in market_gaps],
            "barriers_to_entry": competitive_analysis.get("barriers_to_entry", [])
        }
    
    async def _generate_market_analysis(self, market_gaps: List[MarketGap],
                                      insights: List[str]) -> Dict[str, Any]:
        """Generate comprehensive market analysis."""
        total_market_size = sum(gap.market_size for gap in market_gaps)
        avg_opportunity_score = sum(gap.opportunity_score for gap in market_gaps) / len(market_gaps) if market_gaps else 0
        
        return {
            "total_addressable_market": total_market_size,
            "average_opportunity_score": avg_opportunity_score,
            "market_gaps_count": len(market_gaps),
            "key_insights": insights,
            "market_segments": list(set(segment for gap in market_gaps for segment in gap.target_segments)),
            "llm_enhanced": True
        }
    
    def _estimate_market_size(self, gap_analysis: Dict[str, Any]) -> float:
        """Estimate market size from LLM analysis."""
        # Use LLM estimates if available, otherwise use defaults
        return gap_analysis.get("estimated_tam", 1000000)
    
    def _calculate_analysis_confidence(self, pain_point_analysis: Dict[str, Any],
                                     market_themes: List[Dict[str, Any]],
                                     competitive_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        confidence_factors = []
        
        # Pain point analysis confidence
        if pain_point_analysis.get("key_insights"):
            confidence_factors.append(0.8)
        
        # Market themes confidence
        if market_themes and len(market_themes) > 0:
            confidence_factors.append(0.9)
        
        # Competitive analysis confidence
        if competitive_analysis.get("key_competitors"):
            confidence_factors.append(0.85)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5


if __name__ == "__main__":
    import asyncio
    import logging
    from dataclasses import asdict
    from llm.backends.ollama import OllamaBackend
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("gap_finder_test")
    
    async def test_gap_finder_agent():
        logger.info("Testing EnhancedGapFinderAgent with prompt templates...")
        
        # Create test pain points
        test_pain_points = [
            {
                "pain_point_id": "pp1",
                "description": "Small businesses struggle to find affordable cybersecurity solutions",
                "severity": "high",
                "frequency": "high",
                "source": "customer interview"
            },
            {
                "pain_point_id": "pp2",
                "description": "SMBs lack technical expertise to implement security measures",
                "severity": "high",
                "frequency": "medium",
                "source": "market survey"
            },
            {
                "pain_point_id": "pp3",
                "description": "Existing security solutions are too complex for small teams",
                "severity": "medium",
                "frequency": "high",
                "source": "customer feedback"
            }
        ]
        
        # Create input data
        input_data = GapFinderInput(
            validated_pain_points=test_pain_points,
            market_context="Small business cybersecurity solutions market with focus on simplicity and affordability",
            analysis_scope="Identify gaps in SMB cybersecurity product offerings",
            include_competitive_analysis=True,
            include_market_sizing=True
        )
        
        # Create agent input
        agent_input = AgentInput(
            data=input_data,
            metadata={"agent_id": "gap-finder-test"}
        )
        
        # Import required classes for LLM backend setup
        from llm.backends.ollama import OllamaBackend
        from llm.base import LLMConfig, LLMBackendType
        from llm.manager import LLMManager
        
        # Create the LLM configuration for the Ollama backend
        config = LLMConfig(
            backend_type=LLMBackendType.OLLAMA,
            model_name="phi4-mini:latest",  # Using a smaller model for testing
            base_url="http://localhost:11434",
            timeout=180  # 3 minute timeout for complex queries
        )
        
        # Create the OllamaBackend with proper config
        llm_backend = OllamaBackend(config=config)
        
        # Create agent instance with ollama as preferred backend instead of claude
        agent = EnhancedGapFinderAgent(agent_id="gap-finder-test")
        agent.preferred_backend = "ollama"  # Override the default claude backend
        
        # We need to initialize the LLM capabilities correctly with proper backend registration
        # This is an async approach that follows the code pattern in LLMAgentMixin initialization
        llm_manager = LLMManager()
        await llm_manager.register_backend(llm_backend, is_default=True)
        
        # Set the LLM manager to the agent
        agent._llm_manager = llm_manager
        
        try:
            # Execute agent's plan phase
            logger.info("Running plan phase...")
            plan_result = await agent.plan(agent_input)
            logger.info(f"Plan complete: {len(plan_result.get('phases', []))} phases identified")
            
            # Execute agent's think phase
            logger.info("Running think phase...")
            think_result = await agent.think(agent_input, plan_result)
            logger.info(f"Think complete: {len(think_result.get('market_themes', []))} market themes identified")
            
            # Execute agent's act phase
            logger.info("Running act phase...")
            act_result = await agent.act(agent_input, plan_result, think_result)
            
            # Log results
            logger.info(f"Market gaps identified: {len(act_result.market_gaps)}")
            logger.info(f"Prioritized opportunities: {len(act_result.prioritized_opportunities)}")
            logger.info(f"Recommendations: {len(act_result.recommendations) if act_result.recommendations else 0}")
            
            # Print execution time
            logger.info(f"Execution time: {act_result.execution_time:.2f} seconds")
            
            # Success!  
            logger.info("Successfully executed all GapFinderAgent phases!")
            return act_result
        
        except Exception as e:
            logger.error(f"Error testing GapFinderAgent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    # Run the test
    result = asyncio.run(test_gap_finder_agent())
    
    if result:
        print("\n===== TEST SUCCESSFUL =====\n")
        print(f"GapFinderAgent completed analysis with {len(result.market_gaps)} market gaps identified")
        
        # Print first market gap as sample
        if result.market_gaps:
            first_gap = result.market_gaps[0]
            print(f"\nSample Market Gap: {first_gap.gap_description}")
            print(f"  Market Size: {first_gap.market_size}")
            print(f"  Opportunity Score: {first_gap.opportunity_score}")
            print(f"  Target Segments: {', '.join(first_gap.target_segments)}")
            print(f"  Solution Ideas: {', '.join(first_gap.solution_ideas[:2])}...")
    else:
        print("\n===== TEST FAILED =====\n")
