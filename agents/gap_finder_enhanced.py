"""
Enhanced GapFinderAgent with LLM integration.

This agent demonstrates how to integrate the Multi-Backend LLM Abstraction Layer
into the existing ScoutAgent architecture.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from agents.base import BaseAgent, AgentInput, AgentOutput, AgentState
from agents.gap_finder import MarketGap, GapFinderInput, GapFinderOutput
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
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the enhanced gap finder agent."""
        super().__init__(name="EnhancedGapFinderAgent", config=config)
        self.description = "AI-powered market gap analysis with LLM insights"
        # Initialize LLM capabilities after base initialization
        LLMAgentMixin.__init__(self)
        self.initialize_llm()
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan the enhanced market gap analysis process."""
        if not isinstance(agent_input.data, GapFinderInput):
            raise ValueError("Expected GapFinderInput data")
        
        input_data = agent_input.data
        
        self.logger.info(f"Planning enhanced gap analysis for {len(input_data.validated_pain_points)} pain points")
        
        plan = {
            "phases": [
                "llm_pain_point_analysis",
                "llm_market_theme_identification", 
                "llm_competitive_landscape_analysis",
                "llm_opportunity_assessment",
                "llm_solution_generation",
                "llm_risk_analysis",
                "results_synthesis"
            ],
            "analysis_scope": input_data.analysis_scope,
            "use_llm": True,
            "llm_backend": self._llm_manager.get_default_backend() if self._llm_manager else "auto",
            "include_competitive_analysis": input_data.include_competitive_analysis,
            "include_market_sizing": input_data.include_market_sizing,
            "expected_duration": 600,  # 10 minutes with LLM processing
            "pain_point_count": len(input_data.validated_pain_points)
        }
        
        return plan
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced thinking phase using LLM analysis."""
        input_data = agent_input.data
        
        self.logger.info("Starting enhanced LLM-powered analysis...")
        
        # Phase 1: LLM Pain Point Analysis
        pain_point_analysis = await self._llm_analyze_pain_points(
            input_data.validated_pain_points,
            input_data.market_context
        )
        
        # Phase 2: Market Theme Identification
        market_themes = await self._llm_identify_market_themes(
            pain_point_analysis,
            input_data.market_context
        )
        
        # Phase 3: Competitive Landscape Analysis (if enabled)
        competitive_analysis = {}
        if input_data.include_competitive_analysis:
            competitive_analysis = await self._llm_analyze_competitive_landscape(
                market_themes,
                input_data.market_context
            )
        
        thoughts = {
            "pain_point_analysis": pain_point_analysis,
            "market_themes": market_themes,
            "competitive_analysis": competitive_analysis,
            "analysis_confidence": self._calculate_analysis_confidence(
                pain_point_analysis, market_themes, competitive_analysis
            ),
            "llm_insights": await self._extract_key_insights(
                pain_point_analysis, market_themes, competitive_analysis
            )
        }
        
        return thoughts
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> GapFinderOutput:
        """Execute enhanced market gap analysis with LLM-generated insights."""
        input_data = agent_input.data
        
        self.logger.info("Executing enhanced gap analysis with LLM insights...")
        
        # Generate market gaps using LLM analysis
        market_gaps = await self._generate_enhanced_market_gaps(
            thoughts["market_themes"],
            thoughts["competitive_analysis"],
            input_data
        )
        
        # Generate LLM-powered competitive landscape
        competitive_landscape = await self._generate_competitive_landscape(
            market_gaps,
            thoughts["competitive_analysis"]
        )
        
        # Generate market analysis with LLM insights
        market_analysis = await self._generate_market_analysis(
            market_gaps,
            thoughts["llm_insights"]
        )
        
        # Prioritize opportunities using LLM scoring
        prioritized_opportunities = await self._llm_prioritize_opportunities(market_gaps)
        
        # Generate LLM recommendations
        recommendations = await self._generate_llm_recommendations(
            market_gaps,
            prioritized_opportunities,
            thoughts["llm_insights"]
        )
        
        # Risk assessment with LLM
        risk_assessment = await self._llm_assess_risks(
            market_gaps,
            input_data.market_context
        )
        
        return GapFinderOutput(
            market_gaps=market_gaps,
            prioritized_opportunities=prioritized_opportunities,
            market_analysis=market_analysis,
            competitive_landscape=competitive_landscape,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            result={
                "enhanced_analysis": True,
                "llm_powered": True,
                "confidence_score": thoughts["analysis_confidence"],
                "key_insights": thoughts["llm_insights"]
            }
        )
    
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
