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
from ..mcp_integration.client.base import MCPClient
from ..mcp_integration.config import load_server_configs
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


@dataclass(kw_only=True)
class ScoutInput:
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


@dataclass(kw_only=True)
class ScoutOutput:
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
        # Initialize BaseAgent with the correct name and agent_id
        BaseAgent.__init__(self, name="scout_agent", agent_id=agent_id)
        # Do not force a backend here; honor global/per-agent config in LLMAgentMixin
        LLMAgentMixin.__init__(self, preferred_backend=None)
        self.name = "scout_agent"  # Used for prompt template loading
        self.research_agent = ResearchAgent()
        self.config = get_config()

    def _normalize_input(self, agent_input: Any) -> ScoutInput:
        """Coerce incoming input into a ScoutInput-like structure.

        Accepts either:
        - a ScoutInput instance
        - an AgentInput with data dict containing required fields
        - a plain dict with required fields
        """
        if isinstance(agent_input, ScoutInput):
            return agent_input
        # If it's an AgentInput, prefer its data payload
        payload = None
        if hasattr(agent_input, "data"):
            payload = agent_input.data
        elif isinstance(agent_input, dict):
            payload = agent_input
        else:
            payload = {}

        payload = payload or {}
        # Map fields with defaults similar to ScoutInput
        target_market = payload.get("target_market") or payload.get("market") or ""
        research_scope = payload.get("research_scope", "comprehensive")
        max_pain_points = int(payload.get("max_pain_points", 10))
        sources = payload.get("sources") or ["reddit", "twitter", "forums", "reviews", "blogs"]
        keywords = payload.get("keywords") or ["pain point", "problem", "frustration", "issue"]
        return ScoutInput(
            target_market=target_market,
            research_scope=research_scope,
            max_pain_points=max_pain_points,
            sources=sources,
            keywords=keywords,
        )
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Plan the pain point discovery process."""
        # Normalize incoming input into ScoutInput
        input_data = self._normalize_input(agent_input)
        self.logger.info(f"Planning pain point discovery for market: {input_data.target_market}")
        
        try:
            # Discover available MCP tools (names + descriptions) for the planner
            tools_catalog = []
            tool_names = []
            try:
                servers = load_server_configs()
                # Aggregate tools from all configured servers (best-effort)
                for srv in servers:
                    url = srv.get("url")
                    if not url:
                        continue
                    client = MCPClient(url)
                    try:
                        tools = await client.list_tools()
                        for t in (tools or []):
                            # Best-effort extraction of fields from MCP tool objects
                            name = getattr(t, "name", None) or getattr(t, "tool", None) or str(t)
                            desc = getattr(t, "description", "")
                            # Some MCP tool objects might expose input schema; capture if available
                            schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                            tools_catalog.append({"name": name, "description": desc, "input_schema": getattr(schema, "model_dump", lambda: schema)() if hasattr(schema, "model_dump") else schema})
                            tool_names.append(name)
                    finally:
                        try:
                            await client.shutdown()
                        except Exception:
                            pass
            except Exception:
                # If discovery fails, proceed with empty catalog; prompt will still work
                tools_catalog = []
                tool_names = []

            # Prepare prompt substitutions
            print('--------- -------        --------------------------------')
            print(tool_names)
            print(tools_catalog)
            print('--------- -------        --------------------------------')
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "max_pain_points": input_data.max_pain_points,
                "sources": json.dumps(input_data.sources),
                "keywords": json.dumps(input_data.keywords),
                "subreddits": json.dumps([k for k in input_data.keywords if isinstance(k, str) and k.startswith("r/")]),
                "limits_json": json.dumps({
                    "per_query_limit": 50,
                    "comment_depth": 2,
                    "comment_limit": 200,
                    "min_num_comments": 5,
                    "min_score": 3
                }),
                "tools_json": json.dumps(tools_catalog, ensure_ascii=False),
                "tool_names_csv": ", ".join(tool_names)
            }

            # Load and render the planning prompt template
            prompt_text = load_prompt_template(template_name="plan.prompt", agent_name=self.name, substitutions=substitutions)

            # Generate plan using LLM (returns string)
            try:
                llm_text = await self.llm_generate(prompt=prompt_text, task_type="plan")
                plan = self._extract_json(llm_text)
                # Post-process to enforce Option A (strict) and add execution code
                plan = self._postprocess_plan(plan, input_data, tools_catalog, tool_names)
                self.logger.info(f"Generated plan with keys: {list(plan.keys())}")
            except Exception:
                # Fallback if LLM fails
                self.logger.warning("LLM plan generation failed, using fallback plan")
                plan = {
                    "phases": [
                        {"name": "collect", "source": "reddit"},
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
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research data to identify pain points."""
        self.logger.info("Thinking about discovered pain points...")
        
        # Normalize input data
        input_data = self._normalize_input(agent_input)

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
            
            # Prepare prompt substitutions
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "plan": json.dumps(self.state.plan),
                "research_results": json.dumps(research_results)
            }
            
            # Load and render the thinking prompt template
            prompt_text = load_prompt_template(template_name="think.prompt", agent_name=self.name, substitutions=substitutions)

            # Generate analysis using LLM (returns string)
            try:
                llm_text = await self.llm_generate(prompt=prompt_text, task_type="think")
                analysis = self._extract_json(llm_text)
                self.logger.info("Generated analysis via LLM")
            except Exception:
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
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> ScoutOutput:
        """Execute pain point discovery and return results."""
        self.logger.info("Executing pain point discovery...")
        
        start_time = datetime.now()
        
        try:
            # Normalize input data
            input_data = self._normalize_input(agent_input)

            # Get research results and analysis from state (ensure available before prompt formatting)
            research_results = getattr(self.state, 'research_results', {})
            analysis = getattr(self.state, 'analysis', {})

            # Load the action prompt template
            prompt_text = load_prompt_template(template_name="act.prompt", agent_name=self.name, substitutions={
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "max_pain_points": input_data.max_pain_points,
                "plan": json.dumps(self.state.plan),
                "analysis": json.dumps(analysis),
                "research_results": json.dumps(research_results)
            })
            
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
            
            # Generate action result using LLM (returns string)
            try:
                llm_text = await self.llm_generate(prompt=prompt_text, task_type="act")
                act_result = self._extract_json(llm_text)
                
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
            except Exception:
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

    def _postprocess_plan(self, plan: Dict[str, Any], input_data: "ScoutInput", tools_catalog: List[Dict[str, Any]], tool_names: List[str]) -> Dict[str, Any]:
        """Enforce Option A: ensure tool nodes have tool/params/code and normalize outputs.

        - Add version: "1.1"
        - Normalize outputs to simple filenames (executor resolves run dir)
        - For each tool node:
            - Ensure `params` exists (fallback to `inputs`)
            - Ensure `code` exists: generate a minimal sandbox snippet or 'no_op'
        - Optionally add collect nodes per available sources if missing.
        """
        try:
            if not isinstance(plan, dict):
                return {}

            plan.setdefault("schema", "scout_plan_v1")
            plan.setdefault("version", "1.1")

            dag = plan.get("dag") or {}
            nodes = list(dag.get("nodes") or [])

            # Helper: simple filename from any path-like string
            def _basename(p: str) -> str:
                try:
                    import os
                    return os.path.basename(p)
                except Exception:
                    return p

            # Build a quick map of existing source-collect nodes
            existing_tools = {n.get("tool"): n for n in nodes if n.get("type") == "tool"}

            # Heuristic mapping: find tools containing source name
            available_tools_by_source = {}
            for name in tool_names:
                lower = (name or "").lower()
                for src in (plan.get("sources") or input_data.sources or []):
                    s = (src or "").lower()
                    if s and s in lower:
                        available_tools_by_source.setdefault(s, []).append(name)

            # Ensure collect nodes for sources with available tools
            desired_sources = [s.lower() for s in (plan.get("sources") or input_data.sources or [])]
            for s in desired_sources:
                tools_for_s = available_tools_by_source.get(s, [])
                if not tools_for_s:
                    continue
                # If no node exists for any of these tools, create one using the first tool
                already_present = any((n.get("type") == "tool" and isinstance(n.get("tool"), str) and s in n.get("tool", "").lower()) for n in nodes)
                if not already_present:
                    tool_name = tools_for_s[0]
                    node_id = f"collect_{s}"
                    outputs = [f"{s}_index.json"]
                    params = {
                        k: plan.get(k) for k in ("keywords", "subreddits", "time_window") if plan.get(k) is not None
                    }
                    code = (
                        f"result = mcp_call(tool=\"{tool_name}\", params={json.dumps(params) if params else '{}'}); "
                        f"save_json(\"{outputs[0]}\", result)"
                    )
                    nodes.append({
                        "id": node_id,
                        "type": "tool",
                        "tool": tool_name,
                        "params": params or {},
                        "code": code,
                        "inputs": {},
                        "outputs": outputs,
                        "deps": ["plan"],
                    })

            # Normalize existing nodes
            for n in nodes:
                # Outputs to filenames
                outs = n.get("outputs") or []
                n["outputs"] = [_basename(o) for o in outs]

                if n.get("type") == "tool":
                    # Ensure tool present
                    tool = n.get("tool")
                    if not tool or not isinstance(tool, str):
                        continue  # will be caught by runtime validation

                    # Ensure params
                    if not isinstance(n.get("params"), dict) or n.get("params") is None:
                        # Best-effort: copy known fields from inputs
                        params = {}
                        for key in ("keywords", "subreddits", "time_window", "per_query_limit", "include_comments", "comment_depth", "comment_limit", "use_cache"):
                            if key in (n.get("inputs") or {}):
                                params[key] = n["inputs"][key]
                            elif key in plan:
                                params[key] = plan[key]
                            elif key in (plan.get("limits") or {}):
                                params[key] = plan["limits"][key]
                        n["params"] = params

                    # Ensure code
                    code = n.get("code")
                    primary_out = (n.get("outputs") or [f"{tool}_output.json"])[0]
                    if not code or not isinstance(code, str) or not code.strip():
                        n["code"] = (
                            f"result = mcp_call(tool=\"{tool}\", params={json.dumps(n.get('params') or {})}); "
                            f"save_json(\"{primary_out}\", result)"
                        )

            # Put back nodes
            dag["nodes"] = nodes
            plan["dag"] = dag

            return plan
        except Exception:
            # On any failure, return original plan
            return plan
    
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
register_agent(ScoutAgent)
