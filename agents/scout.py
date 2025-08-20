"""
ScoutAgent - Pain Point Discovery Agent

This agent specializes in discovering pain points in target markets using
web research, social media analysis, and user feedback collection.
"""

import asyncio
import json
import asyncio
import datetime
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from scout_agent.memory.manifest_manager import ManifestManager

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from ..config import get_config
from ..mcp_integration.client.base import MCPClient
from ..mcp_integration.config import load_server_configs
from ..llm.utils import LLMAgentMixin, load_prompt_template
from ..llm.base import LLMBackendType
from ..services.agents.code.service import CodeExecutionService
from dataclasses import dataclass, asdict
import textwrap
import traceback


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
    subreddits: List[str] = None
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ScoutOutput to a dictionary for JSON serialization."""
        return {
            "pain_points": [pp.to_dict() for pp in self.pain_points],
            "total_discovered": self.total_discovered,
            "market_summary": self.market_summary,
            "confidence_score": self.confidence_score,
            "sources_used": self.sources_used,
            "research_duration": self.research_duration
        }


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
        subreddits = payload.get("subreddits") or []
        
        # Debug log the extracted subreddits
        self.logger.info(f"DEBUG: _normalize_input extracted subreddits: {subreddits}")
        
        return ScoutInput(
            target_market=target_market,
            research_scope=research_scope,
            max_pain_points=max_pain_points,
            sources=sources,
            keywords=keywords,
            subreddits=subreddits,
        )
    
    async def plan(self, agent_input: AgentInput, run_id: Optional[str] = None) -> Dict[str, Any]:
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
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope,
                "max_pain_points": input_data.max_pain_points,
                "sources": json.dumps(input_data.sources),
                "keywords": json.dumps(input_data.keywords),
                "subreddits": json.dumps(getattr(input_data, 'subreddits', [])),
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
        
        # Ensure a run_id is present and store it for later stages
        try:
            dag = plan.get("dag") or {}
            # Prefer run_id from orchestrator parameter, then from state, then from plan, then generate new
            state_run_id = getattr(self.state, "run_id", None)
            final_run_id = run_id or state_run_id or dag.get("run_id") or plan.get("run_id")
            if not final_run_id:
                # Use same format as main_orchestrated.py to avoid duplicate folders
                final_run_id = f"scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dag["run_id"] = final_run_id
            plan["dag"] = dag
            plan["run_id"] = final_run_id
            # persist in state
            setattr(self.state, "run_id", final_run_id)
        except Exception:
            pass

        self.state.plan = plan

        # Initialize consolidated manifest file for this run
        try:
            project_root = Path(__file__).resolve().parents[2]
            run_dir = project_root / "data" / "runs" / (plan.get("dag", {}).get("run_id") or plan.get("run_id") or getattr(self.state, "run_id", "dev_run"))
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "run_manifest.json"
            # Load existing if present, otherwise start fresh
            try:
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            except Exception:
                manifest = {}
            # Persist core run metadata and seed stages.plan
            manifest["run_id"] = plan.get("run_id")
            manifest["dag"] = plan.get("dag") or {}
            stages = manifest.setdefault("stages", {})
            stages["plan"] = {
                "data": plan,
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            self.logger.info(f"Initialized manifest at {manifest_path}")
        except Exception as _e:
            # Non-fatal; downstream will create/append as needed
            self.logger.warning(f"Failed to initialize run_manifest.json: {_e}")

        return plan
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze collected Reddit data to identify pain points."""
        self.logger.info("Thinking about discovered pain points from collected Reddit data...")
        
        # Normalize input data
        input_data = self._normalize_input(agent_input)
        
        try:
            # First, try to get collect data from agent state (set by orchestrator)
            reddit_data = getattr(self.state, 'collect_data', None)
            
            if reddit_data:
                self.logger.info("Found collect data in agent state (from orchestrator)")
                self.logger.info(f"Agent state data type: {type(reddit_data)}")
                if isinstance(reddit_data, dict):
                    self.logger.info(f"Agent state data keys: {list(reddit_data.keys())}")
                    for key, value in reddit_data.items():
                        if isinstance(value, (list, dict)):
                            self.logger.info(f"  state.{key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                        else:
                            self.logger.info(f"  state.{key}: {type(value)} = {str(value)[:100]}...")
            else:
                # Fallback: try to get data from manifest (for compatibility with non-orchestrated runs)
                self.logger.info("No collect data in agent state, trying manifest...")
                
                # Get the run_id from the plan (try multiple possible locations)
                run_id = None
                if plan:
                    run_id = plan.get("run_metadata", {}).get("run_id")
                    if not run_id:
                        run_id = plan.get("run_id")
                    if not run_id:
                        run_id = plan.get("dag", {}).get("run_id")
                if not run_id:
                    # If still not found, use the one from agent state if available
                    run_id = getattr(self.state, "run_id", None)
                if not run_id:
                    raise ValueError("No run_id found in plan or agent state")
                
                self.logger.info(f"Using run_id: {run_id}")
                
                # Determine the manifest path
                run_dir = self._get_run_dir(run_id)
                manifest_path = run_dir / "run_manifest.json"
                
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Manifest not found at: {manifest_path}")
                
                # Load the manifest to access collected Reddit data
                manifest_manager = ManifestManager(manifest_path)
                
                # Get the collect node ID from the manifest
                collect_node_id = "scout_collect"  # Default standardized name
                manifest = manifest_manager.get_manifest()
                
                # Try to get data from stages first (orchestrator format)
                if "stages" in manifest and collect_node_id in manifest["stages"]:
                    stage_data = manifest["stages"][collect_node_id]
                    # Look for Reddit data in source-specific location
                    if "reddit" in stage_data:
                        reddit_data = stage_data["reddit"]
                        self.logger.info(f"Found collect data in manifest stages.{collect_node_id}.reddit")
                        self.logger.info(f"Reddit data type: {type(reddit_data)}")
                        if isinstance(reddit_data, dict):
                            self.logger.info(f"Reddit data keys: {list(reddit_data.keys())}")
                    # Fallback: try direct data field for backward compatibility
                    elif "data" in stage_data:
                        reddit_data = stage_data["data"]
                        self.logger.info(f"Found collect data in manifest stages.{collect_node_id}.data (fallback)")
                        self.logger.info(f"Stage data type: {type(reddit_data)}")
                        if isinstance(reddit_data, dict):
                            self.logger.info(f"Stage data keys: {list(reddit_data.keys())}")
                
                # If not found in stages, try node outputs (fallback)
                if not reddit_data:
                    reddit_data = manifest_manager.get_node_output(collect_node_id)
                    if reddit_data:
                        self.logger.info(f"Found collect data in manifest node outputs")
                        self.logger.info(f"Node output data type: {type(reddit_data)}")
                        if isinstance(reddit_data, dict):
                            self.logger.info(f"Node output keys: {list(reddit_data.keys())}")
                
                # If we still don't have data, raise an error
                if not reddit_data:
                    self.logger.error(f"No Reddit data found for node: {collect_node_id}")
                    self.logger.error(f"Manifest stages keys: {list(manifest.get('stages', {}).keys())}")
                    raise ValueError(f"No Reddit data found for node: {collect_node_id}")
            
            # Log the raw reddit_data structure for debugging
            self.logger.info(f"Raw Reddit data type: {type(reddit_data)}")
            if isinstance(reddit_data, dict):
                self.logger.info(f"Reddit data keys: {list(reddit_data.keys())}")
                for key, value in reddit_data.items():
                    if isinstance(value, (list, dict)):
                        self.logger.info(f"  {key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                    else:
                        self.logger.info(f"  {key}: {type(value)} = {str(value)[:100]}...")
            else:
                self.logger.info(f"Reddit data content: {str(reddit_data)[:200]}...")
            
            # Extract threads and comments from the collected data
            threads = reddit_data.get("threads", [])
            comments = reddit_data.get("comments", [])
            
            # Check if data is nested under 'result' key (common MCP tool pattern)
            if not threads and not comments and "result" in reddit_data:
                self.logger.info("Checking nested 'result' key for Reddit data")
                result_data = reddit_data["result"]
                self.logger.info(f"Result data type: {type(result_data)}")
                if isinstance(result_data, dict):
                    self.logger.info(f"Result data keys: {list(result_data.keys())}")
                    threads = result_data.get("threads", [])
                    comments = result_data.get("comments", [])
                    # Log nested structure
                    for key, value in result_data.items():
                        if isinstance(value, (list, dict)):
                            self.logger.info(f"  result.{key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'N/A'} items")
                        else:
                            self.logger.info(f"  result.{key}: {type(value)} = {str(value)[:100]}...")
            
            # Log what we found after extraction
            self.logger.info(f"Found {len(threads)} threads and {len(comments)} comments in the Reddit data")
            
            if not threads:
                self.logger.warning("No Reddit threads found in collected data")
                self.logger.warning(f"Final Reddit data keys: {list(reddit_data.keys()) if isinstance(reddit_data, dict) else 'not a dict'}")
            
            # Prepare prompt substitutions
            substitutions = {
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope
            }
            
            # Load and render the thinking prompt template
            prompt_text = load_prompt_template(template_name="think.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Add the Reddit data to the prompt
            # Limit the data to avoid exceeding token limits
            max_threads = 10
            max_comments_per_thread = 20
            
            # Prepare a condensed version of the Reddit data
            condensed_threads = threads[:max_threads]
            
            # Add comments to their respective threads
            thread_comments = {}
            for comment in comments:
                thread_id = comment.get("link_id", "").replace("t3_", "")
                if thread_id not in thread_comments:
                    thread_comments[thread_id] = []
                if len(thread_comments[thread_id]) < max_comments_per_thread:
                    thread_comments[thread_id].append(comment)
            
            # Add comments to threads
            for thread in condensed_threads:
                thread_id = thread.get("id", "").replace("t3_", "")
                thread["comments"] = thread_comments.get(thread_id, [])
            
            # Add the condensed Reddit data to the prompt
            prompt_text += f"\n\n# REDDIT DATA\n{json.dumps(condensed_threads, indent=2)}"
            
            # Generate analysis using LLM (returns string)
            try:
                llm_text = await self.llm_generate(prompt=prompt_text, task_type="think")
                analysis = self._extract_json(llm_text)
                self.logger.info("Generated analysis via LLM")
            except Exception as e:
                # Fallback if LLM fails
                self.logger.warning(f"LLM analysis generation failed: {str(e)}. Using fallback analysis.")
                analysis = {
                    "per_thread_summaries": [],
                    "pains": [],
                    "themes": [],
                    "error": "LLM analysis failed",
                    "total_threads_analyzed": len(threads),
                    "next_steps": ["retry_analysis", "validate_data"]
                }
                
            # Save analysis for the act phase
            self.state.analysis = analysis
            
            # Write analysis to manifest
            self._write_stage_output("think", analysis)
        
        except Exception as e:
            self.logger.error(f"Error in think phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback analysis
            analysis = {
                "per_thread_summaries": [],
                "pains": [],
                "themes": [],
                "error": f"Error in think phase: {str(e)}",
                "total_threads_analyzed": 0,
                "next_steps": ["check_data_collection", "retry_analysis"],
                "debug_info": {
                    "has_agent_state_data": hasattr(self.state, 'collect_data'),
                    "plan_provided": plan is not None
                }
            }
        
        return analysis
    
    def _get_run_dir(self, run_id: str) -> Path:
        """Get the run directory for a given run_id.
        
        Args:
            run_id: The run ID
            
        Returns:
            Path to the run directory
        """
        # Project root at ScoutAgent/ (not scout_agent/)
        root = Path(__file__).resolve().parents[2]
        run_dir = root / "data" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Using run directory: {run_dir}")
        return run_dir
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> ScoutOutput:
        """Consolidate and validate pain points from think stage analysis."""
        self.logger.info("Consolidating and validating pain points from think stage...")
        
        start_time = datetime.now()
        
        try:
            # Normalize input data
            input_data = self._normalize_input(agent_input)

            # Get the analysis from the think stage (this is our primary input)
            analysis = thoughts if thoughts else getattr(self.state, 'analysis', {})
            
            if not analysis:
                raise ValueError("No analysis available from think stage")
            
            # Load the action prompt template with think stage analysis
            prompt_text = load_prompt_template(template_name="act.prompt", agent_name=self.name, substitutions={
                "target_market": input_data.target_market,
                "research_scope": input_data.research_scope
            })
            
            # Add the think stage analysis to the prompt
            prompt_text += f"\n\n# THINK STAGE ANALYSIS\n{json.dumps(analysis, indent=2)}"
            
            # Generate consolidated pain points using LLM
            try:
                llm_text = await self.llm_generate(prompt=prompt_text, task_type="act")
                act_result = self._extract_json(llm_text)
                self.logger.info("Generated consolidated pain points via LLM")
                
                # Create pain point objects from the consolidated response
                pain_points = []
                for pp_data in act_result.get("pain_points", []):
                    pain_point = PainPoint(
                        description=pp_data.get("statement", pp_data.get("description", "")),
                        severity=pp_data.get("severity", "medium"),
                        market=input_data.target_market,
                        source="reddit",  # Primary source from our data collection
                        evidence=[str(ev.get("text", ev)) if isinstance(ev, dict) else str(ev) for ev in pp_data.get("evidence", [])],
                        frequency=pp_data.get("frequency_indicators", {}).get("thread_mentions", 0),
                        impact_score=pp_data.get("confidence_score", 0.5) * 10,  # Convert to 1-10 scale
                        tags=pp_data.get("tags", []),
                        discovered_at=datetime.now().isoformat()
                    )
                    pain_points.append(pain_point)
                
                # Prepare output using the consolidated result
                output = ScoutOutput(
                    pain_points=pain_points,
                    total_discovered=act_result.get("consolidation_summary", {}).get("consolidated_pains", len(pain_points)),
                    market_summary=f"Consolidated {act_result.get('consolidation_summary', {}).get('raw_pains_analyzed', 0)} raw pain points into {len(pain_points)} validated pain points",
                    confidence_score=act_result.get("consolidation_summary", {}).get("average_confidence", 0.0),
                    sources_used=act_result.get("sources_used", ["reddit"]),
                    research_duration=(datetime.now() - start_time).total_seconds()
                )
                
            except Exception as e:
                # Fallback: use raw analysis data if LLM consolidation fails
                self.logger.warning(f"LLM consolidation failed: {str(e)}. Using raw analysis data.")
                
                # Extract pain points directly from think stage analysis
                raw_pains = analysis.get("pains", [])
                pain_points = []
                
                for i, pain_data in enumerate(raw_pains):
                    pain_point = PainPoint(
                        description=pain_data.get("statement", f"Pain point {i+1}"),
                        severity="medium",  # Default severity
                        market=input_data.target_market,
                        source="reddit",
                        evidence=pain_data.get("evidence", []),
                        frequency=pain_data.get("support_count", 1),
                        impact_score=5.0,  # Default impact
                        tags=pain_data.get("tags", []),
                        discovered_at=datetime.now().isoformat()
                    )
                    pain_points.append(pain_point)
                
                # Prepare fallback output
                output = ScoutOutput(
                    pain_points=pain_points,
                    total_discovered=len(raw_pains),
                    market_summary=f"Extracted {len(pain_points)} pain points from think stage analysis",
                    confidence_score=0.6,  # Moderate confidence for fallback
                    sources_used=["reddit"],
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
        
        # Write output to manifest using the to_dict method
        self._write_stage_output("act", output.to_dict())
        
        self.logger.info(f"Found {len(output.pain_points)} validated pain points")
        return output
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain non-JSON content."""
        import re
        
        # Default fallback structure
        fallback = {
            "per_thread_summaries": [],
            "pains": [],
            "themes": [],
            "error": "Failed to parse JSON",
            "total_threads_analyzed": 0,
            "next_steps": ["retry_analysis", "check_data_quality"]
        }
        
        if not text:
            self.logger.error("Empty text provided to _extract_json")
            return fallback
            
        # First try to extract content from code blocks
        content = ""
        try:
            # Extract from markdown code blocks with priority
            if "```json" in text:
                # Extract content between ```json and ```
                content = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                # Extract content between ``` and ```
                content = text.split("```")[1].split("```")[0].strip()
            else:
                # Try to extract JSON directly - find the first { and the last }
                start = text.find('{')
                end = text.rfind('}')
                if start >= 0 and end > start:
                    content = text[start:end+1].strip()
                else:
                    content = text.strip()
            
            # Fix common JSON issues
            content = content.replace(",}", "}").replace(",]", "]")
            
            # Try to parse the JSON directly first
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Initial JSON parsing failed: {str(e)}. Attempting fixes...")
                
                # More aggressive fixes for common JSON issues
                # 1. Fix trailing commas in objects and arrays
                content = re.sub(r',\s*}', '}', content)
                content = re.sub(r',\s*\]', ']', content)
                
                # 2. Fix missing quotes around keys
                content = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', content)
                
                # 3. Fix unquoted values (true, false, null)
                content = re.sub(r':\s*true\s*([,}])', r':true\1', content)
                content = re.sub(r':\s*false\s*([,}])', r':false\1', content)
                content = re.sub(r':\s*null\s*([,}])', r':null\1', content)
                
                # 4. Fix unterminated strings
                content = re.sub(r'"([^"]*)$', r'"\1"', content)
                
                # Try parsing with the fixes
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 5. Fix single quotes used instead of double quotes
                    # This is tricky because we need to avoid replacing single quotes in text
                    try:
                        # Replace only single quotes that appear to be for keys or string values
                        content = re.sub(r"'([^']*)'\s*:", r'"\1":', content)  # For keys
                        content = re.sub(r":\s*'([^']*)'([,}\]])", r':"\1"\2', content)  # For values
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Try replacing all single quotes with double quotes as a last resort
                        content = content.replace("'", '"')
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Standard JSON parsing attempts failed: {str(e)}")
                            
                            # 6. Try to reconstruct a valid JSON structure
                            try:
                                # Extract key-value pairs using regex
                                pattern = r'"([^"]+)"\s*:\s*([^,}\]]+)'  # Match "key": value
                                matches = re.findall(pattern, content)
                                if matches:
                                    reconstructed = {}
                                    for key, value in matches:
                                        # Clean and parse the value
                                        value = value.strip()
                                        if value.startswith('"') and value.endswith('"'):
                                            reconstructed[key] = value[1:-1]  # String
                                        elif value.lower() == 'true':
                                            reconstructed[key] = True
                                        elif value.lower() == 'false':
                                            reconstructed[key] = False
                                        elif value.lower() == 'null':
                                            reconstructed[key] = None
                                        else:
                                            try:
                                                # Try to parse as number
                                                reconstructed[key] = float(value) if '.' in value else int(value)
                                            except ValueError:
                                                reconstructed[key] = value  # Keep as string
                                    
                                    if reconstructed:
                                        self.logger.info(f"Successfully reconstructed JSON with {len(reconstructed)} keys")
                                        return reconstructed
                            except Exception as recon_err:
                                self.logger.warning(f"JSON reconstruction failed: {str(recon_err)}")
                            
                            # 7. As a last resort, try to extract valid JSON objects using regex
                            try:
                                # Find all JSON-like objects in the text
                                pattern = r'\{[^\{\}]*(\(\{[^\{\}]*\}\)[^\{\}]*)*\}'
                                matches = re.findall(pattern, content)
                                if matches:
                                    for match in matches:
                                        try:
                                            if isinstance(match, tuple):
                                                for submatch in match:
                                                    try:
                                                        return json.loads(submatch)
                                                    except:
                                                        continue
                                            else:
                                                return json.loads(match)
                                        except:
                                            continue
                                
                                # Try another pattern for nested objects
                                pattern = r'\{[^\{\}]*(\{[^\{\}]*\}[^\{\}]*)*\}'
                                matches = re.findall(pattern, content)
                                if matches:
                                    for match in matches:
                                        try:
                                            return json.loads(match)
                                        except:
                                            continue
                            except Exception as regex_err:
                                self.logger.warning(f"Regex extraction failed: {str(regex_err)}")
                            
                            # 8. If all else fails, try to create a minimal valid structure
                            self.logger.warning("All JSON parsing methods failed, returning fallback structure")
                            # Try to extract any useful information from the text
                            try:
                                # Look for thread summaries
                                thread_pattern = r'thread.*?summary.*?:.*?([^,}\]]+)'
                                thread_matches = re.findall(thread_pattern, content, re.IGNORECASE)
                                if thread_matches:
                                    fallback["per_thread_summaries"] = [{'summary': m.strip()} for m in thread_matches]
                                
                                # Look for pain points
                                pain_pattern = r'pain.*?:.*?([^,}\]]+)'
                                pain_matches = re.findall(pain_pattern, content, re.IGNORECASE)
                                if pain_matches:
                                    fallback["pains"] = [{'description': m.strip()} for m in pain_matches]
                                
                                # Look for themes
                                theme_pattern = r'theme.*?:.*?([^,}\]]+)'
                                theme_matches = re.findall(theme_pattern, content, re.IGNORECASE)
                                if theme_matches:
                                    fallback["themes"] = [{'name': m.strip()} for m in theme_matches]
                                
                                fallback["error"] = f"JSON parsing failed: {str(e)}"
                                fallback["partial_content"] = content[:200] + "..." if len(content) > 200 else content
                            except Exception:
                                pass
                            
                            return fallback
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {str(e)}\n{traceback.format_exc()}\nContent: {content[:200] + '...' if len(content) > 200 else content}")
            return fallback

    def _postprocess_plan(self, plan: Dict[str, Any], input_data: "ScoutInput", tools_catalog: List[Dict[str, Any]], tool_names: List[str]) -> Dict[str, Any]:
        """Generate mini-plans for tool calls only, compatible with orchestrator.
        
        The orchestrator handles the main DAG structure (plan -> collect -> think -> act).
        This method only generates tool nodes that the orchestrator will integrate.
        """
        try:
            if not isinstance(plan, dict):
                return {"dag": {"nodes": []}}

            # Generate only tool nodes for the orchestrator to integrate
            tool_nodes = []
            
            # Heuristic mapping: find tools containing source name
            available_tools_by_source: Dict[str, List[str]] = {}
            for name in tool_names:
                lower = (name or "").lower()
                for src in (plan.get("sources") or input_data.sources or []):
                    s = (src or "").lower()
                    if s and s in lower:
                        available_tools_by_source.setdefault(s, []).append(name)

            # Create tool nodes for available sources
            desired_sources = [s.lower() for s in (plan.get("sources") or input_data.sources or [])]
            for s in desired_sources:
                tools_for_s = available_tools_by_source.get(s, [])
                if not tools_for_s:
                    continue
                    
                tool_name = tools_for_s[0]
                node_id = f"collect_{s}"
                
                # Prepare parameters for the tool
                params = {
                    "keywords": input_data.keywords,
                    "per_query_limit": getattr(input_data, 'per_query_limit', 50),
                    "include_comments": getattr(input_data, 'include_comments', True),
                    "comment_depth": getattr(input_data, 'comment_depth', 2),
                    "comment_limit": getattr(input_data, 'comment_limit', 200),
                    "use_cache": True
                }
                
                # Add subreddits if available from input_data
                subreddits = getattr(input_data, 'subreddits', [])
                self.logger.info(f"DEBUG: input_data subreddits: {subreddits}")
                self.logger.info(f"DEBUG: subreddits type: {type(subreddits)}")
                if subreddits:
                    params["subreddits"] = subreddits
                    self.logger.info(f"DEBUG: Added subreddits to params: {subreddits}")
                else:
                    self.logger.warning(f"DEBUG: No subreddits found or empty list: {subreddits}")
                
                self.logger.info(f"DEBUG: Final params for {tool_name}: {json.dumps(params, indent=2)}")
                
                # Generate MCP tool call code
                code = f'''
# MCP tool call for {tool_name}
import json
from pathlib import Path

# Define parameters as Python dict
params = {repr(params)}

# Call the MCP tool (synchronous wrapper)
result = mcp_call("{tool_name}", params)

# Save result to manifest
save_to_manifest("stages.scout_collect.reddit", result)

print(f"DEBUG: {tool_name} completed, result keys: {{list(result.keys()) if isinstance(result, dict) else 'not dict'}}")
'''.strip()

                # Create tool node compatible with orchestrator expectations
                tool_node = {
                    "id": node_id,
                    "type": "tool",
                    "tool": tool_name,
                    "params": params,
                    "code": code
                }
                self.logger.info(f"DEBUG: Created tool node {node_id} with params: {json.dumps(tool_node['params'], indent=2)}")
                
                tool_nodes.append(tool_node)

            # Return mini-plan with only tool nodes for orchestrator integration
            mini_plan = {
                "dag": {
                    "nodes": tool_nodes
                },
                "metadata": {
                    "target_market": input_data.target_market,
                    "research_scope": input_data.research_scope,
                    "sources": input_data.sources,
                    "keywords": input_data.keywords,
                    "subreddits": getattr(input_data, 'subreddits', [])
                }
            }
            
            return mini_plan
        except Exception as e:
            self.logger.error(f"Error in _postprocess_plan: {e}")
            return {"dag": {"nodes": []}}

    async def _execute_plan_non_agent_nodes(self, plan: Dict[str, Any], *, run_dir_override: Optional[Path] = None) -> Dict[str, Any]:
        """Execute all non-agent DAG nodes (e.g., tool/code) using sandboxed code execution.

        - Reads run_id from plan['dag']['run_id']
        - Executes nodes honoring simple dependency order; runs ready nodes in parallel
        - Each node's `code` is wrapped with a prelude providing mcp_call/save_json helpers
        """
        if not isinstance(plan, dict) or not isinstance(plan.get("dag"), dict):
            return {"completed": [], "failed": []}
        dag = plan["dag"]
        nodes: List[Dict[str, Any]] = list(dag.get("nodes") or [])
        if not nodes:
            return {"completed": [], "failed": []}

        # Compute run directory: override > dag.run_id > plan.run_id > state.run_id
        run_id = dag.get("run_id") or plan.get("run_id") or getattr(self.state, "run_id", None) or "dev_run"
        if run_dir_override is not None:
            run_dir = Path(run_dir_override)
        else:
            project_root = Path(__file__).resolve().parents[2]
            run_dir = (project_root / "data" / "runs" / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Filter non-agent nodes
        target_nodes = [n for n in nodes if (n.get("type") or "").lower() != "agent"]
        if not target_nodes:
            return {"completed": [], "failed": []}

        # Index nodes by id and deps
        node_map = {n.get("id") or f"node_{i}": n for i, n in enumerate(target_nodes)}
        for i, (nid, n) in enumerate(list(node_map.items())):
            n.setdefault("id", nid)
            n.setdefault("deps", n.get("dependencies") or [])

        # Consider agent nodes satisfied ONLY if their declared outputs already exist under run_dir
        def _all_outputs_exist(n: Dict[str, Any]) -> bool:
            outputs = n.get("outputs")
            if not outputs:
                return False
            
            # Handle both list format ["stages.scout_collect.reddit"] and dict format {"location": "stages.scout_collect.reddit"}
            if isinstance(outputs, dict):
                location = outputs.get("location")
                outs = [location] if location else []
            elif isinstance(outputs, list):
                outs = outputs
            else:
                return False
            
            if not outs:
                return False
            try:
                # If outputs are manifest sections like "stages.*", verify presence in manifest
                if all(isinstance(o, str) and o.startswith("stages.") for o in outs):
                    manifest_path = run_dir / "run_manifest.json"
                    if not manifest_path.exists():
                        return False
                    try:
                        manifest = json.loads(manifest_path.read_text())
                    except Exception:
                        return False
                    for o in outs:
                        keys = str(o).split(".")  # e.g., ["stages", "plan"]
                        cur = manifest
                        for k in keys:
                            if not isinstance(cur, dict) or k not in cur:
                                return False
                            cur = cur[k]
                    return True

                # Otherwise treat outputs as files
                for o in outs:
                    path_str = str(o).replace("{run_id}", run_id)
                    p = (run_dir / path_str) if not Path(path_str).is_absolute() else Path(path_str)
                    if not p.exists():
                        return False
                return True
            except Exception:
                return False

        pre_satisfied_agents: set[str] = set()
        for n in nodes:
            if (n.get("type") or "").lower() == "agent" and n.get("id"):
                if _all_outputs_exist(n):
                    pre_satisfied_agents.add(n["id"])  # e.g., plan is done if plan.json exists

        # completed/failed track ONLY target (non-agent) nodes
        completed: set[str] = set()
        failed: set[str] = set()
        # satisfied is used for dependency checks and includes completed target nodes + pre-satisfied agent nodes
        satisfied: set[str] = set(pre_satisfied_agents)

        # Identify the plan/manifest file to update as a running manifest
        plan_file: Optional[Path] = None
        for candidate in (run_dir / "run_manifest.json", run_dir / "scout_plan.json", run_dir / "plan.json"):
            if candidate.exists():
                plan_file = candidate
                break

        def _update_manifest(nid: str, status: str, artifacts: Optional[List[str]] = None, 
                      error: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None,
                      metrics: Optional[Dict[str, Any]] = None):
            """Update manifest with node execution status."""
            try:
                # Resolve manifest dynamically: prefer run_manifest.json if it now exists
                target_manifest = None
                mf = run_dir / "run_manifest.json"
                if mf.exists():
                    target_manifest = mf
                elif plan_file and plan_file.exists():
                    target_manifest = plan_file
                if not target_manifest:
                    return  # No manifest to update yet

                # Create a ManifestManager instance
                manifest_manager = ManifestManager(target_manifest)
                
                # Update the node status
                manifest_manager.update_node_status(
                    node_id=nid,
                    state=status
                )
                
                # Process artifacts if provided
                artifact_list = []
                if artifacts:
                    for artifact in artifacts:
                        if isinstance(artifact, str):
                            # Simple path string
                            try:
                                size = Path(run_dir / artifact).stat().st_size if Path(run_dir / artifact).exists() else 0
                                artifact_list.append({
                                    "path": artifact,
                                    "type": Path(artifact).suffix.lstrip(".") or "txt",
                                    "size_bytes": size
                                })
                            except Exception as e:
                                self.logger.warning(f"Failed to process artifact {artifact}: {e}")
                                artifact_list.append({
                                    "path": artifact,
                                    "type": "unknown",
                                    "size_bytes": 0
                                })
                        elif isinstance(artifact, dict):
                            # Already in the right format
                            artifact_list.append(artifact)
                
                # Store output data/artifacts without overwriting existing stage data
                if data or artifact_list:
                    manifest_manager.store_node_output(
                        node_id=nid,
                        data=(data or {}),
                        artifacts=artifact_list if artifact_list else None
                    )
                
                # Record metrics if provided
                if metrics:
                    manifest_manager.record_metrics(
                        node_id=nid,
                        metrics=metrics
                    )
                
                # Record error if provided
                if error:
                    manifest_manager.record_error(
                        node_id=nid,
                        error_message=error.get("stderr", "Unknown error"),
                        error_type="execution_error"
                    )
                
                # Update overall run status based on node status
                if status == "failed":
                    manifest_manager.update_run_status("failed")
                elif status == "completed":
                    # Check if all nodes are completed
                    manifest = manifest_manager.get_manifest()
                    if "dag" in manifest and "nodes" in manifest["dag"]:
                        all_nodes = manifest["dag"]["nodes"]
                        node_runs = manifest.get("node_runs", {})
                        
                        # Count how many nodes are completed vs total
                        total_nodes = len(all_nodes)
                        completed_nodes = 0
                        
                        for node in all_nodes:
                            node_id = node.get("id")
                            if node_id and node_id in node_runs:
                                node_status = node_runs[node_id].get("status")
                                if node_status == "completed":
                                    completed_nodes += 1
                        
                        # Update progress percentage
                        if total_nodes > 0:
                            progress = int((completed_nodes / total_nodes) * 100)
                            manifest_manager.update_run_metadata({"progress": progress})
                            
                            # If all nodes are completed, mark the run as completed
                            if completed_nodes == total_nodes:
                                manifest_manager.update_run_status("completed")
                elif status == "running":
                    try:
                        rm = manifest_manager.get_manifest().get("run_metadata", {})
                        if rm.get("status") == "initialized":
                            manifest_manager.update_run_status("running")
                    except Exception:
                        # Best effort: mark as running
                        manifest_manager.update_run_status("running")
                
            except Exception as e:
                # Best-effort only; log but don't fail the execution
                self.logger.warning(f"Failed to update manifest for node {nid}: {e}")
                pass

        async def exec_node(n: Dict[str, Any]):
            nid = n["id"]
            lang = (n.get("language") or "python").lower()
            code = (n.get("code") or "").strip()
            start_time = time.time()
            
            if not code:
                self.logger.warning(f"Node {nid} has no code; skipping")
                completed.add(nid)
                return

            # Determine declared outputs
            outs = n.get("outputs") or []
            has_wild = any("*" in str(o) for o in outs)
            preferred_out = None
            for o in outs:
                s = str(o)
                if "*" not in s:
                    preferred_out = s.replace("{run_id}", run_id)
                    break
            preferred_out = preferred_out or "tool_output.json"

            # Build execution prelude with helpers bound to this run_dir
            # Inject project root into sys.path so imports like 'scout_agent.*' work from sandboxed temp file
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            prelude_template = textwrap.dedent(r''' 
import json, os, asyncio
import sys
from pathlib import Path

RUN_DIR = Path(r"__RUN_DIR__")
RUN_DIR.mkdir(parents=True, exist_ok=True)
PROJ_ROOT = Path(r"__PROJ_ROOT__")
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scout_agent.mcp_integration.client.multi import MultiMCPClient
from scout_agent.mcp_integration.config import load_server_configs

def save_json(rel_path: str, obj):
    p = RUN_DIR / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                pass
        json.dump(obj, f, indent=2)

def log_to_file_prelude(message):
    """Log to both console and file from within sandbox prelude"""
    import datetime
    log_dir = Path("/tmp/scout_sandbox_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"prelude_debug_{timestamp}.log"
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().isoformat()}: {message}\n")
        f.flush()

def save_to_manifest(section_key: str, obj):
    """Save data to a specific section in the run manifest."""
    log_to_file_prelude(f"DEBUG: PRELUDE save_to_manifest called with section_key='{section_key}', obj type={type(obj)}")
    log_to_file_prelude(f"DEBUG: PRELUDE TESTING - This should appear in sandbox logs!")
    
    manifest_path = RUN_DIR / "run_manifest.json"
    log_to_file_prelude(f"DEBUG: PRELUDE Manifest path: {manifest_path}")
    log_to_file_prelude(f"DEBUG: PRELUDE Manifest exists: {manifest_path.exists()}")
    
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            log_to_file_prelude(f"DEBUG: Loaded existing manifest with keys: {list(manifest.keys())}")
        else:
            manifest = {}
            log_to_file_prelude(f"DEBUG: Created new manifest")
    except Exception as e:
        log_to_file_prelude(f"ERROR: Failed to load manifest: {e}")
        manifest = {}
    
    # Parse section key like "stages.collect_reddit" -> ["stages", "collect_reddit"]
    keys = section_key.split(".")
    log_to_file_prelude(f"DEBUG: Parsed keys: {keys}")
    
    current = manifest
    for key in keys[:-1]:
        current = current.setdefault(key, {})
        log_to_file_prelude(f"DEBUG: Navigated to key '{key}', current keys: {list(current.keys()) if isinstance(current, dict) else 'not dict'}")
    
    # Set the data
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
            log_to_file_prelude(f"DEBUG: Parsed string obj to dict with keys: {list(obj.keys()) if isinstance(obj, dict) else 'not dict'}")
        except Exception as e:
            log_to_file_prelude(f"DEBUG: Failed to parse string obj as JSON: {e}")
            pass
    
    # Ensure we have valid data to save
    if obj is None:
        log_to_file_prelude(f"WARNING: Attempting to save None object to {section_key}")
        obj = {"warning": "No data was returned from the tool call"}
    
    log_to_file_prelude(f"DEBUG: Final obj type: {type(obj)}")
    if isinstance(obj, dict):
        log_to_file_prelude(f"DEBUG: Final obj keys: {list(obj.keys())}")
        if "threads" in obj:
            log_to_file_prelude(f"DEBUG: Found {len(obj['threads'])} threads in obj")
    
    # For stages, ensure we have the proper structure and save the actual data
    if keys[0] == "stages":
        stage_name = keys[-1]
        log_to_file_prelude(f"DEBUG: Processing stage '{stage_name}'")
        current.setdefault(stage_name, {})
        
        # Ensure we have a proper data structure for Reddit threads
        if stage_name == "scout_collect" or (len(keys) > 2 and keys[1] == "scout_collect" and keys[2] == "reddit"):
            log_to_file_prelude(f"DEBUG: Special handling for scout_collect Reddit data")
            # Special handling for MCP tool responses with nested JSON
            if isinstance(obj, dict):
                # Check if this is an MCP response with content structure
                if "content" in obj and isinstance(obj["content"], list) and len(obj["content"]) > 0:
                    content_item = obj["content"][0]
                    if isinstance(content_item, dict) and "text" in content_item:
                        # Try to parse the nested JSON in the text field
                        try:
                            log_to_file_prelude(f"DEBUG: Found nested JSON in MCP response text field, attempting to parse")
                            nested_data = json.loads(content_item["text"])
                            if isinstance(nested_data, dict):
                                obj = nested_data  # Replace with the parsed nested data
                                log_to_file_prelude(f"DEBUG: Successfully parsed nested JSON with keys: {list(obj.keys())}")
                        except Exception as e:
                            log_to_file_prelude(f"DEBUG: Failed to parse nested JSON in text field: {e}")
                
                # If threads key doesn't exist, initialize it
                if "threads" not in obj:
                    log_to_file_prelude(f"DEBUG: No threads key found, initializing empty list")
                    obj["threads"] = []
                    
                # Add execution details if not present
                if "execution_details" not in obj:
                    thread_count = len(obj.get("threads", []))
                    comment_count = sum(len(t.get("comments", [])) for t in obj.get("threads", []))
                    obj["execution_details"] = {
                        "total_threads_collected": thread_count,
                        "total_comments_collected": comment_count
                    }
                    log_to_file_prelude(f"DEBUG: Added execution details: {thread_count} threads, {comment_count} comments")
                    
                log_to_file_prelude(f"DEBUG: Saving {len(obj.get('threads', []))} threads to manifest")
        
        current[stage_name]["data"] = obj
        current[stage_name]["updated_at"] = __import__("datetime").datetime.now().isoformat()
        current[stage_name]["status"] = "completed"
        log_to_file_prelude(f"DEBUG: Set data for stage '{stage_name}'")
    else:
        current[keys[-1]] = {
            "data": obj,
            "updated_at": __import__("datetime").datetime.now().isoformat()
        }
        log_to_file_prelude(f"DEBUG: Set data for non-stage key '{keys[-1]}'")
    
    # Write the updated manifest
    try:
        manifest_json = json.dumps(manifest, indent=2)
        log_to_file_prelude(f"DEBUG: Generated manifest JSON ({len(manifest_json)} chars)")
        manifest_path.write_text(manifest_json)
        log_to_file_prelude(f"DEBUG: Successfully wrote manifest to {manifest_path}")
        
        # Verify the write
        if manifest_path.exists():
            verify_content = manifest_path.read_text()
            log_to_file_prelude(f"DEBUG: Verified manifest file exists and has {len(verify_content)} chars")
        else:
            log_to_file_prelude(f"ERROR: Manifest file does not exist after write!")
        log_to_file_prelude(f"DEBUG: Manifest saved successfully")
    except Exception as e:
        log_to_file_prelude(f"ERROR: Failed to save manifest: {e}")
        import traceback
        log_to_file_prelude(f"DEBUG: Traceback: {traceback.format_exc()}")

def read_from_manifest(section_key: str):
    """Read data from a specific section in the run manifest."""
    manifest_path = RUN_DIR / "run_manifest.json"
    try:
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text())
        
        # Parse section key like "stages.scout_collect.reddit" -> ["stages", "scout_collect", "reddit"]
        keys = section_key.split(".")
        current = manifest
        for key in keys:
            current = current.get(key, {})
        
        return current.get("data") if isinstance(current, dict) else current
    except Exception:
        return None

    def _ensure_payload(self, res) -> Dict[str, Any]:
        """Ensure the MCP response is converted to a proper dict payload.
        
        Args:
            res: MCP response object
            
        Returns:
            Dict containing the parsed payload
        """
        try:
            self.logger.info(f"MCP Response type: {type(res)}")
            if hasattr(res, "content") and res.content is not None:
                self.logger.info(f"Content type: {type(res.content)}")
                if isinstance(res.content, list) and res.content:
                    content_item = res.content[0]
                    self.logger.info(f"First content item type: {type(content_item)}")
                    
                    # Log the full structure of the content item
                    if hasattr(content_item, "__dict__"):
                        item_dict = content_item.__dict__
                        self.logger.info(f"Content item attributes: {list(item_dict.keys())}")
                        for attr, val in item_dict.items():
                            if isinstance(val, str) and len(val) > 100:
                                self.logger.info(f"  {attr}: {type(val)} (length: {len(val)}) = {val[:100]}...")
                            else:
                                self.logger.info(f"  {attr}: {type(val)} = {val}")
                    
                    if hasattr(content_item, "text"):
                        content = content_item.text
                        self.logger.info(f"Text content type: {type(content)}, length: {len(content) if isinstance(content, str) else 'N/A'}")
                        self.logger.info(f"First 200 chars: {content[:200] if isinstance(content, str) else str(content)[:200]}")
                        try:
                            parsed_json = json.loads(content)
                            self.logger.info(f"Successfully parsed JSON with keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'not a dict'}")
                            return parsed_json
                        except json.JSONDecodeError as e:
                            self.logger.error(f"JSON decode error: {e}")
                            # Return partial content for debugging
                            return {"raw": content[:500], "error": f"JSON decode error: {e}"}
                    else:
                        self.logger.warning("Content item has no text attribute")
                        # Try to convert the content item to a dict if possible
                        if hasattr(content_item, "__dict__"):
                            try:
                                item_dict = content_item.__dict__
                                self.logger.info(f"Converted content item to dict with keys: {list(item_dict.keys())}")
                                return {"content": [item_dict]}
                            except Exception as e:
                                self.logger.error(f"Failed to convert content item to dict: {e}")
                        return {"raw": str(content_item), "error": "No text attribute in content item"}
                else:
                    self.logger.warning("Content is not a list or is empty")
                    # Try to convert the content to a dict if possible
                    if hasattr(res.content, "__dict__"):
                        try:
                            content_dict = res.content.__dict__
                            self.logger.info(f"Converted content to dict with keys: {list(content_dict.keys())}")
                            return {"content": [content_dict]}
                        except Exception as e:
                            self.logger.error(f"Failed to convert content to dict: {e}")
                    return {"raw": str(res.content), "error": "Content is not a list or is empty"}
            else:
                self.logger.warning("Response has no content attribute or content is None")
                # Try to get raw text or other attributes
                if hasattr(res, "text"):
                    try:
                        return json.loads(res.text)
                    except Exception:
                        return {"raw": res.text[:500], "error": "No content attribute, using text"}
                elif hasattr(res, "body"):
                    try:
                        return json.loads(res.body)
                    except Exception:
                        return {"raw": str(res.body)[:500], "error": "No content attribute, using body"}
                else:
                    # Try to convert the response object itself to a dict
                    if hasattr(res, "__dict__"):
                        try:
                            res_dict = res.__dict__
                            self.logger.info(f"Converted response to dict with keys: {list(res_dict.keys())}")
                            return res_dict
                        except Exception as e:
                            self.logger.error(f"Failed to convert response to dict: {e}")
                    return {"raw": str(res)[:500], "error": "No content, text, or body attributes"}
        except Exception as e:
            self.logger.error(f"Exception in _ensure_payload: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "raw": str(res)[:500] if res else "None"}

def mcp_call(tool: str, params: dict):
    def _ensure_payload_local(res):
        """Local version of _ensure_payload for sandboxed execution with double-nested JSON parsing"""
        try:
            if hasattr(res, "content") and res.content is not None:
                if isinstance(res.content, list) and res.content:
                    content_item = res.content[0]
                    if hasattr(content_item, "text"):
                        content = content_item.text
                        try:
                            # First layer: Parse the outer JSON structure
                            first_parse = json.loads(content)
                            print(f"DEBUG: First parse keys: {list(first_parse.keys()) if isinstance(first_parse, dict) else 'not dict'}")
                            
                            # Check if this is the double-nested structure we expect
                            if isinstance(first_parse, dict) and "content" in first_parse:
                                content_list = first_parse["content"]
                                print(f"DEBUG: Found content list with {len(content_list)} items")
                                if isinstance(content_list, list) and content_list:
                                    inner_item = content_list[0]
                                    print(f"DEBUG: Inner item keys: {list(inner_item.keys()) if isinstance(inner_item, dict) else 'not dict'}")
                                    if isinstance(inner_item, dict) and "text" in inner_item:
                                        inner_text = inner_item["text"]
                                        print(f"DEBUG: Found inner text, attempting to parse as JSON")
                                        try:
                                            # Second layer: Parse the nested JSON string
                                            second_parse = json.loads(inner_text)
                                            print(f"DEBUG: Successfully parsed double-nested JSON, found keys: {list(second_parse.keys()) if isinstance(second_parse, dict) else 'not dict'}")
                                            if isinstance(second_parse, dict) and "threads" in second_parse:
                                                print(f"DEBUG: Found {len(second_parse['threads'])} threads in parsed data")
                                            return second_parse
                                        except json.JSONDecodeError as e2:
                                            print(f"DEBUG: Failed to parse inner JSON string: {e2}")
                                            print(f"DEBUG: Inner text sample: {inner_text[:200]}...")
                                            # Return the first parse if second fails
                                            return first_parse
                            
                            # If not double-nested, return the first parse
                            print(f"DEBUG: No double-nesting detected, returning first parse")
                            return first_parse
                            
                        except json.JSONDecodeError as e:
                            print(f"DEBUG: Failed to parse outer JSON: {e}")
                            return {"raw": content[:500], "error": str(e)}
                    else:
                        return {"raw": str(content_item)[:500], "error": "No text attribute"}
                else:
                    return {"raw": str(res.content), "error": "Content is not a list or is empty"}
            else:
                # Try to get raw text or other attributes
                if hasattr(res, "text"):
                    try:
                        return json.loads(res.text)
                    except Exception:
                        return {"raw": res.text[:500], "error": "No content attribute, using text"}
                elif hasattr(res, "body"):
                    try:
                        return json.loads(res.body)
                    except Exception:
                        return {"raw": str(res.body)[:500], "error": "No content attribute, using body"}
                else:
                    # Try to convert the response object itself to a dict
                    if hasattr(res, "__dict__"):
                        try:
                            res_dict = res.__dict__
                            return res_dict
                        except Exception as e:
                            return {"raw": str(res)[:500], "error": f"Failed to convert response to dict: {e}"}
                    return {"raw": str(res)[:500], "error": "No content, text, or body attributes"}
        except Exception as e:
            print(f"DEBUG: Exception in _ensure_payload_local: {e}")
            return {"error": str(e), "raw": str(res)[:500] if res else "None"}
    
    async def _run():
        # Set up dedicated logging for sandboxed execution
        import os
        import datetime
        
        # Create logs directory if it doesn't exist
        log_dir = Path("/tmp/scout_sandbox_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"mcp_call_{tool}_{timestamp}.log"
        
        def log_to_file(message):
            """Log to both console and file"""
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now().isoformat()}: {message}\n")
                f.flush()
        
        log_to_file(f"\n=== MCP CALL START ===")
        log_to_file(f"Tool: {tool}")
        log_to_file(f"Parameters: {json.dumps(params, indent=2)}")
        log_to_file(f"Parameter types: {[(k, type(v).__name__) for k, v in params.items()]}")
        log_to_file(f"Has subreddits: {'subreddits' in params}")
        if 'subreddits' in params:
            log_to_file(f"Subreddits value: {params['subreddits']}")
        log_to_file(f"Log file: {log_file}")
        
        try:
            # Load server configs
            log_to_file("\n--- Loading server configs ---")
            servers = load_server_configs()
            log_to_file(f"Loaded {len(servers)} server configs:")
            for i, server in enumerate(servers):
                log_to_file(f"  Server {i}: {server.get('name', 'unnamed')} - {server.get('command', 'no command')}")
                log_to_file(f"    Full server config: {json.dumps(server, indent=4)}")
            
            # Initialize client
            log_to_file("\n--- Initializing MultiMCPClient ---")
            client = MultiMCPClient(servers)
            log_to_file("Client created, calling initialize()")
            await client.initialize()
            log_to_file("Client initialized successfully")
            
            # List available tools (skip if method doesn't exist)
            try:
                log_to_file("\n--- Available tools ---")
                if hasattr(client, 'list_tools'):
                    tools = await client.list_tools()
                    log_to_file(f"Found {len(tools)} available tools:")
                    for tool_info in tools:
                        log_to_file(f"  - {tool_info.get('name', 'unnamed')}: {tool_info.get('description', 'no description')}")
                        log_to_file(f"    Full tool info: {json.dumps(tool_info, indent=4, default=str)}")
                    
                    # Check if our tool exists
                    tool_names = [t.get('name') for t in tools]
                    if tool not in tool_names:
                        log_to_file(f"WARNING: Tool '{tool}' not found in available tools: {tool_names}")
                    else:
                        log_to_file(f"Tool '{tool}' found in available tools")
                else:
                    log_to_file("Client does not have list_tools method, skipping tool listing")
            except Exception as e:
                log_to_file(f"Error listing tools: {e}")
                import traceback
                log_to_file(f"Tools listing traceback: {traceback.format_exc()}")
            
            try:
                # Make the tool call
                log_to_file(f"\n--- Calling tool {tool} ---")
                log_to_file(f"Calling with params: {json.dumps(params, indent=2)}")
                result = await client.call_tool(tool, params or {})
                log_to_file(f"Tool call completed successfully")
                log_to_file(f"Raw result type: {type(result)}")
                log_to_file(f"Raw result (full): {json.dumps(result, indent=2, default=str)}")
                
                # Process the result
                log_to_file("\n--- Processing result ---")
                processed_result = _ensure_payload_local(result)
                log_to_file(f"Processed result type: {type(processed_result)}")
                log_to_file(f"Processed result (full): {json.dumps(processed_result, indent=2, default=str)}")
                
                if isinstance(processed_result, dict):
                    log_to_file(f"Processed result keys: {list(processed_result.keys())}")
                    for key, value in processed_result.items():
                        if isinstance(value, list):
                            log_to_file(f"  {key}: list with {len(value)} items")
                            if key == 'threads' and len(value) > 0:
                                log_to_file(f"    First thread keys: {list(value[0].keys()) if value[0] else 'empty thread'}")
                                log_to_file(f"    First thread sample: {json.dumps(value[0], indent=4, default=str)}")
                        elif isinstance(value, dict):
                            log_to_file(f"  {key}: dict with keys {list(value.keys())}")
                            log_to_file(f"    {key} content: {json.dumps(value, indent=4, default=str)}")
                        else:
                            log_to_file(f"  {key}: {type(value)} = {str(value)[:100]}..." if len(str(value)) > 100 else f"  {key}: {value}")
                
                # Check if we have threads in the result
                if isinstance(processed_result, dict) and 'threads' in processed_result:
                    thread_count = len(processed_result['threads'])
                    log_to_file(f"\n✓ Found {thread_count} threads in result")
                    if thread_count > 0:
                        log_to_file(f"First thread sample: {json.dumps(processed_result['threads'][0], indent=4, default=str)}")
                else:
                    log_to_file(f"\n⚠ No threads found in result")
                    if isinstance(processed_result, dict):
                        log_to_file(f"Available keys: {list(processed_result.keys())}")
                
                log_to_file(f"\n=== MCP CALL SUCCESS ===")
                return processed_result
                
            except Exception as e:
                log_to_file(f"\n❌ ERROR during tool call: {e}")
                log_to_file(f"Error type: {type(e).__name__}")
                import traceback
                log_to_file(f"Full traceback:\n{traceback.format_exc()}")
                error_result = {"error": f"Tool call error: {e}", "tool": tool, "params": params, "error_type": type(e).__name__}
                log_to_file(f"\n=== MCP CALL FAILED ===")
                return error_result
                
        except Exception as e:
            log_to_file(f"\n❌ ERROR setting up MCP client: {e}")
            log_to_file(f"Error type: {type(e).__name__}")
            import traceback
            log_to_file(f"Full traceback:\n{traceback.format_exc()}")
            error_result = {"error": f"MCP client setup error: {e}", "tool": tool, "params": params, "error_type": type(e).__name__}
            log_to_file(f"\n=== MCP SETUP FAILED ===")
            return error_result
            
        finally:
            try:
                if 'client' in locals():
                    log_to_file("\n--- Shutting down client ---")
                    await client.shutdown()
                    log_to_file("Client shutdown complete")
            except Exception as e:
                log_to_file(f"Error during client shutdown: {e}")
    
    result = asyncio.run(_run())
    print(f"\n=== MCP CALL END ===\n")
    return result
''')
            prelude = (
                prelude_template
                .replace("__RUN_DIR__", str(run_dir))
                .replace("__PROJ_ROOT__", str(PROJECT_ROOT))
            )

            postlude = ""
            if has_wild:
                postlude = f"\n# --- Auto-persist index for wildcard outputs ---\ntry:\n    save_json(\"{preferred_out}\", result)\nexcept Exception as _e:\n    pass\n"

            # Parallelization support: fan-out by list param if specified and tool/params are present
            parallel_by = n.get("parallelize_by")
            tool_name = n.get("tool")
            params = n.get("params") or {}
            fan_items: List[Any] = []
            if parallel_by and tool_name and isinstance(params, dict):
                seq = params.get(parallel_by)
                if isinstance(seq, list) and seq:
                    fan_items = seq

            async def run_code_with_text(code_text: str):
                svc = CodeExecutionService()
                svc.setup_direct(working_dir=str(run_dir))
                return await svc.execute_code(code_text, language=lang, timeout=n.get("timeout") or 60)

            if fan_items:
                _update_manifest(nid, "running")
                # Build and run per-item code using tool/params (ignore provided code to make clean overrides)
                import re
                def slugify(s: str) -> str:
                    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(s)).strip("_") or "item"

                generated_files: List[str] = []
                tasks = []
                for item in fan_items:
                    item_params = dict(params)
                    # Replace the list with a singleton list for this item
                    item_params[parallel_by] = [item]
                    slug = slugify(item)
                    out_name = f"{Path(preferred_out).stem}_{slug}.json"
                    generated_files.append(out_name)
                    per_code = prelude + f"\n# Auto-generated per-item tool call for {parallel_by}={item!r}\n" \
                        + f"result = mcp_call(tool=\"{tool_name}\", params={json.dumps(item_params)}); " \
                        + f"save_json(\"{out_name}\", result)\n"
                    tasks.append(run_code_with_text(per_code))

                await asyncio.gather(*tasks, return_exceptions=False)
                # Write index file with generated file list
                try:
                    (run_dir / preferred_out).write_text(json.dumps({"files": generated_files}, indent=2))
                except Exception:
                    pass
                completed.add(nid)
                
                # Calculate execution duration
                duration_seconds = time.time() - start_time
                
                # Track metrics
                metrics = {
                    "duration_seconds": duration_seconds
                }
                
                _update_manifest(
                    nid, 
                    "completed", 
                    artifacts=generated_files,
                    metrics=metrics
                )
            else:
                full_code = prelude + "\n\n# --- Node code begins ---\n" + code + postlude + "\n# --- Node code ends ---\n"
                self.logger.info(f"Starting node {nid} (language={lang})")
                _update_manifest(nid, "running")
                exec_result = await run_code_with_text(full_code)
                if not getattr(exec_result, "success", False):
                    # Persist error details
                    err_info = {
                        "exec_id": getattr(exec_result, "exec_id", None),
                        "stderr": getattr(exec_result, "error", None),
                        "stdout": getattr(exec_result, "output", None),
                    }
                    (run_dir / f"{nid}_error.json").write_text(json.dumps(err_info, indent=2))
                    failed.add(nid)
                    
                    # Calculate execution duration
                    duration_seconds = time.time() - start_time
                    
                    # Track metrics
                    metrics = {
                        "duration_seconds": duration_seconds
                    }
                    
                    _update_manifest(
                        nid, 
                        "failed", 
                        error=err_info,
                        metrics=metrics
                    )
                    raise RuntimeError(f"Node {nid} failed during execution")
                # On success, record artifacts and log
                try:
                    produced: list[str] = []
                    for o in outs:
                        o_str = str(o).replace("{run_id}", run_id)
                        if "*" in o_str:
                            for p in run_dir.glob(o_str):
                                try:
                                    produced.append(str(p.relative_to(run_dir)))
                                except Exception:
                                    produced.append(str(p))
                        else:
                            p = (run_dir / o_str) if not Path(o_str).is_absolute() else Path(o_str)
                            if p.exists():
                                try:
                                    produced.append(str(p.relative_to(run_dir)))
                                except Exception:
                                    produced.append(str(p))
                    # Write per-node artifact manifest
                    manifest = {"node": nid, "artifacts": sorted(produced)}
                    (run_dir / f"{nid}_artifacts.json").write_text(json.dumps(manifest, indent=2))
                    self.logger.info(f"Completed node {nid}; artifacts: {len(produced)} files")
                    
                    # Calculate execution duration
                    duration_seconds = time.time() - start_time
                    
                    # Track metrics
                    metrics = {
                        "duration_seconds": duration_seconds
                    }
                    
                    # Add LLM metrics if available
                    if hasattr(self, "llm") and hasattr(self.llm, "last_usage"):
                        llm_metrics = {
                            "tokens_used": getattr(self.llm.last_usage, "total_tokens", 0),
                            "cost": getattr(self.llm.last_usage, "cost", 0.0),
                            "backend": getattr(self.llm, "backend_type", None),
                            "model": getattr(self.llm, "model", None)
                        }
                        metrics.update(llm_metrics)
                    
                    _update_manifest(
                        nid, 
                        "completed", 
                        artifacts=manifest["artifacts"],
                        metrics=metrics
                    )
                except Exception as _e:
                    self.logger.warning(f"Node {nid} completed but failed to record artifacts: {_e}")
                completed.add(nid)

        # Simple dependency-driven execution with parallel batches
        while len(completed | failed) < len(node_map):
            ready = [
                nid for nid, n in node_map.items()
                if nid not in completed and nid not in failed
                and all(d in satisfied for d in (n.get("deps") or []))
            ]
            if not ready:
                break
            # Execute ready nodes concurrently
            batch = [exec_node(node_map[nid]) for nid in ready]
            results = await asyncio.gather(*batch, return_exceptions=True)
            # Log exceptions
            for nid, res in zip(ready, results):
                if isinstance(res, Exception):
                    self.logger.error(f"Execution error in node {nid}: {res}")
                    failed.add(nid)
                else:
                    # Mark node as satisfied for downstream deps
                    satisfied.add(nid)

        summary = {"completed": sorted(list(completed)), "failed": sorted(list(failed))}
        if failed:
            self.logger.error(f"Non-agent stage had failures: {summary['failed']}")
        return summary

    async def collect(self, *, plan_path: Optional[str] = None, plan: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Public entry to run the collect stage (non-agent DAG nodes).

        Args:
            plan_path: Optional filesystem path to plan.json
            plan: Optional plan dict (if already loaded)

        Returns: summary dict with completed and failed node ids.
        """
        try:
            selected_run_dir: Optional[Path] = None
            if plan is None:
                if not plan_path:
                    raise ValueError("Either plan or plan_path must be provided")
                p = Path(plan_path)
                if not p.exists():
                    raise FileNotFoundError(f"Plan not found: {p}")
                plan = json.loads(p.read_text())
                selected_run_dir = p.parent
            # Determine run_id preference: explicit arg > state > plan dag/run_id 
            # This ensures orchestrator's run_id takes precedence
            dag = plan.get("dag") or {}
            state_run_id = getattr(self.state, "run_id", None)
            chosen_run_id = run_id or state_run_id or dag.get("run_id") or plan.get("run_id")
            if chosen_run_id:
                dag["run_id"] = chosen_run_id
                plan["dag"] = dag
                plan["run_id"] = chosen_run_id
                setattr(self.state, "run_id", chosen_run_id)

            # Prefer using the directory of plan_path if provided, to keep artifacts together for testing
            return await self._execute_plan_non_agent_nodes(plan, run_dir_override=selected_run_dir)
        except Exception as e:
            self.logger.error(f"Collect stage error: {e}\n{traceback.format_exc()}")
            return {"completed": [], "failed": [str(e)]}
    
    def _extract_pain_points(self, research_results: Dict[str, Any], market: str) -> List[PainPoint]:
        """Extract pain points from research data."""
        pain_points = []
        
        # Extract from actual research data - no mock data
        if not research_results:
            return pain_points
            
        # Process real Reddit threads and other sources
        threads = research_results.get("threads", [])
        for thread in threads:
            post = thread.get("post", {})
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            subreddit = post.get("subreddit", "")
            
            # Simple keyword-based pain point extraction
            pain_indicators = ["problem", "issue", "bug", "error", "fail", "broken", "difficult", "hard", "annoying", "frustrating"]
            text = f"{title} {selftext}".lower()
            
            if any(indicator in text for indicator in pain_indicators):
                # Extract a meaningful description from the title
                description = title if title else "Unspecified issue"
                
                # Determine severity based on keywords
                severity = "high" if any(word in text for word in ["critical", "urgent", "blocking", "broken"]) else "medium"
                
                # Calculate basic impact score
                impact_score = 7.0 + (post.get("score", 0) / 100.0)  # Base 7.0 + Reddit score influence
                impact_score = min(impact_score, 10.0)  # Cap at 10.0
                
                pain_point = PainPoint(
                    description=description,
                    severity=severity,
                    market=market,
                    source="reddit",
                    evidence=[f"Reddit post: {post.get('permalink', '')}"],
                    frequency=1,  # Each thread counts as 1 occurrence
                    impact_score=impact_score,
                    tags=[subreddit] if subreddit else [],
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
    
    def _write_stage_output(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Write stage output to manifest section using ManifestManager."""
        try:
            run_id = getattr(self.state, "run_id", "unknown")
            from pathlib import Path
            
            # Determine run directory
            # project_root should be ScoutAgent/ (not scout_agent/)
            project_root = Path(__file__).resolve().parents[2]
            run_dir = project_root / "data" / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "run_manifest.json"
            
            self.logger.info(f"Writing stage {stage_name} output to manifest at: {manifest_path}")
            
            # Use ManifestManager for consistent manifest operations
            from scout_agent.memory.manifest_manager import ManifestManager
            manifest_manager = ManifestManager(manifest_path, create_if_missing=True)
            
            # Store the stage output
            manifest_manager.store_node_output(stage_name, data)
            
            # Update node status to completed
            manifest_manager.update_node_status(
                node_id=stage_name,
                state="completed"
            )
            
            self.logger.info(f"Stage {stage_name} output written to manifest successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to write stage {stage_name} output: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

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
