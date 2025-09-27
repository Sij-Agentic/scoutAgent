"""
ScoutAgent - Pain Point Discovery Agent

This agent specializes in discovering pain points in target markets using
web research, social media analysis, and user feedback collection.
"""

import asyncio
import json
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

            # NEW APPROACH: Generate metadata only, then construct DAG programmatically
            # Phase 1: LLM generates focused metadata (much smaller, no truncation)
            metadata_prompt = load_prompt_template(template_name="plan_metadata.prompt", agent_name=self.name, substitutions=substitutions)
            
            self.logger.info("Phase 1: Generating research metadata with LLM")
            llm_text = await self.llm_generate(prompt=metadata_prompt, task_type="plan", max_tokens=1000)
            metadata = self._extract_json(llm_text)
            
            # Fallback if metadata extraction fails
            if not isinstance(metadata, dict):
                self.logger.warning("Metadata extraction failed, using fallback metadata")
                metadata = {
                    "enhanced_keywords": input_data.keywords or ["issue", "problem", "frustration"],
                    "optimized_subreddits": getattr(input_data, 'subreddits', []) or ["programming", "webdev"],
                    "research_strategy": {
                        "primary_focus": "pain point discovery",
                        "data_sources": input_data.sources or ["reddit"],
                        "collection_priority": "threads_with_comments"
                    },
                    "metadata": {
                        "target_market": input_data.target_market,
                        "research_scope": input_data.research_scope,
                        "max_pain_points": input_data.max_pain_points
                    }
                }
            
            # Phase 2: Programmatically construct DAG from metadata
            self.logger.info("Phase 2: Constructing DAG programmatically from metadata")
            from scout_agent.templates.dag_node_templates import create_dag_from_metadata
            
            available_sources = input_data.sources or ["reddit"]
            plan = create_dag_from_metadata(metadata, available_sources)
            
            # Debug logging
            self.logger.info(f"DEBUG: _extract_json returned type: {type(plan)}")
            if isinstance(plan, dict):
                self.logger.info(f"DEBUG: Plan keys: {list(plan.keys())}")
                # Check if DAG structure is valid
                dag = plan.get("dag", {})
                if isinstance(dag, dict):
                    nodes = dag.get("nodes", [])
                    self.logger.info(f"DEBUG: DAG has {len(nodes)} nodes")
                    if nodes:
                        self.logger.info(f"DEBUG: First node keys: {list(nodes[0].keys()) if nodes[0] else 'empty'}")
                else:
                    self.logger.error(f"DEBUG: DAG is not a dict, type: {type(dag)}")
            else:
                self.logger.error(f"DEBUG: Plan content (first 200 chars): {str(plan)[:200]}")
            
            # Debug: Log raw LLM response length and save for analysis
            self.logger.info(f"DEBUG: Raw LLM response length: {len(llm_text) if llm_text else 0}")
            
            # Debug: Save raw LLM response for debugging
            try:
                debug_dir = Path("debug")
                debug_dir.mkdir(exist_ok=True)
                debug_file = debug_dir / f"scout_plan_llm_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"LLM Response Length: {len(llm_text)}\n")
                    f.write("="*50 + "\n")
                    f.write(llm_text or "NO RESPONSE")
                self.logger.info(f"DEBUG: Saved raw LLM response to {debug_file}")
            except Exception as debug_err:
                self.logger.warning(f"Failed to save debug LLM response: {debug_err}")
            
            # Safety check - ensure plan is a dictionary
            if not isinstance(plan, dict):
                self.logger.error(f"LLM response parsing failed, plan type: {type(plan)}")
                plan = {
                    "dag": {"nodes": []},
                    "metadata": {
                        "target_market": input_data.target_market,
                        "research_scope": input_data.research_scope,
                        "sources": input_data.sources,
                        "keywords": input_data.keywords
                    },
                    "error": "Failed to parse LLM response as JSON"
                }
            
            # Post-process to preserve LLM-generated code
            plan = self._postprocess_plan(plan, input_data, tools_catalog, tool_names)
            self.logger.info(f"Generated plan with keys: {list(plan.keys())}")
        except Exception as e:
            self.logger.error(f"Error in plan phase: {str(e)}\n{traceback.format_exc()}")
            # No fallback - just raise the exception to indicate failure
            raise ValueError(f"Failed to generate plan: {str(e)}")
        
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
            # Persist core run metadata and seed stages.scout_plan
            manifest["run_id"] = plan.get("run_id")
            manifest["dag"] = plan.get("dag") or {}
            stages = manifest.setdefault("stages", {})
            
            # Clean up the plan data to fix JSON string issues
            try:
                cleaned_plan = self._clean_plan_data(plan)
                self.logger.debug("Successfully cleaned plan data")
            except Exception as clean_err:
                self.logger.warning(f"Plan cleaning failed: {clean_err}, using original plan")
                cleaned_plan = plan
            
            stages["scout_plan"] = {
                "data": cleaned_plan,
                "status": "completed",
                "updated_at": datetime.now().isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            self.logger.info(f"Initialized manifest at {manifest_path}")
        except Exception as _e:
            # Non-fatal; downstream will create/append as needed
            self.logger.warning(f"Failed to initialize run_manifest.json: {_e}")

        return plan
    
    def _clean_plan_data(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Clean plan data to fix JSON string issues and prevent double encoding."""
        import json
        import re
        
        def clean_value(value):
            """Recursively clean values that might be JSON strings."""
            if isinstance(value, str):
                # Be more conservative - only try to parse if it's obviously JSON
                stripped = value.strip()
                
                # Only parse if it looks like complete JSON and has matching brackets
                if ((stripped.startswith('[') and stripped.endswith(']')) or \
                   (stripped.startswith('{') and stripped.endswith('}'))):
                    
                    # Additional validation - check if brackets are balanced
                    if stripped.startswith('['):
                        open_count = stripped.count('[')
                        close_count = stripped.count(']')
                    else:
                        open_count = stripped.count('{')
                        close_count = stripped.count('}')
                    
                    if open_count == close_count and len(stripped) > 2:
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(stripped)
                            return parsed
                        except json.JSONDecodeError:
                            # If parsing fails, return the original string
                            return value
                
                return value
            elif isinstance(value, dict):
                # Recursively clean dictionary values
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                # Recursively clean list values
                return [clean_value(item) for item in value]
            else:
                return value
        
        return clean_value(plan)
    
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
            
            # Initialize variables
            threads = []
            comments = []
            
            # Extract threads and comments from the collected data
            # First, try to parse JSON strings from content field (MCP tool format)
            if "sources" in reddit_data and "reddit" in reddit_data["sources"]:
                reddit_source = reddit_data["sources"]["reddit"]
                if "content" in reddit_source and isinstance(reddit_source["content"], list):
                    self.logger.info("Found Reddit data in sources.reddit.content format")
                    for content_item in reddit_source["content"]:
                        if isinstance(content_item, dict) and "text" in content_item:
                            try:
                                # Parse the JSON string from the text field
                                parsed_data = json.loads(content_item["text"])
                                if "threads" in parsed_data:
                                    threads = parsed_data["threads"]
                                    self.logger.info(f"Parsed {len(threads)} threads from JSON content")
                                if "comments" in parsed_data:
                                    comments = parsed_data["comments"]
                                    self.logger.info(f"Parsed {len(comments)} comments from JSON content")
                                break
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse JSON from content: {e}")
                                continue
            
            # Check if data is nested under 'data' key (manifest structure)
            if not threads and "data" in reddit_data and isinstance(reddit_data["data"], dict):
                self.logger.info("Found nested 'data' key in Reddit data")
                data_content = reddit_data["data"]
                threads = data_content.get("threads", [])
                comments = data_content.get("comments", [])
                self.logger.info(f"Data content keys: {list(data_content.keys())}")
            elif not threads:
                # Direct access (fallback)
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
            
            # Add the Reddit data to the prompt using a chunked approach to avoid timeouts
            # Limit the data to avoid exceeding token limits
            # TEMPORARY: Moderate cost limiting for testing
            max_threads = 3  # MODERATE LIMITING: Increase from 2 to 3 threads per batch
            max_comments_per_thread = 8  # MODERATE LIMITING: Increase from 5 to 8 comments
            max_threads_total = min(6, len(threads))  # MODERATE LIMITING: Increase from 3 to 6 threads total
            
            # Prepare threads for processing
            all_threads = threads[:max_threads_total]
            
            # Add comments to their respective threads
            thread_comments = {}
            for comment in comments:
                thread_id = comment.get("link_id", "").replace("t3_", "")
                if thread_id not in thread_comments:
                    thread_comments[thread_id] = []
                if len(thread_comments[thread_id]) < max_comments_per_thread:
                    thread_comments[thread_id].append(comment)
            
            # Process threads in batches to avoid LLM timeout (in parallel)
            batch_count = (len(all_threads) + max_threads - 1) // max_threads  # Ceiling division
            
            self.logger.info(f"Processing {len(all_threads)} threads in {batch_count} batches in parallel")
            
            # Prepare all batches
            batch_tasks = []
            
            async def process_batch(batch_idx, batch_threads):
                try:
                    # Add comments to threads in this batch
                    for thread in batch_threads:
                        thread_id = thread.get("id", "").replace("t3_", "")
                        thread["comments"] = thread_comments.get(thread_id, [])
                    
                    # Create a batch-specific prompt
                    batch_prompt = prompt_text + f"\n\n# REDDIT DATA (Batch {batch_idx+1}/{batch_count})\n{json.dumps(batch_threads, indent=2)}"
                    
                    self.logger.info(f"Processing batch {batch_idx+1}/{batch_count} with {len(batch_threads)} threads")
                    llm_text = await self.llm_generate(prompt=batch_prompt, task_type="think")
                    batch_analysis = self._extract_json(llm_text)
                    
                    if batch_analysis:
                        self.logger.info(f"Successfully analyzed batch {batch_idx+1}")
                        return batch_analysis
                    else:
                        self.logger.warning(f"Failed to extract JSON from batch {batch_idx+1} LLM response")
                        return None
                except Exception as e:
                    self.logger.warning(f"Batch {batch_idx+1} analysis failed: {str(e)}")
                    return None
            
            # Create tasks for all batches
            for batch_idx in range(batch_count):
                start_idx = batch_idx * max_threads
                end_idx = min(start_idx + max_threads, len(all_threads))
                batch_threads = all_threads[start_idx:end_idx]
                batch_tasks.append(process_batch(batch_idx, batch_threads))
            
            # Run all batches in parallel
            analysis_results = await asyncio.gather(*batch_tasks)
            # Filter out None results
            analysis_results = [result for result in analysis_results if result]
            
            # Merge the analysis results from all batches
            if analysis_results:
                # Combine per-thread summaries with deduplication
                per_thread_summaries = []
                seen_thread_ids = set()
                
                for result in analysis_results:
                    if isinstance(result, dict) and "per_thread_summaries" in result:
                        summaries = result.get("per_thread_summaries", [])
                        for summary in summaries:
                            # Check if this is a dict with an id or thread_id field
                            thread_id = None
                            if isinstance(summary, dict):
                                thread_id = summary.get("id") or summary.get("thread_id")
                            
                            # Add if no ID (can't deduplicate) or if ID not seen before
                            if not thread_id or thread_id not in seen_thread_ids:
                                per_thread_summaries.append(summary)
                                if thread_id:
                                    seen_thread_ids.add(thread_id)
                
                # Combine pain points with deduplication by description
                all_pains = []
                seen_pain_descriptions = set()
                
                for result in analysis_results:
                    if isinstance(result, dict) and "pains" in result:
                        pains = result.get("pains", [])
                        for pain in pains:
                            # Get a description to use for deduplication
                            pain_desc = None
                            if isinstance(pain, dict):
                                pain_desc = pain.get("description")
                            elif isinstance(pain, str):
                                pain_desc = pain
                                
                            # Add if no description (can't deduplicate) or if not seen before
                            if not pain_desc or pain_desc not in seen_pain_descriptions:
                                all_pains.append(pain)
                                if pain_desc:
                                    seen_pain_descriptions.add(pain_desc)
                
                # Combine themes (ensure they're hashable strings)
                all_themes = set()
                for result in analysis_results:
                    if isinstance(result, dict) and "themes" in result:
                        themes = result.get("themes", [])
                        # Ensure each theme is a hashable type (string)
                        for theme in themes:
                            if isinstance(theme, str):
                                all_themes.add(theme)
                            elif isinstance(theme, dict) and "name" in theme:
                                # If themes are objects with a name field
                                all_themes.add(theme["name"])
                            elif isinstance(theme, (list, tuple)) and len(theme) > 0:
                                # If themes are lists/tuples, use the first element if it's a string
                                if isinstance(theme[0], str):
                                    all_themes.add(theme[0])
                
                # Create the final analysis with enhanced metadata
                analysis = {
                    "per_thread_summaries": per_thread_summaries,
                    "pains": all_pains,
                    "themes": list(all_themes),
                    "total_threads_analyzed": len(all_threads),
                    "total_threads_summarized": len(per_thread_summaries),
                    "total_pain_points_found": len(all_pains),
                    "total_themes_identified": len(all_themes),
                    "batches_processed": len(analysis_results),
                    "total_batches": batch_count,
                    "processing_method": "parallel_batched",
                    "processing_timestamp": datetime.now().isoformat()
                }
                self.logger.info(f"Successfully merged analysis from {len(analysis_results)} batches")
            else:
                # Fallback if all LLM calls fail
                self.logger.warning("All LLM analysis attempts failed. Using fallback analysis.")
                analysis = {
                    "per_thread_summaries": [],
                    "pains": [],
                    "themes": [],
                    "error": "LLM analysis failed for all batches",
                    "total_threads_analyzed": len(threads),
                    "total_threads_summarized": 0,
                    "total_pain_points_found": 0,
                    "total_themes_identified": 0,
                    "batches_processed": 0,
                    "total_batches": batch_count,
                    "processing_method": "parallel_batched_failed",
                    "processing_timestamp": datetime.now().isoformat(),
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
        
        # Default fallback structure for plan generation
        fallback = {
            "dag": {
                "nodes": []
            },
            "metadata": {
                "target_market": "unknown",
                "research_scope": "focused",
                "sources": [],
                "keywords": []
            },
            "error": "Failed to parse JSON"
        }
        
        # Ensure fallback is always a dictionary
        if not isinstance(fallback, dict):
            self.logger.error(f"Fallback is not a dictionary: {type(fallback)}")
            fallback = {
                "dag": {"nodes": []},
                "metadata": {"target_market": "unknown", "research_scope": "focused", "sources": [], "keywords": []},
                "error": "Failed to parse JSON"
            }
        
        if not text:
            self.logger.error("Empty text provided to _extract_json")
            return fallback
            
        # First try to extract content from code blocks
        content = ""
        try:
            # Extract from markdown code blocks with priority
            if "```json" in text:
                # Extract content between ```json and ``` - find proper pair
                start_marker = text.find("```json")
                if start_marker != -1:
                    # Start after ```json and optional newline
                    start_content = start_marker + 7  # len("```json")
                    if text[start_content:start_content+1] == "\n":
                        start_content += 1
                    
                    # Find the closing ```
                    end_marker = text.find("```", start_content)
                    if end_marker != -1:
                        content = text[start_content:end_marker].strip()
                    else:
                        content = text[start_content:].strip()
            elif "```" in text:
                # Extract content between ``` and ``` - find proper pair
                start_marker = text.find("```")
                if start_marker != -1:
                    # Find the first newline after ```
                    start_content = text.find("\n", start_marker)
                    if start_content == -1:
                        start_content = start_marker + 3
                    else:
                        start_content += 1
                    
                    # Find the closing ```
                    end_marker = text.find("```", start_content)
                    if end_marker != -1:
                        content = text[start_content:end_marker].strip()
                    else:
                        content = text[start_content:].strip()
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
            
            # CONSERVATIVE JSON fixes - only apply if JSON parsing fails
            import re
            
            # First, try parsing without any fixes
            try:
                test_parse = json.loads(content)
                # If parsing succeeds, don't apply any fixes!
                self.logger.debug("JSON is already valid, skipping fixes")
            except json.JSONDecodeError:
                # Only apply fixes if JSON parsing fails
                self.logger.debug("JSON parsing failed, applying conservative fixes")
                
                # Only fix obvious escape sequence issues, nothing else
                def conservative_json_fixes(text):
                    # Only fix clearly broken escape patterns
                    # Pattern: "word\\" -> "word" (only if it's clearly wrong)
                    text = re.sub(r'"([^"]*?)\\+"(\s*[,\]}])', r'"\1"\2', text)
                    
                    # Fix standalone escaped quotes that are clearly wrong
                    text = re.sub(r'(?<!\\)\\"(?![\\"])', '"', text)
                    
                    return text
                
                content = conservative_json_fixes(content)
            
            self.logger.debug("Applied JSON escape sequence fixes")
            
            # Check if content is empty or not JSON-like
            if not content or not content.strip():
                self.logger.warning("Empty content after extraction")
                return fallback
            
            # Check if content looks like JSON (starts with { or [)
            if not (content.strip().startswith(('{', '['))):
                self.logger.warning(f"Content doesn't look like JSON: {content[:100]}...")
                return fallback
            
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
        
        # Final safety check - ensure we always return a dictionary
        if not isinstance(fallback, dict):
            self.logger.error(f"_extract_json returned non-dict type: {type(fallback)}")
            return {
                "dag": {"nodes": []},
                "metadata": {"target_market": "unknown", "research_scope": "focused", "sources": [], "keywords": []},
                "error": "JSON extraction failed"
            }
        
        return fallback

    def _postprocess_plan(self, plan: Dict[str, Any], input_data: "ScoutInput", tools_catalog: List[Dict[str, Any]], tool_names: List[str]) -> Dict[str, Any]:
        """Process the LLM-generated plan, preserving tool nodes and code.
        
        This method ensures the plan has the expected structure but preserves
        the tool nodes and code generated by the LLM.
        """
        try:
            if not isinstance(plan, dict):
                self.logger.warning(f"Plan is not a dictionary (type: {type(plan)}), returning empty plan")
                return {"dag": {"nodes": []}}
            
            # If "dag" exists but is not a dict, attempt to coerce into a proper DAG
            dag = plan.get("dag")
            if dag is not None and not isinstance(dag, dict):
                self.logger.warning(f"Plan has 'dag' of type {type(dag)}; coercing to empty dag with nodes[]")
                plan["dag"] = {"nodes": []}

            # If plan appears to be a single node (flat fields), wrap it into dag.nodes
            candidate_node_keys = {"id", "type", "tool", "params", "code", "language", "outputs"}
            if ("dag" not in plan or not isinstance(plan.get("dag"), dict) or not isinstance(plan.get("dag", {}).get("nodes", []), list)) and any(k in plan for k in candidate_node_keys):
                node: Dict[str, Any] = {k: plan[k] for k in candidate_node_keys if k in plan}
                plan = {
                    "dag": {
                        "nodes": [node]
                    },
                    "metadata": plan.get("metadata", {
                        "target_market": input_data.target_market,
                        "research_scope": input_data.research_scope,
                        "sources": input_data.sources,
                        "keywords": input_data.keywords,
                        "subreddits": getattr(input_data, 'subreddits', [])
                    })
                }
                self.logger.info("Wrapped flat plan into dag.nodes with a single node")
            
            # Check if the plan already has a DAG with nodes
            dag = plan.get("dag", {})
            if not isinstance(dag, dict):
                dag = {"nodes": []}
                plan["dag"] = dag
            nodes = dag.get("nodes", [])
            if not isinstance(nodes, list):
                self.logger.warning(f"Plan dag.nodes is not a list (type: {type(nodes)}); resetting to []")
                nodes = []
                plan["dag"]["nodes"] = nodes
            
            if not nodes:
                self.logger.warning("No nodes found in LLM-generated plan")
            else:
                self.logger.info(f"Found {len(nodes)} nodes in LLM-generated plan")
                for i, node in enumerate(nodes):
                    if not isinstance(node, dict):
                        self.logger.warning(f"Node {i} is not a dict (type: {type(node)}); skipping")
                        continue
                    node_id = node.get("id", f"node_{i}")
                    node_type = node.get("type", "unknown")
                    node_tool = node.get("tool", "none")
                    has_code = "code" in node
                    self.logger.info(f"Node {i}: id={node_id}, type={node_type}, tool={node_tool}, has_code={has_code}")
            
            # Ensure the plan has the expected structure
            if "metadata" not in plan:
                plan["metadata"] = {
                    "target_market": input_data.target_market,
                    "research_scope": input_data.research_scope,
                    "sources": input_data.sources,
                    "keywords": input_data.keywords,
                    "subreddits": getattr(input_data, 'subreddits', [])
                }
            
            # Return the plan with minimal modifications
            return plan
        except Exception as e:
            self.logger.error(f"Error in _postprocess_plan: {e}")
            return {"dag": {"nodes": []}}


    async def collect(self, *, plan_path: Optional[str] = None, plan: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Public entry to run the collect stage using AgentCodeExecutor service.

        Args:
            plan_path: Optional filesystem path to plan.json
            plan: Optional plan dict (if already loaded)
            run_id: Optional run ID to use

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
    
    async def _execute_plan_non_agent_nodes(self, plan: Dict[str, Any], run_dir_override: Optional[Path] = None):
        """Execute tool nodes from the DAG plan by directly calling MCP tools."""
        self.logger.info("Executing DAG tool nodes via direct MCP calls...")
        
        # Import MCP client for direct tool calls
        from scout_agent.mcp_integration.client.multi import MultiMCPClient
        from scout_agent.mcp_integration.config import load_server_configs
        
        # Get DAG nodes from plan
        self.logger.info(f"Plan structure keys: {list(plan.keys())}")
        
        # Try different possible locations for DAG nodes
        dag_nodes = []
        if "stages" in plan and "scout_plan" in plan["stages"]:
            dag_nodes = plan.get("stages", {}).get("scout_plan", {}).get("data", {}).get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in stages.scout_plan.data.dag.nodes")
        elif "dag_metadata" in plan:
            dag_nodes = plan.get("dag_metadata", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag_metadata.nodes")
        elif "dag" in plan:
            dag_nodes = plan.get("dag", {}).get("nodes", [])
            self.logger.info(f"Found {len(dag_nodes)} nodes in dag.nodes")
        
        # Filter tool nodes (loosened): accept nodes that declare a tool or contain code
        tool_nodes = [
            node for node in dag_nodes
            if ("tool_name" in node) or ("tool" in node) or ("code" in node)
        ]
        self.logger.info(f"Found {len(tool_nodes)} tool nodes to execute")
        
        if not tool_nodes:
            self.logger.warning("No tool nodes found in plan")
            return {"completed": [], "failed": []}
        
        # Initialize MCP client
        server_configs = load_server_configs()
        multi_client = MultiMCPClient(server_configs)
        try:
            await multi_client.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            return {"completed": [], "failed": ["MCP client initialization failed"]}
        
        # Execute each tool node
        manifest_manager = ManifestManager(run_dir_override or "default")
        aggregated_sources: Dict[str, Any] = {}
        
        for i, node in enumerate(tool_nodes):
            try:
                node_id = node.get("node_id") or node.get("id") or f"tool_node_{i}_{node.get('tool', node.get('tool_name', 'unknown'))}"
                tool_name = node.get("tool_name") or node.get("tool")
                inputs = node.get("inputs") or node.get("params") or {}
                
                self.logger.info(f"Executing tool node: {node_id} with tool: {tool_name}")
                
                # Resolve template variables in inputs before calling MCP tool
                resolved_inputs = self._resolve_template_variables(inputs, manifest_manager)
                
                # Make direct MCP tool call
                result = await multi_client.call_tool(tool_name, resolved_inputs)
                
                # Process and store result
                if result:
                    # Extract content from MCP response
                    if hasattr(result, 'content') and result.content:
                        content_item = result.content[0] if isinstance(result.content, list) else result.content
                        if hasattr(content_item, 'text'):
                            try:
                                # Try to parse as JSON
                                result_data = json.loads(content_item.text)
                            except json.JSONDecodeError:
                                # If not JSON, store as text
                                result_data = {"output": content_item.text}
                        else:
                            result_data = {"raw_result": str(content_item)}
                    else:
                        result_data = {"raw_result": str(result)}
                    
                    # Determine outputs and store
                    declared_outputs = node.get("outputs") or []
                    if declared_outputs:
                        for out_key in declared_outputs:
                            # Standardize scout collect outputs: map stages.research.* → stages.scout_collect.*
                            if out_key.startswith("stages.research."):
                                out_key = out_key.replace("stages.research.", "stages.scout_collect.")
                            # Also aggregate under sources for the collector summary
                            try:
                                suffix = out_key.split("stages.scout_collect.", 1)[1]
                                aggregated_sources[suffix] = result_data
                            except Exception:
                                pass
                            # Best-effort: store as a stage node output entry for downstream template resolution
                            try:
                                manifest_manager.store_node_output(out_key, result_data)
                            except Exception as _e:
                                self.logger.debug(f"Unable to store output by path {out_key}: {_e}")
                    else:
                        # Fallback: store by node id
                        output_key = node.get("output_manifest_key", f"{node_id}_output")
                        manifest_manager.store_node_output(output_key, result_data)
                    
                    self.logger.info(f"Successfully executed and stored results for node: {node_id}")
                else:
                    self.logger.error(f"Tool node {node_id} returned no result")
                    
            except Exception as e:
                self.logger.error(f"Error executing tool node {node.get('node_id', 'unknown')}: {str(e)}")
                # Continue with other nodes even if one fails
                continue
        
        # Build normalized summary with aggregated sources for downstream stages
        summary_completed = [
            (n.get("node_id") or n.get("id") or f"node_{i}") for i, n in enumerate(tool_nodes)
        ]
        collect_summary = {
            "completed": summary_completed,
            "failed": [],
            "sources": aggregated_sources
        }
        
        # Write the collect summary to manifest under stages.scout_collect.data
        try:
            manifest = manifest_manager.get_manifest() or {}
            stages = manifest.get("stages", {})
            stages.setdefault("scout_collect", {})["data"] = collect_summary
            manifest["stages"] = stages
            manifest_manager._write_manifest(manifest)  # use internal write to persist the staged update
        except Exception as _e:
            self.logger.debug(f"Unable to persist collect summary to manifest: {_e}")
        
        return collect_summary
    
    def _resolve_template_variables(self, inputs: Dict[str, Any], manifest_manager: ManifestManager) -> Dict[str, Any]:
        """Resolve ${...} template variables in tool inputs using manifest data."""
        import re
        
        resolved_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and "${" in value:
                # Find all template variables in the format ${variable_name}
                template_pattern = r'\$\{([^}]+)\}'
                matches = re.findall(template_pattern, value)
                
                # Check if the entire value is a single template variable
                if len(matches) == 1 and value.strip() == f"${{{matches[0]}}}":
                    # Direct replacement - preserve original data type
                    match = matches[0]
                    try:
                        # Parse the template variable (e.g., "triage_content_pp1_output.contents")
                        if '.' in match:
                            node_key, field_path = match.split('.', 1)
                        else:
                            node_key, field_path = match, None
                        
                        # First try to get the node output using the key directly
                        node_output = manifest_manager.get_node_output(node_key)
                        
                        # If not found, try to find the actual node_id that maps to this output key
                        if not node_output:
                            # Look through all stages to find a node that has this output_manifest_key
                            manifest = manifest_manager.get_manifest()
                            for stage_key, stage_data in manifest.get("stages", {}).items():
                                # Check if this stage was stored with the output key we're looking for
                                if stage_key == node_key and "data" in stage_data:
                                    node_output = stage_data["data"]
                                    break
                        
                        if node_output:
                            # Navigate to the specific field if specified
                            if field_path:
                                field_value = self._get_nested_field(node_output, field_path)
                            else:
                                field_value = node_output
                            
                            if field_value is not None:
                                # Direct assignment preserves data type (list, dict, etc.)
                                resolved_inputs[key] = field_value
                                self.logger.info(f"Resolved template variable ${{{match}}} to {type(field_value).__name__} with {len(field_value) if isinstance(field_value, (list, dict)) else 'N/A'} items")
                            else:
                                self.logger.warning(f"Template variable ${{{match}}} resolved to None")
                                resolved_inputs[key] = value  # Keep original if resolution fails
                        else:
                            self.logger.warning(f"No data found for template variable ${{{match}}}")
                            resolved_inputs[key] = value  # Keep original if resolution fails
                    except Exception as e:
                        self.logger.error(f"Error resolving template variable ${{{match}}}: {e}")
                        resolved_inputs[key] = value  # Keep original on error
                else:
                    # Multiple template variables or mixed content - string replacement
                    resolved_value = value
                    for match in matches:
                        try:
                            if '.' in match:
                                node_key, field_path = match.split('.', 1)
                            else:
                                node_key, field_path = match, None
                            
                            node_output = manifest_manager.get_node_output(node_key)
                            if node_output:
                                if field_path:
                                    field_value = self._get_nested_field(node_output, field_path)
                                else:
                                    field_value = node_output
                                
                                if field_value is not None:
                                    resolved_value = resolved_value.replace(f"${{{match}}}", str(field_value))
                        except Exception as e:
                            self.logger.error(f"Error resolving template variable ${{{match}}}: {e}")
                    
                    resolved_inputs[key] = resolved_value
            else:
                # No template variables, keep as-is
                resolved_inputs[key] = value
        
        return resolved_inputs
    
    def _get_nested_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Navigate nested dictionary/list structures using dot notation and array indexing."""
        try:
            current = data
            parts = field_path.split('.')
            
            for part in parts:
                if '[*]' in part:
                    # Handle array expansion like "results[*].link"
                    field_name = part.replace('[*]', '')
                    if field_name and field_name in current:
                        current = current[field_name]
                    
                    # If current is a list, extract the remaining field from each item
                    if isinstance(current, list):
                        remaining_parts = parts[parts.index(part) + 1:]
                        if remaining_parts:
                            remaining_path = '.'.join(remaining_parts)
                            result = []
                            for item in current:
                                if isinstance(item, dict):
                                    nested_value = self._get_nested_field(item, remaining_path)
                                    if nested_value is not None:
                                        result.append(nested_value)
                            return result
                        else:
                            return current
                elif part in current:
                    current = current[part]
                else:
                    return None
            
            return current
        except Exception as e:
            self.logger.error(f"Error navigating nested field '{field_path}': {e}")
            return None
    
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
            
            # Always use agent-prefixed stage names for multi-agent support
            agent_prefixed_stage = f"scout_{stage_name}"
            self.logger.info(f"Writing stage {agent_prefixed_stage} output to manifest at: {manifest_path}")
            
            # Use ManifestManager for consistent manifest operations
            from scout_agent.memory.manifest_manager import ManifestManager
            manifest_manager = ManifestManager(manifest_path, create_if_missing=True)
            
            # Store the stage output with agent-prefixed stage name
            manifest_manager.store_node_output(agent_prefixed_stage, data)
            
            # Update node status to completed with agent-prefixed stage name
            manifest_manager.update_node_status(
                node_id=agent_prefixed_stage,
                state="completed"
            )
            
            self.logger.info(f"Stage {agent_prefixed_stage} output written to manifest successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to write stage {agent_prefixed_stage} output: {e}")
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
