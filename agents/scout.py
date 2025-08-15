"""
ScoutAgent - Pain Point Discovery Agent

This agent specializes in discovering pain points in target markets using
web research, social media analysis, and user feedback collection.
"""

import asyncio
import json
import traceback
from typing import Dict, List, Any, Optional
import textwrap
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from .research_agent import ResearchAgent
from ..config import get_config
from ..mcp_integration.client.base import MCPClient
from ..mcp_integration.config import load_server_configs
from ..llm.utils import LLMAgentMixin, load_prompt_template
from ..llm.base import LLMBackendType
from ..services.agents.code.service import CodeExecutionService


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
        
        # Ensure a run_id is present and store it for later stages
        try:
            dag = plan.get("dag") or {}
            run_id = dag.get("run_id") or plan.get("run_id")
            if not run_id:
                run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            dag["run_id"] = run_id
            plan["dag"] = dag
            plan["run_id"] = run_id
            # persist in state
            setattr(self.state, "run_id", run_id)
        except Exception:
            pass

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
                        # Use Python literal repr for params so booleans are True/False, not JSON true/false
                        params_literal = repr(n.get('params') or {})
                        n["code"] = (
                            f"result = mcp_call(tool=\"{tool}\", params={params_literal}); "
                            f"save_json(\"{primary_out}\", result)"
                        )

            # Put back nodes
            dag["nodes"] = nodes
            plan["dag"] = dag

            return plan
        except Exception:
            # On any failure, return original plan
            return plan

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
            outs = n.get("outputs") or []
            if not outs:
                return False
            try:
                for o in outs:
                    # Resolve basic placeholders like {run_id}
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

        # Identify the plan file to update as a running manifest
        plan_file: Optional[Path] = None
        for candidate in (run_dir / "scout_plan.json", run_dir / "plan.json"):
            if candidate.exists():
                plan_file = candidate
                break

        def _update_manifest(nid: str, status: str, *, artifacts: Optional[List[str]] = None, error: Optional[Dict[str, Any]] = None):
            if not plan_file:
                return
            try:
                data = json.loads(plan_file.read_text())
            except Exception:
                data = {}
            try:
                ts = datetime.now().isoformat()
                dag_data = data.get("dag") or {}
                nodes_data = list(dag_data.get("nodes") or [])
                for node in nodes_data:
                    if (node.get("id") or "") == nid:
                        run_meta = node.get("run") or {}
                        run_meta["status"] = status
                        run_meta["updated_at"] = ts
                        if artifacts is not None:
                            run_meta["artifacts"] = artifacts
                        if error is not None:
                            run_meta["error"] = error
                        node["run"] = run_meta
                        break
                dag_data["nodes"] = nodes_data
                dag_data["updated_at"] = ts
                data["dag"] = dag_data
                plan_file.write_text(json.dumps(data, indent=2))
            except Exception:
                # Best-effort only; ignore manifest write errors
                pass

        async def exec_node(n: Dict[str, Any]):
            nid = n["id"]
            lang = (n.get("language") or "python").lower()
            code = (n.get("code") or "").strip()
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
            prelude = textwrap.dedent(f"""
import json, os, asyncio
import sys
from pathlib import Path

RUN_DIR = Path(r"{run_dir}")
RUN_DIR.mkdir(parents=True, exist_ok=True)
PROJ_ROOT = Path(r"{PROJECT_ROOT}")
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

def _ensure_payload(res):
    try:
        content = res.content[0].text if getattr(res, "content", None) else "{{}}"
        return json.loads(content)
    except Exception:
        return {{"raw": str(res)}}

def mcp_call(tool: str, params: dict):
    async def _run():
        servers = load_server_configs()
        client = MultiMCPClient(servers)
        await client.initialize()
        try:
            result = await client.call_tool(tool, params or {{}})
            return _ensure_payload(result)
        finally:
            await client.shutdown()
    return asyncio.run(_run())
            """)

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
                _update_manifest(nid, "completed", artifacts=generated_files)
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
                    _update_manifest(nid, "failed", error=err_info)
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
                    _update_manifest(nid, "completed", artifacts=manifest["artifacts"]) 
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
            # Determine run_id preference: explicit arg > plan dag/run_id > state
            dag = plan.get("dag") or {}
            chosen_run_id = run_id or dag.get("run_id") or plan.get("run_id") or getattr(self.state, "run_id", None)
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
