"""
ValidatorAgent - Pain Point Validation Agent

This agent specializes in validating pain points discovered and ranked by the ScoutAgent and ScreenerAgent
using external data sources like Reddit, Hacker News, Google, Twitter, and review sites.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from scout_agent.config import get_config
from scout_agent.llm.utils import LLMAgentMixin, load_prompt_template
from scout_agent.memory.manifest_manager import ManifestManager
from scout_agent.mcp_integration.client.base import MCPClient
from scout_agent.mcp_integration.config import load_server_configs
from scout_agent.services.execution.code_executor import AgentCodeExecutor


@dataclass
class ValidatorInput:
    """Input for ValidatorAgent."""
    pain_points: List[Dict[str, Any]] = field(default_factory=list)  # Pain points from ScreenerAgent
    target_market: str = ""  # Market context for validation
    validation_depth: str = "moderate"  # Validation depth (light, moderate, deep)
    market_context: str = ""  # Additional market context
    include_competitor_analysis: bool = True  # Whether to include competitor analysis
    
    # Base AgentInput fields
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_agent_input(cls, agent_input: AgentInput):
        """Create ValidatorInput from standard AgentInput."""
        # Extract pain points from data field
        data = agent_input.data
        pain_points = []
        
        if isinstance(data, list):
            pain_points = data
        elif isinstance(data, dict):
            pain_points = data.get("pain_points") or data.get("filtered_pain_points") or data.get("items") or []
        
        # Extract context fields
        context = agent_input.context or {}
        target_market = context.get("target_market", "")
        validation_depth = context.get("validation_depth", "moderate")
        market_context = context.get("market_context", "")
        include_competitor_analysis = bool(context.get("include_competitor_analysis", True))
        
        return cls(
            pain_points=pain_points,
            target_market=target_market,
            validation_depth=validation_depth,
            market_context=market_context,
            include_competitor_analysis=include_competitor_analysis,
            data=agent_input.data,
            metadata=agent_input.metadata,
            context=agent_input.context
        )


@dataclass
class ValidatorOutput:
    """Output from ValidatorAgent."""
    validated_pain_points: List[Dict[str, Any]]  # Validated pain points with evidence
    validation_summary: str  # Summary of validation findings
    market_insights: Dict[str, Any]  # Additional market insights discovered
    confidence_score: float  # Confidence in the validation (0.0-1.0)
    
    def to_agent_output(self) -> AgentOutput:
        """Convert to standard AgentOutput for compatibility."""
        return AgentOutput(
            result={
                "validated_pain_points": self.validated_pain_points,
                "validation_summary": self.validation_summary,
                "market_insights": self.market_insights,
                "confidence_score": self.confidence_score
            },
            metadata={
                "total_validated": len(self.validated_pain_points)
            },
            logs=[],
            execution_time=0.0,
            success=True
        )


class ValidatorAgent(BaseAgent, LLMAgentMixin):
    """
    ValidatorAgent for validating pain points.
    
    Uses external data sources to validate pain points and gather supporting evidence.
    """
    
    def __init__(self, agent_id: str = None):
        """Initialize the ValidatorAgent."""
        BaseAgent.__init__(self, name="validator", agent_id=agent_id)
        LLMAgentMixin.__init__(self)  # Use default backend
        self.name = "validator_agent"  # Used for prompt template loading
        self.config = get_config()
        self.research_client = None
        
        # Use default backend for all tasks - don't override any backend preferences
        
        
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Execute the validation process."""
        self.start_time = time.time()
        
        try:
            # Convert to ValidatorInput
            input_data = ValidatorInput.from_agent_input(agent_input)
            
            # Execute the three phases
            self._update_status('planning')
            plan = await self.plan(input_data)
            
            self._update_status('thinking')
            research_data = await self.think(input_data, plan)
            
            self._update_status('acting')
            result = await self.act(input_data, plan, research_data)
            
            execution_time = time.time() - self.start_time
            
            # Create output
            output = AgentOutput(
                result=result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'plan': plan,
                    'research_data': research_data
                },
                logs=self.execution_logs,
                execution_time=execution_time,
                success=True
            )
            
            self._update_status('completed')
            return output
            
        except Exception as e:
            self.logger.error(f"Error in ValidatorAgent: {str(e)}")
            execution_time = time.time() - self.start_time
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def _init_research_client(self):
        """Initialize the research client."""
        if self.research_client is None:
            try:
                # Connect to the research tools server
                self.research_client = MCPClient("research-tools", "http://127.0.0.1:8002/sse")
                self.logger.info("Connected to research tools server")
            except Exception as e:
                self.logger.error(f"Failed to connect to research tools server: {e}")
                raise
    
    async def plan(self, agent_input: AgentInput, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Plan the validation strategy for each pain point."""
        # Normalize incoming input into ValidatorInput
        input_data = self._normalize_input(agent_input)
        self.logger.info(f"Planning validation for {len(input_data.pain_points)} pain points")
        
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
                "pain_points": json.dumps(input_data.pain_points, indent=2),
                "pain_point_count": len(input_data.pain_points),
                "validation_depth": input_data.validation_depth,
                "market_context": input_data.market_context,
                "include_competitor_analysis": str(input_data.include_competitor_analysis).lower(),
                "tools_json": json.dumps(tools_catalog, ensure_ascii=False),
                "tool_names_csv": ", ".join(tool_names)
            }
            
            # Load and render the planning prompt template
            prompt_text = load_prompt_template(template_name="plan.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate plan using LLM
            self.logger.info("Calling LLM to generate validation plan...")
            llm_text = await self.llm_generate(prompt=prompt_text, task_type="plan")
            plan = self._extract_json(llm_text)
            # Post-process to ensure plan has the expected structure
            plan = self._postprocess_plan(plan, input_data)
            
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
                final_run_id = f"validator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dag["run_id"] = final_run_id
            plan["dag"] = dag
            plan["run_id"] = final_run_id
            # persist in state
            setattr(self.state, "run_id", final_run_id)
        except Exception:
            pass

        self.state.plan = plan
        return plan
            
    def _normalize_input(self, agent_input: Any) -> ValidatorInput:
        """Normalize incoming input into ValidatorInput."""
        if isinstance(agent_input, ValidatorInput):
            return agent_input
        
        # Use the from_agent_input class method to create a ValidatorInput
        return ValidatorInput.from_agent_input(agent_input)
        
    def _generate_research_node(self, pain_point_id: str, pain_point_desc: str, 
                               keywords: List[str], context: str, node_variant: str, 
                               depth: str = "medium", max_results: int = 10) -> Dict[str, Any]:
        """Generate a comprehensive_research node for a pain point with specific keywords.
        
        Args:
            pain_point_id: The ID of the pain point
            pain_point_desc: The description of the pain point
            keywords: The keywords to use for this research node
            context: The market context
            node_variant: The variant identifier for this node (e.g., "q1", "q2")
            depth: The research depth ("light", "medium", "deep")
            max_results: Maximum results per source
            
        Returns:
            A complete node definition with code for the comprehensive_research tool
        """
        # Create a unique node ID and output path
        node_id = f"validator_collect_{pain_point_id}_{node_variant}"
        output_path = f"stages.validator_collect.{pain_point_id}.{node_variant}"
        
        # Generate the code for the node
        code = (
            "# MCP tool call for comprehensive_research\n"
            "import json\n"
            "params = {\n"
            f"  \"topic\": \"{pain_point_desc}\",\n"
            f"  \"context\": \"{context}\",\n"
            f"  \"keywords\": {json.dumps(keywords)},\n"
            "  \"sources\": [\"reddit\", \"hn\", \"google\", \"twitter\", \"reviews\"],\n"
            f"  \"depth\": \"{depth}\",\n"
            f"  \"max_results_per_source\": {max_results}\n"
            "}\n"
            f"result = mcp_call(\"comprehensive_research\", params)\n"
            f"save_to_manifest(\"{output_path}\", result)\n"
            f"print(f\"DEBUG: Comprehensive research completed for {pain_point_id} ({node_variant})\")\n"
            "# Print the result as JSON so it can be captured by the execution framework\n"
            "print(json.dumps(result))\n"
        )
        
        # Create and return the complete node
        return {
            "id": node_id,
            "type": "tool",
            "tool": "comprehensive_research",
            "params": {
                "topic": pain_point_desc,
                "context": context,
                "keywords": keywords,
                "sources": ["reddit", "hn", "google", "twitter", "reviews"],
                "depth": depth,
                "max_results_per_source": max_results
            },
            "code": code,
            "language": "python",
            "outputs": [output_path]
        }

    def _distribute_keywords(self, search_queries: List[str], num_nodes: int) -> List[List[str]]:
        """Distribute search queries across multiple nodes.
        
        Args:
            search_queries: List of search queries to distribute
            num_nodes: Number of nodes to distribute across
            
        Returns:
            List of keyword lists, one for each node
        """
        if not search_queries:
            return [[] for _ in range(num_nodes)]
            
        # Ensure we have at least one keyword per node
        if len(search_queries) < num_nodes:
            # Duplicate some keywords to ensure each node has at least one
            extended_queries = search_queries * (num_nodes // len(search_queries) + 1)
            search_queries = extended_queries[:num_nodes]
        
        # Distribute keywords across nodes
        result = [[] for _ in range(num_nodes)]
        for i, query in enumerate(search_queries):
            node_idx = i % num_nodes
            result[node_idx].append(query)
            
        return result

    def _postprocess_plan(self, plan: Dict[str, Any], input_data: ValidatorInput) -> Dict[str, Any]:
        """Process the LLM-generated plan, ensuring it has the expected structure."""
        try:
            if not isinstance(plan, dict):
                self.logger.warning("Plan is not a dictionary, returning empty plan")
                return {"dag": {"nodes": []}}
            
            # Check if the plan already has a DAG with nodes
            dag = plan.get("dag", {})
            nodes = dag.get("nodes", [])
            
            if nodes:
                self.logger.info(f"Found {len(nodes)} nodes in LLM-generated plan")
                for i, node in enumerate(nodes):
                    node_id = node.get("id", f"node_{i}")
                    node_type = node.get("type", "unknown")
                    node_tool = node.get("tool", "none")
                    has_code = "code" in node
                    self.logger.info(f"Node {i}: id={node_id}, type={node_type}, tool={node_tool}, has_code={has_code}")
            
            # Ensure the plan has the expected structure
            if "metadata" not in plan:
                plan["metadata"] = {
                    "validation_depth": input_data.validation_depth,
                    "market_context": input_data.market_context,
                    "include_competitor_analysis": input_data.include_competitor_analysis
                }
                
                # If validation_strategies exists, move it to metadata
                if "validation_strategies" in plan:
                    plan["metadata"]["validation_strategies"] = plan.pop("validation_strategies")
            
            # Get validation strategies from metadata
            strategies = plan.get("metadata", {}).get("validation_strategies") if isinstance(plan.get("metadata"), dict) else None
            
            # Generate nodes from validation strategies
            dag = plan.get("dag") or {}
            nodes = dag.get("nodes") or []
            
            # If we already have nodes, don't regenerate them
            if nodes:
                self.logger.info("Using existing nodes from plan")
                return plan
                
            self.logger.info("No nodes found in plan, generating from validation strategies")
            
            # Determine number of nodes per pain point based on validation depth
            nodes_per_pain_point = 2  # Default for light/medium depth
            if input_data.validation_depth == "deep":
                nodes_per_pain_point = 3
            
            try:
                # Process each pain point
                generated_nodes = []
                for idx, pp in enumerate(input_data.pain_points):
                    pain_id = pp.get("id") or f"pp{idx+1}"
                    desc = pp.get("description", "")
                    
                    # Find matching strategy
                    strategy = None
                    if isinstance(strategies, list):
                        for s in strategies:
                            if s.get("pain_point_id") == pain_id:
                                strategy = s
                                break
                    
                    # Get search queries from strategy or fallback to description
                    search_queries = []
                    research_depth = input_data.validation_depth or "medium"
                    max_results = 10
                    
                    if strategy:
                        search_queries = strategy.get("search_queries") or []
                        if strategy.get("research_depth"):
                            research_depth = strategy.get("research_depth")
                        if strategy.get("max_results_per_source"):
                            max_results = strategy.get("max_results_per_source")
                    
                    if not search_queries and desc:
                        search_queries = [desc]
                    
                    # Distribute keywords across nodes
                    keyword_groups = self._distribute_keywords(search_queries, nodes_per_pain_point)
                    
                    # Generate a node for each keyword group
                    for node_idx, keywords in enumerate(keyword_groups):
                        node_variant = f"q{node_idx+1}"
                        node = self._generate_research_node(
                            pain_id, 
                            desc, 
                            keywords, 
                            input_data.market_context or input_data.target_market,
                            node_variant,
                            research_depth,
                            max_results
                        )
                        generated_nodes.append(node)
                        
                self.logger.info(f"Generated {len(generated_nodes)} research nodes from validation strategies")
                dag["nodes"] = generated_nodes
                plan["dag"] = dag
                
            except Exception as gen_e:
                self.logger.error(f"Error generating nodes from strategies: {gen_e}")
                # Fall back to synthesizing minimal nodes if generation fails
                self.logger.warning("Node generation failed; synthesizing fallback collect nodes")
                print("[validator_plan] Fallback: synthesizing collect nodes due to node generation failure")
                
                # Use existing fallback logic
                synthesized_nodes: List[Dict[str, Any]] = []
                for idx, pp in enumerate(input_data.pain_points):
                    pain_id = pp.get("id") or f"pp{idx+1}"
                    desc = pp.get("description", "")
                    # try to find strategy queries
                    kw = []
                    if isinstance(strategies, list):
                        for s in strategies:
                            if s.get("pain_point_id") == pain_id:
                                kw = s.get("search_queries") or []
                                break
                    if not kw and desc:
                        kw = [desc]
                    
                    # Generate a single fallback node
                    node = self._generate_research_node(
                        pain_id,
                        desc,
                        kw,
                        input_data.market_context or input_data.target_market,
                        "fallback",
                        input_data.validation_depth or "medium",
                        10
                    )
                    synthesized_nodes.append(node)
                
                dag["nodes"] = synthesized_nodes
                plan["dag"] = dag

            # Return the plan with the generated nodes
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

    async def _execute_plan_non_agent_nodes(self, plan: Dict[str, Any], run_dir_override: Optional[Path] = None) -> Dict[str, Any]:
        """Execute the non-agent nodes in the plan using the AgentCodeExecutor."""
        self.logger.info("Executing non-agent nodes in the plan")
        
        try:
            # Extract the DAG from the plan
            dag = plan.get("dag", {})
            nodes = dag.get("nodes", [])
            
            # Filter out non-agent nodes (tool nodes)
            tool_nodes = [n for n in nodes if n.get("type") == "tool"]
            self.logger.info(f"Found {len(tool_nodes)} tool nodes in the plan")
            
            if not tool_nodes:
                self.logger.warning("No tool nodes found in the plan")
                return {"completed": [], "failed": []}
            
            # Initialize the code executor service
            code_executor = AgentCodeExecutor()
            await code_executor._initialize(None)
            await code_executor._start()
            
            # Initialize manifest manager for storing results
            run_id = dag.get("run_id") or plan.get("run_id")
            run_dir = run_dir_override or Path(f"data/runs/{run_id}") if run_id else None
            if run_dir:
                manifest_path = run_dir / "run_manifest.json"
                manifest_manager = ManifestManager(manifest_path, create_if_missing=False)
            else:
                manifest_manager = None
                self.logger.warning("No run directory available, tool results won't be persisted")
            
            # Execute each tool node
            completed_nodes = []
            failed_nodes = []
            tool_results = {}  # Store actual results for stage persistence
            
            for node in tool_nodes:
                node_id = node.get("id", "unknown")
                tool_name = node.get("tool", "")
                code = node.get("code", "")
                
                self.logger.info(f"Executing tool node {node_id} with tool {tool_name}")
                
                try:
                    # Execute the code in the sandbox with proper prelude
                    result = await code_executor.wrap_and_execute(
                        node=node,
                        agent_id="validator",
                        stage="collect",
                        context={
                            "run_dir": str(run_dir) if run_dir else None,
                            "workflow_id": run_id,
                            "node_id": node_id,
                            "tool": tool_name
                        }
                    )
                    
                    # Check if execution was successful
                    if result.success:
                        self.logger.info(f"Successfully executed tool node {node_id}")
                        completed_nodes.append(node_id)
                        
                        # Extract and store the tool result data
                        if hasattr(result, 'output') and result.output:
                            try:
                                # Extract JSON from raw execution output that contains debug logs + JSON
                                if isinstance(result.output, str):
                                    result_data = self._extract_json_from_execution_output(result.output)
                                else:
                                    result_data = result.output
                                
                                # Store result in tool_results for stage persistence
                                tool_results[node_id] = result_data
                                
                                # Note: We don't store individual tool outputs under the declared output paths
                                # because ManifestManager.store_node_output treats paths like "stages.validator_collect.pp1.q1" 
                                # as literal keys rather than nested paths. Instead, we accumulate all results
                                # and store them properly in the nested structure at the end.
                                
                                self.logger.info(f"Successfully executed and stored results for node: {node_id}")
                            except Exception as e:
                                self.logger.warning(f"Failed to parse/store result for {node_id}: {e}")
                        else:
                            self.logger.warning(f"Tool node {node_id} returned no output")
                    else:
                        error = result.error if hasattr(result, 'error') else "Unknown error"
                        self.logger.error(f"Failed to execute tool node {node_id}: {error}")
                        failed_nodes.append(f"{node_id}: {error}")
                except Exception as e:
                    self.logger.error(f"Error executing tool node {node_id}: {str(e)}")
                    failed_nodes.append(f"{node_id}: {str(e)}")
            
            # Store collected tool results in proper nested structure
            if manifest_manager and tool_results:
                try:
                    manifest = manifest_manager.get_manifest()
                    stages = manifest.setdefault("stages", {})
                    validator_collect_stage = stages.setdefault("validator_collect", {})
                    
                    # Create properly nested structure: pp1: {q1: data, q2: data}, pp2: {q1: data, q2: data}
                    nested_data = {}
                    for node_id, result_data in tool_results.items():
                        # Extract pp and q from node_id like "validator_collect_pp1_q1"
                        import re
                        match = re.search(r'pp(\d+)_q(\d+)', node_id)
                        if match:
                            pp_num = match.group(1)
                            q_num = match.group(2)
                            pp_key = f"pp{pp_num}"
                            q_key = f"q{q_num}"
                            
                            if pp_key not in nested_data:
                                nested_data[pp_key] = {}
                            nested_data[pp_key][q_key] = result_data
                        else:
                            # Fallback: store under the original node_id
                            nested_data[node_id] = result_data
                    
                    validator_collect_stage["data"] = nested_data
                    validator_collect_stage["updated_at"] = datetime.now().isoformat()
                    manifest_manager._save()
                    
                    pp_keys = list(nested_data.keys())
                    self.logger.info(f"Persisted validator_collect data with pain point keys: {pp_keys}")
                    for pp_key, queries in nested_data.items():
                        if isinstance(queries, dict):
                            q_keys = list(queries.keys())
                            self.logger.info(f"  {pp_key}: {q_keys}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to persist validator_collect stage data: {e}")
            
            # Return the summary
            summary = {
                "completed": completed_nodes,
                "failed": failed_nodes
            }
            
            self.logger.info(f"Completed {len(completed_nodes)} tool nodes, failed {len(failed_nodes)} tool nodes")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error executing non-agent nodes: {str(e)}\n{traceback.format_exc()}")
            return {"completed": [], "failed": [str(e)]}
    
    async def think(self, input_data: ValidatorInput, plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze research data from the collect stage for each pain point."""
        self.logger.info("Analyzing research data from collect stage")
        
        try:
            # Use plan from parameter or from agent state
            if plan is None:
                plan = getattr(self.state, "plan", None)
                
            # If still None, try to get from manifest
            if plan is None:
                self.logger.info("No plan provided, attempting to load from manifest")
                plan = self._get_stage_output("plan")
                
            if plan is None:
                raise ValueError("No validation plan available")
            
            # Extract the validation strategies
            strategies = plan.get("validation_strategies", [])
            if not strategies:
                self.logger.warning("No validation strategies found in plan, checking metadata")
                strategies = plan.get("metadata", {}).get("validation_strategies", [])
                
                # If still not found, check if there are pain points in input_data and generate basic strategies
                if not strategies and hasattr(input_data, 'pain_points') and input_data.pain_points:
                    self.logger.warning("Generating basic validation strategies from pain points")
                    strategies = []
                    for idx, pp in enumerate(input_data.pain_points):
                        pain_id = pp.get("id") or f"pp{idx+1}"
                        desc = pp.get("description", "")
                        strategies.append({
                            "pain_point_id": pain_id,
                            "pain_point_description": desc,
                            "search_queries": [desc] if desc else [],
                            "research_depth": "medium",
                            "max_results_per_source": 10
                        })
                    self.logger.info(f"Generated {len(strategies)} basic validation strategies")
                
                if not strategies:
                    raise ValueError("No validation strategies found in plan and could not generate from input")
            
            # Get research data from collect stage
            collect_data = self._get_stage_output("collect")
            if not collect_data:
                self.logger.warning("No collect data found in manifest, checking for tool node outputs")
                
                # Try to extract data from completed tool nodes in the manifest
                collect_summary = self._get_collect_tool_outputs(plan)
                if not collect_summary or not collect_summary.get("completed"):
                    self.logger.error("No collect data or tool outputs found")
                    raise ValueError("No research data available from collect stage")
            
            # Process the collected research data
            processed_results = self._process_collect_research_data(collect_data, strategies)
            
            # Create the research data result
            research_data = {
                "pain_point_research": processed_results,
                "total_pain_points_researched": len(processed_results),
                "successful_research": len([p for p in processed_results.values() if "error" not in p]),
                "failed_research": len([p for p in processed_results.values() if "error" in p]),
                "research_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Completed research for {research_data['successful_research']} pain points")
            
            # Analyze research data for each pain point sequentially, but sources in parallel
            analyzed_data = {}
            
            # Process each pain point sequentially
            for pain_point_id, pain_point_data in processed_results.items():
                if "error" in pain_point_data:
                    self.logger.warning(f"Skipping pain point {pain_point_id} due to error in data")
                    continue
                
                self.logger.info(f"Processing pain point {pain_point_id}")
                
                # Find the corresponding strategy to get the pain point description
                pain_point_desc = next((s.get("pain_point_description", "") 
                                      for s in strategies if s.get("pain_point_id") == pain_point_id), "")
                
                # For each pain point, process sources in parallel
                source_tasks = []
                for source, source_data in pain_point_data.get("sources", {}).items():
                    task = self._analyze_research_data(
                        pain_point_id=pain_point_id,
                        pain_point_description=pain_point_desc,
                        research_data=source_data,
                        source=source
                    )
                    source_tasks.append(task)
                
                # Execute source tasks in parallel for this pain point
                if source_tasks:
                    self.logger.info(f"Analyzing {len(source_tasks)} sources for pain point {pain_point_id}")
                    source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
                    
                    # Process results for this pain point
                    for result in source_results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Analysis task for pain point {pain_point_id} failed: {str(result)}")
                            continue
                        
                        source = result.get("source")
                        if source:
                            if pain_point_id not in analyzed_data:
                                analyzed_data[pain_point_id] = {}
                            
                            analyzed_data[pain_point_id][source] = result
                    
                    self.logger.info(f"Completed analysis for pain point {pain_point_id}")
                else:
                    self.logger.warning(f"No sources to analyze for pain point {pain_point_id}")
            
            # No need for additional processing of analysis_results as we've already built analyzed_data
            
            # Create the final analyzed research data - no need to store raw_research again
            # as it's already available in the collect stage output
            final_research_data = {
                "analyzed_research": analyzed_data,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_pain_points_analyzed": len(analyzed_data),
                "successful_analysis": sum(1 for pp_data in analyzed_data.values() if pp_data)
            }
            
            self.logger.info(f"Completed analysis for {len(analyzed_data)} pain points")
            
            # Write analyzed research data to manifest
            self._write_stage_output("think", final_research_data)
            
            # Store in agent state for act phase
            self.state.research_data = final_research_data
            
            return final_research_data
            
        except Exception as e:
            self.logger.error(f"Error in think phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback: generate basic research data
            pain_points = []
            if hasattr(input_data, 'pain_points'):
                pain_points = input_data.pain_points
            elif isinstance(input_data, dict) and 'pain_points' in input_data:
                pain_points = input_data['pain_points']
            
            fallback_research = self._generate_fallback_research(pain_points)
            
            # Write fallback research to manifest
            self._write_stage_output("think", fallback_research)
            
            # Store in agent state for act phase
            self.state.research_data = fallback_research
            
            return fallback_research
    
    def _get_collect_tool_outputs(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool outputs from completed nodes in the manifest."""
        try:
            # Get the DAG from the plan
            dag = plan.get("dag", {})
            if not dag:
                return {}
                
            # Find all completed tool nodes
            completed_nodes = {}
            for node_id, node in dag.items():
                if node.get("type") == "tool" and node.get("status") == "completed":
                    completed_nodes[node_id] = node
            
            # Extract the outputs
            tool_outputs = {
                "completed": bool(completed_nodes),
                "nodes": completed_nodes
            }
            
            return tool_outputs
        except Exception as e:
            self.logger.error(f"Error extracting tool outputs: {str(e)}")
            return {}
            
    def _process_collect_research_data(self, collect_data: Dict[str, Any], strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process research data from collect stage for each pain point.

        Accepts multiple shapes:
        - { pain_point_id: { query_id: { data: {source: {content: [...]}}}} }
        - { pain_point_id: [ { data: {...} }, ... ] }
        - { sources: {reddit|hackernews|google: {content: [...]}} } (aggregate)
        - Lists of the above
        """
        processed_results = {}
        
        try:
            # If collect_data is None or empty, return empty results
            if not collect_data:
                self.logger.warning("No collect data available")
                return processed_results
                
            # Normalize to iterable of (pain_point_id, queries_like)
            items_iter: List[Any] = []
            if isinstance(collect_data, dict):
                # Check if we have the old flat structure with keys like "validator_collect_pp1_q1"
                flat_keys = [k for k in collect_data.keys() if k.startswith("validator_collect_")]
                if flat_keys:
                    # Convert old flat structure to nested
                    self.logger.info(f"Converting old flat structure with {len(flat_keys)} tool results to nested format")
                    nested_data = {}
                    for flat_key, tool_data in collect_data.items():
                        if flat_key.startswith("validator_collect_"):
                            import re
                            match = re.search(r'pp(\d+)_q(\d+)', flat_key)
                            if match:
                                pp_num = match.group(1)
                                q_num = match.group(2)
                                pp_key = f"pp{pp_num}"
                                q_key = f"q{q_num}"
                                
                                if pp_key not in nested_data:
                                    nested_data[pp_key] = {}
                                nested_data[pp_key][q_key] = tool_data
                        elif flat_key not in ("completed", "failed"):
                            # Keep non-tool keys as-is
                            nested_data[flat_key] = tool_data
                    
                    items_iter = list(nested_data.items())
                # Aggregate form with sources at top level
                elif "sources" in collect_data and isinstance(collect_data.get("sources"), dict):
                    items_iter = [("pp_aggregate", {"aggregate": {"data": collect_data}})]
                # Or nested under data
                elif "data" in collect_data and isinstance(collect_data.get("data"), dict) \
                        and "sources" in collect_data["data"] and isinstance(collect_data["data"]["sources"], dict):
                    items_iter = [("pp_aggregate", {"aggregate": {"data": collect_data["data"]}})]
                # If we only have bookkeeping keys, there's nothing to process
                elif set(collect_data.keys()).issubset({"completed", "failed"}):
                    self.logger.info("Collect data contains only completion bookkeeping; no sources to process")
                    return processed_results
                else:
                    # Assume new nested structure: pp1: {q1: data, q2: data}
                    items_iter = list(collect_data.items())
            elif isinstance(collect_data, list):
                # Treat as a single aggregate list under an implicit pain point
                items_iter = [("pp_list", collect_data)]
            else:
                self.logger.warning(f"Unexpected collect_data type: {type(collect_data)}")
                return processed_results
            
            try:
                keys_preview = list(collect_data.keys()) if isinstance(collect_data, dict) else [f"len={len(collect_data)}"]
                self.logger.info(f"Processing collect data with keys: {keys_preview}")
            except Exception:
                pass
            
            # Map strategies to pain point IDs for easier lookup
            strategy_map = {}
            for strategy in strategies:
                pain_point_id = strategy.get("pain_point_id")
                if pain_point_id:
                    strategy_map[pain_point_id] = strategy
                
            # Process each pain point in the collect data
            # At this point, items_iter should contain (pp1, {q1: data, q2: data}) pairs
            for pain_point_id, queries in items_iter:
                # Skip bookkeeping keys that may have leaked into iteration
                if pain_point_id in ("completed", "failed"):
                    continue
                
                # pain_point_id should now be pp1, pp2, etc.
                actual_pain_point_id = pain_point_id
                # Get the strategy for this pain point if available
                strategy = strategy_map.get(actual_pain_point_id)
                pain_point_desc = strategy.get("pain_point_description", "") if strategy else ""
                
                # Initialize pain point data structure
                pain_point_data = {
                    "pain_point_id": actual_pain_point_id,
                    "pain_point_description": pain_point_desc,
                    "sources": {}
                }
                
                # Helper to add content entries to a source, parsing JSON strings when present
                def _extend_source(source_name: str, source_payload: Any) -> None:
                    if source_name not in pain_point_data["sources"]:
                        pain_point_data["sources"][source_name] = []
                    if isinstance(source_payload, dict) and "content" in source_payload and isinstance(source_payload["content"], list):
                        for entry in source_payload["content"]:
                            if isinstance(entry, dict) and "text" in entry and isinstance(entry["text"], str):
                                text = entry["text"]
                                try:
                                    parsed = json.loads(text)
                                    # If parsed is a dict containing a list (threads, posts, web_results, reviews), flatten
                                    flattened = None
                                    for key in ("threads", "posts", "web_results", "reviews"):
                                        if isinstance(parsed, dict) and key in parsed and isinstance(parsed[key], list):
                                            flattened = parsed[key]
                                            break
                                    if flattened is not None:
                                        pain_point_data["sources"][source_name].extend(flattened)
                                    else:
                                        pain_point_data["sources"][source_name].append(parsed)
                                except Exception:
                                    pain_point_data["sources"][source_name].append(entry)
                            else:
                                pain_point_data["sources"][source_name].append(entry)
                    elif isinstance(source_payload, list):
                        pain_point_data["sources"][source_name].extend(source_payload)
                    elif isinstance(source_payload, dict):
                        pain_point_data["sources"][source_name].append(source_payload)
                
                # Process each query for this pain point
                # queries should now be a dict like {q1: data, q2: data}
                if isinstance(queries, dict):
                    query_iter = queries.items()
                elif isinstance(queries, list):
                    # fabricate ids for list entries
                    query_iter = [(f"q{idx+1}", q) for idx, q in enumerate(queries)]
                else:
                    query_iter = []
                
                for query_id, query_data in query_iter:
                    # Extract raw_result from the tool output if present
                    if isinstance(query_data, dict) and "raw_result" in query_data:
                        raw_result = query_data["raw_result"]
                        # Try to extract JSON from the raw result
                        try:
                            # Look for JSON at the end of the raw result string
                            import re
                            json_match = re.search(r'\{.*\}$', raw_result, re.DOTALL)
                            if json_match:
                                json_str = json_match.group()
                                parsed_data = json.loads(json_str)
                                # Transform to expected format with query_data
                                query_data = parsed_data
                            else:
                                self.logger.warning(f"No JSON found in raw_result for {query_id}")
                                continue
                        except Exception as e:
                            self.logger.warning(f"Failed to parse JSON from raw_result for {query_id}: {e}")
                            continue
                    try:
                        self.logger.info(f"Processing query {query_id} for pain point {actual_pain_point_id}")
                        if isinstance(query_data, dict):
                            # Prefer nested data->sources
                            if "data" in query_data and isinstance(query_data["data"], dict):
                                for source, source_data in query_data["data"].items():
                                    _extend_source(source, source_data)
                            # Or direct sources
                            elif "sources" in query_data and isinstance(query_data["sources"], dict):
                                for source, source_data in query_data["sources"].items():
                                    _extend_source(source, source_data)
                            # Or aggregate form
                            elif all(k in query_data for k in ("topic", "context", "keywords")) and "data" in query_data:
                                data_block = query_data.get("data", {})
                                if isinstance(data_block, dict):
                                    for source, source_data in data_block.items():
                                        _extend_source(source, source_data)
                        elif isinstance(query_data, list):
                            for entry in query_data:
                                if isinstance(entry, dict):
                                    if "data" in entry and isinstance(entry["data"], dict):
                                        for source, source_data in entry["data"].items():
                                            _extend_source(source, source_data)
                                    elif "sources" in entry and isinstance(entry["sources"], dict):
                                        for source, source_data in entry["sources"].items():
                                            _extend_source(source, source_data)
                    except Exception as _e:
                        self.logger.warning(f"Skipping malformed query {query_id}: {_e}")
                
                # If we found data for this pain point, add it to results
                if pain_point_data["sources"]:
                    processed_results[actual_pain_point_id] = pain_point_data
                else:
                    # No data found for this pain point
                    processed_results[actual_pain_point_id] = {
                        "pain_point_id": actual_pain_point_id,
                        "pain_point_description": pain_point_desc,
                        "error": "No research data found for this pain point",
                        "sources": {}
                    }
            
            return processed_results
        except Exception as e:
            self.logger.error(f"Error processing collect research data: {str(e)}")
            # Return empty results on error
            return processed_results
            
    async def _analyze_research_data(
        self,
        pain_point_id: str,
        pain_point_description: str,
        research_data: List[Dict[str, Any]],
        source: str
    ) -> Dict[str, Any]:
        """Analyze research data for a specific pain point and source.
        
        This method performs the following analysis:
        1. Discards irrelevant entries that don't relate to the pain point
        2. Determines if pertinent entries support or contradict the pain point
        3. Condenses similar entries to avoid redundancy
        4. Provides justification for the analysis
        """
        self.logger.info(f"Analyzing research data for pain point {pain_point_id} from {source}")
        
        try:
            # Load the think prompt template
            prompt_template = self._load_prompt_template("think")
            
            # Format the research data as a string for the prompt
            research_data_str = json.dumps(research_data, indent=2)
            
            # Format the prompt with the pain point and research data
            prompt = prompt_template.format(
                pain_point=pain_point_id,
                pain_point_description=pain_point_description,
                research_data=research_data_str,
                source=source
            )
            
            # Call the LLM to analyze the research data using llm_generate
            # Don't specify a specific backend - use the default backend configuration
            response = await self.llm_generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.2  # Lower temperature for more focused analysis
            )
            
            # Extract JSON from the response
            analysis_result = self._extract_json(response)
            
            # Validate the analysis result structure
            if not analysis_result or not isinstance(analysis_result, dict):
                raise ValueError(f"Invalid analysis result structure: {analysis_result}")
            
            # Ensure the pain_point_id and source are correctly set
            analysis_result["pain_point_id"] = pain_point_id
            analysis_result["source"] = source
            
            # Validate and structure the analyzed entries
            if "entries" not in analysis_result:
                analysis_result["entries"] = {
                    "supporting": [],
                    "contradicting": [],
                    "neutral": [],
                    "irrelevant": []
                }
                
            # Ensure we have the expected categories
            for category in ["supporting", "contradicting", "neutral", "irrelevant"]:
                if category not in analysis_result["entries"]:
                    analysis_result["entries"][category] = []
                    
            # Validate that each entry has the required fields
            for category in ["supporting", "contradicting", "neutral"]:
                for entry in analysis_result["entries"].get(category, []):
                    if not all(key in entry for key in ["content", "justification"]):
                        self.logger.warning(f"Entry in {category} missing required fields: {entry}")
                        
            # Add summary counts
            analysis_result["summary"] = {
                "total_entries": len(research_data),
                "supporting_count": len(analysis_result["entries"].get("supporting", [])),
                "contradicting_count": len(analysis_result["entries"].get("contradicting", [])),
                "neutral_count": len(analysis_result["entries"].get("neutral", [])),
                "irrelevant_count": len(analysis_result["entries"].get("irrelevant", []))
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for pain point {pain_point_id} from {source}: {str(e)}")
            # Return a minimal structure with error information
            return {
                "pain_point_id": pain_point_id,
                "source": source,
                "error": str(e),
                "entries": {
                    "supporting": [],
                    "contradicting": [],
                    "neutral": [],
                    "irrelevant": []
                },
                "summary": {
                    "total_entries": 0,
                    "supporting_count": 0,
                    "contradicting_count": 0,
                    "neutral_count": 0,
                    "irrelevant_count": 0
                },
                "confidence_level": "low",
                "analysis_summary": f"Analysis failed: {str(e)}"
            }
    
    def _extract_condensed_evidence(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract condensed evidence from research data.
        
        This extracts only supporting and contradicting evidence from the research data,
        organizing it by pain point ID for easier consumption by the LLM.
        """
        self.logger.info("Extracting condensed evidence from research data")
        
        condensed_evidence = {}
        
        # Check if we have the expected structure
        if not research_data or not isinstance(research_data, dict):
            self.logger.warning("Invalid research data structure for evidence extraction")
            return condensed_evidence
            
        # Try to get analyzed_research from the research data
        analyzed_research = None
        
        # Check if research_data has pain_point_research directly
        if "pain_point_research" in research_data:
            analyzed_research = research_data["pain_point_research"]
        # Check if it's in the data field (from think stage output)
        elif "data" in research_data and isinstance(research_data["data"], dict):
            if "analyzed_research" in research_data["data"]:
                analyzed_research = research_data["data"]["analyzed_research"]
            elif "pain_point_research" in research_data["data"]:
                analyzed_research = research_data["data"]["pain_point_research"]
                
        if not analyzed_research or not isinstance(analyzed_research, dict):
            self.logger.warning("No analyzed research found in research data")
            return condensed_evidence
            
        # Process each pain point
        for pain_point_id, sources_data in analyzed_research.items():
            pain_point_evidence = {
                "supporting": [],
                "contradicting": []
            }
            
            # Process each source for this pain point
            if isinstance(sources_data, dict):
                for source, source_data in sources_data.items():
                    # Skip non-dict source data
                    if not isinstance(source_data, dict):
                        continue
                        
                    # Extract entries from the source data
                    entries = source_data.get("entries", {})
                    
                    # Extract supporting and contradicting evidence
                    for entry in entries.get("supporting", []):
                        if isinstance(entry, dict):
                            evidence_item = {
                                "source": source,
                                "content": entry.get("content", ""),
                                "justification": entry.get("justification", "")
                            }
                            pain_point_evidence["supporting"].append(evidence_item)
                    
                    for entry in entries.get("contradicting", []):
                        if isinstance(entry, dict):
                            evidence_item = {
                                "source": source,
                                "content": entry.get("content", ""),
                                "justification": entry.get("justification", "")
                            }
                            pain_point_evidence["contradicting"].append(evidence_item)
            
            # Add evidence for this pain point to the condensed evidence
            condensed_evidence[pain_point_id] = pain_point_evidence
        
        self.logger.info(f"Extracted condensed evidence for {len(condensed_evidence)} pain points")
        return condensed_evidence
    
    async def act(self, input_data: ValidatorInput, plan: Dict[str, Any] = None, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze research data and validate pain points."""
        self.logger.info("Analyzing research data and validating pain points")
        from pathlib import Path
        import json
        import os
        
        start_time = datetime.now()
        debug_info = {"has_research_data": False, "plan_provided": bool(plan)}
        
        try:
            # Ensure input_data is ValidatorInput
            if not isinstance(input_data, ValidatorInput):
                input_data = self._normalize_input(input_data)
            
            # Get pain points from input data
            pain_points = input_data.pain_points
            target_market = input_data.target_market
            
            # Get research data from parameters, agent state, or manifest
            if not research_data:
                # Try to get from agent state
                research_data = getattr(self.state, 'research_data', None)
                if research_data:
                    self.logger.info("Retrieved research data from agent state")
                
                # If still not found, try to get from manifest
                if not research_data and hasattr(self, 'manifest_manager'):
                    # Try different node IDs to find the research data
                    node_ids = [f"{self.name}_think", "validator_think", "think"]
                    
                    for node_id in node_ids:
                        research_data = self.manifest_manager.get_node_output(node_id)
                        if research_data:
                            self.logger.info(f"Retrieved research data from manifest with node_id: {node_id}")
                            break
            
            if not research_data:
                self.logger.error("No research data available from think stage")
                raise ValueError("No research data available from think stage")
            
            debug_info["has_research_data"] = True
            
            # Extract condensed evidence from research data
            condensed_evidence = self._extract_condensed_evidence(research_data)
            
            # Save condensed evidence for debugging
            debug_dir = Path(os.getcwd()) / "debug"
            debug_dir.mkdir(exist_ok=True)
            with open(debug_dir / "condensed_evidence.json", "w") as f:
                json.dump(condensed_evidence, f, indent=2)
            
            # Load prompt template with pain points, market focus, and condensed evidence
            prompt = load_prompt_template(
                template_name="act.prompt",
                agent_name="validator_agent",
                substitutions={
                    "pain_points": json.dumps(pain_points, indent=2),
                    "market_focus": target_market,
                    "condensed_evidence": json.dumps(condensed_evidence, indent=2)
                }
            )
                
            # Generate validation analysis using LLM
            self.logger.info("Calling LLM to analyze research data and validate pain points...")
            response = await self.llm_generate(prompt=prompt, task_type="analysis")
            
            # Extract JSON from response
            validation_result = self._extract_json(response)
            
            if not validation_result:
                self.logger.warning("Failed to extract JSON from LLM response, using fallback")
                # Create fallback validation result
                validation_result = {
                    "validated_pain_points": [],
                    "rejected_pain_points": [],
                    "validation_summary": "Failed to extract validation results from LLM response",
                    "validation_timestamp": datetime.now().isoformat(),
                    "error": "JSON extraction failed"
                }
            
            # Add metadata to validation result
            validation_result["validation_timestamp"] = datetime.now().isoformat()
            validation_result["processing_time"] = (datetime.now() - start_time).total_seconds()
            validation_result["pain_points_analyzed"] = len(pain_points)
            validation_result["evidence_sources_used"] = list(set([ev["source"] for pp_id in condensed_evidence for ev in condensed_evidence[pp_id]["supporting"] + condensed_evidence[pp_id]["contradicting"]]))
            
            # Write validation result to manifest
            if hasattr(self, 'manifest_manager'):
                self.logger.info("Writing stage validator_act output to manifest")
                self.manifest_manager.store_node_output(f"{self.name}_act", validation_result)
                self.logger.info("Stage validator_act output written to manifest successfully")
            
            return validation_result
        except Exception as e:
            self.logger.error(f"Error in act phase: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Create error result
            error_result = {
                "validated_pain_points": [],
                "rejected_pain_points": [],
                "validation_summary": f"Error in validation: {str(e)}",
                "validation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "debug_info": debug_info
            }
            
            # Write error result to manifest
            if hasattr(self, 'manifest_manager'):
                self.logger.info("Writing stage validator_act output to manifest")
                self.manifest_manager.store_node_output(f"{self.name}_act", error_result)
                self.logger.info("Stage validator_act output written to manifest successfully")
            
            return error_result
            
            # Write validation result to manifest
            self._write_stage_output("act", validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in act phase: {str(e)}\n{traceback.format_exc()}")
            # Create fallback validation result
            fallback_validation = {
                "validated_pain_points": [],
                "rejected_pain_points": [],
                "validation_summary": f"Error in validation: {str(e)}",
                "validation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "debug_info": {
                    "has_research_data": research_data is not None,
                    "plan_provided": plan is not None
                }
            }
            
            # Write fallback result to manifest
            self._write_stage_output("act", fallback_validation)
            
            return fallback_validation
            

    
    async def _execute_research_for_pain_point(
        self, 
        pain_point_id: str, 
        pain_point_desc: str, 
        search_queries: List[str],
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Execute research for a single pain point using multiple data sources."""
        self.logger.info(f"Researching pain point: {pain_point_id}")
        
        result = {
            "pain_point_id": pain_point_id,
            "pain_point_description": pain_point_desc,
            "sources": {},
            "comprehensive_data": {}
        }
        
        try:
            # Use the comprehensive_research tool to get data from all sources at once
            try:
                # Extract keywords from search queries
                keywords = []
                for query in search_queries:
                    if query != pain_point_desc and query.strip():
                        keywords.append(query.strip())
                
                # Make a single call to comprehensive_research
                comprehensive_response = await self.research_client.call(
                    "comprehensive_research",
                    topic=pain_point_desc,
                    context=pain_point_desc,  # Use description as context too
                    keywords=keywords,
                    sources=data_sources,
                    depth="basic",  # COST LIMITING: Reduce from "medium" to "basic"
                    max_results_per_source=5  # COST LIMITING: Reduce from 10 to 5
                )
                
                # Extract content from response
                comprehensive_content = comprehensive_response.get("content", [{}])[0]
                comprehensive_text = comprehensive_content.get("text", "{}")
                comprehensive_data = json.loads(comprehensive_text)
                
                # Store the comprehensive data
                result["comprehensive_data"] = comprehensive_data
                
                # Also organize data by source for backward compatibility
                if "data" in comprehensive_data:
                    source_data = comprehensive_data["data"]
                    for source, data in source_data.items():
                        if source not in result["sources"]:
                            result["sources"][source] = []
                        result["sources"][source].append({
                            "query": pain_point_desc,
                            "data": data
                        })
                
            except Exception as e:
                self.logger.error(f"Comprehensive research failed for pain point {pain_point_id}: {str(e)}")
                # Fall back to individual source queries if comprehensive research fails
                await self._execute_individual_source_queries(result, search_queries, data_sources)
            
            # Add metadata to result
            result["queries_executed"] = len(search_queries)
            result["sources_used"] = list(result["sources"].keys())
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Research failed for pain point {pain_point_id}: {str(e)}")
            raise
    
    async def _execute_individual_source_queries(
        self, 
        result: Dict[str, Any], 
        search_queries: List[str],
        data_sources: List[str]
    ) -> None:
        """Execute individual queries for each source as a fallback."""
        self.logger.info("Falling back to individual source queries")
        
        for query in search_queries:
            # Clean and format the query
            formatted_query = query.strip()
            if not formatted_query:
                continue
            
            # Execute searches on each data source
            if "reddit" in data_sources:
                try:
                    reddit_response = await self.research_client.call(
                        "reddit_research", 
                        query=formatted_query,
                        post_limit=10,
                        include_comments=True,
                        comment_limit=20
                    )
                    # Extract content from response
                    reddit_content = reddit_response.get("content", [{}])[0]
                    reddit_text = reddit_content.get("text", "{}")
                    reddit_data = json.loads(reddit_text)
                    
                    # Store in result
                    if "reddit" not in result["sources"]:
                        result["sources"]["reddit"] = []
                    result["sources"]["reddit"].append({
                        "query": formatted_query,
                        "data": reddit_data
                    })
                except Exception as e:
                    self.logger.error(f"Reddit search failed for query '{formatted_query}': {str(e)}")
            
            if "hn" in data_sources:
                try:
                    hn_response = await self.research_client.call(
                        "hackernews_research", 
                        query=formatted_query,
                        post_limit=10,
                        include_comments=True
                    )
                    # Extract content from response
                    hn_content = hn_response.get("content", [{}])[0]
                    hn_text = hn_content.get("text", "{}")
                    hn_data = json.loads(hn_text)
                    
                    # Store in result
                    if "hn" not in result["sources"]:
                        result["sources"]["hn"] = []
                    result["sources"]["hn"].append({
                        "query": formatted_query,
                        "data": hn_data
                    })
                except Exception as e:
                    self.logger.error(f"HN search failed for query '{formatted_query}': {str(e)}")
            
            if "google" in data_sources:
                try:
                    google_response = await self.research_client.call(
                        "google_research", 
                        query=formatted_query,
                        search_limit=10,
                        include_news=True
                    )
                    # Extract content from response
                    google_content = google_response.get("content", [{}])[0]
                    google_text = google_content.get("text", "{}")
                    google_data = json.loads(google_text)
                    
                    # Store in result
                    if "google" not in result["sources"]:
                        result["sources"]["google"] = []
                    result["sources"]["google"].append({
                        "query": formatted_query,
                        "data": google_data
                    })
                except Exception as e:
                    self.logger.error(f"Google search failed for query '{formatted_query}': {str(e)}")
            
            if "twitter" in data_sources:
                try:
                    twitter_response = await self.research_client.call(
                        "twitter_research", 
                        query=formatted_query,
                        limit=10
                    )
                    # Extract content from response
                    twitter_content = twitter_response.get("content", [{}])[0]
                    twitter_text = twitter_content.get("text", "{}")
                    twitter_data = json.loads(twitter_text)
                    
                    # Store in result
                    if "twitter" not in result["sources"]:
                        result["sources"]["twitter"] = []
                    result["sources"]["twitter"].append({
                        "query": formatted_query,
                        "data": twitter_data
                    })
                except Exception as e:
                    self.logger.error(f"Twitter search failed for query '{formatted_query}': {str(e)}")
            
            if "reviews" in data_sources:
                try:
                    reviews_response = await self.research_client.call(
                        "product_research", 
                        product=formatted_query,
                        review_limit=5
                    )
                    # Extract content from response
                    reviews_content = reviews_response.get("content", [{}])[0]
                    reviews_text = reviews_content.get("text", "{}")
                    reviews_data = json.loads(reviews_text)
                    
                    # Store in result
                    if "reviews" not in result["sources"]:
                        result["sources"]["reviews"] = []
                    result["sources"]["reviews"].append({
                        "query": formatted_query,
                        "data": reviews_data
                    })
                except Exception as e:
                    self.logger.error(f"Reviews search failed for query '{formatted_query}': {str(e)}")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text with robust error handling."""
        import re
        import json
        
        # Default fallback result
        default_result = {
            "status": "fallback",
            "message": "Failed to parse JSON from response"
        }
        
        if not text or not isinstance(text, str):
            self.logger.error("Empty or non-string response from LLM")
            return default_result
        
        # Try to find JSON in markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(.+?)\s*```'
        matches = re.search(code_block_pattern, text, re.DOTALL)
        
        if matches:
            self.logger.info("Found JSON in code block")
            json_str = matches.group(1)
        else:
            # Try to find JSON without code blocks
            self.logger.info("No code block found, trying to parse entire response as JSON")
            json_str = text.strip()
            
            # Try to find JSON object within the text using regex
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.search(json_pattern, text, re.DOTALL)
            if json_matches:
                self.logger.info("Found JSON-like structure in response")
                json_str = json_matches.group(0)
            else:
                # As a final attempt, slice from first { to last }
                first_idx = text.find('{')
                last_idx = text.rfind('}')
                if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
                    json_str = text[first_idx:last_idx+1]
        
        # Clean the JSON string
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)  # Remove trailing commas
        
        try:
            parsed_json = json.loads(json_str)
            self.logger.info("Successfully parsed JSON")
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            try:
                # Try again with more aggressive cleaning
                self.logger.info("Attempting more aggressive JSON cleaning")
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Add quotes to keys
                
                parsed_json = json.loads(json_str)
                self.logger.info("Successfully parsed JSON after aggressive cleaning")
                return parsed_json
            except Exception as e:
                self.logger.error(f"Failed to parse JSON after cleaning: {e}")
                return default_result
    
    def _generate_fallback_plan(self, pain_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback validation plan if LLM fails."""
        validation_strategies = []
        
        for i, pain_point in enumerate(pain_points):
            # Extract pain point fields
            pain_id = pain_point.get("id", f"pp{i+1}")
            description = pain_point.get("description", "Unknown pain point")
            
            # Generate search queries
            search_queries = [
                description,
                f"{description} problem",
                f"{description} solution"
            ]
            
            validation_strategies.append({
                "pain_point_id": pain_id,
                "pain_point_description": description,
                "search_queries": search_queries,
                "data_sources": ["reddit", "hn", "google", "twitter", "reviews"],
                "validation_approach": "Search for evidence supporting or refuting the pain point"
            })
        
        return {
            "validation_strategies": validation_strategies,
            "validation_approach": "Fallback validation plan using basic search queries",
            "data_sources": ["reddit", "hn", "google", "twitter", "reviews"]
        }
    
    def _generate_fallback_research(self, pain_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback research data if research fails."""
        analyzed_data = {}
        
        for i, pain_point in enumerate(pain_points):
            # Extract pain point fields
            pain_id = pain_point.get("id", f"pp{i+1}")
            description = pain_point.get("description", "Unknown pain point")
            
            # Create empty analysis result
            analyzed_data[pain_id] = {
                "fallback": {
                    "pain_point_id": pain_id,
                    "source": "fallback",
                    "relevant_evidence": [],
                    "key_insights": ["No data available"],
                    "preliminary_validation_score": 0.0,
                    "confidence_level": "low",
                    "analysis_summary": "Research failed, using fallback empty data"
                }
            }
        
        return {
            "analyzed_research": analyzed_data,
            "total_pain_points_analyzed": len(pain_points),
            "successful_analysis": 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_fallback_validation(self, pain_points: List[Dict[str, Any]], research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fallback validation if analysis fails."""
        validated_pain_points = []
        
        for i, pain_point in enumerate(pain_points):
            # Extract pain point fields
            pain_id = pain_point.get("id", f"pp{i+1}")
            description = pain_point.get("description", "Unknown pain point")
            
            # Create validated pain point with minimal data
            validated_pain_points.append({
                "id": pain_id,
                "description": description,
                "validation_score": 0.5,  # Neutral score
                "evidence": [],
                "validation_status": "inconclusive",
                "justification": "Fallback validation due to analysis failure"
            })
        
        return {
            "validated_pain_points": validated_pain_points,
            "validation_summary": f"Fallback validation of {len(validated_pain_points)} pain points",
            "market_insights": {
                "trends": [],
                "competitors": [],
                "user_sentiment": "neutral"
            },
            "confidence_score": 0.3  # Low confidence for fallback
        }
    
    def _write_stage_output(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Write stage output to manifest section using ManifestManager."""
        try:
            # Get run_id from state, if not available check for run_dir attribute
            run_id = getattr(self.state, "run_id", None)
            
            # If we have a run_dir attribute, use that directly
            run_dir = getattr(self.state, "run_dir", None)
            manifest_manager = getattr(self.state, "manifest_manager", None)
            
            if not run_id and not run_dir and not manifest_manager:
                self.logger.warning("No run_id, run_dir, or manifest_manager available. Using fallback.")
                run_id = "unknown"
            
            if not manifest_manager:
                # Determine run directory if we don't have a manifest_manager
                # project_root should be ScoutAgent/ (not scout_agent/)
                project_root = Path(__file__).resolve().parents[2]
                
                if not run_dir:
                    run_dir = project_root / "data" / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                
                manifest_path = run_dir / "run_manifest.json"
                self.logger.info(f"Creating manifest manager for path: {manifest_path}")
                manifest_manager = ManifestManager(manifest_path, create_if_missing=True)
            
            # Always use agent-prefixed stage names for multi-agent support
            agent_prefixed_stage = f"validator_{stage_name}"
            self.logger.info(f"Writing stage {agent_prefixed_stage} output to manifest")
            
            # Store the stage output with agent-prefixed stage name
            manifest_manager.store_node_output(agent_prefixed_stage, data)
            
            # Update node status to completed with agent-prefixed stage name
            manifest_manager.update_node_status(
                node_id=agent_prefixed_stage,
                state="completed"
            )
            
            self.logger.info(f"Stage {agent_prefixed_stage} output written to manifest successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to write stage {stage_name} output: {e}")
            self.logger.error(traceback.format_exc())
    
    def _get_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get the output from a stage from manifest."""
        try:
            # Get manifest_manager from state if available
            manifest_manager = getattr(self.state, "manifest_manager", None)
            
            if not manifest_manager:
                # Try to get run_id or run_dir from state
                run_id = getattr(self.state, "run_id", None)
                run_dir = getattr(self.state, "run_dir", None)
                
                if not run_id and not run_dir:
                    self.logger.warning(f"No run_id or run_dir available for getting {stage_name} output")
                    return None
                
                # Determine run directory
                project_root = Path(__file__).resolve().parents[2]
                if not run_dir:
                    run_dir = project_root / "data" / "runs" / run_id
                
                manifest_path = run_dir / "run_manifest.json"
                
                if not manifest_path.exists():
                    self.logger.warning(f"Manifest not found at: {manifest_path}")
                    return None
                
                # Create ManifestManager
                manifest_manager = ManifestManager(manifest_path)
            
            # Prefer stages section first (often richer than node output summaries)
            node_id = f"validator_{stage_name}"
            manifest = manifest_manager.get_manifest()
            if "stages" in manifest and node_id in manifest["stages"]:
                stage_entry = manifest["stages"][node_id]
                if isinstance(stage_entry, dict) and "data" in stage_entry:
                    self.logger.info(f"Found {stage_name} stage data in manifest stages.{node_id}.data")
                    return stage_entry["data"]
                self.logger.info(f"Found {stage_name} stage data in manifest stages.{node_id}")
                return stage_entry

            # Fallback: node output (may be only completion summary)
            stage_data = manifest_manager.get_node_output(node_id)
            if stage_data:
                self.logger.info(f"Found {stage_name} stage data in manifest node: {node_id}")
                # If node output looks like bookkeeping-only, try to fetch stages data anyway
                if isinstance(stage_data, dict) and set(stage_data.keys()).issubset({"completed", "failed"}):
                    stage_entry = manifest.get("stages", {}).get(node_id)
                    if isinstance(stage_entry, dict) and "data" in stage_entry:
                        self.logger.info(f"Node output is summary; using stages.{node_id}.data instead")
                        return stage_entry["data"]
                return stage_data
            
            self.logger.warning(f"No {stage_name} stage data found in manifest")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting {stage_name} output: {e}")
            return None

    def _extract_json_from_execution_output(self, raw_output: str) -> Dict[str, Any]:
        """Extract JSON data from raw execution output containing debug logs + JSON.
        
        The execution output typically contains debug logs followed by the actual JSON result.
        This method attempts to extract and parse the JSON portion.
        """
        try:
            # Method 1: The Python script should print the JSON result at the end
            # Look for the actual return value which is the last line after all debug logs
            lines = raw_output.strip().split('\n')
            
            # Try to find a line that contains valid JSON (often the last line)
            # With our fix, the complete JSON result should be printed as the last line
            for line in reversed(lines):
                line = line.strip()
                if line and line.startswith('{') and line.endswith('}'):
                    try:
                        parsed_data = json.loads(line)
                        if isinstance(parsed_data, dict):
                            # Prioritize complete results with the data field
                            if 'topic' in parsed_data and 'data' in parsed_data:
                                self.logger.debug(f"Successfully extracted complete JSON result with data field")
                                return parsed_data
                            elif 'topic' in parsed_data:
                                self.logger.debug(f"Successfully extracted JSON from line with keys: {list(parsed_data.keys())}")
                                return parsed_data
                            elif len(parsed_data) > 2:  # Any substantial result
                                self.logger.debug(f"Successfully extracted JSON object with {len(parsed_data)} keys")
                                return parsed_data
                    except json.JSONDecodeError:
                        continue
            
            # Method 2: Look for complete JSON object at the end of output
            # Use a more robust pattern that handles nested structures
            json_match = re.search(r'\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\}$', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_data = json.loads(json_str)
                    self.logger.debug(f"Successfully extracted JSON with keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'not dict'}")
                    return parsed_data
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Found JSON-like text but failed to parse: {e}")
            
            # Method 3: Look for the last substantial JSON object in the output
            # This pattern matches balanced braces more accurately
            json_pattern = r'\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\}'
            json_matches = re.findall(json_pattern, raw_output, re.DOTALL)
            
            # Try matches from the end, looking for ones with substantial content
            for json_candidate in reversed(json_matches):
                try:
                    parsed_data = json.loads(json_candidate)
                    if isinstance(parsed_data, dict):
                        # Prefer results that have the expected comprehensive_research structure
                        if 'topic' in parsed_data and 'data' in parsed_data:
                            self.logger.debug(f"Found complete comprehensive_research result with data field")
                            return parsed_data
                        elif 'topic' in parsed_data:
                            self.logger.debug(f"Found partial comprehensive_research result (missing data field)")
                            return parsed_data
                        elif len(parsed_data) > 3:  # Any substantial JSON object
                            self.logger.debug(f"Found substantial JSON object with {len(parsed_data)} keys")
                            return parsed_data
                except json.JSONDecodeError:
                    continue
            
            # Method 4: Try to extract from any line that looks like JSON
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        parsed_data = json.loads(line)
                        if isinstance(parsed_data, dict) and len(parsed_data) > 1:
                            self.logger.debug(f"Extracted JSON from individual line")
                            return parsed_data
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, return the raw output wrapped
            self.logger.warning(f"No valid JSON found in execution output, storing raw result")
            return {"raw_result": raw_output}
            
        except Exception as e:
            self.logger.error(f"Error extracting JSON from execution output: {e}")
            return {"raw_result": raw_output, "extraction_error": str(e)}

    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        try:
            from scout_agent.llm.utils import load_prompt_template
            # First try with the specific agent name
            try:
                return load_prompt_template(f"{template_name}.prompt", agent_name=self.name)
            except FileNotFoundError:
                # If not found, try without .prompt extension
                return load_prompt_template(template_name, agent_name=self.name)
        except Exception as e:
            self.logger.error(f"Error loading prompt template {template_name}: {str(e)}")
            # Return a minimal fallback prompt
            return "You are a ValidatorAgent. Please analyze the provided data and return a JSON response."


# Register the agent
from .base import register_agent
register_agent(ValidatorAgent)