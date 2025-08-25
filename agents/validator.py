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
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        self.name = "validator_agent"  # Used for prompt template loading
        self.config = get_config()
        self.research_client = None
        
        # Set preferred backend to Ollama for all tasks
        self.preferred_backend = "ollama"
        self.task_backend_preferences = {
            "validation": "ollama",
            "plan": "ollama",
            "research": "ollama",
            "analysis": "ollama"
        }
        
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
        
    def _postprocess_plan(self, plan: Dict[str, Any], input_data: ValidatorInput) -> Dict[str, Any]:
        """Process the LLM-generated plan, ensuring it has the expected structure."""
        try:
            if not isinstance(plan, dict):
                self.logger.warning("Plan is not a dictionary, returning empty plan")
                return {"dag": {"nodes": []}}
            
            # Check if the plan already has a DAG with nodes
            dag = plan.get("dag", {})
            nodes = dag.get("nodes", [])
            
            if not nodes:
                self.logger.warning("No nodes found in LLM-generated plan")
            else:
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
            
            # Execute each tool node
            completed_nodes = []
            failed_nodes = []
            
            for node in tool_nodes:
                node_id = node.get("id", "unknown")
                tool_name = node.get("tool", "")
                code = node.get("code", "")
                
                self.logger.info(f"Executing tool node {node_id} with tool {tool_name}")
                
                try:
                    # Execute the code in the sandbox
                    result = await code_executor.execute_code(
                        code=code,
                        node_id=node_id,
                        run_id=dag.get("run_id") or plan.get("run_id"),
                        run_dir_override=run_dir_override
                    )
                    
                    # Check if execution was successful
                    if result.get("success", False):
                        self.logger.info(f"Successfully executed tool node {node_id}")
                        completed_nodes.append(node_id)
                    else:
                        error = result.get("error", "Unknown error")
                        self.logger.error(f"Failed to execute tool node {node_id}: {error}")
                        failed_nodes.append(f"{node_id}: {error}")
                except Exception as e:
                    self.logger.error(f"Error executing tool node {node_id}: {str(e)}")
                    failed_nodes.append(f"{node_id}: {str(e)}")
            
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
        """Execute validation research based on the plan."""
        self.logger.info("Executing validation research based on plan")
        
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
            
            # Initialize research client if not already done
            await self._init_research_client()
            
            # Extract the validation strategies
            strategies = plan.get("validation_strategies", [])
            if not strategies:
                raise ValueError("No validation strategies found in plan")
            
            # Execute research for each pain point in parallel
            research_tasks = []
            
            for strategy in strategies:
                pain_point_id = strategy.get("pain_point_id")
                pain_point_desc = strategy.get("pain_point_description", "")
                search_queries = strategy.get("search_queries", [])
                data_sources = strategy.get("data_sources", ["reddit", "hn", "google", "twitter", "reviews"])
                
                if not pain_point_id or not pain_point_desc or not search_queries:
                    self.logger.warning(f"Incomplete strategy for pain point: {pain_point_id}")
                    continue
                
                task = self._execute_research_for_pain_point(
                    pain_point_id=pain_point_id,
                    pain_point_desc=pain_point_desc,
                    search_queries=search_queries,
                    data_sources=data_sources
                )
                research_tasks.append(task)
            
            # Execute all research tasks in parallel
            research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = {}
            for i, result in enumerate(research_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Research task {i} failed: {str(result)}")
                    # Add empty result for this pain point
                    pain_point_id = strategies[i].get("pain_point_id", f"unknown_{i}")
                    processed_results[pain_point_id] = {
                        "error": str(result),
                        "sources": {}
                    }
                else:
                    # Add successful result
                    pain_point_id = result.get("pain_point_id")
                    if pain_point_id:
                        processed_results[pain_point_id] = result
            
            # Create the research data result
            research_data = {
                "pain_point_research": processed_results,
                "total_pain_points_researched": len(processed_results),
                "successful_research": len([r for r in research_results if not isinstance(r, Exception)]),
                "failed_research": len([r for r in research_results if isinstance(r, Exception)]),
                "research_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Completed research for {research_data['successful_research']} pain points")
            
            # Write research data to manifest
            self._write_stage_output("think", research_data)
            
            # Store in agent state for act phase
            self.state.research_data = research_data
            
            return research_data
            
        except Exception as e:
            self.logger.error(f"Error in think phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback: generate basic research data
            fallback_research = self._generate_fallback_research(input_data.pain_points)
            
            # Write fallback research to manifest
            self._write_stage_output("think", fallback_research)
            
            # Store in agent state for act phase
            self.state.research_data = fallback_research
            
            return fallback_research
    
    async def act(self, input_data: ValidatorInput, plan: Dict[str, Any] = None, research_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze research data and validate pain points."""
        self.logger.info("Analyzing research data and validating pain points")
        
        try:
            # Use plan and research_data from parameters or from agent state
            if plan is None:
                plan = getattr(self.state, "plan", None)
            if research_data is None:
                research_data = getattr(self.state, "research_data", None)
                
            # If still None, try to get from manifest
            if plan is None:
                self.logger.info("No plan provided, attempting to load from manifest")
                plan = self._get_stage_output("plan")
            if research_data is None:
                self.logger.info("No research data provided, attempting to load from manifest")
                research_data = self._get_stage_output("think")
                
            if plan is None:
                raise ValueError("No validation plan available")
            if research_data is None:
                raise ValueError("No research data available")
            
            # Load prompt template
            prompt = load_prompt_template(
                template_name="act.prompt",
                agent_name="validator_agent",
                substitutions={
                    "pain_points": json.dumps(input_data.pain_points, indent=2),
                    "market_focus": input_data.target_market,
                    "research_data": json.dumps(research_data, indent=2)
                }
            )
            
            # Generate validation analysis using LLM
            self.logger.info("Calling LLM to analyze research data and validate pain points...")
            response = await self.llm_generate(prompt=prompt, task_type="analysis")
            
            # Extract JSON from response
            validation_result = self._extract_json(response)
            
            # Validate the result structure
            if not validation_result.get('validated_pain_points'):
                self.logger.warning("No validated pain points found in LLM response, using fallback")
                validation_result = self._generate_fallback_validation(input_data.pain_points, research_data)
            else:
                self.logger.info(f"Generated validation for {len(validation_result.get('validated_pain_points', []))} pain points")
            
            # Write validation result to manifest
            self._write_stage_output("act", validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in act phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback: generate basic validation
            fallback_validation = self._generate_fallback_validation(input_data.pain_points, research_data)
            
            # Write fallback validation to manifest
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
                    depth="medium",
                    max_results_per_source=10
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
            json_str = text
            
            # Try to find JSON object within the text using regex
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.search(json_pattern, text, re.DOTALL)
            if json_matches:
                self.logger.info("Found JSON-like structure in response")
                json_str = json_matches.group(0)
        
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
        pain_point_research = {}
        
        for i, pain_point in enumerate(pain_points):
            # Extract pain point fields
            pain_id = pain_point.get("id", f"pp{i+1}")
            description = pain_point.get("description", "Unknown pain point")
            
            # Create empty research result
            pain_point_research[pain_id] = {
                "pain_point_id": pain_id,
                "pain_point_description": description,
                "sources": {},
                "queries_executed": 0,
                "sources_used": [],
                "timestamp": datetime.now().isoformat(),
                "error": "Research failed, using fallback empty data"
            }
        
        return {
            "pain_point_research": pain_point_research,
            "total_pain_points_researched": len(pain_points),
            "successful_research": 0,
            "failed_research": len(pain_points),
            "research_timestamp": datetime.now().isoformat()
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
            
            # Try to get data from validator_stage_name node
            node_id = f"validator_{stage_name}"
            stage_data = manifest_manager.get_node_output(node_id)
            
            if stage_data:
                self.logger.info(f"Found {stage_name} stage data in manifest node: {node_id}")
                return stage_data
            
            # Try to get data from stages section
            manifest = manifest_manager.get_manifest()
            if "stages" in manifest and node_id in manifest["stages"]:
                stage_data = manifest["stages"][node_id]
                if "data" in stage_data:
                    self.logger.info(f"Found {stage_name} stage data in manifest stages.{node_id}.data")
                    return stage_data["data"]
            
            self.logger.warning(f"No {stage_name} stage data found in manifest")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting {stage_name} output: {e}")
            return None


# Register the agent
from .base import register_agent
register_agent(ValidatorAgent)