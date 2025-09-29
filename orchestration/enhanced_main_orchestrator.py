"""
Enhanced Main Orchestrator for the complete ScoutAgent workflow.

This orchestrator incorporates learnings from the enhanced development orchestrator
to properly handle agent dependencies, data flow, and state management.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

from scout_agent.dag.engine import DAGEngine
from scout_agent.dag.node import DAGNode, NodeStatus, NodeResult, NodeConfig, NodeType
from scout_agent.custom_logging.logger import get_logger
from scout_agent.llm.utils import initialize_llm_backends
from scout_agent.llm.manager import get_llm_manager
from scout_agent.agents.base import BaseAgent, AgentInput
from scout_agent.memory.manifest_manager import ManifestManager
from scout_agent.orchestration.agent_orchestrator import AgentOrchestrator, AgentStageConfig

logger = get_logger("enhanced_main_orchestrator")

class EnhancedMainOrchestrator(AgentOrchestrator):
    """
    Enhanced main orchestrator that properly handles the complete ScoutAgent workflow.
    
    This orchestrator addresses the key issues in the original workflow:
    1. Proper agent dependency handling
    2. Robust data extraction and passing
    3. Correct agent state management
    4. Proper DAG execution order
    """
    
    def __init__(self, run_id: str = None):
        """Initialize the enhanced main orchestrator."""
        super().__init__(run_id)
        
        # Define the complete dependency map for the workflow
        self.dependency_map = {
            "scout": [],  # No dependencies
            "screener": ["scout_act"],  # Depends on scout_act
            "validator": ["scout_act"],  # Depends on scout_act (parallel to screener)
            "gap_finder": ["screener_act", "validator_act"],  # Depends on both screener and validator
            "builder": ["gap_finder_act"],  # Depends on gap_finder_act
            "writer": ["builder_act", "gap_finder_act"]  # Depends on both builder_act and gap_finder_act for complete data
        }
        
        # Track extracted data for debugging
        self.extracted_data = {}
        
        # Track execution results
        self.results = {}
        
        logger.info("Initialized enhanced main orchestrator")
    
    def register_agent(self, agent_id: str, stage_config: AgentStageConfig, agent: BaseAgent):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: ID of the agent
            stage_config: Stage configuration for the agent
            agent: Agent instance
        """
        # Store agent and config
        self.agents[agent_id] = agent
        self.agent_configs[agent_id] = stage_config
        
        # Set up agent state with run information
        if hasattr(agent, 'state'):
            setattr(agent.state, "run_id", self.run_id)
            setattr(agent.state, "run_dir", self.run_dir)
            if hasattr(self, 'manifest_manager'):
                setattr(agent.state, "manifest_manager", self.manifest_manager)
        
        logger.info(f"Registered agent {agent_id} with enhanced state management")
    
    def _build_dag_with_dependencies(self):
        """
        Build DAG with proper agent dependencies.
        
        This method overrides the base DAG building to ensure proper execution order.
        """
        # Clear existing DAG
        self.dag_engine = DAGEngine()
        
        # Add all agent stages as nodes
        for agent_id, stage_config in self.agent_configs.items():
            for stage in stage_config.stages:
                node_id = f"{agent_id}_{stage}"
                node = DAGNode(
                    node_id=node_id,
                    name=f"{agent_id.title()} {stage.title()}",
                    node_type=NodeType.AGENT,
                    agent_name=agent_id,
                    config=NodeConfig(
                        metadata={
                            "agent_id": agent_id,
                            "stage": stage
                        }
                    )
                )
                self.dag_engine.add_node(node)
        
        # Add dependencies based on our dependency map
        for agent_id, dependencies in self.dependency_map.items():
            if not dependencies:
                continue
                
            for dependency in dependencies:
                # Find the final stage of the dependency agent
                dependency_agent = dependency.split("_")[0]
                dependency_stage = dependency.split("_")[1]
                
                # Find the first stage of the current agent
                current_agent_config = self.agent_configs.get(agent_id)
                if current_agent_config and current_agent_config.stages:
                    first_stage = current_agent_config.stages[0]
                    current_node = f"{agent_id}_{first_stage}"
                    dependency_node = f"{dependency_agent}_{dependency_stage}"
                    
                    # Add edge: dependency → current agent
                    if dependency_node in self.dag_engine.graph.nodes() and current_node in self.dag_engine.graph.nodes():
                        self.dag_engine.add_edge(dependency_node, current_node)
                        logger.info(f"Added dependency: {dependency_node} → {current_node}")
        
        # Add internal agent stage dependencies (plan → collect → think → act)
        for agent_id, stage_config in self.agent_configs.items():
            stages = stage_config.stages
            for i in range(len(stages) - 1):
                current_stage = stages[i]
                next_stage = stages[i + 1]
                current_node = f"{agent_id}_{current_stage}"
                next_node = f"{agent_id}_{next_stage}"
                
                if current_node in self.dag_engine.graph.nodes() and next_node in self.dag_engine.graph.nodes():
                    self.dag_engine.add_edge(current_node, next_node)
                    logger.info(f"Added internal dependency: {current_node} → {next_node}")
        
        # Add agent-to-agent dependencies to enforce proper order
        # Order: scout → screener → validator → gap_finder → builder → writer
        agent_order = ["scout", "screener", "validator", "gap_finder", "builder", "writer"]
        
        for i in range(len(agent_order) - 1):
            current_agent = agent_order[i]
            next_agent = agent_order[i + 1]
            
            # Only add dependency if both agents exist
            if current_agent in self.agent_configs and next_agent in self.agent_configs:
                # Get the last stage of current agent and first stage of next agent
                current_agent_stages = self.agent_configs[current_agent].stages
                next_agent_stages = self.agent_configs[next_agent].stages
                
                if current_agent_stages and next_agent_stages:
                    current_agent_last_stage = current_agent_stages[-1]  # Last stage (usually "act")
                    next_agent_first_stage = next_agent_stages[0]  # First stage (usually "plan" or "think")
                    
                    current_node = f"{current_agent}_{current_agent_last_stage}"
                    next_node = f"{next_agent}_{next_agent_first_stage}"
                    
                    if current_node in self.dag_engine.graph.nodes() and next_node in self.dag_engine.graph.nodes():
                        self.dag_engine.add_edge(current_node, next_node)
                        logger.info(f"Added agent dependency: {current_node} → {next_node}")
        
        # Get execution order
        execution_order = self.dag_engine.get_execution_order()
        logger.info(f"Enhanced DAG execution order: {execution_order}")
        
        return execution_order
    
    async def initialize(self, agent_input: AgentInput):
        """
        Initialize the enhanced orchestrator with proper DAG dependencies.
        """
        # Call base initialize but suppress its noisy DAG order logging
        original_level = getattr(self, "logger", None) and getattr(self.logger, "level", None)
        try:
            if hasattr(self, "logger"):
                try:
                    import logging as _logging
                    self.logger.setLevel(_logging.WARNING)
                except Exception:
                    pass
            await super().initialize(agent_input)
        finally:
            if hasattr(self, "logger") and original_level is not None:
                try:
                    self.logger.setLevel(original_level)
                except Exception:
                    pass
        
        # Rebuild DAG with proper enhanced dependencies
        self._build_dag_with_dependencies()
        
        # Enforce sequential execution to guarantee stage ordering across agents
        try:
            # Run nodes one-at-a-time to avoid race between plan and collect
            self.dag_engine.max_concurrent = 1
            logger.info("Set DAG execution to sequential mode (max_concurrent=1)")
        except Exception:
            pass
        
        # Ensure LLM backends are initialized before any agent stages run
        try:
            manager = get_llm_manager()
            if not manager.get_available_backends():
                logger.info("Initializing LLM backends before execution")
                await initialize_llm_backends()
            # Ensure a sane default if deepseek is unavailable
            backends = manager.get_available_backends()
            default_backend = manager.get_default_backend()
            if "deepseek" not in backends and backends:
                fb = backends[0]
                if hasattr(manager, "set_default_backend"):
                    try:
                        manager.set_default_backend(fb)
                        logger.warning(f"Deepseek unavailable; defaulting LLM backend to {fb}")
                    except Exception as _e:
                        logger.warning(f"Failed to set default backend to {fb}: {_e}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM backends: {e}")
        
        # Post-build concise validation
        validation = self.dag_engine.validate()
        if not validation.get("valid", True):
            logger.error(f"Enhanced DAG invalid: {validation['errors']}")
        else:
            logger.info("Enhanced DAG validated successfully")
            # Sanity: log predecessors for critical nodes
            try:
                g = self.dag_engine.graph
                for nid in ["scout_collect", "scout_think", "screener_think"]:
                    if nid in g.nodes():
                        preds = list(g.predecessors(nid))
                        logger.info(f"Node {nid} predecessors: {preds}")
            except Exception:
                pass
        
        logger.info("Enhanced orchestrator initialized with proper DAG dependencies")
    
    def _extract_data_from_manifest(self, node_id: str) -> Dict[str, Any]:
        """
        Extract data from manifest for a specific node.
        
        Args:
            node_id: ID of the node to extract data for
            
        Returns:
            Extracted data dictionary
        """
        if not self.manifest_manager:
            logger.warning(f"No manifest manager available for {node_id}")
            return {}
        
        try:
            manifest = self.manifest_manager.get_manifest()
            if not manifest:
                logger.warning(f"No manifest available for {node_id}")
                return {}
            
            # Try multiple extraction strategies
            data = {}
            
            # Strategy 1: Direct node output
            if "nodes" in manifest and node_id in manifest["nodes"]:
                node_data = manifest["nodes"][node_id]
                if "output" in node_data:
                    data.update(node_data["output"])
                    logger.info(f"Extracted data from nodes.{node_id}.output")
            
            # Strategy 2: Stages section
            if "stages" in manifest:
                for stage_name, stage_data in manifest["stages"].items():
                    if node_id in stage_data:
                        data.update(stage_data[node_id])
                        logger.info(f"Extracted data from stages.{stage_name}.{node_id}")
            
            # Strategy 3: Top-level entries
            if node_id in manifest:
                data.update(manifest[node_id])
                logger.info(f"Extracted data from top-level {node_id}")
            
            # Strategy 4: Agent output entries
            agent_output_key = f"{node_id}_output"
            if agent_output_key in manifest:
                data.update(manifest[agent_output_key])
                logger.info(f"Extracted data from {agent_output_key}")
            
            if data:
                logger.info(f"Successfully extracted data for {node_id}: {len(data)} keys")
            else:
                logger.warning(f"No data found for {node_id}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data for {node_id}: {e}")
            return {}
    
    def _extract_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pain points from data."""
        pain_points = []
        
        # Try various keys
        for key in ["pain_points", "top_pain_points", "validated_pain_points", "data"]:
            if key in data and isinstance(data[key], list):
                pain_points.extend(data[key])
        
        return pain_points
    
    def _extract_gaps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract gaps from data."""
        gaps = {}
        
        # Try various keys
        for key in ["gaps", "identified_market_gaps", "strategic_recommendations", "data"]:
            if key in data:
                gaps[key] = data[key]
        
        return gaps
    
    def _extract_solution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solution from data."""
        solution = {}
        
        # Try various keys
        for key in ["solution", "business_solution", "product_strategy", "data"]:
            if key in data:
                solution[key] = data[key]
        
        return solution
    
    def _get_previous_stage_data(self, agent_id: str, stage: str) -> Dict[str, Any]:
        """
        Get data from previous stages for the current agent.
        
        Args:
            agent_id: ID of the agent
            stage: Current stage name
            
        Returns:
            Dictionary of previous stage data
        """
        previous_data = {}
        
        # Get dependencies for this agent
        dependencies = self.dependency_map.get(agent_id, [])
        
        for dependency in dependencies:
            dependency_data = self._get_stage_result(dependency)
            if dependency_data:
                previous_data[dependency] = dependency_data
        
        return previous_data
    
    def _get_stage_result(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """
        Get result from a specific stage.
        
        Args:
            stage_id: ID of the stage (e.g., "scout_plan")
            
        Returns:
            Stage result or None if not found
        """
        if stage_id in self.results:
            return self.results[stage_id]
        
        # Try to get from manifest
        if self.manifest_manager:
            return self.manifest_manager.get_node_output(stage_id)
        
        return None
    
    async def execute(self, agent_input: AgentInput):
        """
        Execute the enhanced orchestrator with proper DAG dependencies.
        
        Args:
            agent_input: Input for the orchestration
            
        Returns:
            Execution results
        """
        logger.info("=== DOCKER DEBUG: STARTING ENHANCED ORCHESTRATOR ===")
        logger.info("Starting enhanced orchestrator execution")
        
        # Initialize with proper DAG dependencies
        await self.initialize(agent_input)
        
        # Execute using the DAG engine with the same pattern as AgentOrchestrator
        execution_state = await self.dag_engine.execute(self._execute_node)
        
        logger.info("Enhanced orchestrator execution completed")
        return execution_state
    
    async def _execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """
        Execute a single node in the DAG.
        
        This is the executor function passed to the DAG engine.
        
        Args:
            node: The node to execute
            inputs: Inputs from predecessor nodes
            
        Returns:
            NodeResult with execution results
        """
        from scout_agent.dag.node import NodeResult, NodeStatus
        from datetime import datetime
        
        start_time = datetime.now()
        
        try:
            if node.node_type == NodeType.AGENT:
                return await self._execute_agent_node(node, inputs)
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
        except Exception as e:
            logger.error(f"Error executing node {node.node_id}: {e}")
            end_time = datetime.now()
            return NodeResult(
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=end_time
            )
    
    async def _execute_agent_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """
        Execute an agent node.
        
        Args:
            node: The agent node to execute
            inputs: Inputs from predecessor nodes
            
        Returns:
            NodeResult with execution results
        """
        from scout_agent.dag.node import NodeResult, NodeStatus
        from datetime import datetime
        
        start_time = datetime.now()
        config = node.config
        
        # Get the agent from metadata
        agent_id = config.metadata.get("agent_id")
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Get the method to call
        stage = config.metadata.get("stage")
        method_name = config.metadata.get("method", stage)
        
        if not hasattr(agent, method_name):
            raise ValueError(f"Agent {agent_id} does not have method {method_name}")
        
        method = getattr(agent, method_name)
        
        # Prepare agent input
        agent_input = self.agent_input
        
        # Call the agent method with appropriate arguments
        if stage == "plan":
            result = await method(agent_input, run_id=self.run_id)
        elif stage == "collect":
            # Get plan result for collect stage
            plan_result = None
            # Prefer predecessor inputs (direct from DAG)
            predecessor_key = f"{agent_id}_plan"
            if predecessor_key in inputs and isinstance(inputs[predecessor_key], dict):
                # Some nodes propagate outputs under 'result' key; support both shapes
                candidate = inputs[predecessor_key].get("result") or inputs[predecessor_key]
                if isinstance(candidate, dict) and ("dag" in candidate or "nodes" in candidate):
                    plan_result = candidate
                    logger.info("Using predecessor inputs for collect stage plan_result")
            # Fallback to cached plan results
            if plan_result is None and hasattr(self, '_plan_results') and agent_id in self._plan_results:
                plan_result = self._plan_results[agent_id]
                logger.info(f"Using cached plan result for collect stage: {len(plan_result.get('dag', {}).get('nodes', []))} tool nodes")
            # Fallback to manifest
            if plan_result is None and self.manifest_manager:
                mm_plan = self.manifest_manager.get_node_output(f"{agent_id}_plan")
                if isinstance(mm_plan, dict):
                    plan_result = mm_plan
                    logger.info("Using manifest fallback for collect stage plan_result")
            # As a last resort, try to generate the plan now (self-heal)
            if plan_result is None:
                logger.warning(f"Collect stage missing plan_result. Inputs keys: {list(inputs.keys())}")
                try:
                    if hasattr(agent, "plan"):
                        logger.info("Self-heal: generating plan inline for collect stage")
                        generated = await agent.plan(agent_input, run_id=self.run_id)
                        plan_result = generated if isinstance(generated, dict) else None
                        if plan_result:
                            if not hasattr(self, '_plan_results'):
                                self._plan_results = {}
                            self._plan_results[agent_id] = plan_result
                            if self.manifest_manager:
                                self.manifest_manager.store_node_output(f"{agent_id}_plan", plan_result)
                            logger.info("Self-heal: plan generated and stored for collect stage")
                except Exception as e:
                    logger.error(f"Self-heal plan generation failed: {e}")
            
            # Always delegate to the agent's collect method to execute tools via its own services
            result = await method(plan=plan_result, run_id=self.run_id)
            
            # Optionally extract tool results from manifest for downstream stages (best-effort)
            try:
                if plan_result:
                    self._extract_tool_results_from_manifest(plan_result)
            except Exception as _e:
                logger.debug(f"Skipping tool result extraction: {_e}")
        elif stage == "think":
            # Check if this agent has a collect stage - only get collect data if it does
            agent_config = self.agent_configs.get(agent_id)
            has_collect_stage = agent_config and "collect" in agent_config.stages
            
            if agent_id == "screener":
                # For ScreenerAgent, get the pain points from ScoutAgent's act stage
                scout_act_data = None
                
                # Try to get scout_act data from message service first
                try:
                    scout_act_data = self.message_service.consume_stage_input(
                        workflow_id=self.run_id,
                        stage_id="scout_act"
                    )
                    if scout_act_data:
                        logger.info("Using message service scout_act data for screener think stage")
                except Exception as e:
                    logger.warning(f"Failed to get scout_act data from message service: {e}")
                
                # If not found in message service, try direct stage output
                if not scout_act_data:
                    scout_act_data = self._stage_outputs.get("scout_act")
                    if scout_act_data:
                        logger.info("Using direct stage output for scout_act data in screener think stage")
                
                # If still not found, try manifest
                if not scout_act_data and self.manifest_manager:
                    scout_act_data = self.manifest_manager.get_node_output("scout_act")
                    if scout_act_data:
                        logger.info("Using manifest fallback for scout_act data in screener think stage")
                
                # Extract pain points from scout_act data
                pain_points = []
                if scout_act_data:
                    # Extract pain points from different possible formats
                    if isinstance(scout_act_data, dict):
                        if "pain_points" in scout_act_data:
                            pain_points = scout_act_data["pain_points"]
                        elif "result" in scout_act_data and isinstance(scout_act_data["result"], dict):
                            pain_points = scout_act_data["result"].get("pain_points", [])
                    
                    logger.info(f"Found {len(pain_points)} pain points for screener from scout_act")
                    
                    # Set pain points as input data for screener
                    self.agent_input.data = pain_points
                
                    # Create a ScreenerInput object from the AgentInput
                    from scout_agent.agents.screener import ScreenerInput
                    screener_input = ScreenerInput.from_agent_input(self.agent_input)
                    result = await method(screener_input)
                else:
                    # This should never happen with proper DAG dependencies, but keep as fallback
                    from scout_agent.agents.screener import ScreenerInput
                    screener_input = ScreenerInput(
                        pain_points=[],
                        target_market=self.agent_input.context.get("target_market", ""),
                        top_k=self.agent_input.context.get("top_k", 5)
                    )
                    result = await method(screener_input)
            elif agent_id == "builder":
                # For BuilderAgent, get the gap finder output from GapFinderAgent's act stage
                gap_finder_act_data = None
                
                # Try to get gap_finder_act data from message service first
                try:
                    gap_finder_act_data = self.message_service.consume_stage_input(
                        workflow_id=self.run_id,
                        stage_id="gap_finder_act"
                    )
                    if gap_finder_act_data:
                        logger.info("Using message service gap_finder_act data for builder think stage")
                except Exception as e:
                    logger.warning(f"Failed to get gap_finder_act data from message service: {e}")
                
                # If not found in message service, try direct stage output
                if not gap_finder_act_data:
                    gap_finder_act_data = self._stage_outputs.get("gap_finder_act")
                    if gap_finder_act_data:
                        logger.info("Using direct stage output for gap_finder_act data in builder think stage")
                
                # If still not found, try manifest
                if not gap_finder_act_data and self.manifest_manager:
                    gap_finder_act_data = self.manifest_manager.get_node_output("gap_finder_act")
                    if gap_finder_act_data:
                        logger.info("Using manifest fallback for gap_finder_act data in builder think stage")
                
                # Extract gaps from gap_finder_act data
                gaps = []
                if gap_finder_act_data:
                    # Extract gaps from different possible formats
                    if isinstance(gap_finder_act_data, dict):
                        if "identified_market_gaps" in gap_finder_act_data:
                            gaps = gap_finder_act_data["identified_market_gaps"]
                        elif "result" in gap_finder_act_data and isinstance(gap_finder_act_data["result"], dict):
                            gaps = gap_finder_act_data["result"].get("identified_market_gaps", [])
                    
                    logger.info(f"Found {len(gaps)} gaps for builder from gap_finder_act")
                    
                    # Set gaps as input data for builder
                    self.agent_input.data = gaps
                
                    # Create a BuilderInput object from the AgentInput
                    from scout_agent.agents.builder import BuilderInput
                    builder_input = BuilderInput.from_agent_input(self.agent_input)
                    result = await method(builder_input)
                else:
                    # This should never happen with proper DAG dependencies, but keep as fallback
                    from scout_agent.agents.builder import BuilderInput
                    builder_input = BuilderInput(
                        market_gaps=[],
                        market_context=self.agent_input.context.get("target_market", "")
                    )
                    result = await method(builder_input)
            else:
                # For other agents, just call the method with agent_input
                result = await method(agent_input)
        elif stage == "act":
            # Get think result for act stage
            think_result = None
            if hasattr(self, '_think_results') and agent_id in self._think_results:
                think_result = self._think_results[agent_id]
                logger.info(f"Using think result for act stage")
            
            if agent_id == "scout":
                # ScoutAgent.act expects (agent_input, plan, thoughts)
                result = await method(agent_input, think_result, think_result)
            elif agent_id == "builder":
                # BuilderAgent.act expects (agent_input, think_result)
                result = await method(agent_input, think_result)
            else:
                # For other agents, just call the method with agent_input and think_result
                result = await method(agent_input, think_result)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Store the result for later stages
        if stage == "plan":
            if not hasattr(self, '_plan_results'):
                self._plan_results = {}
            self._plan_results[agent_id] = result
        elif stage == "think":
            if not hasattr(self, '_think_results'):
                self._think_results = {}
            self._think_results[agent_id] = result
        
        # Convert result to dictionary if it's a Pydantic model or has to_dict method
        serializable_result = result
        if hasattr(result, 'to_dict'):
            serializable_result = result.to_dict()
        elif hasattr(result, 'model_dump'):
            serializable_result = result.model_dump()
        elif hasattr(result, 'dict'):
            serializable_result = result.dict()
        
        # Store in manifest
        if self.manifest_manager:
            self.manifest_manager.store_node_output(node.node_id, serializable_result)
        
        end_time = datetime.now()
        return NodeResult(
            success=True,
            output=serializable_result,
            start_time=start_time,
            end_time=end_time
        )
    
    async def _execute_tool_nodes_with_code_executor(self, plan_result: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """
        Execute tool nodes using the code executor.
        
        Args:
            plan_result: The plan result containing tool nodes
            agent_id: The agent ID
            
        Returns:
            Execution results
        """
        if not self.code_executor:
            logger.warning("No code executor available")
            return {"completed": [], "failed": []}
        
        dag_data = plan_result.get("dag", {})
        nodes = dag_data.get("nodes", [])
        
        completed = []
        failed = []
        
        for node in nodes:
            if node.get("type") == "tool":
                tool_node_id = node.get("id")
                code = node.get("code")
                
                if code:
                    try:
                        logger.info(f"Executing tool node: {tool_node_id}")
                        result = await self.code_executor.execute_code(code, "python")
                        completed.append(tool_node_id)
                        logger.info(f"Successfully executed tool node: {tool_node_id}")
                    except Exception as e:
                        logger.error(f"Failed to execute tool node {tool_node_id}: {e}")
                        failed.append(tool_node_id)
                else:
                    logger.warning(f"No code found for tool node: {tool_node_id}")
                    failed.append(tool_node_id)
        
        return {"completed": completed, "failed": failed}
    
    def _extract_tool_results_from_manifest(self, plan_result: Dict[str, Any]) -> None:
        """
        Extract tool results from manifest after collect stage completes.
        
        Args:
            plan_result: The plan result containing tool node information
        """
        if not self.manifest_manager or not plan_result:
            return
            
        dag_data = plan_result.get("dag", {})
        nodes = dag_data.get("nodes", [])
        
        manifest = self.manifest_manager.get_manifest()
        stages = manifest.get("stages", {})
        
        for node in nodes:
            if node.get("type") == "tool":
                tool_node_id = node.get("id")
                if tool_node_id:
                    # Look for tool result in manifest - try multiple locations
                    tool_data = None
                    
                    # Primary location: stages[tool_node_id].data
                    tool_data = stages.get(tool_node_id, {}).get("data")
                    
                    # Fallback 1: Check if data is stored directly in stages[tool_node_id]
                    if not tool_data:
                        tool_data = stages.get(tool_node_id)
                        if tool_data and isinstance(tool_data, dict) and "data" not in tool_data:
                            # Data is stored directly, not nested under 'data' key
                            pass
                        else:
                            tool_data = None
                    
                    # Fallback 2: Check for alternative storage patterns
                    if not tool_data:
                        # Try looking for the data in different manifest structures
                        for stage_name, stage_data in stages.items():
                            if isinstance(stage_data, dict) and tool_node_id in str(stage_data):
                                logger.debug(f"Found potential match in stage {stage_name} for {tool_node_id}")
                                tool_data = stage_data.get("data")
                                if tool_data:
                                    break
                    
                    # Fallback 3: Debug logging for Docker troubleshooting
                    if not tool_data:
                        logger.warning(f"No tool result found in manifest for {tool_node_id}")
                        logger.debug(f"Available stages: {list(stages.keys())}")
                        logger.debug(f"Looking for tool_node_id: {tool_node_id}")
                        logger.debug(f"Stages structure: {json.dumps({k: type(v).__name__ for k, v in stages.items()}, indent=2)}")
                    
                    if tool_data:
                        if not hasattr(self, '_tool_results'):
                            self._tool_results = {}
                        self._tool_results[tool_node_id] = tool_data
                        logger.info(f"Extracted tool result for direct passing: {tool_node_id}")
                    else:
                        logger.warning(f"No tool result found in manifest for {tool_node_id} after all fallback attempts")
        
        logger.info(f"Extracted {len(getattr(self, '_tool_results', {}))} tool results for direct passing")
    
    async def execute_agent_stage(self, agent_id: str, stage: str, method, plan_result=None, think_result=None):
        """
        Enhanced execution of agent stages with proper data flow handling.
        
        This method overrides the base implementation to provide robust data extraction
        and agent-specific input processing.
        """
        logger.info(f"=== DOCKER DEBUG: EXECUTING {agent_id} {stage} STAGE ===")
        logger.info(f"Executing {agent_id} {stage} stage with enhanced data flow")
        
        # Handle different stages with appropriate data flow
        if stage == "plan":
            return await self._execute_plan_stage(agent_id, method)
        elif stage == "collect":
            return await self._execute_collect_stage(agent_id, method, plan_result)
        elif stage == "think":
            return await self._execute_think_stage(agent_id, method)
        elif stage == "act":
            return await self._execute_act_stage(agent_id, method, think_result)
        else:
            logger.error(f"Unknown stage: {stage}")
            return {"error": f"Unknown stage: {stage}"}
    
    async def _execute_agent_stage(self, agent_id: str, stage: str, method, plan_result=None, think_result=None):
        """
        Internal method that calls the enhanced execution.
        """
        return await self.execute_agent_stage(agent_id, stage, method, plan_result, think_result)
    
    async def _execute_plan_stage(self, agent_id: str, method):
        """Execute plan stage with proper input handling."""
        logger.info(f"Executing plan stage for {agent_id}")
        
        # For plan stages, use the original agent input
        if agent_id in ["scout", "validator", "gap_finder"]:
            return await method(self.agent_input)
        else:
            # Other agents don't have plan stages
            return {"error": f"Agent {agent_id} does not have a plan stage"}
    
    async def _execute_collect_stage(self, agent_id: str, method, plan_result):
        """Execute collect stage with plan result."""
        logger.info(f"Executing collect stage for {agent_id}")
        
        if agent_id in ["scout", "validator", "gap_finder"]:
            return await method(self.agent_input, plan_result)
        else:
            return {"error": f"Agent {agent_id} does not have a collect stage"}
    
    async def _execute_think_stage(self, agent_id: str, method):
        """Execute think stage with proper data extraction."""
        logger.info(f"Executing think stage for {agent_id}")
        
        # Get dependencies for this agent
        dependencies = self.dependency_map.get(agent_id, [])
        
        if not dependencies:
            # No dependencies, use original input
            return await method(self.agent_input)
        
        # Extract data from dependencies
        extracted_data = self._extract_dependency_data(dependencies)
        
        # Process data based on agent type
        processed_input = self._process_agent_input(agent_id, extracted_data)
        
        return await method(processed_input)
    
    async def _execute_act_stage(self, agent_id: str, method, think_result):
        """Execute act stage with think result and proper data flow."""
        logger.info(f"Executing act stage for {agent_id}")
        
        # Get dependencies for this agent
        dependencies = self.dependency_map.get(agent_id, [])
        
        if not dependencies:
            # No dependencies, use original input and think result
            return await method(self.agent_input, think_result)
        
        # Extract data from dependencies
        extracted_data = self._extract_dependency_data(dependencies)
        
        # Process data based on agent type
        processed_input = self._process_agent_input(agent_id, extracted_data)
        
        return await method(processed_input, think_result)
    
    def _extract_dependency_data(self, dependencies: List[str]) -> Dict[str, Any]:
        """
        Extract data from multiple dependencies using robust strategies.
        
        Args:
            dependencies: List of dependency node IDs
            
        Returns:
            Dictionary of extracted data
        """
        extracted_data = {}
        
        for dependency in dependencies:
            logger.info(f"Extracting data from dependency: {dependency}")
            
            # Try multiple extraction strategies
            data = self._extract_data_from_manifest(dependency)
            
            if data:
                extracted_data[dependency] = data
                logger.info(f"Successfully extracted data from {dependency}")
            else:
                logger.warning(f"No data found for dependency {dependency}")
        
        return extracted_data
    
    def _extract_data_from_manifest(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract data from manifest using multiple strategies.
        
        Args:
            node_id: ID of the node to extract data from
            
        Returns:
            Extracted data or None if not found
        """
        if not hasattr(self, 'manifest_manager') or not self.manifest_manager:
            logger.error("No manifest manager available")
            return None
            
        # Strategy 1: Direct node output
        data = self.manifest_manager.get_node_output(node_id)
        if data:
            logger.info(f"Found data for {node_id} in direct node output")
            return data
            
        # Strategy 2: Stages section
        manifest = self.manifest_manager.get_manifest()
        if "stages" in manifest and node_id in manifest["stages"]:
            stage_data = manifest["stages"][node_id]
            if "data" in stage_data:
                logger.info(f"Found data for {node_id} in stages.{node_id}.data")
                return stage_data["data"]
                
        # Strategy 3: Look for custom top-level entries
        if node_id in manifest:
            logger.info(f"Found data for {node_id} in top-level manifest entry")
            return manifest[node_id]
            
        # Strategy 4: Look for agent_output entries
        agent_prefix = node_id.split("_")[0]
        if f"{agent_prefix}_output" in manifest:
            logger.info(f"Found data in {agent_prefix}_output")
            return manifest[f"{agent_prefix}_output"]
            
        logger.warning(f"No data found for {node_id} in manifest")
        return None
    
    def _process_agent_input(self, agent_id: str, extracted_data: Dict[str, Any]) -> AgentInput:
        """
        Process extracted data into appropriate agent input.
        
        Args:
            agent_id: ID of the agent
            extracted_data: Extracted data from dependencies
            
        Returns:
            Processed AgentInput
        """
        # Create a copy of the original input
        processed_input = AgentInput(
            data=self.agent_input.data.copy() if self.agent_input.data else {},
            metadata=self.agent_input.metadata.copy() if self.agent_input.metadata else {},
            context=self.agent_input.context.copy() if self.agent_input.context else {}
        )
        
        # Process based on agent type
        if agent_id == "screener":
            self._process_screener_input(processed_input, extracted_data)
        elif agent_id == "validator":
            self._process_validator_input(processed_input, extracted_data)
        elif agent_id == "gap_finder":
            self._process_gap_finder_input(processed_input, extracted_data)
        elif agent_id == "builder":
            self._process_builder_input(processed_input, extracted_data)
        elif agent_id == "writer":
            self._process_writer_input(processed_input, extracted_data)
        
        return processed_input
    
    def _process_screener_input(self, input_data: AgentInput, extracted_data: Dict[str, Any]):
        """Process input data for screener agent."""
        scout_act_data = extracted_data.get("scout_act", {})
        pain_points = self._extract_pain_points(scout_act_data)
        
        if pain_points:
            logger.info(f"Found {len(pain_points)} pain points for screener")
            input_data.data = pain_points
        else:
            logger.warning("No pain points found for screener")
    
    def _process_validator_input(self, input_data: AgentInput, extracted_data: Dict[str, Any]):
        """Process input data for validator agent."""
        scout_act_data = extracted_data.get("scout_act", {})
        pain_points = self._extract_pain_points(scout_act_data)
        
        if pain_points:
            logger.info(f"Found {len(pain_points)} pain points for validator")
            input_data.data = pain_points
        else:
            logger.warning("No pain points found for validator")
    
    def _process_gap_finder_input(self, input_data: AgentInput, extracted_data: Dict[str, Any]):
        """Process input data for gap finder agent."""
        screener_act_data = extracted_data.get("screener_act", {})
        validator_act_data = extracted_data.get("validator_act", {})
        
        # Extract validated pain points from validator
        validated_pain_points = self._extract_validated_pain_points(validator_act_data)
        
        if validated_pain_points:
            logger.info(f"Found {len(validated_pain_points)} validated pain points for gap finder")
            input_data.data = validated_pain_points
        else:
            logger.warning("No validated pain points found for gap finder")
    
    def _process_builder_input(self, input_data: AgentInput, extracted_data: Dict[str, Any]):
        """Process input data for builder agent."""
        gap_finder_act_data = extracted_data.get("gap_finder_act", {})
        gaps = self._extract_gaps(gap_finder_act_data)
        
        if gaps:
            logger.info(f"Found {len(gaps)} gaps for builder")
            input_data.data = gaps
        else:
            logger.warning("No gaps found for builder")
    
    def _process_writer_input(self, input_data: AgentInput, extracted_data: Dict[str, Any]):
        """Process input data for writer agent."""
        builder_act_data = extracted_data.get("builder_act", {})
        
        if builder_act_data:
            logger.info(f"Found builder_act data for writer: {len(str(builder_act_data))} chars")
            # Pass the full builder_act data as the main data
            input_data.data = builder_act_data
            
            # Also add gap_finder data to context if available
            gap_finder_data = extracted_data.get("gap_finder_act", {})
            if gap_finder_data:
                if not input_data.context:
                    input_data.context = {}
                input_data.context["gap_finder_output"] = gap_finder_data
                logger.info(f"Added gap_finder data to writer context: {len(str(gap_finder_data))} chars")
        else:
            logger.warning("No builder_act data found for writer")
    
    def _extract_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pain points from various data formats."""
        pain_points = []
        
        if isinstance(data, dict):
            # Direct pain_points field
            if "pain_points" in data:
                pain_points = data["pain_points"]
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "pain_points" in data["result"]:
                    pain_points = data["result"]["pain_points"]
            # Try to find any array field that might contain pain points
            elif not pain_points:
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        if any(pp.get("description") for pp in value):
                            pain_points = value
                            break
        
        return pain_points
    
    def _extract_validated_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract validated pain points from validator output."""
        validated_pain_points = []
        
        if isinstance(data, dict):
            if "validated_pain_points" in data:
                validated_pain_points = data["validated_pain_points"]
            elif "result" in data and isinstance(data["result"], dict):
                if "validated_pain_points" in data["result"]:
                    validated_pain_points = data["result"]["validated_pain_points"]
        
        return validated_pain_points
    
    def _extract_gaps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract gaps from gap finder output."""
        gaps = []
        
        if isinstance(data, dict):
            if "identified_market_gaps" in data:
                gaps = data["identified_market_gaps"]
            elif "result" in data and isinstance(data["result"], dict):
                if "identified_market_gaps" in data["result"]:
                    gaps = data["result"]["identified_market_gaps"]
        
        return gaps
    
    def _extract_solution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract solution from builder output."""
        solution = {}
        
        if isinstance(data, dict):
            if "business_solution_summary" in data:
                solution = data
            elif "result" in data and isinstance(data["result"], dict):
                solution = data["result"]
        
        return solution
