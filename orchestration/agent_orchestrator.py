"""
Agent Orchestrator for ScoutAgent.

This module provides a specialized orchestrator that manages agent lifecycles
and integrates with the DAG engine for workflow execution. It implements the
"Option A" approach where the plan stage dynamically modifies the DAG.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from scout_agent.dag.engine import DAGEngine
from scout_agent.dag.node import DAGNode, NodeStatus, NodeResult, NodeConfig, NodeType
from scout_agent.custom_logging.logger import get_logger
from scout_agent.agents.base import BaseAgent
from scout_agent.memory.manifest_manager import ManifestManager


class AgentStageConfig:
    """Configuration for agent stages."""
    
    def __init__(self, stages: List[str]):
        """
        Initialize agent stage configuration.
        
        Args:
            stages: List of stages this agent supports (e.g., ["plan", "collect", "think", "act"])
        """
        self.stages = stages


class AgentOrchestrator:
    """
    Orchestrator for agent-based workflows using DAG engine.
    
    This orchestrator manages agent lifecycles and integrates with the DAG engine
    for workflow execution. It implements the "Option A" approach where the plan
    stage dynamically modifies the DAG.
    """
    
    def __init__(self, run_id: str = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            run_id: Optional run ID for this orchestration
        """
        self.run_id = run_id or f"scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = get_logger(f"AgentOrchestrator[{self.run_id}]")
        
        # Core components
        self.dag_engine = DAGEngine()
        self.agents = {}  # agent_id -> agent instance
        self.agent_configs = {}  # agent_id -> AgentStageConfig
        
        # State tracking
        self.run_dir = None
        self.manifest_manager = None
        
        # Direct data passing between stages
        self._stage_outputs = {}  # node_id -> output data
        self._tool_results = {}  # tool_node_id -> tool result data
        
        self.logger.info(f"AgentOrchestrator initialized with run_id: {self.run_id}")
    
    def register_agent(self, agent_id: str, config: AgentStageConfig, agent: BaseAgent = None) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: Agent identifier
            config: Agent stage configuration
            agent: Agent instance (optional)
        """
        if agent is None:
            raise ValueError("Agent instance must be provided")
            
        self.agents[agent_id] = agent
        self.agent_configs[agent_id] = config
        
        self.logger.info(
            f"Registered agent {agent} with stages: {config.stages}"
        )
    
    def _get_direct_collect_data(self) -> Dict[str, Any]:
        """
        Get collect data from direct tool results (primary method).
        
        Returns:
            Aggregated tool results from direct storage
        """
        collect_data = {}
        threads = []
        comments = []
        
        # Aggregate all tool results from direct storage
        for tool_id, tool_result in self._tool_results.items():
            if isinstance(tool_result, dict):
                tool_threads = tool_result.get("threads", [])
                tool_comments = tool_result.get("comments", [])
                
                if tool_threads:
                    threads.extend(tool_threads)
                if tool_comments:
                    comments.extend(tool_comments)
        
        if threads or comments:
            collect_data = {
                "threads": threads,
                "comments": comments,
                "total_threads": len(threads),
                "total_comments": len(comments)
            }
            self.logger.info(f"Direct tool results: {len(threads)} threads, {len(comments)} comments")
        
        return collect_data

    def _aggregate_tool_results(self) -> Dict[str, Any]:
        """
        Aggregate tool results from manifest for passing to agent stages (fallback method).
        
        Returns:
            Aggregated tool results
        """
        if not self.manifest_manager:
            return {}
        
        manifest = self.manifest_manager.get_manifest()
        stages = manifest.get("stages", {})
        
        # Look for collect tool results
        collect_data = {}
        threads = []
        comments = []
        
        for stage_id, stage_data in stages.items():
            if "collect" in stage_id.lower() and isinstance(stage_data, dict):
                data = stage_data.get("data", {})
                if isinstance(data, dict):
                    stage_threads = data.get("threads", [])
                    stage_comments = data.get("comments", [])
                    
                    if stage_threads:
                        threads.extend(stage_threads)
                    if stage_comments:
                        comments.extend(stage_comments)
        
        if threads or comments:
            collect_data = {
                "threads": threads,
                "comments": comments,
                "total_threads": len(threads),
                "total_comments": len(comments)
            }
            self.logger.info(f"Aggregated tool results: {len(threads)} threads, {len(comments)} comments")
        
        return collect_data

    def _extract_tool_results_from_manifest(self, plan_result: Dict[str, Any]) -> None:
        """
        Extract tool results from manifest after collect stage completes.
        This populates _tool_results for direct passing to subsequent stages.
        
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
                    # Look for tool result in manifest
                    tool_data = stages.get(tool_node_id, {}).get("data")
                    if tool_data:
                        self._tool_results[tool_node_id] = tool_data
                        self.logger.info(f"Extracted tool result for direct passing: {tool_node_id}")
                        if isinstance(tool_data, dict) and "threads" in tool_data:
                            self.logger.info(f"Found {len(tool_data['threads'])} threads in {tool_node_id}")
                    else:
                        self.logger.warning(f"No tool result found in manifest for {tool_node_id}")
        
        self.logger.info(f"Extracted {len(self._tool_results)} tool results for direct passing")

    def setup_run_directory(self) -> Path:
        """
        Set up the run directory for this orchestration.
        
        Returns:
            Path to the run directory
        """
        # Project root at ScoutAgent/ (not scout_agent/)
        root = Path(__file__).resolve().parents[2]
        run_dir = root / "data" / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_dir = run_dir
        self.manifest_manager = ManifestManager(run_dir / "run_manifest.json", create_if_missing=True)
        
        self.logger.info(f"Set up run directory: {run_dir}")
        return run_dir
    
    def _create_agent_stage_node(self, agent_id: str, stage: str, dependencies: List[str] = None) -> DAGNode:
        """
        Create a DAG node for an agent stage.
        
        Args:
            agent_id: Agent identifier
            stage: Stage name (plan, collect, think, act)
            dependencies: List of node IDs this node depends on
            
        Returns:
            DAGNode for the agent stage
        """
        node_id = f"{agent_id}_{stage}"
        agent_name = agent_id
        if isinstance(agent_id, str):
            agent_name = agent_id.capitalize()
            
        # Create a proper NodeConfig object
        node_config = NodeConfig()
        
        # Create the node with proper NodeConfig
        node = DAGNode(
            node_id=node_id,
            name=f"{agent_name} {stage.capitalize()}",
            node_type=NodeType.AGENT,
            config=node_config,  # Use NodeConfig object
            dependencies=dependencies or []
        )
        
        # Add agent-specific metadata to the config
        node.config.metadata = {
            "agent_id": agent_id,
            "stage": stage,
            "method": stage
        }
        
        return node
    
    def build_initial_dag(self) -> None:
        """
        Build the initial DAG with agent stage nodes.
        
        This creates the high-level structure with agent stages,
        but tool nodes will be added dynamically during execution.
        """
        # Clear any existing DAG
        self.dag_engine.graph.clear()
        
        # Track node dependencies
        last_nodes = {}  # agent_id -> last node_id
        
        # For each agent, add its stages in sequence
        for agent_id, config in self.agent_configs.items():
            prev_node_id = None
            
            for stage in config.stages:
                # For the first stage of each agent, check if it depends on other agents
                dependencies = []
                if prev_node_id:
                    dependencies.append(prev_node_id)
                
                # Create the node for this stage
                node = self._create_agent_stage_node(agent_id, stage, dependencies)
                self.dag_engine.add_node(node)
                
                # If there are dependencies from the previous node, add edges
                for dep in dependencies:
                    self.dag_engine.add_edge(dep, node.node_id)
                
                # Update tracking
                prev_node_id = node.node_id
                last_nodes[agent_id] = node.node_id
        
        # Validate the DAG
        validation = self.dag_engine.validate()
        if not validation["valid"]:
            self.logger.error(f"Invalid DAG: {validation['errors']}")
            raise ValueError(f"Invalid DAG: {validation['errors']}")
        
        self.logger.info(f"Built initial DAG with {len(self.dag_engine.graph.nodes)} nodes")
    
    def build_workflow_dag(self, workflow_config: Dict[str, Any]) -> None:
        """
        Build a DAG for a specific workflow based on configuration.
        
        Args:
            workflow_config: Configuration for the workflow
        """
        # Set up basic agent stages
        self.build_initial_dag()
        
        # Store workflow config in manifest
        if self.manifest_manager:
            self.manifest_manager.store_workflow_config(workflow_config)
        
        self.logger.info(f"Built workflow DAG with config: {workflow_config}")
    
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
        start_time = datetime.now()
        
        try:
            if node.node_type == NodeType.AGENT:
                return await self._execute_agent_node(node, inputs)
            elif node.node_type == NodeType.FUNCTION:  # Tool nodes handled by agent collect
                raise ValueError(f"Tool nodes should be executed by agent collect() method, not orchestrator")
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
        except Exception as e:
            self.logger.error(f"Error executing node {node.node_id}: {e}")
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
            # For collect, pass the plan result with tool nodes
            plan_result = None
            if hasattr(self, '_plan_results') and agent_id in self._plan_results:
                plan_result = self._plan_results[agent_id]
                self.logger.info(f"Passing plan with tool nodes to collect stage")
                result = await method(plan=plan_result, run_id=self.run_id)
                
                # After collect completes, extract tool results from manifest for direct passing
                self._extract_tool_results_from_manifest(plan_result)
            else:
                self.logger.warning(f"No plan result found for agent {agent_id}")
                result = await method(self.agent_input)
        elif stage == "think":
            # Get collect data from direct tool results first, then fallback to aggregated results
            collect_data = self._get_direct_collect_data()
            if not collect_data:
                collect_data = self._aggregate_tool_results()
                self.logger.info("Using aggregated tool results as fallback for think stage")
            else:
                self.logger.info("Using direct tool results for think stage")
                
            if collect_data:
                setattr(agent.state, 'collect_data', collect_data)
                self.logger.info(f"Set collect_data in agent state for think: {len(collect_data.get('threads', []))} threads")
            result = await method(self.agent_input)
        elif stage == "act":
            # Get plan and think results for act stage
            plan_result = None
            think_result = None
            if hasattr(self, '_plan_results') and agent_id in self._plan_results:
                plan_result = self._plan_results[agent_id]
            
            # Try to get think result from direct stage outputs first, then fallback to manifest
            think_node_id = f"{agent_id}_think"
            think_result = self._stage_outputs.get(think_node_id)
            if not think_result and self.manifest_manager:
                think_result = self.manifest_manager.get_node_output(think_node_id)
                self.logger.info("Using manifest fallback for think result in act stage")
            else:
                self.logger.info("Using direct stage output for think result in act stage")
                
            result = await method(self.agent_input, plan_result, think_result)
        else:
            result = await method(self.agent_input)
        
        # Convert result to dict if it has a to_dict method (for JSON serialization)
        serializable_result = result
        if hasattr(result, 'to_dict') and callable(result.to_dict):
            serializable_result = result.to_dict()
            self.logger.info(f"Converted {type(result).__name__} to dict for serialization")
        
        # Special handling for plan stage: dynamically update DAG
        if stage == "plan" and serializable_result:
            await self._update_dag_from_plan(agent_id, serializable_result, node.node_id)
        
        # Store result for direct passing to subsequent stages
        self._stage_outputs[node.node_id] = serializable_result
        
        # Store result in manifest with standardized stage naming
        if self.manifest_manager:
            # Use agent_id + stage for consistent naming (e.g., scout_plan, scout_collect)
            stage_id = f"{agent_id}_{stage}"
            
            # For collect stage, don't overwrite existing data - just update status
            if stage == "collect":
                # Only update the status, preserve any existing data (like Reddit data from sandbox)
                self.manifest_manager.update_node_status(stage_id, NodeStatus.COMPLETED)
                self.logger.info(f"Updated collect stage status without overwriting existing data: {stage_id}")
            else:
                # For other stages, store the result normally
                self.manifest_manager.store_node_output(stage_id, serializable_result)
                self.manifest_manager.update_node_status(stage_id, NodeStatus.COMPLETED)
        
        end_time = datetime.now()
        return NodeResult(
            success=True,
            output=serializable_result,
            start_time=start_time,
            end_time=end_time
        )
    
    # Tool node execution removed - handled by agent's collect() method
    
    async def _update_dag_from_plan(self, agent_id: str, plan_result: Dict[str, Any], plan_node_id: str) -> None:
        """
        Store the plan result for the collect stage to use.
        
        The collect stage will execute the tool nodes from the plan.
        
        Args:
            agent_id: ID of the agent that produced the plan
            plan_result: Output from the plan stage
            plan_node_id: ID of the plan node
        """
        # Store the plan result for the collect stage to access
        if not hasattr(self, '_plan_results'):
            self._plan_results = {}
        self._plan_results[agent_id] = plan_result
        
        # Extract tool node count for logging
        dag_data = plan_result.get("dag", {})
        nodes = dag_data.get("nodes", [])
        tool_count = len([n for n in nodes if n.get("type") == "tool"])
        
        self.logger.info(f"Stored plan with {tool_count} tool nodes for collect stage execution")
    
    # Code generation removed - only LLM generates code via plan prompt template
    
    def get_dag_structure(self) -> Dict[str, Any]:
        """Get the current DAG structure.
        
        Returns:
            Dictionary with nodes and edges
        """
        nodes = []
        edges = []
        
        for node_id, node_data in self.dag_engine.graph.nodes(data=True):
            node = node_data.get("node")
            if node:
                # Convert NodeConfig to dict for serialization
                if hasattr(node.config, "__dict__"):
                    config_dict = {
                        "retry_count": node.config.retry_count,
                        "timeout_seconds": node.config.timeout_seconds,
                        "skip_on_failure": node.config.skip_on_failure,
                        "parallel": node.config.parallel,
                        "metadata": node.config.metadata
                    }
                else:
                    config_dict = node.config
                
                # Convert NodeType enum to string for serialization
                node_type_str = str(node.node_type.name) if hasattr(node.node_type, "name") else str(node.node_type)
                
                nodes.append({
                    "id": node.node_id,
                    "name": node.name,
                    "type": node_type_str,
                    "config": config_dict
                })
        
        for source, target in self.dag_engine.graph.edges():
            edges.append({
                "source": source,
                "target": target
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    async def initialize(self, agent_input: Any) -> None:
        """
        Initialize the orchestrator with agent input.
        
        Args:
            agent_input: Input for the workflow
        """
        # Set up run directory
        self.setup_run_directory()
        
        # Store agent_input for use in node execution
        self.agent_input = agent_input
        
        # Build initial DAG
        self.build_initial_dag()
        
        # Store agent input in manifest
        if self.manifest_manager:
            if hasattr(agent_input, "to_dict"):
                input_dict = agent_input.to_dict()
            elif hasattr(agent_input, "data"):
                input_dict = agent_input.data
            else:
                input_dict = agent_input
            
            # Update run metadata with input information
            self.manifest_manager.update_run_metadata({
                "input": input_dict,
                "run_id": self.run_id,
                "start_time": datetime.now().isoformat()
            })
        
        self.logger.info(f"Initialized orchestrator with input: {agent_input}")
    
    async def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Returns:
            Results of the workflow execution
        """
        # Ensure run directory is set up
        if not self.run_dir:
            self.setup_run_directory()
        
        # Validate the DAG
        validation = self.dag_engine.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid DAG: {validation['errors']}")
        
        # Execute the DAG
        self.logger.info(f"Starting workflow execution with {len(self.dag_engine.graph.nodes())} nodes")
        execution_state = await self.dag_engine.execute(self._execute_node)
        
        # Save final state
        state_path = self.run_dir / "final_state.json"
        self.dag_engine.save_state(str(state_path))
        
        # Compile results
        results = {
            "run_id": self.run_id,
            "execution_time": execution_state.duration,
            "completed_nodes": execution_state.completed_nodes,
            "failed_nodes": execution_state.failed_nodes,
            "total_nodes": execution_state.total_nodes,
            "progress": execution_state.progress,
            "status": "completed" if not execution_state.failed_nodes else "failed"
        }
        
        # Update manifest with final status
        if self.manifest_manager:
            self.manifest_manager.update_run_status(
                "completed" if not execution_state.failed_nodes else "failed"
            )
            self.manifest_manager.update_run_metadata({
                "end_time": datetime.now().isoformat(),
                "duration": execution_state.duration
            })
        
        self.logger.info(f"Workflow execution completed in {execution_state.duration:.2f}s")
        return results


# Example agent stage configurations
AGENT_STAGE_CONFIGS = {
    "scout": AgentStageConfig(stages=["plan", "collect", "think", "act"]),
    "screener": AgentStageConfig(stages=["plan", "think", "act"]),
    "validator": AgentStageConfig(stages=["plan", "collect", "think", "act"]),
    "gap_finder": AgentStageConfig(stages=["plan", "collect", "think", "act"]),
    "builder": AgentStageConfig(stages=["think", "act"]),
    "writer": AgentStageConfig(stages=["think", "act"])
}


def create_orchestrator(run_id: str = None) -> AgentOrchestrator:
    """
    Create an agent orchestrator.
    
    Args:
        run_id: Optional run ID
        
    Returns:
        Configured AgentOrchestrator instance
    """
    return AgentOrchestrator(run_id)
