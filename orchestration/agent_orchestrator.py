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
            # Handle different node types
            if node.node_type == NodeType.AGENT:
                return await self._execute_agent_node(node, inputs)
            elif node.node_type == NodeType.FUNCTION:  # Using FUNCTION for tool nodes
                return await self._execute_tool_node(node, inputs)
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
            raise ValueError(f"Method {method_name} not found on agent {agent_id}")
        
        method = getattr(agent, method_name)
        
        # Prepare parameters based on method signature
        import inspect
        sig = inspect.signature(method)
        params = {}
        
        # Always provide agent_input if the method accepts it
        if "agent_input" in sig.parameters and hasattr(self, "agent_input"):
            params["agent_input"] = self.agent_input
            
        # Add run_id only if the method accepts it
        if "run_id" in sig.parameters:
            params["run_id"] = self.run_id
        
        # Get plan results from previous nodes
        plan_result = None
        for dep_id, dep_output in inputs.items():
            if "plan" in dep_id.lower():
                plan_result = dep_output
                break
        
        # If no plan found in inputs, try to get it from the manifest
        if not plan_result and self.manifest_manager:
            # Try to find a plan node for this agent
            plan_node_id = f"{agent_id}_plan"
            plan_result = self.manifest_manager.get_node_output(plan_node_id)
            
        # Get think results from previous nodes
        think_result = None
        for dep_id, dep_output in inputs.items():
            if "think" in dep_id.lower():
                think_result = dep_output
                break
                
        # If no think result found in inputs, try to get it from the manifest
        if not think_result and self.manifest_manager:
            # Try to find a think node for this agent
            think_node_id = f"{agent_id}_think"
            think_result = self.manifest_manager.get_node_output(think_node_id)
        
        # Add plan parameter if required
        if "plan" in sig.parameters and plan_result:
            params["plan"] = plan_result
            
        # Add thoughts parameter if required
        if "thoughts" in sig.parameters and think_result:
            params["thoughts"] = think_result
        
        # Execute the agent method
        self.logger.info(f"Executing agent node: {node.node_id}")
        result = await method(**params)
        
        # Convert result to dict if it has a to_dict method (for JSON serialization)
        serializable_result = result
        if hasattr(result, 'to_dict') and callable(result.to_dict):
            serializable_result = result.to_dict()
            self.logger.info(f"Converted {type(result).__name__} to dict for serialization")
        
        # Special handling for plan stage: dynamically update DAG
        if stage == "plan" and serializable_result:
            await self._update_dag_from_plan(agent_id, serializable_result, node.node_id)
        
        # Store result in manifest
        if self.manifest_manager:
            self.manifest_manager.store_node_output(node.node_id, serializable_result)
            self.manifest_manager.update_node_status(node.node_id, NodeStatus.COMPLETED)
        
        end_time = datetime.now()
        return NodeResult(
            success=True,
            output=serializable_result,
            start_time=start_time,
            end_time=end_time
        )
    
    async def _execute_tool_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """
        Execute a tool node.
        
        Args:
            node: The tool node to execute
            inputs: Inputs from predecessor nodes
            
        Returns:
            NodeResult with execution results
        """
        start_time = datetime.now()
        
        # Get the tool configuration from metadata
        tool_name = node.config.metadata.get("tool")
        params = node.config.metadata.get("params", {})
        code = node.config.metadata.get("code")
        
        # Find the agent responsible for executing this tool
        agent_id = node.config.metadata.get("agent_id")
        if not agent_id or agent_id not in self.agents:
            # Default to the first agent that has collect capability
            for aid, agent in self.agents.items():
                if "collect" in self.agent_configs[aid].stages:
                    agent_id = aid
                    break
        
        if not agent_id:
            raise ValueError("No agent found to execute tool node")
        
        agent = self.agents[agent_id]
        
        # Execute the tool
        self.logger.info(f"Executing tool node: {node.node_id} with tool: {tool_name}")
        
        # Assume the agent has a collect method that can execute tools
        if not hasattr(agent, "collect"):
            raise ValueError(f"Agent {agent_id} does not have a collect method")
        
        # Create a mini-plan with just this tool node
        tool_plan = {
            "dag": {
                "nodes": [{
                    "id": node.node_id,
                    "type": "tool",
                    "tool": tool_name,
                    "params": params,
                    "code": code
                }]
            }
        }
        
        # Execute the tool using the agent's collect method
        result = await agent.collect(
            plan=tool_plan,
            run_id=self.run_id
        )
        
        # Store result in manifest
        if self.manifest_manager:
            self.manifest_manager.store_node_output(node.node_id, result)
            self.manifest_manager.update_node_status(
                node.node_id, 
                NodeStatus.COMPLETED if node.node_id in result.get("completed", []) else NodeStatus.FAILED
            )
        
        end_time = datetime.now()
        return NodeResult(
            success=node.node_id in result.get("completed", []),
            output=result,
            start_time=start_time,
            end_time=end_time
        )
    
    async def _update_dag_from_plan(self, agent_id: str, plan_result: Dict[str, Any], plan_node_id: str) -> None:
        """
        Update the DAG based on the plan output.
        
        This implements Option A, where the plan stage dynamically modifies the DAG.
        
        Args:
            agent_id: ID of the agent that produced the plan
            plan_result: Output from the plan stage
            plan_node_id: ID of the plan node
        """
        # Extract DAG structure from plan
        dag_data = plan_result.get("dag", {})
        nodes = dag_data.get("nodes", [])
        
        # Find the next agent stage after plan (usually collect)
        next_stage = None
        agent_stages = self.agent_configs[agent_id].stages
        plan_index = agent_stages.index("plan") if "plan" in agent_stages else -1
        if plan_index >= 0 and plan_index + 1 < len(agent_stages):
            next_stage = agent_stages[plan_index + 1]
        
        if not next_stage:
            self.logger.warning(f"No next stage found after plan for agent {agent_id}")
            return
        
        next_stage_node_id = f"{agent_id}_{next_stage}"
        
        # Find tool nodes in the plan
        tool_nodes = []
        for node_data in nodes:
            if node_data.get("type") == "tool":
                # Create NodeConfig for tool node
                tool_config = NodeConfig()
                tool_config.metadata = {
                    "tool": node_data.get("tool"),
                    "params": node_data.get("params", {}),
                    "code": node_data.get("code"),
                    "agent_id": agent_id  # Associate with the agent that created the plan
                }
                
                tool_node = DAGNode(
                    node_id=node_data.get("id", f"tool_{len(tool_nodes)}"),
                    name=node_data.get("name", f"Tool {len(tool_nodes)}"),
                    node_type=NodeType.FUNCTION,
                    config=tool_config,
                    dependencies=[plan_node_id]  # Depend on the plan node
                )
                tool_nodes.append(tool_node)
        
        # Add tool nodes to the DAG
        for tool_node in tool_nodes:
            self.dag_engine.add_node(tool_node)
            self.dag_engine.add_edge(plan_node_id, tool_node.node_id)
            
            # Make the next stage depend on this tool node
            if next_stage_node_id in self.dag_engine.graph.nodes():
                self.dag_engine.add_edge(tool_node.node_id, next_stage_node_id)
        
        self.logger.info(f"Updated DAG with {len(tool_nodes)} tool nodes from plan")
    
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
