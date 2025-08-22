"""
ReplayOrchestrator for checkpoint replay and development mode.

This module provides orchestrators for replaying workflows from checkpoints
and developing new agent stages using data from existing runs.
"""

import os
import json
import asyncio
import shutil
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime

from scout_agent.dag.engine import DAGEngine
from scout_agent.dag.node import DAGNode, NodeStatus, NodeResult, NodeConfig, NodeType
from scout_agent.custom_logging.logger import get_logger
from scout_agent.agents.base import BaseAgent, AgentInput
from scout_agent.memory.manifest_manager import ManifestManager
from scout_agent.orchestration.agent_orchestrator import AgentOrchestrator, AgentStageConfig


class ReplayOrchestrator(AgentOrchestrator):
    """
    Orchestrator for replaying workflows from checkpoints.
    
    This orchestrator allows:
    - Replaying from a specific stage
    - Replaying until a specific stage
    - Skipping specific stages
    - Loading data from existing manifest
    """
    
    def __init__(self, 
                run_id: str, 
                from_stage: Optional[str] = None,
                to_stage: Optional[str] = None,
                skip_stages: Optional[List[str]] = None,
                new_run_id: Optional[str] = None):
        """
        Initialize the replay orchestrator.
        
        Args:
            run_id: ID of the run to replay
            from_stage: Optional stage to start replay from (inclusive)
            to_stage: Optional stage to replay until (inclusive)
            skip_stages: Optional list of stages to skip during replay
            new_run_id: Optional new run ID for the replay (if None, uses the original run_id)
        """
        # If new_run_id is provided, use it instead of the original run_id
        self.source_run_id = run_id
        replay_run_id = new_run_id or run_id
        
        super().__init__(replay_run_id)
        
        self.replay_mode = True
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.skip_stages = skip_stages or []
        self.completed_stages = set()
        
        self.logger.info(f"Initialized replay orchestrator for run: {run_id}")
        if new_run_id:
            self.logger.info(f"Using new run ID for replay: {new_run_id}")
        if from_stage:
            self.logger.info(f"Will replay from stage: {from_stage}")
        if to_stage:
            self.logger.info(f"Will replay until stage: {to_stage}")
        if skip_stages:
            self.logger.info(f"Will skip stages: {skip_stages}")
    
    def setup_run_directory(self) -> Path:
        """
        Set up the run directory for this orchestration.
        
        For replay mode with new_run_id, copy the manifest from source_run_id.
        
        Returns:
            Path to the run directory
        """
        # Project root at ScoutAgent/ (not scout_agent/)
        root = Path(__file__).resolve().parents[2]
        run_dir = root / "data" / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_dir = run_dir
        
        # If this is a new run based on an existing one, copy the manifest
        if self.run_id != self.source_run_id:
            source_dir = root / "data" / "runs" / self.source_run_id
            source_manifest = source_dir / "run_manifest.json"
            target_manifest = run_dir / "run_manifest.json"
            
            if source_manifest.exists():
                if not target_manifest.exists():
                    shutil.copy(source_manifest, target_manifest)
                    self.logger.info(f"Copied manifest from {source_manifest} to {target_manifest}")
            else:
                self.logger.warning(f"Source manifest not found: {source_manifest}")
        
        # Initialize manifest manager
        self.manifest_manager = ManifestManager(run_dir / "run_manifest.json", create_if_missing=True)
        
        # Update run metadata for replay
        if self.manifest_manager:
            self.manifest_manager.update_run_metadata({
                "replay": {
                    "source_run_id": self.source_run_id,
                    "from_stage": self.from_stage,
                    "to_stage": self.to_stage,
                    "skip_stages": self.skip_stages,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        self.logger.info(f"Set up run directory: {run_dir}")
        return run_dir
    
    def build_replay_dag(self) -> None:
        """
        Build the DAG for replay, marking completed stages.
        
        This modifies the standard DAG to:
        1. Mark stages before from_stage as completed
        2. Mark skipped stages as completed
        3. Remove stages after to_stage
        """
        # First build the normal DAG
        self.build_initial_dag()
        
        # Get the execution order
        execution_order = self.dag_engine.get_execution_order()
        
        # Determine which stages to mark as completed
        if self.from_stage and self.from_stage in execution_order:
            from_idx = execution_order.index(self.from_stage)
            # Mark all stages before from_stage as completed
            for i in range(from_idx):
                node_id = execution_order[i]
                self._mark_node_completed(node_id)
                self.completed_stages.add(node_id)
        
        # Mark skipped stages as completed
        for node_id in self.skip_stages:
            if node_id in self.dag_engine.graph.nodes():
                self._mark_node_completed(node_id)
                self.completed_stages.add(node_id)
        
        # If to_stage is specified, remove all nodes after it
        if self.to_stage and self.to_stage in execution_order:
            to_idx = execution_order.index(self.to_stage)
            # Remove all nodes after to_stage
            nodes_to_remove = []
            for i in range(to_idx + 1, len(execution_order)):
                nodes_to_remove.append(execution_order[i])
            
            for node_id in nodes_to_remove:
                if node_id in self.dag_engine.graph:
                    self.dag_engine.graph.remove_node(node_id)
        
        # Log the modified execution order
        new_order = self.dag_engine.get_execution_order()
        self.logger.info(f"Replay DAG execution order: {new_order}")
        self.logger.info(f"Marked {len(self.completed_stages)} stages as completed")
    
    def _mark_node_completed(self, node_id: str) -> None:
        """
        Mark a node as completed.
        
        Args:
            node_id: ID of the node to mark as completed
        """
        if node_id not in self.dag_engine.graph.nodes():
            self.logger.warning(f"Node {node_id} not found in DAG")
            return
        
        node_data = self.dag_engine.graph.nodes[node_id]
        node = node_data.get("node")
        if not node:
            self.logger.warning(f"Node {node_id} has no node data")
            return
        
        # Create a dummy result
        result = NodeResult(
            success=True,
            output={},
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        
        # Update node status
        node.update_status(NodeStatus.COMPLETED, result)
        self.logger.debug(f"Marked node {node_id} as completed")
    
    async def hydrate_from_manifest(self) -> None:
        """
        Load data from manifest into memory services.
        
        This populates:
        - StageMessageService with stage outputs
        - _stage_outputs for legacy compatibility
        - _tool_results for collect stages
        """
        # Hydrate completed stages
        for node_id in self.completed_stages:
            # Get data from manifest
            data = self.manifest_manager.get_node_output(node_id)
            if data:
                # Store in message service for subsequent stages
                if self.message_service:
                    self.message_service.publish_stage_output(
                        workflow_id=self.run_id,
                        stage_id=node_id,
                        data=data
                    )
                    
                # Also store in legacy _stage_outputs
                self._stage_outputs[node_id] = data
                
                # If this is a collect stage, extract tool results
                if "_collect" in node_id:
                    agent_id = node_id.split("_")[0]
                    self._extract_tool_results_for_agent(agent_id)
                    
        self.logger.info(f"Hydrated data for {len(self.completed_stages)} completed stages")
    
    def _extract_tool_results_for_agent(self, agent_id: str) -> None:
        """
        Extract tool results for an agent from the manifest.
        
        Args:
            agent_id: ID of the agent
        """
        # Get the agent's plan result
        plan_node_id = f"{agent_id}_plan"
        plan_result = self.manifest_manager.get_node_output(plan_node_id)
        
        if not plan_result:
            self.logger.warning(f"No plan result found for agent {agent_id}")
            return
        
        # Extract tool results using existing method
        self._extract_tool_results_from_manifest(plan_result)
    
    def _load_agent_input_from_manifest(self) -> AgentInput:
        """
        Load agent input from manifest.
        
        Returns:
            AgentInput object loaded from manifest
        """
        manifest = self.manifest_manager.get_manifest()
        
        # Try to get from run_metadata
        if "run_metadata" in manifest and "input" in manifest["run_metadata"]:
            input_data = manifest["run_metadata"]["input"]
            
            # Create AgentInput from data
            from scout_agent.agents.base import AgentInput
            agent_input = AgentInput(
                data=input_data.get("data", {}),
                metadata=input_data.get("metadata", {}),
                context=input_data.get("context", {})
            )
            
            self.logger.info("Loaded agent input from manifest")
            return agent_input
        
        # Fallback to empty input
        self.logger.warning("No agent input found in manifest, using empty input")
        from scout_agent.agents.base import AgentInput
        return AgentInput(data={}, metadata={}, context={})
    
    async def initialize(self, agent_input: Optional[AgentInput] = None) -> None:
        """
        Initialize the replay orchestrator.
        
        Args:
            agent_input: Optional agent input (if None, loads from manifest)
        """
        # If agent_input not provided, load from manifest
        if agent_input is None:
            agent_input = self._load_agent_input_from_manifest()
            
        # Call parent initialize
        await super().initialize(agent_input)
        
        # Build the replay DAG
        self.build_replay_dag()
        
        # Hydrate services with manifest data
        await self.hydrate_from_manifest()
    
    async def _execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """
        Execute a single node in the DAG.
        
        Overrides parent method to skip already completed nodes.
        
        Args:
            node: The node to execute
            inputs: Inputs from predecessor nodes
            
        Returns:
            NodeResult with execution results
        """
        node_id = node.node_id
        
        # If node is marked as completed, skip execution
        if node_id in self.completed_stages:
            self.logger.info(f"Skipping already completed node: {node_id}")
            
            # Create a success result
            start_time = datetime.now()
            end_time = datetime.now()
            
            # Try to get the actual result from manifest
            output_data = self.manifest_manager.get_node_output(node_id) or {}
            
            return NodeResult(
                success=True,
                output=output_data,
                start_time=start_time,
                end_time=end_time
            )
        
        # Otherwise, execute normally
        return await super()._execute_node(node, inputs)
