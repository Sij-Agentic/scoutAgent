"""
DevelopmentOrchestrator for developing new agent stages.

This module provides an orchestrator for developing new agent stages
using data from existing runs.
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
from scout_agent.orchestration.replay_orchestrator import ReplayOrchestrator


class DevelopmentOrchestrator(ReplayOrchestrator):
    """
    Orchestrator for developing new agent stages.
    
    This orchestrator allows:
    - Using data from an existing run
    - Developing specific stages of a new agent
    - Creating a new run with the developed agent stages
    """
    
    def __init__(self, 
                source_run_id: str,
                agent_id: str,
                stages: List[str],
                new_run_id: Optional[str] = None):
        """
        Initialize the development orchestrator.
        
        Args:
            source_run_id: ID of the source run to use for data
            agent_id: ID of the agent being developed
            stages: List of stages to develop
            new_run_id: Optional new run ID (if None, generates one)
        """
        # Generate new run_id if not provided
        self.dev_agent_id = agent_id
        self.dev_stages = stages
        self.dev_run_id = new_run_id or f"dev_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize with replay mode
        super().__init__(
            run_id=source_run_id,
            new_run_id=self.dev_run_id
        )
        
        self.development_mode = True
        
        self.logger.info(f"Initialized development orchestrator for agent: {agent_id}")
        self.logger.info(f"Developing stages: {stages}")
        self.logger.info(f"Using source run: {source_run_id}")
        self.logger.info(f"Development run ID: {self.dev_run_id}")
    
    def register_agent_for_development(self, agent_id: str, agent_config: AgentStageConfig, agent_instance: BaseAgent) -> None:
        """
        Register an agent for development with only the specified stages.
        
        Args:
            agent_id: Agent identifier
            agent_config: Full agent stage configuration
            agent_instance: Agent instance
        """
        if agent_id != self.dev_agent_id:
            # Register non-development agents normally
            self.register_agent(agent_id, agent_config, agent_instance)
            return
        
        # Create a filtered config with only the specified stages
        filtered_stages = [s for s in agent_config.stages if s in self.dev_stages]
        if not filtered_stages:
            self.logger.warning(f"No matching stages found for development in {agent_id}")
            return
        
        # Create a new config with only the specified stages
        filtered_config = AgentStageConfig(stages=filtered_stages)
        
        # Register with filtered config
        self.register_agent(agent_id, filtered_config, agent_instance)
        self.logger.info(f"Registered agent {agent_id} for development with stages: {filtered_stages}")
    
    def build_development_dag(self) -> None:
        """
        Build a DAG that includes only the stages being developed and their dependencies.
        
        This creates a DAG with:
        1. All non-development agent nodes marked as completed
        2. Only the specified stages of the development agent
        3. Proper dependencies between development stages and other agents
        """
        # Clear any existing DAG
        self.dag_engine.graph.clear()
        
        # First, add all non-development agent nodes as completed
        for agent_id, config in self.agent_configs.items():
            if agent_id != self.dev_agent_id:
                for stage in config.stages:
                    node_id = f"{agent_id}_{stage}"
                    node = self._create_agent_stage_node(agent_id, stage, [])
                    self.dag_engine.add_node(node)
                    # Mark as completed since we're not executing these
                    node.update_status(NodeStatus.COMPLETED)
                    self.completed_stages.add(node_id)
        
        # Then add development agent nodes with proper dependencies
        prev_node_id = None
        for stage in self.agent_configs[self.dev_agent_id].stages:
            node_id = f"{self.dev_agent_id}_{stage}"
            dependencies = []
            
            # Add dependency on previous stage if it exists
            if prev_node_id:
                dependencies.append(prev_node_id)
            
            # Add external dependencies based on stage
            if stage == self.agent_configs[self.dev_agent_id].stages[0]:
                # First stage might depend on another agent's output
                external_dep = self._get_external_dependency()
                if external_dep:
                    dependencies.append(external_dep)
            
            # Create and add the node
            node = self._create_agent_stage_node(self.dev_agent_id, stage, dependencies)
            self.dag_engine.add_node(node)
            
            # Add edges for dependencies
            for dep in dependencies:
                if dep in self.dag_engine.graph.nodes():
                    self.dag_engine.add_edge(dep, node_id)
            
            prev_node_id = node_id
        
        # Validate the DAG
        validation = self.dag_engine.validate()
        if not validation["valid"]:
            self.logger.error(f"Invalid development DAG: {validation['errors']}")
            raise ValueError(f"Invalid development DAG: {validation['errors']}")
        
        # Log the execution order
        execution_order = self.dag_engine.get_execution_order()
        self.logger.info(f"Development DAG execution order: {execution_order}")
    
    def _get_external_dependency(self) -> Optional[str]:
        """
        Determine external dependency based on agent type.
        
        Returns:
            Node ID of the dependency or None
        """
        dependency_map = {
            "scout": None,  # Scout has no dependencies
            "screener": "scout_act",
            "validator": "screener_act",
            "gap_finder": "validator_act",
            "builder": "gap_finder_act",
            "writer": "builder_act"
        }
        return dependency_map.get(self.dev_agent_id)
    
    async def prepare_input_for_new_agent(self) -> None:
        """
        Prepare input data for the new agent based on its dependencies.
        
        This loads data from the source run and prepares it for the new agent.
        """
        # Get the external dependency
        dependency = self._get_external_dependency()
        if not dependency:
            self.logger.info(f"Agent {self.dev_agent_id} has no external dependencies")
            return
        
        # Get data from the dependency - try multiple paths in the manifest
        dependency_data = self.manifest_manager.get_node_output(dependency)
        
        # If no data found, try looking in the stages section
        if not dependency_data:
            manifest = self.manifest_manager.get_manifest()
            if "stages" in manifest and dependency in manifest["stages"]:
                stage_data = manifest["stages"][dependency]
                if "data" in stage_data:
                    dependency_data = stage_data["data"]
                    self.logger.info(f"Found dependency data in manifest stages.{dependency}.data")
        
        if not dependency_data:
            self.logger.warning(f"No data found for dependency {dependency}")
            return
        
        # Prepare input data based on agent type
        if self.dev_agent_id == "screener":
            # Screener needs pain points from scout_act
            pain_points = []
            
            # Try to extract pain points from different possible locations
            if isinstance(dependency_data, dict):
                # Direct pain_points field
                if "pain_points" in dependency_data:
                    pain_points = dependency_data["pain_points"]
                    self.logger.info("Found pain points in direct pain_points field")
                    
                # Inside result field
                elif "result" in dependency_data and isinstance(dependency_data["result"], dict):
                    if "pain_points" in dependency_data["result"]:
                        pain_points = dependency_data["result"]["pain_points"]
                        self.logger.info("Found pain points in result.pain_points field")
                
                # Try to find any array field that might contain pain points
                if not pain_points:
                    for key, value in dependency_data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            if any(pp.get("description") for pp in value):
                                pain_points = value
                                self.logger.info(f"Found potential pain points in {key} field")
                                break
            
            # Log the raw dependency data for debugging
            self.logger.info(f"Raw dependency data keys: {list(dependency_data.keys()) if isinstance(dependency_data, dict) else 'not a dict'}")
            
            # If still no pain points, create a dummy one for testing
            if not pain_points:
                self.logger.warning("No pain points found in dependency data, creating dummy data for testing")
                pain_points = [
                    {
                        "id": "dummy_1",
                        "description": "Dummy pain point for testing",
                        "severity": "medium",
                        "market": "Test Market",
                        "source": "test",
                        "evidence": ["This is a test pain point"],
                        "frequency": 1,
                        "impact_score": 5.0
                    }
                ]
            
            self.logger.info(f"Found {len(pain_points)} pain points for screener from scout_act")
            self.agent_input.data = pain_points
            
            # Set run_dir and manifest_manager in agent state for screener
            agent = self.agents.get(self.dev_agent_id)
            if agent:
                setattr(agent.state, "run_id", self.run_id)
                setattr(agent.state, "run_dir", self.run_dir)
                setattr(agent.state, "manifest_manager", self.manifest_manager)
                self.logger.info(f"Set run_dir and manifest_manager in {self.dev_agent_id} agent state")
        
        elif self.dev_agent_id == "validator":
            # Validator needs top pain points from screener_act
            if isinstance(dependency_data, dict):
                top_pain_points = dependency_data.get("top_pain_points", [])
                if not top_pain_points and "result" in dependency_data:
                    top_pain_points = dependency_data["result"].get("top_pain_points", [])
                
                self.logger.info(f"Found {len(top_pain_points)} top pain points for validator from screener_act")
                self.agent_input.data = top_pain_points
                
                # Set run_dir and manifest_manager in agent state for validator
                agent = self.agents.get(self.dev_agent_id)
                if agent:
                    setattr(agent.state, "run_id", self.run_id)
                    setattr(agent.state, "run_dir", self.run_dir)
                    setattr(agent.state, "manifest_manager", self.manifest_manager)
                    self.logger.info(f"Set run_dir and manifest_manager in {self.dev_agent_id} agent state")
        
        # Add more agent types as needed
    
    async def initialize(self, agent_input: Optional[AgentInput] = None) -> None:
        """
        Initialize the development orchestrator.
        
        Args:
            agent_input: Optional agent input (if None, loads from manifest)
        """
        # If agent_input not provided, load from manifest
        if agent_input is None:
            agent_input = self._load_agent_input_from_manifest()
            
        # Call parent initialize
        await super().initialize(agent_input)
        
        # Build the development DAG instead of replay DAG
        self.build_development_dag()
        
        # Hydrate services with manifest data
        await self.hydrate_from_manifest()
        
        # Prepare input for the new agent
        await self.prepare_input_for_new_agent()
        
        # Update run metadata for development
        if self.manifest_manager:
            self.manifest_manager.update_run_metadata({
                "development": {
                    "source_run_id": self.source_run_id,
                    "agent_id": self.dev_agent_id,
                    "stages": self.dev_stages,
                    "timestamp": datetime.now().isoformat()
                }
            })
