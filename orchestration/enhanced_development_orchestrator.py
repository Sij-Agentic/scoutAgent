"""
EnhancedDevelopmentOrchestrator for robust development of new agent stages.

This module extends the DevelopmentOrchestrator with improved data extraction
and passing capabilities to handle transitions between any agents and stages.
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
from scout_agent.agents.base import BaseAgent, AgentInput
from scout_agent.memory.manifest_manager import ManifestManager
from scout_agent.orchestration.agent_orchestrator import AgentStageConfig
from scout_agent.orchestration.development_orchestrator import DevelopmentOrchestrator

logger = get_logger("enhanced_development_orchestrator")

os.environ.setdefault("SCOUT_LLM_DEFAULT_BACKEND", "deepseek")
os.environ.setdefault("SCOUT_LLM_DEFAULT_MODEL", "deepseek-chat")


class EnhancedDevelopmentOrchestrator(DevelopmentOrchestrator):
    """
    Enhanced orchestrator for developing new agent stages with improved data handling.
    
    This orchestrator addresses several issues in the original DevelopmentOrchestrator:
    1. Robust data extraction from manifests
    2. Proper data passing between stages
    3. Handling of empty DAG nodes
    4. Consistent run directory management
    """
    
    def __init__(self, 
                source_run_id: str,
                agent_id: str,
                stages: List[str],
                new_run_id: Optional[str] = None,
                debug_mode: bool = False):
        """
        Initialize the enhanced development orchestrator.
        
        Args:
            source_run_id: ID of the source run to use for data
            agent_id: ID of the agent being developed
            stages: List of stages to develop
            new_run_id: Optional new run ID (if None, generates one)
            debug_mode: Enable additional debugging output
        """
        super().__init__(
            source_run_id=source_run_id,
            agent_id=agent_id,
            stages=stages,
            new_run_id=new_run_id
        )
        
        self.debug_mode = debug_mode
        self.dependency_map = {
            "scout": [],
            "screener": ["scout_act"],
            "validator": ["screener_act"],
            "gap_finder": ["validator_act"],
            "builder": ["gap_finder_act"],
            "writer": ["builder_act"]
        }
        
        # Track extracted data for debugging
        self.extracted_data = {}
        
        logger.info("Initialized enhanced development orchestrator")
    
    async def prepare_input_for_new_agent(self) -> None:
        """
        Enhanced preparation of input data for the new agent.
        
        This method robustly extracts data from the source manifest and
        ensures it's properly passed to the agent being developed.
        """
        # Get the dependencies for this agent
        dependencies = self.dependency_map.get(self.dev_agent_id, [])
        
        if not dependencies:
            logger.info(f"Agent {self.dev_agent_id} has no external dependencies")
            return
        
        # For each dependency, extract relevant data
        for dependency in dependencies:
            logger.info(f"Processing dependency: {dependency}")
            
            # Extract data using multiple strategies
            dependency_data = self._extract_data_from_manifest(dependency)
            
            if not dependency_data:
                logger.warning(f"No data found for dependency {dependency}")
                continue
                
            # Process the data based on agent type
            self._process_dependency_data(dependency, dependency_data)
    
    def _extract_data_from_manifest(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract data from manifest using multiple strategies.
        
        Args:
            node_id: ID of the node to extract data from
            
        Returns:
            Extracted data or None if not found
        """
        if not self.manifest_manager:
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
    
    def _process_dependency_data(self, dependency: str, data: Dict[str, Any]) -> None:
        """
        Process dependency data based on agent type.
        
        Args:
            dependency: Dependency node ID
            data: Extracted data
        """
        # Store for debugging
        self.extracted_data[dependency] = data
        
        # Handle based on agent type and dependency
        if self.dev_agent_id == "screener" and dependency == "scout_act":
            self._process_screener_input(data)
        elif self.dev_agent_id == "validator" and dependency == "screener_act":
            self._process_validator_input(data)
        elif self.dev_agent_id == "gap_finder" and dependency == "validator_act":
            self._process_gap_finder_input(data)
        elif self.dev_agent_id == "builder" and dependency == "gap_finder_act":
            self._process_builder_input(data)
        elif self.dev_agent_id == "writer" and dependency == "builder_act":
            self._process_writer_input(data)
        else:
            logger.warning(f"No specific processing for {self.dev_agent_id} with dependency {dependency}")
    
    def _process_screener_input(self, data: Dict[str, Any]) -> None:
        """
        Process input data for screener agent.
        
        Args:
            data: Data from scout_act
        """
        pain_points = self._extract_pain_points(data)
        
        if pain_points:
            logger.info(f"Found {len(pain_points)} pain points for screener")
            self.agent_input.data = pain_points
            
            # Set context if available
            if isinstance(data, dict):
                if "market_summary" in data:
                    self.agent_input.context["target_market"] = data["market_summary"]
                if "top_k" in data:
                    self.agent_input.context["top_k"] = data["top_k"]
                elif "metadata" in data and isinstance(data["metadata"], dict) and "top_k" in data["metadata"]:
                    self.agent_input.context["top_k"] = data["metadata"]["top_k"]
                else:
                    self.agent_input.context["top_k"] = 5  # Default
        else:
            logger.warning("No pain points found for screener")
    
    def _process_validator_input(self, data: Dict[str, Any]) -> None:
        """
        Process input data for validator agent.
        
        Args:
            data: Data from screener_act
        """
        top_pain_points = self._extract_top_pain_points(data)
        
        if top_pain_points:
            logger.info(f"Found {len(top_pain_points)} top pain points for validator")
            self.agent_input.data = top_pain_points
        else:
            logger.warning("No top pain points found for validator")
    
    def _process_gap_finder_input(self, data: Dict[str, Any]) -> None:
        """
        Process input data for gap_finder agent.
        
        Args:
            data: Data from validator_act
        """
        validated_pain_points = self._extract_validated_pain_points(data)
        
        if validated_pain_points:
            logger.info(f"Found {len(validated_pain_points)} validated pain points for gap finder")
            self.agent_input.data = validated_pain_points
        else:
            logger.warning("No validated pain points found for gap finder")
    
    def _process_builder_input(self, data: Dict[str, Any]) -> None:
        """
        Process input data for builder agent.
        
        Args:
            data: Data from gap_finder_act
        """
        gaps = self._extract_gaps(data)
        
        if gaps:
            logger.info(f"Found {len(gaps)} gaps for builder")
            self.agent_input.data = gaps
        else:
            logger.warning("No gaps found for builder")
    
    def _process_writer_input(self, data: Dict[str, Any]) -> None:
        """
        Process input data for writer agent.
        
        Args:
            data: Data from builder_act
        """
        solution = self._extract_solution(data)
        
        if solution:
            logger.info("Found solution for writer")
            self.agent_input.data = solution
        else:
            logger.warning("No solution found for writer")
    
    def _extract_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract pain points from various data formats.
        
        Args:
            data: Data that might contain pain points
            
        Returns:
            List of pain points
        """
        pain_points = []
        
        if isinstance(data, dict):
            # Direct pain_points field
            if "pain_points" in data:
                pain_points = data["pain_points"]
                logger.info("Found pain points in direct pain_points field")
                
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "pain_points" in data["result"]:
                    pain_points = data["result"]["pain_points"]
                    logger.info("Found pain points in result.pain_points field")
            
            # Try to find any array field that might contain pain points
            if not pain_points:
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        if any(pp.get("description") for pp in value):
                            pain_points = value
                            logger.info(f"Found potential pain points in {key} field")
                            break
        
        # If still no pain points and debug mode, create dummy data
        if not pain_points and self.debug_mode:
            logger.warning("Creating dummy pain points for debugging")
            pain_points = [
                {
                    "id": "dummy_1",
                    "description": "Dummy pain point for testing",
                    "severity": "high",
                    "market": "Test Market",
                    "source": "test",
                    "evidence": ["This is a test pain point"],
                    "frequency": 3,
                    "impact_score": 8.5
                },
                {
                    "id": "dummy_2",
                    "description": "Another dummy pain point for testing",
                    "severity": "medium",
                    "market": "Test Market",
                    "source": "test",
                    "evidence": ["This is another test pain point"],
                    "frequency": 2,
                    "impact_score": 6.0
                }
            ]
        
        return pain_points
    
    def _extract_top_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract top pain points from screener output.
        
        Args:
            data: Data that might contain top pain points
            
        Returns:
            List of top pain points
        """
        top_pain_points = []
        
        if isinstance(data, dict):
            # Direct top_pain_points field
            if "top_pain_points" in data:
                top_pain_points = data["top_pain_points"]
                logger.info("Found top pain points in direct top_pain_points field")
                
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "top_pain_points" in data["result"]:
                    top_pain_points = data["result"]["top_pain_points"]
                    logger.info("Found top pain points in result.top_pain_points field")
        
        return top_pain_points
    
    def _extract_validated_pain_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract validated pain points from validator output.
        
        Args:
            data: Data that might contain validated pain points
            
        Returns:
            List of validated pain points
        """
        validated_pain_points = []
        
        if isinstance(data, dict):
            # Direct validated_pain_points field
            if "validated_pain_points" in data:
                validated_pain_points = data["validated_pain_points"]
                logger.info("Found validated pain points in direct validated_pain_points field")
                
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "validated_pain_points" in data["result"]:
                    validated_pain_points = data["result"]["validated_pain_points"]
                    logger.info("Found validated pain points in result.validated_pain_points field")
        
        return validated_pain_points
    
    def _extract_gaps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract gaps from gap finder output.
        
        Args:
            data: Data that might contain gaps
            
        Returns:
            List of gaps
        """
        gaps = []
        
        if isinstance(data, dict):
            # Direct gaps field
            if "gaps" in data:
                gaps = data["gaps"]
                logger.info("Found gaps in direct gaps field")
                
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "gaps" in data["result"]:
                    gaps = data["result"]["gaps"]
                    logger.info("Found gaps in result.gaps field")
        
        return gaps
    
    def _extract_solution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract solution from builder output.
        
        Args:
            data: Data that might contain solution
            
        Returns:
            Solution data
        """
        solution = {}
        
        if isinstance(data, dict):
            # Direct solution field
            if "solution" in data:
                solution = data["solution"]
                logger.info("Found solution in direct solution field")
                
            # Inside result field
            elif "result" in data and isinstance(data["result"], dict):
                if "solution" in data["result"]:
                    solution = data["result"]["solution"]
                    logger.info("Found solution in result.solution field")
                else:
                    # The entire result might be the solution
                    solution = data["result"]
                    logger.info("Using entire result as solution")
        
        return solution
    
    async def _execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """
        Enhanced node execution with proper data passing.
        
        Args:
            node: The node to execute
            inputs: Inputs from predecessor nodes
            
        Returns:
            NodeResult with execution results
        """
        node_id = node.node_id
        
        # If node is marked as completed, skip execution
        if node_id in self.completed_stages:
            logger.info(f"Skipping already completed node: {node_id}")
            
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
        
        # For agent nodes, ensure proper setup
        if node.node_type == NodeType.AGENT:
            config = node.config
            agent_id = config.metadata.get("agent_id")
            stage = config.metadata.get("stage")
            
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Set run information in agent state
                setattr(agent.state, "run_id", self.run_id)
                setattr(agent.state, "run_dir", self.run_dir)
                setattr(agent.state, "manifest_manager", self.manifest_manager)
                
                # Special handling for the agent being developed
                if agent_id == self.dev_agent_id:
                    # For the first stage, ensure agent has the input data
                    if stage == self.agent_configs[agent_id].stages[0]:
                        logger.info(f"Setting up first stage for {agent_id}: {stage}")
                        
                        # Create appropriate input based on agent type
                        self._setup_agent_input(agent, stage)
                    
                    # For subsequent stages, ensure data is passed from previous stages
                    else:
                        prev_stage = self._get_previous_stage(agent_id, stage)
                        if prev_stage:
                            logger.info(f"Setting up stage {stage} with data from {prev_stage}")
                            
                            # Get data from previous stage
                            prev_data = self._stage_outputs.get(prev_stage)
                            if prev_data:
                                # Store in agent state
                                setattr(agent.state, f"{prev_stage}_output", prev_data)
                    
                    # Special direct execution for screener agent
                    if agent_id == "screener":
                        # Get the method to call
                        method_name = stage
                        if not hasattr(agent, method_name):
                            return await super()._execute_node(node, inputs)
                        
                        method = getattr(agent, method_name)
                        
                        # For screener think stage
                        if stage == "think":
                            logger.info("Direct execution of screener think stage")
                            start_time = datetime.now()
                            
                            try:
                                # Create screener input
                                from scout_agent.agents.screener import ScreenerInput
                                screener_input = ScreenerInput(
                                    pain_points=self.agent_input.data or [],
                                    target_market=self.agent_input.context.get("target_market", "Software Development"),
                                    top_k=self.agent_input.context.get("top_k", 5)
                                )
                                
                                # Call the method directly
                                result = await method(screener_input)
                                
                                # Create success result
                                end_time = datetime.now()
                                node_result = NodeResult(
                                    success=True,
                                    output=result,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                                
                                # Update node status
                                node.update_status(NodeStatus.COMPLETED, node_result)
                                
                                # Store result
                                self._stage_outputs[node_id] = result
                                
                                # Store in agent state for act stage
                                setattr(agent.state, "evaluations", result)
                                
                                logger.info(f"Direct execution of {node_id} completed successfully")
                                return node_result
                                
                            except Exception as e:
                                # Create failure result
                                end_time = datetime.now()
                                logger.error(f"Error in direct execution of {node_id}: {str(e)}")
                                return NodeResult(
                                    success=False,
                                    error=str(e),
                                    start_time=start_time,
                                    end_time=end_time
                                )
                        
                        # For screener act stage
                        elif stage == "act":
                            logger.info("Direct execution of screener act stage")
                            start_time = datetime.now()
                            
                            try:
                                # Create screener input
                                from scout_agent.agents.screener import ScreenerInput
                                screener_input = ScreenerInput(
                                    pain_points=self.agent_input.data or [],
                                    target_market=self.agent_input.context.get("target_market", "Software Development"),
                                    top_k=self.agent_input.context.get("top_k", 5)
                                )
                                
                                # Get evaluations from agent state
                                evaluations = getattr(agent.state, "evaluations", None)
                                
                                if not evaluations:
                                    # Try to get from stage outputs
                                    evaluations = self._stage_outputs.get("screener_think")
                                
                                if not evaluations:
                                    logger.warning("No evaluations found for act stage")
                                    evaluations = {
                                        "evaluations": [],
                                        "evaluation_summary": "No evaluations available"
                                    }
                                
                                # Call the method directly
                                result = await method(screener_input, evaluations)
                                
                                # Create success result
                                end_time = datetime.now()
                                node_result = NodeResult(
                                    success=True,
                                    output=result,
                                    start_time=start_time,
                                    end_time=end_time
                                )
                                
                                # Update node status
                                node.update_status(NodeStatus.COMPLETED, node_result)
                                
                                # Store result
                                self._stage_outputs[node_id] = result
                                
                                logger.info(f"Direct execution of {node_id} completed successfully")
                                return node_result
                                
                            except Exception as e:
                                # Create failure result
                                end_time = datetime.now()
                                logger.error(f"Error in direct execution of {node_id}: {str(e)}")
                                return NodeResult(
                                    success=False,
                                    error=str(e),
                                    start_time=start_time,
                                    end_time=end_time
                                )
        
        # Execute normally
        return await super()._execute_node(node, inputs)
    
    def _setup_agent_input(self, agent: BaseAgent, stage: str) -> None:
        """
        Set up agent input based on agent type and stage.
        
        Args:
            agent: The agent instance
            stage: The stage being executed
        """
        agent_id = agent.agent_id
        
        # Handle based on agent type
        if agent_id == "screener":
            from scout_agent.agents.screener import ScreenerInput
            
            # Create screener input
            screener_input = ScreenerInput(
                pain_points=self.agent_input.data or [],
                target_market=self.agent_input.context.get("target_market", "Software Development"),
                top_k=self.agent_input.context.get("top_k", 5)
            )
            
            # Store in agent state
            setattr(agent.state, "screener_input", screener_input)
            
            logger.info(f"Set up screener input with {len(screener_input.pain_points)} pain points")
        
        # Add other agent types as needed
    
    def _get_previous_stage(self, agent_id: str, current_stage: str) -> Optional[str]:
        """
        Get the previous stage for an agent.
        
        Args:
            agent_id: Agent ID
            current_stage: Current stage
            
        Returns:
            Previous stage ID or None
        """
        stages = self.agent_configs.get(agent_id).stages
        if not stages:
            return None
            
        try:
            current_idx = stages.index(current_stage)
            if current_idx > 0:
                prev_stage = stages[current_idx - 1]
                return f"{agent_id}_{prev_stage}"
        except ValueError:
            pass
            
        return None
    
    def dump_debug_info(self, output_dir: Optional[str] = None) -> None:
        """
        Dump debug information to files.
        
        Args:
            output_dir: Optional output directory (defaults to debug)
        """
        if not self.debug_mode:
            return
            
        # Create debug directory
        debug_dir = Path(output_dir or "debug")
        debug_dir.mkdir(exist_ok=True)
        
        # Dump extracted data
        timestamp = int(datetime.now().timestamp())
        data_file = debug_dir / f"extracted_data_{timestamp}.json"
        
        try:
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(self.extracted_data, f, indent=2, default=str)
                
            logger.info(f"Dumped extracted data to {data_file}")
        except Exception as e:
            logger.error(f"Failed to dump debug info: {e}")
