"""
Core DAG Engine for ScoutAgent orchestration.

This module provides the main DAG orchestration system using NetworkX for graph
management and async execution for parallel processing of independent tasks.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx
from pathlib import Path
import json

from .node import DAGNode, NodeStatus, NodeResult
from ..custom_logging.logger import get_logger


@dataclass
class ExecutionState:
    """Tracks the overall execution state of a DAG."""
    workflow_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    skipped_nodes: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Duration of execution in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def progress(self) -> float:
        """Progress as a percentage (0-100)."""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100


class DAGValidator:
    """Validates DAG structure and execution readiness."""
    
    def __init__(self):
        self.logger = get_logger("DAGValidator")
    
    def validate_dag(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Comprehensive DAG validation."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if it's a valid DAG
        if not nx.is_directed_acyclic_graph(graph):
            validation_result["valid"] = False
            validation_result["errors"].append("Graph contains cycles")
        
        # Check for required attributes
        for node_id, node_data in graph.nodes(data=True):
            node = node_data.get("node")
            if not node:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Node {node_id} missing node data")
                continue
                
            if not node.name:
                validation_result["warnings"].append(f"Node {node_id} missing name")
        
        # Check for root node
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if len(root_nodes) != 1:
            validation_result["warnings"].append(
                f"Expected exactly one root node, found {len(root_nodes)}"
            )
        
        # Check for disconnected components
        if not nx.is_weakly_connected(graph):
            validation_result["warnings"].append("Graph has disconnected components")
        
        return validation_result
    
    def get_ready_nodes(self, graph: nx.DiGraph, completed_nodes: Set[str]) -> List[str]:
        """Get nodes ready for execution based on dependencies."""
        ready_nodes = []
        
        for node_id in graph.nodes():
            if node_id in completed_nodes:
                continue
                
            node_data = graph.nodes[node_id]
            node = node_data.get("node")
            
            if not node:
                continue
                
            if node.is_ready(completed_nodes):
                ready_nodes.append(node_id)
        
        return ready_nodes


class DAGEngine:
    """Main DAG orchestration engine."""
    
    def __init__(self, max_concurrent: int = 4):
        self.logger = get_logger("DAGEngine")
        self.max_concurrent = max_concurrent
        self.validator = DAGValidator()
        self.graph = nx.DiGraph()
        self.execution_state = None
        
    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG."""
        self.graph.add_node(node.node_id, node=node)
        self.logger.info(f"Added node: {node.node_id} ({node.name})")
    
    def add_edge(self, from_node: str, to_node: str, **kwargs) -> None:
        """Add a directed edge between nodes."""
        self.graph.add_edge(from_node, to_node, **kwargs)
        self.logger.info(f"Added edge: {from_node} -> {to_node}")
    
    def build_from_nodes(self, nodes: List[DAGNode]) -> None:
        """Build DAG from a list of nodes with dependencies."""
        self.graph.clear()
        
        # Add all nodes
        for node in nodes:
            self.add_node(node)
        
        # Add edges based on dependencies
        for node in nodes:
            for dep in node.dependencies:
                if dep in self.graph.nodes():
                    self.add_edge(dep, node.node_id)
    
    def validate(self) -> Dict[str, Any]:
        """Validate the current DAG structure."""
        return self.validator.validate_dag(self.graph)
    
    def get_execution_order(self) -> List[str]:
        """Get topological execution order."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            self.logger.error(f"Failed to get execution order: {e}")
            return []
    
    def get_ready_nodes(self, completed_nodes: Set[str]) -> List[str]:
        """Get nodes ready for execution."""
        return self.validator.get_ready_nodes(self.graph, completed_nodes)
    
    async def execute_node(self, node_id: str, executor_func: Callable) -> NodeResult:
        """Execute a single node with the provided executor function."""
        node_data = self.graph.nodes[node_id]
        node = node_data["node"]
        
        self.logger.info(f"Starting execution of node: {node_id} ({node.name})")
        
        try:
            # Update status to running
            node.update_status(NodeStatus.RUNNING)
            
            # Prepare inputs from dependencies
            inputs = self._prepare_node_inputs(node_id)
            node.inputs.update(inputs)
            
            # Execute the node
            result = await executor_func(node, inputs)
            
            # Update node with result
            node.update_status(NodeStatus.COMPLETED, result)
            
            self.logger.info(
                f"Completed node: {node_id} ({node.name}) "
                f"in {result.duration:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute node {node_id}: {e}")
            
            result = NodeResult(
                success=False,
                error=str(e),
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            
            node.update_status(NodeStatus.FAILED, result)
            return result
    
    def _prepare_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """Prepare inputs for a node from its dependencies."""
        inputs = {}
        
        # Collect outputs from all predecessor nodes
        for predecessor in self.graph.predecessors(node_id):
            pred_node = self.graph.nodes[predecessor]["node"]
            if pred_node.result and pred_node.result.success:
                inputs[predecessor] = pred_node.outputs
        
        return inputs
    
    async def execute(self, executor_func: Callable) -> ExecutionState:
        """Execute the entire DAG."""
        validation = self.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid DAG: {validation['errors']}")
        
        self.execution_state = ExecutionState(
            workflow_id=str(hash(frozenset(self.graph.nodes())))
        )
        
        self.logger.info(f"Starting DAG execution with {len(self.graph.nodes)} nodes")
        
        completed_nodes = set()
        failed_nodes = set()
        
        try:
            while len(completed_nodes) + len(failed_nodes) < len(self.graph.nodes()):
                # Get nodes ready for execution
                ready_nodes = self.get_ready_nodes(completed_nodes)
                ready_nodes = [n for n in ready_nodes if n not in failed_nodes]
                
                if not ready_nodes:
                    # Check if we're stuck (all remaining nodes depend on failed nodes)
                    remaining = set(self.graph.nodes()) - completed_nodes - failed_nodes
                    if remaining:
                        self.logger.warning(f"Workflow stuck: {len(remaining)} nodes blocked by failures")
                        break
                    break
                
                # Limit concurrent execution
                batch_size = min(len(ready_nodes), self.max_concurrent)
                current_batch = ready_nodes[:batch_size]
                
                self.logger.info(f"Executing batch of {len(current_batch)} nodes")
                
                # Execute batch concurrently
                tasks = [
                    self.execute_node(node_id, executor_func)
                    for node_id in current_batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for node_id, result in zip(current_batch, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Node {node_id} failed with exception: {result}")
                        failed_nodes.add(node_id)
                        self.execution_state.failed_nodes += 1
                    elif result.success:
                        completed_nodes.add(node_id)
                        self.execution_state.completed_nodes += 1
                    else:
                        # Handle retry logic here
                        node = self.graph.nodes[node_id]["node"]
                        if node.config.retry_count > 0 and node_id not in failed_nodes:
                            self.logger.info(f"Retrying node {node_id}")
                            node.config.retry_count -= 1
                            node.update_status(NodeStatus.RETRYING)
                        else:
                            failed_nodes.add(node_id)
                            self.execution_state.failed_nodes += 1
                            
                            if node.config.skip_on_failure:
                                self.logger.info(f"Skipping node {node_id} due to failure")
                                self.execution_state.skipped_nodes += 1
                            else:
                                self.logger.error(f"Node {node_id} failed permanently")
                
                # Update execution state
                self.execution_state.total_nodes = len(self.graph.nodes())
                
                self.logger.info(
                    f"Progress: {self.execution_state.progress:.1f}% "
                    f"({self.execution_state.completed_nodes}/"
                    f"{self.execution_state.total_nodes} completed, "
                    f"{self.execution_state.failed_nodes} failed, "
                    f"{self.execution_state.skipped_nodes} skipped)"
                )
        
        except Exception as e:
            self.logger.error(f"DAG execution failed: {e}")
            raise
        
        finally:
            self.execution_state.end_time = datetime.now()
            self.logger.info(
                f"DAG execution completed in {self.execution_state.duration:.2f}s"
            )
        
        return self.execution_state
    
    def save_state(self, filepath: str) -> None:
        """Save DAG state to file."""
        state = {
            "nodes": [
                node_data["node"].to_dict()
                for _, node_data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "data": self.graph[u][v]
                }
                for u, v in self.graph.edges()
            ],
            "execution_state": {
                "workflow_id": self.execution_state.workflow_id if self.execution_state else None,
                "start_time": self.execution_state.start_time.isoformat() if self.execution_state else None,
                "end_time": self.execution_state.end_time.isoformat() if self.execution_state and self.execution_state.end_time else None,
                "total_nodes": self.execution_state.total_nodes if self.execution_state else 0,
                "completed_nodes": self.execution_state.completed_nodes if self.execution_state else 0,
                "failed_nodes": self.execution_state.failed_nodes if self.execution_state else 0,
                "skipped_nodes": self.execution_state.skipped_nodes if self.execution_state else 0
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"DAG state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load DAG state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.graph.clear()
        
        # Load nodes
        for node_data in state["nodes"]:
            node = DAGNode.from_dict(node_data)
            self.add_node(node)
        
        # Load edges
        for edge_data in state["edges"]:
            self.add_edge(edge_data["from"], edge_data["to"], **edge_data.get("data", {}))
        
        self.logger.info(f"DAG state loaded from {filepath}")
