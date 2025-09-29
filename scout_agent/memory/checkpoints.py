"""
Checkpoint system for workflow resume capabilities.

Provides checkpoint creation, restoration, and management for long-running
workflows, enabling recovery from failures and incremental progress.
"""

import json
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dag.engine import DAGEngine
from dag.node import DAGNode, NodeStatus
from memory.persistence import get_memory_manager
from memory.context import get_context_manager
from custom_logging.logger import get_logger


@dataclass
class Checkpoint:
    """Represents a single checkpoint in workflow execution."""
    
    checkpoint_id: str
    workflow_id: str
    timestamp: datetime
    
    # DAG state
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution progress
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    running_nodes: Set[str] = field(default_factory=set)
    
    # Context state
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "nodes": self.nodes,
            "edges": self.edges,
            "completed_nodes": list(self.completed_nodes),
            "failed_nodes": list(self.failed_nodes),
            "running_nodes": list(self.running_nodes),
            "context_snapshot": self.context_snapshot,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            nodes=data.get("nodes", {}),
            edges=data.get("edges", []),
            completed_nodes=set(data.get("completed_nodes", [])),
            failed_nodes=set(data.get("failed_nodes", [])),
            running_nodes=set(data.get("running_nodes", [])),
            context_snapshot=data.get("context_snapshot", {}),
            metadata=data.get("metadata", {})
        )


class CheckpointManager:
    """Manages checkpoints for workflow execution."""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.context_manager = get_context_manager()
        self.logger = get_logger("CheckpointManager")
    
    def create_checkpoint(self, dag_engine: DAGEngine, context_id: str, 
                         checkpoint_id: Optional[str] = None) -> str:
        """Create a checkpoint of current workflow state."""
        
        if not checkpoint_id:
            checkpoint_id = f"{dag_engine.execution_state.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Collect node states
        nodes_data = {}
        for node_id, node_data in dag_engine.graph.nodes(data=True):
            node = node_data.get("node")
            if node:
                nodes_data[node_id] = node.to_dict()
        
        # Collect edges
        edges_data = []
        for u, v in dag_engine.graph.edges():
            edge_data = dag_engine.graph[u][v]
            edges_data.append({
                "from": u,
                "to": v,
                "data": dict(edge_data)
            })
        
        # Get context snapshot
        context = self.context_manager.get_context(context_id)
        context_snapshot = context.to_dict() if context else {}
        
        # Create checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=dag_engine.execution_state.workflow_id,
            timestamp=datetime.now(),
            nodes=nodes_data,
            edges=edges_data,
            completed_nodes=set(),  # Will be populated from node states
            failed_nodes=set(),
            running_nodes=set(),
            context_snapshot=context_snapshot,
            metadata={
                "total_nodes": len(nodes_data),
                "progress": dag_engine.execution_state.progress if dag_engine.execution_state else 0
            }
        )
        
        # Determine node states from saved data
        for node_id, node_data in nodes_data.items():
            status = node_data.get("status")
            if status == "completed":
                checkpoint.completed_nodes.add(node_id)
            elif status == "failed":
                checkpoint.failed_nodes.add(node_id)
            elif status == "running":
                checkpoint.running_nodes.add(node_id)
        
        # Save checkpoint
        self.memory_manager.save_checkpoint(
            dag_engine.execution_state.workflow_id,
            checkpoint.to_dict()
        )
        
        self.logger.info(
            f"Created checkpoint {checkpoint_id} "
            f"({len(checkpoint.completed_nodes)}/{len(nodes_data)} completed)"
        )
        
        return checkpoint_id
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[DAGEngine]:
        """Restore DAG engine from checkpoint."""
        
        checkpoint_data = self.memory_manager.load_checkpoint(checkpoint_id)
        if not checkpoint_data:
            self.logger.error(f"Checkpoint not found: {checkpoint_id}")
            return None
        
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        
        # Create new DAG engine
        from ..dag.engine import DAGEngine
        dag_engine = DAGEngine()
        
        # Restore nodes
        for node_id, node_data in checkpoint.nodes.items():
            from ..dag.node import DAGNode
            node = DAGNode.from_dict(node_data)
            dag_engine.add_node(node)
        
        # Restore edges
        for edge_data in checkpoint.edges:
            dag_engine.add_edge(edge_data["from"], edge_data["to"])
        
        # Restore context
        context = self.context_manager.get_or_create_context(
            checkpoint.context_snapshot.get("context_id", checkpoint.workflow_id)
        )
        context.from_dict(checkpoint.context_snapshot)
        
        self.logger.info(
            f"Restored from checkpoint {checkpoint_id} "
            f"({len(checkpoint.completed_nodes)} completed, "
            f"{len(checkpoint.failed_nodes)} failed)"
        )
        
        return dag_engine
    
    def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a workflow."""
        checkpoint_ids = self.memory_manager.list_checkpoints(workflow_id)
        checkpoints = []
        
        for checkpoint_id in checkpoint_ids:
            checkpoint_data = self.memory_manager.load_checkpoint(checkpoint_id)
            if checkpoint_data:
                checkpoint = Checkpoint.from_dict(checkpoint_data)
                checkpoints.append({
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "timestamp": checkpoint.timestamp,
                    "completed_nodes": len(checkpoint.completed_nodes),
                    "failed_nodes": len(checkpoint.failed_nodes),
                    "total_nodes": checkpoint.metadata.get("total_nodes", 0),
                    "progress": checkpoint.metadata.get("progress", 0)
                })
        
        return checkpoints
    
    def get_latest_checkpoint(self, workflow_id: str) -> Optional[str]:
        """Get the latest checkpoint ID for a workflow."""
        checkpoints = self.list_checkpoints(workflow_id)
        if checkpoints:
            return checkpoints[0]["checkpoint_id"]
        return None
    
    def should_create_checkpoint(self, dag_engine: DAGEngine, 
                               checkpoint_interval: int = 5) -> bool:
        """Determine if a checkpoint should be created."""
        if not dag_engine.execution_state:
            return False
        
        # Create checkpoint every N completed nodes
        completed = dag_engine.execution_state.completed_nodes
        return completed > 0 and completed % checkpoint_interval == 0
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            # Memory manager doesn't have delete_checkpoint, so we'll use the store directly
            memory_manager = get_memory_manager()
            key = f"checkpoint_{checkpoint_id}"
            
            # This is a workaround since MemoryStore interface is abstract
            if hasattr(memory_manager.store, 'delete'):
                success = memory_manager.store.delete(key)
                if success:
                    self.logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return success
            
            # Fallback: manually delete if using FileMemoryStore
            from .persistence import FileMemoryStore
            if isinstance(memory_manager.store, FileMemoryStore):
                from pathlib import Path
                base_path = Path(memory_manager.store.base_path)
                for ext in ['.json', '.pkl']:
                    file_path = base_path / f"{hashlib.md5(key.encode()).hexdigest()}{ext}"
                    if file_path.exists():
                        file_path.unlink()
                        self.logger.info(f"Deleted checkpoint file: {checkpoint_id}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def cleanup_old_checkpoints(self, workflow_id: str, keep_last: int = 5) -> int:
        """Clean up old checkpoints, keeping only the latest N."""
        checkpoints = self.list_checkpoints(workflow_id)
        
        if len(checkpoints) <= keep_last:
            return 0
        
        deleted = 0
        for checkpoint_info in checkpoints[keep_last:]:
            checkpoint_id = checkpoint_info["checkpoint_id"]
            if self.delete_checkpoint(checkpoint_id):
                deleted += 1
        
        self.logger.info(f"Cleaned up {deleted} old checkpoints for {workflow_id}")
        return deleted


# Global checkpoint manager
import hashlib
_global_checkpoint_manager = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance."""
    global _global_checkpoint_manager
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager()
    return _global_checkpoint_manager
