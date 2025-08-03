"""
DAG Node definitions for ScoutAgent orchestration.

This module defines the data structures and classes for DAG nodes,
representing individual tasks (agent executions or custom functions)
within the workflow system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import uuid
from datetime import datetime


class NodeStatus(Enum):
    """Possible states of a DAG node during execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class NodeType(Enum):
    """Types of DAG nodes supported by the system."""
    AGENT = "agent"  # Agent execution node
    FUNCTION = "function"  # Custom function node
    CONDITIONAL = "conditional"  # Branching logic
    LOOP = "loop"  # Iterative processing
    ROOT = "root"  # Workflow anchor node


@dataclass
class NodeConfig:
    """Configuration for a DAG node."""
    retry_count: int = 3
    timeout_seconds: int = 300
    skip_on_failure: bool = False
    parallel: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Result of a node execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Duration of execution in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class DAGNode:
    """A single node in the DAG representing a task."""
    
    # Core identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Node type and execution
    node_type: NodeType = NodeType.AGENT
    agent_name: Optional[str] = None  # For agent nodes
    function: Optional[str] = None  # For function nodes (function name)
    
    # Configuration
    config: NodeConfig = field(default_factory=NodeConfig)
    
    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[NodeResult] = None
    
    # Input/output handling
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize any derived attributes."""
        if not self.name:
            self.name = f"{self.node_type.value}_{self.node_id}"
    
    def is_ready(self, completed_nodes: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_nodes for dep in self.dependencies)
    
    def update_status(self, status: NodeStatus, result: Optional[NodeResult] = None):
        """Update node status and result."""
        self.status = status
        self.result = result
        self.updated_at = datetime.now()
        
        if result and result.start_time:
            self.outputs.update({"result": result.output})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "node_type": self.node_type.value,
            "agent_name": self.agent_name,
            "function": self.function,
            "config": {
                "retry_count": self.config.retry_count,
                "timeout_seconds": self.config.timeout_seconds,
                "skip_on_failure": self.config.skip_on_failure,
                "parallel": self.config.parallel,
                "metadata": self.config.metadata
            },
            "status": self.status.value,
            "result": {
                "success": self.result.success if self.result else None,
                "output": self.result.output if self.result else None,
                "error": self.result.error if self.result else None,
                "duration": self.result.duration if self.result else None,
                "metadata": self.result.metadata if self.result else {}
            } if self.result else None,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGNode":
        """Create node from dictionary."""
        node = cls(
            node_id=data["node_id"],
            name=data["name"],
            description=data.get("description", ""),
            node_type=NodeType(data["node_type"]),
            agent_name=data.get("agent_name"),
            function=data.get("function"),
            config=NodeConfig(**data.get("config", {})),
            dependencies=data.get("dependencies", []),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {})
        )
        
        if data.get("result"):
            result_data = data["result"]
            node.result = NodeResult(
                success=result_data.get("success", False),
                output=result_data.get("output"),
                error=result_data.get("error"),
                metadata=result_data.get("metadata", {})
            )
        
        return node


@dataclass
class Edge:
    """Represents a directed edge between DAG nodes."""
    
    from_node: str
    to_node: str
    
    # Data flow specification
    input_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "input_mapping": self.input_mapping,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """Create edge from dictionary."""
        return cls(
            from_node=data["from_node"],
            to_node=data["to_node"],
            input_mapping=data.get("input_mapping", {})
        )
