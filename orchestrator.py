"""
High-level orchestrator for ScoutAgent workflows.

Provides a unified interface for managing DAG workflows, coordinating agents,
handling communication, monitoring progress, and managing checkpoints.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import dataclass, field

from .dag.engine import DAGEngine
from .dag.templates.pain_point_discovery import create_pain_point_discovery_workflow
from .dag.templates.market_research import create_market_research_workflow, create_competitor_analysis_workflow, create_customer_research_workflow
from .agents.base import get_registry, AgentInput
from .agents.communication import get_event_bus
from .memory.persistence import MemoryManager
from .memory.context import get_context_manager
from .memory.checkpoints import CheckpointManager
from .custom_logging.logger import get_logger


class WorkflowStatus:
    """Status tracking for workflows."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowExecution:
    """Represents a workflow execution."""
    
    execution_id: str
    workflow_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_id: Optional[str] = None


class ScoutOrchestrator:
    """Main orchestrator for ScoutAgent workflows."""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.logger = get_logger("orchestrator")
        
        # Core components
        self.registry = get_registry()
        self.memory_manager = MemoryManager()
        self.context_manager = get_context_manager()
        self.checkpoint_manager = CheckpointManager()
        
        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_templates = {
            "pain_point_discovery": create_pain_point_discovery_workflow,
            "market_research": create_market_research_workflow,
            "competitor_analysis": create_competitor_analysis_workflow,
            "customer_research": create_customer_research_workflow
        }
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        self.logger.info("ScoutOrchestrator initialized")
    
    async def create_workflow(self, 
                            workflow_type: str, 
                            config: Dict[str, Any],
                            execution_id: Optional[str] = None) -> str:
        """
        Create a new workflow execution.
        
        Args:
            workflow_type: Type of workflow (e.g., 'pain_point_discovery')
            config: Configuration for the workflow
            execution_id: Optional custom execution ID
            
        Returns:
            execution_id for tracking the workflow
        """
        
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        execution_id = execution_id or str(uuid.uuid4())[:8]
        
        # Create workflow execution record
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_type=workflow_type,
            status=WorkflowStatus.PENDING,
            start_time=datetime.now(),
            config=config
        )
        
        self.active_workflows[execution_id] = execution
        
        # Create context for this execution
        context = self.context_manager.get_or_create_context(execution_id)
        context.set("workflow_config", config)
        context.set("workflow_type", workflow_type)
        context.set("execution_id", execution_id)
        
        # Initialize message bus
        await self.registry.start_message_bus(execution_id)
        
        self.logger.info(f"Created workflow: {workflow_type} ({execution_id})")
        
        return execution_id
    
    async def execute_workflow(self, 
                             execution_id: str,
                             checkpoint_resume: bool = False) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            execution_id: ID of the workflow to execute
            checkpoint_resume: Whether to resume from checkpoint
            
        Returns:
            Workflow execution results
        """
        
        if execution_id not in self.active_workflows:
            raise ValueError(f"Unknown execution ID: {execution_id}")
        
        execution = self.active_workflows[execution_id]
        
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Emit start event
            await self._emit_event("workflow_started", {
                "execution_id": execution_id,
                "workflow_type": execution.workflow_type,
                "config": execution.config
            })
            
            # Resume from checkpoint if requested
            if checkpoint_resume:
                checkpoint = await self.checkpoint_manager.restore_checkpoint(execution_id)
                if checkpoint:
                    execution.checkpoint_id = checkpoint.checkpoint_id
                    self.logger.info(f"Resumed from checkpoint: {checkpoint.checkpoint_id}")
            
            # Create workflow DAG
            workflow_func = self.workflow_templates[execution.workflow_type]
            dag_engine = workflow_func(**execution.config)
            
            # Set up progress monitoring
            event_bus = get_event_bus(execution_id)
            
            def on_node_complete(node_id: str, result: Dict[str, Any]):
                asyncio.create_task(self._emit_event("node_completed", {
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "result": result
                }))
            
            def on_node_error(node_id: str, error: str):
                asyncio.create_task(self._emit_event("node_error", {
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "error": error
                }))
            
            event_bus.on("node_completed", on_node_complete)
            event_bus.on("node_error", on_node_error)
            
            # Create checkpoint before execution
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                execution_id,
                {
                    "workflow_type": execution.workflow_type,
                    "config": execution.config,
                    "status": "starting"
                }
            )
            execution.checkpoint_id = checkpoint_id
            
            # Execute the workflow
            results = await dag_engine.execute()
            
            # Update execution
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.results = results
            
            # Save final results
            await self.memory_manager.save_workflow_state(
                execution_id,
                {
                    "execution": execution.__dict__,
                    "results": results,
                    "final_checkpoint": checkpoint_id
                }
            )
            
            # Emit completion event
            await self._emit_event("workflow_completed", {
                "execution_id": execution_id,
                "results": results,
                "execution_time": (execution.end_time - execution.start_time).total_seconds()
            })
            
            self.logger.info(f"Workflow completed: {execution_id}")
            
            return results
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error = str(e)
            
            await self._emit_event("workflow_failed", {
                "execution_id": execution_id,
                "error": str(e)
            })
            
            self.logger.error(f"Workflow failed: {execution_id} - {e}")
            raise
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow."""
        if execution_id not in self.active_workflows:
            return False
        
        execution = self.active_workflows[execution_id]
        if execution.status != WorkflowStatus.RUNNING:
            return False
        
        execution.status = WorkflowStatus.PAUSED
        
        # Create checkpoint
        checkpoint_id = await self.checkpoint_manager.create_checkpoint(
            execution_id,
            {
                "workflow_type": execution.workflow_type,
                "config": execution.config,
                "status": "paused"
            }
        )
        
        await self._emit_event("workflow_paused", {
            "execution_id": execution_id,
            "checkpoint_id": checkpoint_id
        })
        
        return True
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a workflow."""
        if execution_id not in self.active_workflows:
            return False
        
        execution = self.active_workflows[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        
        await self._emit_event("workflow_cancelled", {
            "execution_id": execution_id
        })
        
        return True
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        if execution_id not in self.active_workflows:
            return None
        
        execution = self.active_workflows[execution_id]
        return {
            "execution_id": execution.execution_id,
            "workflow_type": execution.workflow_type,
            "status": execution.status,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "config": execution.config,
            "error": execution.error
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        return [
            {
                "execution_id": exec.execution_id,
                "workflow_type": exec.workflow_type,
                "status": exec.status,
                "start_time": exec.start_time.isoformat()
            }
            for exec in self.active_workflows.values()
        ]
    
    async def get_workflow_results(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a completed workflow."""
        if execution_id not in self.active_workflows:
            return None
        
        execution = self.active_workflows[execution_id]
        if execution.status != WorkflowStatus.COMPLETED:
            return None
        
        return execution.results
    
    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def off_event(self, event_type: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered handlers."""
        if event_type not in self.event_handlers:
            return
        
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop all message buses
        for context_id in list(self.registry._message_buses.keys()):
            await self.registry.stop_message_bus(context_id)
        
        # Clear message buses
        self.registry.clear_message_buses()
        
        # Clear active workflows
        self.active_workflows.clear()
        
        self.logger.info("Orchestrator cleanup completed")
    
    async def get_execution_history(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get the execution history for a workflow."""
        # This would integrate with the DAG engine's execution tracking
        # For now, return basic info
        if execution_id not in self.active_workflows:
            return []
        
        execution = self.active_workflows[execution_id]
        return [{
            "timestamp": execution.start_time.isoformat(),
            "event": "workflow_started",
            "data": {
                "workflow_type": execution.workflow_type,
                "config": execution.config
            }
        }]
    
    async def get_workflow_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of a workflow."""
        if execution_id not in self.active_workflows:
            return {}
        
        execution = self.active_workflows[execution_id]
        
        # Get checkpoint info
        checkpoints = await self.checkpoint_manager.list_checkpoints(execution_id)
        
        # Get context info
        context = self.context_manager.get_context(execution_id)
        context_keys = list(context.data.keys()) if context else []
        
        return {
            "execution_id": execution_id,
            "workflow_type": execution.workflow_type,
            "status": execution.status,
            "duration": (
                (execution.end_time - execution.start_time).total_seconds()
                if execution.end_time else
                (datetime.now() - execution.start_time).total_seconds()
            ),
            "checkpoints": len(checkpoints),
            "context_keys": context_keys,
            "config": execution.config,
            "results_summary": {
                "has_results": bool(execution.results),
                "result_keys": list(execution.results.keys()) if execution.results else []
            }
        }


# Global orchestrator instance
_orchestrator: Optional[ScoutOrchestrator] = None


def get_orchestrator(config: Optional[Any] = None) -> ScoutOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ScoutOrchestrator(config)
    return _orchestrator


# Convenience functions
async def run_pain_point_discovery(
    target_market: str,
    research_focus: str = "comprehensive",
    max_pain_points: int = 10,
    validation_depth: str = "moderate",
    **kwargs
) -> Dict[str, Any]:
    """Run a pain point discovery workflow."""
    orchestrator = get_orchestrator()
    
    config = {
        "target_market": target_market,
        "research_focus": research_focus,
        "max_pain_points": max_pain_points,
        "validation_depth": validation_depth,
        **kwargs
    }
    
    execution_id = await orchestrator.create_workflow("pain_point_discovery", config)
    return await orchestrator.execute_workflow(execution_id)


async def run_market_research(
    market: str,
    research_scope: str = "comprehensive",
    **kwargs
) -> Dict[str, Any]:
    """Run a market research workflow."""
    orchestrator = get_orchestrator()
    
    config = {
        "market": market,
        "research_scope": research_scope,
        **kwargs
    }
    
    execution_id = await orchestrator.create_workflow("market_research", config)
    return await orchestrator.execute_workflow(execution_id)
