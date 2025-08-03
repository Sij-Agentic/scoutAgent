"""
Parallel Executor for DAG orchestration.

This module provides the async execution engine for running DAG nodes,
including agent execution, custom functions, and error handling.
"""

import asyncio
from typing import Any, Dict, Callable, Optional
from datetime import datetime
import traceback

from .node import DAGNode, NodeStatus, NodeResult, NodeType
from ..agents.base import AgentRegistry
from ..custom_logging.logger import get_logger


class NodeExecutor:
    """Handles execution of individual DAG nodes."""
    
    def __init__(self):
        self.logger = get_logger("NodeExecutor")
        self.registry = AgentRegistry()
    
    async def execute_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """Execute a single DAG node based on its type."""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing {node.node_type.value} node: {node.name}")
            
            if node.node_type == NodeType.AGENT:
                result = await self._execute_agent_node(node, inputs)
            elif node.node_type == NodeType.FUNCTION:
                result = await self._execute_function_node(node, inputs)
            elif node.node_type == NodeType.CONDITIONAL:
                result = await self._execute_conditional_node(node, inputs)
            elif node.node_type == NodeType.LOOP:
                result = await self._execute_loop_node(node, inputs)
            elif node.node_type == NodeType.ROOT:
                result = await self._execute_root_node(node, inputs)
            else:
                raise ValueError(f"Unsupported node type: {node.node_type}")
            
            end_time = datetime.now()
            
            return NodeResult(
                success=True,
                output=result,
                start_time=start_time,
                end_time=end_time,
                metadata={"node_type": node.node_type.value}
            )
            
        except asyncio.TimeoutError:
            self.logger.error(f"Node {node.name} timed out after {node.config.timeout_seconds}s")
            return NodeResult(
                success=False,
                error=f"Timeout after {node.config.timeout_seconds}s",
                start_time=start_time,
                end_time=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Node {node.name} failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return NodeResult(
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                metadata={"traceback": traceback.format_exc()}
            )
    
    async def _execute_agent_node(self, node: DAGNode, inputs: Dict[str, Any]) -> Any:
        """Execute an agent-based node."""
        if not node.agent_name:
            raise ValueError("Agent node missing agent_name")
        
        agent_class = self.registry.get_agent_class(node.agent_name)
        if not agent_class:
            raise ValueError(f"Unknown agent: {node.agent_name}")
        
        # Create agent instance
        agent = agent_class()
        
        # Prepare agent inputs
        agent_inputs = self._prepare_agent_inputs(inputs, node.inputs)
        
        # Execute agent
        self.logger.info(f"Running agent: {node.agent_name}")
        
        # Use asyncio.wait_for for timeout handling
        return await asyncio.wait_for(
            agent.execute(**agent_inputs),
            timeout=node.config.timeout_seconds
        )
    
    async def _execute_function_node(self, node: DAGNode, inputs: Dict[str, Any]) -> Any:
        """Execute a custom function node."""
        if not node.function:
            raise ValueError("Function node missing function name")
        
        # Get function from registry or globals
        func = self._get_function(node.function)
        if not func:
            raise ValueError(f"Unknown function: {node.function}")
        
        # Prepare function inputs
        func_inputs = self._prepare_function_inputs(inputs, node.inputs)
        
        self.logger.info(f"Running function: {node.function}")
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(**func_inputs),
                timeout=node.config.timeout_seconds
            )
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, func, **func_inputs),
                timeout=node.config.timeout_seconds
            )
    
    async def _execute_conditional_node(self, node: DAGNode, inputs: Dict[str, Any]) -> Any:
        """Execute a conditional branching node."""
        # For now, implement basic conditional logic
        # This can be extended with more complex branching
        condition_func = node.inputs.get("condition")
        true_branch = node.inputs.get("true_branch")
        false_branch = node.inputs.get("false_branch")
        
        if not condition_func:
            raise ValueError("Conditional node missing condition")
        
        # Evaluate condition
        if callable(condition_func):
            if asyncio.iscoroutinefunction(condition_func):
                result = await condition_func(inputs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, condition_func, inputs)
        else:
            result = bool(condition_func)
        
        return {"branch": true_branch if result else false_branch, "condition_result": result}
    
    async def _execute_loop_node(self, node: DAGNode, inputs: Dict[str, Any]) -> Any:
        """Execute a loop node."""
        # Basic loop implementation
        items = node.inputs.get("items", [])
        loop_func = node.inputs.get("loop_function")
        
        if not loop_func:
            raise ValueError("Loop node missing loop_function")
        
        results = []
        
        for item in items:
            if asyncio.iscoroutinefunction(loop_func):
                result = await loop_func(item, inputs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, loop_func, item, inputs)
            results.append(result)
        
        return {"results": results, "count": len(results)}
    
    async def _execute_root_node(self, node: DAGNode, inputs: Dict[str, Any]) -> Any:
        """Execute the root node (usually just passes through inputs)."""
        return {"status": "started", "inputs": inputs}
    
    def _prepare_agent_inputs(self, dependency_inputs: Dict[str, Any], 
                            node_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for agent execution."""
        combined_inputs = {}
        
        # Merge dependency outputs
        for dep_id, dep_outputs in dependency_inputs.items():
            combined_inputs[f"dep_{dep_id}"] = dep_outputs
        
        # Add direct node inputs
        combined_inputs.update(node_inputs)
        
        return combined_inputs
    
    def _prepare_function_inputs(self, dependency_inputs: Dict[str, Any],
                               node_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for function execution."""
        combined_inputs = {}
        
        # Merge dependency outputs
        for dep_id, dep_outputs in dependency_inputs.items():
            if isinstance(dep_outputs, dict):
                combined_inputs.update(dep_outputs)
            else:
                combined_inputs[f"dep_{dep_id}"] = dep_outputs
        
        # Add direct node inputs (override dependency inputs if needed)
        combined_inputs.update(node_inputs)
        
        return combined_inputs
    
    def _get_function(self, function_name: str) -> Optional[Callable]:
        """Get a function by name from registry or globals."""
        # Try to get from global namespace
        global_vars = globals()
        if function_name in global_vars:
            return global_vars[function_name]
        
        # Try to import from string
        try:
            module_name, func_name = function_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[func_name])
            return getattr(module, func_name)
        except (ValueError, ImportError, AttributeError):
            pass
        
        return None


class ParallelExecutor:
    """Orchestrates parallel execution of DAG nodes."""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.node_executor = NodeExecutor()
        self.logger = get_logger("ParallelExecutor")
    
    async def execute_dag(self, dag_engine, progress_callback: Optional[Callable] = None) -> None:
        """Execute the entire DAG with parallel processing."""
        
        async def executor_func(node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
            """Wrapper for node execution."""
            return await self.node_executor.execute_node(node, inputs)
        
        # Execute with progress callback
        if progress_callback:
            original_execute = executor_func
            
            async def progress_executor(node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
                result = await original_execute(node, inputs)
                progress_callback(node, result)
                return result
            
            executor_func = progress_executor
        
        return await dag_engine.execute(executor_func)
    
    async def execute_single_node(self, node: DAGNode, inputs: Dict[str, Any]) -> NodeResult:
        """Execute a single node (for testing)."""
        return await self.node_executor.execute_node(node, inputs)
