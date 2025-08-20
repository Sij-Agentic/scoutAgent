"""
AgentCodeExecutor - Wrapper combining PreludeService + CodeExecutionService for agent-agnostic execution.

This service provides a unified interface for executing agent code with proper prelude injection,
result extraction, and error handling.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from scout_agent.service_registry import ServiceBase, service, requires, inject
from scout_agent.custom_logging import get_logger
from scout_agent.services.agents.code.service import CodeExecutionService, CodeExecResult


@service(name="agent_code_executor", singleton=True)
@requires("prelude")
@requires("stage_message")
@requires("code_execution", optional=True)
class AgentCodeExecutor(ServiceBase):
    """
    Wrapper service combining PreludeService + CodeExecutionService for agent-agnostic execution.
    
    This service provides:
    - Prelude injection and code wrapping
    - Result extraction and formatting
    - Error handling and logging
    - Execution context management
    """
    
    def __init__(self):
        """Initialize the agent code executor."""
        super().__init__(name="agent_code_executor", version="1.0.0")
        self.logger = get_logger("service.agent_code_executor")
        self.prelude_service = None
        self.message_service = None
        self.code_execution_service = None
    
    @inject("prelude")
    def set_prelude_service(self, prelude_service):
        """Inject prelude service dependency."""
        self.prelude_service = prelude_service
    
    @inject("stage_message")
    def set_message_service(self, message_service):
        """Inject stage message service dependency."""
        self.message_service = message_service
    
    @inject("code_execution")
    def set_code_execution_service(self, code_execution_service):
        """Inject code execution service dependency."""
        self.code_execution_service = code_execution_service
    
    async def _initialize(self, registry) -> bool:
        """Initialize the agent code executor."""
        self.logger.info("Initializing agent code executor")
        
        try:
            # Create fallback services if not injected
            if not self.prelude_service:
                from .prelude_service import PreludeService
                self.prelude_service = PreludeService()
                await self.prelude_service._initialize(registry)
                await self.prelude_service._start()
                self.logger.info("Created fallback prelude service")
                
            if not self.code_execution_service:
                self.code_execution_service = CodeExecutionService()
                await self.code_execution_service._initialize(None)
                await self.code_execution_service._start()
                self.logger.info("Created fallback code execution service")
            
            self.logger.info("Agent code executor initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent code executor: {e}")
            return False
    
    async def _start(self) -> bool:
        """Start the agent code executor."""
        self.logger.info("Starting agent code executor")
        return True
    
    async def _stop(self) -> bool:
        """Stop the agent code executor."""
        self.logger.info("Stopping agent code executor")
        return True
    
    async def wrap_and_execute(self, node: Dict[str, Any], agent_id: str, stage: str, context: Dict[str, Any]) -> CodeExecResult:
        """
        Wrap node code with prelude and execute it.
        
        Args:
            node: Node configuration with code to execute
            agent_id: Agent identifier (e.g., "scout", "screener")
            stage: Stage name (e.g., "collect", "think")
            context: Execution context including run_dir, workflow_id, etc.
            
        Returns:
            CodeExecResult with execution details
        """
        if not self.prelude_service or not self.code_execution_service:
            raise RuntimeError("Required services not available")
        
        # Extract context
        run_dir = context.get("run_dir")
        workflow_id = context.get("workflow_id")
        node_id = node.get("id", f"{agent_id}_{stage}_node")
        
        if not run_dir:
            raise ValueError("run_dir must be provided in context")
        
        run_dir = Path(run_dir)
        
        try:
            # Generate prelude for this agent/stage
            prelude = await self.prelude_service.generate_prelude(
                agent_id=agent_id,
                stage=stage,
                run_dir=run_dir,
                context=context
            )
            
            # Get node code
            node_code = node.get("code", "").strip()
            if not node_code:
                return CodeExecResult(
                    success=False,
                    error=f"No code provided for node {node_id}",
                    exec_id=f"{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            # Combine prelude with node code
            full_code = f"{prelude}\n\n# Node code\n{node_code}"
            
            # Execute the wrapped code
            language = node.get("language", "python")
            timeout = context.get("timeout", 300)  # 5 minutes default
            
            self.logger.info(f"Executing wrapped code for {agent_id}_{stage} (node: {node_id})")
            
            result = await self.code_execution_service.execute_code(
                code=full_code,
                language=language,
                timeout=timeout
            )
            
            # Process the result
            if result.success:
                # Try to extract structured data from output
                extracted_data = self._extract_result_data(result, node)
                
                # Store result in message service if workflow_id provided
                if workflow_id and extracted_data and self.message_service:
                    # Use synchronous version instead of async
                    self.message_service.publish_stage_output(
                        workflow_id=workflow_id,
                        stage_id=f"{agent_id}_{stage}",
                        data=extracted_data
                    )
                
                self.logger.info(f"Successfully executed {node_id}")
            else:
                self.logger.error(f"Execution failed for {node_id}: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in wrap_and_execute for {node_id}: {e}")
            return CodeExecResult(
                success=False,
                error=f"Execution wrapper error: {str(e)}",
                exec_id=f"{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    async def execute_tool_nodes(self, nodes: List[Dict[str, Any]], agent_id: str, stage: str, context: Dict[str, Any]) -> List[CodeExecResult]:
        """
        Execute multiple tool nodes for an agent stage.
        
        Args:
            nodes: List of node configurations
            agent_id: Agent identifier
            stage: Stage name
            context: Execution context
            
        Returns:
            List of CodeExecResult objects
        """
        results = []
        
        # Execute nodes in parallel where possible
        # For now, execute sequentially to maintain order and dependencies
        for node in nodes:
            result = await self.wrap_and_execute(node, agent_id, stage, context)
            results.append(result)
            
            # If a node fails and it's marked as critical, stop execution
            if not result.success and node.get("critical", False):
                self.logger.error(f"Critical node {node.get('id')} failed, stopping execution")
                break
        
        successful_count = sum(1 for r in results if r.success)
        self.logger.info(f"Executed {len(results)} nodes for {agent_id}_{stage}: {successful_count} successful")
        
        return results
    
    def _extract_result_data(self, result, node):
        """
        Extract structured data from execution result.
        
        Args:
            result: Code execution result
            node: Node configuration
            
        Returns:
            Extracted data or None if no structured data found
        """
        if not result.success or not result.output:
            return None
        
        try:
            # Look for JSON output in stdout
            output_lines = result.output.strip().split('\n')
            
            # Try to find complete JSON objects in the output
            # First, try to parse the entire output as JSON
            try:
                full_output = result.output.strip()
                if full_output.startswith('{') and full_output.endswith('}'):
                    return json.loads(full_output)
            except json.JSONDecodeError:
                pass
                
            # Try to find JSON in individual lines
            for line in output_lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            
            # Look for specific markers in output
            json_markers = ['JSON_OUTPUT:', 'RESULT:', 'DATA:', 'DEBUG:']
            for marker in json_markers:
                if marker in result.output:
                    json_part = result.output.split(marker, 1)[1].strip()
                    try:
                        # Try to find where the JSON object ends
                        brace_count = 0
                        end_pos = 0
                        for i, char in enumerate(json_part):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                        
                        if end_pos > 0:
                            json_obj = json_part[:end_pos]
                            return json.loads(json_obj)
                        else:
                            return json.loads(json_part)
                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.debug(f"Failed to parse JSON after marker {marker}: {e}")
                        continue
            
            # Try to find any JSON-like structure in the output
            import re
            json_pattern = r'\{[^{}]*((\{[^{}]*\})[^{}]*)*\}'
            matches = re.findall(json_pattern, result.output)
            for match in matches:
                try:
                    if match and isinstance(match, tuple):
                        match = match[0]  # Get the first group
                    json_str = match.strip()
                    if json_str:
                        return json.loads(json_str)
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # If no structured data found, return basic execution info
            return {
                "execution_successful": True,
                "output_lines": len(output_lines),
                "execution_time": result.duration_ms,
                "node_id": node.get("id"),
                "tool": node.get("tool")
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to extract result data: {e}")
            return None
    
    async def execute_with_dependencies(self, node: Dict[str, Any], agent_id: str, stage: str, context: Dict[str, Any], dependencies: List[str] = None) -> CodeExecResult:
        """
        Execute a node with dependency data injection.
        
        Args:
            node: Node configuration
            agent_id: Agent identifier
            stage: Stage name
            context: Execution context
            dependencies: List of dependency stage IDs
            
        Returns:
            CodeExecResult with execution details
        """
        if dependencies and self.message_service:
            workflow_id = context.get("workflow_id")
            if workflow_id:
                # Get dependency data
                dependency_data = await self.message_service.get_dependency_data(
                    workflow_id=workflow_id,
                    stage_id=f"{agent_id}_{stage}",
                    dependency_stages=dependencies
                )
                
                # Inject dependency data into context
                context = context.copy()
                context["dependency_data"] = dependency_data
                
                self.logger.info(f"Injected dependency data for {agent_id}_{stage}: {list(dependency_data.keys())}")
        
        return await self.wrap_and_execute(node, agent_id, stage, context)
    
    async def execute_dag_nodes(self, dag_nodes: List[Dict[str, Any]], agent_id: str, stage: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a list of DAG nodes with dependency resolution.
        
        Args:
            dag_nodes: List of DAG node configurations
            agent_id: Agent identifier
            stage: Stage name
            context: Execution context
            
        Returns:
            Dictionary with execution results and summary
        """
        if not dag_nodes:
            return {"results": [], "summary": {"total": 0, "successful": 0, "failed": 0}}
        
        # Simple dependency resolution - execute in order for now
        # TODO: Implement proper topological sorting for parallel execution
        results = []
        successful = 0
        failed = 0
        
        for node in dag_nodes:
            node_id = node.get("id", f"node_{len(results)}")
            dependencies = node.get("dependencies", [])
            
            self.logger.info(f"Executing DAG node {node_id} with dependencies: {dependencies}")
            
            result = await self.execute_with_dependencies(
                node=node,
                agent_id=agent_id,
                stage=stage,
                context=context,
                dependencies=dependencies
            )
            
            results.append({
                "node_id": node_id,
                "result": result,
                "success": result.success
            })
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
                # Stop on critical failure
                if node.get("critical", False):
                    self.logger.error(f"Critical DAG node {node_id} failed, stopping execution")
                    break
        
        summary = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "completion_rate": successful / len(results) if results else 0
        }
        
        self.logger.info(f"DAG execution complete for {agent_id}_{stage}: {summary}")
        
        return {
            "results": results,
            "summary": summary
        }
