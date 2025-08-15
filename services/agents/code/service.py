"""
Code Execution Service Implementation.

This module provides a service for code generation, validation, and execution
within sandboxed environments. It handles code operations that were previously
managed by code agents.
"""

import os
import uuid
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum

from scout_agent.service_registry import ServiceBase, service, requires, inject
from scout_agent.custom_logging import get_logger


class ExecutionEnvironment(Enum):
    """Supported execution environments for running code."""
    LOCAL = "local"
    DOCKER = "docker"
    WASM = "wasm"  # WebAssembly


class CodeExecResult:
    """Result of a code execution operation."""
    
    def __init__(self, 
                success: bool,
                output: str = "",
                error: str = "",
                exec_id: str = "",
                duration_ms: int = 0):
        """
        Initialize the execution result.
        
        Args:
            success: Whether execution was successful
            output: Standard output (stdout)
            error: Error output (stderr)
            exec_id: Unique execution ID
            duration_ms: Execution time in milliseconds
        """
        self.success = success
        self.output = output
        self.error = error
        self.exec_id = exec_id or str(uuid.uuid4())
        self.duration_ms = duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exec_id": self.exec_id,
            "duration_ms": self.duration_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeExecResult':
        """Create execution result from dictionary."""
        return cls(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error", ""),
            exec_id=data.get("exec_id", ""),
            duration_ms=data.get("duration_ms", 0)
        )


@service(name="code_execution", singleton=True)
@requires("config", optional=True)
@requires("logging", optional=True)
class CodeExecutionService(ServiceBase):
    """
    Service for code generation, validation, and execution.
    
    This service handles code operations within sandboxed environments,
    providing security isolation and resource limits.
    """
    
    def __init__(self):
        """Initialize the code execution service."""
        super().__init__(name="code_execution", version="1.0.0")
        self.logger = get_logger("service.code_execution")
        self.config = None
        self.working_dir = None
        self.execution_history = {}
        self.resource_limits = {
            "timeout_seconds": 30,
            "max_memory_mb": 500,
            "max_processes": 5,
            "network_access": False
        }
        self.supported_languages = {
            "python": {
                "extension": ".py",
                "command": "python",
                "docker_image": "python:3.10-slim"
            },
            "javascript": {
                "extension": ".js",
                "command": "node",
                "docker_image": "node:16-alpine"
            },
            "bash": {
                "extension": ".sh",
                "command": "bash",
                "docker_image": "bash:latest"
            }
            # More languages can be added here
        }
        
    def setup_direct(self, config=None, logger=None, working_dir=None):
        """
        Setup the service directly without using the service registry.
        Useful for testing and standalone usage.
        
        Args:
            config: Configuration service or dict
            logger: Logger instance
            working_dir: Working directory for code execution
        """
        if logger:
            self.logger = logger
        if config:
            self.config = config
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
    
    async def _initialize(self, registry) -> bool:
        """
        Initialize the code execution service.
        
        Args:
            registry: Service registry for accessing dependencies
            
        Returns:
            True if initialization was successful
        """
        self.logger.info("Initializing code execution service")
        
        try:
            # Handle direct initialization (no registry)
            if registry is None:
                self.logger.info("Initializing code execution service directly (no registry)")
                if not self.working_dir:
                    # Set up default working directory
                    self.working_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
                    self.logger.info(f"Created temporary working directory: {self.working_dir}")
                return True
                
            # Try to get config service if available
            try:
                self.config = registry.get_service("config")
            except Exception as e:
                self.logger.warning(f"Could not get config service: {e}. Using default configuration.")
                # Create a minimal config object for testing
                class MinimalConfig:
                    def get_config_value(self, key, default=None):
                        return default
                    def get_llm_backend_for_agent(self, agent_name):
                        return {}
                self.config = MinimalConfig()
            
            # Try to get logging service if we need to update logger
            try:
                logging_service = registry.get_service("logging")
                self.logger = logging_service.get_logger("code_execution_service")
            except Exception as e:
                # Keep using default logger if logging service not available
                self.logger.warning(f"Could not get logging service: {e}. Using default logger.")
            
            # Set up working directory
            try:
                working_dir = self.config.get_config_value(
                    "code_execution.working_dir",
                    "/tmp/scout_agent/code_execution"
                )
                self.working_dir = Path(working_dir)
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Using working directory: {self.working_dir}")
            except Exception as e:
                self.logger.warning(f"Error setting working directory from config: {e}. Using default.")
                self.working_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
                self.logger.info(f"Created temporary working directory: {self.working_dir}")
            
            # Get resource limits
            try:
                resource_limits = self.config.get_config_value(
                    "code_execution.resource_limits", 
                    self.resource_limits
                )
                self.resource_limits.update(resource_limits)
            except Exception as e:
                self.logger.warning(f"Error getting resource limits from config: {e}. Using defaults.")
            
            # Get LLM backend preference
            try:
                self.llm_backend_config = self.config.get_llm_backend_for_agent("code") or {}
            except Exception as e:
                self.logger.warning(f"Could not get LLM backend config: {e}. Using default.")
                self.llm_backend_config = {}
            
            self.logger.info("Code execution service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize code execution service: {e}")
            return False
    
    async def _start(self) -> bool:
        """
        Start the code execution service.
        
        Validates the execution environment and dependencies.
        
        Returns:
            True if startup was successful
        """
        self.logger.info("Starting code execution service")
        
        try:
            # Ensure working directory exists
            if not self.working_dir:
                self.working_dir = Path(tempfile.mkdtemp(prefix="code_exec_"))
                self.logger.info(f"Created temporary working directory: {self.working_dir}")
            elif not self.working_dir.exists():
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created working directory: {self.working_dir}")
            
            # Validate language support
            for lang, config in self.supported_languages.items():
                cmd = config["command"]
                try:
                    subprocess.run(
                        [cmd, "--version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self.logger.info(f"{lang.capitalize()} interpreter available: {cmd}")
                except (subprocess.SubprocessError, FileNotFoundError):
                    self.logger.warning(f"{lang.capitalize()} interpreter not found: {cmd}")
            
            self.logger.info("Code execution service started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start code execution service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """
        Stop the code execution service.
        
        Cleans up any running processes or temporary files.
        
        Returns:
            True if shutdown was successful
        """
        self.logger.info("Stopping code execution service")
        
        try:
            # Implement cleanup logic here
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop code execution service: {e}")
            return False
    
    async def generate_code(self, 
                          prompt: str,
                          language: str = "python",
                          context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Generate code using an LLM based on a prompt.
        
        Args:
            prompt: Description of the code to generate
            language: Target programming language
            context: Additional context for code generation
            
        Returns:
            Tuple of (success, generated_code)
        """
        self.logger.info(f"Generating {language} code from prompt")
        
        try:
            # TODO: Implement LLM integration for code generation
            # For now, this is a placeholder
            sample_code = f"# Generated {language} code based on prompt: {prompt}\n"
            
            if language == "python":
                sample_code += "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()"
            elif language == "javascript":
                sample_code += "function main() {\n    console.log('Hello, World!');\n}\n\nmain();"
            elif language == "bash":
                sample_code += "#!/bin/bash\necho 'Hello, World!'"
                
            return True, sample_code
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return False, f"# Error generating code: {str(e)}"
    
    def validate_code(self, code: str, language: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate code statically without execution.
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.logger.info(f"Validating {language} code")
        issues = []
        
        try:
            if language == "python":
                # Use pylint for validation
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
                    f.write(code)
                    f.flush()
                    
                    result = subprocess.run(
                        ["pylint", "--output-format=json", f.name],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.stdout.strip():
                        import json
                        pylint_issues = json.loads(result.stdout)
                        for issue in pylint_issues:
                            issues.append({
                                "line": issue.get("line", 0),
                                "column": issue.get("column", 0),
                                "message": issue.get("message", "Unknown issue"),
                                "severity": issue.get("type", "warning")
                            })
            
            # Add validators for other languages as needed
            
            # If no issues or only warnings, consider code valid
            is_valid = all(issue["severity"] != "error" for issue in issues)
            return is_valid, issues
            
        except Exception as e:
            self.logger.error(f"Code validation failed: {e}")
            issues.append({
                "line": 0,
                "column": 0,
                "message": f"Validation error: {str(e)}",
                "severity": "error"
            })
            return False, issues
    
    async def execute_code(self, 
                        code: str, 
                        language: str,
                        env: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
                        inputs: Optional[str] = None,
                        timeout: Optional[int] = None) -> CodeExecResult:
        """
        Execute code in the specified environment.
        
        Args:
            code: Code to execute
            language: Programming language
            env: Execution environment (local, docker, etc.)
            inputs: Optional inputs for stdin
            timeout: Execution timeout in seconds (overrides default)
            
        Returns:
            CodeExecResult with execution details
        """
        exec_id = str(uuid.uuid4())
        self.logger.info(f"Executing {language} code [id: {exec_id}]")
        
        # Use configured timeout or default
        timeout_seconds = timeout or self.resource_limits["timeout_seconds"]
        
        try:
            if language not in self.supported_languages:
                return CodeExecResult(
                    success=False,
                    error=f"Unsupported language: {language}",
                    exec_id=exec_id
                )
                
            lang_info = self.supported_languages[language]
            
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(
                suffix=lang_info["extension"],
                mode="w",
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
                
            # Choose execution strategy based on environment
            if env == ExecutionEnvironment.LOCAL:
                result = await self._execute_local(
                    temp_file_path, 
                    lang_info["command"],
                    inputs,
                    timeout_seconds,
                    exec_id
                )
                
            elif env == ExecutionEnvironment.DOCKER:
                result = await self._execute_docker(
                    temp_file_path,
                    lang_info["docker_image"],
                    inputs,
                    timeout_seconds,
                    exec_id
                )
                
            elif env == ExecutionEnvironment.WASM:
                # WebAssembly execution not yet implemented
                result = CodeExecResult(
                    success=False,
                    error="WebAssembly execution not yet implemented",
                    exec_id=exec_id
                )
            
            else:
                result = CodeExecResult(
                    success=False,
                    error=f"Unsupported execution environment: {env.value}",
                    exec_id=exec_id
                )
                
            # Store execution history
            self.execution_history[exec_id] = result.to_dict()
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return CodeExecResult(
                success=False,
                error=f"Execution error: {str(e)}",
                exec_id=exec_id
            )
    
    async def _execute_local(self, 
                           file_path: str, 
                           command: str,
                           inputs: Optional[str],
                           timeout: int,
                           exec_id: str) -> CodeExecResult:
        """
        Execute code locally with resource constraints.
        
        Args:
            file_path: Path to temporary file with code
            command: Command to execute the code
            inputs: Optional inputs for stdin
            timeout: Execution timeout in seconds
            exec_id: Execution ID
            
        Returns:
            CodeExecResult with execution details
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare command
            cmd = [command, file_path]
            
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if inputs else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir) if self.working_dir else None
            )
            
            # Run with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(inputs.encode() if inputs else None),
                    timeout=timeout
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                return CodeExecResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode(),
                    exec_id=exec_id,
                    duration_ms=duration_ms
                )
                
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except:
                    pass
                    
                return CodeExecResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    exec_id=exec_id,
                    duration_ms=timeout * 1000
                )
                
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return CodeExecResult(
                success=False,
                error=f"Execution failed: {str(e)}",
                exec_id=exec_id,
                duration_ms=duration_ms
            )
    
    async def _execute_docker(self,
                            file_path: str,
                            docker_image: str,
                            inputs: Optional[str],
                            timeout: int,
                            exec_id: str) -> CodeExecResult:
        """
        Execute code in a Docker container for isolation.
        
        Args:
            file_path: Path to temporary file with code
            docker_image: Docker image to use
            inputs: Optional inputs for stdin
            timeout: Execution timeout in seconds
            exec_id: Execution ID
            
        Returns:
            CodeExecResult with execution details
        """
        import time
        start_time = time.time()
        
        try:
            # Get file details
            file_name = os.path.basename(file_path)
            container_name = f"scout_code_exec_{exec_id}"
            
            # Docker command to run the file
            docker_cmd = [
                "docker", "run",
                "--name", container_name,
                "--rm",  # Remove container after execution
                "-m", f"{self.resource_limits['max_memory_mb']}m",  # Memory limit
                "-v", f"{file_path}:/app/{file_name}",  # Mount file
                "--network", "none" if not self.resource_limits["network_access"] else "bridge",
                docker_image
            ]
            
            # Add command to execute within container
            if docker_image.startswith("python"):
                docker_cmd.extend(["python", f"/app/{file_name}"])
            elif docker_image.startswith("node"):
                docker_cmd.extend(["node", f"/app/{file_name}"])
            elif docker_image.startswith("bash"):
                docker_cmd.extend(["bash", f"/app/{file_name}"])
            
            # Create process
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE if inputs else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Run with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(inputs.encode() if inputs else None),
                    timeout=timeout
                )
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                return CodeExecResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode(),
                    exec_id=exec_id,
                    duration_ms=duration_ms
                )
                
            except asyncio.TimeoutError:
                try:
                    # Kill the container if it's still running
                    await asyncio.create_subprocess_exec(
                        "docker", "kill", container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                except:
                    pass
                    
                return CodeExecResult(
                    success=False,
                    error=f"Docker execution timed out after {timeout} seconds",
                    exec_id=exec_id,
                    duration_ms=timeout * 1000
                )
                
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return CodeExecResult(
                success=False,
                error=f"Docker execution failed: {str(e)}",
                exec_id=exec_id,
                duration_ms=duration_ms
            )
            
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent execution history.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of execution results
        """
        history = list(self.execution_history.values())
        return sorted(history, key=lambda x: x.get("exec_id", ""), reverse=True)[:limit]
        
    def get_execution_result(self, exec_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific execution result.
        
        Args:
            exec_id: Execution ID
            
        Returns:
            Execution result or None if not found
        """
        return self.execution_history.get(exec_id)
        
    async def create_dag_task(self, code: str, language: str, task_id: str = None) -> Dict[str, Any]:
        """
        Create a DAG task for code execution.
        
        This method creates a task definition that can be added to a DAG.
        When executed, the task will use this service to run the code.
        
        Args:
            code: Code to execute
            language: Programming language
            task_id: Optional task ID
            
        Returns:
            DAG task definition
        """
        task_id = task_id or f"code_exec_{str(uuid.uuid4())[:8]}"
        
        # Create task definition
        task = {
            "id": task_id,
            "type": "code_execution",
            "params": {
                "code": code,
                "language": language,
                "environment": ExecutionEnvironment.LOCAL.value
            },
            "dependencies": []
        }
        
        return task


# Global service instance for convenience
_code_execution_service_instance: Optional[CodeExecutionService] = None


def get_code_execution_service() -> CodeExecutionService:
    """
    Get the global code execution service instance.
    
    Returns:
        CodeExecutionService instance
    """
    global _code_execution_service_instance
    
    if _code_execution_service_instance is None:
        try:
            from service_registry import get_registry
            
            # Check if registered in service registry
            registry = get_registry()
            
            if registry.has_service_instance("code_execution"):
                _code_execution_service_instance = registry.get_service("code_execution")
            else:
                # Create and register a new instance
                _code_execution_service_instance = CodeExecutionService()
                # Register in the registry
                registry.register_instance(_code_execution_service_instance)
        except Exception as e:
            # If registry access fails, create a standalone instance
            logger = get_logger("code_execution_service")
            logger.warning(f"Could not access service registry: {e}. Creating standalone instance.")
            _code_execution_service_instance = CodeExecutionService()
            
    return _code_execution_service_instance
