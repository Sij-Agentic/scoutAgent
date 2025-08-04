"""
Base Agent class for ScoutAgent.
Provides the foundation for all specialized agents with planning/thinking/action separation.
"""

import uuid
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from custom_logging import get_logger
from config import get_config


@dataclass
class AgentInput:
    """Input data structure for agents."""
    data: Any
    metadata: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentOutput:
    """Output data structure from agents."""
    result: Any
    metadata: Dict[str, Any]
    logs: List[str]
    execution_time: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    name: str
    status: str  # 'idle', 'planning', 'thinking', 'acting', 'completed', 'failed'
    iteration: int
    max_iterations: int
    start_time: float
    last_update: float
    memory: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base agent class with planning/thinking/action separation.
    
    All ScoutAgent agents inherit from this base class.
    """
    
    def __init__(self, 
                 name: str,
                 agent_id: Optional[str] = None,
                 config: Optional[Any] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name/type
            agent_id: Unique identifier (auto-generated if None)
            config: Configuration instance
        """
        self.name = name
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or get_config()
        self.logger = get_logger(f"agent.{name}")
        
        # Agent state
        self.state = AgentState(
            agent_id=self.agent_id,
            name=name,
            status='idle',
            iteration=0,
            max_iterations=self.config.agent.max_iterations,
            start_time=time.time(),
            last_update=time.time(),
            memory={}
        )
        
        # Performance tracking
        self.execution_logs = []
        self.start_time = None
        
        self.logger.info(f"Initialized agent: {self.name} ({self.agent_id})")
    
    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Main execution method following planning/thinking/action pattern.
        
        Args:
            agent_input: Input data for the agent
            
        Returns:
            AgentOutput: Results from agent execution
        """
        self.start_time = time.time()
        self.logger.info(f"Starting execution of agent {self.name}")
        
        try:
            # Planning phase
            self._update_status('planning')
            plan = self.plan(agent_input)
            self.logger.debug(f"Plan created: {plan}")
            
            # Thinking phase
            self._update_status('thinking')
            thoughts = self.think(agent_input, plan)
            self.logger.debug(f"Thoughts: {thoughts}")
            
            # Action phase
            self._update_status('acting')
            result = self.act(agent_input, plan, thoughts)
            self.logger.debug(f"Action result: {result}")
            
            # Create output
            execution_time = time.time() - self.start_time
            output = AgentOutput(
                result=result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'plan': plan,
                    'thoughts': thoughts
                },
                logs=self.execution_logs,
                execution_time=execution_time,
                success=True
            )
            
            self._update_status('completed')
            self.logger.info(f"Agent {self.name} completed successfully in {execution_time:.2f}s")
            return output
            
        except Exception as e:
            execution_time = time.time() - self.start_time
            self.logger.error(f"Agent {self.name} failed: {str(e)}")
            
            output = AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
            
            self._update_status('failed')
            return output
    
    @abstractmethod
    def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Planning phase: Determine what needs to be done.
        
        Args:
            agent_input: Input data
            
        Returns:
            Dict containing the plan
        """
        pass
    
    @abstractmethod
    def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thinking phase: Analyze and reason about the plan.
        
        Args:
            agent_input: Input data
            plan: The plan from planning phase
            
        Returns:
            Dict containing thoughts/analysis
        """
        pass
    
    @abstractmethod
    def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Any:
        """
        Action phase: Execute the actual work.
        
        Args:
            agent_input: Input data
            plan: The plan from planning phase
            thoughts: Thoughts from thinking phase
            
        Returns:
            The result of the action
        """
        pass
    
    def _update_status(self, status: str):
        """Update agent status and log the change."""
        old_status = self.state.status
        self.state.status = status
        self.state.last_update = time.time()
        self.logger.debug(f"Agent {self.name} status: {old_status} -> {status}")
    
    def log(self, message: str, level: str = 'info'):
        """Add a log entry to execution logs."""
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message
        }
        self.execution_logs.append(log_entry)
        
        # Also log to main logger
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state as dictionary."""
        return asdict(self.state)
    
    def save_state(self, file_path: str):
        """Save agent state to file."""
        state_data = {
            'state': self.get_state(),
            'logs': self.execution_logs,
            'config': asdict(self.config)
        }
        
        with open(file_path, 'w') as f:
            import json
            json.dump(state_data, f, indent=2, default=str)
        
        self.logger.debug(f"Agent state saved to {file_path}")
    
    def load_state(self, file_path: str):
        """Load agent state from file."""
        if not Path(file_path).exists():
            self.logger.warning(f"State file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            import json
            state_data = json.load(f)
        
        # Restore state
        state_dict = state_data.get('state', {})
        self.state = AgentState(**state_dict)
        self.execution_logs = state_data.get('logs', [])
        
        self.logger.debug(f"Agent state loaded from {file_path}")
    
    def reset(self):
        """Reset agent to initial state."""
        self.state = AgentState(
            agent_id=self.agent_id,
            name=self.name,
            status='idle',
            iteration=0,
            max_iterations=self.config.agent.max_iterations,
            start_time=time.time(),
            last_update=time.time(),
            memory={}
        )
        self.execution_logs = []
        self.logger.info(f"Agent {self.name} reset to initial state")
    
    def __str__(self):
        return f"BaseAgent(name='{self.name}', id='{self.agent_id}', status='{self.state.status}')"
    
    def __repr__(self):
        return self.__str__()


class AgentRegistry:
    """
    Registry for managing all available agents.
    """
    
    def __init__(self):
        self.logger = get_logger("agent_registry")
        self.agents: Dict[str, type] = {}
        self.instances: Dict[str, BaseAgent] = {}
        self._message_buses: Dict[str, Any] = {}
        self.logger.info("AgentRegistry initialized")
    
    def register_agent(self, agent_class: type) -> None:
        """
        Register an agent class.
        
        Args:
            agent_class: Agent class to register (must inherit from BaseAgent)
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must inherit from BaseAgent: {agent_class}")
        
        agent_name = agent_class.__name__
        self.agents[agent_name] = agent_class
        
        # Also register lowercase version for convenience
        lowercase_name = agent_name.lower().replace('agent', '')
        if lowercase_name:
            self.agents[lowercase_name] = agent_class
        
        self.logger.info(f"Registered agent: {agent_name}")
        if lowercase_name:
            self.logger.info(f"Registered agent alias: {lowercase_name}")
    
    def create_agent(self, agent_name: str, **kwargs) -> BaseAgent:
        """
        Create an instance of a registered agent.
        
        Args:
            agent_name: Name of the registered agent
            **kwargs: Arguments to pass to agent constructor
            
        Returns:
            BaseAgent instance
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent not registered: {agent_name}")
        
        agent_class = self.agents[agent_name]
        agent_instance = agent_class(name=agent_name, **kwargs)
        
        self.instances[agent_instance.agent_id] = agent_instance
        self.logger.info(f"Created agent instance: {agent_name} ({agent_instance.agent_id})")
        
        return agent_instance
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent instance by ID."""
        return self.instances.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all registered agent types."""
        return list(self.agents.keys())
    
    def list_instances(self) -> Dict[str, str]:
        """List all agent instances with their names."""
        return {agent_id: instance.name for agent_id, instance in self.instances.items()}
    
    def remove_agent(self, agent_id: str):
        """Remove an agent instance."""
        if agent_id in self.instances:
            agent_name = self.instances[agent_id].name
            del self.instances[agent_id]
            self.logger.info(f"Removed agent instance: {agent_name} ({agent_id})")
    
    def clear_instances(self):
        """Clear all agent instances."""
        self.instances.clear()
        self.logger.info("Cleared all agent instances")
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a registered agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent not registered: {agent_name}")
        
        agent_class = self.agents[agent_name]
        return {
            'name': agent_name,
            'class_name': agent_class.__name__,
            'module': agent_class.__module__,
            'doc': agent_class.__doc__,
            'registered_instances': [
                aid for aid, inst in self.instances.items() 
                if inst.name == agent_name
            ]
        }
    
    def get_message_bus(self, context_id: str) -> Any:
        """Get or create a message bus for a context."""
        from .communication import MessageBus
        if context_id not in self._message_buses:
            self._message_buses[context_id] = MessageBus(context_id)
        return self._message_buses[context_id]
    
    async def start_message_bus(self, context_id: str) -> None:
        """Start the message bus for a context."""
        bus = self.get_message_bus(context_id)
        await bus.start()
    
    async def stop_message_bus(self, context_id: str) -> None:
        """Stop the message bus for a context."""
        if context_id in self._message_buses:
            await self._message_buses[context_id].stop()
    
    def clear_message_buses(self) -> None:
        """Clear all message buses."""
        self._message_buses.clear()


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def register_agent(agent_class: type, name: Optional[str] = None):
    """Register an agent class with the global registry."""
    get_registry().register_agent(agent_class)


def create_agent(agent_name: str, **kwargs) -> BaseAgent:
    """Create an agent instance using the global registry."""
    return get_registry().create_agent(agent_name, **kwargs)


def get_agent_class(agent_name: str) -> Optional[type]:
    """Get an agent class by name from the global registry."""
    registry = get_registry()
    return registry.agents.get(agent_name)


if __name__ == "__main__":
    # Test the base agent system
    print("Testing BaseAgent and AgentRegistry...")
    
    # Create a simple test agent
    class TestAgent(BaseAgent):
        def plan(self, agent_input):
            return {"task": "test", "steps": ["step1", "step2"]}
        
        def think(self, agent_input, plan):
            return {"analysis": "test analysis", "confidence": 0.9}
        
        def act(self, agent_input, plan, thoughts):
            return {"result": "test completed", "data": agent_input.data}
    
    # Register and create agent
    register_agent(TestAgent)
    agent = create_agent("TestAgent")
    
    # Test execution
    from agents.base import AgentInput
    result = agent.execute(AgentInput(data="test data", metadata={"test": True}))
    
    print(f"Agent created: {agent}")
    print(f"Registry agents: {get_registry().list_agents()}")
    print(f"Execution result: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    print("Base agent system test completed!")
