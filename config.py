"""
Configuration management for ScoutAgent.
Handles environment variables, DAG settings, timeouts, and validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import re
from dataclasses import dataclass, asdict

# Try to import dotenv, but handle gracefully if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass  # No-op if dotenv not available

from custom_logging import get_logger

logger = get_logger("config")


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    log_dir: str = "./logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_tracing: bool = True
    trace_dir: str = "./logs/traces"


@dataclass
class DAGConfig:
    """DAG execution configuration."""
    max_depth: int = 10
    max_agents: int = 50
    timeout_seconds: int = 300
    enable_parallel: bool = True
    max_concurrent: int = 5
    retry_attempts: int = 3
    retry_delay: int = 1


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    max_iterations: int = 10
    timeout_seconds: int = 60
    memory_limit_mb: int = 512
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class SearchConfig:
    """Search and scraping configuration."""
    max_results: int = 100
    rate_limit_delay: float = 1.0
    user_agent: str = "ScoutAgent/1.0"
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3


@dataclass
class MemoryConfig:
    """Memory storage configuration."""
    backend: str = "local"  # local, chroma, faiss
    storage_dir: str = "./memory"
    max_entries: int = 10000
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class APIConfig:
    """API keys and external service configuration."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    bing_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None


@dataclass
class LLMRoutingConfig:
    """LLM routing configuration with global and per-agent overrides."""
    default_backend: Optional[str] = None
    default_model: Optional[str] = None
    per_agent_backend: Dict[str, str] = None  # keys: agent name (lowercase)
    per_agent_model: Dict[str, str] = None    # keys: agent name (lowercase)

    def __post_init__(self):
        if self.per_agent_backend is None:
            self.per_agent_backend = {}
        if self.per_agent_model is None:
            self.per_agent_model = {}


@dataclass
class ScoutConfig:
    """Main configuration class for ScoutAgent."""
    
    # Core settings
    project_name: str = "scout_agent"
    environment: str = "development"  # development, staging, production
    debug: bool = False
    
    # Sub-configurations - use default_factory for mutable defaults
    logging: LoggingConfig = None
    dag: DAGConfig = None
    agent: AgentConfig = None
    search: SearchConfig = None
    memory: MemoryConfig = None
    api: APIConfig = None
    llm_routing: LLMRoutingConfig = None
    
    # File paths
    config_file: Optional[str] = None
    prompts_dir: str = "./prompts"
    data_dir: str = "./data"
    output_dir: str = "./output"
    
    def __post_init__(self):
        """Initialize mutable defaults and validate."""
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.dag is None:
            self.dag = DAGConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.search is None:
            self.search = SearchConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.llm_routing is None:
            self.llm_routing = LLMRoutingConfig()
        
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        validation_errors = []
        
        # Validate logging level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging.level not in valid_levels:
            validation_errors.append(f"Invalid log level: {self.logging.level}")
        
        # Validate DAG settings
        if self.dag.max_depth < 1:
            validation_errors.append("DAG max_depth must be >= 1")
        if self.dag.max_agents < 1:
            validation_errors.append("DAG max_agents must be >= 1")
        if self.dag.timeout_seconds < 1:
            validation_errors.append("DAG timeout must be >= 1 second")
        
        # Validate agent settings
        if self.agent.max_iterations < 1:
            validation_errors.append("Agent max_iterations must be >= 1")
        if self.agent.timeout_seconds < 1:
            validation_errors.append("Agent timeout must be >= 1 second")
        
        # Validate memory backend
        valid_backends = {"local", "chroma", "faiss"}
        if self.memory.backend not in valid_backends:
            validation_errors.append(f"Invalid memory backend: {self.memory.backend}")
        
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Config validation error: {error}")
            raise ValueError(f"Configuration validation failed: {validation_errors}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.logging.log_dir,
            self.logging.trace_dir,
            self.memory.storage_dir,
            self.prompts_dir,
            self.data_dir,
            self.output_dir,
            # Data subfolders
            str(Path(self.data_dir) / "reddit_cache"),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


class ConfigManager:
    """Configuration manager for loading and managing ScoutAgent config."""
    
    def __init__(self):
        self.logger = get_logger("config_manager")
        if DOTENV_AVAILABLE:
            load_dotenv()  # Load environment variables from .env file
            self.logger.debug("Loaded environment variables from .env")
        else:
            self.logger.warning("python-dotenv not available, using system environment variables only")
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   env_prefix: str = "SCOUT_") -> ScoutConfig:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_file: Path to JSON configuration file
            env_prefix: Prefix for environment variables
            
        Returns:
            ScoutConfig instance
        """
        self.logger.info("Loading ScoutAgent configuration...")
        
        # Start with default config
        config_dict = asdict(ScoutConfig())
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.logger.info(f"Loading config from file: {config_file}")
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            config_dict = self._deep_merge(config_dict, file_config)
        
        # Override with environment variables
        env_config = self._load_from_env(env_prefix)
        config_dict = self._deep_merge(config_dict, env_config)
        
        # Create and validate config
        try:
            # Convert nested dicts to dataclass instances
            config = ScoutConfig(
                project_name=config_dict.get('project_name', 'scout_agent'),
                environment=config_dict.get('environment', 'development'),
                debug=config_dict.get('debug', False),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                dag=DAGConfig(**config_dict.get('dag', {})),
                agent=AgentConfig(**config_dict.get('agent', {})),
                search=SearchConfig(**config_dict.get('search', {})),
                memory=MemoryConfig(**config_dict.get('memory', {})),
                api=APIConfig(**config_dict.get('api', {})),
                llm_routing=LLMRoutingConfig(**config_dict.get('llm_routing', {})),
                config_file=config_dict.get('config_file'),
                prompts_dir=config_dict.get('prompts_dir', './prompts'),
                data_dir=config_dict.get('data_dir', './data'),
                output_dir=config_dict.get('output_dir', './output'),
            )
            self.logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_env(self, prefix: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Simple mappings
        simple_mappings = {
            f"{prefix}PROJECT_NAME": ["project_name"],
            f"{prefix}ENVIRONMENT": ["environment"],
            f"{prefix}DEBUG": ["debug"],
            f"{prefix}LOG_LEVEL": ["logging", "level"],
            f"{prefix}LOG_DIR": ["logging", "log_dir"],
            f"{prefix}MAX_FILE_SIZE": ["logging", "max_file_size"],
            f"{prefix}BACKUP_COUNT": ["logging", "backup_count"],
            f"{prefix}ENABLE_TRACING": ["logging", "enable_tracing"],
            f"{prefix}TRACE_DIR": ["logging", "trace_dir"],
            f"{prefix}DAG_MAX_DEPTH": ["dag", "max_depth"],
            f"{prefix}DAG_MAX_AGENTS": ["dag", "max_agents"],
            f"{prefix}DAG_TIMEOUT": ["dag", "timeout_seconds"],
            f"{prefix}ENABLE_PARALLEL": ["dag", "enable_parallel"],
            f"{prefix}MAX_CONCURRENT": ["dag", "max_concurrent"],
            f"{prefix}RETRY_ATTEMPTS": ["dag", "retry_attempts"],
            f"{prefix}RETRY_DELAY": ["dag", "retry_delay"],
            f"{prefix}AGENT_MAX_ITERATIONS": ["agent", "max_iterations"],
            f"{prefix}AGENT_TIMEOUT": ["agent", "timeout_seconds"],
            f"{prefix}MEMORY_LIMIT_MB": ["agent", "memory_limit_mb"],
            f"{prefix}ENABLE_CACHING": ["agent", "enable_caching"],
            f"{prefix}CACHE_TTL": ["agent", "cache_ttl"],
            f"{prefix}SEARCH_MAX_RESULTS": ["search", "max_results"],
            f"{prefix}RATE_LIMIT_DELAY": ["search", "rate_limit_delay"],
            f"{prefix}USER_AGENT": ["search", "user_agent"],
            f"{prefix}SEARCH_TIMEOUT": ["search", "timeout_seconds"],
            f"{prefix}MAX_RETRIES": ["search", "max_retries"],
            f"{prefix}BACKOFF_FACTOR": ["search", "backoff_factor"],
            f"{prefix}MEMORY_BACKEND": ["memory", "backend"],
            f"{prefix}MEMORY_STORAGE_DIR": ["memory", "storage_dir"],
            f"{prefix}MAX_ENTRIES": ["memory", "max_entries"],
            f"{prefix}EMBEDDING_MODEL": ["memory", "embedding_model"],
            f"{prefix}CHUNK_SIZE": ["memory", "chunk_size"],
            f"{prefix}CHUNK_OVERLAP": ["memory", "chunk_overlap"],
            f"{prefix}PROMPTS_DIR": ["prompts_dir"],
            f"{prefix}DATA_DIR": ["data_dir"],
            f"{prefix}OUTPUT_DIR": ["output_dir"],
        }
        
        # API keys
        api_keys = {
            f"{prefix}OPENAI_API_KEY": ["api", "openai_api_key"],
            f"{prefix}ANTHROPIC_API_KEY": ["api", "anthropic_api_key"],
            f"{prefix}SERPAPI_KEY": ["api", "serpapi_key"],
            f"{prefix}BING_API_KEY": ["api", "bing_api_key"],
            f"{prefix}GEMINI_API_KEY": ["api", "gemini_api_key"],
            f"{prefix}DEEPSEEK_API_KEY": ["api", "deepseek_api_key"],
            f"{prefix}HUGGINGFACE_TOKEN": ["api", "huggingface_token"],
        }
        
        # Process simple mappings
        for env_var, path in {**simple_mappings, **api_keys}.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, path, self._convert_type(value))

        # LLM routing overrides
        # Global defaults
        default_backend = os.getenv(f"{prefix}LLM_DEFAULT_BACKEND")
        if default_backend:
            self._set_nested_value(env_config, ["llm_routing", "default_backend"], default_backend.lower())
        default_model = os.getenv(f"{prefix}LLM_DEFAULT_MODEL")
        if default_model:
            self._set_nested_value(env_config, ["llm_routing", "default_model"], default_model)

        # Per-agent overrides: SCOUT_LLM_AGENT_<AGENT>_BACKEND and _MODEL
        # Normalize agent key to lowercase
        pattern = re.compile(rf"^{re.escape(prefix)}LLM_AGENT_([A-Z0-9_]+)_(BACKEND|MODEL)$")
        for key, value in os.environ.items():
            m = pattern.match(key)
            if not m or not value:
                continue
            agent_key = m.group(1).lower()
            kind = m.group(2)
            if kind == "BACKEND":
                self._set_nested_value(env_config, ["llm_routing", "per_agent_backend", agent_key], value.lower())
            elif kind == "MODEL":
                self._set_nested_value(env_config, ["llm_routing", "per_agent_model", agent_key], value)

        return env_config
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """Convert string environment variable to appropriate type."""
        # Boolean conversion
        if value.lower() in {"true", "1", "yes", "on"}:
            return True
        elif value.lower() in {"false", "0", "no", "off"}:
            return False
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, dictionary: Dict[str, Any], path: List[str], value: Any):
        """Set a nested dictionary value using a path."""
        current = dictionary
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_env_template(self, path: str = ".env.template"):
        """Create a .env template file with expected SCOUT_* environment variables.

        This writes commented guidance and placeholder values for keys consumed by
        ConfigManager._load_from_env(). Existing file will be overwritten to keep
        it up-to-date with the current schema.
        """
        self.logger.info(f"Writing environment template to: {path}")
        lines = [
            "# ScoutAgent environment template",
            "# Copy to .env and fill in values as needed",
            "",
            "# --- Core ---",
            "SCOUT_PROJECT_NAME=scout_agent",
            "SCOUT_ENVIRONMENT=development",
            "SCOUT_DEBUG=false",
            "",
            "# --- Logging ---",
            "SCOUT_LOG_LEVEL=INFO",
            "SCOUT_LOG_DIR=./logs",
            "SCOUT_LOG_MAX_FILE_SIZE=10485760",
            "SCOUT_LOG_BACKUP_COUNT=5",
            "SCOUT_ENABLE_TRACING=true",
            "SCOUT_TRACE_DIR=./logs/traces",
            "",
            "# --- DAG ---",
            "SCOUT_DAG_MAX_DEPTH=10",
            "SCOUT_DAG_MAX_AGENTS=50",
            "SCOUT_DAG_TIMEOUT_SECONDS=300",
            "SCOUT_DAG_ENABLE_PARALLEL=true",
            "SCOUT_DAG_MAX_CONCURRENT=5",
            "SCOUT_DAG_RETRY_ATTEMPTS=3",
            "SCOUT_DAG_RETRY_DELAY=1",
            "",
            "# --- Agent ---",
            "SCOUT_AGENT_MAX_ITERATIONS=10",
            "SCOUT_AGENT_TIMEOUT_SECONDS=60",
            "SCOUT_AGENT_MEMORY_LIMIT_MB=512",
            "SCOUT_AGENT_ENABLE_CACHING=true",
            "SCOUT_AGENT_CACHE_TTL=3600",
            "",
            "# --- Search ---",
            "SCOUT_SEARCH_MAX_RESULTS=100",
            "SCOUT_SEARCH_RATE_LIMIT_DELAY=1.0",
            "SCOUT_SEARCH_USER_AGENT=ScoutAgent/1.0",
            "SCOUT_SEARCH_TIMEOUT_SECONDS=30",
            "SCOUT_SEARCH_MAX_RETRIES=3",
            "SCOUT_SEARCH_BACKOFF_FACTOR=0.3",
            "",
            "# --- Memory ---",
            "SCOUT_MEMORY_BACKEND=local",
            "SCOUT_MEMORY_STORAGE_DIR=./memory",
            "SCOUT_MEMORY_MAX_ENTRIES=10000",
            "SCOUT_MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2",
            "SCOUT_MEMORY_CHUNK_SIZE=512",
            "SCOUT_MEMORY_CHUNK_OVERLAP=50",
            "",
            "# --- Paths ---",
            "SCOUT_PROMPTS_DIR=./prompts",
            "SCOUT_DATA_DIR=./data",
            "SCOUT_OUTPUT_DIR=./output",
            "",
            "# --- API Keys (leave blank if not used) ---",
            "SCOUT_OPENAI_API_KEY=",
            "SCOUT_ANTHROPIC_API_KEY=",
            "SCOUT_SERPAPI_KEY=",
            "SCOUT_BING_API_KEY=",
            "SCOUT_GEMINI_API_KEY=",
            "SCOUT_DEEPSEEK_API_KEY=",
            "SCOUT_HUGGINGFACE_TOKEN=",
            "",
            "# --- LLM Routing (global defaults) ---",
            "# Choose a default backend and model when not specified by code.",
            "# Supported backends: openai, claude, gemini, deepseek, ollama",
            "SCOUT_LLM_DEFAULT_BACKEND=",
            "SCOUT_LLM_DEFAULT_MODEL=",
            "",
            "# --- Per-Agent LLM overrides ---",
            "# Set a specific backend or model per agent. Use agent name in UPPERCASE,",
            "# non-alphanumeric replaced by underscores. Examples:",
            "#   SCOUT_LLM_AGENT_SCREENER_BACKEND=ollama",
            "#   SCOUT_LLM_AGENT_SCREENER_MODEL=phi4-mini:latest",
            "#   SCOUT_LLM_AGENT_VALIDATOR_BACKEND=openai",
            "#   SCOUT_LLM_AGENT_VALIDATOR_MODEL=gpt-4o",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.logger.info("Environment template written successfully")
        
    def save_config(self, config: ScoutConfig, config_file: str):
        """Save configuration to JSON file."""
        self.logger.info(f"Saving configuration to: {config_file}")
        
        config_dict = asdict(config)
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info("Configuration saved successfully")


# Global configuration instance
_config_instance: Optional[ScoutConfig] = None


def get_config() -> ScoutConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        manager = ConfigManager()
        _config_instance = manager.load_config()
    return _config_instance


def init_config(config_file: Optional[str] = None) -> ScoutConfig:
    """Initialize configuration with optional config file."""
    global _config_instance
    manager = ConfigManager()
    _config_instance = manager.load_config(config_file)
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    print("Testing ScoutAgent configuration...")
    
    # Create config manager
    manager = ConfigManager()
    
    # Create .env template
    manager.create_env_template()
    
    # Load default config
    config = manager.load_config()
    
    print("\nLoaded configuration:")
    print(json.dumps(asdict(config), indent=2))
    
    print("\nConfiguration validation passed!")
    print("Environment template created: .env.template")