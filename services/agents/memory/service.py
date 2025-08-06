"""
Memory Service Implementation.

This module provides a service for storing, retrieving, and searching memories
across various storage backends.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Tuple

from service_registry import ServiceBase, service, requires, inject
from custom_logging import get_logger


class MemoryBackendType(Enum):
    """Supported memory storage backends."""
    FILE_SYSTEM = "file_system"
    VECTOR_DB = "vector_db"
    SQL_DB = "sql_db"


class MemoryType(Enum):
    """Types of memories that can be stored."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    JSON = "json"
    EMBEDDING = "embedding"


class Memory:
    """Representation of a stored memory."""
    
    def __init__(self,
                memory_id: str = None,
                content: Any = None,
                memory_type: MemoryType = MemoryType.TEXT,
                metadata: Dict[str, Any] = None,
                tags: List[str] = None,
                timestamp: float = None,
                embedding: List[float] = None):
        """
        Initialize a memory instance.
        
        Args:
            memory_id: Unique identifier for the memory
            content: The content of the memory
            memory_type: Type of memory (text, code, image, etc.)
            metadata: Additional metadata about the memory
            tags: Tags for categorizing the memory
            timestamp: Creation time
            embedding: Vector embedding for semantic search
        """
        self.memory_id = memory_id or str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.tags = tags or []
        self.timestamp = timestamp or datetime.now().timestamp()
        self.embedding = embedding
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        return cls(
            memory_id=data.get("memory_id"),
            content=data.get("content"),
            memory_type=MemoryType(data.get("memory_type", "text")),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            timestamp=data.get("timestamp"),
            embedding=data.get("embedding")
        )


class MemorySearchResult:
    """Result of a memory search operation."""
    
    def __init__(self,
                memory: Memory,
                similarity_score: float = 0.0):
        """
        Initialize a search result.
        
        Args:
            memory: The matched memory
            similarity_score: Semantic similarity score (0.0 to 1.0)
        """
        self.memory = memory
        self.similarity_score = similarity_score
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "memory": self.memory.to_dict() if self.memory else None,
            "similarity_score": self.similarity_score
        }
    
    
class MemoryBackend:
    """Base class for memory storage backends."""
    
    def __init__(self, backend_type: MemoryBackendType, config: Dict[str, Any] = None):
        """
        Initialize the memory backend.
        
        Args:
            backend_type: Type of storage backend
            config: Backend configuration
        """
        self.backend_type = backend_type
        self.config = config or {}
        self.logger = get_logger(f"memory_backend.{backend_type.value}")
        
    async def initialize(self) -> bool:
        """Initialize the backend storage."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def store(self, memory: Memory) -> bool:
        """Store a memory in the backend."""
        raise NotImplementedError("Subclasses must implement store()")
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        raise NotImplementedError("Subclasses must implement delete()")
    
    async def list(self, 
                 limit: int = 100,
                 tags: List[str] = None,
                 memory_type: MemoryType = None,
                 from_timestamp: float = None,
                 to_timestamp: float = None) -> List[Memory]:
        """List memories with optional filters."""
        raise NotImplementedError("Subclasses must implement list()")
    
    async def search(self,
                   query: str,
                   limit: int = 10,
                   tags: List[str] = None,
                   memory_type: MemoryType = None,
                   threshold: float = 0.0) -> List[MemorySearchResult]:
        """Search memories by content."""
        raise NotImplementedError("Subclasses must implement search()")
        
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory fields."""
        raise NotImplementedError("Subclasses must implement update()")


class FileSystemBackend(MemoryBackend):
    """File system based memory storage backend."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the file system backend."""
        super().__init__(MemoryBackendType.FILE_SYSTEM, config)
        self.storage_dir = Path(self.config.get("storage_dir", "./memories"))
        self.index_file = self.storage_dir / "memory_index.json"
        self.memory_index = {}
        
    async def initialize(self) -> bool:
        """Initialize the file system storage."""
        try:
            # Create storage directory if it doesn't exist
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Load index if it exists
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.memory_index = json.load(f)
                    
            self.logger.info(f"Initialized file system backend with {len(self.memory_index)} memories")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize file system backend: {e}")
            return False
    
    async def _save_index(self) -> bool:
        """Save the memory index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.memory_index, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save memory index: {e}")
            return False
    
    async def store(self, memory: Memory) -> bool:
        """Store a memory in the file system."""
        try:
            # Create memory directory if it doesn't exist
            memory_dir = self.storage_dir / "data"
            memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a filename
            memory_file = memory_dir / f"{memory.memory_id}.json"
            
            # Save memory to file
            with open(memory_file, 'w') as f:
                json.dump(memory.to_dict(), f, indent=2)
                
            # Update index
            self.memory_index[memory.memory_id] = {
                "file": str(memory_file.relative_to(self.storage_dir)),
                "type": memory.memory_type.value,
                "tags": memory.tags,
                "timestamp": memory.timestamp
            }
            
            # Save index
            await self._save_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory from the file system."""
        try:
            if memory_id not in self.memory_index:
                return None
                
            # Get file path
            rel_path = self.memory_index[memory_id]["file"]
            memory_file = self.storage_dir / rel_path
            
            # Load memory from file
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
                
            return Memory.from_dict(memory_data)
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from the file system."""
        try:
            if memory_id not in self.memory_index:
                return False
                
            # Get file path
            rel_path = self.memory_index[memory_id]["file"]
            memory_file = self.storage_dir / rel_path
            
            # Delete file if it exists
            if memory_file.exists():
                memory_file.unlink()
                
            # Update index
            del self.memory_index[memory_id]
            await self._save_index()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
            
    async def list(self, 
                 limit: int = 100,
                 tags: List[str] = None,
                 memory_type: MemoryType = None,
                 from_timestamp: float = None,
                 to_timestamp: float = None) -> List[Memory]:
        """List memories with optional filters."""
        try:
            results = []
            count = 0
            
            # Apply filters
            for memory_id, info in self.memory_index.items():
                # Filter by type
                if memory_type and info["type"] != memory_type.value:
                    continue
                    
                # Filter by tags (any match)
                if tags and not any(tag in info["tags"] for tag in tags):
                    continue
                    
                # Filter by timestamp range
                if from_timestamp and info["timestamp"] < from_timestamp:
                    continue
                    
                if to_timestamp and info["timestamp"] > to_timestamp:
                    continue
                    
                # Retrieve the full memory
                memory = await self.retrieve(memory_id)
                if memory:
                    results.append(memory)
                    count += 1
                    
                    # Check limit
                    if count >= limit:
                        break
                        
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            return []
    
    async def search(self,
                   query: str,
                   limit: int = 10,
                   tags: List[str] = None,
                   memory_type: MemoryType = None,
                   threshold: float = 0.0) -> List[MemorySearchResult]:
        """Search memories by content."""
        try:
            # For file system backend, we're just doing simple text search
            # A real vector DB would do semantic search with embeddings
            results = []
            
            # Get memories with filters
            memories = await self.list(
                limit=limit * 2,  # Get more than we need to filter
                tags=tags,
                memory_type=memory_type
            )
            
            # Simple text search
            for memory in memories:
                # Skip non-text memories
                if memory.memory_type not in [MemoryType.TEXT, MemoryType.CODE]:
                    continue
                    
                # Check if content contains the query (case insensitive)
                content = str(memory.content).lower()
                query_lower = query.lower()
                
                if query_lower in content:
                    # Simple similarity score based on substring position
                    # This is not a real semantic score, just a placeholder
                    position = content.find(query_lower)
                    length = len(content)
                    score = 1.0 - (position / length) if length > 0 else 0.0
                    
                    if score >= threshold:
                        results.append(MemorySearchResult(memory, score))
                        
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory fields."""
        try:
            # Get the existing memory
            memory = await self.retrieve(memory_id)
            if not memory:
                return False
                
            # Update fields
            for key, value in updates.items():
                if key == 'content':
                    memory.content = value
                elif key == 'metadata':
                    memory.metadata.update(value)
                elif key == 'tags':
                    memory.tags = value
                    
            # Store updated memory
            return await self.store(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False


@service(name="memory", singleton=True)
@requires("config", optional=True)
@requires("logging", optional=True)
class MemoryService(ServiceBase):
    """
    Service for storing, retrieving, and searching memories.
    
    This service handles memory operations across various storage backends
    and provides a unified API for other components to access memories.
    """
    
    def __init__(self):
        """Initialize the memory service."""
        super().__init__(name="memory", version="1.0.0")
        self.logger = get_logger("service.memory")
        self.config = None
        self.backend = None
        self.backend_type = None
        self.query_cache = {}  # Simple LRU cache for search results
        self.max_cache_size = 100
        
    def setup_direct(self, config=None, logger=None, backend_config=None):
        """
        Setup the service directly without using the service registry.
        Useful for testing and standalone usage.
        
        Args:
            config: Configuration service or dict
            logger: Logger instance
            backend_config: Optional backend configuration
        """
        if logger:
            self.logger = logger
        if config:
            self.config = config
            
        # Create a default backend if needed
        if not self.backend and backend_config is None:
            backend_config = {
                "data_dir": "/tmp/scout_agent/memory_test"
            }
        
    async def _initialize(self, registry) -> bool:
        """
        Initialize the memory service.
        
        Args:
            registry: Service registry for accessing dependencies
            
        Returns:
            True if initialization was successful
        """
        self.logger.info("Initializing memory service")
        
        try:
            # Handle direct initialization (no registry)
            if registry is None:
                self.logger.info("Initializing memory service directly (no registry)")
                if not self.backend:
                    # Set up default backend if none exists
                    backend_config = {
                        "data_dir": "/tmp/scout_agent/memory_test"
                    }
                    self.backend = FileSystemBackend(backend_config)
                    self.backend_type = MemoryBackendType.FILE_SYSTEM
                    await self.backend.initialize()
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
                self.logger = logging_service.get_logger("memory_service")
            except Exception as e:
                # Keep using default logger if logging service not available
                self.logger.warning(f"Could not get logging service: {e}. Using default logger.")
            
            # Determine backend type from config
            backend_type_str = self.config.get_config_value(
                "memory.backend",
                "file_system"  # Default backend
            )
            
            # Get backend type enum
            try:
                self.backend_type = MemoryBackendType(backend_type_str)
            except ValueError:
                self.logger.warning(
                    f"Invalid backend type: {backend_type_str}. "
                    "Falling back to file_system."
                )
                self.backend_type = MemoryBackendType.FILE_SYSTEM
            
            # Get backend config
            backend_config = self.config.get_config_value("memory.backend_config", {})
            
            # Create backend
            if self.backend_type == MemoryBackendType.VECTOR_DB:
                try:
                    # Try to import and use vector backend
                    from services.agents.memory.vector_backend import VectorDatabaseBackend
                    self.backend = VectorDatabaseBackend(backend_config)
                except (ImportError, Exception) as e:
                    self.logger.warning(
                        f"Vector database backend not available: {e}. "
                        "Falling back to file system backend."
                    )
                    self.backend = FileSystemBackend(backend_config)
                    self.backend_type = MemoryBackendType.FILE_SYSTEM
            else:
                self.backend = FileSystemBackend(backend_config)
                
            # Initialize backend
            await self.backend.initialize()
            
            # Get cache size
            self.max_cache_size = self.config.get_config_value(
                "memory.cache_size", 
                100
            )
                
            # Get LLM backend preference
            try:
                self.llm_backend_config = self.config.get_llm_backend_for_agent("memory") or {}
            except (AttributeError, Exception) as e:
                self.logger.warning(f"Could not get LLM backend config: {e}. Using default.")
                self.llm_backend_config = {}
                    
            self.logger.info(
                f"Memory service initialized with {self.backend_type.value} backend"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory service: {e}")
            return False
    
    async def _start(self) -> bool:
        """Start the memory service."""
        self.logger.info("Starting memory service")
        try:
            if not self.backend:
                self.logger.warning("Backend not initialized during start, initializing default backend")
                backend_config = {"data_dir": "/tmp/scout_agent/memory_test"}
                self.backend = FileSystemBackend(backend_config)
                self.backend_type = MemoryBackendType.FILE_SYSTEM
                await self.backend.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start memory service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """Stop the memory service and clean up resources."""
        self.logger.info("Stopping memory service")
        # Clear cache
        self.query_cache = {}
        return True
        
    async def create_memory(self, 
                           content: Any, 
                           memory_type: Union[MemoryType, str] = MemoryType.TEXT,
                           metadata: Dict[str, Any] = None,
                           tags: List[str] = None,
                           memory_id: str = None) -> Optional[str]:
        """
        Create a new memory.
        
        Args:
            content: Memory content (text, code, etc.)
            memory_type: Type of memory
            metadata: Additional memory metadata
            tags: Tags for categorizing the memory
            memory_id: Optional ID (will be generated if not provided)
            
        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Ensure backend exists
            if not self.backend:
                self.logger.warning("Backend not initialized, creating default backend")
                backend_config = {"data_dir": "/tmp/scout_agent/memory_test"}
                self.backend = FileSystemBackend(backend_config)
                self.backend_type = MemoryBackendType.FILE_SYSTEM
                await self.backend.initialize()
                
            # Convert string type to enum if needed
            if isinstance(memory_type, str):
                try:
                    memory_type = MemoryType(memory_type)
                except ValueError:
                    memory_type = MemoryType.TEXT
                    
            # Create memory object
            memory = Memory(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                tags=tags
            )
            
            # Store in backend
            success = await self.backend.store(memory)
            if success:
                # Clear cache on writes
                self.query_cache = {}
                return memory.memory_id
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create memory: {e}")
            return None
            
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory as a dictionary or None if not found
        """
        try:
            memory = await self.backend.retrieve(memory_id)
            if memory:
                return memory.to_dict()
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
            
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: Memory identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful
        """
        try:
            success = await self.backend.update(memory_id, updates)
            if success:
                # Clear cache on writes
                self.query_cache = {}
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
            
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if successful
        """
        try:
            success = await self.backend.delete(memory_id)
            if success:
                # Clear cache on writes
                self.query_cache = {}
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
            
    async def list_memories(self, 
                           limit: int = 100,
                           tags: List[str] = None,
                           memory_type: Union[MemoryType, str] = None,
                           from_timestamp: float = None,
                           to_timestamp: float = None) -> List[Dict[str, Any]]:
        """
        List memories with optional filters.
        
        Args:
            limit: Maximum number of memories to return
            tags: Filter by tags
            memory_type: Filter by memory type
            from_timestamp: Filter by minimum timestamp
            to_timestamp: Filter by maximum timestamp
            
        Returns:
            List of memories as dictionaries
        """
        try:
            # Convert string type to enum if needed
            if isinstance(memory_type, str):
                try:
                    memory_type = MemoryType(memory_type)
                except ValueError:
                    memory_type = None
                    
            # Generate cache key
            cache_key = f"list:{limit}:{tags}:{memory_type}:{from_timestamp}:{to_timestamp}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
                
            memories = await self.backend.list(
                limit=limit,
                tags=tags,
                memory_type=memory_type,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            
            result = [memory.to_dict() for memory in memories]
            
            # Cache result
            if len(self.query_cache) >= self.max_cache_size:
                self._prune_cache()
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            return []
            
    async def search_memories(self,
                            query: str,
                            limit: int = 10,
                            tags: List[str] = None,
                            memory_type: Union[MemoryType, str] = None,
                            threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search memories semantically.
        
        Args:
            query: Search query
            limit: Maximum number of results
            tags: Filter by tags
            memory_type: Filter by memory type
            threshold: Minimum similarity score
            
        Returns:
            List of memory results with similarity scores
        """
        try:
            # Convert string type to enum if needed
            if isinstance(memory_type, str):
                try:
                    memory_type = MemoryType(memory_type)
                except ValueError:
                    memory_type = None
                    
            # Generate cache key
            cache_key = f"search:{query}:{limit}:{tags}:{memory_type}:{threshold}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
                
            results = await self.backend.search(
                query=query,
                limit=limit,
                tags=tags,
                memory_type=memory_type,
                threshold=threshold
            )
            
            # Convert to dictionaries with scores
            search_results = []
            for result in results:
                memory_dict = result.memory.to_dict()
                memory_dict["similarity_score"] = result.similarity_score
                search_results.append(memory_dict)
                
            # Cache result
            if len(self.query_cache) >= self.max_cache_size:
                self._prune_cache()
            self.query_cache[cache_key] = search_results
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
            
    def _prune_cache(self):
        """Remove oldest items from the cache."""
        # Remove 25% of the cache
        items_to_remove = max(1, int(self.max_cache_size * 0.25))
        for _ in range(items_to_remove):
            if self.query_cache:
                self.query_cache.pop(next(iter(self.query_cache)))
                
    async def create_dag_task(self, 
                             operation: str, 
                             params: Dict[str, Any],
                             task_id: str = None) -> Dict[str, Any]:
        """
        Create a DAG task for memory operations.
        
        This method creates a task definition that can be added to a DAG.
        When executed, the task will use this service to perform the memory operation.
        
        Args:
            operation: Operation type ('create', 'get', 'update', 'delete', 'search')
            params: Operation parameters
            task_id: Optional task ID
            
        Returns:
            DAG task definition
        """
        task_id = task_id or f"memory_{operation}_{str(uuid.uuid4())[:8]}"
        
        # Create task definition
        task = {
            "id": task_id,
            "type": "memory_operation",
            "params": {
                "operation": operation,
                "operation_params": params
            },
            "dependencies": []
        }
        
        return task


# Global service instance for convenience
_memory_service_instance: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """
    Get the global memory service instance.
    
    Returns:
        MemoryService instance
    """
    global _memory_service_instance
    
    if _memory_service_instance is None:
        try:
            from service_registry import get_registry
            
            # Check if registered in service registry
            registry = get_registry()
            
            if registry.has_service_instance("memory"):
                _memory_service_instance = registry.get_service("memory")
            else:
                # Create and register a new instance
                _memory_service_instance = MemoryService()
                # Register in the registry
                registry.register_instance(_memory_service_instance)
        except Exception as e:
            # If registry access fails, create a standalone instance
            logger = get_logger("memory_service")
            logger.warning(f"Could not access service registry: {e}. Creating standalone instance.")
            _memory_service_instance = MemoryService()
            
    return _memory_service_instance
