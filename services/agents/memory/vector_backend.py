"""
Vector Database Backend for Memory Service.

This module provides a vector database implementation for memory storage
with semantic search capabilities.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    # Optional dependencies for vector embeddings and storage
    import faiss
    import torch
    import sentence_transformers
    VECTOR_SUPPORT = True
except ImportError:
    VECTOR_SUPPORT = False

from .service import (
    MemoryBackend, MemoryBackendType, Memory, MemoryType, MemorySearchResult
)


class VectorDatabaseBackend(MemoryBackend):
    """Vector database backend with embedding and semantic search."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the vector database backend."""
        super().__init__(MemoryBackendType.VECTOR_DB, config)
        self.file_backend = None  # Will use file backend for storage
        self.embedding_model = None
        self.index = None
        self.vector_dim = 384  # Default embedding dimension
        self.memory_to_index = {}  # Map memory ID to index position
        self.index_to_memory = {}  # Map index position to memory ID
        
    async def initialize(self) -> bool:
        """Initialize the vector database."""
        if not VECTOR_SUPPORT:
            self.logger.error("Vector database dependencies not available")
            return False
            
        try:
            # Initialize file backend for actual storage
            from .service import FileSystemBackend
            self.file_backend = FileSystemBackend(self.config)
            await self.file_backend.initialize()
            
            # Get vector dimension from config
            self.vector_dim = self.config.get("vector_dim", 384)
            
            # Load or create FAISS index
            index_path = Path(self.config.get("index_path", "./vector_index"))
            index_file = index_path / "memory_vectors.index"
            
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                self.logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create a new index
                self.index = faiss.IndexFlatL2(self.vector_dim)
                index_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created new FAISS index with dimension {self.vector_dim}")
                
            # Load mapping files
            mapping_file = index_path / "memory_mapping.npz"
            if mapping_file.exists():
                mappings = np.load(str(mapping_file), allow_pickle=True)
                self.memory_to_index = mappings["memory_to_index"].item() if "memory_to_index" in mappings else {}
                self.index_to_memory = mappings["index_to_memory"].item() if "index_to_memory" in mappings else {}
                
            # Initialize embedding model
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = sentence_transformers.SentenceTransformer(model_name)
            
            self.logger.info(f"Vector database backend initialized with {len(self.memory_to_index)} mappings")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            return False
            
    async def _save_mappings(self) -> bool:
        """Save index mappings to disk."""
        try:
            index_path = Path(self.config.get("index_path", "./vector_index"))
            mapping_file = index_path / "memory_mapping.npz"
            
            np.savez(
                str(mapping_file),
                memory_to_index=self.memory_to_index,
                index_to_memory=self.index_to_memory
            )
            
            # Also save the index
            index_file = index_path / "memory_vectors.index"
            faiss.write_index(self.index, str(index_file))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save vector mappings: {e}")
            return False
            
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
            
        # Generate embedding
        embedding = self.embedding_model.encode(text)
        return embedding.astype(np.float32)
        
    async def store(self, memory: Memory) -> bool:
        """Store memory and its embedding."""
        try:
            # First store in file backend
            success = await self.file_backend.store(memory)
            if not success:
                return False
                
            # For text or code memories, create and store embedding
            if memory.memory_type in [MemoryType.TEXT, MemoryType.CODE]:
                # Generate embedding if not provided
                if memory.embedding is None:
                    memory.embedding = self._get_embedding(str(memory.content)).tolist()
                    # Update the stored memory with embedding
                    await self.file_backend.store(memory)
                
                # Add to FAISS index
                vector = np.array([memory.embedding], dtype=np.float32)
                
                if memory.memory_id in self.memory_to_index:
                    # Update existing vector
                    idx = self.memory_to_index[memory.memory_id]
                    # This is a simplification - FAISS doesn't directly support updates
                    # In a real system, you'd need to remove and re-add, or use a mutable index
                    self.logger.warning("Vector update not fully supported, adding as new vector")
                    
                # Add as new vector
                idx = self.index.ntotal
                self.index.add(vector)
                
                # Update mappings
                self.memory_to_index[memory.memory_id] = idx
                self.index_to_memory[idx] = memory.memory_id
                
                # Save mappings
                await self._save_mappings()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory in vector database: {e}")
            return False
            
    async def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory from file backend."""
        return await self.file_backend.retrieve(memory_id)
        
    async def delete(self, memory_id: str) -> bool:
        """Delete memory and its vector if exists."""
        try:
            # Delete from file backend
            success = await self.file_backend.delete(memory_id)
            
            # Remove from vector index if it exists
            if memory_id in self.memory_to_index:
                # In real FAISS usage, you'd need to rebuild the index
                # This is a simplification - we just update the mappings
                idx = self.memory_to_index[memory_id]
                del self.memory_to_index[memory_id]
                if idx in self.index_to_memory:
                    del self.index_to_memory[idx]
                
                # Save mappings
                await self._save_mappings()
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory from vector database: {e}")
            return False
            
    async def list(self, 
                 limit: int = 100,
                 tags: List[str] = None,
                 memory_type: MemoryType = None,
                 from_timestamp: float = None,
                 to_timestamp: float = None) -> List[Memory]:
        """List memories with filters (delegates to file backend)."""
        return await self.file_backend.list(
            limit=limit,
            tags=tags,
            memory_type=memory_type,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        
    async def search(self,
                   query: str,
                   limit: int = 10,
                   tags: List[str] = None,
                   memory_type: MemoryType = None,
                   threshold: float = 0.0) -> List[MemorySearchResult]:
        """Semantic search using vector embeddings."""
        try:
            # Create query embedding
            query_vector = self._get_embedding(query)
            query_vector = np.expand_dims(query_vector, axis=0)
            
            # Perform vector search
            k = min(limit * 4, self.index.ntotal)  # Get more results to filter
            if k == 0:
                return []
                
            distances, indices = self.index.search(query_vector, k)
            
            # Convert to memory results
            results = []
            for i, idx in enumerate(indices[0]):
                # Get memory ID
                if idx not in self.index_to_memory:
                    continue
                    
                memory_id = self.index_to_memory[idx]
                
                # Get the memory
                memory = await self.retrieve(memory_id)
                if not memory:
                    continue
                    
                # Filter by type
                if memory_type and memory.memory_type != memory_type:
                    continue
                    
                # Filter by tags
                if tags and not any(tag in memory.tags for tag in tags):
                    continue
                    
                # Calculate similarity score (0 to 1, where 1 is most similar)
                # FAISS returns squared L2 distance, convert to similarity
                distance = distances[0][i]
                similarity = 1.0 / (1.0 + distance)
                
                if similarity >= threshold:
                    results.append(MemorySearchResult(memory, similarity))
                    
                    # Check if we have enough results after filtering
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            # Fall back to text search
            return await self.file_backend.search(query, limit, tags, memory_type, threshold)
            
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory fields."""
        try:
            # Get the existing memory
            memory = await self.retrieve(memory_id)
            if not memory:
                return False
                
            # Update fields
            content_updated = False
            for key, value in updates.items():
                if key == 'content':
                    memory.content = value
                    content_updated = True
                elif key == 'metadata':
                    memory.metadata.update(value)
                elif key == 'tags':
                    memory.tags = value
                    
            # If content updated, regenerate embedding
            if content_updated and memory.memory_type in [MemoryType.TEXT, MemoryType.CODE]:
                memory.embedding = self._get_embedding(str(memory.content)).tolist()
                
            # Store updated memory (will handle vector update)
            return await self.store(memory)
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
