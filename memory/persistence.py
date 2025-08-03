"""
Memory persistence system for ScoutAgent.

Provides long-term storage and retrieval of workflow states, agent memories,
and execution contexts across sessions.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib

from ..custom_logging.logger import get_logger


class MemoryStore:
    """Base class for memory storage backends."""
    
    def save(self, key: str, data: Any) -> None:
        """Save data with given key."""
        raise NotImplementedError
    
    def load(self, key: str) -> Optional[Any]:
        """Load data by key."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        raise NotImplementedError
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        raise NotImplementedError


class FileMemoryStore(MemoryStore):
    """File-based memory storage using JSON and pickle."""
    
    def __init__(self, base_path: str = "memory_store"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("FileMemoryStore")
    
    def _get_file_path(self, key: str, use_pickle: bool = False) -> Path:
        """Get file path for given key."""
        # Create a safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        extension = ".pkl" if use_pickle else ".json"
        return self.base_path / f"{safe_key}{extension}"
    
    def save(self, key: str, data: Any, use_pickle: bool = False) -> None:
        """Save data to file."""
        file_path = self._get_file_path(key, use_pickle)
        
        try:
            if use_pickle or not isinstance(data, (dict, list, str, int, float, bool)):
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Saved data for key: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data for key {key}: {e}")
            raise
    
    def load(self, key: str) -> Optional[Any]:
        """Load data from file."""
        # Try JSON first, then pickle
        json_path = self._get_file_path(key, False)
        pickle_path = self._get_file_path(key, True)
        
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load JSON for key {key}: {e}")
        
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load pickle for key {key}: {e}")
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete data files for key."""
        json_path = self._get_file_path(key, False)
        pickle_path = self._get_file_path(key, True)
        
        deleted = False
        if json_path.exists():
            json_path.unlink()
            deleted = True
        
        if pickle_path.exists():
            pickle_path.unlink()
            deleted = True
        
        if deleted:
            self.logger.info(f"Deleted data for key: {key}")
        
        return deleted
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """List all stored keys."""
        keys = []
        for file_path in self.base_path.glob("*.json"):
            # We can't reverse the hash, so return the hash as key
            key = file_path.stem
            keys.append(key)
        
        for file_path in self.base_path.glob("*.pkl"):
            key = file_path.stem
            if key not in keys:
                keys.append(key)
        
        return keys


class MemoryManager:
    """Manages memory persistence across workflows and sessions."""
    
    def __init__(self, store: Optional[MemoryStore] = None):
        self.store = store or FileMemoryStore()
        self.logger = get_logger("MemoryManager")
        self.session_cache = {}
    
    def save_workflow_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        """Save complete workflow state."""
        key = f"workflow_{workflow_id}"
        state_with_meta = {
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "type": "workflow_state"
        }
        self.store.save(key, state_with_meta)
    
    def load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state."""
        key = f"workflow_{workflow_id}"
        data = self.store.load(key)
        return data["state"] if data else None
    
    def save_agent_memory(self, agent_name: str, memory: Dict[str, Any]) -> None:
        """Save agent-specific memory."""
        key = f"agent_{agent_name}"
        memory_with_meta = {
            "memory": memory,
            "timestamp": datetime.now().isoformat(),
            "type": "agent_memory"
        }
        self.store.save(key, memory_with_meta)
    
    def load_agent_memory(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent-specific memory."""
        key = f"agent_{agent_name}"
        data = self.store.load(key)
        return data["memory"] if data else {}
    
    def save_checkpoint(self, workflow_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """Save execution checkpoint."""
        checkpoint_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        key = f"checkpoint_{checkpoint_id}"
        
        checkpoint_with_meta = {
            "checkpoint": checkpoint_data,
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "type": "checkpoint"
        }
        
        self.store.save(key, checkpoint_with_meta)
        self.logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        key = f"checkpoint_{checkpoint_id}"
        data = self.store.load(key)
        return data["checkpoint"] if data else None
    
    def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoints for a workflow."""
        all_keys = self.store.list_keys()
        checkpoints = []
        
        for key in all_keys:
            if key.startswith("checkpoint_") and workflow_id in key:
                checkpoint_id = key.replace("checkpoint_", "")
                checkpoints.append(checkpoint_id)
        
        return sorted(checkpoints, reverse=True)
    
    def save_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Save shared context between agents."""
        key = f"context_{context_id}"
        context_with_meta = {
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "type": "shared_context"
        }
        self.store.save(key, context_with_meta)
    
    def load_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Load shared context."""
        key = f"context_{context_id}"
        data = self.store.load(key)
        return data["context"] if data else {}
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Save session data."""
        key = f"session_{session_id}"
        session_with_meta = {
            "session": session_data,
            "timestamp": datetime.now().isoformat(),
            "type": "session"
        }
        self.store.save(key, session_with_meta)
        self.session_cache[session_id] = session_data
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data."""
        if session_id in self.session_cache:
            return self.session_cache[session_id]
        
        key = f"session_{session_id}"
        data = self.store.load(key)
        
        if data:
            session_data = data["session"]
            self.session_cache[session_id] = session_data
            return session_data
        
        return None
    
    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Clean up old data based on age."""
        # This is a placeholder - would need to implement age-based cleanup
        self.logger.info(f"Cleanup placeholder: would remove data older than {max_age_days} days")
        return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        keys = self.store.list_keys()
        
        stats = {
            "total_keys": len(keys),
            "workflow_states": len([k for k in keys if k.startswith("workflow_")]),
            "agent_memories": len([k for k in keys if k.startswith("agent_")]),
            "checkpoints": len([k for k in keys if k.startswith("checkpoint_")]),
            "contexts": len([k for k in keys if k.startswith("context_")]),
            "sessions": len([k for k in keys if k.startswith("session_")])
        }
        
        return stats


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager
