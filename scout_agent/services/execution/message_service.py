"""
StageMessageService - Direct in-memory message passing between stages with manifest fallback.

This service provides efficient data routing between agent stages with workflow isolation
and automatic fallback to manifest persistence for recovery.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from scout_agent.service_registry import ServiceBase, service, requires
from scout_agent.custom_logging import get_logger
from scout_agent.memory.manifest_manager import ManifestManager


@service(name="stage_message", singleton=True)
@requires("config", optional=True)
class StageMessageService(ServiceBase):
    """
    Service for direct in-memory message passing between stages with manifest fallback.
    
    This service provides:
    - Primary: In-memory message queues per workflow
    - Secondary: Manifest persistence for recovery
    - Cross-agent data routing
    - Workflow isolation and cleanup
    """
    
    def __init__(self):
        """Initialize the stage message service."""
        super().__init__()
        self.logger = get_logger("stage_message_service")
        self._workflow_messages = {}  # workflow_id -> {stage_id -> {data, timestamp, consumed}}
        self._workflow_locks = {}     # workflow_id -> asyncio.Lock
        self._locks = {}              # workflow_id -> asyncio.Lock (for backward compatibility)
        self._workflow_metadata = {}  # workflow_id -> metadata
        self._messages = {}           # Simple key-value store for direct messages
    
    async def _initialize(self, registry) -> bool:
        """Initialize the stage message service."""
        self.logger.info("Initializing stage message service")
        
        try:
            # Get config service if available
            if registry:
                try:
                    self.config = registry.get_service("config")
                except Exception as e:
                    self.logger.warning(f"Could not get config service: {e}")
            
            self.logger.info("Stage message service initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize stage message service: {e}")
            return False
    
    async def _start(self) -> bool:
        """Start the stage message service."""
        self.logger.info("Starting stage message service")
        return True
    
    async def _stop(self) -> bool:
        """Stop the stage message service."""
        self.logger.info("Stopping stage message service")
        return True
    
    def _get_workflow_lock(self, workflow_id: str) -> asyncio.Lock:
        """Get or create a lock for the workflow."""
        if workflow_id not in self._locks:
            self._locks[workflow_id] = asyncio.Lock()
            # Also store in _workflow_locks for consistency
            self._workflow_locks[workflow_id] = self._locks[workflow_id]
        return self._locks[workflow_id]
    
    def _ensure_workflow(self, workflow_id: str) -> None:
        """Ensure workflow storage exists."""
        if workflow_id not in self._workflow_messages:
            self._workflow_messages[workflow_id] = {}
            self._workflow_metadata[workflow_id] = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "stage_count": 0
            }
    
    def publish_stage_output(self, workflow_id: str, stage_id: str, data: Dict[str, Any]) -> None:
        """
        Publish stage output data for consumption by subsequent stages.
        
        Args:
            workflow_id: Workflow identifier
            stage_id: Stage identifier (e.g., "scout_collect", "screener_think")
            data: Output data from the stage
        """
        try:
            # Ensure workflow exists
            self._ensure_workflow(workflow_id)
            
            # Store in memory
            self._workflow_messages[workflow_id][stage_id] = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "consumed": False
            }
            
            # Update metadata
            self._workflow_metadata[workflow_id]["last_updated"] = datetime.now().isoformat()
            self._workflow_metadata[workflow_id]["stage_count"] = len(self._workflow_messages[workflow_id])
            
            self.logger.info(f"Published stage output: {workflow_id}/{stage_id}")
            
            # Also save to manifest as fallback - use synchronous version
            self._save_to_manifest_fallback_sync(workflow_id, stage_id, data)
        except Exception as e:
            self.logger.warning(f"Failed to publish via message service: {e}")
    
    def consume_stage_input(self, workflow_id: str, stage_id: str, source: str = None) -> Optional[Dict[str, Any]]:
        """
        Consume input data for a stage from previous stages.
        
        Args:
            workflow_id: Workflow identifier
            stage_id: Stage identifier that wants to consume data
            source: Optional source identifier (e.g., 'reddit') for source-specific data
            
        Returns:
            Input data for the stage or None if not available
        """
        try:
            self._ensure_workflow(workflow_id)
            
            # Extract agent_id from stage_id (e.g., 'scout_think' -> 'scout')
            agent_id = stage_id.split('_')[0] if '_' in stage_id else ''
            
            # If this is a think stage and source is specified, try to get source-specific collect data
            if stage_id.endswith('_think') and source:
                # Try to get source-specific data first
                source_data = self._load_collect_from_manifest_sync(workflow_id, agent_id, source)
                if source_data:
                    self.logger.info(f"Found source-specific {source} data for {stage_id}")
                    return source_data
            
            # For in-memory messages
            if self._workflow_messages[workflow_id]:
                # If this is a think stage, try to find collect stage data
                if stage_id.endswith('_think'):
                    collect_stage_id = f"{agent_id}_collect"
                    if collect_stage_id in self._workflow_messages[workflow_id]:
                        message = self._workflow_messages[workflow_id][collect_stage_id]
                        message["consumed"] = True
                        data = message.get("data")
                        
                        # If source is specified and data contains that source
                        if source and isinstance(data, dict) and source in data:
                            self.logger.info(f"Consumed source-specific {source} data from {collect_stage_id}")
                            return data[source]
                        
                        self.logger.info(f"Consumed stage output: {workflow_id}/{collect_stage_id}")
                        return data
                
                # Get the most recent stage output as fallback
                latest_stage = None
                latest_timestamp = None
                
                for s_id, message in self._workflow_messages[workflow_id].items():
                    # Skip the current stage
                    if s_id == stage_id:
                        continue
                        
                    timestamp = message.get("timestamp")
                    if latest_timestamp is None or (timestamp and timestamp > latest_timestamp):
                        latest_stage = s_id
                        latest_timestamp = timestamp
                
                if latest_stage:
                    message = self._workflow_messages[workflow_id][latest_stage]
                    message["consumed"] = True
                    self.logger.info(f"Consumed stage output: {workflow_id}/{latest_stage}")
                    return message.get("data")
            
            # Try fallback from manifest with stage-specific logic
            if stage_id.endswith('_think'):
                # For think stage, try to find collect data with source
                if source:
                    source_data = self._load_collect_from_manifest_sync(workflow_id, agent_id, source)
                    if source_data:
                        return source_data
                
                # Try to find any collect data
                collect_stage_id = f"{agent_id}_collect"
                collect_data = self._load_from_manifest_fallback_sync(workflow_id, collect_stage_id)
                if collect_data:
                    return collect_data
            
            # General fallback
            return self._load_from_manifest_fallback_sync(workflow_id, stage_id)
            
        except Exception as e:
            self.logger.warning(f"Failed to consume via message service: {e}")
            return None
    
    def get_collect_data(self, workflow_id: str, agent_id: str, source: str) -> Dict[str, Any]:
        """
        Get collect data for an agent from a specific source.
        
        Args:
            workflow_id: Workflow ID
            agent_id: Agent ID
            source: Data source (reddit, twitter, etc.)
            
        Returns:
            Collect data or empty dict if not found
        """
        # Always use agent-prefixed stage names for consistency
        stage_id = f"{agent_id}_collect"
        key = f"{workflow_id}:{stage_id}:{source}"
        
        # Try to get from in-memory messages first
        data = self._messages.get(key, {})
        if data:
            self.logger.info(f"Found collect data for {agent_id}/{source} in memory")
            return data
            
        # If not in memory, try to load from manifest
        try:
            # Try to get from workflow messages
            if workflow_id in self._workflow_messages and stage_id in self._workflow_messages[workflow_id]:
                stage_data = self._workflow_messages[workflow_id][stage_id]
                if "data" in stage_data:
                    data = stage_data["data"]
                    # Check if data has source-specific structure
                    if isinstance(data, dict):
                        # Direct source key in data
                        if source in data:
                            self.logger.info(f"Found {source} data in workflow messages")
                            return data[source]
                        # Reddit-specific structure check
                        if source == 'reddit' and ('threads' in data or 'comments' in data):
                            self.logger.info(f"Found Reddit threads/comments in workflow messages")
                            return data
                    return data
                    
            # Try source-specific data loading
            source_data = self._load_collect_from_manifest_sync(workflow_id, agent_id, source)
            if source_data:
                self.logger.info(f"Found source-specific {source} data via _load_collect_from_manifest_sync")
                return source_data
                
            # Fallback to manifest with standard path
            manifest_data = self._load_from_manifest_fallback_sync(workflow_id, stage_id)
            if manifest_data:
                # Check if manifest_data has source-specific structure
                if isinstance(manifest_data, dict):
                    # Direct source key in data
                    if source in manifest_data:
                        self.logger.info(f"Found {source} data in manifest")
                        return manifest_data[source]
                    # Reddit-specific structure check
                    if source == 'reddit' and ('threads' in manifest_data or 'comments' in manifest_data):
                        self.logger.info(f"Found Reddit threads/comments in manifest")
                        return manifest_data
                return manifest_data
                
            # Try legacy node ID format as last resort
            legacy_collect_id = f"collect_{source}"
            legacy_data = self._load_from_manifest_fallback_sync(workflow_id, legacy_collect_id)
            if legacy_data:
                self.logger.info(f"Found legacy data for {legacy_collect_id}")
                return legacy_data
                
            # Legacy fallback - try non-prefixed stage name (for backward compatibility)
            legacy_stage_id = "collect"
            legacy_data = self._load_from_manifest_fallback_sync(workflow_id, legacy_stage_id)
            if legacy_data and source in legacy_data:
                self.logger.warning(f"Using legacy non-prefixed stage name '{legacy_stage_id}' for {agent_id}/{source}")
                return legacy_data[source]
                
            # Additional legacy fallback - try source-specific node ID (e.g., collect_reddit)
            source_specific_id = f"collect_{source}"
            source_data = self._load_from_manifest_fallback_sync(workflow_id, source_specific_id)
            if source_data:
                self.logger.warning(f"Using legacy source-specific node ID '{source_specific_id}' for {agent_id}/{source}")
                return source_data
                
        except Exception as e:
            self.logger.warning(f"Error retrieving collect data for {agent_id}/{source}: {e}")
            
        # Return empty dict if nothing found
        return {}
    
    async def _load_from_manifest_fallback(self, workflow_id: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """Load data from manifest fallback storage (async version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                manifest_manager = ManifestManager(manifest_path)
                data = manifest_manager.get_node_output(stage_id)
                if data:
                    self.logger.debug(f"Loaded from manifest fallback: {stage_id}")
                    return data
        except Exception as e:
            self.logger.warning(f"Failed to load from manifest fallback: {e}")
        
        return None
        
    def _load_from_manifest_fallback_sync(self, workflow_id: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """Load data from manifest fallback storage (synchronous version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                from scout_agent.memory.manifest_manager import ManifestManager
                manifest_manager = ManifestManager(manifest_path)
                data = manifest_manager.get_node_output(stage_id)
                if data:
                    self.logger.debug(f"Loaded from manifest fallback (sync): {stage_id}")
                    return data
        except Exception as e:
            self.logger.warning(f"Failed to load from manifest fallback (sync): {e}")
        
        return None
    
    def cleanup_workflow(self, workflow_id: str) -> None:
        """
        Clean up workflow data from memory.
        
        Args:
            workflow_id: Workflow identifier to clean up
        """
        try:
            if workflow_id in self._workflow_messages:
                stage_count = len(self._workflow_messages[workflow_id])
                del self._workflow_messages[workflow_id]
                del self._workflow_metadata[workflow_id]
                if workflow_id in self._locks:
                    del self._locks[workflow_id]
                if workflow_id in self._workflow_locks:
                    del self._workflow_locks[workflow_id]
                
                self.logger.info(f"Cleaned up workflow {workflow_id} ({stage_count} stages)")
        except Exception as e:
            self.logger.warning(f"Error during workflow cleanup: {e}")
    
    async def _save_to_manifest_fallback(self, workflow_id: str, stage_id: str, data: Dict[str, Any]) -> None:
        """Save data to manifest as fallback storage (async version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                manifest_manager = ManifestManager(manifest_path)
                manifest_manager.store_node_output(stage_id, data)
                self.logger.debug(f"Saved to manifest fallback: {stage_id}")
        except Exception as e:
            self.logger.warning(f"Failed to save to manifest fallback: {e}")
            
    def _save_to_manifest_fallback_sync(self, workflow_id: str, stage_id: str, data: Dict[str, Any]) -> None:
        """Save data to manifest as fallback storage (synchronous version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                manifest_manager = ManifestManager(manifest_path)
                
                # Check if this is source-specific data (e.g., Reddit)
                if stage_id.endswith('_collect') and isinstance(data, dict):
                    # For collect stage, check if data contains source-specific information
                    if 'threads' in data and ('source' in data or any(t.get('source') == 'reddit' for t in data.get('threads', []) if isinstance(t, dict))):
                        # This is Reddit data, store it in source-specific location
                        source_data = {"reddit": data}
                        self.logger.info(f"Storing Reddit-specific data for {stage_id}")
                        manifest_manager.store_node_output(stage_id, source_data)
                    else:
                        # Regular data storage
                        manifest_manager.store_node_output(stage_id, data)
                else:
                    # Regular data storage for non-collect stages
                    manifest_manager.store_node_output(stage_id, data)
                    
                self.logger.debug(f"Saved to manifest fallback: {stage_id}")
        except Exception as e:
            self.logger.warning(f"Failed to save to manifest fallback: {e}")
    
    async def _load_from_manifest_fallback(self, workflow_id: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """Load data from manifest fallback storage (async version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                manifest_manager = ManifestManager(manifest_path)
                data = manifest_manager.get_node_output(stage_id)
                if data:
                    self.logger.debug(f"Loaded from manifest fallback: {stage_id}")
                    return data
        except Exception as e:
            self.logger.warning(f"Failed to load from manifest fallback: {e}")
        
        return None
        
    def _load_from_manifest_fallback_sync(self, workflow_id: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """Load data from manifest fallback storage (synchronous version)."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if manifest_path.exists():
                manifest_manager = ManifestManager(manifest_path)
                
                # First try to get data directly from the specified stage_id
                data = manifest_manager.get_node_output(stage_id)
                if data:
                    self.logger.debug(f"Loaded from manifest fallback: {stage_id}")
                    return data
                
                # If this is a think stage looking for collect data, try alternative paths
                if stage_id.endswith('_think'):
                    # Extract agent_id from stage_id (e.g., 'scout_think' -> 'scout')
                    agent_id = stage_id.split('_')[0] if '_' in stage_id else ''
                    
                    # Try to find collect data for this agent
                    collect_stage_id = f"{agent_id}_collect" if agent_id else "collect"
                    collect_data = manifest_manager.get_node_output(collect_stage_id)
                    
                    if collect_data:
                        # Check if it has source-specific data
                        if isinstance(collect_data, dict) and 'reddit' in collect_data:
                            self.logger.info(f"Found source-specific Reddit data for {collect_stage_id}")
                            return collect_data
                        else:
                            self.logger.debug(f"Loaded collect data from manifest: {collect_stage_id}")
                            return collect_data
                    
                    # Try legacy node ID format (e.g., 'collect_reddit')
                    legacy_collect_id = "collect_reddit"
                    legacy_data = manifest_manager.get_node_output(legacy_collect_id)
                    if legacy_data:
                        self.logger.info(f"Found legacy data for {legacy_collect_id}")
                        return legacy_data
                        
        except Exception as e:
            self.logger.warning(f"Failed to load from manifest fallback: {e}")
        
        return None
    
    def _load_collect_from_manifest_sync(self, workflow_id: str, agent_id: str, source: str) -> Optional[Dict[str, Any]]:
        """Load collect data from manifest with source-specific handling."""
        try:
            # Determine manifest path
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = project_root / "data" / "runs" / workflow_id / "run_manifest.json"
            
            if not manifest_path.exists():
                return None
            
            manifest_manager = ManifestManager(manifest_path)
            manifest = manifest_manager.get_manifest()
            
            collect_stage_id = f"{agent_id}_collect"
            
            # Check stages for source-specific data
            stages = manifest.get("stages", {})
            
            # First try standardized path: stages.{agent_id}_collect.{source}
            if collect_stage_id in stages:
                stage_data = stages[collect_stage_id]
                
                # Look for source-specific data (e.g., stages.scout_collect.reddit)
                if source in stage_data:
                    source_data = stage_data[source]
                    self.logger.info(f"Found source-specific data at {collect_stage_id}.{source}")
                    if isinstance(source_data, dict) and "data" in source_data:
                        return source_data["data"]
                    else:
                        return source_data
                
                # Fallback to general data
                if "data" in stage_data:
                    data = stage_data["data"]
                    if isinstance(data, dict):
                        # Look for source in the data
                        if source in data:
                            self.logger.info(f"Found {source} data in {collect_stage_id}.data")
                            return data[source]
                        # Look for threads/comments (common Reddit structure)
                        if "threads" in data or "comments" in data:
                            self.logger.info(f"Found threads/comments in {collect_stage_id}.data")
                            return data
                    return data
            
            # Try legacy node ID format (e.g., 'collect_reddit')
            legacy_collect_id = f"collect_{source}"
            if legacy_collect_id in stages:
                legacy_data = stages[legacy_collect_id].get("data")
                if legacy_data:
                    self.logger.info(f"Found legacy data at {legacy_collect_id}")
                    return legacy_data
                    
            # Try direct node output as last resort
            direct_data = manifest_manager.get_node_output(legacy_collect_id)
            if direct_data:
                self.logger.info(f"Found direct node output for {legacy_collect_id}")
                return direct_data
            
            self.logger.debug(f"No collect data found in manifest for {collect_stage_id}/{source}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load collect data from manifest: {e}")
            return None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow status information or None if not found
        """
        if workflow_id not in self._workflow_metadata:
            return None
        
        metadata = self._workflow_metadata[workflow_id].copy()
        metadata["stages"] = list(self._workflow_messages.get(workflow_id, {}).keys())
        metadata["active"] = workflow_id in self._workflow_messages
        
        return metadata
    
    def list_workflows(self) -> List[str]:
        """
        List all active workflow IDs.
        
        Returns:
            List of workflow IDs
        """
        return list(self._workflow_messages.keys())
    
    def get_stage_data(self, workflow_id: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific stage.
        
        Args:
            workflow_id: Workflow identifier
            stage_id: Stage identifier
            
        Returns:
            Stage data or None if not found
        """
        if workflow_id not in self._workflow_messages:
            return None
        
        return self._workflow_messages[workflow_id].get(stage_id)
    
    def get_dependency_data(self, workflow_id: str, stage_id: str, dependency_stages: List[str]) -> Dict[str, Any]:
        """
        Get data from multiple dependency stages.
        
        Args:
            workflow_id: Workflow identifier
            stage_id: Current stage identifier
            dependency_stages: List of stage IDs this stage depends on
            
        Returns:
            Dictionary mapping dependency stage IDs to their data
        """
        dependency_data = {}
        
        try:
            self._ensure_workflow(workflow_id)
            
            for dep_stage in dependency_stages:
                if dep_stage in self._workflow_messages[workflow_id]:
                    stage_data = self._workflow_messages[workflow_id][dep_stage]
                    dependency_data[dep_stage] = stage_data["data"]
                else:
                    # Try manifest fallback
                    fallback_data = self._load_from_manifest_fallback_sync(workflow_id, dep_stage)
                    if fallback_data:
                        dependency_data[dep_stage] = fallback_data
            
            self.logger.info(f"Retrieved dependency data for {stage_id}: {list(dependency_data.keys())}")
        except Exception as e:
            self.logger.warning(f"Failed to get dependency data: {e}")
            
        return dependency_data
