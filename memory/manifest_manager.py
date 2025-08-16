"""
ManifestManager for standardized manifest operations.

This module provides a centralized way to manage Scout run manifests with
consistent structure, status tracking, and output handling.
"""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class ManifestManager:
    """Manages Scout run manifests with standardized structure and operations."""

    def __init__(self, manifest_path: Union[str, Path], create_if_missing: bool = True):
        """Initialize the manifest manager.
        
        Args:
            manifest_path: Path to the manifest file
            create_if_missing: Create the manifest file if it doesn't exist
        """
        self.manifest_path = Path(manifest_path)
        self._manifest = None
        
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                if create_if_missing:
                    self._create_empty_manifest()
                else:
                    raise ValueError(f"Invalid JSON in manifest: {manifest_path}")
        elif create_if_missing:
            self._create_empty_manifest()
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        # Ensure the manifest has the required structure
        self._ensure_structure()
    
    def _create_empty_manifest(self) -> None:
        """Create an empty manifest with the required structure."""
        run_id = self.manifest_path.parent.name
        now = datetime.datetime.now().isoformat()
        
        self._manifest = {
            "schema": "scout_plan_v2",
            "version": "2.0",
            "run_id": run_id,
            "run_metadata": {
                "started_at": now,
                "updated_at": now,
                "status": "initialized",
                "overall_progress": 0.0
            },
            "dag": {
                "nodes": []
            },
            "stages": {}
        }
        self._save()
    
    def _ensure_structure(self) -> None:
        """Ensure the manifest has the required structure."""
        # Upgrade from v1 to v2 if needed
        if self._manifest.get("schema") == "scout_plan_v1":
            self._upgrade_from_v1()
        
        # Ensure required top-level keys
        self._manifest.setdefault("schema", "scout_plan_v2")
        self._manifest.setdefault("version", "2.0")
        self._manifest.setdefault("run_id", self.manifest_path.parent.name)
        
        # Ensure run_metadata
        if "run_metadata" not in self._manifest:
            now = datetime.datetime.now().isoformat()
            self._manifest["run_metadata"] = {
                "started_at": now,
                "updated_at": now,
                "status": "initialized",
                "overall_progress": 0.0
            }
        
        # Ensure DAG structure
        if "dag" not in self._manifest:
            self._manifest["dag"] = {"nodes": []}
        elif "nodes" not in self._manifest["dag"]:
            self._manifest["dag"]["nodes"] = []
        
        # Ensure stages
        self._manifest.setdefault("stages", {})
        
        # Update timestamp
        self._manifest["run_metadata"]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save changes
        self._save()
    
    def _upgrade_from_v1(self) -> None:
        """Upgrade manifest from v1 to v2 format."""
        # Copy existing data
        old_manifest = dict(self._manifest)
        run_id = old_manifest.get("run_id", self.manifest_path.parent.name)
        now = datetime.datetime.now().isoformat()
        
        # Create new structure
        new_manifest = {
            "schema": "scout_plan_v2",
            "version": "2.0",
            "run_id": run_id,
            "run_metadata": {
                "started_at": now,
                "updated_at": now,
                "status": "running",
                "overall_progress": 0.0
            },
            "target_market": old_manifest.get("target_market", ""),
            "time_window": old_manifest.get("time_window", "12m"),
            "sources": old_manifest.get("sources", []),
            "keywords": old_manifest.get("keywords", []),
            "subreddits": old_manifest.get("subreddits", []),
            "limits": old_manifest.get("limits", {}),
            "dag": old_manifest.get("dag", {"nodes": []}),
            "stages": old_manifest.get("stages", {})
        }
        
        # Update node structure if nodes exist
        if "nodes" in new_manifest["dag"]:
            for node in new_manifest["dag"]["nodes"]:
                # Add status if not present
                if "status" not in node:
                    node_id = node.get("id", "unknown")
                    stage_status = "completed" if node_id in new_manifest["stages"] else "pending"
                    
                    node["status"] = {
                        "state": stage_status,
                        "started_at": now,
                        "updated_at": now,
                        "duration_seconds": 0,
                        "attempts": 0,
                        "retries_remaining": 3,
                        "error": None
                    }
                
                # Add metrics if not present
                if "metrics" not in node:
                    node["metrics"] = {
                        "tokens_used": 0,
                        "cost": 0.0,
                        "backend": None,
                        "model": None
                    }
                
                # Add outputs structure if not present
                if "outputs" in node and isinstance(node["outputs"], list) and len(node["outputs"]) > 0:
                    output_location = node["outputs"][0]
                    node["outputs"] = {
                        "location": output_location,
                        "artifacts": []
                    }
        
        self._manifest = new_manifest
    
    def _save(self) -> None:
        """Save the manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))
    
    def get_manifest(self) -> Dict[str, Any]:
        """Get the full manifest."""
        return self._manifest
    
    def update_run_status(self, status: str, progress: Optional[float] = None) -> None:
        """Update the run status and progress.
        
        Args:
            status: New status (initialized, running, completed, failed)
            progress: Overall progress as a fraction (0.0 to 1.0)
        """
        self._manifest["run_metadata"]["status"] = status
        self._manifest["run_metadata"]["updated_at"] = datetime.datetime.now().isoformat()
        
        if progress is not None:
            self._manifest["run_metadata"]["overall_progress"] = max(0.0, min(1.0, progress))
        
        self._save()
    
    def update_node_status(
        self, 
        node_id: str, 
        state: str, 
        error: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        attempt: bool = False
    ) -> None:
        """Update a node's status.
        
        Args:
            node_id: ID of the node to update
            state: New state (pending, running, completed, failed)
            error: Error details if state is 'failed'
            duration_seconds: Duration of execution in seconds
            attempt: Increment the attempt counter
        """
        node = self._get_node(node_id)
        if not node:
            # If node not found in DAG, check if it exists in stages
            if node_id in self._manifest["stages"]:
                # Update stage status directly
                self._manifest["stages"][node_id]["status"] = state
                self._manifest["stages"][node_id]["updated_at"] = datetime.datetime.now().isoformat()
                if error is not None:
                    self._manifest["stages"][node_id]["error"] = error
                if duration_seconds is not None:
                    self._manifest["stages"][node_id]["duration_seconds"] = duration_seconds
                self._save()
            return
        
        # Initialize status if not present
        if "status" not in node:
            now = datetime.datetime.now().isoformat()
            node["status"] = {
                "state": "pending",
                "started_at": now,
                "updated_at": now,
                "duration_seconds": 0,
                "attempts": 0,
                "retries_remaining": 3,
                "error": None
            }
        
        # Update status
        node["status"]["state"] = state
        node["status"]["updated_at"] = datetime.datetime.now().isoformat()
        
        if state == "running" and node["status"].get("state") != "running":
            node["status"]["started_at"] = datetime.datetime.now().isoformat()
        
        if error is not None:
            node["status"]["error"] = error
        
        if duration_seconds is not None:
            node["status"]["duration_seconds"] = duration_seconds
        
        if attempt:
            node["status"]["attempts"] += 1
            if node["status"]["retries_remaining"] > 0:
                node["status"]["retries_remaining"] -= 1
        
        # Update stage status
        stage_id = node.get("id")
        if stage_id:
            self._manifest["stages"].setdefault(stage_id, {})
            self._manifest["stages"][stage_id]["status"] = state
            self._manifest["stages"][stage_id]["updated_at"] = datetime.datetime.now().isoformat()
            
            if error is not None:
                self._manifest["stages"][stage_id]["error"] = error
            
            if duration_seconds is not None:
                self._manifest["stages"][stage_id]["duration_seconds"] = duration_seconds
        
        # Update run status based on node states
        if state == "failed":
            self._manifest["run_metadata"]["status"] = "failed"
        elif state == "completed":
            # Check if all nodes are completed
            all_completed = all(
                n.get("status", {}).get("state") == "completed"
                for n in self._manifest["dag"]["nodes"]
            )
            if all_completed:
                self._manifest["run_metadata"]["status"] = "completed"
        
        # Calculate overall progress
        total_nodes = len(self._manifest["dag"]["nodes"])
        if total_nodes > 0:
            completed = sum(1 for n in self._manifest["dag"]["nodes"] 
                          if n.get("status", {}).get("state") == "completed")
            progress = completed / total_nodes
            self._manifest["run_metadata"]["overall_progress"] = progress
        
        self._save()
    
    def record_metrics(self, node_id: str, metrics: Dict[str, Any]) -> None:
        """Record metrics for a node.
        
        Args:
            node_id: ID of the node
            metrics: Metrics to record (tokens_used, cost, backend, model, etc.)
        """
        node = self._get_node(node_id)
        if not node:
            return
        
        # Initialize metrics if not present
        if "metrics" not in node:
            node["metrics"] = {
                "tokens_used": 0,
                "cost": 0.0,
                "backend": None,
                "model": None
            }
            
        # Update metrics with new values
        for key, value in metrics.items():
            node["metrics"][key] = value
            
        # Save changes
        self._save()
        
    def store_node_output(self, node_id: str, data: Dict[str, Any], artifacts: Optional[List[Dict[str, Any]]] = None) -> None:
        """Store output data for a node in the manifest.
        
        Args:
            node_id: The ID of the node
            data: The output data to store
            artifacts: Optional list of file artifacts produced by the node
        """
        manifest = self.get_manifest()
        
        # Ensure stages section exists
        if "stages" not in manifest:
            manifest["stages"] = {}
            
        # Ensure node section exists in stages
        if node_id not in manifest["stages"]:
            manifest["stages"][node_id] = {}
        
        # For plan stage, avoid duplicating the entire plan data structure
        if node_id == "plan" and "dag" in data:
            # Store only essential metadata, not the entire plan
            manifest["stages"][node_id]["data"] = {
                "summary": "Plan completed successfully",
                "reference": "See root level for complete plan data"
            }
        else:
            # Store data normally for other nodes
            manifest["stages"][node_id]["data"] = data
        
        # Store artifacts if provided
        if artifacts:
            manifest["stages"][node_id]["artifacts"] = artifacts
            
        # Update timestamp
        manifest["stages"][node_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save the manifest
        self._save()
        
    def get_node_output(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node's output data.
        
        Args:
            node_id: ID of the node
            
        Returns:
            The node's output data or None if not found
        """
        if node_id in self._manifest["stages"] and "data" in self._manifest["stages"][node_id]:
            return self._manifest["stages"][node_id]["data"]
        return None
    
    def _get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            The node dict or None if not found
        """
        for node in self._manifest["dag"]["nodes"]:
            if node.get("id") == node_id:
                return node
        return None
        
    def update_run_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update run metadata with additional information.
        
        Args:
            metadata: Dictionary of metadata to update/add to run_metadata
        """
        for key, value in metadata.items():
            self._manifest["run_metadata"][key] = value
            
        # Always update the timestamp
        self._manifest["run_metadata"]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save changes
        self._save()
        
    def record_error(self, node_id: str, error_message: str, error_type: str = "execution_error") -> None:
        """Record an error for a node.
        
        Args:
            node_id: ID of the node that encountered the error
            error_message: Error message to record
            error_type: Type of error (default: execution_error)
        """
        error_data = {
            "type": error_type,
            "message": error_message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update node status with error
        self.update_node_status(
            node_id=node_id,
            state="failed",
            error=error_data
        )
