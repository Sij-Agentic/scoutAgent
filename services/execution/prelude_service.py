"""
PreludeService - Agent-agnostic prelude generation for sandboxed code execution.

This service extracts and generalizes the prelude logic from ScoutAgent to support
multiple agents with configurable data paths and helper functions.
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List
from scout_agent.service_registry import ServiceBase, service, requires
from scout_agent.custom_logging import get_logger


@service(name="prelude", singleton=True)
@requires("config", optional=True)
class PreludeService(ServiceBase):
    """
    Service for generating agent-agnostic prelude functions for sandboxed code execution.
    
    This service provides:
    - Agent-agnostic helper function generation
    - Configurable data paths (stages.{agent_id}_{stage}.{source})
    - Prelude caching for performance
    - Extensible helper function registry
    """
    
    def __init__(self):
        """Initialize the prelude service."""
        super().__init__(name="prelude", version="1.0.0")
        self.logger = get_logger("service.prelude")
        self.config = None
        self._helper_functions = {}
        self._prelude_cache = {}
        
        # Register default helper functions
        self._register_default_helpers()
    
    async def _initialize(self, registry) -> bool:
        """Initialize the prelude service."""
        self.logger.info("Initializing prelude service")
        
        try:
            # Get config service if available
            if registry:
                try:
                    self.config = registry.get_service("config")
                except Exception as e:
                    self.logger.warning(f"Could not get config service: {e}")
            
            self.logger.info("Prelude service initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize prelude service: {e}")
            return False
    
    async def _start(self) -> bool:
        """Start the prelude service."""
        self.logger.info("Starting prelude service")
        return True
    
    async def _stop(self) -> bool:
        """Stop the prelude service."""
        self.logger.info("Stopping prelude service")
        return True
    
    def _register_default_helpers(self) -> None:
        """Register default helper functions."""
        # Save JSON helper
        self._helper_functions["save_json"] = {
            "code": textwrap.dedent('''
                def save_json(rel_path: str, obj):
                    """Save an object as JSON to a relative path within the run directory."""
                    p = RUN_DIR / rel_path
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w", encoding="utf-8") as f:
                        if isinstance(obj, str):
                            try:
                                obj = json.loads(obj)
                            except Exception:
                                pass
                        json.dump(obj, f, indent=2)
            ''').strip(),
            "description": "Save an object as JSON to a relative path"
        }
        
        # Logging helper
        self._helper_functions["log_to_file_prelude"] = {
            "code": textwrap.dedent('''
                def log_to_file_prelude(message):
                    """Log to both console and file from within sandbox prelude"""
                    import datetime
                    log_dir = Path("/tmp/scout_sandbox_logs")
                    log_dir.mkdir(exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_file = log_dir / f"prelude_debug_{timestamp}.log"
                    print(message)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.datetime.now().isoformat()}: {message}\\n")
                        f.flush()
            ''').strip(),
            "description": "Log messages to both console and file for debugging"
        }
        
        # Manifest save helper
        self._helper_functions["save_to_manifest"] = {
            "code": self._generate_save_to_manifest_code(),
            "description": "Save data to a specific section in the run manifest"
        }
        
        # Manifest read helper
        self._helper_functions["read_from_manifest"] = {
            "code": textwrap.dedent('''
                def read_from_manifest(section_key: str):
                    """Read data from a specific section in the run manifest."""
                    manifest_path = RUN_DIR / "run_manifest.json"
                    try:
                        if not manifest_path.exists():
                            return None
                        manifest = json.loads(manifest_path.read_text())
                        
                        # Parse section key like "stages.scout_collect.reddit" -> ["stages", "scout_collect", "reddit"]
                        keys = section_key.split(".")
                        current = manifest
                        for key in keys:
                            current = current.get(key, {})
                        
                        return current.get("data") if isinstance(current, dict) else current
                    except Exception:
                        return None
            ''').strip(),
            "description": "Read data from a specific section in the run manifest"
        }
        
        # MCP call helper
        self._helper_functions["mcp_call"] = {
            "code": self._generate_mcp_call_code(),
            "description": "Make MCP tool calls with proper response parsing"
        }
    
    def _generate_save_to_manifest_code(self) -> str:
        """Generate the save_to_manifest helper function code."""
        return textwrap.dedent('''
            def save_to_manifest(section_key: str, obj):
                """Save data to a specific section in the run manifest."""
                log_to_file_prelude(f"DEBUG: PRELUDE save_to_manifest called with section_key='{section_key}', obj type={type(obj)}")
                
                manifest_path = RUN_DIR / "run_manifest.json"
                log_to_file_prelude(f"DEBUG: PRELUDE Manifest path: {manifest_path}")
                log_to_file_prelude(f"DEBUG: PRELUDE Manifest exists: {manifest_path.exists()}")
                
                try:
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text())
                        log_to_file_prelude(f"DEBUG: Loaded existing manifest with keys: {list(manifest.keys())}")
                    else:
                        manifest = {}
                        log_to_file_prelude(f"DEBUG: Created new manifest")
                except Exception as e:
                    log_to_file_prelude(f"ERROR: Failed to load manifest: {e}")
                    manifest = {}
                
                # Parse section key like "stages.collect_reddit" -> ["stages", "collect_reddit"]
                keys = section_key.split(".")
                log_to_file_prelude(f"DEBUG: Parsed keys: {keys}")
                
                current = manifest
                for key in keys[:-1]:
                    current = current.setdefault(key, {})
                    log_to_file_prelude(f"DEBUG: Navigated to key '{key}', current keys: {list(current.keys()) if isinstance(current, dict) else 'not dict'}")
                
                # Set the data
                if isinstance(obj, str):
                    try:
                        obj = json.loads(obj)
                        log_to_file_prelude(f"DEBUG: Parsed string obj to dict with keys: {list(obj.keys()) if isinstance(obj, dict) else 'not dict'}")
                    except Exception as e:
                        log_to_file_prelude(f"DEBUG: Failed to parse string obj as JSON: {e}")
                        pass
                
                # Ensure we have valid data to save
                if obj is None:
                    log_to_file_prelude(f"WARNING: Attempting to save None object to {section_key}")
                    obj = {"warning": "No data was returned from the tool call"}
                
                log_to_file_prelude(f"DEBUG: Final obj type: {type(obj)}")
                if isinstance(obj, dict):
                    log_to_file_prelude(f"DEBUG: Final obj keys: {list(obj.keys())}")
                    if "threads" in obj:
                        log_to_file_prelude(f"DEBUG: Found {len(obj['threads'])} threads in obj")
                
                # For stages, ensure we have the proper structure and save the actual data
                if keys[0] == "stages":
                    stage_name = keys[-1]
                    log_to_file_prelude(f"DEBUG: Processing stage '{stage_name}'")
                    current.setdefault(stage_name, {})
                    
                    # Handle source-specific data paths (e.g., stages.scout_collect.reddit)
                    if len(keys) > 2:
                        source_name = keys[-1]
                        stage_name = keys[-2]
                        log_to_file_prelude(f"DEBUG: Source-specific save: stage='{stage_name}', source='{source_name}'")
                        # Save directly to the stage without nesting
                        current[source_name] = obj
                        log_to_file_prelude(f"DEBUG: Saved {source_name} data directly to stage")
                    else:
                        # Standard stage data
                        current[stage_name]["data"] = obj
                        current[stage_name]["updated_at"] = __import__("datetime").datetime.now().isoformat()
                        current[stage_name]["status"] = "completed"
                    
                    log_to_file_prelude(f"DEBUG: Set data for stage '{stage_name}'")
                else:
                    current[keys[-1]] = {
                        "data": obj,
                        "updated_at": __import__("datetime").datetime.now().isoformat()
                    }
                    log_to_file_prelude(f"DEBUG: Set data for non-stage key '{keys[-1]}'")
                
                # Write the updated manifest
                try:
                    manifest_json = json.dumps(manifest, indent=2)
                    log_to_file_prelude(f"DEBUG: Generated manifest JSON ({len(manifest_json)} chars)")
                    manifest_path.write_text(manifest_json)
                    log_to_file_prelude(f"DEBUG: Successfully wrote manifest to {manifest_path}")
                    
                    # Verify the write
                    if manifest_path.exists():
                        verify_content = manifest_path.read_text()
                        log_to_file_prelude(f"DEBUG: Verified manifest file exists and has {len(verify_content)} chars")
                    else:
                        log_to_file_prelude(f"ERROR: Manifest file does not exist after write!")
                    log_to_file_prelude(f"DEBUG: Manifest saved successfully")
                except Exception as e:
                    log_to_file_prelude(f"ERROR: Failed to save manifest: {e}")
                    import traceback
                    log_to_file_prelude(f"DEBUG: Traceback: {traceback.format_exc()}")
        ''').strip()
    
    def _generate_mcp_call_code(self) -> str:
        """Generate the mcp_call helper function code."""
        return textwrap.dedent('''
            def mcp_call(tool: str, params: dict):
                """Make MCP tool calls with proper response parsing."""
                def _ensure_payload_local(res):
                    """Local version of _ensure_payload for sandboxed execution with double-nested JSON parsing"""
                    try:
                        if hasattr(res, "content") and res.content is not None:
                            if isinstance(res.content, list) and res.content:
                                content_item = res.content[0]
                                if hasattr(content_item, "text"):
                                    content = content_item.text
                                    try:
                                        # First layer: Parse the outer JSON structure
                                        first_parse = json.loads(content)
                                        print(f"DEBUG: First parse keys: {list(first_parse.keys()) if isinstance(first_parse, dict) else 'not dict'}")
                                        
                                        # Check if this is the double-nested structure we expect
                                        if isinstance(first_parse, dict) and "content" in first_parse:
                                            content_list = first_parse["content"]
                                            print(f"DEBUG: Found content list with {len(content_list)} items")
                                            if isinstance(content_list, list) and content_list:
                                                inner_item = content_list[0]
                                                print(f"DEBUG: Inner item keys: {list(inner_item.keys()) if isinstance(inner_item, dict) else 'not dict'}")
                                                if isinstance(inner_item, dict) and "text" in inner_item:
                                                    inner_text = inner_item["text"]
                                                    print(f"DEBUG: Found inner text, attempting to parse as JSON")
                                                    try:
                                                        # Second layer: Parse the nested JSON string
                                                        second_parse = json.loads(inner_text)
                                                        print(f"DEBUG: Successfully parsed double-nested JSON, found keys: {list(second_parse.keys()) if isinstance(second_parse, dict) else 'not dict'}")
                                                        if isinstance(second_parse, dict) and "threads" in second_parse:
                                                            print(f"DEBUG: Found {len(second_parse['threads'])} threads in parsed data")
                                                        return second_parse
                                                    except json.JSONDecodeError as e2:
                                                        print(f"DEBUG: Failed to parse inner JSON string: {e2}")
                                                        print(f"DEBUG: Inner text sample: {inner_text[:200]}...")
                                                        # Return the first parse if second fails
                                                        return first_parse
                                        
                                        # If not double-nested, return the first parse
                                        print(f"DEBUG: No double-nesting detected, returning first parse")
                                        return first_parse
                                        
                                    except json.JSONDecodeError as e:
                                        print(f"DEBUG: Failed to parse outer JSON: {e}")
                                        return {"raw": content[:500], "error": str(e)}
                                else:
                                    return {"raw": str(content_item)[:500], "error": "No text attribute"}
                            else:
                                return {"raw": str(res.content), "error": "Content is not a list or is empty"}
                        else:
                            # Try to get raw text or other attributes
                            if hasattr(res, "text"):
                                try:
                                    return json.loads(res.text)
                                except Exception:
                                    return {"raw": res.text[:500], "error": "No content attribute, using text"}
                            elif hasattr(res, "body"):
                                try:
                                    return json.loads(res.body)
                                except Exception:
                                    return {"raw": str(res.body)[:500], "error": "No content attribute, using body"}
                            else:
                                # Try to convert the response object itself to a dict
                                if hasattr(res, "__dict__"):
                                    try:
                                        res_dict = res.__dict__
                                        return res_dict
                                    except Exception as e:
                                        return {"raw": str(res)[:500], "error": f"Failed to convert response to dict: {e}"}
                                return {"raw": str(res)[:500], "error": "No content, text, or body attributes"}
                    except Exception as e:
                        print(f"DEBUG: Exception in _ensure_payload_local: {e}")
                        return {"error": str(e), "raw": str(res)[:500] if res else "None"}
                
                async def _run():
                    # Set up dedicated logging for sandboxed execution
                    import os
                    import datetime
                    
                    # Create logs directory if it doesn't exist
                    log_dir = Path("/tmp/scout_sandbox_logs")
                    log_dir.mkdir(exist_ok=True)
                    
                    # Create timestamped log file
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_file = log_dir / f"mcp_call_{tool}_{timestamp}.log"
                    
                    def log_to_file(message):
                        """Log to both console and file"""
                        print(message)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{datetime.datetime.now().isoformat()}: {message}\\n")
                            f.flush()
                    
                    log_to_file(f"DEBUG: Starting MCP call to {tool} with params: {params}")
                    
                    try:
                        # Load server configs and create multi-client
                        configs = load_server_configs()
                        log_to_file(f"DEBUG: Loaded {len(configs)} server configs")
                        
                        multi_client = MultiMCPClient(configs)
                        log_to_file(f"DEBUG: Created MultiMCPClient")
                        
                        # Initialize the client
                        await multi_client.initialize()
                        log_to_file(f"DEBUG: MultiMCPClient initialized")
                        
                        # Make the tool call
                        log_to_file(f"DEBUG: Calling tool {tool} with params: {json.dumps(params, indent=2)}")
                        result = await multi_client.call_tool(tool, params)
                        log_to_file(f"DEBUG: Tool call completed, result type: {type(result)}")
                        
                        # Process the result
                        payload = _ensure_payload_local(result)
                        log_to_file(f"DEBUG: Processed payload type: {type(payload)}")
                        if isinstance(payload, dict):
                            log_to_file(f"DEBUG: Payload keys: {list(payload.keys())}")
                            if "threads" in payload:
                                log_to_file(f"DEBUG: Found {len(payload['threads'])} threads in payload")
                        
                        # Cleanup
                        await multi_client.shutdown()
                        log_to_file(f"DEBUG: MultiMCPClient shutdown complete")
                        
                        return payload
                        
                    except Exception as e:
                        log_to_file(f"ERROR: MCP call failed: {e}")
                        import traceback
                        log_to_file(f"DEBUG: Traceback: {traceback.format_exc()}")
                        return {"error": str(e)}
                
                # Run the async function
                return asyncio.run(_run())
        ''').strip()
    
    async def generate_prelude(self, agent_id: str, stage: str, run_dir: Path, context: Dict[str, Any] = None) -> str:
        """
        Generate agent-agnostic prelude for sandboxed code execution.
        
        Args:
            agent_id: Agent identifier (e.g., "scout", "screener")
            stage: Stage name (e.g., "collect", "think")
            run_dir: Run directory path
            context: Additional context for prelude generation
            
        Returns:
            Generated prelude code as string
        """
        cache_key = f"{agent_id}_{stage}_{str(run_dir)}"
        
        # Check cache first
        if cache_key in self._prelude_cache:
            self.logger.debug(f"Using cached prelude for {agent_id}_{stage}")
            return self._prelude_cache[cache_key]
        
        context = context or {}
        project_root = Path(__file__).resolve().parents[3]  # Go up to ScoutAgent root
        
        # Generate base prelude with imports and setup
        base_prelude = textwrap.dedent(f'''
            import json, os, asyncio
            import sys
            from pathlib import Path
            
            RUN_DIR = Path(r"{run_dir}")
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            PROJ_ROOT = Path(r"{project_root}")
            if str(PROJ_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJ_ROOT))
            
            from scout_agent.mcp_integration.client.multi import MultiMCPClient
            from scout_agent.mcp_integration.config import load_server_configs
        ''').strip()
        
        # Add helper functions
        helper_code_parts = []
        for helper_name, helper_info in self._helper_functions.items():
            helper_code_parts.append(f"\n# {helper_info['description']}")
            helper_code_parts.append(helper_info['code'])
        
        helper_code = "\n\n".join(helper_code_parts)
        
        # Combine base prelude with helpers
        full_prelude = f"{base_prelude}\n\n{helper_code}\n"
        
        # Cache the result
        self._prelude_cache[cache_key] = full_prelude
        
        self.logger.info(f"Generated prelude for {agent_id}_{stage} ({len(full_prelude)} chars)")
        return full_prelude
    
    async def register_helper_function(self, name: str, code: str, description: str = "") -> None:
        """
        Register a new helper function.
        
        Args:
            name: Function name
            code: Function implementation code
            description: Function description
        """
        self._helper_functions[name] = {
            "code": code.strip(),
            "description": description
        }
        
        # Clear cache to force regeneration
        self._prelude_cache.clear()
        
        self.logger.info(f"Registered helper function: {name}")
    
    async def get_cached_prelude(self, agent_id: str, stage: str, run_dir: Path) -> Optional[str]:
        """
        Get cached prelude if available.
        
        Args:
            agent_id: Agent identifier
            stage: Stage name
            run_dir: Run directory path
            
        Returns:
            Cached prelude or None if not found
        """
        cache_key = f"{agent_id}_{stage}_{str(run_dir)}"
        return self._prelude_cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear the prelude cache."""
        self._prelude_cache.clear()
        self.logger.info("Prelude cache cleared")
    
    def get_helper_functions(self) -> Dict[str, Dict[str, str]]:
        """Get all registered helper functions."""
        return self._helper_functions.copy()
