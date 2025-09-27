"""
ScreenerAgent - Pain Point Ranking Agent

This agent specializes in evaluating and ranking pain points discovered by the ScoutAgent
based on problem clarity, willingness to pay, automatability, source diversity, and recency/frequency.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from scout_agent.config import get_config
from scout_agent.llm.utils import LLMAgentMixin, load_prompt_template
from scout_agent.memory.manifest_manager import ManifestManager


@dataclass
class ScreenerInput:
    """Input for ScreenerAgent."""
    pain_points: List[Dict[str, Any]]  # Pain points from ScoutAgent
    target_market: str  # Market context for evaluation
    top_k: int = 5  # Number of top pain points to return
    
    @classmethod
    def from_agent_input(cls, agent_input: AgentInput):
        """Create ScreenerInput from standard AgentInput."""
        # Extract pain points from data field
        pain_points = agent_input.data
        
        # Extract context fields
        context = agent_input.context or {}
        target_market = context.get("target_market", "")
        top_k = context.get("top_k", 5)
        
        return cls(
            pain_points=pain_points,
            target_market=target_market,
            top_k=top_k
        )


@dataclass
class ScreenerOutput:
    """Output from ScreenerAgent."""
    ranked_pain_points: List[Dict[str, Any]]  # All pain points with rankings
    top_pain_points: List[Dict[str, Any]]  # Top K pain points
    ranking_justification: str  # Explanation of ranking methodology
    ranking_stats: Dict[str, Any]  # Statistics about the ranking
    confidence_score: float  # Confidence in the ranking (0.0-1.0)
    
    def to_agent_output(self) -> AgentOutput:
        """Convert to standard AgentOutput for compatibility."""
        return AgentOutput(
            result={
                "ranked_pain_points": self.ranked_pain_points,
                "top_pain_points": self.top_pain_points,
                "ranking_justification": self.ranking_justification,
                "ranking_stats": self.ranking_stats,
                "confidence_score": self.confidence_score
            },
            metadata={
                "top_k": len(self.top_pain_points),
                "total_evaluated": len(self.ranked_pain_points)
            },
            logs=[],
            execution_time=0.0,
            success=True
        )


class ScreenerAgent(BaseAgent, LLMAgentMixin):
    """
    ScreenerAgent for evaluating and ranking pain points.
    
    Uses criteria-based evaluation to rank pain points by their potential
    and select the most promising ones.
    """
    
    def __init__(self, agent_id: str = None):
        """Initialize the ScreenerAgent."""
        BaseAgent.__init__(self, name="screener", agent_id=agent_id)
        LLMAgentMixin.__init__(self, preferred_backend=None)
        self.name = "screener"
        self.config = get_config()
        
    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Execute the screening process."""
        self.start_time = time.time()
        
        try:
            # Convert to ScreenerInput
            input_data = ScreenerInput.from_agent_input(agent_input)
            
            # Skip plan phase as we only have think and act
            self._update_status('thinking')
            evaluations = await self.think(input_data)
            
            self._update_status('acting')
            result = await self.act(input_data, evaluations)
            
            execution_time = time.time() - self.start_time
            
            # Create output
            output = AgentOutput(
                result=result,
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'evaluations': evaluations
                },
                logs=self.execution_logs,
                execution_time=execution_time,
                success=True
            )
            
            self._update_status('completed')
            return output
            
        except Exception as e:
            self.logger.error(f"Error in ScreenerAgent: {str(e)}")
            execution_time = time.time() - self.start_time
            return AgentOutput(
                result=None,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                logs=self.execution_logs,
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def plan(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Plan phase is not used in ScreenerAgent."""
        return {"status": "skipped", "reason": "ScreenerAgent uses only think and act phases"}
    
    async def think(self, input_data: ScreenerInput) -> Dict[str, Any]:
        """Evaluate pain points against criteria."""
        self.logger.info(f"Evaluating {len(input_data.pain_points)} pain points")
        
        # Check if we have any pain points to evaluate
        if not input_data.pain_points:
            self.logger.warning("No pain points provided for evaluation")
            fallback_evaluations = self._generate_fallback_evaluations([])
            self._write_stage_output("think", fallback_evaluations)
            self.state.evaluations = fallback_evaluations
            return fallback_evaluations
        
        # Save input pain points for debugging
        try:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"screener_input_pain_points_{int(time.time())}.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(input_data.pain_points, f, indent=2)
                self.logger.info(f"Saved input pain points to {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to save input pain points: {e}")
        
        try:
            # Load prompt template
            prompt = load_prompt_template(
                template_name="think.prompt",
                agent_name="screener_agent",  # Use explicit folder name
                substitutions={
                    "pain_points": json.dumps(input_data.pain_points, indent=2),
                    "market_focus": input_data.target_market,
                    "top_k": input_data.top_k
                }
            )
            
            # Debug the prompt path
            self.logger.info(f"Loading prompt from screener_agent/think.prompt")
            
            # Save the prompt for debugging
            try:
                debug_file = debug_dir / f"screener_think_prompt_{int(time.time())}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
                    self.logger.info(f"Saved prompt to {debug_file}")
            except Exception as e:
                self.logger.error(f"Failed to save prompt: {e}")
            
            # Generate evaluations using LLM
            self.logger.info("Calling LLM to generate evaluations...")
            response = await self.llm_generate(prompt=prompt, task_type="evaluation")
            self.logger.info(f"Received LLM response of length: {len(response) if response else 0}")
            
            # Extract JSON from response
            evaluations = self._extract_json(response)
            
            # Validate the evaluations structure
            if not evaluations.get('evaluations'):
                self.logger.warning("No evaluations found in LLM response, using fallback")
                evaluations = self._generate_fallback_evaluations(input_data.pain_points)
            else:
                self.logger.info(f"Generated evaluations for {len(evaluations.get('evaluations', []))} pain points")
            
            # Write evaluations to manifest
            self._write_stage_output("think", evaluations)
            
            # Store in agent state for act phase
            self.state.evaluations = evaluations
            
            return evaluations
            
        except Exception as e:
            self.logger.error(f"Error in think phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback: generate basic evaluations
            self.logger.info("Using fallback evaluation due to error")
            fallback_evaluations = self._generate_fallback_evaluations(input_data.pain_points)
            
            # Write fallback evaluations to manifest
            self._write_stage_output("think", fallback_evaluations)
            
            # Store in agent state for act phase
            self.state.evaluations = fallback_evaluations
            
            return fallback_evaluations
    
    async def act(self, input_data: ScreenerInput, evaluations: Dict[str, Any] = None) -> Dict[str, Any]:
        """Rank pain points and select top K based on evaluations from think stage."""
        self.logger.info("Ranking pain points and selecting top candidates")
        
        try:
            # Use evaluations from parameter or from agent state
            if evaluations is None:
                evaluations = getattr(self.state, "evaluations", None)
                
            # If still None, try to get from manifest
            if evaluations is None:
                self.logger.info("No evaluations provided, attempting to load from manifest")
                evaluations = self._get_think_output()
                
            if evaluations is None:
                raise ValueError("No evaluations available for ranking")
            
            # Extract the evaluations list
            eval_list = evaluations.get("evaluations", [])
            if not eval_list:
                raise ValueError("No evaluations found in evaluation data")
                
            # Sort by total score (descending)
            ranked_pain_points = sorted(eval_list, key=lambda x: x.get("total_score", 0), reverse=True)
            
            # Add rank field
            for i, pain_point in enumerate(ranked_pain_points):
                pain_point["rank"] = i + 1
            
            # Select top K
            # Derive top_k robustly from input_data or its context
            derived_top_k = getattr(input_data, 'top_k', None)
            if derived_top_k is None:
                context = getattr(input_data, 'context', None)
                if isinstance(context, dict):
                    derived_top_k = context.get('top_k')
            if derived_top_k is None:
                derived_top_k = 10
            
            # ULTRA-AGGRESSIVE: Override for ultra-fast HTML testing  
            max_top_k_for_testing = 1  # ULTRA-AGGRESSIVE LIMITING: Only 1 pain point for fastest testing
            if derived_top_k > max_top_k_for_testing:
                self.logger.info(f"COST LIMITING: Reducing top_k from {derived_top_k} to {max_top_k_for_testing} for testing")
                derived_top_k = max_top_k_for_testing
            top_k = min(int(derived_top_k), len(ranked_pain_points))
            top_pain_points = ranked_pain_points[:top_k]
            
            # Create simplified versions of top pain points for the output
            simplified_top_points = []
            for point in top_pain_points:
                simplified_top_points.append({
                    "id": point.get("id", ""),
                    "description": point.get("description", ""),
                    "total_score": point.get("total_score", 0),
                    "rank": point.get("rank", 0)
                })
            
            # Calculate statistics
            scores = [p.get("total_score", 0) for p in ranked_pain_points]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Score distribution
            high_threshold = avg_score * 1.2
            low_threshold = avg_score * 0.8
            high_count = sum(1 for s in scores if s >= high_threshold)
            low_count = sum(1 for s in scores if s <= low_threshold)
            medium_count = len(scores) - high_count - low_count
            
            # Create the ranking result
            ranking_result = {
                "ranked_pain_points": ranked_pain_points,
                "top_pain_points": simplified_top_points,
                "ranking_justification": f"Selected the top {top_k} pain points based on their total evaluation scores across all criteria.",
                "ranking_stats": {
                    "total_evaluated": len(ranked_pain_points),
                    "highest_score": max(scores) if scores else 0,
                    "lowest_score": min(scores) if scores else 0,
                    "average_score": round(avg_score, 1),
                    "score_distribution": {
                        "high_potential": high_count,
                        "medium_potential": medium_count,
                        "low_potential": low_count
                    }
                },
                "confidence_score": 0.9  # High confidence since this is deterministic
            }
            
            self.logger.info(f"Generated ranking with {len(ranking_result['top_pain_points'])} top pain points")
            
            # Write ranking to manifest
            self._write_stage_output("act", ranking_result)
            
            return ranking_result
            
        except Exception as e:
            self.logger.error(f"Error in act phase: {str(e)}\n{traceback.format_exc()}")
            # Fallback: generate basic ranking
            # Derive top_k for fallback similarly to main path
            derived_top_k = getattr(input_data, 'top_k', None)
            if derived_top_k is None:
                context = getattr(input_data, 'context', None)
                if isinstance(context, dict):
                    derived_top_k = context.get('top_k')
            if derived_top_k is None:
                derived_top_k = 10
            fallback_ranking = self._generate_fallback_ranking(evaluations, int(derived_top_k))
            
            # Write fallback ranking to manifest
            self._write_stage_output("act", fallback_ranking)
            
            return fallback_ranking
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text with robust error handling."""
        import re
        import json
        import os
        from pathlib import Path
        
        # Default fallback result
        default_result = {
            "status": "fallback",
            "message": "Failed to parse JSON from response"
        }
        
        # Log the raw response for debugging
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        debug_file = debug_dir / f"screener_think_response_{int(time.time())}.txt"
        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write("=== RAW LLM RESPONSE ===\n")
                f.write(text or "")
                self.logger.info(f"Saved raw LLM response to {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to save debug file: {e}")
        
        if not text or not isinstance(text, str):
            self.logger.error("Empty or non-string response from LLM")
            return default_result
        
        # Try to find JSON in markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(.+?)\s*```'
        matches = re.search(code_block_pattern, text, re.DOTALL)
        
        if matches:
            self.logger.info("Found JSON in code block")
            json_str = matches.group(1)
        else:
            # Try to find JSON without code blocks
            self.logger.info("No code block found, trying to parse entire response as JSON")
            json_str = text
            
            # Try to find JSON object within the text using regex
            json_pattern = r'\{[\s\S]*\}'
            json_matches = re.search(json_pattern, text, re.DOTALL)
            if json_matches:
                self.logger.info("Found JSON-like structure in response")
                json_str = json_matches.group(0)
        
        # Clean the JSON string
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)  # Remove trailing commas
        
        # Log the extracted JSON string
        try:
            with open(debug_file, "a", encoding="utf-8") as f:
                f.write("\n\n=== EXTRACTED JSON STRING ===\n")
                f.write(json_str or "")
        except Exception:
            pass
        
        try:
            parsed_json = json.loads(json_str)
            self.logger.info("Successfully parsed JSON")
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            try:
                # Try again with more aggressive cleaning
                self.logger.info("Attempting more aggressive JSON cleaning")
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Add quotes to keys
                
                # Log the cleaned JSON string
                try:
                    with open(debug_file, "a", encoding="utf-8") as f:
                        f.write("\n\n=== CLEANED JSON STRING ===\n")
                        f.write(json_str or "")
                except Exception:
                    pass
                
                parsed_json = json.loads(json_str)
                self.logger.info("Successfully parsed JSON after aggressive cleaning")
                return parsed_json
            except Exception as e:
                self.logger.error(f"Failed to parse JSON after cleaning: {e}")
                return default_result
    
    def _generate_fallback_evaluations(self, pain_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback evaluations if LLM fails."""
        evaluations = []
        
        # Handle empty pain_points list
        if not pain_points:
            self.logger.warning("No pain points provided for fallback evaluation")
            return {
                "evaluations": [],
                "evaluation_summary": "No pain points available for evaluation"
            }
        
        for i, pain_point in enumerate(pain_points):
            # Extract pain point fields
            pain_id = pain_point.get("id", f"pp{i+1}")
            description = pain_point.get("description", "Unknown pain point")
            
            # Handle missing description
            if not description or description == "Unknown pain point":
                # Try to extract description from other fields
                if "title" in pain_point:
                    description = pain_point["title"]
                elif "summary" in pain_point:
                    description = pain_point["summary"]
                elif "text" in pain_point:
                    description = pain_point["text"][:100] + "..."  # Truncate long text
            
            # Generate random-ish scores based on available data
            problem_clarity = min(len(description) / 20, 10) if description else 5.0  # Longer descriptions get higher clarity scores
            willingness_to_pay = pain_point.get("impact_score", 5) * 0.8  # Base WTP on impact score
            automatability = 7.0  # Default middle-high score for automatability
            source_diversity = min(len(pain_point.get("evidence", [])), 10)  # Score based on evidence count
            recency_frequency = min(pain_point.get("frequency", 5), 10)  # Score based on frequency
            
            # Calculate total score
            total_score = problem_clarity + willingness_to_pay + automatability + source_diversity + recency_frequency
            
            evaluations.append({
                "id": pain_id,
                "description": description or "No description available",
                "scores": {
                    "problem_clarity": round(problem_clarity, 1),
                    "willingness_to_pay": round(willingness_to_pay, 1),
                    "automatability": round(automatability, 1),
                    "source_diversity": round(source_diversity, 1),
                    "recency_frequency": round(recency_frequency, 1)
                },
                "total_score": round(total_score, 1),
                "justification": "Fallback evaluation based on available data"
            })
        
        # Save fallback evaluations for debugging
        try:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"screener_fallback_evaluations_{int(time.time())}.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump({
                    "evaluations": evaluations,
                    "evaluation_summary": f"Fallback evaluation of {len(evaluations)} pain points"
                }, f, indent=2)
                self.logger.info(f"Saved fallback evaluations to {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to save fallback evaluations: {e}")
        
        return {
            "evaluations": evaluations,
            "evaluation_summary": f"Fallback evaluation of {len(evaluations)} pain points"
        }
    
    def _generate_fallback_ranking(self, evaluations: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Generate fallback ranking if the main ranking process fails."""
        # Extract evaluations
        eval_list = evaluations.get("evaluations", [])
        if not eval_list:
            # Create dummy evaluations if none exist
            eval_list = [
                {
                    "id": f"dummy_{i}",
                    "description": f"Fallback pain point {i+1}",
                    "total_score": 50 - i,  # Descending scores
                    "justification": "Automatically generated fallback"
                }
                for i in range(min(10, top_k * 2))  # Create enough for top_k
            ]
        
        # Sort by total score (descending)
        ranked_pain_points = sorted(eval_list, key=lambda x: x.get("total_score", 0), reverse=True)
        
        # Add rank field
        for i, pain_point in enumerate(ranked_pain_points):
            pain_point["rank"] = i + 1
        
        # Select top K
        top_k = min(top_k, len(ranked_pain_points))
        top_pain_points = ranked_pain_points[:top_k]
        
        # Create simplified versions of top pain points for the output
        simplified_top_points = []
        for point in top_pain_points:
            simplified_top_points.append({
                "id": point.get("id", ""),
                "description": point.get("description", ""),
                "total_score": point.get("total_score", 0),
                "rank": point.get("rank", 0)
            })
        
        # Calculate statistics
        scores = [p.get("total_score", 0) for p in ranked_pain_points]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Score distribution
        high_threshold = avg_score * 1.2
        low_threshold = avg_score * 0.8
        high_count = sum(1 for s in scores if s >= high_threshold)
        low_count = sum(1 for s in scores if s <= low_threshold)
        medium_count = len(scores) - high_count - low_count
        
        return {
            "ranked_pain_points": ranked_pain_points,
            "top_pain_points": simplified_top_points,
            "ranking_justification": f"Fallback ranking based on total evaluation scores. Selected the top {top_k} pain points.",
            "ranking_stats": {
                "total_evaluated": len(ranked_pain_points),
                "highest_score": max(scores) if scores else 0,
                "lowest_score": min(scores) if scores else 0,
                "average_score": round(avg_score, 1),
                "score_distribution": {
                    "high_potential": high_count,
                    "medium_potential": medium_count,
                    "low_potential": low_count
                }
            },
            "confidence_score": 0.5  # Medium confidence for fallback
        }
    
    def _write_stage_output(self, stage_name: str, data: Dict[str, Any]) -> None:
        """Write stage output to manifest section using ManifestManager."""
        try:
            # Get run_id from state, if not available check for run_dir attribute
            run_id = getattr(self.state, "run_id", None)
            
            # If we have a run_dir attribute, use that directly
            run_dir = getattr(self.state, "run_dir", None)
            manifest_manager = getattr(self.state, "manifest_manager", None)
            
            if not run_id and not run_dir and not manifest_manager:
                self.logger.warning("No run_id, run_dir, or manifest_manager available. Using fallback.")
                run_id = "unknown"
            
            if not manifest_manager:
                # Determine run directory if we don't have a manifest_manager
                # project_root should be ScoutAgent/ (not scout_agent/)
                project_root = Path(__file__).resolve().parents[2]
                
                if not run_dir:
                    run_dir = project_root / "data" / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                
                manifest_path = run_dir / "run_manifest.json"
                self.logger.info(f"Creating manifest manager for path: {manifest_path}")
                manifest_manager = ManifestManager(manifest_path, create_if_missing=True)
            
            # Always use agent-prefixed stage names for multi-agent support
            agent_prefixed_stage = f"screener_{stage_name}"
            self.logger.info(f"Writing stage {agent_prefixed_stage} output to manifest")
            
            # Store the stage output with agent-prefixed stage name
            manifest_manager.store_node_output(agent_prefixed_stage, data)
            
            # Update node status to completed with agent-prefixed stage name
            manifest_manager.update_node_status(
                node_id=agent_prefixed_stage,
                state="completed"
            )
            
            self.logger.info(f"Stage {agent_prefixed_stage} output written to manifest successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to write stage {stage_name} output: {e}")
            self.logger.error(traceback.format_exc())
    
    def _get_think_output(self) -> Optional[Dict[str, Any]]:
        """Get the output from the think stage from manifest."""
        try:
            # Get manifest_manager from state if available
            manifest_manager = getattr(self.state, "manifest_manager", None)
            
            if not manifest_manager:
                # Try to get run_id or run_dir from state
                run_id = getattr(self.state, "run_id", None)
                run_dir = getattr(self.state, "run_dir", None)
                
                if not run_id and not run_dir:
                    self.logger.warning("No run_id or run_dir available for getting think output")
                    return None
                
                # Determine run directory
                project_root = Path(__file__).resolve().parents[2]
                if not run_dir:
                    run_dir = project_root / "data" / "runs" / run_id
                
                manifest_path = run_dir / "run_manifest.json"
                
                if not manifest_path.exists():
                    self.logger.warning(f"Manifest not found at: {manifest_path}")
                    return None
                
                # Create ManifestManager
                manifest_manager = ManifestManager(manifest_path)
            
            # Try to get data from screener_think node
            think_node_id = "screener_think"
            think_data = manifest_manager.get_node_output(think_node_id)
            
            if think_data:
                self.logger.info(f"Found think stage data in manifest node: {think_node_id}")
                return think_data
            
            # Try to get data from stages section
            manifest = manifest_manager.get_manifest()
            if "stages" in manifest and think_node_id in manifest["stages"]:
                stage_data = manifest["stages"][think_node_id]
                if "data" in stage_data:
                    self.logger.info(f"Found think stage data in manifest stages.{think_node_id}.data")
                    return stage_data["data"]
            
            self.logger.warning(f"No think stage data found in manifest")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting think output: {e}")
            return None


# Register the agent
from .base import register_agent
register_agent(ScreenerAgent)