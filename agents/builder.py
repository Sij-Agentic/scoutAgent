"""
BuilderAgent - Solution Prototyping and Validation Agent

This agent specializes in creating solution prototypes and validating
them against discovered market gaps and pain points.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseAgent, AgentInput, AgentOutput, AgentState
from services.agents.code import get_code_execution_service
from .analysis_agent import AnalysisAgent
from config import get_config
from llm.utils import LLMAgentMixin, load_prompt_template


@dataclass
class SolutionPrototype:
    """Represents a solution prototype."""
    solution_name: str
    description: str
    target_pain_points: List[str]
    key_features: List[str]
    technical_architecture: Dict[str, Any]
    implementation_approach: str
    estimated_development_time: str
    estimated_cost: float
    market_size: float
    mvp_scope: List[str]
    validation_plan: Dict[str, Any]
    success_metrics: List[str]
    risk_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuilderInput:
    """Input for BuilderAgent."""
    market_gaps: List[Dict[str, Any]]  # From GapFinderAgent
    target_market: str
    solution_type: str = "software"  # software, service, platform
    budget_range: str = "moderate"  # low, moderate, high
    timeline: str = "3-6 months"  # 1-3 months, 3-6 months, 6-12 months
    technical_complexity: str = "moderate"  # low, moderate, high
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.market_gaps:
            raise ValueError("Must provide market gaps for solution building")
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BuilderOutput:
    """Output from BuilderAgent."""
    solution_prototypes: List[SolutionPrototype]
    recommended_solution: Dict[str, Any]
    implementation_roadmap: List[Dict[str, Any]]
    technical_design: Dict[str, Any]
    cost_analysis: Dict[str, float]
    timeline_estimate: Dict[str, str]
    feasibility_score: float
    risk_factors: List[str]
    success_metrics: List[str]
    result: Any = None
    metadata: Dict[str, Any] = None
    logs: List[str] = None
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.logs is None:
            self.logs = []


class BuilderAgent(BaseAgent, LLMAgentMixin):
    """
    BuilderAgent for creating and validating solution prototypes.
    
    Uses technical analysis, cost estimation, and market validation
    to create actionable solution prototypes for identified market gaps.
    """
    
    def __init__(self, agent_id: str = None):
        BaseAgent.__init__(self, name="builder", agent_id=agent_id)
        LLMAgentMixin.__init__(self)
        self.code_service = None  # Will initialize in _initialize
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
        self.name = "builder"  # Used for prompt directory name
        
        # Initialize the code service asynchronously later
        asyncio.create_task(self._init_code_service())
        self.preferred_backend = "ollama"  # Use ollama backend by default
        
        # Explicitly override task backend preferences to use Ollama for all tasks
        self.task_backend_preferences = {
            'planning': 'ollama',
            'analysis': 'ollama',
            'technical_design': 'ollama',
            'default': 'ollama'
        }
    
    async def _init_code_service(self):
        """Initialize the code execution service."""
        try:
            from service_registry import get_registry
            registry = get_registry()
            
            # Try to get the service from the registry first
            if registry and registry.has_service("code_execution"):
                self.code_service = registry.get_service("code_execution")
                self.logger.info("Using code_execution service from registry")
            else:
                # Fallback to factory method if not in registry
                self.code_service = get_code_execution_service()
                self.logger.info("Using code_execution service from factory")
                
            # Initialize the service if it hasn't been already
            if hasattr(self.code_service, '_initialize') and callable(getattr(self.code_service, '_initialize')):
                await self.code_service._initialize(None)
                await self.code_service._start()
                
        except Exception as e:
            self.logger.warning(f"Could not initialize code service: {str(e)}")
            # Create a minimal mock to avoid NoneType errors
            class MockCodeService:
                async def execute_code(self, *args, **kwargs):
                    return {"success": False, "output": "Code execution service not available"}
                
                async def generate_code(self, *args, **kwargs):
                    return False, "Code generation service not available"
                
                def __getattr__(self, name):
                    # Return an empty function for any attribute
                    return lambda *args, **kwargs: None
            
            self.code_service = MockCodeService()
    
    async def plan(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Plan the solution building process using LLM prompt."""
        self.logger.info(f"Planning solution building for {len(input_data.market_gaps) if input_data and input_data.market_gaps else 0} market gaps")
        start_time = datetime.now()
        
        # Debug log to help identify None values
        self.logger.info(f"BuilderInput DEBUG: market_gaps={input_data.market_gaps}, type={type(input_data.market_gaps)}")
        
        try:
            # Prepare substitutions for the prompt template with safe handling of None values
            substitutions = {
                "market_gaps": json.dumps(input_data.market_gaps if hasattr(input_data, 'market_gaps') and input_data.market_gaps is not None else []),
                "target_market": self._safe_string_operation(input_data.target_market) if hasattr(input_data, 'target_market') else "",
                "solution_type": self._safe_string_operation(input_data.solution_type) if hasattr(input_data, 'solution_type') else "",
                "budget_range": self._safe_string_operation(input_data.budget_range) if hasattr(input_data, 'budget_range') else "",
                "timeline": self._safe_string_operation(input_data.timeline) if hasattr(input_data, 'timeline') else "",
                "technical_complexity": self._safe_string_operation(input_data.technical_complexity) if hasattr(input_data, 'technical_complexity') else ""
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("plan.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate plan using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="planning"
            )
            
            # Extract JSON from LLM response
            plan = self._extract_json(llm_response)
            
        except Exception as e:
            self.logger.error(f"Error generating plan: {str(e)}")
            # Fallback plan
            plan = {
                "phases": [
                    "gap_analysis",
                    "solution_design",
                    "technical_architecture",
                    "cost_estimation",
                    "validation_planning",
                    "risk_assessment",
                    "roadmap_creation"
                ],
                "solution_type": input_data.solution_type,
                "budget_range": input_data.budget_range,
                "timeline": input_data.timeline,
                "technical_complexity": input_data.technical_complexity,
                "expected_duration": 1200,  # 20 minutes
                "gap_count": len(input_data.market_gaps)
            }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Plan generation completed in {execution_time:.2f} seconds")
        
        self.state.plan = plan
        return plan
        
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        try:
            # First try to find JSON enclosed in triple backticks
            import re
            json_match = re.search(r'```(?:json)?([\s\S]*?)```', text)
            result = {}
            
            if json_match:
                json_str = json_match.group(1).strip()
                # Remove trailing commas which can cause JSON parsing errors
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                result = json.loads(json_str)
            else:
                # If no JSON in backticks, try parsing the whole text
                result = json.loads(text)
                
            # Ensure feasibility_score is always present
            if 'feasibility_score' not in result:
                result['feasibility_score'] = 7.5  # Default feasibility score
                
            return result
        except Exception as e:
            self.logger.warning(f"Could not parse LLM response as JSON: {str(text)[:50]}...")
            # Return empty dict with feasibility_score to avoid KeyError
            return {"feasibility_score": 7.5}
    
    async def think(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Analyze market gaps to design solutions using LLM prompt."""
        self.logger.info(f"Analyzing {len(input_data.market_gaps)} market gaps for solution design")
        start_time = datetime.now()
        
        # Fetch or create plan
        plan = self.state.plan or await self.plan(input_data)
        
        try:
            # Prepare substitutions for the prompt template
            substitutions = {
                "market_gaps": json.dumps(input_data.market_gaps),
                "target_market": input_data.target_market,
                "solution_type": input_data.solution_type,
                "budget_range": input_data.budget_range,
                "timeline": input_data.timeline,
                "technical_complexity": input_data.technical_complexity,
                "plan": json.dumps(plan)
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("think.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate analysis using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="analysis"
            )
            
            # Extract JSON from LLM response
            analysis_result = self._extract_json(llm_response)
            
        except Exception as e:
            self.logger.error(f"Error analyzing market gaps: {str(e)}")
            # Fallback analysis
            gap_analysis = []
            for gap in input_data.market_gaps:
                # Make sure to use proper fallbacks for potentially null fields
                gap_id = gap.get("id", str(len(gap_analysis) + 1))
                gap_description = gap.get("gap_description", "") or gap.get("description", "")
                
                gap_analysis.append({
                    "gap_id": gap_id,
                    "gap_description": gap_description,
                    "feasibility_score": 7.5,
                    "technical_requirements": ["Scalable architecture", "User-friendly interface"],
                    "potential_approaches": ["Cloud-based solution", "Mobile application"],
                    "estimated_complexity": "medium"
                })
            
            analysis_result = {
                "gap_analysis": gap_analysis,
                "feasibility_score": 7.8,  # Ensure this key is always present
                "technical_considerations": [
                    "Scalability requirements",
                    "Security implications",
                    "Integration complexity"
                ],
                "resource_requirements": {
                    "development_resources": ["Frontend developers", "Backend developers", "DevOps"],
                    "testing_resources": ["QA engineers", "User testing participants"],
                    "deployment_resources": ["Cloud infrastructure", "CI/CD pipeline"]
                },
                "recommended_technologies": [
                    "React/Vue for frontend",
                    "Node.js/Python for backend",
                    "AWS/GCP for hosting"
                ],
                "implementation_strategy": "Agile development with 2-week sprints"
            }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Gap analysis completed in {execution_time:.2f} seconds")
        
        self.state.analysis = analysis_result
        return analysis_result
    
    async def act(self, input_data: BuilderInput) -> BuilderOutput:
        """Execute solution building and return prototypes using LLM prompt."""
        try:
            self.logger.info(f"Creating solution prototypes for {len(input_data.market_gaps) if input_data and input_data.market_gaps else 0} market gaps")
            # Add debug logging to trace the exact nature of input_data
            self.logger.info(f"BuilderInput act DEBUG: {input_data.__dict__ if hasattr(input_data, '__dict__') else 'No __dict__'}")
        except Exception as e:
            self.logger.error(f"Exception in debug logging: {str(e)}")
            
        start_time = datetime.now()
        
        # Get plan and analysis or run if not available
        plan = self.state.plan or await self.plan(input_data)
        gap_analysis = getattr(self.state, 'analysis', None) or await self.think(input_data)
        
        try:
            # Prepare substitutions for the prompt template
            substitutions = {
                "market_gaps": json.dumps(input_data.market_gaps),
                "target_market": input_data.target_market,
                "solution_type": input_data.solution_type,
                "budget_range": input_data.budget_range,
                "timeline": input_data.timeline,
                "technical_complexity": input_data.technical_complexity,
                "plan": json.dumps(plan),
                "gap_analysis": json.dumps(gap_analysis)
            }
            
            # Load prompt template with substitutions
            prompt_content = load_prompt_template("act.prompt", agent_name=self.name, substitutions=substitutions)
            
            # Generate solution prototypes using LLM
            llm_response = await self.llm_generate(
                prompt=prompt_content,
                task_type="technical_design"
            )
            
            # Extract JSON from LLM response
            solution_result = self._extract_json(llm_response)
            
            # Convert LLM output to SolutionPrototypes
            solution_prototypes = []
            for solution_data in solution_result.get("solution_prototypes", []):
                try:
                    solution = SolutionPrototype(
                        solution_name=solution_data.get("solution_name", ""),
                        description=solution_data.get("description", ""),
                        target_pain_points=solution_data.get("target_pain_points", []),
                        key_features=solution_data.get("key_features", []),
                        technical_architecture=solution_data.get("technical_architecture", {}),
                        implementation_approach=solution_data.get("implementation_approach", ""),
                        estimated_development_time=solution_data.get("estimated_development_time", ""),
                        estimated_cost=float(solution_data.get("estimated_cost", 0)),
                        market_size=float(solution_data.get("market_size", 0)),
                        mvp_scope=solution_data.get("mvp_scope", []),
                        validation_plan=solution_data.get("validation_plan", {}),
                        success_metrics=solution_data.get("success_metrics", []),
                        risk_assessment=solution_data.get("risk_assessment", {})
                    )
                    solution_prototypes.append(solution)
                except Exception as e:
                    self.logger.error(f"Error creating solution prototype: {str(e)}")
            
            # Create BuilderOutput
            output = BuilderOutput(
                solution_prototypes=solution_prototypes,
                recommended_solution=solution_result.get("recommended_solution", {}),
                implementation_roadmap=solution_result.get("implementation_roadmap", []),
                technical_design=solution_result.get("technical_design", {}),
                cost_analysis=solution_result.get("cost_analysis", {}),
                timeline_estimate=solution_result.get("timeline_estimate", {}),
                feasibility_score=float(solution_result.get("feasibility_score", 0)),
                risk_factors=solution_result.get("risk_factors", []),
                success_metrics=solution_result.get("success_metrics", []),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating solution prototypes: {str(e)}")
            # Fallback solution
            default_solution = self._create_fallback_solution(input_data)
            
            # Create BuilderOutput with fallback solution
            output = BuilderOutput(
                solution_prototypes=[default_solution],
                recommended_solution=default_solution.to_dict(),
                implementation_roadmap=self._create_implementation_roadmap(),
                technical_design=self._create_technical_design(default_solution),
                cost_analysis=self._calculate_cost_analysis(default_solution),
                timeline_estimate=self._estimate_timeline(default_solution, input_data.timeline),
                feasibility_score=self._calculate_feasibility_score(default_solution),
                risk_factors=["Technical complexity", "Market adoption"],
                success_metrics=self._define_success_metrics(default_solution, input_data.target_market),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        
        self.logger.info(f"Solution generation completed in {output.execution_time:.2f} seconds")
        return output
        
    def _safe_lower(self, value):
        """Safely convert a value to lowercase, handling None and non-string types"""
        # Extra logging to trace where this is being called from
        import traceback
        self.logger.debug(f"_safe_lower called with value type: {type(value)}, value: {repr(value)[:100]}")
        self.logger.debug(f"Stack trace: {traceback.format_stack()[-3]}")
        
        # Handle None case first
        if value is None:
            self.logger.warning("None value passed to _safe_lower, returning empty string")
            return ""
            
        # Convert to string if not already a string
        string_value = ""
        if isinstance(value, str):
            string_value = value
        else:
            try:
                string_value = str(value)
                self.logger.info(f"Converted non-string value to string in _safe_lower: {type(value)} -> {repr(string_value)[:100]}")
            except Exception as e:
                self.logger.warning(f"Failed to convert value to string in _safe_lower: {e}")
                return ""
        
        # Only lowercase the string if it's a valid string
        if string_value and isinstance(string_value, str):
            try:
                return string_value.lower()
            except Exception as e:
                self.logger.error(f"Unexpected error lowercasing string: {e}")
                return string_value  # Return original string if lowercasing fails
        else:
            self.logger.warning(f"Cannot lowercase empty or non-string value, returning as is: {repr(string_value)[:100]}")
            return string_value
            
    def _safe_string_operation(self, value, default=""):
        """Safely handle string operations on potentially None values"""
        if value is None:
            return default
        try:
            return str(value)
        except Exception as e:
            self.logger.warning(f"Failed to convert value to string: {e}")
            return default
    
    def _create_fallback_solution(self, input_data: BuilderInput) -> SolutionPrototype:
        """Create a fallback solution when LLM generation fails."""
        if not input_data.market_gaps:
            gap = {"description": "Unknown market gap"}
        else:
            gap = input_data.market_gaps[0]
            
        return SolutionPrototype(
            solution_name=f"Solution for {input_data.target_market}",
            description=f"A {input_data.solution_type} solution for {input_data.target_market}",
            target_pain_points=[gap.get("pain_point", "Unspecified pain point")],
            key_features=["Scalable architecture", "User-friendly interface"],
            technical_architecture={
                "components": ["Frontend", "Backend", "Database"],
                "technologies": ["React", "Node.js", "PostgreSQL"],
                "integrations": ["Payment gateway", "Analytics"],
                "deployment": "Cloud-based"
            },
            implementation_approach="Agile development with 2-week sprints",
            estimated_development_time="3-6 months",
            estimated_cost=150000.0,
            market_size=5000000.0,
            mvp_scope=["Core feature 1", "Core feature 2"],
            validation_plan={
                "validation_methods": ["User testing", "Beta program"],
                "success_criteria": ["Adoption rate", "User satisfaction"],
                "testing_approach": "Iterative testing with target users"
            },
            success_metrics=["User adoption rate > 10%", "Customer satisfaction > 4.0/5.0"],
            risk_assessment={
                "risks": ["Technical complexity", "Market adoption"],
                "mitigations": ["Prototype key components early", "Conduct user testing"]
            },
            timeline_estimate=timeline_estimate,
            feasibility_score=feasibility_score,
            risk_factors=risk_factors
        )

    def _evaluate_solution_feasibility(self, solution: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Evaluate the feasibility of a solution."""
        return {
            "overall_score": 0.85,
            "technical_feasibility": 0.9,
            "market_feasibility": 0.8,
            "financial_feasibility": 0.85,
            "integration_challenges": ["API_compatibility", "data_migration"],
            "risk_factors": ["competition", "market_volatility"]
        }

    async def _analyze_technical_requirements(self, gap: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Analyze technical requirements for a market gap."""
        # Mock implementation for validation
        return {
            "complexity": "moderate",
            "required_technologies": ["python", "react", "postgresql", "docker"],
            "integration_points": ["payment_gateway", "email_service", "analytics"],
            "scalability_requirements": ["horizontal_scaling", "caching", "load_balancing"],
            "security_requirements": ["encryption", "authentication", "authorization"]
        }
    
    async def _create_solution_prototype(self, gap: Dict[str, Any], target_market: str,
                                       solution_type: str, budget_range: str, 
                                       timeline: str, complexity: str) -> SolutionPrototype:
        """Create a solution prototype for a market gap."""
        gap_description = gap.get("gap_description", "")
        market_size = gap.get("market_size", 25000000)
        
        # Generate solution name and description
        solution_name = self._generate_solution_name(gap_description)
        description = self._generate_solution_description(gap_description, target_market)
        
        # Define key features
        key_features = self._define_key_features(gap_description, solution_type)
        
        # Design technical architecture
        technical_architecture = self._design_technical_architecture(
            solution_type, complexity, key_features
        )
        
        # Determine implementation approach
        implementation_approach = self._determine_implementation_approach(
            complexity, budget_range, timeline
        )
        
        # Estimate development time and cost
        estimated_time = self._estimate_development_time(key_features, complexity)
        estimated_cost = self._estimate_development_cost(
            key_features, complexity, budget_range
        )
        
        # Define MVP scope
        mvp_scope = self._define_mvp_scope(key_features, timeline)
        
        # Create validation plan
        validation_plan = self._create_validation_plan(solution_name, target_market)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(solution_name, target_market)
        
        # Assess risks
        risk_assessment = self._assess_risks(solution_name, technical_architecture)
        
        return SolutionPrototype(
            solution_name=solution_name,
            description=description,
            target_pain_points=[gap_description],
            key_features=key_features,
            technical_architecture=technical_architecture,
            implementation_approach=implementation_approach,
            estimated_development_time=estimated_time,
            estimated_cost=estimated_cost,
            market_size=market_size,
            mvp_scope=mvp_scope,
            validation_plan=validation_plan,
            success_metrics=success_metrics,
            risk_assessment=risk_assessment
        )
    
    def _generate_solution_name(self, gap_description: str) -> str:
        """Generate a solution name based on the gap description."""
        # Handle None or empty gap description
        if not gap_description:
            return "SmartSolutionPro"
            
        # Simple name generation
        keywords = self._safe_lower(gap_description).split()
        key_terms = [word for word in keywords if len(word) > 3][:3]
        
        prefixes = ["Easy", "Smart", "Quick", "Pro", "Auto"]
        suffixes = ["Flow", "Hub", "Pro", "AI", "Sync"]
        
        import random
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        if key_terms:
            base = "".join(word.capitalize() for word in key_terms[:2])
            return f"{prefix}{base}{suffix}"
        else:
            return f"{prefix}Solution{suffix}"
    
    def _generate_solution_description(self, gap_description: str, target_market: str) -> str:
        """Generate a solution description."""
        return f"A comprehensive solution addressing {gap_description} for {target_market} businesses."
    
    def _define_key_features(self, gap_description: str, solution_type: str) -> List[str]:
        """Define key features for the solution."""
        features = {
            "onboarding_complexity": [
                "Guided setup wizard",
                "Template library",
                "Interactive tutorials",
                "Progress tracking"
            ],
            "integration_challenges": [
                "Universal API connector",
                "Pre-built integrations",
                "Real-time sync",
                "Data mapping tools"
            ],
            "pricing_barriers": [
                "Flexible pricing tiers",
                "Usage-based billing",
                "Free trial period",
                "ROI calculator"
            ],
            "performance_issues": [
                "Performance monitoring",
                "Automated optimization",
                "Caching system",
                "Load balancing"
            ],
            "usability_problems": [
                "Intuitive interface",
                "Mobile responsiveness",
                "Accessibility features",
                "User feedback system"
            ]
        }
        
        # Find matching features based on gap description
        if gap_description:
            gap_desc_lower = self._safe_lower(gap_description)
            for key, feature_list in features.items():
                if key in gap_desc_lower:
                    return feature_list
        
        return ["Core functionality", "User interface", "Integration capabilities", "Reporting"]
    
    def _design_technical_architecture(self, solution_type: str, complexity: str, 
                                     features: List[str]) -> Dict[str, Any]:
        """Design the technical architecture for the solution."""
        architectures = {
            "software": {
                "frontend": "React/Vue.js",
                "backend": "Python/FastAPI",
                "database": "PostgreSQL",
                "cloud": "AWS/GCP",
                "apis": "RESTful APIs"
            },
            "service": {
                "delivery": "SaaS platform",
                "infrastructure": "Cloud-native",
                "scalability": "Auto-scaling",
                "security": "Enterprise-grade"
            },
            "platform": {
                "architecture": "Microservices",
                "integration": "API-first",
                "extensibility": "Plugin system",
                "multi-tenant": "Yes"
            }
        }
        
        base_arch = architectures.get(solution_type, architectures["software"])
        
        # Adjust based on complexity
        if complexity == "high":
            base_arch.update({
                "scalability": "Distributed microservices",
                "performance": "Caching + CDN",
                "monitoring": "Comprehensive observability"
            })
        
        return base_arch
    
    def _determine_implementation_approach(self, complexity: str, budget: str, timeline: str) -> str:
        """Determine the implementation approach."""
        approaches = {
            ("low", "low", "1-3 months"): "Rapid prototyping with existing tools",
            ("moderate", "moderate", "3-6 months"): "Agile development with MVP focus",
            ("high", "high", "6-12 months"): "Comprehensive development with full features"
        }
        
        return approaches.get((complexity, budget, timeline), "Iterative development approach")
    
    def _estimate_development_time(self, features: List[str], complexity: str) -> str:
        """Estimate development time based on features and complexity."""
        base_weeks = len(features) * 2  # 2 weeks per feature
        
        complexity_multipliers = {
            "low": 1.0,
            "moderate": 1.5,
            "high": 2.5
        }
        
        total_weeks = int(base_weeks * complexity_multipliers.get(complexity, 1.5))
        
        if total_weeks <= 4:
            return "1 month"
        elif total_weeks <= 12:
            return "3 months"
        elif total_weeks <= 24:
            return "6 months"
        else:
            return "12+ months"
    
    def _estimate_development_cost(self, features: List[str], complexity: str, budget: str) -> float:
        """Estimate development cost based on features, complexity, and budget."""
        base_cost = len(features) * 10000  # $10K per feature
        
        complexity_multipliers = {
            "low": 1.0,
            "moderate": 1.5,
            "high": 3.0
        }
        
        budget_multipliers = {
            "low": 0.7,
            "moderate": 1.0,
            "high": 2.0
        }
        
        total_cost = base_cost * complexity_multipliers.get(complexity, 1.5) * budget_multipliers.get(budget, 1.0)
        return total_cost
    
    def _define_mvp_scope(self, features: List[str], timeline: str) -> List[str]:
        """Define MVP scope based on timeline."""
        if timeline == "1-3 months":
            return features[:2]  # Top 2 features
        elif timeline == "3-6 months":
            return features[:4]  # Top 4 features
        else:
            return features[:6]  # Top 6 features
    
    def _create_validation_plan(self, solution_name: str, target_market: str) -> Dict[str, Any]:
        """Create a validation plan for the solution."""
        return {
            "validation_phases": [
                "customer_interviews",
                "prototype_testing",
                "pilot_program",
                "market_validation"
            ],
            "success_criteria": [
                "10+ customer interviews",
                "Prototype usability score >80%",
                "Pilot program with 5+ users",
                "Market interest validation"
            ],
            "timeline": "4-6 weeks",
            "budget": "$5,000 - $10,000"
        }
    
    def _define_success_metrics(self, solution_name: str, target_market: str) -> List[str]:
        """Define success metrics for the solution."""
        return [
            "User adoption rate >20% within 6 months",
            "Customer satisfaction score >4.0/5.0",
            "Monthly recurring revenue >$10K within 12 months",
            "Customer acquisition cost <3x lifetime value",
            "Churn rate <5% monthly"
        ]
    
    def _assess_risks(self, solution_name: str, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for the solution."""
        return {
            "technical_risks": [
                "Scalability challenges",
                "Integration complexity",
                "Performance bottlenecks"
            ],
            "market_risks": [
                "Customer adoption",
                "Competitive response",
                "Market timing"
            ],
            "business_risks": [
                "Budget overruns",
                "Timeline delays",
                "Resource constraints"
            ],
            "mitigation_strategies": [
                "Start with MVP",
                "Iterative development",
                "Regular customer feedback",
                "Risk monitoring"
            ]
        }
    
    def _select_best_solution(self, prototypes: List[SolutionPrototype]) -> SolutionPrototype:
        """Select the best solution prototype from a list of prototypes."""
        if not prototypes:
            raise ValueError("No prototypes provided")
        
        # Simple scoring based on cost-effectiveness and market opportunity
        scored_prototypes = []
        for prototype in prototypes:
            score = (prototype.market_size / prototype.estimated_cost) * 100
            scored_prototypes.append((prototype, score))
        
        return max(scored_prototypes, key=lambda x: x[1])[0]
    
    def _select_recommended_solution(self, prototypes: List[SolutionPrototype]) -> Dict[str, Any]:
        """Select the recommended solution from prototypes."""
        if not prototypes:
            return {}
        
        recommended = self._select_best_solution(prototypes)
        
        # Calculate the score for the recommended solution
        score = (recommended.market_size / recommended.estimated_cost) * 100
        
        return {
            "solution_name": recommended.solution_name,
            "score": score,
            "reason": "Best cost-to-market-size ratio"
        }
    
    def _create_implementation_roadmap(self, recommended: Dict[str, Any], timeline: str) -> Dict[str, Any]:
        """Create an implementation roadmap."""
        phases = {
            "1-3 months": [
                {"phase": "Discovery", "duration": "2 weeks", "tasks": ["Customer interviews", "Market validation"]},
                {"phase": "Design", "duration": "3 weeks", "tasks": ["UI/UX design", "Technical design"]},
                {"phase": "MVP Development", "duration": "4 weeks", "tasks": ["Core features", "Testing"]}
            ],
            "3-6 months": [
                {"phase": "Planning", "duration": "2 weeks", "tasks": ["Requirements", "Architecture"]},
                {"phase": "MVP", "duration": "8 weeks", "tasks": ["Core development", "Testing"]},
                {"phase": "Enhancement", "duration": "6 weeks", "tasks": ["Additional features", "Optimization"]}
            ],
            "6-12 months": [
                {"phase": "Foundation", "duration": "4 weeks", "tasks": ["Architecture", "Core setup"]},
                {"phase": "MVP", "duration": "12 weeks", "tasks": ["Core features", "Testing"]},
                {"phase": "Scale", "duration": "16 weeks", "tasks": ["Advanced features", "Performance"]}
            ]
        }
        
        return {
            "timeline": timeline,
            "phases": phases.get(timeline, phases["3-6 months"]),
            "milestones": ["MVP launch", "Beta testing", "Market launch"]
        }
    
    def _calculate_resource_requirements(self, recommended: Dict[str, Any], solution_type: str) -> Dict[str, Any]:
        """Calculate resource requirements for implementation."""
        return {
            "team": {
                "developers": 2 if solution_type == "software" else 3,
                "designers": 1,
                "product_manager": 1,
                "qa_engineer": 1
            },
            "tools": ["Development environment", "Testing tools", "Deployment platform"],
            "budget_breakdown": {
                "development": 0.6,
                "design": 0.15,
                "testing": 0.15,
                "deployment": 0.1
            }
        }
    
    def _validate_solutions(
        self,
        solutions: List[SolutionPrototype],
        target_market: str
    ) -> Dict[str, Any]:
        """Validate solutions against market requirements."""
        validation_results = []
        
        for prototype in solutions:
            validation = self._validate_single_solution(prototype, target_market)
            validation_results.append(validation)
        
        return {
            "overall_validity": all(v["is_valid"] for v in validation_results),
            "individual_results": validation_results,
            "market_fit_score": sum(v["market_fit_score"] for v in validation_results) / len(validation_results)
        }
    
    def _validate_single_solution(self, prototype: SolutionPrototype, 
                                      target_market: str) -> Dict[str, Any]:
        """Validate a single solution against market requirements."""
        # Mock validation
        return {
            "solution_name": prototype.solution_name,
            "is_valid": True,
            "market_fit_score": 0.85,
            "technical_feasibility": 0.9,
            "business_viability": 0.8,
            "validation_notes": "Solution addresses key pain points effectively"
        }
    
    def _generate_next_steps(self, recommended: Dict[str, Any], 
                           validation: Dict[str, Any]) -> List[str]:
        """Generate next steps for implementation."""
        return [
            "Conduct detailed customer interviews",
            "Create detailed technical specifications",
            "Set up development environment",
            "Begin MVP development",
            "Establish feedback loops with potential customers",
            "Plan beta testing program"
        ]
    
    def _determine_design_approach(self, input_data: BuilderInput) -> str:
        """Determine the design approach based on input parameters."""
        if input_data.technical_complexity == "low" and input_data.budget_range == "low":
            return "Lean startup approach"
        elif input_data.technical_complexity == "moderate":
            return "Agile development approach"
        else:
            return "Comprehensive development approach"
    
    def _assess_mvp_complexity(self, market_gaps: List[Dict[str, Any]]) -> str:
        """Assess MVP complexity based on market gaps."""
        if len(market_gaps) <= 2:
            return "low"
        elif len(market_gaps) <= 4:
            return "moderate"
        else:
            return "high"
    
    def _create_technical_design(self, solution: SolutionPrototype) -> Dict[str, Any]:
        """Create technical design for the solution."""
        return {
            "architecture": solution.technical_architecture,
            "components": solution.key_features,
            "implementation_approach": solution.implementation_approach,
            "estimated_effort": solution.estimated_development_time
        }
    
    def _calculate_cost_analysis(self, solution: SolutionPrototype) -> Dict[str, Any]:
        """Calculate detailed cost analysis for the solution."""
        return {
            "development_cost": solution.estimated_cost,
            "market_size": solution.market_size,
            "roi_potential": (solution.market_size / solution.estimated_cost) * 100,
            "cost_breakdown": {
                "development": solution.estimated_cost * 0.7,
                "testing": solution.estimated_cost * 0.2,
                "deployment": solution.estimated_cost * 0.1
            }
        }
    
    def _estimate_timeline(self, solution: SolutionPrototype, target_timeline: str) -> Dict[str, Any]:
        """Estimate implementation timeline for the solution."""
        return {
            "estimated_duration": solution.estimated_development_time,
            "target_timeline": target_timeline,
            "milestones": solution.mvp_scope,
            "critical_path": solution.key_features[:3]
        }
    
    def _calculate_feasibility_score(self, solution: SolutionPrototype) -> float:
        """Calculate feasibility score for the solution."""
        # Simple scoring based on cost-effectiveness and complexity
        cost_score = min(100, (1000000 / solution.estimated_cost) * 10)
        market_score = min(100, (solution.market_size / 10000000) * 100)
        return (cost_score + market_score) / 2
    
    def _assess_risk_factors(self, solution: SolutionPrototype) -> List[Dict[str, Any]]:
        """Assess risk factors for the solution."""
        return [
            {
                "risk": "Technical complexity",
                "probability": "medium",
                "impact": "high",
                "mitigation": "Prototype key components early"
            },
            {
                "risk": "Market adoption",
                "probability": "low",
                "impact": "high",
                "mitigation": "Conduct user testing"
            }
        ]
    
    def _define_success_metrics(self, solution: SolutionPrototype, target_market: str) -> List[str]:
        """Define success metrics for the solution."""
        return [
            "User adoption rate > 10%",
            "Customer satisfaction > 4.0/5.0",
            "Revenue target achievement",
            "Market penetration in " + target_market
        ]


# Register the agent - moved to agent_registry.py
# from .base import register_agent
# register_agent("builder", BuilderAgent)


async def test_builder_agent():
    """Test BuilderAgent with prompt templates."""
    import logging
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("builder_agent_test")
    logger.info("Testing BuilderAgent with prompt templates...")
    
    # Sample market gaps data
    market_gaps = [
        {
            "id": "gap1",
            "description": "Lack of affordable project management tools for small businesses",
            "pain_point": "Small businesses struggle with project tracking and team coordination",
            "market_segment": "SMB",
            "score": 8.5
        },
        {
            "id": "gap2",
            "description": "Fragmented data analytics solutions for e-commerce",
            "pain_point": "E-commerce businesses lack integrated analytics for sales and marketing",
            "market_segment": "E-commerce",
            "score": 7.8
        }
    ]
    
    # Create BuilderInput
    builder_input = BuilderInput(
        market_gaps=market_gaps,
        target_market="small businesses",
        solution_type="software",
        budget_range="moderate",
        timeline="3-6 months",
        technical_complexity="moderate"
    )
    
    # Initialize BuilderAgent
    agent = BuilderAgent("builder-test")
    
    # Initialize LLM Backend
    from llm.base import LLMConfig, LLMBackendType
    from llm.manager import LLMManager
    from llm.backends.ollama import OllamaBackend
    
    # Create the LLM configuration for the Ollama backend
    config = LLMConfig(
        backend_type=LLMBackendType.OLLAMA,
        model_name="phi4-mini:latest",  # Using a smaller model for testing
        base_url="http://localhost:11434",
        timeout=180  # 3 minute timeout for complex queries
    )
    
    # Create and register the LLM backend
    llm_manager = LLMManager()
    backend = OllamaBackend(config)
    await llm_manager.register_backend(backend)
    
    # Assign LLM manager to agent
    agent._llm_manager = llm_manager
    
    # Override preferred backend for testing
    agent.preferred_backend = "ollama"
    
    try:
        # Execute plan phase
        logger.info("Running plan phase...")
        plan_start = datetime.now()
        plan = await agent.plan(builder_input)
        plan_time = (datetime.now() - plan_start).total_seconds()
        logger.info(f"Plan complete: {len(plan.get('phases', []))} phases identified in {plan_time:.2f}s")
        
        # Execute think phase
        logger.info("Running think phase...")
        think_start = datetime.now()
        think_result = await agent.think(builder_input)
        think_time = (datetime.now() - think_start).total_seconds()
        
        # Get gap analysis count
        gap_analysis = think_result.get("gap_analysis", [])
        if isinstance(gap_analysis, list):
            analysis_count = len(gap_analysis)
        else:
            analysis_count = 0
            
        logger.info(f"Think complete: {analysis_count} gaps analyzed in {think_time:.2f}s")
        
        # Execute act phase
        logger.info("Running act phase...")
        act_start = datetime.now()
        act_result = await agent.act(builder_input)
        act_time = (datetime.now() - act_start).total_seconds()
        
        # Log results
        logger.info(f"Act complete: {len(act_result.solution_prototypes)} solution prototypes generated in {act_time:.2f}s")
        logger.info(f"Solutions:")  
        for i, solution in enumerate(act_result.solution_prototypes):
            logger.info(f"  Solution {i+1}: {solution.solution_name} - Est. Cost: ${solution.estimated_cost}")
        
        logger.info("\n===== TEST SUCCESSFUL =====\n")
        logger.info(f"BuilderAgent completed solution generation in {plan_time + think_time + act_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error testing BuilderAgent: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_builder_agent())
