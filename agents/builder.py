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

from agents.base import BaseAgent, AgentInput, AgentOutput, AgentState
from agents.code_agent import CodeAgent
from agents.analysis_agent import AnalysisAgent
from config import get_config


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


class BuilderAgent(BaseAgent):
    """
    BuilderAgent for creating and validating solution prototypes.
    
    Uses technical analysis, cost estimation, and market validation
    to create actionable solution prototypes for identified market gaps.
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)
        self.code_agent = CodeAgent()
        self.analysis_agent = AnalysisAgent()
        self.config = get_config()
    
    async def plan(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Plan the solution building process."""
        self.logger.info(f"Planning solution building for {len(input_data.market_gaps)} market gaps")
        
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
        
        self.state.plan = plan
        return plan
    
    async def think(self, input_data: BuilderInput) -> Dict[str, Any]:
        """Analyze market gaps to design solutions."""
        self.logger.info("Analyzing market gaps for solution design...")
        
        # Analyze technical requirements
        tech_analysis = await self._analyze_technical_requirements(
            input_data.market_gaps,
            input_data.solution_type
        )
        
        # Evaluate solution feasibility
        feasibility = self._evaluate_solution_feasibility(
            input_data.market_gaps,
            getattr(input_data, 'budget_range', {'min': 1000, 'max': 50000}),
            input_data.timeline,
            getattr(input_data, 'technical_complexity', 'moderate')
        )
        
        # Prepare solution design strategy
        design_strategy = {
            "solution_count": len(input_data.market_gaps),
            "technical_complexity": input_data.technical_complexity,
            "feasibility_score": feasibility["overall_score"],
            "design_approach": self._determine_design_approach(input_data),
            "mvp_complexity": self._assess_mvp_complexity(input_data.market_gaps),
            "integration_challenges": feasibility["integration_challenges"]
        }
        
        return design_strategy
    
    async def act(self, input_data: BuilderInput) -> BuilderOutput:
        """Execute solution building and return prototypes."""
        self.logger.info("Executing solution building...")
        
        start_time = datetime.now()
        
        # Create solution prototypes
        solution_prototypes = []
        for gap in input_data.market_gaps:
            prototype = await self._create_solution_prototype(
                gap,
                input_data.target_market,
                input_data.solution_type,
                input_data.budget_range,
                input_data.timeline,
                input_data.technical_complexity
            )
            solution_prototypes.append(prototype)
        
        # Select recommended solution
        recommended_solution = self._select_recommended_solution(solution_prototypes)
        
        # Create implementation roadmap
        implementation_roadmap = self._create_implementation_roadmap(
            recommended_solution,
            input_data.timeline
        )
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(
            recommended_solution,
            input_data.solution_type
        )
        
        # Validate solutions
        validation_results = self._validate_solutions(
            solution_prototypes,
            input_data.target_market
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(
            recommended_solution,
            validation_results
        )
        
        best_solution = self._select_best_solution(solution_prototypes)
        technical_design = self._create_technical_design(best_solution)
        roadmap = self._create_implementation_roadmap(best_solution, input_data.timeline)
        cost_analysis = self._calculate_cost_analysis(best_solution)
        timeline_estimate = self._estimate_timeline(best_solution, input_data.timeline)
        feasibility_score = self._calculate_feasibility_score(best_solution)
        risk_factors = self._assess_risk_factors(best_solution)
        success_metrics = self._define_success_metrics(best_solution, input_data.target_market)
        
        return BuilderOutput(
            solution_prototypes=solution_prototypes,
            recommended_solution=recommended_solution,
            implementation_roadmap=roadmap,
            technical_design=technical_design,
            cost_analysis=cost_analysis,
            timeline_estimate=timeline_estimate,
            feasibility_score=feasibility_score,
            risk_factors=risk_factors,
            success_metrics=success_metrics
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
        # Simple name generation
        keywords = gap_description.lower().split()
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
        for key, feature_list in features.items():
            if key in gap_description.lower():
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
