"""
Analysis Agent for ScoutAgent.

Specialized agent for data analysis, pattern recognition, and insights generation.
"""

from typing import Dict, Any, List, Optional, Union
import json
import statistics
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
import math

import asyncio
from .base import BaseAgent, AgentInput, AgentOutput
from config import get_config
from llm.utils import LLMAgentMixin, load_prompt_template

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first valid JSON object from a string."""
    # Look for a JSON block enclosed in ```json ... ```
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        text = match.group(1)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback for cases where the JSON is not perfectly formed
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                return None
    return None



class AnalysisAgent(BaseAgent, LLMAgentMixin):
    """
    Analysis agent for data processing, pattern recognition, and insights.
    
    Capabilities:
    - Statistical analysis and data profiling
    - Pattern recognition and anomaly detection
    - Trend analysis and forecasting
    - Text analysis and sentiment analysis
    - Data visualization recommendations
    """
    
    def __init__(self, name="analysis", **kwargs):
        super().__init__(name=name, **kwargs)
        LLMAgentMixin.__init__(self, preferred_backend='ollama')
        self.config = get_config()
    
    async def plan(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Plan analysis strategy based on input data using an LLM.
        
        Args:
            agent_input: Contains data and analysis parameters
            
        Returns:
            Analysis plan with methodology and expected outputs
        """
        data = agent_input.data
        data_type = self._determine_data_type(data)
        data_sample = str(data)[:500]  # Provide a sample of the data

        prompt = load_prompt_template(
            "plan.prompt", 
            agent_name=self.name,
            substitutions={
                'data_type': data_type,
                'data_sample': data_sample
            }
        )

        self.log("Generating analysis plan with LLM...")
        response_str = await self.llm_generate(prompt)
        plan = _extract_json(response_str)

        if plan:
            self.log(f"Analysis plan created for {data_type} data")
            # Ensure data_type is in the plan for the act method
            plan['data_type'] = data_type
        else:
            self.log("Failed to parse LLM response as JSON.", level='error')
            plan = {"error": "Failed to generate a valid plan.", "data_type": data_type}

        return plan
    
    async def think(self, agent_input: AgentInput, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data characteristics and determine approach using an LLM.
        
        Args:
            agent_input: Original input
            plan: Analysis plan
            
        Returns:
            Data insights and approach recommendations
        """
        data = agent_input.data
        data_sample = str(data)[:1000]
        plan_str = json.dumps(plan, indent=2)

        prompt = load_prompt_template(
            "think.prompt",
            agent_name=self.name,
            substitutions={
                'data_sample': data_sample,
                'plan': plan_str
            }
        )

        self.log("Generating data insights with LLM...")
        response_str = await self.llm_generate(prompt)
        insights = _extract_json(response_str)
        if not insights:
            self.log("Failed to parse LLM response as JSON.", level='error')
            insights = {}

        self.log(f"Data analysis insights generated.")
        return insights
    
    async def act(self, agent_input: AgentInput, plan: Dict[str, Any], thoughts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis based on plan and insights.
        
        Args:
            agent_input: Original input
            plan: Analysis plan
            thoughts: Data insights and recommendations
            
        Returns:
            Comprehensive analysis results
        """
        data = agent_input.data
        data_type = plan['data_type']
        analysis_type = plan['analysis_type']
        
        results = {
            'original_data': data,
            'data_type': data_type,
            'analysis_type': analysis_type,
            'executed_analyses': [],
            'findings': {},
            'insights': [],
            'recommendations': [],
            'metadata': {}
        }
        
        # Execute analyses based on data type and plan
        if data_type == 'numerical':
            results['executed_analyses'].extend(self._analyze_numerical_data(data))
        elif data_type == 'text':
            results['executed_analyses'].extend(self._analyze_text_data(data))
        elif data_type == 'categorical':
            results['executed_analyses'].extend(self._analyze_categorical_data(data))
        elif data_type == 'time_series':
            results['executed_analyses'].extend(self._analyze_time_series_data(data))
        elif data_type == 'structured':
            results['executed_analyses'].extend(self._analyze_structured_data(data))
        
        # Generate insights and recommendations
        results['findings'] = self._compile_findings(results['executed_analyses'])
        results['insights'] = self._generate_insights(results['findings'])
        results['recommendations'] = self._generate_recommendations(results['findings'])
        
        # Add metadata
        quality_assessment = thoughts.get('quality_assessment', {})
        results['metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_size': len(data) if hasattr(data, '__len__') else str(data).__len__(),
            'processing_time': time.time() - self.state.start_time,
            'quality_score': quality_assessment.get('overall_score', 0.0),
            'quality_comments': quality_assessment.get('comments', 'N/A'),
            'reasoning': thoughts.get('reasoning', 'N/A'),
            'challenges': thoughts.get('challenges', [])
        }
        
        self.log(f"Analysis completed - generated {len(results['insights'])} insights")
        return results
    
    def _determine_data_type(self, data: Any) -> str:
        """Determine the type of input data."""
        if isinstance(data, (list, tuple)):
            if all(isinstance(item, (int, float)) for item in data):
                return 'numerical'
            elif all(isinstance(item, str) for item in data):
                return 'text'
            else:
                return 'mixed'
        elif isinstance(data, dict):
            return 'structured'
        elif isinstance(data, str):
            # Check if it's JSON
            try:
                json.loads(data)
                return 'structured'
            except json.JSONDecodeError:
                return 'text'
        else:
            return 'unknown'
    
    def _select_methodology(self, data_type: str) -> str:
        """Select appropriate analysis methodology."""
        methodologies = {
            'numerical': 'statistical_analysis',
            'text': 'nlp_processing',
            'categorical': 'frequency_analysis',
            'time_series': 'temporal_analysis',
            'structured': 'multi_dimensional_analysis',
            'mixed': 'hybrid_analysis'
        }
        return methodologies.get(data_type, 'general_analysis')
    
    def _create_analysis_steps(self, data_type: str) -> List[str]:
        """Create analysis steps based on data type."""
        steps = {
            'numerical': [
                'data_profiling',
                'statistical_summary',
                'distribution_analysis',
                'outlier_detection',
                'correlation_analysis'
            ],
            'text': [
                'text_preprocessing',
                'tokenization',
                'sentiment_analysis',
                'keyword_extraction',
                'topic_modeling'
            ],
            'categorical': [
                'frequency_distribution',
                'mode_analysis',
                'category_relationships',
                'chi_square_tests'
            ],
            'time_series': [
                'temporal_patterns',
                'trend_analysis',
                'seasonality_detection',
                'forecasting'
            ],
            'structured': [
                'schema_analysis',
                'missing_data_analysis',
                'data_relationships',
                'cross_field_analysis'
            ]
        }
        return steps.get(data_type, ['basic_analysis'])
    
    def _define_outputs(self, data_type: str) -> List[str]:
        """Define expected analysis outputs."""
        outputs = {
            'numerical': ['statistics', 'distributions', 'correlations', 'outliers'],
            'text': ['sentiment_scores', 'keywords', 'topics', 'readability'],
            'categorical': ['frequencies', 'modes', 'associations'],
            'time_series': ['trends', 'seasonality', 'forecasts'],
            'structured': ['schema', 'relationships', 'quality_metrics']
        }
        return outputs.get(data_type, ['summary'])
    
    def _define_validation_criteria(self, data_type: str) -> List[str]:
        """Define validation criteria for analysis."""
        criteria = {
            'numerical': ['data_completeness', 'statistical_significance', 'outlier_threshold'],
            'text': ['text_quality', 'sentiment_accuracy', 'keyword_relevance'],
            'categorical': ['category_coverage', 'frequency_balance'],
            'time_series': ['temporal_completeness', 'trend_significance'],
            'structured': ['schema_consistency', 'data_integrity']
        }
        return criteria.get(data_type, ['basic_validation'])
    
    def _analyze_data_characteristics(self, data: Any, data_type: str) -> Dict[str, Any]:
        """Analyze data characteristics."""
        characteristics = {
            'size': len(data) if hasattr(data, '__len__') else 1,
            'type': type(data).__name__,
            'structure': self._describe_structure(data),
            'value_range': self._get_value_range(data, data_type),
            'distribution_shape': self._describe_distribution(data, data_type)
        }
        return characteristics
    
    def _assess_data_quality(self, data: Any, data_type: str) -> Dict[str, Any]:
        """Assess data quality."""
        quality = {
            'completeness': self._calculate_completeness(data),
            'consistency': self._check_consistency(data),
            'accuracy': self._assess_accuracy(data),
            'overall_score': 0.0
        }
        
        # Calculate overall score
        weights = {'completeness': 0.4, 'consistency': 0.3, 'accuracy': 0.3}
        quality['overall_score'] = sum(
            quality[key] * weights[key] 
            for key in weights if key in quality
        )
        
        return quality
    
    def _detect_patterns(self, data: Any, data_type: str) -> List[Dict[str, Any]]:
        """Detect patterns in data."""
        patterns = []
        
        if data_type == 'numerical':
            patterns.extend(self._detect_numerical_patterns(data))
        elif data_type == 'text':
            patterns.extend(self._detect_text_patterns(data))
        elif data_type == 'categorical':
            patterns.extend(self._detect_categorical_patterns(data))
        
        return patterns
    
    def _detect_anomalies(self, data: Any, data_type: str) -> List[Dict[str, Any]]:
        """Detect anomalies in data."""
        anomalies = []
        
        if data_type == 'numerical':
            anomalies.extend(self._detect_numerical_anomalies(data))
        elif data_type == 'text':
            anomalies.extend(self._detect_text_anomalies(data))
        
        return anomalies
    
    def _recommend_approach(self, data: Any, data_type: str) -> List[str]:
        """Recommend analysis approaches."""
        recommendations = []
        
        if data_type == 'numerical':
            recommendations.extend([
                'Use descriptive statistics',
                'Consider outlier detection',
                'Apply normalization if needed'
            ])
        elif data_type == 'text':
            recommendations.extend([
                'Use NLP preprocessing',
                'Consider sentiment analysis',
                'Apply topic modeling'
            ])
        
        return recommendations
    
    def _assess_complexity(self, data: Any, data_type: str) -> str:
        """Assess analysis complexity."""
        size = len(data) if hasattr(data, '__len__') else 1
        
        if size < 100:
            return 'simple'
        elif size < 1000:
            return 'moderate'
        else:
            return 'complex'
    
    def _analyze_numerical_data(self, data: List[Union[int, float]]) -> List[Dict[str, Any]]:
        """Perform numerical data analysis."""
        if not data or not all(isinstance(x, (int, float)) for x in data):
            return []
        
        analyses = []
        
        # Basic statistics
        analyses.append({
            'type': 'descriptive_statistics',
            'results': {
                'count': len(data),
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'mode': statistics.mode(data) if len(set(data)) < len(data) else None,
                'std_dev': statistics.stdev(data) if len(data) > 1 else 0,
                'min': min(data),
                'max': max(data),
                'range': max(data) - min(data)
            }
        })
        
        # Distribution analysis
        analyses.append({
            'type': 'distribution_analysis',
            'results': {
                'skewness': self._calculate_skewness(data),
                'kurtosis': self._calculate_kurtosis(data),
                'quartiles': self._calculate_quartiles(data)
            }
        })
        
        # Outlier detection
        outliers = self._detect_numerical_outliers(data)
        if outliers:
            analyses.append({
                'type': 'outlier_detection',
                'results': {'outliers': outliers, 'count': len(outliers)}
            })
        
        return analyses
    
    def _analyze_text_data(self, data: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Perform text data analysis."""
        if isinstance(data, str):
            texts = [data]
        else:
            texts = data
        
        if not texts or not all(isinstance(t, str) for t in texts):
            return []
        
        analyses = []
        
        # Basic text metrics
        combined_text = ' '.join(texts)
        word_count = len(combined_text.split())
        char_count = len(combined_text)
        
        analyses.append({
            'type': 'text_metrics',
            'results': {
                'total_documents': len(texts),
                'total_words': word_count,
                'total_characters': char_count,
                'avg_words_per_doc': word_count / len(texts),
                'avg_chars_per_doc': char_count / len(texts)
            }
        })
        
        # Keyword extraction
        keywords = self._extract_keywords(texts)
        analyses.append({
            'type': 'keyword_extraction',
            'results': {'top_keywords': keywords[:10]}
        })
        
        # Sentiment analysis (mock)
        sentiment = self._analyze_sentiment(texts)
        analyses.append({
            'type': 'sentiment_analysis',
            'results': sentiment
        })
        
        return analyses
    
    def _analyze_categorical_data(self, data: List[str]) -> List[Dict[str, Any]]:
        """Perform categorical data analysis."""
        if not data or not all(isinstance(x, str) for x in data):
            return []
        
        analyses = []
        
        # Frequency distribution
        frequency = Counter(data)
        analyses.append({
            'type': 'frequency_distribution',
            'results': {
                'frequencies': dict(frequency),
                'unique_categories': len(frequency),
                'most_common': frequency.most_common(5),
                'least_common': frequency.most_common()[-5:]
            }
        })
        
        # Mode analysis
        if frequency:
            mode_value = frequency.most_common(1)[0]
            analyses.append({
                'type': 'mode_analysis',
                'results': {
                    'mode': mode_value[0],
                    'mode_frequency': mode_value[1],
                    'mode_percentage': (mode_value[1] / len(data)) * 100
                }
            })
        
        return analyses
    
    def _analyze_time_series_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform time series analysis."""
        # Mock implementation
        return [{
            'type': 'time_series_analysis',
            'results': {
                'trend': 'increasing',
                'seasonality': False,
                'forecast': [1, 2, 3, 4, 5]
            }
        }]
    
    def _analyze_structured_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform structured data analysis."""
        analyses = []
        
        # Schema analysis
        schema = self._analyze_schema(data)
        analyses.append({
            'type': 'schema_analysis',
            'results': schema
        })
        
        # Field analysis
        for field, value in data.items():
            field_analysis = self._analyze_field(field, value)
            analyses.append({
                'type': 'field_analysis',
                'field': field,
                'results': field_analysis
            })
        
        return analyses
    
    def _compile_findings(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile findings from all analyses."""
        findings = {
            'summary': {},
            'detailed_results': analyses,
            'key_metrics': {},
            'warnings': [],
            'successes': []
        }
        
        # Extract key metrics
        for analysis in analyses:
            if 'results' in analysis:
                findings['key_metrics'].update(analysis['results'])
        
        return findings
    
    def _generate_insights(self, findings: Dict[str, Any]) -> List[str]:
        """Generate insights from findings."""
        insights = []
        
        # Generate insights based on findings
        if 'descriptive_statistics' in findings['key_metrics']:
            stats = findings['key_metrics']['descriptive_statistics']
            insights.append(f"Data contains {stats['count']} observations")
            insights.append(f"Mean value is {stats['mean']:.2f}")
            insights.append(f"Standard deviation is {stats['std_dev']:.2f}")
        
        if 'frequencies' in findings['key_metrics']:
            frequencies = findings['key_metrics']['frequencies']
            insights.append(f"Found {len(frequencies)} unique categories")
        
        return insights
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Generate recommendations based on findings
        recommendations.append("Consider visualizing key metrics")
        recommendations.append("Validate data quality before further analysis")
        recommendations.append("Document findings and share with stakeholders")
        
        return recommendations
    
    # Helper methods for numerical analysis
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of numerical data."""
        if len(data) < 3:
            return 0.0
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        skewness = (sum((x - mean) ** 3 for x in data) / n) / (std_dev ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of numerical data."""
        if len(data) < 4:
            return 0.0
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        if std_dev == 0:
            return 0.0
        
        n = len(data)
        kurtosis = (sum((x - mean) ** 4 for x in data) / n) / (std_dev ** 4) - 3
        return kurtosis
    
    def _calculate_quartiles(self, data: List[float]) -> Dict[str, float]:
        """Calculate quartiles of numerical data."""
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        def percentile(p):
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_data[f]
            return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
        
        return {
            'q1': percentile(25),
            'q2': percentile(50),
            'q3': percentile(75)
        }
    
    def _detect_numerical_outliers(self, data: List[float]) -> List[float]:
        """Detect outliers in numerical data using IQR method."""
        if len(data) < 4:
            return []
        
        sorted_data = sorted(data)
        q1 = sorted_data[len(sorted_data) // 4]
        q3 = sorted_data[3 * len(sorted_data) // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return [x for x in data if x < lower_bound or x > upper_bound]
    
    def _detect_numerical_patterns(self, data: List[float]) -> List[Dict[str, Any]]:
        """Detect patterns in numerical data."""
        patterns = []
        
        # Check for increasing/decreasing trends
        if len(data) > 1:
            differences = [data[i+1] - data[i] for i in range(len(data)-1)]
            avg_diff = statistics.mean(differences)
            
            if avg_diff > 0:
                patterns.append({'type': 'trend', 'description': 'increasing trend'})
            elif avg_diff < 0:
                patterns.append({'type': 'trend', 'description': 'decreasing trend'})
        
        return patterns
    
    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """Extract keywords from text data."""
        combined_text = ' '.join(texts).lower()
        words = re.findall(r'\b\w+\b', combined_text)
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        word_freq = Counter(filtered_words)
        return [word for word, _ in word_freq.most_common(20)]
    
    def _analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Perform sentiment analysis on text data (mock implementation)."""
        # Mock sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing']
        
        combined_text = ' '.join(texts).lower()
        positive_count = sum(1 for word in positive_words if word in combined_text)
        negative_count = sum(1 for word in negative_words if word in combined_text)
        
        total = positive_count + negative_count
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.0}
        
        score = (positive_count - negative_count) / total
        sentiment = 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        
        return {'sentiment': sentiment, 'score': score}
    
    def _detect_text_patterns(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns in text data."""
        patterns = []
        
        # Check for repeated phrases
        combined_text = ' '.join(texts)
        words = combined_text.split()
        
        # Find repeated word sequences
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if combined_text.count(phrase) > 1:
                patterns.append({
                    'type': 'repeated_phrase',
                    'phrase': phrase,
                    'frequency': combined_text.count(phrase)
                })
        
        return patterns
    
    def _detect_text_anomalies(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect anomalies in text data."""
        anomalies = []
        
        # Check for unusual length
        lengths = [len(text) for text in texts]
        avg_length = statistics.mean(lengths)
        std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
        
        for i, text in enumerate(texts):
            if abs(len(text) - avg_length) > 2 * std_length:
                anomalies.append({
                    'type': 'length_anomaly',
                    'index': i,
                    'length': len(text),
                    'expected_range': (avg_length - 2*std_length, avg_length + 2*std_length)
                })
        
        return anomalies
    
    def _detect_categorical_patterns(self, data: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns in categorical data."""
        patterns = []
        
        frequency = Counter(data)
        
        # Check for dominant category
        if frequency:
            most_common = frequency.most_common(1)[0]
            if most_common[1] > len(data) * 0.5:
                patterns.append({
                    'type': 'dominant_category',
                    'category': most_common[0],
                    'percentage': (most_common[1] / len(data)) * 100
                })
        
        return patterns
    
    def _describe_structure(self, data: Any) -> str:
        """Describe the structure of the data."""
        if isinstance(data, dict):
            return f"Dictionary with {len(data)} keys"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        elif isinstance(data, str):
            return f"String with {len(data)} characters"
        else:
            return f"{type(data).__name__} object"
    
    def _get_value_range(self, data: Any, data_type: str) -> Optional[Dict[str, Any]]:
        """Get value range for numerical data."""
        if data_type == 'numerical' and isinstance(data, list):
            try:
                numbers = [float(x) for x in data if isinstance(x, (int, float))]
                if numbers:
                    return {'min': min(numbers), 'max': max(numbers)}
            except (ValueError, TypeError):
                pass
        return None
    
    def _describe_distribution(self, data: Any, data_type: str) -> str:
        """Describe the distribution of the data."""
        if data_type == 'numerical' and isinstance(data, list):
            try:
                numbers = [float(x) for x in data if isinstance(x, (int, float))]
                if len(numbers) > 1:
                    mean = statistics.mean(numbers)
                    std_dev = statistics.stdev(numbers)
                    return f"Normal-like distribution (mean={mean:.2f}, std={std_dev:.2f})"
            except (ValueError, TypeError):
                pass
        return "Distribution not calculated"
    
    def _calculate_completeness(self, data: Any) -> float:
        """Calculate data completeness score."""
        if isinstance(data, dict):
            total_fields = len(data)
            non_empty_fields = sum(1 for v in data.values() if v is not None and v != '')
            return non_empty_fields / total_fields if total_fields > 0 else 1.0
        elif isinstance(data, list):
            non_empty_items = sum(1 for item in data if item is not None and item != '')
            return non_empty_items / len(data) if data else 1.0
        return 1.0
    
    def _check_consistency(self, data: Any) -> float:
        """Check data consistency."""
        # Simple consistency check
        return 0.9  # Mock implementation
    
    def _assess_accuracy(self, data: Any) -> float:
        """Assess data accuracy."""
        # Mock accuracy assessment
        return 0.85  # Mock implementation
    
    def _analyze_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema of structured data."""
        schema = {}
        for key, value in data.items():
            schema[key] = {
                'type': type(value).__name__,
                'nullable': value is None,
                'empty': value == '' if isinstance(value, str) else False
            }
        return schema
    
    def _analyze_field(self, field: str, value: Any) -> Dict[str, Any]:
        """Analyze a specific field in structured data."""
        return {
            'field_name': field,
            'data_type': type(value).__name__,
            'value': str(value)[:100] + '...' if len(str(value)) > 100 else str(value),
            'analysis': f"Field {field} contains {type(value).__name__} data"
        }


if __name__ == "__main__":
    # Test the analysis agent
    async def main():
        agent = AnalysisAgent()
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
        test_input = AgentInput(
            data=test_data,
            metadata={'analysis_type': 'exploratory'}
        )
        
        result = await agent.execute(test_input)
        print(f"Analysis completed: {result.success}")
        # The structure of the result will now depend on the LLM's output
        # so we print it directly to inspect.
        print(json.dumps(result.result, indent=2))

    asyncio.run(main())
