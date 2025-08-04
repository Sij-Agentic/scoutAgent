# Multi-Backend LLM Abstraction Layer

A comprehensive abstraction layer for integrating multiple Large Language Model (LLM) backends into the ScoutAgent system. This implementation provides a unified interface for working with Ollama, OpenAI, Claude (Anthropic), and Gemini models.

## Features

- **Multi-Backend Support**: Seamlessly switch between Ollama, OpenAI, Claude, and Gemini
- **Backend-Agnostic Interface**: Unified API regardless of the underlying LLM provider
- **Health Monitoring**: Automatic health checks and failover capabilities
- **Async/Await Support**: Full asynchronous operation for optimal performance
- **Streaming Support**: Real-time streaming responses from all backends
- **Error Handling**: Comprehensive error handling with retry logic and fallbacks
- **Agent Integration**: Easy integration with existing ScoutAgent architecture
- **Configuration Management**: Flexible configuration through environment variables

## Architecture

```
llm/
├── __init__.py                 # Main exports
├── base.py                     # Base classes and interfaces
├── manager.py                  # LLM manager and orchestration
├── utils.py                    # Utility functions and agent mixin
└── backends/
    ├── __init__.py            # Backend exports
    ├── ollama.py              # Ollama backend implementation
    ├── openai.py              # OpenAI backend implementation
    ├── claude.py              # Claude/Anthropic backend implementation
    └── gemini.py              # Google Gemini backend implementation
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_llm.txt
```

### 2. Configure API Keys

Create a `.env` file or set environment variables:

```bash
# OpenAI
SCOUT_OPENAI_API_KEY=your_openai_api_key

# Anthropic Claude
SCOUT_ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
SCOUT_GEMINI_API_KEY=your_gemini_api_key

# Ollama (local, no API key needed)
# Make sure Ollama is running on http://localhost:11434
```

### 3. Basic Usage

```python
import asyncio
from llm.manager import initialize_llm_backends, get_llm_manager
from llm.base import LLMRequest

async def main():
    # Initialize backends
    await initialize_llm_backends()
    
    # Get manager
    manager = get_llm_manager()
    
    # Create request
    request = LLMRequest(
        messages=[{"role": "user", "content": "What is artificial intelligence?"}],
        temperature=0.7,
        max_tokens=200
    )
    
    # Generate response
    response = await manager.generate(request)
    print(response.content)

asyncio.run(main())
```

### 4. Agent Integration

```python
from llm.utils import LLMAgentMixin
from agents.base import BaseAgent

class MyEnhancedAgent(BaseAgent, LLMAgentMixin):
    async def plan(self, agent_input):
        # Use LLM for planning
        plan_prompt = f"Create a plan for: {agent_input.data}"
        plan = await self.llm_generate(plan_prompt)
        return {"plan": plan}
    
    async def think(self, agent_input, plan):
        # Use LLM for analysis
        analysis = await self.llm_analyze_text(
            str(agent_input.data), 
            analysis_type="strategic"
        )
        return analysis
    
    async def act(self, agent_input, plan, thoughts):
        # Use LLM for action generation
        action_prompt = f"Based on plan: {plan} and analysis: {thoughts}, what actions should be taken?"
        actions = await self.llm_generate(action_prompt)
        return {"actions": actions}
```

## Backend Configuration

### Ollama (Local)

```python
from llm.base import LLMConfig, LLMBackendType
from llm.backends import OllamaBackend

config = LLMConfig(
    backend_type=LLMBackendType.OLLAMA,
    model_name="llama2",
    base_url="http://localhost:11434",
    temperature=0.7,
    max_tokens=2048
)

backend = OllamaBackend(config)
```

### OpenAI

```python
config = LLMConfig(
    backend_type=LLMBackendType.OPENAI,
    model_name="gpt-4",
    api_key="your_api_key",
    temperature=0.7,
    max_tokens=4096
)

backend = OpenAIBackend(config)
```

### Claude (Anthropic)

```python
config = LLMConfig(
    backend_type=LLMBackendType.CLAUDE,
    model_name="claude-3-sonnet-20240229",
    api_key="your_api_key",
    temperature=0.7,
    max_tokens=4096
)

backend = ClaudeBackend(config)
```

### Gemini (Google)

```python
config = LLMConfig(
    backend_type=LLMBackendType.GEMINI,
    model_name="gemini-pro",
    api_key="your_api_key",
    temperature=0.7,
    max_tokens=4096
)

backend = GeminiBackend(config)
```

## Advanced Features

### Backend Switching

```python
# Use specific backend
response = await manager.generate(request, backend_type="openai")

# Automatic fallback if primary backend fails
response = await manager.generate(request)  # Uses default with fallback
```

### Streaming Responses

```python
async for chunk in manager.stream_generate(request):
    print(chunk, end="", flush=True)
```

### Health Monitoring

```python
# Check backend health
health_status = manager.get_all_health_status()
healthy_backends = manager.get_healthy_backends()

# Manual health check
is_healthy = await backend.health_check()
```

### Error Handling

```python
from llm.base import LLMError, LLMRateLimitError, LLMTimeoutError

try:
    response = await manager.generate(request)
except LLMRateLimitError:
    print("Rate limit exceeded, waiting...")
    await asyncio.sleep(60)
except LLMTimeoutError:
    print("Request timed out")
except LLMError as e:
    print(f"LLM error: {e}")
```

## Utility Functions

### Quick Generation

```python
from llm.utils import quick_llm_generate

response = await quick_llm_generate(
    "Summarize the benefits of AI",
    temperature=0.3,
    max_tokens=200
)
```

### Text Analysis

```python
from llm.utils import LLMAgentMixin

class AnalysisAgent(LLMAgentMixin):
    async def analyze_sentiment(self, text):
        return await self.llm_analyze_text(text, analysis_type="sentiment")
    
    async def extract_insights(self, data):
        return await self.llm_extract_insights(data, insight_type="business")
```

### Prompt Templates

```python
from llm.utils import AgentPromptTemplates

# Market analysis template
prompt = AgentPromptTemplates.market_analysis(
    pain_points=["Manual processes", "High costs"],
    market_context="SaaS industry"
)

response = await manager.generate(LLMRequest(messages=prompt.to_messages()))
```

## Enhanced Agent Example

The `EnhancedGapFinderAgent` demonstrates full LLM integration:

```python
from agents.gap_finder_enhanced import EnhancedGapFinderAgent
from agents.gap_finder import GapFinderInput
from agents.base import AgentInput

# Create input
gap_input = GapFinderInput(
    validated_pain_points=[
        {"description": "Manual inventory tracking", "severity": "high"}
    ],
    market_context="Small business software",
    analysis_scope="comprehensive"
)

# Execute enhanced agent
agent = EnhancedGapFinderAgent()
result = await agent.execute(AgentInput(data=gap_input, metadata={}))

print(f"Found {len(result.result.market_gaps)} market gaps")
print(f"Key insights: {result.result.result['key_insights']}")
```

## Testing

### Run Basic Tests

```python
python test_llm_integration.py
```

### Run with Pytest

```bash
pip install pytest pytest-asyncio
pytest test_llm_integration.py -v
```

### Run Integration Example

```python
python examples/llm_integration_example.py
```

## Configuration Options

### Environment Variables

All configuration can be done through environment variables with the `SCOUT_` prefix:

```bash
# Core LLM settings
SCOUT_LLM_DEFAULT_BACKEND=openai
SCOUT_LLM_TIMEOUT=60
SCOUT_LLM_RETRY_ATTEMPTS=3
SCOUT_LLM_RETRY_DELAY=1.0

# Backend-specific settings
SCOUT_OPENAI_API_KEY=your_key
SCOUT_OPENAI_MODEL=gpt-4
SCOUT_ANTHROPIC_API_KEY=your_key
SCOUT_ANTHROPIC_MODEL=claude-3-sonnet-20240229
SCOUT_GEMINI_API_KEY=your_key
SCOUT_GEMINI_MODEL=gemini-pro
SCOUT_OLLAMA_BASE_URL=http://localhost:11434
SCOUT_OLLAMA_MODEL=llama2
```

### Programmatic Configuration

```python
from config import get_config

config = get_config()
config.api.openai_api_key = "your_key"
config.api.anthropic_api_key = "your_key"
config.api.gemini_api_key = "your_key"
```

## Performance Considerations

### Async Best Practices

- Always use `await` with LLM operations
- Use `asyncio.gather()` for concurrent requests
- Implement proper error handling and timeouts

### Rate Limiting

- Each backend has its own rate limits
- The manager automatically handles retries with exponential backoff
- Monitor health status to avoid hitting limits

### Memory Management

- Large responses are handled efficiently
- Streaming is recommended for long-form content
- Cleanup resources with `manager.cleanup()`

## Troubleshooting

### Common Issues

1. **No backends available**: Check API keys and network connectivity
2. **Rate limit errors**: Implement backoff strategies or use multiple backends
3. **Timeout errors**: Increase timeout settings or reduce request complexity
4. **Model not found**: Verify model names are correct for each backend

### Debug Logging

```python
import logging
logging.getLogger("llm").setLevel(logging.DEBUG)
```

### Health Monitoring

```python
# Check backend status
status = manager.get_llm_status()
print(f"Available backends: {status['backends']}")
print(f"Healthy backends: {status['healthy_backends']}")
```

## Contributing

### Adding New Backends

1. Create a new backend class inheriting from `LLMBackend`
2. Implement required methods: `initialize()`, `generate()`, `stream_generate()`, `health_check()`
3. Add backend to `backends/__init__.py`
4. Update `manager.py` initialization logic
5. Add tests in `test_llm_integration.py`

### Example New Backend

```python
from llm.base import LLMBackend, LLMRequest, LLMResponse

class CustomBackend(LLMBackend):
    async def initialize(self) -> bool:
        # Initialize your backend
        return True
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Implement generation logic
        pass
    
    async def stream_generate(self, request: LLMRequest):
        # Implement streaming logic
        pass
    
    async def health_check(self) -> bool:
        # Check backend health
        return True
```

## License

This LLM abstraction layer is part of the ScoutAgent project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test files for examples
3. Examine the integration example
4. Check backend-specific documentation
