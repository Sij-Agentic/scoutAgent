# MCP - Tool Service Implementation

This directory contains the implementation of the MCP (Master Control Program) tool service for the ScoutAgent system. The MCP provides a framework for registering, discovering, and executing tools across multiple servers using Server-Sent Events (SSE) for communication.

## Components

### Client

- `client/sse.py` - SSE client implementation for connecting to MCP servers
- `client/base.py` - Base client interfaces and implementations
- `client/multi.py` - Multi-server client for managing multiple MCP server connections

### Server

- `server/sse.py` - SSE server implementation for serving tools
- `server/base.py` - Base server interfaces and tool registration

### Models

- `models.py` - Pydantic models for MCP requests, responses, and configurations

### Sandbox

- `sandbox/proxy.py` - Tool proxy for providing safe access to tools from sandboxed code
- `sandbox/executor.py` - Sandboxed code execution with tool access

### Service

- `service.py` - Service integration with the ScoutAgent service registry

## Configuration

MCP servers are configured in `config/mcp_servers.yaml`. An example configuration is provided in `config/mcp_servers.yaml.example`.

## Usage

### Basic Usage

```python
from mcp.service import MCPToolService

# Get service instance
service = MCPToolService.get_instance()

# List available tools
tools = service.get_available_tools()

# Call a tool
result = await service.call_tool("tool_name", {"param": "value"})

# Execute code with tool access
result = await service.execute_code_with_tools("""
result = tools.tool_name(param="value")
print(result)
""")
```

### Advanced Usage - Direct Client

```python
from mcp.client.sse import sse_client

# Create client
client = await sse_client("http://localhost:8000")

# List tools
tools = await client.list_tools()

# Call tool
response = await client.call_tool("tool_name", {"param": "value"})
```

## Implementation Notes

This implementation has fully replaced the reference implementation that was previously in the `MCP-inspiration` directory. Key improvements:

1. Production-ready SSE-based tool communication
2. Multi-server tool discovery and routing
3. Sandboxed code execution with tool proxy access
4. Clean integration with the ScoutAgent service registry
5. Comprehensive testing suite

## Testing

Run the basic tests:
```
python -m mcp.tests.test_simple
```

Run the full integration tests:
```
python -m mcp.tests.test_full_integration
```

Note: Full integration tests require `uvicorn` to be installed.
