"""
Core models for MCP (Model Context Protocol) implementation.

This module provides the data structures used throughout the MCP implementation,
including tools schemas, communication types, and protocol-specific structures.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union, TypedDict
from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Content types for MCP responses."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    ERROR = "error"
    BINARY = "binary"


class TextContent(BaseModel):
    """Text content response from a tool."""
    type: str = "text"
    text: str


class ImageContent(BaseModel):
    """Image content response from a tool."""
    type: str = "image"
    url: str
    data: Optional[bytes] = None
    mime_type: str = "image/png"


class FileContent(BaseModel):
    """File content response from a tool."""
    type: str = "file"
    filename: str
    data: bytes
    mime_type: str = "application/octet-stream"


class ErrorContent(BaseModel):
    """Error content response from a tool."""
    type: str = "error"
    message: str
    code: Optional[str] = None


Content = Union[TextContent, ImageContent, FileContent, ErrorContent]


class ToolSchema(BaseModel):
    """JSON Schema for tool input validation."""
    type: str = Field(default="object")
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    
    class Config:
        extra = "allow"


class Tool(BaseModel):
    """Representation of a tool provided by an MCP server."""
    name: str
    description: str
    inputSchema: ToolSchema
    version: Optional[str] = "1.0.0"
    deprecated: bool = False
    
    @validator("inputSchema", pre=True)
    def validate_schema(cls, v):
        """Convert dict to ToolSchema if needed."""
        if isinstance(v, dict):
            return ToolSchema(**v)
        return v


class ToolResponse(BaseModel):
    """Response from a tool execution."""
    content: List[Content] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution with processed content."""
    raw_response: ToolResponse
    result: Any = None
    error: Optional[str] = None
    success: bool = True


class ServerInfo(BaseModel):
    """Information about an MCP server."""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)


class ToolsResponse(BaseModel):
    """Response containing available tools from a server."""
    tools: List[Tool] = Field(default_factory=list)


class ServerOptions(BaseModel):
    """Options for initializing a server."""
    timeout: Optional[float] = 60.0
    max_concurrent_requests: Optional[int] = 10


class ToolRequest(BaseModel):
    """Request to call a tool."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Example tool input/output model classes

class SearchInput(BaseModel):
    """Input for search tools."""
    query: str
    max_results: int = Field(default=10, description="Maximum number of results to return")


class URLListOutput(BaseModel):
    """Output containing a list of URLs."""
    result: List[str]


class UrlInput(BaseModel):
    """Input containing a URL."""
    url: str


class SummaryInput(BaseModel):
    """Input for summarization tools."""
    url: str
    prompt: Optional[str] = None


class CodeInput(BaseModel):
    """Input containing code to execute."""
    code: str
    timeout: float = 30.0


class CodeOutput(BaseModel):
    """Output from code execution."""
    result: str
    error: Optional[str] = None
    success: bool = True


class ServerConfig(BaseModel):
    """MCP server configuration."""
    id: str
    name: str
    url: str
    timeout: float = 30.0
    headers: Dict[str, str] = Field(default_factory=dict)
    prefix_tools: bool = False
