from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, List, Optional


# Generic wrapper for tool responses when returning JSON text
class ToolResult(BaseModel):
    success: bool = True
    result: Any = None
    error: Optional[str] = None


# Example/common schemas (extend as needed)
class SearchInput(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=50)


class SearchItem(BaseModel):
    title: str
    url: str
    snippet: str


class SearchOutput(BaseModel):
    results: List[SearchItem]


class ReadFileInput(BaseModel):
    path: str


class WriteFileInput(BaseModel):
    path: str
    content: str


class ListDirectoryInput(BaseModel):
    path: str


class FileInfo(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: Optional[int] = None


class ListDirectoryOutput(BaseModel):
    files: List[FileInfo]
