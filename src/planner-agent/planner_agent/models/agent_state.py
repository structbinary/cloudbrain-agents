# Copyright (C) 2025 StructBinary
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from typing import Any, Dict, List, Optional, Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class MultiAgentState(BaseModel):
    """State schema for multi-agent planning workflow."""
    messages: Annotated[list, add_messages]
    user_query:str
    status: Optional[str] = None
    question: Optional[str] = None
    refined_task_list: Optional[List[str]] = None
    selected_agent: Optional[List[Dict[str, Any]]] = None
    agent_capabilities: Optional[List[str]] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    final_response: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    context_id: Optional[str] = None
    next: Optional[Literal[
        "task_decomposition",
        "definition_checker",
        "a2a_agent_mapper",
        "mcp_server_mapper",
        "__end__"
    ]] = None
    resume_value: Optional[Any] = None

    class Config:
        extra = "allow"



class A2AAgentCardResponse(BaseModel):
    selected_agent: Optional[List[Dict[str, Any]]] = None
    refined_task_list: Optional[List[str]] = None
    status: Optional[str] = None
    question: Optional[str] = None


class MCPNodeResponse(BaseModel):
    agent_responses: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    question: Optional[str] = None


class TaskDecomposition(BaseModel):
    """Task decomposition."""
    task_list: List[str] = Field(description="Task list")
    status: Literal["input_required", "completed", "error"] = Field(description="Status of the task decomposition")
    question: Optional[str] = Field(description="Question to ask user when input is required or error message")