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