"""
Custom Handoff Tools for Supervisor using langgraph-supervisor.

This module implements custom handoff tools that:
- Pass session_id and task_id to agents
- Maintain our existing state structure
- Work with langgraph-supervisor's create_supervisor() function
"""

from typing import Annotated, Dict, Any, Optional
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

from aws_orchestrator_agent.utils.logger import AgentLogger

# Create logger
handoff_logger = AgentLogger("SUPERVISOR_HANDOFF")


def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:
    """
    Create a custom handoff tool that passes session_id and task_id to agents.
    
    Args:
        agent_name: Name of the target agent
        name: Tool name (if None, will be auto-generated)
        description: Tool description (if None, will be auto-generated)
        
    Returns:
        BaseTool: Custom handoff tool
    """
    
    # Auto-generate name and description if not provided
    if name is None:
        name = f"transfer_to_{agent_name}"
    if description is None:
        description = f"Transfer task to {agent_name}"

    @tool(name, description=description)
    def handoff_to_agent(
        # Task description for the LLM to populate
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # Injected state from the supervisor
        state: Annotated[dict, InjectedState],
        # Injected tool call ID
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """
        Handoff to a specific agent with session_id and task_id.
        
        This tool:
        1. Extracts session_id and task_id from the supervisor state
        2. Creates a tool message for the handoff
        3. Returns a Command to transfer control to the target agent
        4. Passes all necessary context including session_id and task_id
        """
        
        # Log the handoff attempt
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Handoff tool called for {agent_name}",
            extra={
                "agent_name": agent_name,
                "task_description": task_description,
                "state_keys": list(state.keys()) if isinstance(state, dict) else "not_dict",
                "state_type": type(state).__name__,
            }
        )
        
        # Extract session_id and task_id from state
        # langgraph-supervisor uses a different state schema, so we need to look in messages
        session_id = None
        task_id = None
        
        # Try to extract from state directly first
        session_id = state.get("session_id")
        task_id = state.get("task_id")
        user_request = state.get("user_request", "")
        
        # If not found in state, try to extract from messages
        if session_id is None or task_id is None or not user_request:
            messages = state.get("messages", [])
            for message in messages:
                if hasattr(message, 'additional_kwargs'):
                    metadata = message.additional_kwargs
                    if session_id is None and 'session_id' in metadata:
                        session_id = metadata['session_id']
                    if task_id is None and 'task_id' in metadata:
                        task_id = metadata['task_id']
                    if not user_request and 'user_request' in metadata:
                        user_request = metadata['user_request']
                elif hasattr(message, 'metadata'):
                    metadata = message.metadata
                    if session_id is None and 'session_id' in metadata:
                        session_id = metadata['session_id']
                    if task_id is None and 'task_id' in metadata:
                        task_id = metadata['task_id']
                    if not user_request and 'user_request' in metadata:
                        user_request = metadata['user_request']
        
        # Log extracted values
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Extracted session_id, task_id, and user_request from state",
            extra={
                "extracted_session_id": session_id,
                "extracted_task_id": task_id,
                "extracted_user_request": user_request,
                "state_has_session_id": "session_id" in state if isinstance(state, dict) else False,
                "state_has_task_id": "task_id" in state if isinstance(state, dict) else False,
                "state_has_user_request": "user_request" in state if isinstance(state, dict) else False,
            }
        )
        
        # Create tool message for the handoff
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        # Get messages from state
        messages = state.get("messages", [])
        
        # Create the state update that will be passed to the target agent
        # This includes all the context the agent needs
        state_update = {
            "messages": messages + [tool_message],
            "active_agent": agent_name,
            "task_description": task_description,
            # Pass our custom fields
            "session_id": session_id,
            "task_id": task_id,
            # Pass the extracted user_request
            "user_request": user_request if user_request else task_description,
            "status": state.get("status", "in_progress"),
        }
        
        # Log the final state update
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Handing off to {agent_name}",
            extra={
                "agent_name": agent_name,
                "state_update_keys": list(state_update.keys()),
                "final_session_id": state_update.get("session_id"),
                "final_task_id": state_update.get("task_id"),
            }
        )
        
        # Return Command to transfer control to the target agent
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update=state_update,
        )

    # Set metadata for langgraph-supervisor
    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    
    return handoff_to_agent


def create_handoff_tools_for_agents(agent_names: list[str]) -> list[BaseTool]:
    """
    Create handoff tools for multiple agents.
    
    Args:
        agent_names: List of agent names to create handoff tools for
        
    Returns:
        List[BaseTool]: List of handoff tools
    """
    tools = []
    
    for agent_name in agent_names:
        tool = create_custom_handoff_tool(
            agent_name=agent_name,
            name=None,  # Auto-generate name
            description=None  # Auto-generate description
        )
        tools.append(tool)
        
        handoff_logger.log_structured(
            level="INFO",
            message=f"Created handoff tool for {agent_name}",
            extra={
                "agent_name": agent_name,
                "tool_name": tool.name,
                "tool_description": tool.description,
            }
        )
    
    return tools
