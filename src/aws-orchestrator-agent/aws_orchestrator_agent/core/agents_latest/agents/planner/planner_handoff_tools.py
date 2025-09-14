"""
Custom Handoff Tools for Planner Sub-Supervisor.

This module implements custom handoff tools for the planner sub-supervisor that:
- Pass planning-specific context between agents
- Manage planning workflow state
- Handle planning phase transitions
- Maintain planning data across handoffs
"""

from typing import Annotated, Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from aws_orchestrator_agent.utils.logger import AgentLogger
import asyncio
from contextlib import asynccontextmanager

# Create logger
logger = AgentLogger("PLANNER_HANDOFF")

def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:

    @tool(name, description=description)
    def handoff_to_agent(
        # you can add additional tool call arguments for the LLM to populate
        # for example, you can ask the LLM to populate a task description for the next agent
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # you can inject the state of the agent that is calling the tool
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        # Enhanced logging for cancellation detection
        logger.log_structured(
            level="INFO",
            message=f"Handoff tool called: {name} -> {agent_name}",
            extra={
                "tool_name": name,
                "target_agent": agent_name,
                "tool_call_id": tool_call_id,
                "task_description_length": len(task_description) if task_description else 0,
                "state_type": type(state).__name__,
                "has_planning_workflow_state": hasattr(state, 'planning_workflow_state'),
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', 'unknown') if hasattr(state, 'planning_workflow_state') else 'unknown',
                "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                "loop_counter": getattr(state.planning_workflow_state, 'loop_counter', 0) if hasattr(state, 'planning_workflow_state') else 0,
                "timestamp": datetime.now().isoformat()
            }
        )
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        messages = getattr(state, "messages", [])
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                # Pass the task description to the next agent
                "task_description": task_description,
                # Pass the user request and other important context
                "user_request": getattr(state, "user_request", task_description),
                "session_id": getattr(state, "session_id", None),
                "task_id": getattr(state, "task_id", None),
                "status": getattr(state, "status", "in_progress"),
                # Pass planning workflow state to maintain context
                "planning_workflow_state": getattr(state, "planning_workflow_state", None),
                "requirements_data": getattr(state, "requirements_data", None),
                "planning_context": f"Handing off to {agent_name} for: {task_description}",
            },
        )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent

def create_handoff_to_requirements_analyzer() -> BaseTool:
    """Create handoff tool for Requirements Analyzer agent."""
    return create_custom_handoff_tool(
        agent_name="requirements_analyzer",
        name="handoff_to_requirements_analyzer",
        description="Transfer control to the Requirements Analyzer agent to analyze user requirements and extract infrastructure needs."
    )


# def create_handoff_to_security_n_best_practices_evaluator() -> BaseTool:
#     """Create handoff tool for security_n_best_practices_evaluator agent."""
#     return create_custom_handoff_tool(
#         agent_name="security_n_best_practices_evaluator",
#         name="handoff_to_security_n_best_practices_evaluator",
#         description="Transfer control to the security_n_best_practices_evaluator to analyze security compliance and best practices for AWS infrastructure."
#     )

def create_handoff_to_execution_planner() -> BaseTool:
    """Create handoff tool for Execution Planner agent."""
    return create_custom_handoff_tool(
        agent_name="execution_planner",
        name="handoff_to_execution_planner",
        description="Transfer control to the Execution Planner agent to create execution plans and assess risks."
    )

def create_handoff_to_planner_complete() -> BaseTool:
    """Create handoff tool to mark planning complete and return to main supervisor."""
    
    @tool
    def handoff_to_planner_complete(
        task_description: Annotated[str, "Summary of completed planning work"],
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """
        Mark planning workflow complete and return to main supervisor.
        
        Args:
            task_description: Summary of completed planning work
            state: Current state from the agent
            tool_call_id: Tool call ID for tracking
            
        Returns:
            Command to end the planning workflow
        """
        # Log handoff tool execution
        logger.log_structured(
            level="INFO",
            message="handoff_to_planner_complete called - planning workflow completing",
            extra={
                "tool_name": "handoff_to_planner_complete",
                "tool_call_id": tool_call_id,
                "task_description_length": len(task_description) if task_description else 0,
                "state_type": type(state).__name__,
                "has_planning_workflow_state": hasattr(state, 'planning_workflow_state'),
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', 'unknown') if hasattr(state, 'planning_workflow_state') else 'unknown',
                "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                "execution_complete": getattr(state.planning_workflow_state, 'execution_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                "requirements_complete": getattr(state.planning_workflow_state, 'requirements_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                "loop_counter": getattr(state.planning_workflow_state, 'loop_counter', 0) if hasattr(state, 'planning_workflow_state') else 0,
                "has_requirements_data": bool(getattr(state, "requirements_data", None)),
                "has_execution_data": bool(getattr(state, "execution_data", None)),
                "has_planning_results": bool(getattr(state, "planning_results", None)),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        tool_message = ToolMessage(
            content=f"Planning workflow completed: {task_description[:100]}..." if task_description else "Planning workflow completed successfully",
            name="handoff_to_planner_complete",
            tool_call_id=tool_call_id,
        )
        
        # Access messages directly from state
        messages = getattr(state, "messages", [])
        
        # Extract planning data from the state
        requirements_data = getattr(state, "requirements_data", None)
        execution_data = getattr(state, "execution_data", None)
        planning_results = getattr(state, "planning_results", None)
        
        # Create planner_data structure that Supervisor expects
        # Ensure all data is properly serialized to plain dicts
        planner_data = {
            "requirements_data": requirements_data.model_dump() if hasattr(requirements_data, 'model_dump') else (requirements_data if requirements_data else {}),
            "execution_data": execution_data.model_dump() if hasattr(execution_data, 'model_dump') else (execution_data if execution_data else {}),
            "planning_results": planning_results.model_dump() if hasattr(planning_results, 'model_dump') else (planning_results if planning_results else {}),
            "planning_complete": True,
            "completion_timestamp": getattr(state, "completion_timestamp", None)
        }
        
        # Log the handoff data being passed
        logger.log_structured(
            level="INFO",
            message="Planner handoff complete - passing data to supervisor",
            extra={
                "planner_data_keys": list(planner_data.keys()) if planner_data else [],
                "has_requirements_data": bool(planner_data.get("requirements_data")),
                "has_execution_data": bool(planner_data.get("execution_data")),
                "has_planning_results": bool(planner_data.get("planning_results")),
                "planning_complete": planner_data.get("planning_complete", False),
                "messages_count": len(messages),
                "task_description": task_description[:100] if task_description else "None"
            }
        )
        
        # Create the command
        command = Command(
            goto="supervisor",  # Return to parent graph (supervisor) - this is the correct way
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": None,
                "task_description": task_description,
                "status": "completed",
                "planning_complete": True,
                "planner_data": planner_data,  # Pass planning data to Supervisor
            },
        )
        
        # Log successful completion before returning
        logger.log_structured(
            level="INFO",
            message="Handoff command created successfully",
            extra={
                "tool_name": "handoff_to_planner_complete",
                "tool_call_id": tool_call_id,
                "command_goto": "supervisor",
                "command_graph": "parent",
                "planner_data_keys": list(planner_data.keys()) if planner_data else [],
                "planner_data_size": len(str(planner_data)) if planner_data else 0,
                "messages_count": len(messages),
                "update_keys": list(command.update.keys()),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return command
    
    # Add metadata for LangGraph tracking
    handoff_to_planner_complete.metadata = {
        "METADATA_KEY_HANDOFF_DESTINATION": "supervisor"
    }
    
    return handoff_to_planner_complete

def create_planner_handoff_tools() -> Dict[str, BaseTool]:
    """
    Create all planner handoff tools.
    
    Returns:
        Dictionary of handoff tools
    """
    return {
        "handoff_to_requirements_analyzer": create_handoff_to_requirements_analyzer(),
        # "handoff_to_tf_security_n_best_practices_evaluator": create_handoff_to_tf_security_n_best_practices_evaluator(),
        # "handoff_to_security_n_best_practices_evaluator": create_handoff_to_security_n_best_practices_evaluator(),
        "handoff_to_execution_planner": create_handoff_to_execution_planner(),
        "handoff_to_planner_complete": create_handoff_to_planner_complete(),
    }
