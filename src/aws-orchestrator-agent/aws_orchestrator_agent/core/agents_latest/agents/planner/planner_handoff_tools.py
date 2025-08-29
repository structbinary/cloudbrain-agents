"""
Custom Handoff Tools for Planner Sub-Supervisor.

This module implements custom handoff tools for the planner sub-supervisor that:
- Pass planning-specific context between agents
- Manage planning workflow state
- Handle planning phase transitions
- Maintain planning data across handoffs
"""

from typing import Annotated, Dict, Any, Optional
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

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
                # Pass workflow state to maintain context
                "workflow_state": getattr(state, "workflow_state", None),
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

# def create_handoff_to_tf_security_n_best_practices_evaluator() -> BaseTool:
#     """Create handoff tool for tf_security_n_best_practices_evaluator agent."""
#     return create_custom_handoff_tool(
#         agent_name="tf_security_n_best_practices_evaluator",
#         name="handoff_to_tf_security_n_best_practices_evaluator",
#         description="Transfer control to the tf_security_n_best_practices_evaluator agent to evaluate security and best practices of the AWS service."
#     )

def create_handoff_to_security_n_best_practices_evaluator() -> BaseTool:
    """Create handoff tool for security_n_best_practices_evaluator agent."""
    return create_custom_handoff_tool(
        agent_name="security_n_best_practices_evaluator",
        name="handoff_to_security_n_best_practices_evaluator",
        description="Transfer control to the security_n_best_practices_evaluator to analyze security compliance and best practices for AWS infrastructure."
    )

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
    ) -> str:
        """
        Mark planning workflow complete and return to main supervisor.
        
        Args:
            task_description: Summary of completed planning work
            state: Current state from the agent
            tool_call_id: Tool call ID for tracking
            
        Returns:
            Command to end the planning workflow
        """
        tool_message = ToolMessage(
            content="Planning workflow completed successfully",
            name="handoff_to_planner_complete",
            tool_call_id=tool_call_id,
        )
        # Access messages directly from state
        messages = getattr(state, "messages", [])
        return Command(
            goto=Command.END,
            graph=Command.PARENT,
            update={
                "messages": messages + [tool_message],
                "active_agent": None,
                "task_description": task_description,
                "status": "completed",
                "planning_complete": True,
            },
        )
    
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
        "handoff_to_security_n_best_practices_evaluator": create_handoff_to_security_n_best_practices_evaluator(),
        "handoff_to_execution_planner": create_handoff_to_execution_planner(),
        "handoff_to_planner_complete": create_handoff_to_planner_complete(),
    }
