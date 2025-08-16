"""
Core Supervisor Agent implementation.

This module provides the main SupervisorAgent class that orchestrates
specialized agent subgraphs using LangGraph's supervisor (tool-calling) pattern
with human-in-the-loop approval and clarification capabilities.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Annotated, TypedDict
# Add langgraph-supervisor imports
from langgraph_supervisor import create_supervisor, create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages

from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.core.agents.supervisor.supervisor_prompts import SUPERVISOR_PROMPT

from aws_orchestrator_agent.core.agents.supervisor.state import (
    StateManager,
    SupervisorState,
    AgentType,
    WorkflowStatus
)
from aws_orchestrator_agent.core.agents.supervisor.state.schemas import ( 
    GenerationState, 
    ValidationState, 
    EditorState
)
from aws_orchestrator_agent.core.agents.supervisor.state.state_transformers import StateTransformer
from aws_orchestrator_agent.core.agents.supervisor.human_interface_agent import create_human_interface_agent
from aws_orchestrator_agent.core.agents.planner import create_planner_agent
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.types import AgentResponse

# Create agent logger for supervisor
supervisor_logger = AgentLogger("SUPERVISOR")


class SupervisorAgent:
    """
    Supervisor Agent for orchestrating specialized infrastructure agents.
    
    This agent uses LangGraph's supervisor (tool-calling) pattern to route
    requests to appropriate specialized agents (Analysis, Generation, Validation, Editor)
    based on LLM-powered decision making, with robust state management via StateManager.
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "supervisor-agent"
    ):
        """
        Initialize the Supervisor Agent.
        
        Args:
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for A2A integration
        self.name = name
        
        # Get LLM configuration from centralized config
        llm_config = self.config_instance.get_llm_config()
        
        # Initialize the LLM model using the centralized provider
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model: {e}",
                extra={"error": str(e)}
            )
            raise
        
        # Get supervisor-specific configuration from centralized config
        self.supervisor_config = {
            "output_mode": self.config_instance.supervisor_output_mode,
            "add_handoff_back_messages": self.config_instance.supervisor_add_handoff_back_messages,
            "max_snapshots": self.config_instance.supervisor_max_snapshots,
            "enable_audit_trail": self.config_instance.supervisor_enable_audit_trail,
            "max_concurrent_workflows": self.config_instance.supervisor_max_concurrent_workflows,
            "workflow_timeout": self.config_instance.supervisor_workflow_timeout,
            "human_approval_required": self.config_instance.supervisor_human_approval_required,
            "validation_always_required": self.config_instance.supervisor_validation_always_required,
            "max_retries": self.config_instance.supervisor_max_retries,
            "timeout_seconds": self.config_instance.supervisor_timeout_seconds
        }
        
        # Initialize StateManager for robust state orchestration
        self.state_manager = StateManager(max_snapshots=self.supervisor_config["max_snapshots"])
        
        # Initialize supervisor state
        self.supervisor_state: Optional[SupervisorState] = None
        
        # Agent subgraphs will be registered here
        self.agents: List[Any] = []
        self.agent_names: Dict[str, str] = {}
        
        # Supervisor graph (will be created after agent registration)
        self.supervisor = None
        
        # Human-in-the-loop support
        self.memory = MemorySaver()
        self.interrupt_handlers: Dict[str, Any] = {}
        
        # Initialize flow.md components
        self.human_interface_agent = create_human_interface_agent(self.config_instance)
        

        
        supervisor_logger.log_structured(
            level="INFO",
            message="Supervisor Agent initialized successfully",
            extra={"max_snapshots": self.supervisor_config["max_snapshots"]}
        )
    
    @log_async
    async def initialize(self) -> None:
        """Initialize the supervisor agent."""
        try:
            # Check if supervisor is ready
            if not self.is_ready():
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Supervisor not ready, initializing with mock agents for testing",
                    extra={"ready": False}
                )
                
                # Create minimal mock agents for testing
                from langgraph.graph import StateGraph
                from langchain_core.messages import HumanMessage, AIMessage
                
                # Create a simple mock agent that just echoes back
                def mock_agent(state):
                    messages = state.get("messages", [])
                    if messages and isinstance(messages[-1], HumanMessage):
                        return {
                            "messages": [AIMessage(content=f"Mock response to: {messages[-1].content}")]
                        }
                    return {"messages": [AIMessage(content="No input received")]}
                
                # Create mock subgraphs for each agent type
                from typing import TypedDict, Annotated
                from langgraph.graph.message import add_messages
                
                # Define a simple state schema for mock agents
                class MockAgentState(TypedDict):
                    messages: Annotated[list, add_messages]
                    # Add fields that need to be shared with planner agent for interrupt handling
                    status: str
                    planning_metadata: dict
                    waiting_for_dependency_input: bool
                    current_dependency_question: str
                    dependency_questions: list
                    dependency_answers: dict
                    user_request: str
                    context_id: str
                    task_id: str
                    agent_name: str
                    request_type: str
                    request_classification_confidence: float
                    requirements_analysis: dict
                    infrastructure_requirements: list
                    architectural_patterns: list
                    security_requirements: list
                    execution_plan: list
                    cost_analysis: dict
                    resource_requirements: dict
                    compliance_requirements: list
                    validation_criteria: list
                    existing_infrastructure: dict
                    affected_resources: list
                    unchanged_resources: list
                    change_impact: dict
                    downtime_required: bool
                    rollback_strategy: str
                    risk_assessment: dict
                    error: str
                    error_context: dict
                    current_step: str
                    completed_steps: list
                    requires_approval: bool
                    approval_context: dict
                    dependency_mapping_complete: bool
                    planning_started_at: str
                    planning_completed_at: str
                    planning_duration: float
                    complexity_score: int
                
                mock_generation_graph = StateGraph(MockAgentState)
                mock_generation_graph.add_node("generation", mock_agent)
                mock_generation_graph.set_entry_point("generation")
                mock_generation_graph.set_finish_point("generation")
                
                mock_validation_graph = StateGraph(MockAgentState)
                mock_validation_graph.add_node("validation", mock_agent)
                mock_validation_graph.set_entry_point("validation")
                mock_validation_graph.set_finish_point("validation")
                
                mock_editor_graph = StateGraph(MockAgentState)
                mock_editor_graph.add_node("editor", mock_agent)
                mock_editor_graph.set_entry_point("editor")
                mock_editor_graph.set_finish_point("editor")
                
                # Create and register the real Planner Agent
                try:
                    planner_agent = create_planner_agent(self.config_instance, custom_config=None, name="planner_agent")
                    planner_graph = planner_agent.get_compiled_graph()
                    
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Successfully created real Planner Agent",
                        extra={"planner_agent_name": planner_agent.name, "planner_graph_ready": planner_graph is not None}
                    )
                    
                    # Register the real Planner Agent
                    self.register_agent_subgraph("planner_agent", planner_graph, "Planner Agent")
                    
                except Exception as planner_error:
                    supervisor_logger.log_structured(
                        level="ERROR",
                        message=f"Failed to create real Planner Agent, falling back to mock: {planner_error}",
                        extra={"error": str(planner_error)}
                    )
                    # Fallback to mock planner if real one fails
                    mock_planner_graph = StateGraph(MockAgentState)
                    mock_planner_graph.add_node("planner", mock_agent)
                    mock_planner_graph.set_entry_point("planner")
                    mock_planner_graph.set_finish_point("planner")
                    self.register_agent_subgraph("planner_agent", mock_planner_graph.compile(name="planner_agent"), "Planner Agent (Mock)")
                
                # Register mock agents for other agent types
                self.register_agent_subgraph("generation_agent", mock_generation_graph.compile(name="generation_agent"), "Generation Agent")
                self.register_agent_subgraph("validation_agent", mock_validation_graph.compile(name="validation_agent"), "Validation Agent")
                self.register_agent_subgraph("editor_agent", mock_editor_graph.compile(name="editor_agent"), "Editor Agent")
                
                # Create the LangGraph supervisor for proper streaming
                try:
                    self.create_supervisor()
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="LangGraph supervisor created successfully with mock agents",
                        extra={"supervisor_ready": True, "mock_agents_ready": True, "agent_count": len(self.agents)}
                    )
                except Exception as e:
                    supervisor_logger.log_structured(
                        level="ERROR",
                        message=f"Failed to create LangGraph supervisor: {e}",
                        extra={"supervisor_ready": False, "mock_agents_ready": True, "error": str(e)}
                    )
                    raise
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Supervisor agent initialized with mock agents",
                    extra={"ready": self.is_ready()}
                )
            else:
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Supervisor agent already ready",
                    extra={"ready": True}
                )
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize supervisor agent: {e}",
                extra={"error": str(e)}
            )
            raise

    @log_sync
    def initialize_workflow(self, user_request: str, mcp_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new workflow with the given user request.
        
        Args:
            user_request: The user's request or query
            mcp_context: Optional MCP context information
        """
        # Extract context_id and task_id from mcp_context
        context_id = mcp_context.get("context_id") if mcp_context else None
        task_id = mcp_context.get("task_id") if mcp_context else None
        
        self.supervisor_state = SupervisorState(
            user_request=user_request,
            mcp_context=mcp_context or {},
            context_id=context_id,
            task_id=task_id,
            workflow_started_at=datetime.utcnow()
        )
        
        # Create initial snapshot
        self.state_manager.create_snapshot(
            self.supervisor_state, 
            "Workflow initialized"
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Initialized workflow: {self.supervisor_state.workflow_id}",
            task_id=self.supervisor_state.workflow_id,
            extra={"user_request_length": len(user_request)}
        )
    
    @log_sync
    def register_agent_subgraph(
        self, 
        agent_name: str, 
        agent_subgraph: Any,
        display_name: Optional[str] = None
    ) -> None:
        """
        Register an agent subgraph with the supervisor.
        
        Args:
            agent_name: Internal name for the agent
            agent_subgraph: LangGraph subgraph instance
            display_name: Human-readable name for the agent
        """
        if agent_name in self.agent_names:
            supervisor_logger.log_structured(
                level="WARNING",
                message=f"Agent {agent_name} already registered, overwriting",
                extra={"agent_name": agent_name}
            )
        
        self.agents.append(agent_subgraph)
        # Use centralized config for agent names
        self.agent_names[agent_name] = display_name or self.config_instance.supervisor_agent_names.get(agent_name, agent_name)
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Registered agent subgraph: {agent_name} -> {self.agent_names[agent_name]}",
            extra={"agent_name": agent_name, "display_name": self.agent_names[agent_name]}
        )
    
    @log_sync
    def create_supervisor(self, prompt: Optional[str] = None) -> None:
        """
        Create supervisor using langgraph-supervisor library with comprehensive state transformation.
        
        Args:
            prompt: Custom prompt for the supervisor (defaults to SUPERVISOR_PROMPT)
        """
        if not self.agents:
            raise ValueError("No agent subgraphs registered. Call register_agent_subgraph() first.")
        
        supervisor_prompt = prompt or SUPERVISOR_PROMPT
        
        try:
            # Create supervisor with auto-generated handoff tools
            # The create_supervisor function will auto-generate:
            # - delegate_to_analysis_agent()
            # - delegate_to_generation_agent()
            # - delegate_to_validation_agent()
            # - delegate_to_editor_agent()
            # - delegate_to_planner_agent()
            
            # Note: State transformation is handled at the agent level
            # Each agent is responsible for handling the state it receives
            self.supervisor = create_supervisor(
                agents=self.agents,
                model=self.model,
                tools=[],  # No custom tools for now - use auto-generated ones
                prompt=supervisor_prompt,
                add_handoff_back_messages=self.supervisor_config["add_handoff_back_messages"],
                output_mode=self.supervisor_config["output_mode"]
            ).compile(
                checkpointer=self.memory  # Use memory for human-in-the-loop persistence
            )
            
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Created langgraph-supervisor with {len(self.agents)} agent subgraphs",
                extra={"agent_count": len(self.agents), "handoff_tools": "auto-generated"}
            )
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to create supervisor: {e}",
                extra={"error": str(e)}
            )
            raise

    
    @log_sync
    def get_workflow_progress(self) -> Dict[str, Any]:
        """
        Get detailed workflow progress information.
        
        Returns:
            Dictionary containing progress information
        """
        if not self.supervisor_state:
            return {"error": "No workflow initialized"}
        
        return self.state_manager.get_workflow_progress(self.supervisor_state)
    
    @log_sync
    def get_snapshot_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available snapshots.
        
        Returns:
            List of snapshot information dictionaries
        """
        return self.state_manager.get_snapshot_info()
    
    @log_sync
    def rollback_workflow(self, snapshot_index: int = -1) -> bool:
        """
        Rollback workflow to a previous snapshot.
        
        Args:
            snapshot_index: Index of snapshot to rollback to (default: latest)
            
        Returns:
            True if rollback was successful
        """
        if not self.supervisor_state:
            supervisor_logger.log_structured(
                level="ERROR",
                message="No workflow to rollback"
            )
            return False
        
        try:
            self.supervisor_state = self.state_manager.rollback_to_snapshot(
                self.supervisor_state, 
                snapshot_index
            )
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Successfully rolled back workflow to snapshot {snapshot_index}",
                task_id=self.supervisor_state.workflow_id
            )
            return True
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to rollback workflow: {e}",
                task_id=self.supervisor_state.workflow_id
            )
            return False
    
    @log_sync
    def export_state_for_debugging(self) -> Dict[str, Any]:
        """
        Export current state for debugging purposes.
        
        Returns:
            Dictionary containing exported state data
        """
        if not self.supervisor_state:
            return {"error": "No workflow initialized"}
        
        return self.state_manager.export_state_for_debugging(self.supervisor_state)
    
    @log_sync
    def get_routing_history(self) -> List[Dict[str, Any]]:
        """Get the routing history for audit and debugging."""
        if self.supervisor_state:
            return self.supervisor_state.routing_history.copy()
        return []
    
    @log_sync
    def get_error_context(self) -> Dict[str, Any]:
        """Get the current error context."""
        if self.supervisor_state:
            return self.supervisor_state.error_context.copy()
        return {}
    
    @log_sync
    def reset_state(self) -> None:
        """Reset the supervisor state."""
        self.supervisor_state = None
        self.state_manager.clear_snapshots()
        supervisor_logger.log_structured(
            level="INFO",
            message="Supervisor state reset"
        )
    
    def _extract_user_request(self, messages: List[Dict[str, Any]]) -> str:
        """Extract user request from messages."""
        for message in messages:
            if message.get("role") == "user":
                return message.get("content", "")
        return "No user request found"

    
    @log_sync
    def get_agent_info(self) -> Dict[str, str]:
        """Get information about registered agents."""
        return self.agent_names.copy()
    


    def _update_supervisor_state(self, item: Dict[str, Any], context_id: str, task_id: str, step_count: int) -> None:
        """
        Update supervisor state with the latest state from LangGraph using StateManager.
        
        Args:
            item: State update from supervisor.astream()
            context_id: Context identifier
            task_id: Task identifier
            step_count: Current step count
        """
        try:
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Updating supervisor state at step {step_count}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "has_supervisor_state": hasattr(self, 'supervisor_state'),
                    "supervisor_state_is_none": self.supervisor_state is None if hasattr(self, 'supervisor_state') else True,
                    "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"
                }
            )
            
            if not hasattr(self, 'supervisor_state') or not self.supervisor_state:
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="No supervisor state to update",
                    task_id=task_id,
                    context_id=context_id
                )
                return
            
            # Create a new supervisor state from the LangGraph item
            # This ensures we get the latest state with proper validation
            updated_state_data = {
                "messages": item.get('messages', []),
                "user_request": item.get('user_request', self.supervisor_state.user_request),
                "workflow_id": self.supervisor_state.workflow_id,
                "current_step": item.get('current_step', self.supervisor_state.current_step),
                "status": item.get('status', self.supervisor_state.status),
                "mcp_context": self.supervisor_state.mcp_context,
                "routing_history": self.supervisor_state.routing_history,
                "error_context": self.supervisor_state.error_context,
                "audit_log": self.supervisor_state.audit_log,
                "total_execution_time": self.supervisor_state.total_execution_time,
                "agent_execution_times": self.supervisor_state.agent_execution_times,
                "workflow_started_at": self.supervisor_state.workflow_started_at,
                "workflow_completed_at": self.supervisor_state.workflow_completed_at,
                "human_approval_required": self.supervisor_state.human_approval_required,
                "approval_context": self.supervisor_state.approval_context
            }
            
            # Preserve existing agent states
            if self.supervisor_state.analysis_state:
                updated_state_data["analysis_state"] = self.supervisor_state.analysis_state
            if self.supervisor_state.generation_state:
                updated_state_data["generation_state"] = self.supervisor_state.generation_state
            if self.supervisor_state.validation_state:
                updated_state_data["validation_state"] = self.supervisor_state.validation_state
            if self.supervisor_state.editor_state:
                updated_state_data["editor_state"] = self.supervisor_state.editor_state
            
            # Create new supervisor state with validation
            new_supervisor_state = SupervisorState(**updated_state_data)
            
            # Validate the new state using StateManager
            if not self.state_manager.validate_state(new_supervisor_state):
                raise ValueError("Invalid supervisor state after update")
            
            # Update agent states if present in the item using StateManager
            self._update_agent_states_via_manager(item, new_supervisor_state)
            
            # Replace the current state with the validated new state
            self.supervisor_state = new_supervisor_state
            
            # Create snapshot for audit trail using StateManager
            self.state_manager.create_snapshot(
                self.supervisor_state,
                f"Step {step_count} completed"
            )
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Updated supervisor state at step {step_count} via StateManager",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "step_count": step_count,
                    "message_count": len(item.get('messages', [])),
                    "status": self.supervisor_state.status,
                    "validation_passed": True
                }
            )
                
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to update supervisor state via StateManager: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={"error": str(e), "step_count": step_count}
            )
            # Try to rollback to previous snapshot if available
            try:
                if self.state_manager.get_snapshot_info():
                    self.supervisor_state = self.state_manager.rollback_to_snapshot(
                        self.supervisor_state, -1
                    )
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Rolled back to previous snapshot after state update failure",
                        task_id=task_id,
                        context_id=context_id
                    )
            except Exception as rollback_error:
                supervisor_logger.log_structured(
                    level="ERROR",
                    message=f"Failed to rollback after state update error: {rollback_error}",
                    task_id=task_id,
                    context_id=context_id
                )

    def _update_agent_states_via_manager(self, item: Dict[str, Any], supervisor_state: SupervisorState) -> None:
        """
        Update agent-specific states using StateManager's merge capabilities.
        
        Args:
            item: State update from supervisor.astream()
            supervisor_state: The supervisor state to update
        """
        try:
            
            # Update generation state if present
            if 'generation_state' in item and item['generation_state']:
                generation_state = GenerationState(**item['generation_state'])
                if self.state_manager.validate_state(generation_state):
                    supervisor_state.generation_state = generation_state
                    supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Updated generation state via StateManager",
                        extra={"generation_complete": generation_state.generation_complete}
                    )
            
            # Update validation state if present
            if 'validation_state' in item and item['validation_state']:
                validation_state = ValidationState(**item['validation_state'])
                if self.state_manager.validate_state(validation_state):
                    supervisor_state.validation_state = validation_state
                    supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Updated validation state via StateManager",
                        extra={"validation_complete": validation_state.validation_complete}
                    )
            
            # Update editor state if present
            if 'editor_state' in item and item['editor_state']:
                editor_state = EditorState(**item['editor_state'])
                if self.state_manager.validate_state(editor_state):
                    supervisor_state.editor_state = editor_state
                    supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Updated editor state via StateManager",
                        extra={"editor_complete": editor_state.editor_complete}
                    )
                        
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to update agent states via StateManager: {e}",
                extra={"error": str(e)}
            )

    def _format_stream_response(self, item, context_id: str, task_id: str, step_count: int) -> AgentResponse:
        """
        Format streaming response with proper metadata and state information.
        
        Args:
            item: Stream item from supervisor.astream()
            context_id: Context identifier
            task_id: Task identifier
            step_count: Current step count
            
        Returns:
            Formatted AgentResponse
        """
        status = item.get('status')
        question = item.get('question')
        final_response = item.get('final_response')

        # Get current workflow progress
        workflow_progress = self.get_workflow_progress() if hasattr(self, 'supervisor_state') and self.supervisor_state else {}

        if status == 'completed':
            return AgentResponse(
                response_type='data',
                is_task_complete=True,
                require_user_input=False,
                content={
                    'status': status,
                    'question': question,
                    'final_response': final_response,
                    'workflow_progress': workflow_progress
                },
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'step_count': step_count,
                    'status': 'completed',
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if hasattr(self, 'supervisor_state') and self.supervisor_state else None
                }
            )
        elif status == 'error' or status == 'failed':
            return AgentResponse(
                response_type='text',
                is_task_complete=False,
                require_user_input=True,
                content=question or 'An error occurred while processing your request.',
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'step_count': step_count,
                    'status': 'failed',
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if hasattr(self, 'supervisor_state') and self.supervisor_state else None
                }
            )
        else:
            return AgentResponse(
                response_type='text',
                is_task_complete=False,
                require_user_input=False,
                content=f'Processing... Status: {status}' if status else 'Processing...',
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'step_count': step_count,
                    'status': 'working',
                    'workflow_progress': workflow_progress,
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if hasattr(self, 'supervisor_state') and self.supervisor_state else None
                }
            )


    
    @log_sync
    def is_ready(self) -> bool:
        """Check if the supervisor is ready for use."""
        model_ready = self.model is not None
        supervisor_ready = self.supervisor is not None
        agents_ready = len(self.agents) > 0
        
        # Log detailed status for debugging
        if not model_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: LLM model not initialized",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        elif not supervisor_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: Supervisor graph not created (call create_supervisor() after registering agents)",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        elif not agents_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: No agent subgraphs registered (call register_agent_subgraph() first)",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        
        # Return true only if all components are ready
        return model_ready and supervisor_ready and agents_ready

    @log_async
    async def stream(
        self,
        query_or_command,
        context_id: str,
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Main async entry point for all user operations.
        Uses astream() as the core execution method with automatic human-in-the-loop.
        Agents will trigger interrupts when they need human feedback using the interrupt() function.
        """
        supervisor_logger.log_structured(
            level="INFO",
            message=f"[stream] START",
            task_id=task_id,
            context_id=context_id,
            extra={"agent_name": self.__class__.__name__, "query_or_command": str(query_or_command)}
        )

        # 1. Initialize supervisor if not ready
        if not self.is_ready():
            await self.initialize()
        
        # 2. Initialize workflow for new queries
        if isinstance(query_or_command, str):
            mcp_context = {"context_id": context_id, "task_id": task_id}
            self.initialize_workflow(query_or_command, mcp_context)

        # 3. Prepare input state
        if isinstance(query_or_command, Command):
            # Resume call: extract resume value and inject into state, reuse session_id as thread_id
            resume_value = query_or_command.resume
            graph_input = {
                "messages": [],
                "user_request": "",
                "context_id": context_id,
                "task_id": task_id,
                "mcp_context": {"context_id": context_id, "task_id": task_id},
                "resume_value": resume_value
            }
            thread_id = context_id  # Reuse for HITL resume
        else:
            # Initial call: build input state with proper message handling
            user_query = query_or_command
            
            # Create a message that includes context information
            context_message = HumanMessage(content=f"Context ID: {context_id}, Task ID: {task_id}\n\n{user_query}")
            
            graph_input = {
                "messages": [context_message],
                "user_request": user_query,
                "context_id": context_id,
                "task_id": task_id,
                "mcp_context": {"context_id": context_id, "task_id": task_id},
                "workflow_id": str(uuid.uuid4()),
                "current_step": "initialized",
                "status": "pending"
            }
            thread_id = str(uuid.uuid4())  # Unique per new user-initiated task

        config = {'configurable': {'thread_id': thread_id}}
        step_count = 0
        
        # 4. Execute using astream() - agents will trigger interrupts when needed
        try:
            async for item in self.supervisor.astream(graph_input, config, stream_mode='values'):
                step_count += 1
                # Debug: Log the detailed structure of the item
                supervisor_logger.log_structured(
                    level="INFO",
                    message=f"[stream] step={step_count} item={item}",
                    task_id=task_id,
                    context_id=context_id,
                    extra={
                        "agent_name": self.__class__.__name__, 
                        "step_count": step_count, 
                        "item": str(item),
                        "item_type": type(item).__name__,
                        "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict",
                        "item_attributes": [attr for attr in dir(item) if not attr.startswith('_')] if not isinstance(item, dict) else "dict"
                    }
                )

                # Handle interrupts (when agents need human feedback)
                # With langgraph_supervisor, interrupts are detected through messages field
                interrupt_detected = False
                interrupt_payload = None
                
                # Check if the item contains interrupt information in the messages
                if isinstance(item, dict) and 'messages' in item:
                    # Look for interrupt messages in the messages list
                    for message in item['messages']:
                        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                            # Check if this message contains interrupt data
                            if 'interrupt_data' in message.additional_kwargs:
                                interrupt_detected = True
                                interrupt_payload = message.additional_kwargs['interrupt_data']
                                break
                            elif 'interrupt_type' in message.additional_kwargs:
                                interrupt_detected = True
                                interrupt_payload = {
                                    'question': message.additional_kwargs.get('current_dependency_question', 'Dependency mapping question'),
                                    'context': message.additional_kwargs.get('interrupt_type', 'dependency_mapping'),
                                    'available_questions': message.additional_kwargs.get('dependency_questions', []),
                                    'partial_analysis': message.additional_kwargs.get('planning_metadata', {}).get('dependency_mapping_partial', {})
                                }
                                break
                
                # Fallback: Check if the item contains interrupt information in the state
                if not interrupt_detected and isinstance(item, dict):
                    # Check for interrupt information in the state
                    if 'status' in item and item['status'] == 'interrupted':
                        interrupt_detected = True
                        interrupt_payload = item.get('planning_metadata', {}).get('interrupt_data', {})
                    
                    # Also check for dependency mapping state
                    elif item.get('waiting_for_dependency_input', False):
                        interrupt_detected = True
                        interrupt_payload = {
                            'question': item.get('current_dependency_question', 'Dependency mapping question'),
                            'context': 'dependency_mapping',
                            'available_questions': item.get('dependency_questions', []),
                            'partial_analysis': item.get('planning_metadata', {}).get('dependency_mapping_partial', {})
                        }
                
                # Legacy check for __interrupt__ (fallback)
                if not interrupt_detected and '__interrupt__' in item:
                    interrupt_detected = True
                    interrupt_payload = item['__interrupt__'][0].value
                
                if interrupt_detected and interrupt_payload:
                    # Log the interrupt for audit trail
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Agent requires human feedback",
                        task_id=task_id,
                        context_id=context_id,
                        extra={
                            "interrupt_payload": str(interrupt_payload),
                            "interrupt_type": interrupt_payload.get('context', 'unknown'),
                            "step_count": step_count
                        }
                    )
                    
                    # Handle different types of interrupts
                    interrupt_type = interrupt_payload.get('context', 'unknown')
                    
                    if interrupt_type == 'dependency_mapping':
                        # Handle dependency mapping questions
                        yield AgentResponse(
                            response_type='dependency_question',
                            is_task_complete=False,
                            require_user_input=True,
                            content=interrupt_payload.get('question', 'Dependency mapping question'),
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'dependency_mapping_question',
                                'interrupt_type': interrupt_type,
                                'available_questions': interrupt_payload.get('available_questions', []),
                                'partial_analysis': interrupt_payload.get('partial_analysis', {}),
                                'thread_id': thread_id
                            }
                        )
                    elif interrupt_type == 'approval_required':
                        # Handle approval requests
                        yield AgentResponse(
                            response_type='approval_required',
                            is_task_complete=False,
                            require_user_input=True,
                            content=interrupt_payload.get('message', 'Approval required'),
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'approval_required',
                                'interrupt_type': interrupt_type,
                                'node': interrupt_payload.get('node', 'unknown'),
                                'thread_id': thread_id
                            }
                        )
                    else:
                        # Generic interrupt handling
                        yield AgentResponse(
                            response_type='human_input',
                            is_task_complete=False,
                            require_user_input=True,
                            content=interrupt_payload.get('question', 'Agent requires human feedback'),
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'agent_requires_feedback',
                                'interrupt_type': interrupt_type,
                                'thread_id': thread_id
                            }
                        )
                    
                    break  # Pause until human responds

                # Handle normal state updates
                # Update our supervisor state with the latest state from LangGraph
                if isinstance(item, dict) and 'messages' in item:
                    # Extract and update supervisor state
                    self._update_supervisor_state(item, context_id, task_id, step_count)
                
                yield self._format_stream_response(item, context_id, task_id, step_count)
                
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Stream execution failed: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={"error": str(e)}
            )
            yield AgentResponse(
                response_type='error',
                is_task_complete=True,
                require_user_input=False,
                content=f'Supervisor execution failed: {str(e)}',
                error=str(e),
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'status': 'failed'
                }
            )
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"DEBUG: [stream] END session_id={context_id}, task_id={task_id}",
            task_id=task_id,
            context_id=context_id,
            extra={"agent_name": self.__class__.__name__}
        )
    
    @log_async
    async def handle_dependency_answer(self, context_id: str, task_id: str, question: str, answer: str) -> Dict[str, Any]:
        """
        Handle a dependency mapping answer and resume the workflow.
        
        This method is called when a user provides an answer to a dependency mapping question.
        It updates the state and resumes the workflow using LangGraph's Command(resume=...) pattern.
        
        Args:
            context_id: The context ID for the workflow
            task_id: The task ID for the workflow
            question: The question that was answered
            answer: The user's answer
            
        Returns:
            Dict containing the result of the operation
        """
        try:
            supervisor_logger.log_structured(
                level="INFO",
                message="Handling dependency answer",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "question": question,
                    "answer": answer,
                    "agent_name": self.name
                }
            )
            
            # Create a Command object to resume the workflow
            from langgraph.types import Command
            
            # The resume value should contain the answer in a format the dependency mapping node expects
            resume_value = {
                "question": question,
                "answer": answer,
                "context_id": context_id,
                "task_id": task_id
            }
            
            # Create the command to resume the workflow
            resume_command = Command(resume=resume_value)
            
            # Resume the workflow by calling stream with the resume command
            # This will continue from where the interrupt occurred
            result = None
            async for response in self.stream(resume_command, context_id, task_id):
                result = response
                # For now, we'll just get the first response
                # In a real implementation, you might want to stream all responses
                break
            
            supervisor_logger.log_structured(
                level="INFO",
                message="Successfully resumed workflow with dependency answer",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "question": question,
                    "answer": answer,
                    "result_type": type(result).__name__ if result else "None"
                }
            )
            
            return {
                "status": "success",
                "message": "Workflow resumed successfully",
                "question": question,
                "answer": answer,
                "context_id": context_id,
                "task_id": task_id,
                "result": result.dict() if result else None
            }
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to handle dependency answer: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "question": question,
                    "answer": answer,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            return {
                "status": "error",
                "message": f"Failed to handle dependency answer: {e}",
                "error_type": type(e).__name__,
                "context_id": context_id,
                "task_id": task_id
            }


# Factory function for easy creation
@log_sync
def create_supervisor_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "supervisor-agent"
) -> SupervisorAgent:
    """
    Factory function to create a Supervisor Agent.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Configured SupervisorAgent instance
    """
    return SupervisorAgent(config=config, custom_config=custom_config, name=name) 