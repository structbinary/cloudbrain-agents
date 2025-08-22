"""
Custom Supervisor Agent using langgraph-supervisor.

This module implements a supervisor agent that coordinates specialized
agents using langgraph-supervisor's create_supervisor() function.
Uses centralized Config and LLMProvider.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import langgraph-supervisor
from langgraph_supervisor import create_supervisor

# Import our custom handoff tools
from .supervisor_handoff_tools import create_handoff_tools_for_agents

from .types import (
    SupervisorState, 
    AgentType, 
    get_state_class, 
    WorkflowStatus, 
    BaseAgent, 
    AgentResponse,
    StateTransformer,
    GenerationState,
    ValidationState,
    EditorState,
    SecurityState,
    CostState
)
from .agents.base_agent import BaseSubgraphAgent
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider

# Create agent logger for supervisor
supervisor_logger = AgentLogger("SUPERVISOR")


class CustomSupervisorAgent(BaseAgent):
    """
    Custom supervisor agent that orchestrates subgraph agents.
    
    This supervisor:
    1. Manages workflow state and routing decisions
    2. Delegates tasks to specialized agents using Send()
    3. Handles human-in-the-loop interruptions
    4. Coordinates state flow between agents
    """
    
    @log_sync
    def __init__(
        self,
        agents: List[BaseSubgraphAgent],
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        name: str = "supervisor-agent"
    ):
        """
        Initialize the supervisor agent with centralized configuration.
        
        Args:
            agents: List of subgraph agents to orchestrate
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
            prompt_template: Custom prompt template for supervisor
            name: Agent name for identification
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for identification
        self._name = name
        
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
        
        # Initialize memory for human-in-the-loop first
        self.memory = MemorySaver()
        
        # Initialize agents and pass the checkpointer
        self.agents = {}
        for agent in agents:
            # Set the checkpointer for each agent
            if hasattr(agent, 'memory'):
                agent.memory = self.memory
            self.agents[agent.name] = agent
        
        # Set prompt template
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        # Initialize supervisor state
        self.supervisor_state: Optional[SupervisorState] = None
        
        # Build the supervisor graph
        self.graph = self._build_supervisor_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Validate that the graph was built correctly
        if not self.compiled_graph:
            raise ValueError("Failed to compile supervisor graph")
        
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Supervisor graph compiled successfully",
            extra={
                "graph_nodes": list(self.graph.nodes.keys()) if hasattr(self.graph, 'nodes') else "unknown",
                "compiled_graph_type": type(self.compiled_graph).__name__
            }
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message="Custom Supervisor Agent initialized successfully with centralized config",
            extra={
                "agent_count": len(agents), 
                "name": name,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "max_snapshots": self.supervisor_config["max_snapshots"]
            }
        )
    
    def _get_default_prompt(self) -> str:
        """Get the default prompt template for the supervisor following LangGraph tutorial pattern."""
        agent_names = list(self.agents.keys())
        agent_descriptions = "\n".join([f"- {name}: {self._get_agent_description(name)}" for name in agent_names])
        
        return f"""
        You are a supervisor managing specialized infrastructure agents.
        
        Available agents:
        {agent_descriptions}
        
        Your responsibilities:
        1. Analyze the user request and determine which agent(s) to delegate to
        2. Route tasks to the appropriate agent using the available transfer_to_* tools
        3. Coordinate the workflow and handle any interruptions
        4. Ensure the final result meets the user's requirements
        
        Instructions:
        - Delegate one agent at a time, do not call agents in parallel
        - Use the transfer_to_* tools to delegate tasks
        - Provide clear task descriptions to agents
        - Handle any human-in-the-loop interruptions
        - Return the final result when all work is complete
        
        Do not do any work yourself - only delegate to the appropriate agents.
        """
    
    def _get_agent_description(self, agent_name: str) -> str:
        """Get description for an agent based on its name."""
        descriptions = {
            "planner_sub_supervisor": "Analyzes requirements and creates execution plans using specialized sub-agents",
            "generation_agent": "Generates Terraform modules from scratch",
            "editor_agent": "Modifies existing Terraform configurations",
            "validation_agent": "Validates Terraform modules and configurations"
        }
        return descriptions.get(agent_name, "Specialized infrastructure agent")
    
    def _build_supervisor_graph(self) -> StateGraph:
        """Build the supervisor StateGraph using langgraph-supervisor."""
        
        # Get agent names for handoff tools
        agent_names = list(self.agents.keys())
        
        # Create custom handoff tools for all agents
        handoff_tools = create_handoff_tools_for_agents(agent_names)
        
        # Create agents list for create_supervisor
        # Each agent should be a compiled graph with a name
        agents = []
        for agent_name, agent in self.agents.items():
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Compiling agent for supervisor",
                extra={
                    "agent_name": agent_name,
                    "agent_type": type(agent).__name__,
                    "agent_has_build_graph": hasattr(agent, 'build_graph'),
                }
            )
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"About to compile agent graph",
                extra={
                    "agent_name": agent_name,
                    "agent_type": type(agent).__name__,
                    "agent_has_build_graph": hasattr(agent, 'build_graph'),
                    "agent_has_name": hasattr(agent, '_name'),
                    "agent_name_value": getattr(agent, '_name', 'unknown')
                }
            )
            
            compiled_agent = agent.build_graph().compile(
                checkpointer=self.memory,
                name=agent_name  # Set the agent name
            )
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Agent compiled successfully",
                extra={
                    "agent_name": agent_name,
                    "compiled_agent_type": type(compiled_agent).__name__,
                    "compiled_agent_has_nodes": hasattr(compiled_agent, 'nodes'),
                }
            )
            
            agents.append(compiled_agent)
        
        # Create supervisor using langgraph-supervisor
        supervisor_graph = create_supervisor(
            agents=agents,
            model=self.model,
            tools=handoff_tools,
            prompt=self.prompt_template,
            add_handoff_back_messages=True,
            output_mode="full_history"
        )
        
        # Debug: Log the supervisor graph structure
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Supervisor graph created with langgraph-supervisor",
            extra={
                "supervisor_graph_type": type(supervisor_graph).__name__,
                "supervisor_graph_has_nodes": hasattr(supervisor_graph, 'nodes'),
                "supervisor_graph_nodes": list(supervisor_graph.nodes.keys()) if hasattr(supervisor_graph, 'nodes') else "unknown",
                "handoff_tools_count": len(handoff_tools),
                "handoff_tool_names": [tool.name for tool in handoff_tools],
            }
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message="Created supervisor using langgraph-supervisor",
            extra={
                "agent_count": len(agents),
                "handoff_tools_count": len(handoff_tools),
                "agent_names": agent_names,
            }
        )
        
        return supervisor_graph
    

    

    
    def _transform_state_for_agent(self, agent_name: str, supervisor_state: SupervisorState, task_description: str) -> Dict[str, Any]:
        """
        Transform supervisor state to agent-specific state.
        
        This method is kept for compatibility but langgraph-supervisor handles
        most state management automatically through the handoff tools.
        
        Args:
            agent_name: Name of the target agent
            supervisor_state: Current supervisor state
            task_description: Task description for the agent
            
        Returns:
            Agent-specific state as dictionary
        """
        try:
            # Get the appropriate state class for this agent
            agent_type = self._get_agent_type_from_name(agent_name)
            
            # For langgraph-supervisor, the handoff tools handle most state transformation
            # This method is kept for any custom transformations we might need
            
            # Basic transformation - langgraph-supervisor handles the rest
            agent_state = {
                "user_request": task_description,
                "session_id": supervisor_state.session_id,
                "task_id": supervisor_state.task_id,
                "status": supervisor_state.status.value if hasattr(supervisor_state.status, 'value') else str(supervisor_state.status),
                "workflow_id": supervisor_state.workflow_id,
            }
            
            # Add context if available
            if supervisor_state.terraform_context:
                agent_state["context"] = supervisor_state.terraform_context
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Transformed state for {agent_name} using langgraph-supervisor pattern",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "state_keys": list(agent_state.keys()),
                    "session_id": agent_state.get("session_id"),
                    "task_id": agent_state.get("task_id"),
                }
            )
            
            return agent_state
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to transform state for {agent_name}: {e}",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "workflow_id": supervisor_state.workflow_id
                }
            )
            # Fallback to basic state
            return {
                "user_request": task_description,
                "session_id": supervisor_state.session_id,
                "task_id": supervisor_state.task_id,
            }
    
    def _get_agent_type_from_name(self, agent_name: str) -> AgentType:
        """Get agent type from agent name."""
        name_to_type = {
            "planner_sub_supervisor": AgentType.PLANNER,
            "generation_agent": AgentType.GENERATION,
            "validation_agent": AgentType.VALIDATION,
            "editor_agent": AgentType.EDITOR,
            "security_agent": AgentType.SECURITY,
            "cost_agent": AgentType.COST
        }
        return name_to_type.get(agent_name, AgentType.PLANNER)
    
    def _merge_agent_state_back_to_supervisor(self, agent_name: str, agent_state: Dict[str, Any], supervisor_state: SupervisorState) -> SupervisorState:
        """
        Merge agent state back into supervisor state.
        
        This method handles the state propagation back from subgraph agents to supervisor.
        For the planner sub-supervisor, it directly uses the agent's output_transform() result.
        
        Args:
            agent_name: Name of the agent that returned
            agent_state: State returned from the agent
            supervisor_state: Current supervisor state
            
        Returns:
            Updated supervisor state with agent results merged
        """
        try:
            agent_type = self._get_agent_type_from_name(agent_name)
            
            # Handle planner sub-supervisor specially (it manages its own state)
            if agent_type == AgentType.PLANNER:
                # The planner's output_transform() returns agent_result, but supervisor expects planner_data
                # Map the structure correctly
                supervisor_updates = {
                    "messages": agent_state.get("messages", []),
                    "planner_data": agent_state.get("agent_result", {}),
                    "status": WorkflowStatus.COMPLETED if agent_state.get("agent_status") == "completed" else WorkflowStatus.IN_PROGRESS,
                    "current_agent": None,  # Planning is complete, move to next phase
                }
                
                # Handle any questions from the planner
                if "agent_metadata" in agent_state:
                    metadata = agent_state["agent_metadata"]
                    if metadata.get("question"):
                        supervisor_updates["question"] = metadata["question"]
            else:
                # Use StateTransformer for other agents
                if agent_type == AgentType.GENERATION:
                    supervisor_updates = StateTransformer.generation_to_supervisor(GenerationState(**agent_state))
                elif agent_type == AgentType.VALIDATION:
                    supervisor_updates = StateTransformer.validation_to_supervisor(ValidationState(**agent_state))
                elif agent_type == AgentType.EDITOR:
                    supervisor_updates = StateTransformer.editor_to_supervisor(EditorState(**agent_state))
                elif agent_type == AgentType.SECURITY:
                    supervisor_updates = StateTransformer.security_to_supervisor(SecurityState(**agent_state))
                elif agent_type == AgentType.COST:
                    supervisor_updates = StateTransformer.cost_to_supervisor(CostState(**agent_state))
                else:
                    # Fallback for unknown agent types
                    supervisor_updates = {}
            
            # Merge messages (LangGraph automatically handles this, but we ensure it's done properly)
            if "messages" in agent_state:
                supervisor_updates["messages"] = agent_state["messages"]
            
            # Create updated supervisor state
            updated_state_data = supervisor_state.model_dump()
            updated_state_data.update(supervisor_updates)
            
            # Handle workflow completion
            if supervisor_updates.get("current_agent") is None:
                updated_state_data.update({
                    "status": WorkflowStatus.COMPLETED,
                    "workflow_completed_at": datetime.now(timezone.utc)
                })
            
            # Handle any errors from agents
            if "error" in agent_state:
                updated_state_data.update({
                    "error": agent_state["error"],
                    "status": WorkflowStatus.FAILED,
                    "current_agent": None
                })
                supervisor_logger.log_structured(
                    level="ERROR",
                    message=f"Agent {agent_name} returned error",
                    extra={
                        "agent_name": agent_name,
                        "error": agent_state["error"],
                        "status": WorkflowStatus.FAILED.value
                    }
                )
            
            # Create and validate the updated supervisor state
            updated_supervisor_state = SupervisorState(**updated_state_data)
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Successfully merged {agent_name} state back to supervisor",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "status": updated_supervisor_state.status.value,
                    "next_agent": supervisor_updates.get("current_agent")
                }
            )
            
            return updated_supervisor_state
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to merge {agent_name} state: {e}",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "agent_state_keys": list(agent_state.keys()) if isinstance(agent_state, dict) else "not_dict"
                }
            )
            # Return original state if merge fails
            return supervisor_state
    
    @property
    def name(self) -> str:
        """Get the name of the supervisor agent."""
        return self._name
    
    @log_async
    async def stream(
        self,
        query_or_command,
        context_id: str,
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Simplified async stream method following the reference pattern.
        
        Features:
        - Simple state management using StateTransformer
        - Clean interrupt detection with '__interrupt__' key
        - Direct status-based response formatting
        - Minimal manual state handling
        """
        supervisor_logger.log_structured(
            level="INFO",
            message=f"[stream] START",
            task_id=task_id,
            context_id=context_id,
            extra={
                "agent_name": self.__class__.__name__, 
                "query_or_command": str(query_or_command),
                "is_resume": isinstance(query_or_command, Command)
            }
        )

        # Simple init/resume handling
        if isinstance(query_or_command, Command):
            # Resume call: extract resume value and inject into state
            resume_value = query_or_command.resume
            graph_input = {
                "messages": [],
                "session_id": context_id,
                "task_id": task_id,
                "user_request": "",
                "status": "resuming",
                "resume_value": resume_value
            }
            thread_id = context_id  # Reuse for HITL resume
        else:
            # Initial call: build input state with new thread_id
            user_query = str(query_or_command)
            
            # Create the initial message with session_id and task_id in metadata
            initial_message = HumanMessage(
                content=user_query,
                additional_kwargs={
                    "session_id": context_id,
                    "task_id": task_id,
                    "user_request": user_query
                }
            )
            
            # For langgraph-supervisor, we need to use a simpler state structure
            graph_input = {
                "messages": [initial_message],
                "session_id": context_id,
                "task_id": task_id,
                "user_request": user_query,
                "status": "pending"
            }
            thread_id = str(uuid.uuid4())  # Unique per new user-initiated task

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
        step_count = 0
        
        # Debug: Log the initial graph input state
        supervisor_logger.log_structured(
            level="DEBUG",
            message=f"Initial graph input state",
            task_id=task_id,
            context_id=context_id,
            extra={
                "graph_input_type": type(graph_input).__name__,
                "session_id": graph_input.get("session_id") if isinstance(graph_input, dict) else getattr(graph_input, "session_id", None),
                "task_id": graph_input.get("task_id") if isinstance(graph_input, dict) else getattr(graph_input, "task_id", None),
                "user_request": graph_input.get("user_request") if isinstance(graph_input, dict) else getattr(graph_input, "user_request", None),
                "messages_count": len(graph_input.get("messages", [])) if isinstance(graph_input, dict) else len(getattr(graph_input, "messages", [])),
            }
        )
        
        try:
            # Use simple astream with values mode like reference
            async for item in self.compiled_graph.astream(graph_input, config, stream_mode='values'):
                step_count += 1
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message=f"[stream] step={step_count}",
                    task_id=task_id,
                    context_id=context_id,
                    extra={
                        "agent_name": self.__class__.__name__,
                        "step_count": step_count,
                        "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"
                    }
                )

                # 1. Handle human-in-the-loop interrupt (simple check like reference)
                if '__interrupt__' in item:
                    interrupt_payload = item['__interrupt__'][0].value  # dict passed to interrupt()
                    yield AgentResponse(
                        response_type='human_input',
                        is_task_complete=False,
                        require_user_input=True,
                        content=interrupt_payload.get('question', 'Input required'),
                        metadata={
                            'session_id': context_id,
                            'task_id': task_id,
                            'agent_name': self.name,
                            'step_count': step_count,
                            'status': 'input_required'
                        }
                    )
                    # Pause streaming until client resumes with feedback
                    break

                # 2. Handle normal state updates (direct status-based responses like reference)
                status = item.get('status')
                question = item.get('question')
                error = item.get('error')

                if status is not None:
                    if status == 'input_required':
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=True,
                            content=item.get('question') or 'More information needed to proceed.',
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'input_required'
                            }
                        )
                    elif status == 'error' or status == 'failed':
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=True,
                            content=item.get('question') or 'An error occurred while processing your request.',
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'failed'
                            }
                        )
                    elif status == 'completed':
                        content_data = {
                            'status': status,
                            'question': item.get('question')
                        }
                        yield AgentResponse(
                            response_type='data',
                            is_task_complete=True,
                            require_user_input=False,
                            content=content_data,
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'completed'
                            }
                        )
                    else:
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=False,
                            content=f'Processing... Status: {status}',
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'working'
                            }
                        )
                else:
                    # Default processing response
                    yield AgentResponse(
                        response_type='text',
                        is_task_complete=False,
                        require_user_input=False,
                        content='Processing...',
                        metadata={
                            'session_id': context_id,
                            'task_id': task_id,
                            'agent_name': self.name,
                            'step_count': step_count,
                            'status': 'working'
                        }
                    )
                    
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Stream execution failed: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "thread_id": thread_id,
                    "step_count": step_count
                }
            )
            yield AgentResponse(
                response_type='error',
                is_task_complete=True,
                require_user_input=False,
                content=f'Error during streaming: {str(e)}',
                error=str(e),
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'error_type': type(e).__name__,
                    'step_count': step_count
                }
            )
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"[stream] END",
            task_id=task_id,
            context_id=context_id,
            extra={
                "agent_name": self.__class__.__name__,
                "thread_id": thread_id,
                "step_count": step_count
            }
        )
    
    
    def _is_agent_return(self, item: Dict[str, Any]) -> bool:
        """Check if item represents a return from a subgraph agent."""
        # Check for agent-specific data fields that indicate completion
        agent_data_fields = [
            "planner_data", "generation_data", "validation_data", 
            "editor_data", "security_data", "cost_data"
        ]
        
        return any(field in item for field in agent_data_fields)
    
    def _extract_agent_name_from_return(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract agent name from return item."""
        # Check for agent-specific data fields
        agent_data_map = {
            "planner_data": "planner_sub_supervisor",
            "generation_data": "generation_agent", 
            "validation_data": "validation_agent",
            "editor_data": "editor_agent",
            "security_data": "security_agent",
            "cost_data": "cost_agent"
        }
        
        for field, agent_name in agent_data_map.items():
            if field in item:
                return agent_name
        
        return None

    def _has_interrupt_data(self, item: Dict[str, Any]) -> bool:
        """Check if item contains interrupt data."""
        return isinstance(item, dict) and 'interrupt_data' in item

    def _extract_interrupt_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interrupt data from item."""
        return item.get('interrupt_data', {})
    
    def _has_interrupt_context(self, item: Dict[str, Any]) -> bool:
        """Check if item contains interrupt context from agent."""
        return isinstance(item, dict) and (
            'interrupt_required' in item or 
            'interrupt_context' in item or
            item.get('interrupt_required', False)
        )
    
    def _extract_interrupt_context(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interrupt context from agent return item."""
        if 'interrupt_context' in item:
            return item['interrupt_context']
        elif 'interrupt_required' in item and item['interrupt_required']:
            # Fallback: create basic interrupt data
            return {
                "context": "agent_interrupt",
                "question": "Agent requires human input",
                "agent_data": item
            }
        else:
            return {}
    
    @log_sync
    def _extract_graph_interrupt_data(self, graph_interrupt: Exception) -> Dict[str, Any]:
        """
        Extract interrupt data from GraphInterrupt exception.
        
        Args:
            graph_interrupt: The GraphInterrupt exception
            
        Returns:
            Extracted interrupt data
        """
        try:
            # GraphInterrupt typically contains the interrupt data in its args
            if hasattr(graph_interrupt, 'args') and graph_interrupt.args:
                # The first argument is usually the interrupt data
                interrupt_arg = graph_interrupt.args[0]
                if hasattr(interrupt_arg, 'value'):
                    return interrupt_arg.value
                elif isinstance(interrupt_arg, dict):
                    return interrupt_arg
                else:
                    return {"message": str(interrupt_arg)}
            
            # Fallback to string representation
            return {"message": str(graph_interrupt)}
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="WARNING",
                message="Failed to extract interrupt data from GraphInterrupt",
                extra={"error": str(e), "graph_interrupt": str(graph_interrupt)}
            )
            return {"message": "Interrupt occurred", "error": str(graph_interrupt)}
    
    # ============================================================================
    # SIMPLIFIED UTILITY METHODS
    # ============================================================================
    
    @log_sync
    def _initialize_workflow(self, user_request: str, context_id: str, task_id: str) -> None:
        """Initialize a new workflow with the given user request."""
        self.supervisor_state = SupervisorState(
            user_request=user_request,
            session_id=context_id,
            task_id=task_id,
            status="pending",
            workflow_started_at=datetime.now()
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Initialized workflow: {self.supervisor_state.workflow_id}",
            task_id=task_id,
            context_id=context_id,
            extra={"user_request_length": len(user_request)}
        )
    
    @log_sync
    def add_agent(self, agent: BaseSubgraphAgent):
        """Add a new agent to the supervisor."""
        self.agents[agent.name] = agent
        # Rebuild graph with new agent
        self.graph = self._build_supervisor_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Added agent to supervisor: {agent.name}",
            extra={"agent_name": agent.name, "total_agents": len(self.agents)}
        )
    
    @log_sync
    def remove_agent(self, agent_name: str):
        """Remove an agent from the supervisor."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            # Rebuild graph without the agent
            self.graph = self._build_supervisor_graph()
            self.compiled_graph = self.graph.compile(checkpointer=self.memory)
            
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Removed agent from supervisor: {agent_name}",
                extra={"agent_name": agent_name, "total_agents": len(self.agents)}
            )
        else:
            supervisor_logger.log_structured(
                level="WARNING",
                message=f"Attempted to remove non-existent agent: {agent_name}",
                extra={"agent_name": agent_name}
            )
    
    @log_sync
    def get_agent_status(self, agent_name: str) -> Optional[str]:
        """Get the status of a specific agent."""
        if agent_name in self.agents:
            return "available"
        return None
    
    @log_sync
    def list_agents(self) -> List[str]:
        """List all available agents."""
        agent_list = list(self.agents.keys())
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Listed available agents",
            extra={"agent_count": len(agent_list), "agents": agent_list}
        )
        return agent_list
    
    @log_sync
    def get_agent_info(self) -> Dict[str, str]:
        """Get information about registered agents."""
        return {name: name for name in self.agents.keys()}
    
    @log_sync
    def is_ready(self) -> bool:
        """Check if the supervisor is ready for use."""
        model_ready = self.model is not None
        supervisor_ready = self.compiled_graph is not None
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
                message="Supervisor not ready: Supervisor graph not compiled",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        elif not agents_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: No agent subgraphs registered",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        
        # Return true only if all components are ready
        return model_ready and supervisor_ready and agents_ready
    
    @log_sync
    def reset_state(self) -> None:
        """Reset the supervisor state."""
        self.supervisor_state = None
        supervisor_logger.log_structured(
            level="INFO",
            message="Supervisor state reset"
        )
    
    @log_sync
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for debugging."""
        return {
            "llm_provider": self.config_instance.get_llm_config().get('provider'),
            "llm_model": self.config_instance.get_llm_config().get('model'),
            "supervisor_config": self.supervisor_config,
            "agent_count": len(self.agents),
            "ready": self.is_ready()
        }


# Factory function for easy supervisor creation
def create_supervisor_agent(
    agents: List[BaseSubgraphAgent],
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    name: str = "supervisor-agent"
) -> CustomSupervisorAgent:
    """
    Create a supervisor agent with the given agents using centralized configuration.
    
    Args:
        agents: List of subgraph agents to orchestrate
        config: Configuration instance (defaults to new Config())
        custom_config: Optional custom configuration to override defaults
        prompt_template: Custom prompt template
        name: Agent name for identification
        
    Returns:
        Configured supervisor agent
    """
    return CustomSupervisorAgent(
        agents=agents,
        config=config,
        custom_config=custom_config,
        prompt_template=prompt_template,
        name=name
    )
