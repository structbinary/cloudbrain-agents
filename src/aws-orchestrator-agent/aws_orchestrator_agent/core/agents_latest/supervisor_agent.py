"""
Custom Supervisor Agent for orchestrating subgraph agents.

This module implements a custom supervisor agent that coordinates specialized
agents (Generation, Editor, Validation, etc.) using LangGraph's Send() pattern
and inherits from the BaseAgent interface. Uses centralized Config and LLMProvider.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from .types import SupervisorState, AgentType, get_state_class, WorkflowStatus, BaseAgent, AgentResponse
from .agents.base_agent import BaseSubgraphAgent
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider

# Create agent logger for supervisor
supervisor_logger = AgentLogger("SUPERVISOR")


def create_handoff_tool(agent_name: str, supervisor_instance):
    """Create a handoff tool for a specific agent."""
    
    def handoff_tool(
        task_description: str,
        state: Annotated[Dict[str, Any], InjectedState]
    ) -> Command:
        """
        Transfer task to agent using Send() primitive following LangGraph tutorial.
        
        This ensures:
        1. Agent receives transformed state via Send()
        2. Agent returns to supervisor via Command.PARENT
        3. Agent state is automatically merged back into supervisor state
        """
        # Create task description message (following tutorial pattern)
        task_description_message = {"role": "user", "content": task_description}
        
        # Build a safe SupervisorState from injected state (may be partial)
        try:
            safe_supervisor_state = SupervisorState(
                messages=state.get("messages", []),
                user_request=state.get("user_request") or task_description,
                session_id=state.get("session_id") or getattr(supervisor_instance.supervisor_state, "session_id", None),
                task_id=state.get("task_id") or getattr(supervisor_instance.supervisor_state, "task_id", None),
            )
        except Exception:
            # Fallback minimal state
            safe_supervisor_state = SupervisorState(
                messages=[],
                user_request=task_description,
                session_id=getattr(supervisor_instance.supervisor_state, "session_id", None),
                task_id=getattr(supervisor_instance.supervisor_state, "task_id", None),
            )

        # Transform supervisor state to agent-specific state
        agent_state = supervisor_instance._transform_state_for_agent(agent_name, safe_supervisor_state, task_description)
        # Ensure session/task identifiers propagate to child state where names differ
        if agent_name == "planner_agent":
            agent_state["session_id"] = getattr(safe_supervisor_state, "session_id", None)
            agent_state["task_id"] = getattr(safe_supervisor_state, "task_id", None)
        
        # Ensure agent_state has the task description message
        if "messages" not in agent_state:
            agent_state["messages"] = []
        agent_state["messages"].insert(0, task_description_message)
        
        supervisor_logger.log_structured(
            level="DEBUG",
            message=f"Handing off to {agent_name}",
            extra={
                "agent_name": agent_name,
                "task_description": task_description,
                "state_keys": list(agent_state.keys())
            }
        )
        
        return Command(
            goto=[Send(agent_name, agent_state)],
            graph=Command.PARENT  # This ensures return to supervisor
        )
    
    # Create the tool manually without decorator
    
    return StructuredTool.from_function(
        func=handoff_tool,
        name=f"transfer_to_{agent_name}",
        description=f"Transfer task to {agent_name}"
    )


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
        
        # Initialize agents
        self.agents = {agent.name: agent for agent in agents}
        
        # Set prompt template
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        # Initialize memory for human-in-the-loop
        self.memory = MemorySaver()
        
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
            "planner_agent": "Analyzes requirements and creates execution plans",
            "generation_agent": "Generates Terraform modules from scratch",
            "editor_agent": "Modifies existing Terraform configurations",
            "validation_agent": "Validates Terraform modules and configurations"
        }
        return descriptions.get(agent_name, "Specialized infrastructure agent")
    
    def _build_supervisor_graph(self) -> StateGraph:
        """Build the supervisor StateGraph following LangGraph tutorial pattern."""
        
        # Create the supervisor graph with SupervisorState
        graph = StateGraph(SupervisorState)
        
        # Create handoff tools for each agent
        handoff_tools = self._create_handoff_tools()
        
        # Create supervisor agent using create_react_agent (following tutorial)
        from langgraph.prebuilt import create_react_agent
        
        supervisor_agent = create_react_agent(
            model=self.model,  # Use the centralized LLM model
            tools=handoff_tools,
            prompt=self.prompt_template,
            name="supervisor"
        )
        
        # Add the supervisor agent node
        graph.add_node("supervisor", supervisor_agent)
        
        # Add subgraph nodes for each agent as compiled runnables using the SAME checkpointer
        # Ensure agents reuse supervisor memory so interrupts can pause/resume across boundaries
        for agent_name, agent in self.agents.items():
            try:
                # Propagate supervisor checkpointer to subgraph agent if present
                if hasattr(agent, 'memory'):
                    agent.memory = self.memory
            except Exception:
                pass
            graph.add_node(agent_name, agent.build_graph().compile(checkpointer=self.memory))
        
        # Set entry point
        graph.set_entry_point("supervisor")
        
        # Add edges from supervisor to agents
        for agent_name in self.agents.keys():
            graph.add_edge("supervisor", agent_name)
        
        # Add edges from agents back to supervisor
        for agent_name in self.agents.keys():
            graph.add_edge(agent_name, "supervisor")
        
        return graph
    

    
    def _create_handoff_tools(self) -> List[Callable]:
        """
        Create handoff tools for each agent following LangGraph tutorial pattern.
        
        Returns:
            List of handoff tool functions
        """
        tools = []
        
        for agent_name, agent in self.agents.items():
            tools.append(create_handoff_tool(agent_name, self))
        
        return tools
    
    def _transform_state_for_agent(self, agent_name: str, supervisor_state: SupervisorState, task_description: str) -> Dict[str, Any]:
        """
        Transform supervisor state to agent-specific state using proper state schemas.
        
        Args:
            agent_name: Name of the target agent
            supervisor_state: Current supervisor state
            task_description: Task description for the agent
            
        Returns:
            Agent-specific state as dictionary
        """
        # Get the appropriate state class for this agent
        agent_type = self._get_agent_type_from_name(agent_name)
        state_class = get_state_class(agent_type)
        
        # Create base state data with proper defaults
        base_state_data = {
            "messages": supervisor_state.messages,
            "user_request": task_description,
            "workflow_id": supervisor_state.workflow_id,
            # Use context_id key consistently for child states
            "session_id": supervisor_state.session_id,
            "task_id": supervisor_state.task_id,
            # Ensure all required fields have proper defaults
            "approval_context": getattr(supervisor_state, 'approval_context', {}) or {},
            "planning_metadata": getattr(supervisor_state, 'planning_metadata', {}) or {},
            "dependency_answers": getattr(supervisor_state, 'dependency_answers', {}) or {},
            "risk_assessment": getattr(supervisor_state, 'risk_assessment', {}) or {},
            "resource_requirements": getattr(supervisor_state, 'resource_requirements', {}) or {},
            "cost_analysis": getattr(supervisor_state, 'cost_analysis', {}) or {},
            "handoff_context": getattr(supervisor_state, 'handoff_context', {}) or {}
        }
        
        # Add agent-specific transformations
        if agent_type == AgentType.PLANNER:
            # Planner needs additional context for dependency mapping
            base_state_data.update({
                "mcp_context": getattr(supervisor_state, 'mcp_context', {}),
                "planning_started_at": datetime.now(timezone.utc)
            })
        
        elif agent_type == AgentType.GENERATION:
            # Generation agent needs requirements from supervisor state
            base_state_data.update({
                "requirements": getattr(supervisor_state, 'terraform_context', {}),
                "module_name": f"module_{supervisor_state.workflow_id[:8]}",
                "module_version": "1.0.0"
            })
        
        elif agent_type == AgentType.VALIDATION:
            # Validation agent needs module references
            base_state_data.update({
                "module_ref": supervisor_state.generated_module_ref,
                "workspace_ref": supervisor_state.workspace_ref
            })
        
        elif agent_type == AgentType.EDITOR:
            # Editor agent needs target config and change request
            base_state_data.update({
                "target_config_ref": supervisor_state.generated_module_ref,
                "change_request": getattr(supervisor_state, 'terraform_context', {})
            })
        
        # Create and validate the agent state
        try:
            # Clean up any None values that might cause validation issues
            cleaned_state_data = {}
            for key, value in base_state_data.items():
                if value is None:
                    # Use appropriate defaults based on field type
                    if key in ['approval_context', 'planning_metadata', 'dependency_answers', 
                              'risk_assessment', 'resource_requirements', 'cost_analysis', 'handoff_context']:
                        cleaned_state_data[key] = {}
                    elif key in ['dependency_questions', 'architectural_patterns', 'security_requirements', 
                                'compliance_requirements', 'execution_plan', 'validation_criteria', 'completed_steps']:
                        cleaned_state_data[key] = []
                    else:
                        cleaned_state_data[key] = value
                else:
                    cleaned_state_data[key] = value
            
            agent_state = state_class(**cleaned_state_data)
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Transformed state for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "state_class": state_class.__name__,
                    "workflow_id": supervisor_state.workflow_id
                }
            )
            return agent_state.model_dump()
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to transform state for {agent_name}: {e}",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "error": str(e),
                    "state_data_keys": list(base_state_data.keys())
                }
            )
            # Fallback to basic state with cleaned data
            return cleaned_state_data
    
    def _get_agent_type_from_name(self, agent_name: str) -> AgentType:
        """Get agent type from agent name."""
        name_to_type = {
            "planner_agent": AgentType.PLANNER,
            "generation_agent": AgentType.GENERATION,
            "validation_agent": AgentType.VALIDATION,
            "editor_agent": AgentType.EDITOR,
            "security_agent": AgentType.SECURITY,
            "cost_agent": AgentType.COST
        }
        return name_to_type.get(agent_name, AgentType.PLANNER)
    
    def _merge_agent_state_back_to_supervisor(self, agent_name: str, agent_state: Dict[str, Any], supervisor_state: SupervisorState) -> SupervisorState:
        """
        Merge agent state back into supervisor state following LangGraph patterns.
        
        This method handles the state propagation back from subgraph agents to supervisor,
        as described in the LangGraph Agent Supervisor tutorial.
        
        Args:
            agent_name: Name of the agent that returned
            agent_state: State returned from the agent
            supervisor_state: Current supervisor state
            
        Returns:
            Updated supervisor state with agent results merged
        """
        agent_type = self._get_agent_type_from_name(agent_name)
        
        # Start with current supervisor state
        updated_state_data = supervisor_state.model_dump()
        
        # Merge messages (LangGraph automatically handles this, but we ensure it's done properly)
        if "messages" in agent_state:
            # Append agent messages to supervisor messages
            updated_state_data["messages"].extend(agent_state["messages"])
        
        # Agent-specific state merging based on agent type
        if agent_type == AgentType.PLANNER:
            # Merge planner results
            if "planning_complete" in agent_state and agent_state["planning_complete"]:
                updated_state_data.update({
                    "terraform_context": agent_state.get("handoff_context", {}),
                    "current_agent": AgentType.GENERATION,  # Next agent
                    "status": WorkflowStatus.IN_PROGRESS
                })
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Planner completed, moving to generation",
                    extra={
                        "agent_name": agent_name,
                        "planning_complete": True,
                        "next_agent": AgentType.GENERATION.value
                    }
                )
        
        elif agent_type == AgentType.GENERATION:
            # Merge generation results
            if "generation_complete" in agent_state and agent_state["generation_complete"]:
                updated_state_data.update({
                    "generated_module_ref": agent_state.get("generated_module_ref"),
                    "current_agent": AgentType.VALIDATION,  # Next agent
                    "status": WorkflowStatus.IN_PROGRESS
                })
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Generation completed, moving to validation",
                    extra={
                        "agent_name": agent_name,
                        "generation_complete": True,
                        "next_agent": AgentType.VALIDATION.value,
                        "module_ref": agent_state.get("generated_module_ref")
                    }
                )
        
        elif agent_type == AgentType.VALIDATION:
            # Merge validation results
            if "validation_complete" in agent_state and agent_state["validation_complete"]:
                updated_state_data.update({
                    "validation_report_ref": agent_state.get("validation_report_ref"),
                    "current_agent": None,  # Workflow complete
                    "status": WorkflowStatus.COMPLETED,
                    "workflow_completed_at": datetime.now(timezone.utc)
                })
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Validation completed, workflow finished",
                    extra={
                        "agent_name": agent_name,
                        "validation_complete": True,
                        "status": WorkflowStatus.COMPLETED.value
                    }
                )
        
        elif agent_type == AgentType.EDITOR:
            # Merge editor results
            if "editor_complete" in agent_state and agent_state["editor_complete"]:
                updated_state_data.update({
                    "generated_module_ref": agent_state.get("modified_config_ref"),
                    "current_agent": AgentType.VALIDATION,  # Next agent
                    "status": WorkflowStatus.IN_PROGRESS
                })
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Editor completed, moving to validation",
                    extra={
                        "agent_name": agent_name,
                        "editor_complete": True,
                        "next_agent": AgentType.VALIDATION.value
                    }
                )
        
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
        try:
            updated_supervisor_state = SupervisorState(**updated_state_data)
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Successfully merged {agent_name} state back to supervisor",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "status": updated_supervisor_state.status.value
                }
            )
            return updated_supervisor_state
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to merge {agent_name} state: {e}",
                extra={
                    "agent_name": agent_name,
                    "error": str(e)
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
        Enhanced async stream method with comprehensive human-in-the-loop support.
        
        Features:
        - Stable session management with consistent thread_id
        - Multi-layered interrupt detection
        - Proper Command resume handling
        - Enhanced state persistence with checkpointer
        - Comprehensive AgentResponse formatting
        - Complete audit trail for interrupt/resume cycles
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

        # 1-3. Minimal init/resume handling using explicit SupervisorState
        is_resume = isinstance(query_or_command, Command)
        if is_resume:
            # Resume: build minimal SupervisorState and append resume_value; reuse session_id as thread_id
            resume_value = query_or_command.resume
            
            # Handle different types of resume data
            if isinstance(resume_value, dict) and resume_value.get('context') == 'dependency_mapping':
                # This is a dependency mapping resume
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Resuming dependency mapping workflow",
                    task_id=task_id,
                    context_id=context_id,
                    extra={
                        "resume_context": "dependency_mapping",
                        "question": resume_value.get('question'),
                        "answer": resume_value.get('answer')
                    }
                )
            
            state = SupervisorState(
                messages=[],
                user_request="",
                session_id=context_id,
                task_id=task_id,
            )
            graph_input = {**state.model_dump(), "resume_value": resume_value}
            thread_id = context_id
            # Ensure supervisor_state is available for downstream tools
            self.supervisor_state = state
        else:
            # Fresh start: add initial HumanMessage and mint a new thread_id
            user_query = str(query_or_command)
            state = SupervisorState(
                messages=[HumanMessage(content=user_query)],
                user_request=user_query,
                session_id=context_id,
                task_id=task_id,
                workflow_started_at=datetime.now(timezone.utc)
            )
            graph_input = state.model_dump()
            thread_id = str(uuid.uuid4())
            # Ensure supervisor_state is available for downstream tools
            self.supervisor_state = state

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
        step_count = 0
        
        # 4. Execute using astream_events() to capture interrupts reliably across subgraphs
        try:
            async for event in self.compiled_graph.astream_events(graph_input, config):
                # Uniform event logging
                try:
                    supervisor_logger.log_structured(
                        level="DEBUG",
                        message="[events] step",
                        task_id=task_id,
                        context_id=context_id,
                        extra={
                            "agent_name": self.__class__.__name__,
                            "event": event.get('event'),
                            "name": event.get('name'),
                            "tags": event.get('tags'),
                            "keys": list((event.get('data') or {}).keys()) if isinstance(event.get('data'), dict) else "no-data",
                            "thread_id": thread_id
                        }
                    )
                except Exception:
                    pass

                # Detect interrupt events
                ev = (event.get('event') or '').lower()
                if 'interrupt' in ev:
                    payload = {}
                    data = event.get('data') or {}
                    # LangGraph often stores the interrupt in data.value
                    val = data.get('value') if isinstance(data, dict) else None
                    if isinstance(val, dict):
                        payload = val
                    elif isinstance(data, dict):
                        payload = data
                    self._log_interrupt_detected(payload, step_count, context_id, task_id, thread_id)
                    yield AgentResponse(
                        response_type='human_input',
                        is_task_complete=False,
                        require_user_input=True,
                        content=(payload.get('question') if isinstance(payload, dict) else 'Input required'),
                        metadata={
                            'session_id': context_id,
                            'task_id': task_id,
                            'agent_name': self.name,
                            'step_count': step_count,
                            'status': 'input_required',
                            'thread_id': thread_id
                        }
                    )
                    return

                # Track values and update state on on_state or on_value events
                if ev.endswith(':value') or ev.endswith('on_chain_end') or ev.endswith('on_node_end'):
                    step_count += 1
                    data = event.get('data') or {}
                    item = data.get('value', {}) if isinstance(data, dict) else {}
                    if isinstance(item, dict):
                        # Check for interrupt context in agent returns
                        if self._has_interrupt_context(item):
                            interrupt_payload = self._extract_interrupt_context(item)
                            self._log_interrupt_detected(interrupt_payload, step_count, context_id, task_id, thread_id)
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
                                    'status': 'input_required',
                                    'thread_id': thread_id,
                                    'interrupt_type': interrupt_payload.get('context', 'unknown')
                                }
                            )
                            return
                        
                        if self._is_agent_return(item):
                            agent_name = self._extract_agent_name_from_return(item)
                            if agent_name and self.supervisor_state:
                                self.supervisor_state = self._merge_agent_state_back_to_supervisor(agent_name, item, self.supervisor_state)
                        self._update_supervisor_state(item, context_id, task_id, step_count)
                    response = self._format_stream_response(item if isinstance(item, dict) else {}, context_id, task_id, step_count)
                    yield response
                
                
                
        except Exception as e:
            # Interrupt-aware handling: surface GraphInterrupt as human_input instead of error
            if hasattr(e, '__class__') and ('GraphInterrupt' in e.__class__.__name__ or 'Interrupt' in e.__class__.__name__):
                interrupt_payload = self._extract_graph_interrupt_data(e)
                self._log_interrupt_detected(interrupt_payload, step_count, context_id, task_id, thread_id)
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
                        'status': 'input_required',
                        'thread_id': thread_id
                    }
                )
                return
            # 9. Enhanced error handling with recovery
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
            yield await self._handle_stream_error(e, context_id, task_id, thread_id)
        
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
    
    # ============================================================================
    # ENHANCED SESSION MANAGEMENT METHODS
    # ============================================================================
    
    # _get_or_create_thread_id removed (unused)
    
    # _prepare_initial_state removed (unused; inlined in stream)
    
    # _prepare_resume_state removed (unused; inlined in stream)
    
    # _extract_resume_context removed (unused)
    
    # ============================================================================
    # ENHANCED INTERRUPT DETECTION METHODS
    # ============================================================================
    
    # _detect_interrupts removed (unused)
    
    # _has_agent_interrupt removed (unused)
    
    # _extract_agent_interrupt removed (unused)
    
    # _requires_human_approval removed (unused)
    
    # _extract_approval_context removed (unused)
    
    # _has_dependency_interrupt removed (unused)
    
    # _extract_dependency_interrupt removed (unused)
    
    def _is_agent_return(self, item: Dict[str, Any]) -> bool:
        """Check if item represents a return from a subgraph agent."""
        # Check for agent completion flags
        completion_flags = [
            "planning_complete", "generation_complete", "validation_complete", 
            "editor_complete", "security_complete", "cost_complete"
        ]
        
        return any(flag in item for flag in completion_flags)
    
    def _extract_agent_name_from_return(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract agent name from return item."""
        # Check for agent-specific completion flags
        agent_completion_map = {
            "planning_complete": "planner_agent",
            "generation_complete": "generation_agent", 
            "validation_complete": "validation_agent",
            "editor_complete": "editor_agent",
            "security_complete": "security_agent",
            "cost_complete": "cost_agent"
        }
        
        for flag, agent_name in agent_completion_map.items():
            if flag in item:
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
    # ENHANCED LOGGING METHODS
    # ============================================================================
    
    @log_sync
    def _log_stream_step(self, step_count: int, item: Any, context_id: str, task_id: str, thread_id: str) -> None:
        """Enhanced logging for stream steps."""
        supervisor_logger.log_structured(
            level="DEBUG",
            message=f"[stream] step={step_count}",
            task_id=task_id,
            context_id=context_id,
            extra={
                "agent_name": self.__class__.__name__, 
                "step_count": step_count, 
                "item_type": type(item).__name__,
                "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict",
                "thread_id": thread_id
            }
        )
    
    @log_sync
    def _log_interrupt_detected(self, interrupt_payload: Dict[str, Any], step_count: int, 
                              context_id: str, task_id: str, thread_id: str) -> None:
        """Log interrupt detection for audit trail."""
        supervisor_logger.log_structured(
            level="INFO",
            message="Agent requires human feedback",
            task_id=task_id,
            context_id=context_id,
            extra={
                "interrupt_payload": str(interrupt_payload),
                "interrupt_type": interrupt_payload.get('context', 'unknown'),
                "step_count": step_count,
                "thread_id": thread_id
            }
        )
    
    # ============================================================================
    # ENHANCED STATE MANAGEMENT METHODS
    # ============================================================================
    
    @log_sync
    def _update_supervisor_state_with_checkpointer(self, item: Dict[str, Any], context_id: str, 
                                                 task_id: str, step_count: int, thread_id: str) -> None:
        """
        Update supervisor state and let LangGraph handle persistence.
        
        Args:
            item: State update from LangGraph
            context_id: Context identifier
            task_id: Task identifier
            step_count: Current step count
            thread_id: Thread identifier
        """
        try:
            # Update local state
            self._update_supervisor_state(item, context_id, task_id, step_count)
            
            # Let LangGraph handle its own state management
            # The checkpointer is automatically managed by LangGraph during graph execution
            # We don't need to manually update it here
            supervisor_logger.log_structured(
                level="DEBUG",
                message="State updated, LangGraph will handle persistence",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "thread_id": thread_id,
                    "step_count": step_count,
                    "state_updated": True
                }
            )
                
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to update supervisor state with checkpointer: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "error": str(e),
                    "thread_id": thread_id,
                    "step_count": step_count
                }
            )
    
    # ============================================================================
    # ENHANCED RESPONSE FORMATTING METHODS
    # ============================================================================
    
    @log_sync
    def _format_interrupt_response(self, interrupt_payload: Dict[str, Any], context_id: str, 
                                 task_id: str, thread_id: str, step_count: int) -> AgentResponse:
        """
        Format interrupt responses with proper metadata.
        
        Args:
            interrupt_payload: Interrupt payload data
            context_id: Context identifier
            task_id: Task identifier
            thread_id: Thread identifier
            step_count: Current step count
            
        Returns:
            Formatted AgentResponse
        """
        interrupt_type = interrupt_payload.get('context', 'unknown')
        
        response_mapping = {
            'dependency_mapping': self._format_dependency_response,
            'approval_required': self._format_approval_response,
            'human_input': self._format_generic_interrupt_response
        }
        
        formatter = response_mapping.get(interrupt_type, self._format_generic_interrupt_response)
        return formatter(interrupt_payload, context_id, task_id, thread_id, step_count)
    
    @log_sync
    def _format_dependency_response(self, payload: Dict[str, Any], context_id: str, 
                                  task_id: str, thread_id: str, step_count: int) -> AgentResponse:
        """Format dependency mapping response."""
        return AgentResponse(
            response_type='dependency_question',
            is_task_complete=False,
            require_user_input=True,
            content=payload.get('question', 'Dependency mapping question'),
            metadata={
                'session_id': context_id,
                'task_id': task_id,
                'agent_name': self.name,
                'step_count': step_count,
                'status': 'dependency_mapping_question',
                'interrupt_type': 'dependency_mapping',
                'available_questions': payload.get('available_questions', []),
                'partial_analysis': payload.get('partial_analysis', {}),
                'thread_id': thread_id
            }
        )
    
    @log_sync
    def _format_approval_response(self, payload: Dict[str, Any], context_id: str, 
                                task_id: str, thread_id: str, step_count: int) -> AgentResponse:
        """Format approval required response."""
        return AgentResponse(
            response_type='approval_required',
            is_task_complete=False,
            require_user_input=True,
            content=payload.get('message', 'Approval required'),
            metadata={
                'session_id': context_id,
                'task_id': task_id,
                'agent_name': self.name,
                'step_count': step_count,
                'status': 'approval_required',
                'interrupt_type': 'approval_required',
                'node': payload.get('node', 'unknown'),
                'thread_id': thread_id
            }
        )
    
    @log_sync
    def _format_generic_interrupt_response(self, payload: Dict[str, Any], context_id: str, 
                                         task_id: str, thread_id: str, step_count: int) -> AgentResponse:
        """Format generic interrupt response."""
        return AgentResponse(
            response_type='human_input',
            is_task_complete=False,
            require_user_input=True,
            content=payload.get('question', 'Agent requires human feedback'),
            metadata={
                'session_id': context_id,
                'task_id': task_id,
                'agent_name': self.name,
                'step_count': step_count,
                'status': 'agent_requires_feedback',
                'interrupt_type': payload.get('context', 'unknown'),
                'thread_id': thread_id
            }
        )
    
    # ============================================================================
    # ENHANCED ERROR HANDLING METHODS
    # ============================================================================
    
    @log_async
    async def _handle_stream_error(self, error: Exception, context_id: str, task_id: str, thread_id: str) -> AgentResponse:
        """
        Enhanced error handling with recovery mechanisms.
        
        Args:
            error: The exception that occurred
            context_id: Context identifier
            task_id: Task identifier
            thread_id: Thread identifier
            
        Returns:
            Error AgentResponse
        """
        supervisor_logger.log_structured(
            level="ERROR",
            message=f"Stream execution failed: {error}",
            task_id=task_id,
            context_id=context_id,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
                "thread_id": thread_id
            }
        )
        
        # Attempt recovery if possible
        recovery_attempted = await self._attempt_error_recovery(error, context_id, task_id, thread_id)
        
        return AgentResponse(
            response_type='error',
            is_task_complete=True,
            require_user_input=False,
            content=f'Supervisor execution failed: {str(error)}',
            error=str(error),
            metadata={
                'session_id': context_id,
                'task_id': task_id,
                'agent_name': self.name,
                'status': 'failed',
                'thread_id': thread_id,
                'recovery_attempted': recovery_attempted
            }
        )
    
    @log_async
    async def _attempt_error_recovery(self, error: Exception, context_id: str, task_id: str, thread_id: str) -> bool:
        """
        Attempt error recovery mechanisms.
        
        Args:
            error: The exception that occurred
            context_id: Context identifier
            task_id: Task identifier
            thread_id: Thread identifier
            
        Returns:
            True if recovery was attempted
        """
        try:
            # For now, just log the recovery attempt
            # In the future, this could include:
            # - State rollback
            # - Resource cleanup
            # - Retry logic
            # - Fallback mechanisms
            
            supervisor_logger.log_structured(
                level="INFO",
                message="Attempting error recovery",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "error_type": type(error).__name__,
                    "thread_id": thread_id,
                    "recovery_mechanism": "logging_only"
                }
            )
            
            return True
            
        except Exception as recovery_error:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Error recovery failed: {recovery_error}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "original_error": str(error),
                    "recovery_error": str(recovery_error),
                    "thread_id": thread_id
                }
            )
            return False
    
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
    def _update_supervisor_state(self, item: Dict[str, Any], context_id: str, task_id: str, step_count: int) -> None:
        """
        Update supervisor state with the latest state from LangGraph using proper validation.
        
        Args:
            item: State update from supervisor.astream()
            context_id: Context identifier
            task_id: Task identifier
            step_count: Current step count
        """
        try:
            if not self.supervisor_state:
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="No supervisor state to update",
                    task_id=task_id,
                    context_id=context_id
                )
                return
            
            # Preserve existing state fields that shouldn't be overwritten
            preserved_fields = {
                "workflow_id": self.supervisor_state.workflow_id,
                "workflow_started_at": self.supervisor_state.workflow_started_at,
                "workflow_completed_at": self.supervisor_state.workflow_completed_at,
                "human_approval_required": self.supervisor_state.human_approval_required,
                "approval_context": self.supervisor_state.approval_context,
                "error": self.supervisor_state.error,
                "retry_count": self.supervisor_state.retry_count,
                "max_retries": self.supervisor_state.max_retries,
                "workspace_ref": self.supervisor_state.workspace_ref,
                "terraform_context": self.supervisor_state.terraform_context,
                "generated_module_ref": self.supervisor_state.generated_module_ref,
                "validation_report_ref": self.supervisor_state.validation_report_ref,
                "security_report_ref": self.supervisor_state.security_report_ref,
                "cost_report_ref": self.supervisor_state.cost_report_ref
            }
            
            # Merge LangGraph item with preserved fields
            updated_state_data = {**item, **preserved_fields}
            
            # Ensure required fields are present
            if "user_request" not in updated_state_data:
                updated_state_data["user_request"] = self.supervisor_state.user_request
            
            if "session_id" not in updated_state_data:
                updated_state_data["session_id"] = context_id
                
            if "task_id" not in updated_state_data:
                updated_state_data["task_id"] = task_id
            
            # Create new supervisor state with proper validation
            new_supervisor_state = SupervisorState(**updated_state_data)
            
            # Replace the current state with the validated new state
            self.supervisor_state = new_supervisor_state
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Updated supervisor state at step {step_count}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "step_count": step_count,
                    "message_count": len(item.get('messages', [])),
                    "status": self.supervisor_state.status,
                    "workflow_id": self.supervisor_state.workflow_id
                }
            )
                
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to update supervisor state: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={"error": str(e), "step_count": step_count}
            )
    
    def _format_stream_response(self, item: Dict[str, Any], context_id: str, task_id: str, step_count: int) -> AgentResponse:
        """Format streaming response with proper metadata and state information."""
        status = item.get('status')
        question = item.get('question')
        final_response = item.get('final_response')

        if status == 'completed':
            return AgentResponse(
                response_type='data',
                is_task_complete=True,
                require_user_input=False,
                content={
                    'status': status,
                    'question': question,
                    'final_response': final_response
                },
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'step_count': step_count,
                    'status': 'completed',
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if self.supervisor_state else None
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
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if self.supervisor_state else None
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
                    'workflow_id': getattr(self.supervisor_state, 'workflow_id', None) if self.supervisor_state else None
                }
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
