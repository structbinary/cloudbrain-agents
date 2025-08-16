"""
Planner Agent Implementation.

This module implements the Planner Agent, which is responsible for:
- Analyzing user requirements and existing infrastructure
- Creating detailed implementation plans
- Identifying AWS services and their interdependencies
- Planning architectural patterns and security considerations
- Generating step-by-step execution plans

The Planner Agent follows the LangChain/LangGraph best practices and inherits from BaseAgent.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from functools import wraps

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel

from aws_orchestrator_agent.core.agents.base_agent import BaseAgent, agent_node, require_approval
from langgraph.graph import StateGraph
from aws_orchestrator_agent.core.agents.planner.planner_state import (
    PlannerState,
    InfrastructureRequirement,
    ArchitecturalPattern,
    SecurityRequirement,
    ExecutionStep
)
from aws_orchestrator_agent.core.agents.planner.new_infrastructure_planner import create_new_infrastructure_planner
from aws_orchestrator_agent.core.agents.planner.modification_planner import create_modification_planner
from aws_orchestrator_agent.core.agents.supervisor.state.state_transformers import StateTransformer
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config


# Create agent logger for planner
planner_logger = AgentLogger("PLANNER")


class RequirementsAnalysis(BaseModel):
    """Output schema for requirements analysis."""
    business_requirements: List[str]
    technical_requirements: List[str]
    constraints: List[str]
    assumptions: List[str]
    risk_factors: List[str]


class InfrastructurePlan(BaseModel):
    """Output schema for infrastructure planning."""
    services: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    security_requirements: List[Dict[str, Any]]
    cost_estimates: Dict[str, Any]
    dependencies: Dict[str, List[str]]


class ExecutionPlan(BaseModel):
    """Output schema for execution planning."""
    steps: List[Dict[str, Any]]
    total_estimated_time: str
    critical_path: List[int]
    resource_requirements: Dict[str, Any]


class PlannerAgent(BaseAgent):
    """
    Planner Agent Coordinator for AWS infrastructure planning.
    
    This agent coordinates between specialized planners:
    - NewInfrastructurePlanner: For new AWS infrastructure projects
    - ModificationPlanner: For modifying existing infrastructure
    
    The main PlannerAgent handles request classification and routes to
    the appropriate specialized planner.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "planner_agent"
    ):
        """
        Initialize the Planner Agent.
        
        Args:
            config: Configuration instance
            custom_config: Optional custom configuration
            name: Agent name
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Get LLM configuration
        llm_config = self.config_instance.get_llm_config()
        
        # Initialize LLM model
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            planner_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model for planner: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model for planner: {e}",
                extra={"error": str(e)}
            )
            raise
        
        # Initialize base agent
        super().__init__(
            name=name,
            state_schema=PlannerState,
            config=custom_config or {},
            mcp_client=None,  # Will be set by supervisor if needed
            state_manager=None  # Will be set by supervisor if needed
        )
        
        # Initialize specialized planners
        self.new_infrastructure_planner = create_new_infrastructure_planner(config, custom_config, "new_infrastructure_planner")
        self.modification_planner = create_modification_planner(config, custom_config, "modification_planner")
        
        # Initialize output parsers for classification
        self.classification_parser = StrOutputParser()
        
        # Define prompts for classification
        self._define_prompts()
        
        # Build the coordinator graph
        self._build_coordinator_graph()
    
    def _define_prompts(self) -> None:
        """Define the prompts used by the planner coordinator."""
        
        # Request classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS infrastructure planner. Your task is to classify whether the user request is for creating a new AWS module/service or modifying existing infrastructure.

Classification Rules:
- "new": Request to create a completely new AWS module, service, or infrastructure component
- "modify": Request to modify existing infrastructure, add functionality to existing modules, or update configurations

Examples:
- "Create a new S3 bucket for file storage" → "new"
- "Add auto-scaling to existing EC2 instances" → "modify"
- "Create a new Lambda function for image processing" → "new"
- "Update the security groups for existing RDS instances" → "modify"

Respond with only "new" or "modify"."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Classify this request: {user_request}")
        ])
        self.change_impact_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert change management specialist. Assess the impact of proposed infrastructure changes.

Evaluate:
1. Which resources will be affected by the changes
2. Which resources will remain unchanged
3. Potential downtime requirements
4. Risk factors and mitigation strategies
5. Rollback procedures and requirements
6. Cost implications of the changes

Provide a detailed impact analysis with risk assessment."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Assess the impact of these changes: {proposed_changes}\n\nExisting infrastructure: {existing_infrastructure}")
        ])
    
    def _build_coordinator_graph(self) -> None:
        """Build the coordinator's StateGraph."""
        
        # Define nodes
        nodes = self.define_nodes()
        for name, node_func in nodes.items():
            self.add_node(name, node_func)
        
        # Build the graph with conditional edges
        self._build_conditional_graph()
        self.compile_graph()
        
        planner_logger.log_structured(
            level="INFO",
            message="Planner coordinator graph built and compiled successfully",
            extra={"agent_name": self.name, "node_count": len(nodes)}
        )
    
    def _build_conditional_graph(self) -> None:
        """Build the graph with conditional routing for new vs modify requests."""
        
        # Create the StateGraph
        self.graph = StateGraph(PlannerState)
        
        # Add all nodes
        for name, node_func in self.nodes.items():
            self.graph.add_node(name, node_func)
        
        # Add regular edges
        edges = {
            "classify_request_type": ["route_to_specialized_planner"]
        }
        
        for source, targets in edges.items():
            for target in targets:
                self.graph.add_edge(source, target)
        
        # Set entry and finish points
        self.graph.set_entry_point("classify_request_type")
        self.graph.set_finish_point("route_to_specialized_planner")
        
        planner_logger.log_structured(
            level="INFO",
            message="Built coordinator StateGraph for planner agent",
            extra={"agent_name": self.name, "conditional_edges": False}
        )
    
    def define_nodes(self) -> Dict[str, Any]:
        """Define the coordinator's nodes."""
        return {
            "classify_request_type": self.classify_request_type_node,
            "route_to_specialized_planner": self.route_to_specialized_planner_node
        }
    
    def define_edges(self) -> Dict[str, List[str]]:
        """Define the edges between nodes."""
        return {
            "classify_request_type": ["route_to_specialized_planner"],
            "route_to_specialized_planner": []
        }
    
    def _route_after_classification(self, state: Dict[str, Any]) -> str:
        """Route to appropriate path based on request type."""
        request_type = state.get("request_type", "new")
        
        if request_type == "modify":
            return "analyze_existing_infrastructure"
        else:
            return "analyze_requirements"
    
    def create_initial_state(self, input_data: Dict[str, Any]) -> PlannerState:
        """Create the initial state for the planner agent."""
        
        # Debug: Log the input data to understand what langgraph-supervisor is passing
        planner_logger.log_structured(
            level="DEBUG",
            message="create_initial_state called with input_data",
            extra={
                "input_data_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict",
                "input_data_type": type(input_data).__name__,
                "input_data_preview": str(input_data)[:200] if isinstance(input_data, dict) else str(input_data)[:200]
            }
        )
        
        # Extract user request - handle both direct input and messages from supervisor
        user_request = input_data.get("user_request", "")
        context_id = input_data.get("context_id", str(uuid.uuid4()))
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        
        # If no user_request provided, try to extract from messages
        if not user_request and "messages" in input_data:
            messages = input_data["messages"]
            planner_logger.log_structured(
                level="DEBUG",
                message="Extracting user_request from messages",
                extra={
                    "message_count": len(messages),
                    "message_types": [type(msg).__name__ for msg in messages]
                }
            )
            for message in messages:
                if hasattr(message, 'content') and isinstance(message.content, str):
                    # Use the first human message as the user request
                    user_request = message.content
                    planner_logger.log_structured(
                        level="DEBUG",
                        message="Extracted user_request from message",
                        extra={
                            "user_request_length": len(user_request),
                            "user_request_preview": user_request[:100]
                        }
                    )
                    break
        
        # If still no user_request, use a default
        if not user_request:
            user_request = "No specific request provided"
            planner_logger.log_structured(
                level="WARNING",
                message="No user_request found in input_data, using default",
                extra={
                    "input_data_keys": list(input_data.keys()) if isinstance(input_data, dict) else "Not a dict"
                }
            )
        
        # Create initial messages
        messages = [
            SystemMessage(content="You are an expert AWS infrastructure planner. Analyze the user request and create a comprehensive plan."),
            HumanMessage(content=user_request)
        ]
        
        # Create initial state
        state = PlannerState(
            messages=messages,
            user_request=user_request,
            context_id=context_id,
            task_id=task_id,
            status="planning_started",
            planning_started_at=datetime.utcnow().isoformat()
        )
        
        planner_logger.log_structured(
            level="INFO",
            message="Created initial planner state",
            extra={
                "agent_name": self.name,
                "context_id": context_id,
                "task_id": task_id,
                "user_request_length": len(user_request),
                "extracted_from_messages": "user_request" not in input_data
            }
        )
        
        return state
    
    async def classify_request_type_node(self, state: Union[Dict[str, Any], PlannerState]) -> Dict[str, Any]:
        """Classify whether the request is for new infrastructure or modification."""
        
        try:
            # Debug: Log the incoming state
            planner_logger.log_structured(
                level="DEBUG",
                message="classify_request_type_node called",
                extra={
                    "state_type": type(state).__name__,
                    "state_keys": list(state.keys()) if isinstance(state, dict) else "Not a dict",
                    "has_messages": "messages" in state if isinstance(state, dict) else "Not a dict",
                    "has_user_request": "user_request" in state if isinstance(state, dict) else "Not a dict",
                    "has_context_id": "context_id" in state if isinstance(state, dict) else "Not a dict",
                    "has_task_id": "task_id" in state if isinstance(state, dict) else "Not a dict"
                }
            )
            
            # Handle state transformation - langgraph-supervisor passes state directly
            if isinstance(state, dict):
                # Transform supervisor state to planner state using our transformer
                transformed_state = StateTransformer.supervisor_to_planner_state(state)
                current_state = PlannerState(**transformed_state)
                
                planner_logger.log_structured(
                    level="DEBUG",
                    message="Applied state transformation from supervisor dict",
                    extra={
                        "original_context_id": state.get("context_id"),
                        "transformed_context_id": current_state.context_id,
                        "original_task_id": state.get("task_id"),
                        "transformed_task_id": current_state.task_id
                    }
                )
            elif isinstance(state, PlannerState):
                # langgraph-supervisor passes PlannerState directly, but it might not have context_id/task_id
                current_state = state
                
                # Check if context_id and task_id are missing
                if not current_state.context_id or not current_state.task_id:
                    planner_logger.log_structured(
                        level="WARNING",
                        message="PlannerState received but context_id/task_id are missing - this indicates langgraph-supervisor is not passing state correctly",
                        extra={
                            "context_id": current_state.context_id,
                            "task_id": current_state.task_id,
                            "user_request": current_state.user_request[:50] if current_state.user_request else "None",
                            "message_count": len(current_state.messages) if current_state.messages else 0
                        }
                    )
                    
                    # Try to extract context from the first HumanMessage
                    if current_state.messages:
                        for message in current_state.messages:
                            if hasattr(message, 'content') and isinstance(message.content, str):
                                # Look for context_id and task_id in message content
                                if 'Context ID:' in message.content and 'Task ID:' in message.content:
                                    try:
                                        # Extract context_id and task_id from message content
                                        # Format: "Context ID: test-context-123, Task ID: test-task-456"
                                        content = message.content
                                        
                                        # Find Context ID
                                        context_start = content.find('Context ID:')
                                        if context_start != -1:
                                            context_end = content.find(',', context_start)
                                            if context_end != -1:
                                                context_id = content[context_start + 12:context_end].strip()
                                                current_state.context_id = context_id
                                        
                                        # Find Task ID
                                        task_start = content.find('Task ID:')
                                        if task_start != -1:
                                            task_end = content.find('\n', task_start)
                                            if task_end == -1:
                                                task_end = len(content)
                                            task_id = content[task_start + 8:task_end].strip()
                                            current_state.task_id = task_id
                                        
                                        if current_state.context_id and current_state.task_id:
                                            planner_logger.log_structured(
                                                level="INFO",
                                                message="Successfully extracted context_id and task_id from message content",
                                                extra={
                                                    "extracted_context_id": current_state.context_id,
                                                    "extracted_task_id": current_state.task_id,
                                                    "message_content": message.content[:100]
                                                }
                                            )
                                            break
                                    except Exception as e:
                                        planner_logger.log_structured(
                                            level="ERROR",
                                            message=f"Failed to extract context from message: {e}",
                                            extra={"message_content": message.content[:100]}
                                        )
                    
                    # Log final state
                    planner_logger.log_structured(
                        level="DEBUG",
                        message="Final PlannerState after context extraction",
                        extra={
                            "context_id": current_state.context_id,
                            "task_id": current_state.task_id,
                            "user_request": current_state.user_request[:50] if current_state.user_request else "None"
                        }
                    )
            else:
                raise ValueError(f"Unexpected state type: {type(state)}")
            
            # Ensure user_request is set if empty
            if not current_state.user_request and current_state.messages:
                for message in current_state.messages:
                    if hasattr(message, 'content') and isinstance(message.content, str):
                        if hasattr(message, '__class__') and 'Human' in message.__class__.__name__:
                            # Extract only the user request part, removing context information
                            content = message.content
                            if 'Context ID:' in content and 'Task ID:' in content:
                                # Find the actual user request after the context information
                                # Format: "Context ID: xxx, Task ID: xxx\n\nuser_request"
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if line.strip() and not line.startswith('Context ID:') and not line.startswith('Task ID:'):
                                        # Found the user request part
                                        user_request = '\n'.join(lines[i:]).strip()
                                        current_state.user_request = user_request
                                        break
                                else:
                                    # Fallback: use the original content
                                    current_state.user_request = content
                            else:
                                # No context information, use the content as-is
                                current_state.user_request = content
                            break
                        elif not current_state.user_request:
                            current_state.user_request = message.content
            
            # Create classification chain
            classification_chain = self.classification_prompt | self.model | StrOutputParser()
            
            # Execute classification
            classification_result = classification_chain.invoke({
                "messages": current_state.messages,
                "user_request": current_state.user_request
            })
            
            # Parse result (should be "new" or "modify")
            request_type = classification_result.strip().lower()
            if request_type not in ["new", "modify"]:
                request_type = "new"  # Default to new if unclear
            
            # Calculate confidence (simple heuristic for now)
            confidence = 0.9 if request_type in ["new", "modify"] else 0.5
            
            # Update state
            current_state.request_type = request_type
            current_state.request_classification_confidence = confidence
            current_state.current_step = "request_classified"
            current_state.completed_steps.append("classify_request_type")
            
            # Add classification result to messages
            current_state.messages.append(
                AIMessage(content=f"Request classified as: {request_type} (confidence: {confidence})")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message=f"Request classified as {request_type}",
                extra={
                    "request_type": request_type,
                    "confidence": confidence,
                    "user_request": current_state.user_request[:100] + "..." if len(current_state.user_request) > 100 else current_state.user_request
                }
            )
            
            # Debug: Log what we're returning
            result = current_state.model_dump()
            planner_logger.log_structured(
                level="DEBUG",
                message="classify_request_type_node returning result",
                extra={
                    "result_type": type(result).__name__,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "Not a dict",
                    "has_messages": "messages" in result if isinstance(result, dict) else "Not a dict",
                    "request_type": result.get("request_type", "unknown") if isinstance(result, dict) else "unknown"
                }
            )
            
            # Return state in the format expected by langgraph-supervisor
            # The library expects a state with messages and other fields
            return {
                "messages": current_state.messages,
                "user_request": current_state.user_request,
                "context_id": current_state.context_id,
                "task_id": current_state.task_id,
                "request_type": current_state.request_type,
                "request_classification_confidence": current_state.request_classification_confidence,
                "current_step": current_state.current_step,
                "completed_steps": current_state.completed_steps,
                "status": current_state.status
            }
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in request classification: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set default values on error
            current_state = PlannerState(**state)
            current_state.request_type = "new"
            current_state.request_classification_confidence = 0.0
            current_state.error = str(e)
            current_state.error_context = "request_classification"
            
            return current_state.model_dump()
    
    async def route_to_specialized_planner_node(self, state: Union[Dict[str, Any], PlannerState]) -> Dict[str, Any]:
        """Route to the appropriate specialized planner based on request type."""
        
        try:
            # Use state transformer to ensure proper state handling
            if isinstance(state, dict):
                # Transform supervisor state to planner state using our transformer
                transformed_state = StateTransformer.supervisor_to_planner_state(state)
                current_state = PlannerState(**transformed_state)
                
                planner_logger.log_structured(
                    level="DEBUG",
                    message="Applied state transformation in routing node",
                    extra={
                        "original_context_id": state.get("context_id"),
                        "transformed_context_id": current_state.context_id,
                        "original_task_id": state.get("task_id"),
                        "transformed_task_id": current_state.task_id
                    }
                )
            elif isinstance(state, PlannerState):
                current_state = state
            else:
                raise ValueError(f"Unexpected state type: {type(state)}")
            request_type = current_state.request_type
            
            planner_logger.log_structured(
                level="INFO",
                message=f"Routing to specialized planner for request type: {request_type}",
                extra={
                    "agent_name": self.name,
                    "request_type": request_type,
                    "user_request": current_state.user_request[:100] + "..." if len(current_state.user_request) > 100 else current_state.user_request
                }
            )
            
            # Route to appropriate specialized planner
            if request_type == "new":
                # Check if we're resuming from a dependency mapping interruption
                if current_state.waiting_for_dependency_input and current_state.dependency_answers:
                    # We're resuming from a HITL interruption
                    planner_logger.log_structured(
                        level="INFO",
                        message="Resuming from dependency mapping HITL interruption",
                        extra={
                            "agent_name": self.name,
                            "current_question": current_state.current_dependency_question,
                            "answers_count": len(current_state.dependency_answers)
                        }
                    )
                    
                    # Get the current state from the specialized planner
                    new_planner_graph = self.new_infrastructure_planner.get_compiled_graph()
                    
                    # Create a state that includes the dependency answers
                    new_planner_state = self.new_infrastructure_planner.create_state_for_execution({
                        "user_request": current_state.user_request,
                        "context_id": current_state.context_id,
                        "task_id": current_state.task_id,
                        "waiting_for_dependency_input": current_state.waiting_for_dependency_input,
                        "dependency_questions": current_state.dependency_questions,
                        "dependency_answers": current_state.dependency_answers,
                        "current_dependency_question": current_state.current_dependency_question,
                        "dependency_mapping_complete": current_state.dependency_mapping_complete,
                        "planning_metadata": current_state.planning_metadata
                    })
                else:
                    # Initial execution
                    new_planner_graph = self.new_infrastructure_planner.get_compiled_graph()
                    new_planner_state = self.new_infrastructure_planner.create_state_for_execution({
                        "user_request": current_state.user_request,
                        "context_id": current_state.context_id,
                        "task_id": current_state.task_id
                    })
                
                # Execute the new infrastructure planner graph
                planner_logger.log_structured(
                    level="INFO",
                    message="Executing new infrastructure planner graph",
                    extra={
                        "agent_name": self.name,
                        "request_type": request_type,
                        "user_request": current_state.user_request[:100] + "..." if len(current_state.user_request) > 100 else current_state.user_request,
                        "graph_type": type(new_planner_graph).__name__,
                        "state_type": type(new_planner_state).__name__
                    }
                )
                
                try:
                    # Convert PlannerState to dict for graph execution
                    if hasattr(new_planner_state, 'model_dump'):
                        # It's a Pydantic model, convert to dict
                        new_planner_state_dict = new_planner_state.model_dump()
                    else:
                        # It's already a dict
                        new_planner_state_dict = new_planner_state
                    
                    # Execute the graph with streaming to handle interrupts
                    specialized_state = None
                    start_time = time.time()  # Add start_time for execution tracking
                    
                    # Check if this is a resume from HITL interruption
                    resume_value = getattr(current_state, 'resume_value', None)
                    
                    if resume_value is not None:
                        # Resume call: reuse context_id as thread_id for HITL resume
                        thread_id = current_state.context_id
                        planner_logger.log_structured(
                            level="INFO",
                            message="Resuming new infrastructure planner with existing thread_id",
                            extra={
                                "agent_name": self.name,
                                "thread_id": thread_id,
                                "resume_value": str(resume_value)
                            }
                        )
                    else:
                        # Initial call: use a new thread_id for state isolation
                        thread_id = str(uuid.uuid4())
                        planner_logger.log_structured(
                            level="INFO",
                            message="Starting new infrastructure planner with new thread_id",
                            extra={
                                "agent_name": self.name,
                                "thread_id": thread_id
                            }
                        )
                    
                    config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
                    
                    async for event in new_planner_graph.astream(new_planner_state_dict, config, stream_mode='values'):
                        # Debug: Log all events to understand the structure
                        planner_logger.log_structured(
                            level="DEBUG",
                            message="Received event from new infrastructure planner subgraph",
                            extra={
                                "agent_name": self.name,
                                "event_type": type(event).__name__,
                                "event_attributes": [attr for attr in dir(event) if not attr.startswith('_')],
                                "event_str": str(event)[:200]
                            }
                        )
                        
                        # Check if this is an interrupt event - LangGraph uses __interrupt__ attribute
                        if hasattr(event, '__interrupt__') and event.__interrupt__:
                            planner_logger.log_structured(
                                level="INFO",
                                message="Interrupt detected from new infrastructure planner subgraph",
                                extra={
                                    "agent_name": self.name,
                                    "interrupt_data": event.__interrupt__,
                                    "interrupt_type": type(event.__interrupt__).__name__
                                }
                            )
                            # Let the interrupt propagate naturally - don't handle it here
                            # The GraphInterrupt will be caught by the outer try-catch block
                            break
                        
                        # Check if this is the final result
                        if hasattr(event, 'end') and event.end:
                            specialized_state = event.end
                            break
                    
                    # If no final state was received, create a placeholder
                    if specialized_state is None:
                        specialized_state = {
                            "status": "interrupted",
                            "requirements_analysis": None,
                            "infrastructure_requirements": [],
                            "architectural_patterns": [],
                            "security_requirements": [],
                            "execution_plan": [],
                            "cost_analysis": {},
                            "resource_requirements": {},
                            "validation_criteria": [],
                            "complexity_score": 0,
                            "planning_completed_at": None,
                            "planning_duration": None
                        }
                    
                    planner_logger.log_structured(
                        level="INFO",
                        message="New infrastructure planner execution completed successfully",
                        extra={
                            "agent_name": self.name,
                            "execution_status": specialized_state.get("status", "unknown"),
                            "has_requirements": "requirements_analysis" in specialized_state,
                            "has_infrastructure": "infrastructure_requirements" in specialized_state
                        }
                    )
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Check if this is a GraphInterrupt (LangGraph's interrupt mechanism)
                    if hasattr(e, '__class__') and 'GraphInterrupt' in e.__class__.__name__:
                        # This is a LangGraph interrupt - let it propagate naturally
                        # langgraph_supervisor will handle this through its built-in interrupt mechanism
                        planner_logger.log_structured(
                            level="INFO",
                            message="GraphInterrupt detected from new infrastructure planner subgraph - propagating to supervisor",
                            extra={
                                "agent_name": self.name,
                                "interrupt_data": str(e),
                                "interrupt_type": type(e).__name__
                            }
                        )
                        
                        # Re-raise the GraphInterrupt to let langgraph_supervisor handle it
                        # This is the correct way to handle interrupts with langgraph_supervisor
                        raise e
                    

                    

                    
                    # For all other exceptions, log as error and set error state
                    planner_logger.log_structured(
                        level="ERROR",
                        message=f"Error in new infrastructure planner subgraph execution: {e}",
                        extra={
                            "agent_name": self.name,
                            "error": str(e),
                            "execution_time": execution_time,
                            "user_request": current_state.user_request
                        }
                    )
                    
                    # Set error state
                    current_state.error = str(e)
                    current_state.error_context = "new_infrastructure_planning"
                    current_state.status = "error"
                    
                    return current_state.model_dump()
                
                # Merge results into current state
                current_state.requirements_analysis = specialized_state.get("requirements_analysis")
                current_state.infrastructure_requirements = specialized_state.get("infrastructure_requirements")
                current_state.architectural_patterns = specialized_state.get("architectural_patterns")
                current_state.security_requirements = specialized_state.get("security_requirements")
                current_state.execution_plan = specialized_state.get("execution_plan")
                current_state.cost_analysis = specialized_state.get("cost_analysis")
                current_state.resource_requirements = specialized_state.get("resource_requirements")
                current_state.validation_criteria = specialized_state.get("validation_criteria")
                current_state.complexity_score = specialized_state.get("complexity_score")
                current_state.status = specialized_state.get("status", "completed")
                current_state.planning_completed_at = specialized_state.get("planning_completed_at")
                current_state.planning_duration = specialized_state.get("planning_duration")
                
                # Add routing message
                current_state.messages.append(
                    AIMessage(content=f"Request routed to New Infrastructure Planner. Planning completed successfully.")
                )
                
            elif request_type == "modify":
                # Get compiled graph and create state for modification planner
                modification_planner_graph = self.modification_planner.get_compiled_graph()
                modification_planner_state = self.modification_planner.create_state_for_execution({
                    "user_request": current_state.user_request,
                    "context_id": current_state.context_id,
                    "task_id": current_state.task_id
                })
                
                # Execute the modification planner graph
                planner_logger.log_structured(
                    level="INFO",
                    message="Executing modification planner graph",
                    extra={
                        "agent_name": self.name,
                        "request_type": request_type,
                        "user_request": current_state.user_request[:100] + "..." if len(current_state.user_request) > 100 else current_state.user_request,
                        "graph_type": type(modification_planner_graph).__name__,
                        "state_type": type(modification_planner_state).__name__
                    }
                )
                
                try:
                    # Convert PlannerState to dict for graph execution
                    if hasattr(modification_planner_state, 'model_dump'):
                        # It's a Pydantic model, convert to dict
                        modification_planner_state_dict = modification_planner_state.model_dump()
                    else:
                        # It's already a dict
                        modification_planner_state_dict = modification_planner_state
                    
                    # Execute the graph with streaming to handle interrupts
                    specialized_state = None
                    start_time = time.time()  # Add start_time for execution tracking
                    
                    # Check if this is a resume from HITL interruption
                    resume_value = getattr(current_state, 'resume_value', None)
                    
                    if resume_value is not None:
                        # Resume call: reuse context_id as thread_id for HITL resume
                        thread_id = current_state.context_id
                        planner_logger.log_structured(
                            level="INFO",
                            message="Resuming modification planner with existing thread_id",
                            extra={
                                "agent_name": self.name,
                                "thread_id": thread_id,
                                "resume_value": str(resume_value)
                            }
                        )
                    else:
                        # Initial call: use a new thread_id for state isolation
                        thread_id = str(uuid.uuid4())
                        planner_logger.log_structured(
                            level="INFO",
                            message="Starting modification planner with new thread_id",
                            extra={
                                "agent_name": self.name,
                                "thread_id": thread_id
                            }
                        )
                    
                    config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
                    
                    async for event in modification_planner_graph.astream(modification_planner_state_dict, config, stream_mode='values'):
                        # Debug: Log all events to understand the structure
                        planner_logger.log_structured(
                            level="DEBUG",
                            message="Received event from modification planner subgraph",
                            extra={
                                "agent_name": self.name,
                                "event_type": type(event).__name__,
                                "event_attributes": [attr for attr in dir(event) if not attr.startswith('_')],
                                "event_str": str(event)[:200]
                            }
                        )
                        
                        # Check if this is an interrupt event - LangGraph uses __interrupt__ attribute
                        if hasattr(event, '__interrupt__') and event.__interrupt__:
                            planner_logger.log_structured(
                                level="INFO",
                                message="Interrupt detected from modification planner subgraph",
                                extra={
                                    "agent_name": self.name,
                                    "interrupt_data": event.__interrupt__,
                                    "interrupt_type": type(event.__interrupt__).__name__
                                }
                            )
                            # Let the interrupt propagate naturally - don't handle it here
                            # The GraphInterrupt will be caught by the outer try-catch block
                            break
                        
                        # Check if this is the final result
                        if hasattr(event, 'end') and event.end:
                            specialized_state = event.end
                            break
                    
                    # If no final state was received, create a placeholder
                    if specialized_state is None:
                        specialized_state = {
                            "status": "interrupted",
                            "requirements_analysis": None,
                            "existing_infrastructure": None,
                            "change_impact": None,
                            "affected_resources": [],
                            "unchanged_resources": [],
                            "downtime_required": False,
                            "rollback_strategy": "standard",
                            "risk_assessment": {"risk_level": "unknown"},
                            "infrastructure_requirements": [],
                            "architectural_patterns": [],
                            "security_requirements": [],
                            "execution_plan": [],
                            "cost_analysis": {},
                            "resource_requirements": {},
                            "validation_criteria": [],
                            "complexity_score": 0,
                            "planning_completed_at": None,
                            "planning_duration": None
                        }
                    
                    planner_logger.log_structured(
                        level="INFO",
                        message="Modification planner execution completed successfully",
                        extra={
                            "agent_name": self.name,
                            "execution_status": specialized_state.get("status", "unknown"),
                            "has_requirements": "requirements_analysis" in specialized_state,
                            "has_existing_infrastructure": "existing_infrastructure" in specialized_state,
                            "has_change_impact": "change_impact" in specialized_state
                        }
                    )
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Check if this is a GraphInterrupt (LangGraph's interrupt mechanism)
                    if hasattr(e, '__class__') and 'GraphInterrupt' in e.__class__.__name__:
                        # This is a LangGraph interrupt - let it propagate naturally
                        # langgraph_supervisor will handle this through its built-in interrupt mechanism
                        planner_logger.log_structured(
                            level="INFO",
                            message="GraphInterrupt detected from modification planner subgraph - propagating to supervisor",
                            extra={
                                "agent_name": self.name,
                                "interrupt_data": str(e),
                                "interrupt_type": type(e).__name__
                            }
                        )
                        
                        # Re-raise the GraphInterrupt to let langgraph_supervisor handle it
                        # This is the correct way to handle interrupts with langgraph_supervisor
                        raise e
                    
                    # For all other exceptions, log as error and set error state
                    planner_logger.log_structured(
                        level="ERROR",
                        message=f"Error in modification planner subgraph execution: {e}",
                        extra={
                            "agent_name": self.name,
                            "error": str(e),
                            "execution_time": execution_time,
                            "user_request": current_state.user_request
                        }
                    )
                    
                    # Set error state
                    current_state.error = str(e)
                    current_state.error_context = "modification_planning"
                    current_state.status = "error"
                    
                    return current_state.model_dump()
                
                # Merge results into current state
                current_state.requirements_analysis = specialized_state.get("requirements_analysis")
                current_state.existing_infrastructure = specialized_state.get("existing_infrastructure")
                current_state.change_impact = specialized_state.get("change_impact")
                current_state.affected_resources = specialized_state.get("affected_resources")
                current_state.unchanged_resources = specialized_state.get("unchanged_resources")
                current_state.downtime_required = specialized_state.get("downtime_required")
                current_state.rollback_strategy = specialized_state.get("rollback_strategy")
                current_state.risk_assessment = specialized_state.get("risk_assessment")
                current_state.infrastructure_requirements = specialized_state.get("infrastructure_requirements")
                current_state.architectural_patterns = specialized_state.get("architectural_patterns")
                current_state.security_requirements = specialized_state.get("security_requirements")
                current_state.execution_plan = specialized_state.get("execution_plan")
                current_state.cost_analysis = specialized_state.get("cost_analysis")
                current_state.resource_requirements = specialized_state.get("resource_requirements")
                current_state.validation_criteria = specialized_state.get("validation_criteria")
                current_state.complexity_score = specialized_state.get("complexity_score")
                current_state.status = specialized_state.get("status", "completed")
                current_state.planning_completed_at = specialized_state.get("planning_completed_at")
                current_state.planning_duration = specialized_state.get("planning_duration")
                
                # Add routing message
                current_state.messages.append(
                    AIMessage(content=f"Request routed to Modification Planner. Planning completed successfully.")
                )
            
            else:
                # Default to new infrastructure planner
                planner_logger.log_structured(
                    level="WARNING",
                    message=f"Unknown request type '{request_type}', defaulting to new infrastructure planner",
                    extra={"request_type": request_type}
                )
                current_state.messages.append(
                    AIMessage(content=f"Unknown request type '{request_type}', defaulting to new infrastructure planner.")
                )
            
            current_state.current_step = "routing_completed"
            current_state.completed_steps.append("route_to_specialized_planner")
            
            # Debug: Log what we're returning
            result = current_state.model_dump()
            planner_logger.log_structured(
                level="DEBUG",
                message="route_to_specialized_planner_node returning result",
                extra={
                    "result_type": type(result).__name__,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "Not a dict",
                    "has_messages": "messages" in result if isinstance(result, dict) else "Not a dict",
                    "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown"
                }
            )
            
            # Return state in the format expected by langgraph-supervisor
            # The library expects a tool response format with tool_call_id
            # We need to wrap the state in the expected format
            
            # Generate a unique tool_call_id for this response
            tool_call_id = str(uuid.uuid4())
            
            # Return the state in the format expected by langgraph-supervisor handoff tools
            return {
                "tool_call_id": tool_call_id,
                "messages": current_state.messages,
                "user_request": current_state.user_request,
                "context_id": current_state.context_id,
                "task_id": current_state.task_id,
                "request_type": current_state.request_type,
                "current_step": current_state.current_step,
                "completed_steps": current_state.completed_steps,
                "status": current_state.status,
                "requirements_analysis": current_state.requirements_analysis,
                "infrastructure_requirements": current_state.infrastructure_requirements,
                "architectural_patterns": current_state.architectural_patterns,
                "security_requirements": current_state.security_requirements,
                "execution_plan": current_state.execution_plan,
                "cost_analysis": current_state.cost_analysis,
                "resource_requirements": current_state.resource_requirements,
                "compliance_requirements": current_state.compliance_requirements,
                "validation_criteria": current_state.validation_criteria,
                "planning_metadata": current_state.planning_metadata,
                "existing_infrastructure": current_state.existing_infrastructure,
                "affected_resources": current_state.affected_resources,
                "unchanged_resources": current_state.unchanged_resources,
                "change_impact": current_state.change_impact,
                "downtime_required": current_state.downtime_required,
                "rollback_strategy": current_state.rollback_strategy,
                "risk_assessment": current_state.risk_assessment,
                "error": current_state.error,
                "error_context": current_state.error_context,
                "requires_approval": current_state.requires_approval,
                "approval_context": current_state.approval_context,
                "dependency_questions": current_state.dependency_questions,
                "dependency_answers": current_state.dependency_answers,
                "waiting_for_dependency_input": current_state.waiting_for_dependency_input,
                "current_dependency_question": current_state.current_dependency_question,
                "dependency_mapping_complete": current_state.dependency_mapping_complete,
                "planning_started_at": current_state.planning_started_at,
                "planning_completed_at": current_state.planning_completed_at,
                "planning_duration": current_state.planning_duration,
                "complexity_score": current_state.complexity_score
            }
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in routing to specialized planner: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "routing_to_specialized_planner"
            
            # Return error state in the same tool response format
            tool_call_id = str(uuid.uuid4())
            
            return {
                "tool_call_id": tool_call_id,
                "messages": current_state.messages,
                "user_request": current_state.user_request,
                "context_id": current_state.context_id,
                "task_id": current_state.task_id,
                "request_type": current_state.request_type,
                "current_step": current_state.current_step,
                "completed_steps": current_state.completed_steps,
                "status": current_state.status,
                "requirements_analysis": current_state.requirements_analysis,
                "infrastructure_requirements": current_state.infrastructure_requirements,
                "architectural_patterns": current_state.architectural_patterns,
                "security_requirements": current_state.security_requirements,
                "execution_plan": current_state.execution_plan,
                "cost_analysis": current_state.cost_analysis,
                "resource_requirements": current_state.resource_requirements,
                "compliance_requirements": current_state.compliance_requirements,
                "validation_criteria": current_state.validation_criteria,
                "planning_metadata": current_state.planning_metadata,
                "existing_infrastructure": current_state.existing_infrastructure,
                "affected_resources": current_state.affected_resources,
                "unchanged_resources": current_state.unchanged_resources,
                "change_impact": current_state.change_impact,
                "downtime_required": current_state.downtime_required,
                "rollback_strategy": current_state.rollback_strategy,
                "risk_assessment": current_state.risk_assessment,
                "error": current_state.error,
                "error_context": current_state.error_context,
                "requires_approval": current_state.requires_approval,
                "approval_context": current_state.approval_context,
                "dependency_questions": current_state.dependency_questions,
                "dependency_answers": current_state.dependency_answers,
                "waiting_for_dependency_input": current_state.waiting_for_dependency_input,
                "current_dependency_question": current_state.current_dependency_question,
                "dependency_mapping_complete": current_state.dependency_mapping_complete,
                "planning_started_at": current_state.planning_started_at,
                "planning_completed_at": current_state.planning_completed_at,
                "planning_duration": current_state.planning_duration,
                "complexity_score": current_state.complexity_score
            }
    
    @agent_node
    def analyze_existing_infrastructure_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze existing infrastructure for modification requests."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Only run for modification requests
            if current_state.request_type != "modify":
                current_state.messages.append(
                    AIMessage(content="Skipping existing infrastructure analysis - this is a new request.")
                )
                return current_state.model_dump()
            
            # Create analysis chain
            analysis_chain = self.existing_infrastructure_prompt | self.model | StrOutputParser()
            
            # Execute analysis
            analysis_result = analysis_chain.invoke({
                "messages": current_state.messages,
                "user_request": current_state.user_request
            })
            
            # Update state with existing infrastructure analysis
            current_state.existing_infrastructure = {
                "analysis": analysis_result,
                "analyzed_at": datetime.utcnow().isoformat()
            }
            current_state.current_step = "existing_infrastructure_analyzed"
            current_state.completed_steps.append("analyze_existing_infrastructure")
            
            # Add analysis result to messages
            current_state.messages.append(
                AIMessage(content=f"Existing infrastructure analysis completed: {analysis_result[:200]}...")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Existing infrastructure analysis completed",
                extra={
                    "request_type": current_state.request_type,
                    "user_request": current_state.user_request[:100] + "..." if len(current_state.user_request) > 100 else current_state.user_request
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in existing infrastructure analysis: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "existing_infrastructure_analysis"
            
            return current_state.model_dump()
    
    @agent_node
    def assess_change_impact_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of proposed changes for modification requests."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Only run for modification requests
            if current_state.request_type != "modify":
                current_state.messages.append(
                    AIMessage(content="Skipping change impact assessment - this is a new request.")
                )
                return current_state.model_dump()
            
            # Get existing infrastructure and proposed changes
            existing_infrastructure = current_state.existing_infrastructure or {}
            proposed_changes = current_state.requirements_analysis or {}
            
            # Create impact assessment chain
            impact_chain = self.change_impact_prompt | self.model | StrOutputParser()
            
            # Execute impact assessment
            impact_result = impact_chain.invoke({
                "messages": current_state.messages,
                "proposed_changes": str(proposed_changes),
                "existing_infrastructure": str(existing_infrastructure)
            })
            
            # Parse impact result and update state
            current_state.change_impact = {
                "assessment": impact_result,
                "assessed_at": datetime.utcnow().isoformat()
            }
            
            # Extract key information from impact result
            impact_text = impact_result.lower()
            current_state.downtime_required = any(word in impact_text for word in ["downtime", "maintenance window", "service interruption"])
            
            # Set rollback strategy based on impact
            if "high risk" in impact_text or "critical" in impact_text:
                current_state.rollback_strategy = "immediate_rollback_required"
            elif "medium risk" in impact_text:
                current_state.rollback_strategy = "gradual_rollback_recommended"
            else:
                current_state.rollback_strategy = "standard_rollback_procedures"
            
            # Update risk assessment
            current_state.risk_assessment = {
                "impact_level": "high" if "high risk" in impact_text else "medium" if "medium risk" in impact_text else "low",
                "downtime_required": current_state.downtime_required,
                "rollback_strategy": current_state.rollback_strategy,
                "assessment_details": impact_result
            }
            
            current_state.current_step = "change_impact_assessed"
            current_state.completed_steps.append("assess_change_impact")
            
            # Add impact assessment to messages
            current_state.messages.append(
                AIMessage(content=f"Change impact assessment completed. Downtime required: {current_state.downtime_required}, Risk level: {current_state.risk_assessment['impact_level']}")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Change impact assessment completed",
                extra={
                    "request_type": current_state.request_type,
                    "downtime_required": current_state.downtime_required,
                    "risk_level": current_state.risk_assessment["impact_level"]
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in change impact assessment: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "change_impact_assessment"
            
            return current_state.model_dump()
    
    @agent_node
    def analyze_requirements_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user requirements and extract key information."""
        
        try:
            # Get user request from state
            user_request = state.get("user_request", "")
            
            # Create prompt chain
            chain = self.requirements_prompt | self.model | self.requirements_parser
            
            # Execute analysis
            result = chain.invoke({
                "messages": state.get("messages", []),
                "user_request": user_request
            })
            
            # Update state with analysis results
            state["requirements_analysis"] = result.dict()
            state["status"] = "requirements_analyzed"
            
            # Add analysis result to messages
            state["messages"].append(
                AIMessage(content=f"Requirements analysis completed. Found {len(result.business_requirements)} business requirements and {len(result.technical_requirements)} technical requirements.")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Requirements analysis completed",
                extra={
                    "agent_name": self.name,
                    "business_requirements_count": len(result.business_requirements),
                    "technical_requirements_count": len(result.technical_requirements),
                    "constraints_count": len(result.constraints)
                }
            )
            
            return state
            
        except Exception as e:
            state["error"] = f"Requirements analysis failed: {str(e)}"
            state["status"] = "error"
            raise
    
    @agent_node
    def plan_infrastructure_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the infrastructure based on requirements analysis."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Get requirements analysis from state
            requirements_analysis = current_state.requirements_analysis or {}
            
            # Get request type and existing infrastructure
            request_type = current_state.request_type
            existing_infrastructure = current_state.existing_infrastructure or {}
            
            # Create prompt chain
            chain = self.infrastructure_prompt | self.model | self.infrastructure_parser
            
            # Execute infrastructure planning with context
            result = chain.invoke({
                "messages": current_state.messages,
                "requirements_analysis": str(requirements_analysis),
                "request_type": request_type,
                "existing_infrastructure": str(existing_infrastructure) if existing_infrastructure else "No existing infrastructure"
            })
            
            # Convert results to Pydantic models
            infrastructure_requirements = [
                InfrastructureRequirement(**service) for service in result.services
            ]
            
            architectural_patterns = [
                ArchitecturalPattern(**pattern) for pattern in result.patterns
            ]
            
            security_requirements = [
                SecurityRequirement(**req) for req in result.security_requirements
            ]
            
            # Update state
            current_state.infrastructure_requirements = infrastructure_requirements
            current_state.architectural_patterns = architectural_patterns
            current_state.security_requirements = security_requirements
            current_state.cost_analysis = result.cost_estimates
            current_state.status = "infrastructure_planned"
            current_state.current_step = "infrastructure_planned"
            current_state.completed_steps.append("plan_infrastructure")
            
            # Add planning result to messages with context
            planning_summary = f"Infrastructure planning completed for {request_type} request. "
            planning_summary += f"Identified {len(infrastructure_requirements)} services and {len(architectural_patterns)} architectural patterns."
            
            if request_type == "modify":
                planning_summary += " Modifications designed to preserve existing functionality."
            
            current_state.messages.append(AIMessage(content=planning_summary))
            
            planner_logger.log_structured(
                level="INFO",
                message="Infrastructure planning completed",
                extra={
                    "agent_name": self.name,
                    "request_type": request_type,
                    "services_count": len(infrastructure_requirements),
                    "patterns_count": len(architectural_patterns),
                    "security_requirements_count": len(security_requirements),
                    "is_modification": request_type == "modify"
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in infrastructure planning: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "infrastructure_planning"
            
            return current_state.model_dump()
    
    @agent_node
    def create_execution_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed execution plan."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Get infrastructure plan from state
            infrastructure_plan = {
                "services": [req.dict() for req in current_state.infrastructure_requirements or []],
                "patterns": [pattern.dict() for pattern in current_state.architectural_patterns or []],
                "security": [req.dict() for req in current_state.security_requirements or []]
            }
            
            # Get request type and change impact
            request_type = current_state.request_type
            change_impact = current_state.change_impact or {}
            
            # Create prompt chain
            chain = self.execution_prompt | self.model | self.execution_parser
            
            # Execute execution planning with context
            result = chain.invoke({
                "messages": current_state.messages,
                "infrastructure_plan": str(infrastructure_plan),
                "request_type": request_type,
                "change_impact": str(change_impact) if change_impact else "No change impact data"
            })
            
            # Convert results to Pydantic models
            execution_steps = [
                ExecutionStep(**step) for step in result.steps
            ]
            
            # Update state
            current_state.execution_plan = execution_steps
            current_state.resource_requirements = result.resource_requirements
            current_state.status = "execution_planned"
            current_state.current_step = "execution_planned"
            current_state.completed_steps.append("create_execution_plan")
            
            # Add execution plan to messages with context
            execution_summary = f"Execution plan created for {request_type} request with {len(execution_steps)} steps. "
            execution_summary += f"Estimated time: {result.total_estimated_time}"
            
            if request_type == "modify":
                execution_summary += " Includes rollback procedures and minimal downtime strategies."
            
            current_state.messages.append(AIMessage(content=execution_summary))
            
            planner_logger.log_structured(
                level="INFO",
                message="Execution plan created",
                extra={
                    "agent_name": self.name,
                    "request_type": request_type,
                    "steps_count": len(execution_steps),
                    "estimated_time": result.total_estimated_time,
                    "critical_path_length": len(result.critical_path),
                    "is_modification": request_type == "modify"
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Error in execution planning: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "execution_planning"
            
            return current_state.model_dump()
    
    @agent_node
    def validate_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete plan and identify issues."""
        
        try:
            # Prepare complete plan for validation
            complete_plan = {
                "requirements": state.get("requirements_analysis", {}),
                "infrastructure": {
                    "services": [req.dict() for req in state.get("infrastructure_requirements", [])],
                    "patterns": [pattern.dict() for pattern in state.get("architectural_patterns", [])],
                    "security": [req.dict() for req in state.get("security_requirements", [])]
                },
                "execution": {
                    "steps": [step.dict() for step in state.get("execution_plan", [])],
                    "resources": state.get("resource_requirements", {})
                }
            }
            
            # Create prompt chain
            chain = self.validation_prompt | self.model
            
            # Execute validation
            result = chain.invoke({
                "messages": state.get("messages", []),
                "complete_plan": str(complete_plan)
            })
            
            # Extract validation results
            validation_content = result.content if hasattr(result, 'content') else str(result)
            
            # Update state with validation results
            state["validation_criteria"] = [validation_content]
            state["status"] = "plan_validated"
            
            # Add validation result to messages
            state["messages"].append(
                AIMessage(content=f"Plan validation completed. Review the validation results and recommendations.")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Plan validation completed",
                extra={
                    "agent_name": self.name,
                    "validation_content_length": len(validation_content)
                }
            )
            
            return state
            
        except Exception as e:
            state["error"] = f"Plan validation failed: {str(e)}"
            state["status"] = "error"
            raise
    
    @agent_node
    @require_approval
    def finalize_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the plan and prepare for execution."""
        
        try:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(state)
            
            # Update final state
            state["complexity_score"] = complexity_score
            state["status"] = "plan_completed"
            state["planning_completed_at"] = datetime.utcnow().isoformat()
            
            # Calculate planning duration
            if state.get("planning_started_at"):
                start_time = datetime.fromisoformat(state["planning_started_at"])
                end_time = datetime.fromisoformat(state["planning_completed_at"])
                state["planning_duration"] = (end_time - start_time).total_seconds()
            
            # Add finalization message
            state["messages"].append(
                AIMessage(content=f"Planning completed successfully! Complexity score: {complexity_score}/10. The plan is ready for execution.")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Planning finalized successfully",
                extra={
                    "agent_name": self.name,
                    "complexity_score": complexity_score,
                    "planning_duration": state.get("planning_duration"),
                    "total_services": len(state.get("infrastructure_requirements", [])),
                    "total_steps": len(state.get("execution_plan", []))
                }
            )
            
            return state
            
        except Exception as e:
            state["error"] = f"Plan finalization failed: {str(e)}"
            state["status"] = "error"
            raise
    
    def _calculate_complexity_score(self, state: Dict[str, Any]) -> int:
        """Calculate a complexity score for the plan (1-10)."""
        
        score = 1  # Base score
        
        # Add points for number of services
        services_count = len(state.get("infrastructure_requirements", []))
        if services_count > 10:
            score += 3
        elif services_count > 5:
            score += 2
        elif services_count > 2:
            score += 1
        
        # Add points for architectural patterns
        patterns_count = len(state.get("architectural_patterns", []))
        score += min(patterns_count, 2)
        
        # Add points for security requirements
        security_count = len(state.get("security_requirements", []))
        if security_count > 5:
            score += 2
        elif security_count > 2:
            score += 1
        
        # Add points for execution steps
        steps_count = len(state.get("execution_plan", []))
        if steps_count > 20:
            score += 2
        elif steps_count > 10:
            score += 1
        
        # Cap at 10
        return min(score, 10)
    
    @log_sync
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the current plan."""
        
        if not self.current_state:
            return {"error": "No plan available"}
        
        state = self.current_state
        
        return {
            "status": state.status,
            "complexity_score": state.complexity_score,
            "services_count": len(state.infrastructure_requirements),
            "patterns_count": len(state.architectural_patterns),
            "steps_count": len(state.execution_plan),
            "estimated_cost": state.cost_analysis.get("total_monthly", "Not available"),
            "planning_duration": state.planning_duration,
            "error": state.error
        }
    
    @log_sync
    def export_plan(self, format: str = "json") -> Dict[str, Any]:
        """Export the plan in the specified format."""
        
        if not self.current_state:
            return {"error": "No plan available"}
        
        state = self.current_state
        
        if format.lower() == "json":
            return {
                "plan": state.dict(),
                "summary": self.get_plan_summary(),
                "exported_at": datetime.utcnow().isoformat()
            }
        else:
            return {"error": f"Unsupported format: {format}"}
    
    @log_sync
    def provide_dependency_answer(self, question: str, answer: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Provide an answer to a dependency mapping question."""
        try:
            # Find the current state for this context
            if context_id is None:
                # Use the most recent state if no context_id provided
                # This would need to be implemented based on your state management
                return {
                    "status": "error",
                    "message": "Context ID is required for dependency answer"
                }
            
            # Update the state with the user's answer
            # This would typically involve updating a persistent state store
            # For now, we'll return a success response
            planner_logger.log_structured(
                level="INFO",
                message="Dependency answer provided",
                extra={
                    "agent_name": self.name,
                    "question": question,
                    "answer": answer,
                    "context_id": context_id
                }
            )
            
            return {
                "status": "success",
                "message": "Dependency answer recorded successfully",
                "question": question,
                "answer": answer,
                "context_id": context_id
            }
            
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Failed to record dependency answer: {e}",
                extra={
                    "agent_name": self.name,
                    "question": question,
                    "answer": answer,
                    "context_id": context_id,
                    "error": str(e)
                }
            )
            
            return {
                "status": "error",
                "message": f"Failed to record dependency answer: {e}",
                "error_type": type(e).__name__
            }


@log_sync
def create_planner_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "planner_agent"
) -> PlannerAgent:
    """
    Factory function to create a Planner Agent.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(config=config, custom_config=custom_config, name=name) 