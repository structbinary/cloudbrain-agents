"""
Planner Agent for Custom Supervisor Architecture.

This module implements the Planner Agent as a BaseSubgraphAgent that works with
the custom supervisor using LangGraph's Send() pattern. It adapts the existing
comprehensive planner implementation to the new architecture.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from functools import wraps
import traceback

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from ..base_agent import BaseSubgraphAgent
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.utils.mcp_client import create_mcp_client
from ...types import PlannerState, AgentType
from .new_infrastructure_subgraph import NewInfrastructureSubgraph

# Create agent logger for planner
planner_logger = AgentLogger("PLANNER_AGENT")


class PlannerAgent(BaseSubgraphAgent):
    """
    Planner Agent that works with the custom supervisor using Send() pattern.
    
    This agent adapts the existing comprehensive planner implementation to work
    with the new custom supervisor architecture following the LangGraph tutorial pattern.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "planner_agent"
    ):
        """
        Initialize the Planner Agent with centralized configuration.
        
        Args:
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
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
            planner_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model for planner agent: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            planner_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model for planner agent: {e}",
                extra={"error": str(e)}
            )
            raise
        
        # Initialize output parsers
        self.requirements_parser = JsonOutputParser(pydantic_object=self._create_requirements_schema())
        self.infrastructure_parser = JsonOutputParser(pydantic_object=self._create_infrastructure_schema())
        self.execution_parser = JsonOutputParser(pydantic_object=self._create_execution_schema())
        self.validation_parser = JsonOutputParser(pydantic_object=self._create_validation_schema())
        
        # Define prompts
        self._define_prompts()
        self.memory = MemorySaver()
        
        # Initialize subgraphs for different request types
        self.new_infrastructure_subgraph = NewInfrastructureSubgraph(
            model=self.model,
            config=self.config_instance,
            mcp_client_factory=create_mcp_client,
            mcp_params={
                "host": self.config_instance.TERRAFORM_MCP_SERVER_HOST,
                "port": self.config_instance.TERRAFORM_MCP_SERVER_PORT,
                "transport": self.config_instance.TERRAFORM_MCP_SERVER_TRANSPORT,
            },
        )
        
        # Build the agent graph
        self._graph = self.build_graph()
        
        planner_logger.log_structured(
            level="INFO",
            message="Planner Agent initialized successfully with custom supervisor architecture",
            extra={
                "agent_name": name,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model']
            }
        )
    
    @property
    def name(self) -> str:
        """Get the name of the planner agent."""
        return self._name
    
    @property
    def state_model(self) -> type[BaseModel]:
        """Get the state model for this agent."""
        return PlannerState
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for the planner agent.
        
        This method creates the workflow graph following the LangGraph tutorial pattern.
        
        Returns:
            StateGraph: The compiled graph for this agent
        """
        # Create the graph with PlannerState
        graph = StateGraph(PlannerState)
        
        # Add nodes for the planning workflow
        graph.add_node("classify_request", self._classify_request_node)
        
        # Add new infrastructure workflow compiled with this agent's checkpointer (will be overwritten by supervisor)
        new_infra_graph = self.new_infrastructure_subgraph.build_graph().compile(checkpointer=self.memory)
        graph.add_node("new_infrastructure_workflow", new_infra_graph)
        
        # Add modify infrastructure workflow (placeholder for now)
        graph.add_node("modify_infrastructure_workflow", self._modify_infrastructure_workflow_node)
        
        # Add an interrupt gate that raises HITL when flags are set by subgraphs
        graph.add_node("interrupt_gate", self._interrupt_gate_node)

        # Add finalization node
        graph.add_node("finalize_plan", self._finalize_plan_node)
        
        # Add conditional routing based on request type
        graph.add_conditional_edges(
            "classify_request",
            self._route_after_classification,
            {
                "new_infrastructure_workflow": "new_infrastructure_workflow",
                "modify_infrastructure_workflow": "modify_infrastructure_workflow"
            }
        )
        
        # Route subgraphs through the interrupt gate before finalization
        graph.add_edge("new_infrastructure_workflow", "interrupt_gate")
        graph.add_edge("modify_infrastructure_workflow", "interrupt_gate")
        
        # Add conditional routing from interrupt gate
        graph.add_conditional_edges(
            "interrupt_gate",
            self._route_after_interrupt_gate,
            {
                "continue": "finalize_plan",
                "interrupt": END
            }
        )
        
        # Set entry and exit points
        graph.set_entry_point("classify_request")
        graph.set_finish_point("finalize_plan")
        
        return graph
    
    # Removed: _new_infrastructure_workflow_node (subgraph mounted natively)
    
    def _interrupt_gate_node(self, state: PlannerState) -> PlannerState:
        """Check for interrupt flags and return state with interrupt context; let graph routing handle propagation."""
        
        # Check for interrupt flags set by nested subgraphs
        if getattr(state, 'waiting_for_dependency_input', False):
            # Extract interrupt data from state
            interrupt_data = {
                "context": "dependency_mapping",
                "question": getattr(state, 'current_dependency_question', "Dependency mapping question"),
                "available_questions": getattr(state, 'dependency_questions', []),
                "partial_analysis": getattr(getattr(state, 'planning_metadata', {}), 'dependency_mapping_partial', {}),
                "dependency_answers": getattr(state, 'dependency_answers', {}),
                "mapping_complete": getattr(state, 'dependency_mapping_complete', False)
            }
            
            planner_logger.log_structured(
                level="INFO",
                message="Interrupt gate detected dependency mapping interrupt from nested subgraph",
                extra={
                    "question": interrupt_data.get('question'),
                    "available_questions_count": len(interrupt_data.get('available_questions', [])),
                    "interrupt_context": "dependency_mapping"
                }
            )
            
            # Set interrupt context in state and return - let the graph routing handle it
            state.interrupt_context = interrupt_data
            state.interrupt_required = True
            return state
            
        elif getattr(state, 'requires_approval', False):
            # Handle approval interrupts
            approval_context = getattr(state, 'approval_context', {})
            interrupt_data = {
                "context": "approval_required",
                "message": approval_context.get('message', "Approval required"),
                "node": getattr(state, 'current_step', "unknown"),
                "approval_context": approval_context
            }
            
            planner_logger.log_structured(
                level="INFO",
                message="Interrupt gate detected approval interrupt from nested subgraph",
                extra={
                    "approval_message": interrupt_data.get('message'),
                    "node": interrupt_data.get('node'),
                    "interrupt_context": "approval_required"
                }
            )
            
            # Set interrupt context in state and return - let the graph routing handle it
            state.interrupt_context = interrupt_data
            state.interrupt_required = True
            return state
        
        # No interrupt needed, return state as-is
        return state
    
    def _route_after_interrupt_gate(self, state: PlannerState) -> str:
        """
        Route after interrupt gate based on whether interrupt is needed.
        
        Args:
            state: Current planner state
            
        Returns:
            str: Route destination - "continue" or "interrupt"
        """
        # Check if interrupt is required
        if getattr(state, 'interrupt_required', False):
            planner_logger.log_structured(
                level="INFO",
                message="Interrupt gate routing to END (interrupt)",
                extra={
                    "interrupt_context": getattr(state, 'interrupt_context', {}).get('context', 'unknown'),
                    "interrupt_required": True
                }
            )
            return "interrupt"
        else:
            planner_logger.log_structured(
                level="INFO",
                message="Interrupt gate routing to finalize_plan (continue)",
                extra={
                    "interrupt_required": False
                }
            )
            return "continue"

    def _modify_infrastructure_workflow_node(self, state: PlannerState) -> PlannerState:
        """Placeholder for modify infrastructure workflow."""
        
        # For now, just mark as completed and add a message
        state.status = "modify_workflow_completed"
        state.current_step = "modify_workflow_completed"
        if not hasattr(state, 'completed_steps'):
            state.completed_steps = []
        state.completed_steps.append("modify_infrastructure_workflow")
        
        # Add placeholder message
        state.messages.append(
            AIMessage(content="Modify infrastructure workflow is not yet implemented. This will be added in a future update.")
        )
        
        planner_logger.log_structured(
            level="INFO",
            message="Modify infrastructure workflow placeholder executed",
            extra={
                "user_request": state.user_request[:100] + "..." if len(state.user_request) > 100 else state.user_request
            }
        )
        
        return state
    
    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Send() payload from supervisor to planner state.
        
        Args:
            send_payload: Data sent from supervisor via Send() primitive
            
        Returns:
            Dict[str, Any]: Transformed state ready for planner processing
        """
        # Extract task description from Send payload
        task_description = ""
        if "messages" in send_payload and send_payload["messages"]:
            task_description = send_payload["messages"][0].get("content", "")
        
        # Transform to planner state format
        return {
            "user_request": task_description,
            "messages": send_payload.get("messages", []),
            "supervisor_context": send_payload.get("context", {}),
            "planning_started_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized"
        }
    
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform planner state back to supervisor state.
        
        Args:
            agent_state: Current planner state
            
        Returns:
            Dict[str, Any]: Data to merge back into supervisor state
        """
        # Only include data that should propagate to supervisor
        return {
            "planning_complete": agent_state.get("planning_complete", False),
            "handoff_context": {
                "requirements_analysis": agent_state.get("requirements_analysis", {}),
                "infrastructure_requirements": agent_state.get("infrastructure_requirements", []),
                "architectural_patterns": agent_state.get("architectural_patterns", []),
                "security_requirements": agent_state.get("security_requirements", []),
                "execution_plan": agent_state.get("execution_plan", []),
                "cost_analysis": agent_state.get("cost_analysis", {}),
                "complexity_score": agent_state.get("complexity_score", 0),
                "planning_metadata": agent_state.get("planning_metadata", {})
            },
            "messages": agent_state.get("messages", []),
            "status": agent_state.get("status", "completed")
        }
    
    def can_interrupt(self) -> bool:
        """Check if the agent can be interrupted for human-in-the-loop."""
        return True
    
    def handle_interrupt(self, state: PlannerState) -> PlannerState:
        """
        Handle interruption for human-in-the-loop interaction.
        
        Args:
            state: Current agent state
            
        Returns:
            PlannerState: Updated state after interrupt handling
        """
        # Check if we need dependency mapping input
        if getattr(state, 'waiting_for_dependency_input', False):
            # Trigger interrupt for dependency question
            raise interrupt({
                "context": "dependency_mapping",
                "question": getattr(state, 'current_dependency_question', "Dependency mapping question"),
                "available_questions": getattr(state, 'dependency_questions', []),
                "partial_analysis": getattr(getattr(state, 'planning_metadata', {}), 'dependency_mapping_partial', {})
            })
        
        # Check if approval is required
        if getattr(state, 'requires_approval', False):
            # Trigger interrupt for approval
            raise interrupt({
                "context": "approval_required",
                "message": getattr(getattr(state, 'approval_context', {}), 'message', "Approval required"),
                "node": getattr(state, 'current_step', "unknown")
            })
        
        return state
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tools available to this agent."""
        return [
            {
                "name": "dependency_mapping_tool",
                "description": "Tool for mapping AWS service dependencies",
                "function": self._dependency_mapping_tool
            }
        ]
    
    def validate_state(self, state: PlannerState) -> bool:
        """Validate the agent state."""
        try:
            # If it's already a PlannerState, it's valid
            if isinstance(state, PlannerState):
                return True
            # If it's a dict, try to create PlannerState from it
            PlannerState(**state)
            return True
        except Exception:
            return False
    
    def preprocess_state(self, state: PlannerState) -> PlannerState:
        """Preprocess state before processing."""
        # Ensure required fields are present
        if not hasattr(state, 'planning_started_at') or not state.planning_started_at:
            state.planning_started_at = datetime.now(timezone.utc).isoformat()
        
        if not hasattr(state, 'status') or not state.status:
            state.status = "initialized"
        
        return state
    
    def postprocess_result(self, result: PlannerState) -> PlannerState:
        """Postprocess result before returning to supervisor."""
        # Mark planning as complete
        result.planning_complete = True
        result.planning_completed_at = datetime.now(timezone.utc).isoformat()
        
        # Calculate planning duration
        if hasattr(result, 'planning_started_at') and result.planning_started_at:
            start_time = datetime.fromisoformat(result.planning_started_at)
            end_time = datetime.fromisoformat(result.planning_completed_at)
            result.planning_duration = (end_time - start_time).total_seconds()
        
        return result
    
    # ============================================================================
    # GRAPH NODES
    # ============================================================================
    
    def _classify_request_node(self, state: PlannerState) -> PlannerState:
        """Classify the request type (new vs modify)."""
        try:
            user_request = state.user_request
            
            # Create classification prompt
            classification_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at classifying infrastructure requests.
                
                Classify the user request as either:
                - "new": For requests to create new infrastructure from scratch
                - "modify": For requests to modify existing infrastructure
                
                Respond with only the classification: "new" or "modify"
                """),
                ("human", "Classify this request: {user_request}")
            ])
            
            # Execute classification
            chain = classification_prompt | self.model | StrOutputParser()
            result = chain.invoke({"user_request": user_request})
            
            # Extract classification
            request_type = result.strip().lower()
            if request_type not in ["new", "modify"]:
                request_type = "new"  # Default to new
            
            # Update state using Pydantic model
            state.request_type = request_type
            state.status = "classified"
            state.current_step = "request_classified"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("classify_request")
            
            # Add classification result to messages
            state.messages.append(
                AIMessage(content=f"Request classified as: {request_type}")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Request classification completed",
                extra={
                    "request_type": request_type,
                    "user_request": user_request[:100] + "..." if len(user_request) > 100 else user_request
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Request classification failed: {str(e)}"
            state.status = "error"
            raise
    
    def _route_after_classification(self, state: PlannerState) -> str:
        """Route to appropriate workflow based on request classification."""
        request_type = getattr(state, 'request_type', 'new')
        
        if request_type == "new":
            return "new_infrastructure_workflow"
        elif request_type == "modify":
            return "modify_infrastructure_workflow"
        else:
            # Default to new infrastructure workflow
            return "new_infrastructure_workflow"
    
    def _analyze_requirements_node(self, state: PlannerState) -> PlannerState:
        """Analyze user requirements and extract key information."""
        try:
            user_request = state.user_request
            request_type = getattr(state, 'request_type', 'new')
            
            # Create requirements analysis prompt
            requirements_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AWS infrastructure requirements analyst.
                
                Analyze the user request and identify:
                1. Business requirements and objectives
                2. Technical requirements and specifications
                3. Constraints and limitations
                4. Assumptions and prerequisites
                5. Risk factors and mitigation strategies
                
                Provide a comprehensive analysis in JSON format.
                """),
                ("human", "Analyze requirements for this {request_type} infrastructure request: {user_request}")
            ])
            
            # Execute analysis
            chain = requirements_prompt | self.model | self.requirements_parser
            result = chain.invoke({
                "user_request": user_request,
                "request_type": request_type
            })
            
            # Debug logging to understand result type
            planner_logger.log_structured(
                level="DEBUG",
                message="Requirements analysis result received",
                extra={
                    "result_type": type(result).__name__,
                    "has_dict_method": hasattr(result, 'dict'),
                    "result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict"
                }
            )
            
            # Update state using Pydantic model
            # Handle both Pydantic objects and dictionaries
            if hasattr(result, 'dict'):
                # It's a Pydantic object
                requirements_data = result.dict()
                business_requirements_count = len(getattr(result, 'business_requirements', []))
                technical_requirements_count = len(getattr(result, 'technical_requirements', []))
            else:
                # It's already a dictionary
                requirements_data = result
                # Extract counts from the dictionary structure
                business_requirements_count = len(result.get('BusinessRequirementsAndObjectives', {}).get('Requirements', []))
                technical_requirements_count = len(result.get('TechnicalRequirementsAndSpecifications', {}).get('Specifications', []))
            
            state.requirements_analysis = requirements_data
            state.status = "requirements_analyzed"
            state.current_step = "requirements_analyzed"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("analyze_requirements")
            
            # Add analysis result to messages
            state.messages.append(
                AIMessage(content=f"Requirements analysis completed. Found {business_requirements_count} business requirements and {technical_requirements_count} technical requirements.")
            )
            
            # Calculate counts based on the actual result structure
            if hasattr(result, 'dict'):
                # It's a Pydantic object
                business_requirements_count = len(getattr(result, 'business_requirements_and_objectives', {}))
                technical_requirements_count = len(getattr(result, 'technical_requirements_and_specifications', {}))
            else:
                # It's already a dictionary - handle the actual structure
                business_requirements_count = len(result.get('business_requirements_and_objectives', {}))
                technical_requirements_count = len(result.get('technical_requirements_and_specifications', {}))
            
            planner_logger.log_structured(
                level="INFO",
                message="Requirements analysis completed",
                extra={
                    "request_type": request_type,
                    "business_requirements_count": business_requirements_count,
                    "technical_requirements_count": technical_requirements_count,
                    "result_structure": list(result.keys()) if isinstance(result, dict) else "not_dict"
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Requirements analysis failed: {str(e)}"
            state.status = "error"
            raise
    
    def _plan_infrastructure_node(self, state: PlannerState) -> PlannerState:
        """Plan the infrastructure based on requirements analysis."""
        try:
            requirements_analysis = getattr(state, 'requirements_analysis', {})
            request_type = getattr(state, 'request_type', 'new')
            
            # Create infrastructure planning prompt
            infrastructure_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AWS architect.
                
                Design infrastructure based on requirements:
                1. Required AWS services with optimal configurations
                2. Modern architectural patterns (microservices, event-driven, serverless, etc.)
                3. Security requirements and best practices
                4. Cost estimates and optimization strategies
                5. Service dependencies and integration points
                
                Provide a comprehensive infrastructure plan in JSON format.
                """),
                ("human", "Create infrastructure plan for {request_type} request based on: {requirements_analysis}")
            ])
            
            # Execute planning
            chain = infrastructure_prompt | self.model | self.infrastructure_parser
            result = chain.invoke({
                "requirements_analysis": str(requirements_analysis),
                "request_type": request_type
            })
            
            # Update state using Pydantic model
            # Handle both Pydantic objects and dictionaries
            if hasattr(result, 'dict'):
                # It's a Pydantic object
                state.infrastructure_requirements = result.services
                state.architectural_patterns = result.patterns
                state.security_requirements = result.security_requirements
                state.cost_analysis = result.cost_estimates
                services_count = len(result.services)
                patterns_count = len(result.patterns)
            else:
                # It's already a dictionary
                state.infrastructure_requirements = result.get('services', [])
                state.architectural_patterns = result.get('patterns', [])
                state.security_requirements = result.get('security_requirements', [])
                state.cost_analysis = result.get('cost_estimates', {})
                services_count = len(result.get('services', []))
                patterns_count = len(result.get('patterns', []))
            
            state.status = "infrastructure_planned"
            state.current_step = "infrastructure_planned"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("plan_infrastructure")
            
            # Add planning result to messages
            state.messages.append(
                AIMessage(content=f"Infrastructure planning completed. Identified {services_count} services and {patterns_count} architectural patterns.")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Infrastructure planning completed",
                extra={
                    "request_type": request_type,
                    "services_count": len(result.services),
                    "patterns_count": len(result.patterns)
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Infrastructure planning failed: {str(e)}"
            state.status = "error"
            raise
    
    def _create_execution_plan_node(self, state: PlannerState) -> PlannerState:
        """Create a detailed execution plan."""
        try:
            infrastructure_plan = {
                "services": getattr(state, 'infrastructure_requirements', []),
                "patterns": getattr(state, 'architectural_patterns', []),
                "security": getattr(state, 'security_requirements', [])
            }
            request_type = getattr(state, 'request_type', 'new')
            
            # Create execution planning prompt
            execution_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert DevOps engineer.
                
                Create a detailed step-by-step execution plan:
                1. Sequential steps with clear actions and commands
                2. Dependencies between steps and services
                3. Time estimates for each phase
                4. Resource requirements and provisioning
                5. Validation criteria for each step
                6. Deployment strategy
                7. Testing phases and validation procedures
                
                Provide a comprehensive execution plan in JSON format.
                """),
                ("human", "Create execution plan for {request_type} request: {infrastructure_plan}")
            ])
            
            # Execute planning
            chain = execution_prompt | self.model | self.execution_parser
            result = chain.invoke({
                "infrastructure_plan": str(infrastructure_plan),
                "request_type": request_type
            })
            
            # Update state using Pydantic model
            # Handle both Pydantic objects and dictionaries
            if hasattr(result, 'dict'):
                # It's a Pydantic object
                state.execution_plan = result.steps
                state.resource_requirements = result.resource_requirements
                steps_count = len(result.steps)
                estimated_time = result.total_estimated_time
            else:
                # It's already a dictionary
                state.execution_plan = result.get('steps', [])
                state.resource_requirements = result.get('resource_requirements', {})
                steps_count = len(result.get('steps', []))
                estimated_time = result.get('total_estimated_time', 'Unknown')
            
            state.status = "execution_planned"
            state.current_step = "execution_planned"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("create_execution_plan")
            
            # Add execution plan to messages
            state.messages.append(
                AIMessage(content=f"Execution plan created with {steps_count} steps. Estimated time: {estimated_time}")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Execution plan created",
                extra={
                    "request_type": request_type,
                    "steps_count": len(result.steps),
                    "estimated_time": result.total_estimated_time
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Execution planning failed: {str(e)}"
            state.status = "error"
            raise
    
    def _validate_plan_node(self, state: PlannerState) -> PlannerState:
        """Validate the complete plan and identify issues."""
        try:
            complete_plan = {
                "requirements": getattr(state, 'requirements_analysis', {}),
                "infrastructure": {
                    "services": getattr(state, 'infrastructure_requirements', []),
                    "patterns": getattr(state, 'architectural_patterns', []),
                    "security": getattr(state, 'security_requirements', [])
                },
                "execution": {
                    "steps": getattr(state, 'execution_plan', []),
                    "resources": getattr(state, 'resource_requirements', {})
                }
            }
            
            # Create validation prompt
            validation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert infrastructure validator.
                
                Review the complete plan and identify:
                1. Security vulnerabilities and compliance gaps
                2. Cost optimization opportunities
                3. Scalability and performance concerns
                4. Best practice violations
                5. Architectural design issues
                6. Deployment strategy problems
                7. Testing and validation gaps
                
                Provide a comprehensive validation report in JSON format.
                """),
                ("human", "Validate this complete plan: {complete_plan}")
            ])
            
            # Execute validation
            chain = validation_prompt | self.model | self.validation_parser
            result = chain.invoke({"complete_plan": str(complete_plan)})
            
            # Update state using Pydantic model
            # Handle both Pydantic objects and dictionaries
            if hasattr(result, 'dict'):
                # It's a Pydantic object
                overall_assessment = result.overall_assessment
                risk_level = result.risk_level
            else:
                # It's already a dictionary
                overall_assessment = result.get('overall_assessment', 'Unknown')
                risk_level = result.get('risk_level', 'Unknown')
            
            state.validation_criteria = [overall_assessment]
            state.status = "plan_validated"
            state.current_step = "plan_validated"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("validate_plan")
            
            # Add validation result to messages
            state.messages.append(
                AIMessage(content=f"Plan validation completed. Overall assessment: {overall_assessment}")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Plan validation completed",
                extra={
                    "overall_assessment": overall_assessment,
                    "risk_level": risk_level
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Plan validation failed: {str(e)}"
            state.status = "error"
            raise
    
    def _finalize_plan_node(self, state: PlannerState) -> PlannerState:
        """Finalize the plan and prepare for handoff to supervisor."""
        try:
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(state)
            
            # Update final state using Pydantic model
            state.complexity_score = complexity_score
            state.status = "plan_completed"
            state.planning_completed_at = datetime.now(timezone.utc).isoformat()
            state.planning_complete = True
            
            # Calculate planning duration
            if hasattr(state, 'planning_started_at') and state.planning_started_at:
                # Ensure planning_started_at is a string
                start_time_str = str(state.planning_started_at)
                if start_time_str and start_time_str != 'None':
                    try:
                        start_time = datetime.fromisoformat(start_time_str)
                        end_time = datetime.fromisoformat(state.planning_completed_at)
                        state.planning_duration = (end_time - start_time).total_seconds()
                    except (ValueError, TypeError) as e:
                        planner_logger.log_structured(
                            level="WARNING",
                            message="Could not calculate planning duration due to invalid datetime format",
                            extra={
                                "planning_started_at": state.planning_started_at,
                                "planning_completed_at": state.planning_completed_at,
                                "error": str(e)
                            }
                        )
                        state.planning_duration = None
            
            # Add finalization message
            state.messages.append(
                AIMessage(content=f"Planning completed successfully! Complexity score: {complexity_score}/10. The plan is ready for execution.")
            )
            
            planner_logger.log_structured(
                level="INFO",
                message="Planning finalized successfully",
                extra={
                    "complexity_score": complexity_score,
                    "planning_duration": getattr(state, 'planning_duration', None),
                    "total_services": len(getattr(state, 'infrastructure_requirements', [])),
                    "total_steps": len(getattr(state, 'execution_plan', []))
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Plan finalization failed: {str(e)}"
            state.status = "error"
            raise
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _calculate_complexity_score(self, state: PlannerState) -> int:
        """Calculate a complexity score for the plan (1-10)."""
        score = 1  # Base score
        
        # Add points for number of services
        services_count = len(getattr(state, 'infrastructure_requirements', []))
        if services_count > 10:
            score += 3
        elif services_count > 5:
            score += 2
        elif services_count > 2:
            score += 1
        
        # Add points for architectural patterns
        patterns_count = len(getattr(state, 'architectural_patterns', []))
        score += min(patterns_count, 2)
        
        # Add points for security requirements
        security_count = len(getattr(state, 'security_requirements', []))
        if security_count > 5:
            score += 2
        elif security_count > 2:
            score += 1
        
        # Add points for execution steps
        steps_count = len(getattr(state, 'execution_plan', []))
        if steps_count > 20:
            score += 2
        elif steps_count > 10:
            score += 1
        
        # Cap at 10
        return min(score, 10)
    
    def _dependency_mapping_tool(self, user_request: str) -> str:
        """Tool for mapping AWS service dependencies."""
        # This would implement the dependency mapping logic
        # For now, return a placeholder
        return f"Dependency mapping analysis for: {user_request}"
    
    def _define_prompts(self) -> None:
        """Define the prompts used by the planner agent."""
        # Prompts are defined inline in the node methods for simplicity
        pass
    
    def _create_requirements_schema(self) -> type[BaseModel]:
        """Create the requirements analysis schema."""
        class RequirementsAnalysis(BaseModel):
            business_requirements_and_objectives: Dict[str, Any] = Field(description="Business requirements and objectives")
            technical_requirements_and_specifications: Dict[str, Any] = Field(description="Technical requirements and specifications")
            constraints_and_limitations: Dict[str, Any] = Field(description="Constraints and limitations")
            assumptions_and_prerequisites: Dict[str, Any] = Field(description="Assumptions and prerequisites")
            risk_factors_and_mitigation_strategies: Dict[str, Any] = Field(description="Risk factors and mitigation strategies")
        
        return RequirementsAnalysis
    
    def _create_infrastructure_schema(self) -> type[BaseModel]:
        """Create the infrastructure planning schema."""
        class InfrastructurePlan(BaseModel):
            services: List[Dict[str, Any]] = Field(description="Required AWS services with configurations")
            patterns: List[Dict[str, Any]] = Field(description="Architectural patterns")
            security_requirements: List[Dict[str, Any]] = Field(description="Security requirements")
            cost_estimates: Dict[str, Any] = Field(description="Cost estimates and breakdown")
            dependencies: Dict[str, List[str]] = Field(description="Service dependencies")
        
        return InfrastructurePlan
    
    def _create_execution_schema(self) -> type[BaseModel]:
        """Create the execution planning schema."""
        class ExecutionPlan(BaseModel):
            steps: List[Dict[str, Any]] = Field(description="Step-by-step execution plan")
            total_estimated_time: str = Field(description="Total estimated execution time")
            critical_path: List[int] = Field(description="Critical path steps")
            resource_requirements: Dict[str, Any] = Field(description="Resource requirements")
        
        return ExecutionPlan
    
    def _create_validation_schema(self) -> type[BaseModel]:
        """Create the validation schema."""
        class ValidationResult(BaseModel):
            validation_results: Dict[str, List[str]] = Field(description="Validation results by category")
            overall_assessment: str = Field(description="Overall assessment")
            recommendations: List[str] = Field(description="Recommendations for improvement")
            risk_level: str = Field(description="Overall risk level")
        
        return ValidationResult


# Factory function for easy creation
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
