"""
Modification Planner Implementation.

This module implements the ModificationPlanner, which is specialized for:
- Modifying existing AWS infrastructure
- Change management and risk assessment
- Minimal disruption strategies
- Rollback planning and procedures
- Impact analysis and dependency management

The ModificationPlanner inherits from BaseAgent and focuses on modification scenarios.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from functools import wraps

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel

from aws_orchestrator_agent.core.agents.base_agent import BaseAgent, agent_node, require_approval
from aws_orchestrator_agent.core.agents.planner.planner_state import (
    PlannerState,
    InfrastructureRequirement,
    ArchitecturalPattern,
    SecurityRequirement,
    ExecutionStep
)
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config


# Create agent logger for modification planner
modification_logger = AgentLogger("MODIFICATION_PLANNER")


class ModificationRequirements(BaseModel):
    """Output schema for modification requirements analysis."""
    business_requirements: List[str]
    technical_requirements: List[str]
    constraints: List[str]
    assumptions: List[str]
    risk_factors: List[str]
    affected_services: List[str]
    unchanged_services: List[str]
    compatibility_requirements: List[str]


class ExistingInfrastructureAnalysis(BaseModel):
    """Output schema for existing infrastructure analysis."""
    current_services: List[Dict[str, Any]]
    current_patterns: List[Dict[str, Any]]
    current_security: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    performance_metrics: Dict[str, Any]
    compliance_status: Dict[str, Any]
    technical_debt: List[str]


class ChangeImpactAssessment(BaseModel):
    """Output schema for change impact assessment."""
    affected_resources: List[str]
    unchanged_resources: List[str]
    risk_level: str
    downtime_required: bool
    rollback_strategy: str
    mitigation_measures: List[str]
    cost_impact: Dict[str, Any]
    timeline_impact: str


class ModificationPlan(BaseModel):
    """Output schema for modification planning."""
    services_to_modify: List[Dict[str, Any]]
    services_to_add: List[Dict[str, Any]]
    services_to_remove: List[Dict[str, Any]]
    security_updates: List[Dict[str, Any]]
    cost_estimates: Dict[str, Any]
    dependencies: Dict[str, List[str]]
    rollback_procedures: List[str]
    testing_strategy: str


class ModificationExecutionPlan(BaseModel):
    """Output schema for modification execution planning."""
    steps: List[Dict[str, Any]]
    total_estimated_time: str
    critical_path: List[int]
    resource_requirements: Dict[str, Any]
    rollback_steps: List[Dict[str, Any]]
    testing_phases: List[str]
    monitoring_checkpoints: List[str]


class ModificationPlanner(BaseAgent):
    """
    Modification Planner for AWS infrastructure changes.
    
    This agent specializes in modifying existing AWS infrastructure,
    focusing on change management, risk assessment, and minimal disruption.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "modification_planner"
    ):
        """
        Initialize the Modification Planner.
        
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
            modification_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model for modification planner: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model for modification planner: {e}",
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
        
        # Initialize output parsers
        self.requirements_parser = JsonOutputParser(pydantic_object=ModificationRequirements)
        self.existing_infra_parser = JsonOutputParser(pydantic_object=ExistingInfrastructureAnalysis)
        self.impact_parser = JsonOutputParser(pydantic_object=ChangeImpactAssessment)
        self.modification_parser = JsonOutputParser(pydantic_object=ModificationPlan)
        self.execution_parser = JsonOutputParser(pydantic_object=ModificationExecutionPlan)
        
        # Define prompts
        self._define_prompts()
        
        # Build the agent graph
        self._build_planner_graph()
    
    def _define_prompts(self) -> None:
        """Define the prompts used by the modification planner."""
        
        # Requirements analysis prompt for modifications
        self.requirements_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS infrastructure planner specializing in infrastructure modifications. Analyze the user request to identify requirements for modifying existing infrastructure.

Analyze the user request and identify:
1. Business requirements and objectives for the modification
2. Technical requirements and specifications
3. Constraints and limitations
4. Assumptions and prerequisites
5. Risk factors and mitigation strategies
6. Services that will be affected by the changes
7. Services that will remain unchanged
8. Compatibility requirements with existing infrastructure

Provide a structured analysis that will guide the modification planning process."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Analyze the following modification request: {user_request}")
        ])
        
        # Existing infrastructure analysis prompt
        self.existing_infrastructure_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS infrastructure analyst. Analyze the existing infrastructure to understand the current state and identify what needs to be modified.

Analyze:
1. Current AWS services and their configurations
2. Existing architectural patterns and their relationships
3. Security configurations and compliance status
4. Resource dependencies and integration points
5. Current performance metrics and bottlenecks
6. Technical debt and areas for improvement
7. Compliance and governance requirements

Provide a comprehensive analysis that will guide the modification planning."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Analyze the existing infrastructure for this modification request: {user_request}")
        ])
        
        # Change impact assessment prompt
        self.impact_assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert change management specialist. Assess the impact of proposed infrastructure modifications.

Evaluate:
1. Which resources will be affected by the changes
2. Which resources will remain unchanged
3. Risk level and potential issues
4. Downtime requirements and maintenance windows
5. Rollback procedures and requirements
6. Mitigation measures and safeguards
7. Cost implications of the changes
8. Timeline impact and scheduling considerations

Provide a detailed impact analysis with risk assessment and mitigation strategies."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Assess the impact of these modifications:\nProposed Changes: {proposed_changes}\nExisting Infrastructure: {existing_infrastructure}")
        ])
        
        # Modification planning prompt
        self.modification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS architect specializing in infrastructure modifications. Create a comprehensive modification plan that minimizes disruption and ensures backward compatibility.

Design:
1. Services that need to be modified with specific changes
2. New services to be added
3. Services to be removed or deprecated
4. Security updates and compliance requirements
5. Cost estimates and optimization strategies
6. Dependencies and integration points
7. Rollback procedures for each change
8. Testing strategy to validate changes

Follow AWS best practices and ensure minimal disruption to existing services."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Create modification plan based on:\nRequirements: {requirements_analysis}\nExisting Infrastructure: {existing_infrastructure}\nImpact Assessment: {impact_assessment}")
        ])
        
        # Execution planning prompt for modifications
        self.execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert DevOps engineer specializing in infrastructure modifications. Create a detailed step-by-step execution plan for safely modifying existing AWS infrastructure.

The plan should include:
1. Sequential steps with clear actions and commands
2. Dependencies between steps and services
3. Time estimates for each phase
4. Resource requirements and provisioning
5. Validation criteria for each step
6. Rollback steps for each modification
7. Testing phases and validation procedures
8. Monitoring checkpoints and health checks
9. Minimal downtime strategies
10. Communication and coordination procedures

Ensure the plan follows change management best practices and includes comprehensive rollback procedures."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Create execution plan for infrastructure modifications:\nModification Plan: {modification_plan}\nImpact Assessment: {impact_assessment}")
        ])
        
        # Validation prompt for modifications
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert infrastructure validator specializing in modifications. Review the complete modification plan and identify any issues, risks, or improvements.

Check for:
1. Security vulnerabilities and compliance gaps
2. Risk of service disruption or downtime
3. Rollback procedure completeness
4. Testing strategy adequacy
5. Dependency conflicts
6. Cost optimization opportunities
7. Best practice violations
8. Change management process gaps

Provide actionable recommendations for improvement and risk mitigation."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Validate the modification plan: {complete_plan}")
        ])
    
    def _build_planner_graph(self) -> None:
        """Build the modification planner's StateGraph."""
        
        # Define nodes
        nodes = self.define_nodes()
        for name, node_func in nodes.items():
            self.add_node(name, node_func)
        
        # Build and compile the graph
        self.build_graph()
        self.compile_graph()
        
        modification_logger.log_structured(
            level="INFO",
            message="Modification planner graph built and compiled successfully",
            extra={"agent_name": self.name, "node_count": len(nodes)}
        )
    
    def define_nodes(self) -> Dict[str, Any]:
        """Define the modification planner's nodes."""
        from aws_orchestrator_agent.core.agents.base_agent import agent_node
        
        # Create wrapper functions with proper names
        async def analyze_requirements_wrapper(state):
            return await self.analyze_requirements_node(state)
        
        async def analyze_existing_infrastructure_wrapper(state):
            return await self.analyze_existing_infrastructure_node(state)
        
        async def assess_change_impact_wrapper(state):
            return await self.assess_change_impact_node(state)
        
        async def plan_modifications_wrapper(state):
            return await self.plan_modifications_node(state)
        
        async def create_execution_plan_wrapper(state):
            return await self.create_execution_plan_node(state)
        
        async def validate_plan_wrapper(state):
            return await self.validate_plan_node(state)
        
        async def finalize_plan_wrapper(state):
            return await self.finalize_plan_node(state)
        
        return {
            "analyze_requirements": agent_node(analyze_requirements_wrapper),
            "analyze_existing_infrastructure": agent_node(analyze_existing_infrastructure_wrapper),
            "assess_change_impact": agent_node(assess_change_impact_wrapper),
            "plan_modifications": agent_node(plan_modifications_wrapper),
            "create_execution_plan": agent_node(create_execution_plan_wrapper),
            "validate_plan": agent_node(validate_plan_wrapper),
            "finalize_plan": agent_node(finalize_plan_wrapper)
        }
    
    def define_edges(self) -> Dict[str, List[str]]:
        """Define the edges between nodes."""
        return {
            "analyze_requirements": ["analyze_existing_infrastructure"],
            "analyze_existing_infrastructure": ["assess_change_impact"],
            "assess_change_impact": ["plan_modifications"],
            "plan_modifications": ["create_execution_plan"],
            "create_execution_plan": ["validate_plan"],
            "validate_plan": ["finalize_plan"],
            "finalize_plan": []
        }
    
    def create_initial_state(self, input_data: Dict[str, Any]) -> PlannerState:
        """Create the initial state for the modification planner."""
        
        # Extract user request
        user_request = input_data.get("user_request", "")
        context_id = input_data.get("context_id", str(uuid.uuid4()))
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        
        # Create initial messages
        messages = [
            SystemMessage(content="You are an expert AWS infrastructure planner specializing in infrastructure modifications. Create a comprehensive plan for safely modifying existing infrastructure."),
            HumanMessage(content=user_request)
        ]
        
        # Create initial state
        state = PlannerState(
            messages=messages,
            user_request=user_request,
            context_id=context_id,
            task_id=task_id,
            request_type="modify",  # Always modify for this planner
            status="planning_started",
            planning_started_at=datetime.utcnow().isoformat()
        )
        
        modification_logger.log_structured(
            level="INFO",
            message="Created initial modification planner state",
            extra={
                "agent_name": self.name,
                "context_id": context_id,
                "task_id": task_id,
                "user_request_length": len(user_request)
            }
        )
        
        return state
    
    async def analyze_requirements_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Analyze user requirements for infrastructure modifications."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Create prompt chain
            chain = self.requirements_prompt | self.model | self.requirements_parser
            
            # Execute analysis
            result = chain.invoke({
                "messages": current_state.messages,
                "user_request": current_state.user_request
            })
            
            # Update state with analysis results
            current_state.requirements_analysis = result.dict()
            current_state.status = "requirements_analyzed"
            current_state.current_step = "requirements_analyzed"
            current_state.completed_steps.append("analyze_requirements")
            
            # Add analysis result to messages
            current_state.messages.append(
                AIMessage(content=f"Modification requirements analysis completed. Found {len(result.business_requirements)} business requirements, {len(result.technical_requirements)} technical requirements, {len(result.affected_services)} affected services, and {len(result.unchanged_services)} unchanged services.")
            )
            
            modification_logger.log_structured(
                level="INFO",
                message="Modification requirements analysis completed",
                extra={
                    "agent_name": self.name,
                    "business_requirements_count": len(result.business_requirements),
                    "technical_requirements_count": len(result.technical_requirements),
                    "affected_services_count": len(result.affected_services),
                    "unchanged_services_count": len(result.unchanged_services)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in modification requirements analysis: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "requirements_analysis"
            
            return current_state.model_dump()
    
    async def analyze_existing_infrastructure_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze existing infrastructure for modification planning."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Create prompt chain
            chain = self.existing_infrastructure_prompt | self.model | self.existing_infra_parser
            
            # Execute analysis
            result = chain.invoke({
                "messages": current_state.messages,
                "user_request": current_state.user_request
            })
            
            # Update state with existing infrastructure analysis
            current_state.existing_infrastructure = result.dict()
            current_state.current_step = "existing_infrastructure_analyzed"
            current_state.completed_steps.append("analyze_existing_infrastructure")
            
            # Add analysis result to messages
            current_state.messages.append(
                AIMessage(content=f"Existing infrastructure analysis completed. Found {len(result.current_services)} current services, {len(result.current_patterns)} architectural patterns, and {len(result.technical_debt)} technical debt items.")
            )
            
            modification_logger.log_structured(
                level="INFO",
                message="Existing infrastructure analysis completed",
                extra={
                    "agent_name": self.name,
                    "current_services_count": len(result.current_services),
                    "current_patterns_count": len(result.current_patterns),
                    "technical_debt_count": len(result.technical_debt)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in existing infrastructure analysis: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "existing_infrastructure_analysis"
            
            return current_state.model_dump()
    
    async def assess_change_impact_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of proposed modifications."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Get requirements analysis and existing infrastructure
            requirements_analysis = current_state.requirements_analysis or {}
            existing_infrastructure = current_state.existing_infrastructure or {}
            
            # Create prompt chain
            chain = self.impact_assessment_prompt | self.model | self.impact_parser
            
            # Execute impact assessment
            result = chain.invoke({
                "messages": current_state.messages,
                "proposed_changes": str(requirements_analysis),
                "existing_infrastructure": str(existing_infrastructure)
            })
            
            # Update state with impact assessment
            current_state.change_impact = result.dict()
            current_state.affected_resources = result.affected_resources
            current_state.unchanged_resources = result.unchanged_resources
            current_state.downtime_required = result.downtime_required
            current_state.rollback_strategy = result.rollback_strategy
            
            # Update risk assessment
            current_state.risk_assessment = {
                "risk_level": result.risk_level,
                "downtime_required": result.downtime_required,
                "rollback_strategy": result.rollback_strategy,
                "mitigation_measures": result.mitigation_measures,
                "cost_impact": result.cost_impact,
                "timeline_impact": result.timeline_impact
            }
            
            current_state.current_step = "change_impact_assessed"
            current_state.completed_steps.append("assess_change_impact")
            
            # Add impact assessment to messages
            current_state.messages.append(
                AIMessage(content=f"Change impact assessment completed. Risk level: {result.risk_level}, Downtime required: {result.downtime_required}, Affected resources: {len(result.affected_resources)}")
            )
            
            modification_logger.log_structured(
                level="INFO",
                message="Change impact assessment completed",
                extra={
                    "agent_name": self.name,
                    "risk_level": result.risk_level,
                    "downtime_required": result.downtime_required,
                    "affected_resources_count": len(result.affected_resources),
                    "unchanged_resources_count": len(result.unchanged_resources)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in change impact assessment: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "change_impact_assessment"
            
            return current_state.model_dump()
    
    async def plan_modifications_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the infrastructure modifications."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Get analysis results
            requirements_analysis = current_state.requirements_analysis or {}
            existing_infrastructure = current_state.existing_infrastructure or {}
            impact_assessment = current_state.change_impact or {}
            
            # Create prompt chain
            chain = self.modification_prompt | self.model | self.modification_parser
            
            # Execute modification planning
            result = chain.invoke({
                "messages": current_state.messages,
                "requirements_analysis": str(requirements_analysis),
                "existing_infrastructure": str(existing_infrastructure),
                "impact_assessment": str(impact_assessment)
            })
            
            # Convert results to Pydantic models
            infrastructure_requirements = [
                InfrastructureRequirement(**service) for service in result.services_to_modify + result.services_to_add
            ]
            
            architectural_patterns = [
                ArchitecturalPattern(**pattern) for pattern in [{"name": "modification", "description": "Infrastructure modification pattern"}]
            ]
            
            security_requirements = [
                SecurityRequirement(**req) for req in result.security_updates
            ]
            
            # Update state
            current_state.infrastructure_requirements = infrastructure_requirements
            current_state.architectural_patterns = architectural_patterns
            current_state.security_requirements = security_requirements
            current_state.cost_analysis = result.cost_estimates
            current_state.status = "modifications_planned"
            current_state.current_step = "modifications_planned"
            current_state.completed_steps.append("plan_modifications")
            
            # Add planning result to messages
            planning_summary = f"Modification planning completed. Services to modify: {len(result.services_to_modify)}, Services to add: {len(result.services_to_add)}, Services to remove: {len(result.services_to_remove)}, Security updates: {len(result.security_updates)}"
            current_state.messages.append(AIMessage(content=planning_summary))
            
            modification_logger.log_structured(
                level="INFO",
                message="Modification planning completed",
                extra={
                    "agent_name": self.name,
                    "services_to_modify_count": len(result.services_to_modify),
                    "services_to_add_count": len(result.services_to_add),
                    "services_to_remove_count": len(result.services_to_remove),
                    "security_updates_count": len(result.security_updates)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in modification planning: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "modification_planning"
            
            return current_state.model_dump()
    
    async def create_execution_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed execution plan for infrastructure modifications."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Get modification plan and impact assessment
            modification_plan = {
                "services": [req.dict() for req in current_state.infrastructure_requirements or []],
                "patterns": [pattern.dict() for pattern in current_state.architectural_patterns or []],
                "security": [req.dict() for req in current_state.security_requirements or []]
            }
            impact_assessment = current_state.change_impact or {}
            
            # Create prompt chain
            chain = self.execution_prompt | self.model | self.execution_parser
            
            # Execute execution planning
            result = chain.invoke({
                "messages": current_state.messages,
                "modification_plan": str(modification_plan),
                "impact_assessment": str(impact_assessment)
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
            
            # Add execution plan to messages
            execution_summary = f"Modification execution plan created with {len(execution_steps)} steps. Estimated time: {result.total_estimated_time}. Rollback steps: {len(result.rollback_steps)}. Testing phases: {len(result.testing_phases)}"
            current_state.messages.append(AIMessage(content=execution_summary))
            
            modification_logger.log_structured(
                level="INFO",
                message="Modification execution plan created",
                extra={
                    "agent_name": self.name,
                    "steps_count": len(execution_steps),
                    "estimated_time": result.total_estimated_time,
                    "rollback_steps_count": len(result.rollback_steps),
                    "testing_phases_count": len(result.testing_phases),
                    "monitoring_checkpoints_count": len(result.monitoring_checkpoints)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in modification execution planning: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "execution_planning"
            
            return current_state.model_dump()
    
    async def validate_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete modification plan."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Prepare complete plan for validation
            complete_plan = {
                "requirements": current_state.requirements_analysis or {},
                "existing_infrastructure": current_state.existing_infrastructure or {},
                "impact_assessment": current_state.change_impact or {},
                "modifications": {
                    "services": [req.dict() for req in current_state.infrastructure_requirements or []],
                    "patterns": [pattern.dict() for pattern in current_state.architectural_patterns or []],
                    "security": [req.dict() for req in current_state.security_requirements or []]
                },
                "execution": {
                    "steps": [step.dict() for step in current_state.execution_plan or []],
                    "resources": current_state.resource_requirements or {}
                }
            }
            
            # Create prompt chain
            chain = self.validation_prompt | self.model
            
            # Execute validation
            result = chain.invoke({
                "messages": current_state.messages,
                "complete_plan": str(complete_plan)
            })
            
            # Extract validation results
            validation_content = result.content if hasattr(result, 'content') else str(result)
            
            # Update state with validation results
            current_state.validation_criteria = [validation_content]
            current_state.status = "plan_validated"
            current_state.current_step = "plan_validated"
            current_state.completed_steps.append("validate_plan")
            
            # Add validation result to messages
            current_state.messages.append(
                AIMessage(content=f"Modification plan validation completed. Review the validation results and recommendations.")
            )
            
            modification_logger.log_structured(
                level="INFO",
                message="Modification plan validation completed",
                extra={
                    "agent_name": self.name,
                    "validation_content_length": len(validation_content)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in modification plan validation: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "plan_validation"
            
            return current_state.model_dump()
    
    @require_approval
    async def finalize_plan_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the modification plan and prepare for execution."""
        
        try:
            # Get current state
            current_state = PlannerState(**state)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(current_state.model_dump())
            
            # Update final state
            current_state.complexity_score = complexity_score
            current_state.status = "plan_completed"
            current_state.planning_completed_at = datetime.utcnow().isoformat()
            current_state.current_step = "plan_completed"
            current_state.completed_steps.append("finalize_plan")
            
            # Calculate planning duration
            if current_state.planning_started_at:
                start_time = datetime.fromisoformat(current_state.planning_started_at)
                end_time = datetime.fromisoformat(current_state.planning_completed_at)
                current_state.planning_duration = (end_time - start_time).total_seconds()
            
            # Add finalization message
            current_state.messages.append(
                AIMessage(content=f"Modification planning completed successfully! Complexity score: {complexity_score}/10. Risk level: {current_state.risk_assessment.get('risk_level', 'unknown')}. The plan is ready for execution.")
            )
            
            modification_logger.log_structured(
                level="INFO",
                message="Modification plan finalized",
                extra={
                    "agent_name": self.name,
                    "complexity_score": complexity_score,
                    "planning_duration": current_state.planning_duration,
                    "risk_level": current_state.risk_assessment.get("risk_level", "unknown"),
                    "total_steps": len(current_state.completed_steps)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error in modification plan finalization: {e}",
                extra={"error": str(e), "user_request": state.get("user_request", "")}
            )
            
            # Set error state
            current_state = PlannerState(**state)
            current_state.error = str(e)
            current_state.error_context = "plan_finalization"
            
            return current_state.model_dump()
    
    def _calculate_complexity_score(self, state: Dict[str, Any]) -> int:
        """Calculate complexity score for modification plan."""
        try:
            # Base complexity
            complexity = 4  # Modifications are inherently more complex
            
            # Add complexity based on number of affected services
            affected_resources = state.get("affected_resources", [])
            if len(affected_resources) > 10:
                complexity += 3
            elif len(affected_resources) > 5:
                complexity += 2
            elif len(affected_resources) > 2:
                complexity += 1
            
            # Add complexity based on risk level
            risk_level = state.get("risk_assessment", {}).get("risk_level", "low")
            if risk_level == "high":
                complexity += 2
            elif risk_level == "medium":
                complexity += 1
            
            # Add complexity based on downtime requirement
            if state.get("downtime_required", False):
                complexity += 1
            
            # Add complexity based on execution steps
            steps_count = len(state.get("execution_plan", []))
            if steps_count > 25:
                complexity += 2
            elif steps_count > 15:
                complexity += 1
            
            # Cap at 10
            return min(complexity, 10)
            
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error calculating complexity score: {e}",
                extra={"error": str(e)}
            )
            return 6  # Default complexity for modifications
    
    @log_sync
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the modification plan."""
        try:
            state = self.get_state()
            if not state:
                return {"error": "No plan state available"}
            
            return {
                "plan_type": "modification",
                "status": state.get("status"),
                "complexity_score": state.get("complexity_score"),
                "affected_resources_count": len(state.get("affected_resources", [])),
                "unchanged_resources_count": len(state.get("unchanged_resources", [])),
                "risk_level": state.get("risk_assessment", {}).get("risk_level", "unknown"),
                "downtime_required": state.get("downtime_required", False),
                "execution_steps_count": len(state.get("execution_plan", [])),
                "planning_duration": state.get("planning_duration"),
                "completed_steps": state.get("completed_steps", [])
            }
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error getting plan summary: {e}",
                extra={"error": str(e)}
            )
            return {"error": str(e)}
    
    @log_sync
    def export_plan(self, format: str = "json") -> Dict[str, Any]:
        """Export the modification plan in the specified format."""
        try:
            state = self.get_state()
            if not state:
                return {"error": "No plan state available"}
            
            if format.lower() == "json":
                return {
                    "plan_type": "modification",
                    "exported_at": datetime.utcnow().isoformat(),
                    "plan_data": state.model_dump() if hasattr(state, 'model_dump') else state
                }
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            modification_logger.log_structured(
                level="ERROR",
                message=f"Error exporting plan: {e}",
                extra={"error": str(e), "format": format}
            )
            return {"error": str(e)}


@log_sync
def create_modification_planner(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "modification_planner"
) -> ModificationPlanner:
    """
    Factory function to create a ModificationPlanner instance.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured ModificationPlanner instance
    """
    return ModificationPlanner(config=config, custom_config=custom_config, name=name) 