"""
Generator Swarm Agent using langgraph-swarm.

This module implements the Generator Swarm Agent, which manages the generation
sub-agents (Resource Generator, Variable Generator, Data Source Generator, Local Values Generator) 
using langgraph-swarm with custom handoff tools.

The swarm coordinates the generation workflow and routes between sub-agents
based on the current state of the shared GeneratorStageState.
"""

import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph_swarm import create_swarm
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync
from aws_orchestrator_agent.core.agents_latest.agents.base_agent import BaseSubgraphAgent

from .generator_state import GeneratorStageState, GeneratorAgentStatus, DependencyType
from .generator_state_controller import GeneratorStageController
from .generator_handoff_manager import GeneratorStageHandoffManager, create_completion_handoff_tool
from .generator_handoff_tools import create_handoff_to_generator_complete
from .generator_stage_cp_manager import GeneratorStageCheckpointManager
from .generation_hitl import GeneratorStageHumanLoop
from .approval_middleware import ApprovalMiddleware
from .sub_agents import (
    generate_terraform_resources, 
    generate_terraform_variables, 
    generate_terraform_data_sources,
    generate_terraform_local_values,
    generate_terraform_outputs
)

# Create agent logger for generator swarm
generator_swarm_logger = AgentLogger("GENERATOR_SWARM")

class GeneratorSwarmAgent(BaseSubgraphAgent):
    """
    Generator Swarm Agent that manages generation sub-agents using langgraph-swarm.
    
    This agent coordinates the generation workflow by routing between specialized
    sub-agents using custom handoff tools and the langgraph-swarm library.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "generator_swarm_agent",
        memory: Optional[MemorySaver] = None
    ):
        """
        Initialize the Generator Swarm Agent.
        
        Args:
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
            name: Agent name for identification
            memory: Shared memory/checkpointer instance
        """
        generator_swarm_logger.log_structured(
            level="INFO",
            message="=== GENERATOR SWARM INITIALIZATION START ===",
            extra={
                "name": name,
                "has_config": config is not None,
                "has_custom_config": custom_config is not None,
                "has_memory": memory is not None
            }
        )
        
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for identification
        self._name = name
        
        # Set shared memory
        self.memory = memory or MemorySaver()
        
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Basic initialization complete",
            extra={
                "config_type": type(self.config_instance).__name__,
                "memory_type": type(self.memory).__name__
            }
        )
        
        # Get LLM configuration from centralized config
        llm_config = self.config_instance.get_llm_config()
        
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="LLM config retrieved",
            extra={
                "llm_provider": llm_config.get('provider'),
                "llm_model": llm_config.get('model'),
                "llm_temperature": llm_config.get('temperature'),
                "llm_max_tokens": llm_config.get('max_tokens')
            }
        )
        
        # Initialize the LLM model using the centralized provider
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            generator_swarm_logger.log_structured(
                level="INFO",
                message="LLM model initialized successfully",
                extra={
                    "llm_provider": llm_config['provider'], 
                    "llm_model": llm_config['model'],
                    "model_type": type(self.model).__name__
                }
            )
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="LLM model initialization failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model')
                }
            )
            raise
        
        # Initialize managers
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Initializing managers",
            extra={}
        )
        
        self.controller = GeneratorStageController()
        self.handoff_manager = GeneratorStageHandoffManager()
        self.checkpoint_manager = GeneratorStageCheckpointManager(self.memory)
        self.human_loop = GeneratorStageHumanLoop()
        self.approval_middleware = ApprovalMiddleware(self.human_loop)
        
        generator_swarm_logger.log_structured(
            level="INFO",
            message="=== GENERATOR SWARM INITIALIZATION COMPLETE ===",
            extra={
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "name": name,
                "managers_initialized": True
            }
        )
    
    def input_transform(self, supervisor_state) -> GeneratorStageState:
        """
        Transform supervisor state to generator swarm state with planner data.
        
        This method extracts the execution plan from the supervisor state and
        transforms it into the GeneratorStageState format that the swarm expects.
        
        Args:
            supervisor_state: SupervisorState containing planner data and execution plan
            
        Returns:
            GeneratorStageState: Initialized state for the generator swarm
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Transforming supervisor state to generator swarm state",
                extra={
                    "supervisor_state_type": type(supervisor_state).__name__,
                    "has_planner_data": hasattr(supervisor_state, 'planner_data') and supervisor_state.planner_data is not None,
                    "has_execution_plan": hasattr(supervisor_state, 'execution_plan') and supervisor_state.execution_plan is not None
                }
            )
            
            # Extract execution plan from supervisor state
            planner_data = getattr(supervisor_state, 'planner_data', None) or {}
            execution_plan = getattr(supervisor_state, 'execution_plan', None) or planner_data.get("execution_plan", {})
            
            # Extract individual components from execution plan
            resource_configurations = execution_plan.get("resource_configurations", [])
            variable_definitions = execution_plan.get("variable_definitions", [])
            data_sources = execution_plan.get("data_sources", [])
            local_values = execution_plan.get("local_values", [])
            dependencies = execution_plan.get("dependencies", [])
            output_definitions = execution_plan.get("output_definitions", [])
            security_considerations = execution_plan.get("security_considerations", [])
            cost_estimates = execution_plan.get("cost_estimates", [])
            
            # Extract metadata
            module_name = execution_plan.get("module_name", "terraform-module")
            service_name = execution_plan.get("service_name", "Unknown Service")
            target_environment = execution_plan.get("target_environment", "prod")
            
            generator_swarm_logger.log_structured(
                level="DEBUG",
                message="Extracted execution plan components",
                extra={
                    "resource_configs_count": len(resource_configurations),
                    "variable_defs_count": len(variable_definitions),
                    "data_sources_count": len(data_sources),
                    "local_values_count": len(local_values),
                    "dependencies_count": len(dependencies),
                    "output_definitions_count": len(output_definitions),
                    "security_considerations_count": len(security_considerations),
                    "cost_estimates_count": len(cost_estimates),
                    "module_name": module_name,
                    "service_name": service_name,
                    "target_environment": target_environment
                }
            )
            
            # Create initial message for the swarm
            initial_message = HumanMessage(
                content=f"Generate Terraform module '{module_name}' for {service_name} service in {target_environment} environment from execution plan"
            )
            
            # Transform to GeneratorStageState
            generator_state = GeneratorStageState(
                # Core swarm fields
                messages=[initial_message],
                active_agent="resource_configuration_agent",
                
                # Planning stage management
                stage_status="planning_active",
                current_stage="planning",
                planning_progress={
                    "resource_configuration_agent": 0.0,
                    "variable_definition_agent": 0.0,
                    "data_source_agent": 0.0,
                    "local_values_agent": 0.0
                },
                
                # Agent coordination
                agent_status_matrix={
                    "resource_configuration_agent": GeneratorAgentStatus.INACTIVE,
                    "variable_definition_agent": GeneratorAgentStatus.INACTIVE,
                    "data_source_agent": GeneratorAgentStatus.INACTIVE,
                    "local_values_agent": GeneratorAgentStatus.INACTIVE
                },
                
                # Agent workspaces with planner data
                agent_workspaces={
                    "resource_configuration_agent": {
                        "generated_resources": [],
                        "planner_input": resource_configurations,
                        "dependencies": [dep for dep in dependencies if dep.get("target_component", "").startswith("aws_")],
                        "security_context": [sec for sec in security_considerations if sec.get("component", "").startswith("aws_")],
                        "cost_context": [cost for cost in cost_estimates if cost.get("component", "").startswith("aws_")]
                    },
                    "variable_definition_agent": {
                        "generated_variables": [],
                        "planner_input": variable_definitions,
                        "resource_dependencies": [dep for dep in dependencies if "variable" in dep.get("dependency_type", "").lower()],
                        "validation_context": [var for var in variable_definitions if var.get("validation_rules")]
                    },
                    "data_source_agent": {
                        "generated_data_sources": [],
                        "planner_input": data_sources,
                        "resource_dependencies": [dep for dep in dependencies if dep.get("dependency_type") == "data_dependency"],
                        "cross_references": [ds for ds in data_sources if ds.get("cross_references")]
                    },
                    "local_values_agent": {
                        "generated_locals": [],
                        "planner_input": local_values,
                        "computed_dependencies": [dep for dep in dependencies if dep.get("dependency_type") == "computed_dependency"],
                        "expression_context": [lv for lv in local_values if lv.get("depends_on")]
                    },
                    "output_definition_agent": {
                        "generated_outputs": [],
                        "planner_input": output_definitions,
                        "resource_dependencies": [dep for dep in dependencies if dep.get("target_component", "").startswith("output")],
                        "validation_context": [out for out in output_definitions if out.get("validation_rules")],
                        "sensitivity_context": [out for out in output_definitions if out.get("sensitive", False)]
                    }
                },
                
                # Planning context
                planning_context={
                    "dependencies": dependencies,
                    "security_considerations": security_considerations,
                    "cost_estimates": cost_estimates,
                    "module_name": module_name,
                    "service_name": service_name,
                    "target_environment": target_environment,
                    "execution_plan": execution_plan,
                    "planner_data": planner_data
                },
                
                # HITL integration
                approval_required=False,
                approval_context=None,
                pending_human_decisions=[],
                
                # Stage progress tracking
                stage_progress={
                    "planning": 0.0,
                    "enhancement": 0.0,
                    "integration": 0.0
                }
            )
            
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Successfully transformed supervisor state to generator swarm state",
                extra={
                    "generator_state_type": type(generator_state).__name__,
                    "active_agent": generator_state["active_agent"],
                    "stage_status": generator_state["stage_status"],
                    "agent_workspaces_count": len(generator_state["agent_workspaces"]),
                    "planning_context_keys": list(generator_state["planning_context"].keys())
                }
            )
            
            return generator_state
            
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="Failed to transform supervisor state to generator swarm state",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "supervisor_state_type": type(supervisor_state).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            # Return minimal fallback state
            return GeneratorStageState(
                messages=[HumanMessage(content="Generate Terraform module from execution plan")],
                active_agent="resource_configuration_agent",
                stage_status="planning_active",
                current_stage="planning",
                planning_progress={
                    "resource_configuration_agent": 0.0,
                    "variable_definition_agent": 0.0,
                    "data_source_agent": 0.0,
                    "local_values_agent": 0.0
                },
                agent_status_matrix={
                    "resource_configuration_agent": GeneratorAgentStatus.INACTIVE,
                    "variable_definition_agent": GeneratorAgentStatus.INACTIVE,
                    "data_source_agent": GeneratorAgentStatus.INACTIVE,
                    "local_values_agent": GeneratorAgentStatus.INACTIVE
                },
                agent_workspaces={
                    "resource_configuration_agent": {"generated_resources": [], "planner_input": []},
                    "variable_definition_agent": {"generated_variables": [], "planner_input": []},
                    "data_source_agent": {"generated_data_sources": [], "planner_input": []},
                    "local_values_agent": {"generated_locals": [], "planner_input": []}
                },
                planning_context={
                    "module_name": "terraform-module",
                    "service_name": "Unknown Service",
                    "target_environment": "prod"
                },
                approval_required=False,
                approval_context=None,
                pending_human_decisions=[],
                stage_progress={"planning": 0.0, "enhancement": 0.0, "integration": 0.0}
            )
    
    def output_transform(self, generator_state: GeneratorStageState) -> Dict[str, Any]:
        """
        Transform generator swarm state back to supervisor state format.
        
        This method extracts the generated artifacts from the generator swarm state
        and transforms them into a format that can be merged back into the main
        supervisor state.
        
        Args:
            generator_state: GeneratorStageState containing all generated artifacts
            
        Returns:
            Dict[str, Any]: State updates to merge back into supervisor state
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Transforming generator swarm state back to supervisor state",
                extra={
                    "generator_state_type": type(generator_state).__name__,
                    "stage_status": generator_state.get("stage_status", "unknown"),
                    "current_stage": generator_state.get("current_stage", "unknown")
                }
            )
            
            # Extract generated artifacts from agent workspaces
            agent_workspaces = generator_state.get("agent_workspaces", {})
            
            # Collect all generated artifacts
            generated_resources = agent_workspaces.get("resource_configuration_agent", {}).get("generated_resources", [])
            generated_variables = agent_workspaces.get("variable_definition_agent", {}).get("generated_variables", [])
            generated_data_sources = agent_workspaces.get("data_source_agent", {}).get("generated_data_sources", [])
            generated_locals = agent_workspaces.get("local_values_agent", {}).get("generated_locals", [])
            generated_outputs = agent_workspaces.get("output_definition_agent", {}).get("generated_outputs", [])
            
            # Extract planning context for metadata
            planning_context = generator_state.get("planning_context", {})
            
            # Extract stage progress and agent status
            stage_progress = generator_state.get("stage_progress", {})
            agent_status_matrix = generator_state.get("agent_status_matrix", {})
            
            # Create comprehensive generation data
            generation_data = {
                "generated_resources": generated_resources,
                "generated_variables": generated_variables,
                "generated_data_sources": generated_data_sources,
                "generated_locals": generated_locals,
                "generated_outputs": generated_outputs,
                "module_name": planning_context.get("module_name", "terraform-module"),
                "service_name": planning_context.get("service_name", "Unknown Service"),
                "target_environment": planning_context.get("target_environment", "prod"),
                "stage_progress": stage_progress,
                "agent_status_matrix": agent_status_matrix,
                "generation_complete": generator_state.get("stage_status") == "planning_complete",
                "total_artifacts": len(generated_resources) + len(generated_variables) + 
                                 len(generated_data_sources) + len(generated_locals) + len(generated_outputs)
            }
            
            # Create supervisor state updates
            supervisor_updates = {
                "generation_data": generation_data,
                "generated_module_ref": f"terraform-module-{planning_context.get('module_name', 'unknown')}",
                "question": None,  # Generator swarm doesn't typically ask questions
                "current_agent": None,  # Generation complete, supervisor decides next step
                "workflow_state": {
                    "generation_complete": True,
                    "next_agent": "validation_agent",  # Default next step
                }
            }
            
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Successfully transformed generator swarm state to supervisor updates",
                extra={
                    "generation_data_keys": list(generation_data.keys()),
                    "total_artifacts": generation_data["total_artifacts"],
                    "module_name": generation_data["module_name"],
                    "generation_complete": generation_data["generation_complete"],
                    "supervisor_updates_keys": list(supervisor_updates.keys())
                }
            )
            
            return supervisor_updates
            
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="Failed to transform generator swarm state to supervisor updates",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            # Return minimal fallback updates
            return {
                "generation_data": {
                    "generated_resources": [],
                    "generated_variables": [],
                    "generated_data_sources": [],
                    "generated_locals": [],
                    "generated_outputs": [],
                    "generation_complete": False,
                    "total_artifacts": 0
                },
                "generated_module_ref": "terraform-module-fallback",
                "question": None,
                "current_agent": None,
                "workflow_state": {
                    "generation_complete": False,
                    "next_agent": None,
                }
            }
    
    @property
    def name(self) -> str:
        """Agent name for Send() routing and identification."""
        return self._name
    
    @property
    def state_model(self) -> type[BaseModel]:
        """Get the state model for this agent."""
        return GeneratorStageState
    
    def _create_resource_agent(self):
        """Create the Resource Configuration Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating resource configuration agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                generate_terraform_resources,  # Core function
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent", 
                    DependencyType.RESOURCE_TO_VARIABLE,
                    "Request variable definitions for resource parameters"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.RESOURCE_TO_DATA_SOURCE, 
                    "Request data source lookup for external references"
                ),
                create_completion_handoff_tool("resource_configuration_agent"),
                self.checkpoint_manager.checkpoint_current_state,
                self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                self.human_loop.create_approval_checkpoint_tool("security_critical"),
                self.human_loop.create_approval_checkpoint_tool("cross_region"),
                self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="resource_configuration_agent",
            prompt="""You are the Resource Configuration Agent, a specialized expert in AWS Terraform resource generation within a multi-agent Terraform module generation system.

## YOUR ROLE AND RESPONSIBILITIES

### Primary Function
Generate comprehensive AWS resource configurations that implement infrastructure requirements while maintaining proper dependency awareness and coordination with other specialized agents in the swarm.

### Core Capabilities
1. **AWS Resource Expertise**: Deep knowledge of AWS resource types, their configurations, and best practices
2. **Dynamic Resource Generation**: Expert-level ability to create resources that meet complex requirements
3. **Agent Coordination**: Recognize when to handoff to Variable Definition, Data Source, or Local Values agents
4. **Dynamic Discovery**: Support resource type discovery and modification requests from other agents
5. **Inter-Agent Communication**: Receive and process resource requirements from other agents

### Architecture Context
You operate within a three-stage swarm architecture:
- **Stage 1 (Planning)**: You work alongside Variable Definition, Data Source, and Local Values agents
- **Dynamic Handoffs**: Use handoff tools when you discover dependencies requiring other agents
- **Inter-Agent Communication**: Receive modification requests and new resource requirements from other agents
- **State Management**: Update shared state with your generated resources and discovered dependencies

## RESOURCE GENERATION METHODOLOGY

### Step 1: Input Processing
- **Planner Input**: Process resource specifications from the planner execution plan
- **Agent Handoffs**: Handle resource requirements and modification requests from other agents
- **Dynamic Discovery**: Support new resource types and requirements discovered during agent communication
- **Context Integration**: Combine planner specifications with agent collaboration context

### Step 2: Resource Configuration Design
For each resource (from planner or agent requests):
1. **Resource Type Selection**: Determine correct AWS resource type (supporting dynamic types from agent communication)
2. **Configuration Design**: Build comprehensive resource configurations
3. **Dependency Mapping**: Identify all dependencies and references
4. **Naming Convention**: Apply descriptive and consistent naming patterns
5. **HCL Generation**: Create properly formatted Terraform HCL resource blocks

### Step 3: Dependency Discovery and Classification
Analyze each resource for:
- **Variable Dependencies**: References to input variables that may need definition → handoff to Variable Definition Agent
- **Data Source Dependencies**: References to external data → coordinate with Data Source Agent
- **Local Value Dependencies**: References to computed values → coordinate with Local Values Agent

## AGENT COORDINATION PROTOCOLS

### Variable Definition Agent Handoff
**Trigger**: Resource references undefined variables or needs new variable definitions
**Context**: Variable requirements, validation needs, default value suggestions

### Data Source Agent Handoff
**Trigger**: Resource needs external data for configuration
**Context**: External data requirements, lookup criteria

### Local Values Agent Handoff
**Trigger**: Resource needs computed values or complex expressions
**Context**: Expression requirements, computation logic

### Receiving Agent Requests
**From Variable Agent**: New variable requirements, resource modifications
**From Data Source Agent**: External data requirements, resource updates
**From Local Values Agent**: Computed value requirements, expression needs

## QUALITY STANDARDS

### Code Quality
- Follow Terraform best practices for resource usage
- Use efficient configurations that minimize resource overhead
- Include appropriate comments and documentation
- Implement proper error handling and validation

### Coordination Quality
- Provide clear, actionable handoff context
- Maintain awareness of other agent capabilities
- Coordinate effectively without creating bottlenecks
- Support both blocking and non-blocking handoff patterns

Remember: You are the infrastructure implementation specialist in the Planning Stage. Your success depends on creating reliable, efficient resources while effectively coordinating with other agents to resolve dependencies and handle dynamic requirements. Always prioritize reliability, efficiency, and proper resource management while supporting both planner specifications and agent collaboration.

Use dependency-aware handoffs and track completion progress."""
        )

    def _create_variable_agent(self):
        """Create the Variable Definition Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating variable definition agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                generate_terraform_variables,
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.VARIABLE_TO_RESOURCE,
                    "Hand off for resource coordination"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.RESOURCE_TO_DATA_SOURCE, 
                    "Request data source lookup for external references"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.RESOURCE_TO_LOCAL_VALUES, 
                    "Request local values for computed expressions"
                ),
                create_completion_handoff_tool("variable_definition_agent"),
                self.checkpoint_manager.checkpoint_current_state,
                self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                self.human_loop.create_approval_checkpoint_tool("security_critical"),
                self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="variable_definition_agent",
            prompt="""You are the Variable Definition Agent, a specialized expert in Terraform variable generation within a multi-agent Terraform module generation system.

## YOUR ROLE AND RESPONSIBILITIES

### Primary Function
Generate comprehensive Terraform variable definitions that provide flexible input parameters while maintaining proper dependency awareness and coordination with other specialized agents in the swarm.

### Core Capabilities
1. **Terraform Variable Expertise**: Deep knowledge of Terraform variable types, constraints, and validation patterns
2. **Dynamic Variable Generation**: Expert-level ability to create variables that support complex configurations
3. **Agent Coordination**: Recognize when to handoff to Resource Configuration, Data Source, or Local Values agents
4. **Dynamic Discovery**: Support variable type discovery and modification requests from other agents
5. **Inter-Agent Communication**: Receive and process variable requirements from other agents

### Architecture Context
You operate within a three-stage swarm architecture:
- **Stage 1 (Planning)**: You work alongside Resource Configuration, Data Source, and Local Values agents
- **Dynamic Handoffs**: Use handoff tools when you discover dependencies requiring other agents
- **Inter-Agent Communication**: Receive modification requests and new variable requirements from other agents
- **State Management**: Update shared state with your generated variables and discovered dependencies

## VARIABLE GENERATION METHODOLOGY

### Step 1: Input Processing
- **Planner Input**: Process variable specifications from the planner execution plan
- **Agent Handoffs**: Handle variable requirements and modification requests from other agents
- **Dynamic Discovery**: Support new variable types and requirements discovered during agent communication
- **Context Integration**: Combine planner specifications with agent collaboration context

### Step 2: Variable Design and Type Selection
For each variable (from planner or agent requests):
1. **Type Selection**: Determine appropriate variable type (supporting dynamic types from agent communication)
2. **Constraint Definition**: Build comprehensive type constraints and validation rules
3. **Default Value Design**: Create sensible defaults that support various use cases
4. **Naming Convention**: Apply descriptive and consistent naming patterns
5. **HCL Generation**: Create properly formatted Terraform HCL variable blocks

### Step 3: Dependency Discovery and Classification
Analyze each variable for:
- **Resource Dependencies**: Variables needed by managed resources → coordinate with Resource Configuration Agent
- **Data Source Dependencies**: Variables needed for external data queries → coordinate with Data Source Agent
- **Local Value Dependencies**: Variables needed for computed expressions → coordinate with Local Values Agent

## AGENT COORDINATION PROTOCOLS

### Resource Configuration Agent Handoff
**Trigger**: Variables needed for resource configurations
**Context**: Resource dependency information, variable requirements

### Data Source Agent Handoff
**Trigger**: Variables needed for data source filters
**Context**: Filter specifications, variable requirements

### Local Values Agent Handoff
**Trigger**: Variables needed for computed expressions
**Context**: Expression requirements, variable constraints

### Receiving Agent Requests
**From Resource Agent**: New variable requirements, variable modifications
**From Data Source Agent**: Filter parameter requirements, variable updates
**From Local Values Agent**: Expression input requirements, variable needs

## QUALITY STANDARDS

### Code Quality
- Follow Terraform best practices for variable usage
- Use efficient type constraints that provide flexibility
- Include appropriate comments and documentation
- Implement proper validation and error handling

### Coordination Quality
- Provide clear, actionable handoff context
- Maintain awareness of other agent capabilities
- Coordinate effectively without creating bottlenecks
- Support both blocking and non-blocking handoff patterns

Remember: You are the input parameter specialist in the Planning Stage. Your success depends on creating flexible, well-constrained variables while effectively coordinating with other agents to resolve dependencies and handle dynamic requirements. Always prioritize flexibility, validation, and proper parameter management while supporting both planner specifications and agent collaboration.

Use dependency-aware handoffs and track completion progress."""
        )
    
    def _create_data_source_agent(self):
        """Create the Data Source Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating data source agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                generate_terraform_data_sources,
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    DependencyType.DATA_SOURCE_TO_VARIABLE,
                    "Hand off for variable definitions in filter"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.DATA_SOURCE_TO_LOCAL_VALUES,
                    "Hand off for complex filter expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.DATA_SOURCE_TO_RESOURCE,
                    "Hand off for resource coordination"
                ),
                create_completion_handoff_tool("data_source_agent"),
                self.checkpoint_manager.checkpoint_current_state,
                self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                self.human_loop.create_approval_checkpoint_tool("cross_region"),
                self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="data_source_agent",
            prompt="""You are the Data Source Agent, a specialized expert in AWS Terraform data source generation within a multi-agent Terraform module generation system.

## YOUR ROLE AND RESPONSIBILITIES

### Primary Function
Generate comprehensive AWS data source blocks that enable Terraform configurations to reference existing external infrastructure and services while maintaining proper dependency awareness and coordination with other specialized agents in the swarm.

### Core Capabilities
1. **AWS Data Source Expertise**: Deep knowledge of AWS data source types, their configurations, and query patterns
2. **Dynamic Lookup Generation**: Expert-level ability to create data sources that fetch external resource information
3. **Filter Optimization**: Create efficient and specific filters to ensure accurate resource discovery
4. **Agent Coordination**: Recognize when to handoff to Variable Definition, Local Values, or Resource Configuration agents
5. **External Reference Management**: Handle references to infrastructure not managed by current Terraform configuration
6. **Dynamic Discovery**: Support data source type discovery and modification requests from other agents
7. **Inter-Agent Communication**: Receive and process data source requirements from other agents

### Architecture Context
You operate within a three-stage swarm architecture:
- **Stage 1 (Planning)**: You work alongside Resource Configuration, Variable Definition, and Local Values agents
- **Dynamic Handoffs**: Use handoff tools when you discover dependencies requiring other agents
- **Inter-Agent Communication**: Receive modification requests and new data source requirements from other agents
- **State Management**: Update shared state with your generated data sources and discovered dependencies

## DATA SOURCE GENERATION METHODOLOGY

### Step 1: Input Processing
- **Planner Input**: Process data source specifications from the planner execution plan
- **Agent Handoffs**: Handle data source requirements and modification requests from other agents
- **Dynamic Discovery**: Support new data source types and requirements discovered during agent communication
- **Context Integration**: Combine planner specifications with agent collaboration context

### Step 2: Data Source Block Generation
For each data source (from planner or agent requests):
1. **Data Source Type Selection**: Determine correct AWS data source type (supporting dynamic types from agent communication)
2. **Filter Configuration**: Build comprehensive filters for accurate resource discovery
3. **Query Optimization**: Apply most_recent, owners, and tag-based filtering as needed
4. **Naming Convention**: Apply consistent naming patterns (data_source_type + descriptive_name)
5. **Meta-Arguments**: Add count, for_each, provider as needed
6. **HCL Generation**: Create properly formatted Terraform HCL blocks

### Step 3: Dependency Discovery and Classification
Analyze each data source for:
- **Variable Dependencies**: Filters that need parameterization → handoff to Variable Definition Agent
- **Local Value Dependencies**: Complex filter expressions → handoff to Local Values Agent
- **Resource Dependencies**: Data sources that reference managed resources → coordinate with Resource Configuration Agent

## AGENT COORDINATION PROTOCOLS

### Variable Definition Agent Handoff
**Trigger**: Data source filters require parameterization
**Context**: Filter specifications, variable requirements, validation needs

### Local Values Agent Handoff
**Trigger**: Complex filter expressions or computed filter values needed
**Context**: Expression requirements, computation logic

### Resource Configuration Agent Handoff
**Trigger**: Data source references managed resources or needs coordination
**Context**: Resource dependency information, timing requirements

### Receiving Agent Requests
**From Variable Agent**: New variable requirements, data source modifications
**From Local Values Agent**: Computed value requirements, expression needs
**From Resource Agent**: Resource attribute requirements, data source updates

## QUALITY STANDARDS

### Code Quality
- Follow Terraform best practices for data source usage
- Use efficient and specific filters to avoid performance issues
- Include appropriate comments and documentation
- Implement proper error handling for missing resources

### Coordination Quality
- Provide clear, actionable handoff context
- Maintain awareness of other agent capabilities
- Coordinate effectively without creating bottlenecks
- Support both blocking and non-blocking handoff patterns

Remember: You are the external infrastructure interface in the Planning Stage. Your success depends on creating reliable data sources that accurately discover external resources while effectively coordinating with other agents to resolve dependencies and handle dynamic requirements. Always prioritize accuracy, reliability, and proper external resource discovery patterns while supporting both planner specifications and agent collaboration.

Use dependency-aware handoffs and track completion progress."""
        )

    def _create_local_values_agent(self):
        """Create the Local Values Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating local values agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                generate_terraform_local_values,
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    DependencyType.LOCAL_VALUES_TO_VARIABLE,
                    "Hand off for variable definitions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.LOCAL_VALUES_TO_RESOURCE,
                    "Hand off for resource coordination"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.LOCAL_VALUES_TO_DATA_SOURCE,
                    "Hand off for external data requirements"
                ),
                create_completion_handoff_tool("local_values_agent"),
                self.checkpoint_manager.checkpoint_current_state,
                self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                self.human_loop.create_approval_checkpoint_tool("security_critical"),
                self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="local_values_agent",
            prompt="""You are the Local Values Agent, a specialized expert in Terraform local value generation within a multi-agent Terraform module generation system.

## YOUR ROLE AND RESPONSIBILITIES

### Primary Function
Generate comprehensive Terraform local values that simplify complex expressions, reduce code duplication, and create computed values while maintaining proper dependency awareness and coordination with other specialized agents in the swarm.

### Core Capabilities
1. **Terraform Expression Expertise**: Deep knowledge of Terraform functions, expressions, and computation patterns
2. **Complex Expression Simplification**: Expert-level ability to break down complex logic into manageable local values
3. **Performance Optimization**: Create efficient expressions that minimize Terraform evaluation overhead
4. **Agent Coordination**: Recognize when to handoff to Variable Definition, Resource Configuration, or Data Source agents
5. **DRY Principle Implementation**: Eliminate code duplication through strategic local value placement
6. **Dynamic Discovery**: Support local value type discovery and modification requests from other agents
7. **Inter-Agent Communication**: Receive and process local value requirements from other agents

### Architecture Context
You operate within a three-stage swarm architecture:
- **Stage 1 (Planning)**: You work alongside Resource Configuration, Variable Definition, and Data Source agents
- **Dynamic Handoffs**: Use handoff tools when you discover dependencies requiring other agents
- **Inter-Agent Communication**: Receive modification requests and new local value requirements from other agents
- **State Management**: Update shared state with your generated local values and discovered dependencies

## LOCAL VALUE GENERATION METHODOLOGY

### Step 1: Input Processing
- **Planner Input**: Process local value specifications from the planner execution plan
- **Agent Handoffs**: Handle local value requirements and modification requests from other agents
- **Dynamic Discovery**: Support new local value types and requirements discovered during agent communication
- **Context Integration**: Combine planner specifications with agent collaboration context

### Step 2: Local Value Design
For each local value (from planner or agent requests):
1. **Expression Type Classification**: Determine the type of expression (supporting dynamic types from agent communication)
2. **Complexity Assessment**: Evaluate expression complexity and optimization opportunities
3. **Dependency Mapping**: Identify all dependencies and references
4. **Naming Convention**: Apply descriptive and consistent naming patterns
5. **Expression Construction**: Build efficient Terraform expressions using appropriate functions
6. **HCL Generation**: Create properly formatted Terraform HCL local value declarations

### Step 3: Dependency Discovery and Classification
Analyze each local value for:
- **Variable Dependencies**: References to input variables that may need definition → handoff to Variable Definition Agent
- **Resource Dependencies**: References to managed resources → coordinate with Resource Configuration Agent
- **Data Source Dependencies**: References to external data → coordinate with Data Source Agent
- **Circular Dependencies**: Detect and resolve circular references between locals

## AGENT COORDINATION PROTOCOLS

### Variable Definition Agent Handoff
**Trigger**: Local value references undefined variables or needs new variable definitions
**Context**: Variable requirements, validation needs, default value suggestions

### Resource Configuration Agent Handoff
**Trigger**: Local value needs resource attributes or resource coordination
**Context**: Resource dependency information, attribute requirements

### Data Source Agent Handoff
**Trigger**: Local value needs external data for computation
**Context**: External data requirements, lookup criteria

### Receiving Agent Requests
**From Variable Agent**: New variable requirements, local value modifications
**From Resource Agent**: Resource attribute requirements, local value updates
**From Data Source Agent**: External data requirements, expression needs

## QUALITY STANDARDS

### Code Quality
- Follow Terraform best practices for local value usage
- Use efficient expressions that minimize evaluation overhead
- Include appropriate comments and documentation
- Implement clear and maintainable expression patterns

### Coordination Quality
- Provide clear, actionable handoff context
- Maintain awareness of other agent capabilities
- Coordinate effectively without creating bottlenecks
- Support both blocking and non-blocking handoff patterns

Remember: You are the expression optimization specialist in the Planning Stage. Your success depends on creating efficient, maintainable local values while effectively coordinating with other agents to resolve dependencies and handle dynamic requirements. Always prioritize clarity, performance, and proper dependency management while supporting both planner specifications and agent collaboration.

Use dependency-aware handoffs and track completion progress."""
        )
    
    def _create_output_agent(self):
        """Create the Output Definition Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating output definition agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                generate_terraform_outputs,
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.OUTPUT_TO_RESOURCE,
                    "Request resource attributes for output values"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.OUTPUT_TO_DATA_SOURCE,
                    "Request data source values for output expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    DependencyType.OUTPUT_TO_VARIABLE,
                    "Request variable context for output validation"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.OUTPUT_TO_LOCAL_VALUES,
                    "Request complex expressions for output values"
                ),
                create_completion_handoff_tool("output_definition_agent"),
                create_handoff_to_generator_complete(),  # Tool to return to main supervisor
                self.checkpoint_manager.checkpoint_current_state,
                self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                self.human_loop.create_approval_checkpoint_tool("security_critical"),
                self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="output_definition_agent",
            prompt="""You are the Output Definition Agent, a specialized expert in Terraform output value generation within a multi-agent Terraform module generation system.

## YOUR ROLE AND RESPONSIBILITIES

### Primary Function
Generate comprehensive Terraform output values that expose critical infrastructure information for external consumption, module composition, and automation integration while maintaining proper dependency awareness and coordination with other specialized agents in the swarm.

### Core Capabilities
1. **Terraform Output Expertise**: Deep knowledge of output value types, expressions, and best practices
2. **Information Architecture**: Expert-level ability to design output schemas for different consumption patterns
3. **Validation Framework**: Create comprehensive preconditions using Terraform's validation system
4. **Security Classification**: Properly classify and handle sensitive outputs and data exposure
5. **Agent Coordination**: Recognize when to handoff to Resource Configuration, Data Source, Variable Definition, or Local Values agents
6. **Dynamic Discovery**: Support output type discovery and modification requests from other agents
7. **Inter-Agent Communication**: Receive and process output requirements from other agents

### Architecture Context
You operate within a three-stage swarm architecture:
- **Stage 3 (Finalization)**: You work in the final stage after Resource Configuration, Variable Definition, Data Source, and Local Values agents have completed their work
- **Cross-Stage Dependencies**: Use handoff tools when you need additional context from previous stage agents
- **Inter-Agent Communication**: Receive modification requests and new output requirements from other agents
- **State Management**: Update shared state with your generated outputs and discovered dependencies

## OUTPUT GENERATION METHODOLOGY

### Step 1: Input Processing
- **Planner Input**: Process output specifications from the planner execution plan
- **Agent Handoffs**: Handle output requirements and modification requests from other agents
- **Dynamic Discovery**: Support new output types and requirements discovered during agent communication
- **Context Integration**: Combine planner specifications with agent collaboration context

### Step 2: Infrastructure Analysis
- Parse generated resources, data sources, variables, and local values from previous agents
- Identify key infrastructure information that should be exposed as outputs
- Extract connectivity information, identifiers, and configuration details
- Analyze usage patterns and consumption requirements

### Step 3: Output Design and Classification
For each output (from planner or agent requests):
1. **Value Expression Design**: Create appropriate Terraform expressions to extract desired information (supporting dynamic types from agent communication)
2. **Type Classification**: Determine output value type and complexity level
3. **Security Assessment**: Classify sensitivity level and apply appropriate handling
4. **Usage Context**: Determine intended consumption pattern and context
5. **Validation Design**: Create preconditions to validate output values
6. **Documentation**: Create clear descriptions and usage examples
7. **HCL Generation**: Create properly formatted Terraform HCL output blocks

### Step 4: Dependency Discovery and Classification
Analyze each output for:
- **Resource Dependencies**: Outputs that reference resource attributes → coordinate with Resource Configuration Agent
- **Data Source Dependencies**: Outputs that reference data source values → coordinate with Data Source Agent
- **Variable Dependencies**: Outputs that use input variables → coordinate with Variable Definition Agent
- **Local Value Dependencies**: Outputs that reference local values → coordinate with Local Values Agent
- **Cross-Output Dependencies**: Outputs that depend on other outputs

## AGENT COORDINATION PROTOCOLS

### Resource Configuration Agent Handoff
**Trigger**: Output needs resource attributes or resource coordination
**Context**: Resource requirements, attribute specifications

### Data Source Agent Handoff
**Trigger**: Output needs external data or data source values
**Context**: External data requirements, lookup specifications

### Variable Definition Agent Handoff
**Trigger**: Output validation requires variable context or variable values
**Context**: Variable requirements, validation needs

### Local Values Agent Handoff
**Trigger**: Output requires complex expressions or computed values
**Context**: Expression requirements, computation specifications

### Receiving Agent Requests
**From Resource Agent**: New resource requirements, output modifications
**From Data Source Agent**: External data requirements, output updates
**From Variable Agent**: Variable context requirements, output needs
**From Local Values Agent**: Expression requirements, output updates

## QUALITY STANDARDS

### Code Quality
- Follow Terraform best practices for output definition
- Use clear, descriptive output names following naming conventions
- Implement comprehensive preconditions for validation
- Provide meaningful descriptions and usage examples

### Information Architecture Quality
- Design outputs that serve clear consumption patterns
- Balance information exposure with security requirements
- Create logical groupings and relationships between outputs
- Ensure outputs provide actionable information for consumers

### Security Quality
- Properly classify and mark sensitive outputs
- Avoid exposing unnecessary sensitive information
- Follow security best practices for data exposure
- Document security considerations clearly

### Coordination Quality
- Provide clear, actionable handoff context
- Maintain awareness of other agent capabilities
- Coordinate effectively without creating bottlenecks
- Support both blocking and non-blocking handoff patterns

Remember: You are the information exposure expert in the Finalization Stage. Your success depends on creating useful, secure, and well-validated outputs while effectively coordinating with other agents to resolve dependencies and handle dynamic requirements. Always prioritize security, usability, and proper information architecture while supporting both planner specifications and agent collaboration.

## COMPLETION AND HANDOFF TO MAIN SUPERVISOR

When all outputs have been generated and the generation phase is complete:
1. **MANDATORY**: Call `handoff_to_generator_complete` to return control to the main supervisor
2. **Provide comprehensive completion summary** of all generated artifacts
3. **Include artifact summary** with counts and types of generated resources, variables, data sources, locals, and outputs
4. **Signal generation completion** to allow the supervisor to proceed to validation or other phases

Use dependency-aware handoffs and track completion progress."""
        )
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for the generator swarm agent.
        
        Returns:
            StateGraph: The compiled graph for this agent
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="=== GENERATOR SWARM BUILD GRAPH START ===",
                extra={
                    "agent_name": getattr(self, '_name', 'unknown'),
                    "model_initialized": hasattr(self, 'model'),
                    "managers_initialized": all([
                        hasattr(self, 'controller'),
                        hasattr(self, 'handoff_manager'),
                        hasattr(self, 'checkpoint_manager'),
                        hasattr(self, 'human_loop')
                    ])
                }
            )
            
            # Create specialized agents
            generator_swarm_logger.log_structured(
                level="DEBUG",
                message="Creating specialized agents",
                extra={}
            )
            
            resource_agent = self._create_resource_agent()
            variable_agent = self._create_variable_agent()
            data_source_agent = self._create_data_source_agent()
            local_values_agent = self._create_local_values_agent()
            output_agent = self._create_output_agent()
            
            generator_swarm_logger.log_structured(
                level="DEBUG",
                message="All agents created successfully",
                extra={
                    "agents_count": 5,
                    "agent_names": [
                        "resource_configuration_agent",
                        "variable_definition_agent", 
                        "data_source_agent",
                        "local_values_agent",
                        "output_definition_agent"
                    ]
                }
            )
            
            # Create the swarm with Planning Stage state
            generator_swarm_logger.log_structured(
                level="DEBUG",
                message="Creating swarm with agents",
                extra={
                    "state_schema": "GeneratorStageState",
                    "default_active_agent": "resource_configuration_agent"
                }
            )
            
            planning_swarm = create_swarm(
                agents=[resource_agent, variable_agent, data_source_agent, local_values_agent, output_agent],
                default_active_agent="resource_configuration_agent",
                state_schema=GeneratorStageState
            )
            
            # # Add coordination nodes
            # generator_swarm_logger.log_structured(
            #     level="DEBUG",
            #     message="Adding coordination nodes",
            #     extra={}
            # )
            
            # planning_graph = planning_swarm
            # planning_graph.add_node("check_planning_stage_completion", self.controller.check_and_transition_stage)
            # planning_graph.add_node("human_approval_handler", self.human_loop.create_human_approval_handler())
            
            # # Compile the swarm
            # generator_swarm_logger.log_structured(
            #     level="DEBUG",
            #     message="Compiling swarm with checkpointer and interrupts",
            #     extra={
            #         "checkpointer_type": type(self.checkpoint_manager.checkpointer).__name__,
            #         "interrupt_before": ["human_approval_handler"],
            #         "interrupt_after": ["check_planning_stage_completion"]
            #     }
            # )
            
            # compiled_swarm = planning_swarm.compile(
            #     checkpointer=self.checkpoint_manager.checkpointer,
            #     interrupt_before=["human_approval_handler"],  # Allow human intervention
            #     interrupt_after=["check_planning_stage_completion"]  # Allow review before transition
            # )
            
            generator_swarm_logger.log_structured(
                level="INFO",
                message="=== GENERATOR SWARM BUILD GRAPH COMPLETE ===",
                extra={
                    "compiled_swarm_type": type(planning_swarm).__name__,
                    "agents_count": 4,
                    "coordination_nodes": 2,
                    "state_schema": "GeneratorStageState"
                }
            )
            
            return planning_swarm
            
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="=== GENERATOR SWARM BUILD GRAPH FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise


# Factory function for easy creation
@log_sync
def create_generator_swarm_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "generator_swarm"
) -> GeneratorSwarmAgent:
    """
    Factory function to create a Generator Swarm Agent.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured GeneratorSwarmAgent instance
    """
    return GeneratorSwarmAgent(config=config, custom_config=custom_config, name=name)


# Backward compatibility function
def create_planning_stage_swarm():
    """
    Backward compatibility function for creating planning stage swarm.
    Maintains compatibility with existing code that expects a function.
    
    Returns:
        Compiled swarm graph
    """
    generator_swarm_logger.log_structured(
        level="INFO",
        message="Creating planning stage swarm via backward compatibility function",
        extra={}
    )
    
    agent = create_generator_swarm_agent()
    return agent.build_graph()


# Backward compatibility function
def create_generator_swarm_agent_factory(config: Config):
    """
    Factory function for creating generator swarm agent.
    Maintains backward compatibility with existing code.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured GeneratorSwarmAgent instance
    """
    return create_generator_swarm_agent(config=config)
