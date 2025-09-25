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
from typing import Dict, Any, Optional, List, Annotated
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph_swarm import create_swarm
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync
from aws_orchestrator_agent.core.agents_latest.agents.base_agent import BaseSubgraphAgent
from aws_orchestrator_agent.core.agents_latest.types import StateTransformer
from .generator_state import GeneratorSwarmState, GeneratorAgentStatus, DependencyType
from aws_orchestrator_agent.core.agents_latest.types import SupervisorState
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

        self.generator_swarm_state = None
        
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
    
    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Send() payload from supervisor to agent state.
        
        The supervisor uses Send() to pass data to subgraph agents. This method
        extracts the task description and transforms it into the agent's state format.
        
        Args:
            send_payload: Data sent from supervisor via Send() primitive
                - messages: List of messages, typically [{"role": "user", "content": task_description}]
                - Any other data included in the Send() call
        
        Returns:
            Dict[str, Any]: Transformed state ready for agent processing
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Transforming Send() payload to generator stage state",
                extra={
                    "send_payload_keys": list(send_payload.keys()),
                    "has_messages": "messages" in send_payload,
                    "messages_count": len(send_payload.get("messages", []))
                }
            )
            
            # Extract task description from Send payload (following base class pattern)
            task_description = ""
            if "messages" in send_payload and send_payload["messages"]:
                task_description = send_payload["messages"][0].get("content", "")
            
            # For generator swarm, we need to reconstruct the full supervisor state
            # from the send_payload and then transform it to GeneratorStageState
            # This is a workaround since the base class expects simple payloads
            
            # Create a minimal SupervisorState-like object for transformation
            # We'll need to get the full supervisor state from somewhere else
            # For now, create a basic structure that StateTransformer can work with
            supervisor_state_dict = {
                "user_request": task_description,
                "messages": send_payload.get("messages", []),
                "session_id": send_payload.get("session_id"),
                "task_id": send_payload.get("task_id"),
                "planner_data": send_payload.get("planner_data", {}),
                "workspace_ref": send_payload.get("workspace_ref"),
                "terraform_context": send_payload.get("terraform_context", {}),
            }
            
            # Create a SupervisorState object for transformation
            supervisor_state = SupervisorState(**supervisor_state_dict)
            
            # Transform to GeneratorSwarmState using existing StateTransformer
            generator_state = StateTransformer.supervisor_to_generator_swarm(supervisor_state)
            
            # Convert GeneratorSwarmState to dict (following base class pattern)
            return generator_state
            
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="Failed to transform supervisor state for generator swarm",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "supervisor_state_type": type(supervisor_state).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            # Return minimal generator state
            from .generator_state import GeneratorSwarmState
            return dict(GeneratorSwarmState(
                internal_messages=[HumanMessage(content="Generate Terraform module from execution plan")],
                active_agent="resource_configuration_agent"
            ))
    
    
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform agent state back to supervisor state.
        
        This method prepares the agent's result for merging back into the
        supervisor's state. Only include data that should propagate to supervisor.
        
        Args:
            agent_state: The final state from agent execution
        
        Returns:
            Dict[str, Any]: Data to merge into supervisor state
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Transforming generator swarm state back to supervisor state",
                extra={
                    "agent_state_type": type(agent_state).__name__,
                    "stage_status": agent_state.get("stage_status", "unknown"),
                    "current_stage": agent_state.get("current_stage", "unknown")
                }
            )
            
            # Convert dict back to GeneratorSwarmState for transformation
            from .generator_state import GeneratorSwarmState
            generator_state = GeneratorSwarmState(**agent_state)
            
            # Transform back to supervisor updates using StateTransformer
            supervisor_updates = StateTransformer.generator_to_supervisor(generator_state)
            
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Successfully transformed generator swarm state to supervisor updates",
                extra={
                    "supervisor_updates_keys": list(supervisor_updates.keys()),
                    "generation_data_keys": list(supervisor_updates.get("generation_data", {}).keys())
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
                    "generated_module": {
                        "resources": [],
                        "variables": [],
                        "data_sources": [],
                        "locals": [],
                        "outputs": []
                    },
                    "status": "error",
                    "error": str(e)
                },
                "status": "failed",
                "current_agent": None
            }
    
    @property
    def name(self) -> str:
        """Agent name for Send() routing and identification."""
        return self._name
    
    @property
    def state_model(self) -> type[BaseModel]:
        """Get the state model for this agent."""
        from .generator_state import GeneratorSwarmState
        return GeneratorSwarmState
    
    def _create_resource_agent(self):
        """Create the Resource Configuration Agent."""
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating resource configuration agent",
            extra={}
        )
        # if input_state is None:
        #     input_state = self.generator_swarm_state
        
        # Create a state injection tool that ensures the current state is available
        @tool("inject_current_state_for_terraform")
        def inject_current_state_for_terraform(
            state: Annotated[Any, InjectedState] = None,
        ) -> dict:
            """Inject the current state into the graph state for terraform resource generation."""
            current_state = getattr(self, 'generator_swarm_state', None)
            if current_state:
                # Merge current_state into the existing state
                merged_state = {**(state or {}), **current_state}
                
                generator_swarm_logger.log_structured(
                    level="DEBUG",
                    message="Injecting current state for terraform resource generation",
                    extra={
                        "current_state_keys": list(current_state.keys()),
                        "merged_state_keys": list(merged_state.keys()),
                        "has_execution_plan_data": "execution_plan_data" in merged_state,
                        "has_agent_workspaces": "agent_workspaces" in merged_state,
                        "has_planning_context": "planning_context" in merged_state
                    }
                )
                
                # Return the merged state as updates
                return merged_state
            else:
                return state or {}
        
        # Create the agent
        agent = create_react_agent(
            model=self.model,
            # state_schema=input_state,
            tools=[
                inject_current_state_for_terraform,  # State injection tool
                generate_terraform_resources,  # Main tool that uses InjectedState
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
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.RESOURCE_TO_LOCAL_VALUES, 
                    "Request local values for computed expressions"
                ),
                create_completion_handoff_tool("resource_configuration_agent")
                # Checkpoint and HITL tools will be added in next release
                # self.checkpoint_manager.checkpoint_current_state,
                # self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                # self.human_loop.create_approval_checkpoint_tool("security_critical"),
                # self.human_loop.create_approval_checkpoint_tool("cross_region"),
                # self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="resource_configuration_agent",
            prompt="""You are the Resource Configuration Agent, an AWS Terraform expert in a multi-agent generation system.

## CORE MISSION
Generate production-ready AWS Terraform resources that implement infrastructure requirements while coordinating with specialized agents for dependencies.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, and Local Values agents
- **Workflow**: Planning stage with dynamic handoffs based on dependencies
- **State**: Shared GeneratorSwarmState with agent workspaces and planning context

## STATE STRUCTURE
The state contains the following key fields:
- `execution_plan_data`: Contains execution plans with resource configurations
- `agent_workspaces.resource_configuration_agent`: Contains your workspace data including planner_input
- `planning_context`: Contains planning requirements and context
- `active_agent`: Current active agent (should be "resource_configuration_agent")

## INPUT PROCESSING
1. **Planner Data**: Process resource_configurations from execution_plan_data
2. **Agent Requests**: Handle handoffs from other agents requiring new resources
3. **Context Integration**: Combine planner specs with agent collaboration context

## CRITICAL: TOOL USAGE
**MANDATORY WORKFLOW:**
1. **FIRST**: Call `inject_current_state_for_terraform` to inject the current state into the graph
   - This tool returns the merged state data that will be used as input for the next tool
2. **THEN**: Call `generate_terraform_resources` to process the input data and generate the actual Terraform resources
   - This tool will automatically receive the state data returned by the injection tool

Do not provide general responses - use the tools to perform the actual work. The state injection tool must be called first to ensure the current state is available for resource generation.

## RESOURCE GENERATION PROTOCOL

### Step 1: Resource Analysis
For each resource specification:
- **Type Selection**: Determine correct AWS resource type (aws_vpc, aws_subnet, aws_security_group, etc.)
- **Configuration Mapping**: Map requirements to Terraform resource arguments
- **Dependency Identification**: Identify variable, data source, and local value dependencies

### Step 2: HCL Generation
Generate properly formatted Terraform HCL:
```hcl
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-vpc"
    Environment = var.environment
  }
}
```

### Step 3: Dependency Coordination
- **Variable Dependencies**: Use handoff_to_variable_definition_agent for undefined variables
- **Data Source Dependencies**: Use handoff_to_data_source_agent for external data lookups
- **Local Value Dependencies**: Use handoff_to_local_values_agent for computed expressions

## QUALITY REQUIREMENTS

### Code Standards
- Follow AWS Terraform best practices
- Use consistent naming: `${var.project_name}-${resource_type}-${descriptive_name}`
- Include appropriate tags for all resources
- Add comments for complex configurations
- Validate resource arguments and constraints

### Coordination Standards
- Provide clear handoff context with specific requirements
- Update agent workspace with generated resources
- Track completion progress accurately
- Handle both blocking and non-blocking handoffs

## ERROR HANDLING
- **Invalid Input**: Log error and request clarification via handoff
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Resource Conflicts**: Coordinate with other agents to resolve
- **Validation Failures**: Provide specific error details and suggested fixes

## COMPLETION CRITERIA
- All planner-specified resources generated
- All agent-requested resources completed
- Dependencies properly coordinated
- Resources validated and properly formatted
- Call resource_configuration_agent_complete_task when done

## EXAMPLES

### Good Resource Generation:
```hcl
resource "aws_security_group" "web" {
  name_prefix = "${var.project_name}-web-"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.web_cidr_blocks]
  }
  
  tags = {
    Name        = "${var.project_name}-web-sg"
    Environment = var.environment
  }
}
```

### Proper Handoff Example:
When resource needs undefined variable:
- **Tool**: handoff_to_variable_definition_agent
- **Context**: "Resource aws_security_group.web requires variable 'web_cidr_blocks' of type list(string) for ingress rules"

Remember: You are the infrastructure implementation specialist. Prioritize reliability, efficiency, and proper coordination with other agents."""
        )
        
        return agent

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
                generate_terraform_variables,  # Core function
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.VARIABLE_TO_RESOURCE,
                    "Request resource coordination for variable dependencies"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.VARIABLE_TO_DATA_SOURCE, 
                    "Request data source lookup for external references"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.VARIABLE_TO_LOCAL_VALUES, 
                    "Request local values for computed expressions"
                ),
                create_completion_handoff_tool("variable_definition_agent")
                # Checkpoint and HITL tools will be added in next release
                # self.checkpoint_manager.checkpoint_current_state,
                # self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                # self.human_loop.create_approval_checkpoint_tool("security_critical"),
                # self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="variable_definition_agent",
            prompt="""You are the Variable Definition Agent, a Terraform variable expert in a multi-agent generation system.

## CORE MISSION
Generate production-ready Terraform variable definitions that provide flexible input parameters while coordinating with specialized agents for dependencies.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, and Local Values agents
- **Workflow**: Planning stage with dynamic handoffs based on dependencies
- **State**: Shared GeneratorStageState with agent workspaces and planning context

## INPUT PROCESSING
1. **Planner Data**: Process variable_definitions from execution_plan_data
2. **Agent Requests**: Handle handoffs from other agents requiring new variables
3. **Context Integration**: Combine planner specs with agent collaboration context

## CRITICAL: TOOL USAGE
**ALWAYS start by calling the `generate_terraform_variables` tool** with the execution plan data, agent workspace, and planning context to process the input data and generate the actual Terraform variables. Do not provide general responses - use the tool to perform the actual work.

**Tool Call Format:**
```json
{
  "execution_plan_data": {...},
  "agent_workspace": {...},
  "planning_context": {...}
}
```

## VARIABLE GENERATION PROTOCOL

### Step 1: Variable Analysis
For each variable specification:
- **Type Selection**: Determine appropriate Terraform type (string, number, bool, list, map, object, etc.)
- **Constraint Mapping**: Map requirements to type constraints and validation rules
- **Default Value Design**: Create sensible defaults that support various use cases
- **Dependency Identification**: Identify resource, data source, and local value dependencies

### Step 2: HCL Generation
Generate properly formatted Terraform HCL:
```hcl
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "The vpc_cidr must be a valid IPv4 CIDR block."
  }
}
```

### Step 3: Dependency Coordination
- **Resource Dependencies**: Use handoff_to_resource_configuration_agent for resource coordination
- **Data Source Dependencies**: Use handoff_to_data_source_agent for external data queries
- **Local Value Dependencies**: Use handoff_to_local_values_agent for computed expressions

## QUALITY REQUIREMENTS

### Code Standards
- Follow Terraform best practices for variable definitions
- Use descriptive names: `project_name`, `environment`, `vpc_cidr`, etc.
- Include comprehensive descriptions for all variables
- Add validation rules for critical variables
- Mark sensitive variables appropriately

### Coordination Standards
- Provide clear handoff context with specific requirements
- Update agent workspace with generated variables
- Track completion progress accurately
- Handle both blocking and non-blocking handoffs

## ERROR HANDLING
- **Invalid Input**: Log error and request clarification via handoff
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Type Conflicts**: Coordinate with other agents to resolve
- **Validation Failures**: Provide specific error details and suggested fixes

## COMPLETION CRITERIA
- All planner-specified variables generated
- All agent-requested variables completed
- Dependencies properly coordinated
- Variables validated and properly formatted
- Call variable_definition_agent_complete_task when done

## EXAMPLES

### Good Variable Generation:
```hcl
variable "instance_type" {
  description = "EC2 instance type for the application servers"
  type        = string
  default     = "t3.micro"
  
  validation {
    condition     = can(regex("^t[0-9]+\\.[a-z]+$", var.instance_type))
    error_message = "Instance type must be a valid EC2 instance type."
  }
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]
  
  validation {
    condition     = length(var.allowed_cidr_blocks) > 0
    error_message = "At least one CIDR block must be specified."
  }
}
```

### Proper Handoff Example:
When variable needs resource coordination:
- **Tool**: handoff_to_resource_configuration_agent
- **Context**: "Variable 'instance_type' is required by aws_instance.web for resource configuration"

Remember: You are the input parameter specialist. Prioritize flexibility, validation, and proper coordination with other agents."""
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
                generate_terraform_data_sources,  # Core function
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    DependencyType.DATA_SOURCE_TO_VARIABLE,
                    "Request variable definitions for data source filters"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    DependencyType.DATA_SOURCE_TO_LOCAL_VALUES,
                    "Request local values for complex filter expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.DATA_SOURCE_TO_RESOURCE,
                    "Request resource coordination for data source dependencies"
                ),
                create_completion_handoff_tool("data_source_agent")
                # self.checkpoint_manager.checkpoint_current_state,
                # self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                # self.human_loop.create_approval_checkpoint_tool("cross_region"),
                # self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="data_source_agent",
            prompt="""You are the Data Source Agent, an AWS Terraform data source expert in a multi-agent generation system.

## CORE MISSION
Generate production-ready AWS Terraform data sources that discover external infrastructure while coordinating with specialized agents for dependencies.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, and Local Values agents
- **Workflow**: Planning stage with dynamic handoffs based on dependencies
- **State**: Shared GeneratorStageState with agent workspaces and planning context

## INPUT PROCESSING
1. **Planner Data**: Process data_sources from execution_plan_data
2. **Agent Requests**: Handle handoffs from other agents requiring new data sources
3. **Context Integration**: Combine planner specs with agent collaboration context

## CRITICAL: TOOL USAGE
**ALWAYS start by calling the `generate_terraform_data_sources` tool** with the execution plan data, agent workspace, and planning context to process the input data and generate the actual Terraform data sources. Do not provide general responses - use the tool to perform the actual work.

**Tool Call Format:**
```json
{
  "execution_plan_data": {...},
  "agent_workspace": {...},
  "planning_context": {...}
}
```

## DATA SOURCE GENERATION PROTOCOL

### Step 1: Data Source Analysis
For each data source specification:
- **Type Selection**: Determine correct AWS data source type (aws_ami, aws_vpc, aws_subnets, etc.)
- **Filter Configuration**: Build efficient filters for accurate resource discovery
- **Dependency Identification**: Identify variable, local value, and resource dependencies

### Step 2: HCL Generation
Generate properly formatted Terraform HCL:
```hcl
data "aws_ami" "latest_amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
```

### Step 3: Dependency Coordination
- **Variable Dependencies**: Use handoff_to_variable_definition_agent for filter parameterization
- **Local Value Dependencies**: Use handoff_to_local_values_agent for complex filter expressions
- **Resource Dependencies**: Use handoff_to_resource_configuration_agent for resource coordination

## QUALITY REQUIREMENTS

### Code Standards
- Follow AWS Terraform best practices for data source usage
- Use efficient and specific filters to avoid performance issues
- Include appropriate comments and documentation
- Implement proper error handling for missing resources
- Use consistent naming: `data.${data_source_type}.${descriptive_name}`

### Coordination Standards
- Provide clear handoff context with specific requirements
- Update agent workspace with generated data sources
- Track completion progress accurately
- Handle both blocking and non-blocking handoffs

## ERROR HANDLING
- **Invalid Input**: Log error and request clarification via handoff
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Filter Conflicts**: Coordinate with other agents to resolve
- **Resource Not Found**: Provide specific error details and suggested fixes

## COMPLETION CRITERIA
- All planner-specified data sources generated
- All agent-requested data sources completed
- Dependencies properly coordinated
- Data sources validated and properly formatted
- Call data_source_agent_complete_task when done

## EXAMPLES

### Good Data Source Generation:
```hcl
data "aws_vpc" "existing" {
  filter {
    name   = "tag:Name"
    values = [var.vpc_name]
  }
  
  filter {
    name   = "state"
    values = ["available"]
  }
}
```

### Proper Handoff Example:
When data source needs undefined variable:
- **Tool**: handoff_to_variable_definition_agent
- **Context**: "Data source aws_vpc.existing requires variable 'vpc_name' of type string for tag filter"

Remember: You are the external infrastructure discovery specialist. Prioritize accuracy, efficiency, and proper coordination with other agents."""
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
                generate_terraform_local_values,  # Core function
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    DependencyType.LOCAL_VALUES_TO_VARIABLE,
                    "Request variable definitions for local value expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    DependencyType.LOCAL_VALUES_TO_RESOURCE,
                    "Request resource coordination for local value dependencies"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    DependencyType.LOCAL_VALUES_TO_DATA_SOURCE,
                    "Request external data for local value computations"
                ),
                create_completion_handoff_tool("local_values_agent")
                # self.checkpoint_manager.checkpoint_current_state,
                # self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                # self.human_loop.create_approval_checkpoint_tool("security_critical"),
                # self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="local_values_agent",
            prompt="""You are the Local Values Agent, a Terraform expression expert in a multi-agent generation system.

## CORE MISSION
Generate production-ready Terraform local values that simplify complex expressions and reduce code duplication while coordinating with specialized agents for dependencies.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, and Local Values agents
- **Workflow**: Planning stage with dynamic handoffs based on dependencies
- **State**: Shared GeneratorStageState with agent workspaces and planning context

## INPUT PROCESSING
1. **Planner Data**: Process local_values from execution_plan_data
2. **Agent Requests**: Handle handoffs from other agents requiring new local values
3. **Context Integration**: Combine planner specs with agent collaboration context

## CRITICAL: TOOL USAGE
**ALWAYS start by calling the `generate_terraform_local_values` tool** with the execution plan data, agent workspace, and planning context to process the input data and generate the actual Terraform local values. Do not provide general responses - use the tool to perform the actual work.

**Tool Call Format:**
```json
{
  "execution_plan_data": {...},
  "agent_workspace": {...},
  "planning_context": {...}
}
```

## LOCAL VALUE GENERATION PROTOCOL

### Step 1: Expression Analysis
For each local value specification:
- **Type Classification**: Determine expression type (computed, derived, transformed, etc.)
- **Complexity Assessment**: Evaluate expression complexity and optimization opportunities
- **Dependency Identification**: Identify variable, resource, and data source dependencies

### Step 2: HCL Generation
Generate properly formatted Terraform HCL:
```hcl
locals {
  # Common tags used across all resources
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  # VPC CIDR calculations
  vpc_cidr_blocks = {
    public  = cidrsubnet(var.vpc_cidr, 8, 0)
    private = cidrsubnet(var.vpc_cidr, 8, 1)
    database = cidrsubnet(var.vpc_cidr, 8, 2)
  }
  
  # Resource naming conventions
  name_prefix = "${var.project_name}-${var.environment}"
}
```

### Step 3: Dependency Coordination
- **Variable Dependencies**: Use handoff_to_variable_definition_agent for undefined variables
- **Resource Dependencies**: Use handoff_to_resource_configuration_agent for resource coordination
- **Data Source Dependencies**: Use handoff_to_data_source_agent for external data

## QUALITY REQUIREMENTS

### Code Standards
- Follow Terraform best practices for local value usage
- Use efficient expressions that minimize evaluation overhead
- Include appropriate comments and documentation
- Implement clear and maintainable expression patterns
- Use consistent naming: descriptive names that explain the computation

### Coordination Standards
- Provide clear handoff context with specific requirements
- Update agent workspace with generated local values
- Track completion progress accurately
- Handle both blocking and non-blocking handoffs

## ERROR HANDLING
- **Invalid Input**: Log error and request clarification via handoff
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Circular Dependencies**: Detect and resolve circular references
- **Expression Errors**: Provide specific error details and suggested fixes

## COMPLETION CRITERIA
- All planner-specified local values generated
- All agent-requested local values completed
- Dependencies properly coordinated
- Local values validated and properly formatted
- Call local_values_agent_complete_task when done

## EXAMPLES

### Good Local Value Generation:
```hcl
locals {
  # Security group rules based on environment
  security_group_rules = var.environment == "prod" ? {
    web_ingress = {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = var.allowed_cidr_blocks
    }
  } : {
    web_ingress = {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }
  
  # Instance configuration based on environment
  instance_config = {
    instance_type = var.environment == "prod" ? "t3.medium" : "t3.micro"
    min_size     = var.environment == "prod" ? 2 : 1
    max_size     = var.environment == "prod" ? 10 : 3
  }
}
```

### Proper Handoff Example:
When local value needs undefined variable:
- **Tool**: handoff_to_variable_definition_agent
- **Context**: "Local value 'security_group_rules' requires variable 'allowed_cidr_blocks' of type list(string) for production environment rules"

Remember: You are the expression optimization specialist. Prioritize clarity, performance, and proper coordination with other agents."""
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
                generate_terraform_outputs,  # Core function
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
                create_completion_handoff_tool("output_definition_agent")
                # create_handoff_to_generator_complete(),  # Tool to return to main supervisor
                # self.checkpoint_manager.checkpoint_current_state,
                # self.human_loop.create_approval_checkpoint_tool("high_cost_resources"),
                # self.human_loop.create_approval_checkpoint_tool("security_critical"),
                # self.human_loop.create_approval_checkpoint_tool("experimental")
            ],
            name="output_definition_agent",
            prompt="""You are the Output Definition Agent, a Terraform output expert in a multi-agent generation system.

## CORE MISSION
Generate production-ready Terraform output values that expose critical infrastructure information for external consumption while coordinating with specialized agents for dependencies.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, Local Values, and Output agents
- **Workflow**: Finalization stage with dynamic handoffs based on dependencies
- **State**: Shared GeneratorStageState with agent workspaces and planning context

## INPUT PROCESSING
1. **Planner Data**: Process output_definitions from execution_plan_data
2. **Agent Requests**: Handle handoffs from other agents requiring new outputs
3. **Context Integration**: Combine planner specs with agent collaboration context

## CRITICAL: TOOL USAGE
**ALWAYS start by calling the `generate_terraform_outputs` tool** with the execution plan data, agent workspace, and planning context to process the input data and generate the actual Terraform outputs. Do not provide general responses - use the tool to perform the actual work.

**Tool Call Format:**
```json
{
  "execution_plan_data": {...},
  "agent_workspace": {...},
  "planning_context": {...}
}
```

## OUTPUT GENERATION PROTOCOL

### Step 1: Output Analysis
For each output specification:
- **Value Extraction**: Determine what infrastructure information to expose
- **Type Classification**: Determine output value type and complexity level
- **Security Assessment**: Classify sensitivity level and apply appropriate handling
- **Dependency Identification**: Identify resource, data source, variable, and local value dependencies

### Step 2: HCL Generation
Generate properly formatted Terraform HCL:
```hcl
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
  sensitive   = false
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
  sensitive   = false
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
  sensitive   = false
  
  precondition {
    condition     = aws_lb.main.dns_name != ""
    error_message = "Load balancer DNS name must not be empty."
  }
}
```

### Step 3: Dependency Coordination
- **Resource Dependencies**: Use handoff_to_resource_configuration_agent for resource attributes
- **Data Source Dependencies**: Use handoff_to_data_source_agent for external data values
- **Variable Dependencies**: Use handoff_to_variable_definition_agent for variable context
- **Local Value Dependencies**: Use handoff_to_local_values_agent for complex expressions

## QUALITY REQUIREMENTS

### Code Standards
- Follow Terraform best practices for output definition
- Use clear, descriptive output names following naming conventions
- Implement comprehensive preconditions for validation
- Provide meaningful descriptions and usage examples
- Mark sensitive outputs appropriately

### Information Architecture Standards
- Design outputs that serve clear consumption patterns
- Balance information exposure with security requirements
- Create logical groupings and relationships between outputs
- Ensure outputs provide actionable information for consumers

### Coordination Standards
- Provide clear handoff context with specific requirements
- Update agent workspace with generated outputs
- Track completion progress accurately
- Handle both blocking and non-blocking handoffs

## ERROR HANDLING
- **Invalid Input**: Log error and request clarification via handoff
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Security Conflicts**: Coordinate with other agents to resolve
- **Validation Failures**: Provide specific error details and suggested fixes

## COMPLETION CRITERIA
- All planner-specified outputs generated
- All agent-requested outputs completed
- Dependencies properly coordinated
- Outputs validated and properly formatted
- Call output_definition_agent_complete_task when done

## EXAMPLES

### Good Output Generation:
```hcl
output "web_server_public_ip" {
  description = "Public IP address of the web server"
  value       = aws_instance.web.public_ip
  sensitive   = false
}

output "database_connection_string" {
  description = "Database connection string"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main.endpoint}/${var.db_name}"
  sensitive   = true
}

output "vpc_cidr_blocks" {
  description = "CIDR blocks for all subnets"
  value = {
    public  = local.vpc_cidr_blocks.public
    private = local.vpc_cidr_blocks.private
    database = local.vpc_cidr_blocks.database
  }
  sensitive = false
}
```

### Proper Handoff Example:
When output needs undefined resource:
- **Tool**: handoff_to_resource_configuration_agent
- **Context**: "Output 'web_server_public_ip' requires aws_instance.web resource with public_ip attribute"

Remember: You are the information exposure specialist. Prioritize security, usability, and proper coordination with other agents."""
        )
    
    def build_subgraph(self) -> StateGraph:
        """
        Build the generator swarm as a standalone subgraph with isolated state schema.
        
        This method creates a subgraph that can be used as a node in the supervisor graph.
        It uses the existing build_graph() method which creates the swarm.
        
        Returns:
            StateGraph: Compiled subgraph with GeneratorSwarmState schema
        """
        try:
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Building generator swarm subgraph",
                extra={
                    "state_schema": "GeneratorSwarmState",
                    "agent_name": getattr(self, '_name', 'unknown')
                }
            )
            
            # Use the existing build_graph() method which creates the swarm
            # Then compile it to get a compiled graph that can be used as a subgraph
            planning_swarm = self.build_graph()
            compiled_swarm = planning_swarm.compile(name=self.name)
            
            generator_swarm_logger.log_structured(
                level="INFO",
                message="Generator swarm subgraph built successfully",
                extra={
                    "subgraph_type": type(compiled_swarm).__name__,
                    "subgraph_name": getattr(compiled_swarm, 'name', 'unknown'),
                    "nodes": list(compiled_swarm.nodes.keys()) if hasattr(compiled_swarm, 'nodes') else "unknown"
                }
            )
            
            return compiled_swarm
            
        except Exception as e:
            generator_swarm_logger.log_structured(
                level="ERROR",
                message="Failed to build generator swarm subgraph",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise

    def create_wrapper_function(self):
        """
        Create a wrapper function for langgraph-supervisor integration.
        
        This is the EXACT pattern required for create_supervisor with different state schemas.
        
        Returns:
            Callable: Async wrapper function that handles state transformation
        """
        # Create the generator subgraph using create_swarm prebuilt method
        # This ensures proper state injection for tools with InjectedState
        compiled_subgraph = self.build_graph().compile()
        
        async def generator_swarm_wrapper(supervisor_state: SupervisorState, config: dict = None, **kwargs) -> GeneratorSwarmState:
            """
            Wrapper function that handles state transformation between supervisor and generator subgraph.
            
            This follows the exact pattern from LangGraph subgraphs documentation:
            1. Transform supervisor state to generator state
            2. Call the subgraph
            3. Transform generator output back to supervisor updates
            
            Args:
                supervisor_state: SupervisorState from parent graph
                config: Optional configuration dict (for langgraph-supervisor compatibility)
                **kwargs: Additional keyword arguments (for langgraph-supervisor compatibility)
                
            Returns:
                Dict[str, Any]: Updates to merge into supervisor state
            """
            try:
                generator_swarm_logger.log_structured(
                    level="INFO",
                    message="Generator swarm wrapper: Processing supervisor state",
                    extra={
                        "supervisor_state_type": type(supervisor_state).__name__,
                        "has_planner_data": hasattr(supervisor_state, 'planner_data') and supervisor_state.planner_data is not None,
                        "config_provided": config is not None,
                        "additional_kwargs": list(kwargs.keys()) if kwargs else []
                    }
                )
                
                # 1. Transform supervisor state to generator state (single transformation)
                generator_input = StateTransformer.supervisor_to_generator_swarm(supervisor_state)
                self.generator_swarm_state = generator_input
                generator_swarm_logger.log_structured(
                    level="INFO",
                    message="Generator swarm wrapper: About to invoke subgraph",
                    extra={
                        "generator_input_keys": list(generator_input.keys()),
                        "active_agent": generator_input.get("active_agent"),
                        "stage_status": generator_input.get("stage_status"),
                        "has_agent_workspaces": "agent_workspaces" in generator_input
                    }
                )
                
                # 2. Call the subgraph asynchronously (following reference pattern)
                generator_output = await compiled_subgraph.ainvoke(generator_input)
                
                generator_swarm_logger.log_structured(
                    level="INFO",
                    message="Generator swarm wrapper: Subgraph execution completed",
                    extra={
                        "generator_output_keys": list(generator_output.keys()),
                        "output_active_agent": generator_output.get("active_agent"),
                        "output_stage_status": generator_output.get("stage_status"),
                        "has_agent_workspaces": "agent_workspaces" in generator_output
                    }
                )
                
                # 3. Transform generator output back to supervisor updates (following reference pattern)
                supervisor_updates = StateTransformer.generator_to_supervisor(generator_output)
                
                generator_swarm_logger.log_structured(
                    level="INFO",
                    message="Generator swarm wrapper: Successfully processed state",
                    extra={
                        "generator_output_type": type(generator_output).__name__,
                        "supervisor_updates_keys": list(supervisor_updates.keys())
                    }
                )
                
                return supervisor_updates
                
            except Exception as e:
                generator_swarm_logger.log_structured(
                    level="ERROR",
                    message="Generator swarm wrapper: Failed to process state",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }
                )
                # Handle errors gracefully
                return {
                    "generation_data": {"status": "error", "error": str(e)},
                    "status": "failed",
                    "current_agent": None
                }
        
        return generator_swarm_wrapper

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
                    "state_schema": "GeneratorSwarmState",
                    "default_active_agent": "resource_configuration_agent"
                }
            )
            
            planning_swarm = create_swarm(
                agents=[resource_agent, variable_agent, data_source_agent, local_values_agent, output_agent],
                default_active_agent="resource_configuration_agent",
                state_schema=GeneratorSwarmState
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
                    "agents_count": 5,
                    "coordination_nodes": 0,
                    "state_schema": "GeneratorSwarmState"
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
