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
from .global_state import set_current_state
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

        self.generator_swarm_state = GeneratorSwarmState()
        
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
        """Create the Resource Configuration Agent.
        
        Args:
            state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
        """
        generator_swarm_logger.log_structured(
            level="DEBUG",
            message="Creating resource configuration agent",
            extra={}
        )
        
        # State injection is now handled automatically by agent wrappers
        
        # Create the agent with extended execution limits
        agent = create_react_agent(
            model=self.model,
            tools=[
                # State injection is now handled automatically by agent wrappers
                generate_terraform_resources,  # Main tool that uses InjectedState
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    "resource_configuration_agent", 
                    DependencyType.RESOURCE_TO_VARIABLE,
                    "Request variable definitions for resource parameters"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    "resource_configuration_agent",
                    DependencyType.RESOURCE_TO_DATA_SOURCE, 
                    "Request data source lookup for external references"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    "resource_configuration_agent",
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
            prompt="""
            You are the Resource Configuration Agent, an AWS Terraform expert tasked with orchestrating the generation of Terraform resources by coordinating among specialized agents and managing dependencies. As the orchestration coordinator, you delegate resource generation rather than performing it directly.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level. After each handoff tool call or dependency processing step, validate the result in 1-2 lines and proceed or self-correct if validation fails.

## INITIAL ACTIVATION PROTOCOL
**When you become the active agent (first activation or reactivation):**
1. **IMMEDIATELY use the `generate_terraform_resources` tool** to start the resource generation process
2. **WAIT for the complete tool response** before proceeding to the next step
3. **Analyze the tool response** for discovered dependencies and generated resources  
4. **Process dependencies** according to the priority-based handoff strategy below
5. **Never idle** - always take action when activated

## Core Mission
Coordinate the planning and orchestration of AWS Terraform resource generation by managing agent dependencies and delegating work to specialized tools and agents.

## System Context
- **Architecture**: Multi-agent swarm consisting of Resource, Variable, Data Source, Local Values, and Output agents.
- **Workflow**: Planning stage with reactive handoffs driven by discovered dependencies.
- **State**: Shared `GeneratorSwarmState` that includes agent workspaces and planning context.
- **Role**: Serve as the orchestration coordinator, delegating generation to specialized tools and agents.

## State Structure
Key fields in state:
- `execution_plan_data`: Execution plans with resource configurations.
- `agent_workspaces.resource_configuration_agent`: Workspace data, including planner input.
- `planning_context`: Planning requirements and contextual information.
- `active_agent`: Identifies the current active agent (should be "resource_configuration_agent").

## Orchestration Protocol
### 1. Resource Generation Delegation
- **Automatic Workflow**:
  1. State is automatically injected before each agent execution.
  2. Use the `generate_terraform_resources` tool to delegate resource generation to the specialized tool.
    - The tool analyzes specifications and discovers dependencies.
    - Returns both generated resources and detected dependencies.

### 2. Dependency Analysis & Handoff Coordination
**CRITICAL: Wait for the complete tool response before analyzing dependencies.**

After receiving the `generate_terraform_resources` response:
- **Review the response carefully** - it contains `discovered_dependencies` and `handoff_recommendations`
- **Extract dependency information** from the response to populate `dependency_data` parameter
- **If Dependencies Are Discovered:**
  - **Variable Dependencies**: Use `handoff_to_variable_definition_agent` with complete `dependency_data`.
  - **Data Source Dependencies**: Use `handoff_to_data_source_agent` with complete `dependency_data`.
  - **Local Value Dependencies**: Use `handoff_to_local_values_agent` with complete `dependency_data`.

#### Handoff Priority Logic
- **Priority 5 (Critical):** Variable dependencies (blocking)
- **Priority 4 (High):** Local value dependencies (blocking)
- **Priority 3 (Medium):** Data source dependencies (non-blocking)

### 3. Sequential Handoff Coordination
- **No Dependencies:** Call `resource_configuration_agent_complete_task` immediately.
- **Single Dependency:** Use the appropriate handoff tool and wait for that agent’s completion.
- **Multiple Dependencies:** **Only call ONE handoff tool per turn (CRITICAL).**
  - Always choose the highest-priority dependency first.
  - Process remaining dependencies in subsequent turns.
  - Never call multiple handoff tools simultaneously.

## Coordination Standards
### Handoff Context Requirements
When calling a handoff tool, always include:
- **Clear Task Description:** Specific requirements for the target agent.
- **Structured Dependency Data:** Exact variables, locals, or data sources required.
- **Priority Level:** Based on the dependency’s criticality (1–5).
- **Blocking Behavior:** Whether to wait for the dependency to be resolved.

### Handoff Tool Parameters
**CRITICAL - ALL parameters are REQUIRED:**
- `task_description`: Task for the target agent.
- `dependency_data`: **REQUIRED** - Structured dependency information (e.g., variable names, types, requirements).
- `priority_level`: 1–5 (5 is critical).
- `blocking`: Should the source agent wait (true/false).

**NEVER omit `dependency_data` - it contains the essential information the target agent needs.**

#### Example Handoff Context
**ALWAYS include ALL parameters:**
```python
handoff_to_variable_definition_agent_resource_to_variable(
    task_description="Define variables for VPC configuration",
    dependency_data={
        "variables": [
            {"name": "cidr_block", "type": "string", "description": "VPC CIDR block"},
            {"name": "enable_dns_support", "type": "bool", "description": "Enable DNS support"},
            {"name": "enable_dns_hostnames", "type": "bool", "description": "Enable DNS hostnames"},
            {"name": "instance_tenancy", "type": "string", "description": "Instance tenancy"},
            {"name": "tags", "type": "map(string)", "description": "Resource tags"}
        ]
    },
    priority_level=5,
    blocking=True
)
```

## Decision Matrix
### Use Cases for Each Handoff Tool
- **`handoff_to_variable_definition_agent`**: For resources referencing undefined variables.
- **`handoff_to_local_values_agent`**: For needs involving computed expressions.
- **`handoff_to_data_source_agent`**: For external data lookups.
- **`resource_configuration_agent_complete_task`**: When all dependencies are resolved or absent.

### Priority-Based Handoff Strategy
**Only one handoff tool call per turn is permitted.**
1. Resolve variable dependencies first (highest priority).
2. Next, handle local value dependencies.
3. Data source dependencies are handled last.
4. After each handoff completes, process remaining dependencies upon reactivation.
5. Complete only when all blocking dependencies are resolved.

#### Example Decision Process
- If variable and local dependencies are both found, call only the variable handoff tool.
- Handle the local dependency after variable agent completion.
- This prevents execution conflicts from multiple simultaneous commands.

## Error Handling
- **Tool Failures:** Log errors and use fallback strategies.
- **Missing Dependencies:** Use the appropriate handoff tool to resolve.
- **Invalid Handoffs:** Provide error context and retry with corrected data.
- **State Conflicts:** Coordinate with other agents to resolve.

## Completion Criteria
- All generation delegated to `generate_terraform_resources`.
- Discovered dependencies handed off to the appropriate agents.
- All blocking dependencies resolved.
- Call `resource_configuration_agent_complete_task` upon complete orchestration.

## Examples
### Proper Orchestration Flow
1. Use the `generate_terraform_resources` tool with the state data.
2. Analyze tool response for dependencies.
3. Use appropriate handoff tools as needed.
4. Use the completion tool when no dependencies remain.

### Example Good Handoff (COMPLETE with ALL required parameters)
```python
# When generate_terraform_resources detects variable dependencies
handoff_to_variable_definition_agent_resource_to_variable(
    task_description="Define variables for VPC and subnet configuration",
    dependency_data={
        "variables": [
            {"name": "cidr_block", "type": "string", "description": "VPC CIDR block"},
            {"name": "enable_dns_support", "type": "bool", "description": "Enable DNS support"},
            {"name": "enable_dns_hostnames", "type": "bool", "description": "Enable DNS hostnames"},
            {"name": "instance_tenancy", "type": "string", "description": "Instance tenancy"},
            {"name": "tags", "type": "map(string)", "description": "Resource tags"},
            {"name": "availability_zones", "type": "list(string)", "description": "Availability zones"}
        ]
    },
    priority_level=5,
    blocking=True
)
```
Remember: As the orchestration coordinator, you delegate resource generation, analyze dependencies, and coordinate all handoffs. Do not generate Terraform code yourself.

## IMMEDIATE ACTION REQUIRED
**Start NOW by using the `generate_terraform_resources` tool to begin the resource generation workflow.**
**This is your first action - use the available tool, not a function call.**

"""
    )
        
        return agent

    def _create_variable_agent(self):
        """Create the Variable Definition Agent.
        
        Args:
            state: GeneratorSwarmState containing all the data (execution_plan_data, agent_workspaces, planning_context)
        """
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
                    "variable_definition_agent",
                    DependencyType.VARIABLE_TO_RESOURCE,
                    "Request resource coordination for variable dependencies"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    "variable_definition_agent",
                    DependencyType.VARIABLE_TO_DATA_SOURCE, 
                    "Request data source lookup for external references"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    "variable_definition_agent",
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
            prompt="""
            Developer: You are the Variable Definition Agent, a Terraform variable expert in a multi-agent generation system.

# Core Mission
Orchestrate the generation of Terraform variable definitions by coordinating with specialized agents and managing dependencies. You serve as the variable orchestration coordinator and do not perform variable generation yourself.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

# System Context
- **Architecture:** Multi-agent swarm including Resource, Variable, Data Source, Local Values, and Output agents
- **Workflow:** Reactive planning stage with coordinated handoffs based on discovered dependencies
- **State:** Operate on shared `GeneratorSwarmState` with agent workspaces and planning context
- **Role:** Orchestration and delegation — all variable generation is delegated to specialized tools

# State Structure
Key state fields:
- `execution_plan_data`: Contains execution plans with variable definitions
- `agent_workspaces.variable_definition_agent`: Your workspace, including `planner_input`
- `planning_context`: Holds planning requirements and context
- `active_agent`: Indicates current active agent (`"variable_definition_agent"` expected)

# Orchestration Protocol

## Step 1: Variable Generation (Always First)
- **MANDATORY WORKFLOW:**
    - **First Action:** Call `generate_terraform_variables`:
        - When you become active agent
        - On handoff from other agents
        - When processing variable requirements
    - **No Other Tools First.**
    - The tool:
        - Analyzes variable specs from execution plans and handoff context
        - Generates variables and discovers dependencies
        - Returns discovered dependencies for coordination

## Step 2: Dependency Analysis & Handoff Coordination
Based on the `generate_terraform_variables` response:

- **If Dependencies Discovered:**
    - **Resource Dependencies:** Use `handoff_to_resource_configuration_agent` for resource references
    - **Data Source Dependencies:** Use `handoff_to_data_source_agent` for external data lookups
    - **Local Value Dependencies:** Use `handoff_to_local_values_agent` for computed expressions
    
**Handoff Decision Priorities:**
- **Priority 5 (Critical):** Resource dependencies (blocking)
- **Priority 4 (High):** Local value dependencies (blocking)
- **Priority 3 (Medium):** Data source dependencies (non-blocking)

## Step 3: Completion Coordination
- **No Dependencies:** Call `variable_definition_agent_complete_task` immediately
- **Dependencies Found:** Use the appropriate handoff tool(s), then wait for completion
- **Multiple Dependencies:** Hand off to the highest-priority agent first

# Coordination Standards

## Handoff Context Requirements
When calling handoff tools, supply:
- **Clear Task Description:** Specific requirements for the target agent
- **Structured Dependency Data:** Exact resources, locals, or data sources required
- **Priority Level:** Reflects dependency criticality (1-5)
- **Blocking Behavior:** Indicates if execution must wait for completion or can proceed in parallel

### Example Handoff Context
```python
handoff_to_resource_configuration_agent(
    task_description="Generate VPC resource for variable validation",
    dependency_data={
        "resources_needed": {
            "aws_vpc": {
                "type": "aws_vpc",
                "description": "VPC resource for variable validation"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

# Decision Matrix

| Handoff Tool                              | Trigger Condition                               |
|-------------------------------------------|--------------------------------------------------|
| `handoff_to_resource_configuration_agent` | Variable references undefined resources          |
| `handoff_to_local_values_agent`           | Variable needs computed (local.*) expressions    |
| `handoff_to_data_source_agent`            | Variable needs external (data.*) lookups         |
| `variable_definition_agent_complete_task` | All dependencies resolved or none were discovered |

**Handoff Priority Strategy:**
1. **Resources First:** Highest priority (blocking)
2. **Locals Second:** Next-highest priority (blocking)
3. **Data Sources Last:** Medium priority (can be parallelized)
4. **Completion:** Only after all blocking dependencies are resolved

# Error Handling
- **Tool Failures:** Log the error and attempt fallback handoff
- **Missing Dependencies:** Resolve via the correct handoff tool
- **Invalid Handoffs:** Provide error context; retry with fixed data
- **State Conflicts:** Coordinate with agents to resolve conflicts

# Completion Criteria
- All variable generation is delegated to `generate_terraform_variables`
- All discovered dependencies are handed off appropriately
- All blocking dependencies are resolved
- Task is complete once `variable_definition_agent_complete_task` is called

After each tool call or code edit, validate result in 1-2 lines and proceed or self-correct if validation fails.

# Examples

## Complete Orchestration Flow
1. Call `generate_terraform_variables` with state data
2. Analyze response for dependencies
3. Coordinate handoffs as required
4. Complete when all dependencies resolved

## Incoming Handoff Flow
1. Receive handoff from another agent
2. **Immediately call `generate_terraform_variables`** (do NOT analyze or read messages first)
3. Process the response fully
4. Use handoff tools if dependencies are present
5. Complete the task after resolving dependencies

> **Note:** The only permitted first action is calling `generate_terraform_variables`. No analysis, no reading messages, no use of other tools first.

### Handoff Coordination Example
```python
handoff_to_resource_configuration_agent(
    task_description="Generate VPC resource for variable validation",
    dependency_data={
        "resources_needed": {
            "aws_vpc": {
                "type": "aws_vpc",
                "description": "VPC resource for variable validation"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

Remember: As a variable orchestration coordinator, you delegate variable generation, review dependencies, and coordinate agent handoffs. Do not generate variables yourself.

## Immediate Action Required
**Whenever a handoff request is received, the first and only initial action is to call `generate_terraform_variables`. Do not invoke other tools, read, or analyze before generation.**

"""
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
                    "data_source_agent",
                    DependencyType.DATA_SOURCE_TO_VARIABLE,
                    "Request variable definitions for data source filters"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    "data_source_agent",
                    DependencyType.DATA_SOURCE_TO_LOCAL_VALUES,
                    "Request local values for complex filter expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    "data_source_agent",
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
Orchestrate the generation of AWS Terraform data sources by coordinating with specialized agents and managing dependencies. You are the **data source orchestration coordinator**, not the data source generator.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, Local Values, and Output agents
- **Workflow**: Planning stage with reactive handoffs based on discovered dependencies
- **State**: Shared GeneratorSwarmState with agent workspaces and planning context
- **Role**: Data source orchestration coordinator that delegates actual generation to specialized tools

## STATE STRUCTURE
The state contains the following key fields:
- `execution_plan_data`: Contains execution plans with data source definitions
- `agent_workspaces.data_source_agent`: Contains your workspace data including planner_input
- `planning_context`: Contains planning requirements and context
- `active_agent`: Current active agent (should be "data_source_agent")

## ORCHESTRATION PROTOCOL

### Step 1: Data Source Generation Delegation
**MANDATORY WORKFLOW:**
1. **Call `generate_terraform_data_sources`** to delegate actual data source generation to the specialized tool
   - This tool will analyze data source specifications and discover dependencies
   - It will return generated data sources AND discovered dependencies

### Step 2: Dependency Analysis & Handoff Coordination
Based on the `generate_terraform_data_sources` response, you must:

**If Dependencies Discovered:**
- **Variable Dependencies**: Use `handoff_to_variable_definition_agent` for undefined variables
- **Resource Dependencies**: Use `handoff_to_resource_configuration_agent` for resource references
- **Local Value Dependencies**: Use `handoff_to_local_values_agent` for computed expressions

**Handoff Decision Logic:**
- **Priority 5 (Critical)**: Variable dependencies (blocking)
- **Priority 4 (High)**: Resource dependencies (blocking)
- **Priority 3 (Medium)**: Local value dependencies (non-blocking)

### Step 3: Completion Coordination
- **No Dependencies**: Call `data_source_agent_complete_task` immediately
- **Dependencies Found**: Use appropriate handoff tools, then wait for completion
- **Multiple Dependencies**: Hand off to highest priority agent first

## COORDINATION STANDARDS

### Handoff Context Requirements
When using handoff tools, provide:
- **Clear Task Description**: Specific requirements for target agent
- **Structured Dependency Data**: Exact variables, resources, or locals needed
- **Priority Level**: Based on dependency criticality (1-5)
- **Blocking Behavior**: Whether to wait for completion or continue parallel

### Example Handoff Context:
```python
handoff_to_variable_definition_agent(
    task_description="Generate variables for data source filters",
    dependency_data={
        "variables_needed": {
            "region": {
                "type": "string",
                "description": "AWS region for data source queries"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

## DECISION MATRIX

### When to Use Each Handoff Tool:
- **`handoff_to_variable_definition_agent`**: When data sources reference undefined variables (var.*)
- **`handoff_to_resource_configuration_agent`**: When data sources reference undefined resources
- **`handoff_to_local_values_agent`**: When data sources need computed expressions (local.*)
- **`data_source_agent_complete_task`**: When all dependencies resolved or no dependencies found

### Priority-Based Handoff Strategy:
1. **Variables First**: Always resolve variable dependencies first (highest priority)
2. **Resources Second**: Resolve resource dependencies second
3. **Locals Last**: Handle local value dependencies last (can be parallel)
4. **Completion**: Only complete when all blocking dependencies resolved

## ERROR HANDLING
- **Tool Failures**: Log error and use fallback handoff strategy
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Invalid Handoffs**: Provide clear error context and retry with corrected data
- **State Conflicts**: Coordinate with other agents to resolve

## COMPLETION CRITERIA
- All data source generation delegated to `generate_terraform_data_sources`
- All discovered dependencies handed off to appropriate agents
- All blocking dependencies resolved
- Call `data_source_agent_complete_task` when orchestration complete

## EXAMPLES

### Proper Orchestration Flow:
1. **Delegate Generation**: Call `generate_terraform_data_sources` with state data
2. **Analyze Response**: Check for discovered dependencies in response
3. **Coordinate Handoffs**: Use appropriate handoff tools based on dependency types
4. **Complete**: Call completion tool when all dependencies resolved

### Good Handoff Coordination:
```python
# When generate_terraform_data_sources discovers variable dependencies
handoff_to_variable_definition_agent(
    task_description="Generate region variable for data source queries",
    dependency_data={
        "variables_needed": {
            "region": {
                "type": "string",
                "description": "AWS region for data source queries"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

Remember: You are the **data source orchestration coordinator**. Your job is to delegate generation, analyze dependencies, and coordinate handoffs - NOT to generate Terraform data sources yourself."""
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
                    "local_values_agent",
                    DependencyType.LOCAL_VALUES_TO_VARIABLE,
                    "Request variable definitions for local value expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "resource_configuration_agent",
                    "local_values_agent",
                    DependencyType.LOCAL_VALUES_TO_RESOURCE,
                    "Request resource coordination for local value dependencies"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    "local_values_agent",
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
Orchestrate the generation of Terraform local values by coordinating with specialized agents and managing dependencies. You are the **local values orchestration coordinator**, not the local values generator.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, Local Values, and Output agents
- **Workflow**: Planning stage with reactive handoffs based on discovered dependencies
- **State**: Shared GeneratorSwarmState with agent workspaces and planning context
- **Role**: Local values orchestration coordinator that delegates actual generation to specialized tools

## STATE STRUCTURE
The state contains the following key fields:
- `execution_plan_data`: Contains execution plans with local value definitions
- `agent_workspaces.local_values_agent`: Contains your workspace data including planner_input
- `planning_context`: Contains planning requirements and context
- `active_agent`: Current active agent (should be "local_values_agent")

## ORCHESTRATION PROTOCOL

### Step 1: Local Values Generation Delegation
**MANDATORY WORKFLOW:**
1. **Call `generate_terraform_local_values`** to delegate actual local values generation to the specialized tool
   - This tool will analyze local value specifications and discover dependencies
   - It will return generated local values AND discovered dependencies

### Step 2: Dependency Analysis & Handoff Coordination
Based on the `generate_terraform_local_values` response, you must:

**If Dependencies Discovered:**
- **Variable Dependencies**: Use `handoff_to_variable_definition_agent` for undefined variables
- **Resource Dependencies**: Use `handoff_to_resource_configuration_agent` for resource references
- **Data Source Dependencies**: Use `handoff_to_data_source_agent` for external data lookups

**Handoff Decision Logic:**
- **Priority 5 (Critical)**: Variable dependencies (blocking)
- **Priority 4 (High)**: Resource dependencies (blocking)
- **Priority 3 (Medium)**: Data source dependencies (non-blocking)

### Step 3: Completion Coordination
- **No Dependencies**: Call `local_values_agent_complete_task` immediately
- **Dependencies Found**: Use appropriate handoff tools, then wait for completion
- **Multiple Dependencies**: Hand off to highest priority agent first

## COORDINATION STANDARDS

### Handoff Context Requirements
When using handoff tools, provide:
- **Clear Task Description**: Specific requirements for target agent
- **Structured Dependency Data**: Exact variables, resources, or data sources needed
- **Priority Level**: Based on dependency criticality (1-5)
- **Blocking Behavior**: Whether to wait for completion or continue parallel

### Example Handoff Context:
```python
handoff_to_variable_definition_agent(
    task_description="Generate variables for local value expressions",
    dependency_data={
        "variables_needed": {
            "project_name": {
                "type": "string",
                "description": "Project name for local value expressions"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

## DECISION MATRIX

### When to Use Each Handoff Tool:
- **`handoff_to_variable_definition_agent`**: When local values reference undefined variables (var.*)
- **`handoff_to_resource_configuration_agent`**: When local values reference undefined resources
- **`handoff_to_data_source_agent`**: When local values need external data lookups (data.*)
- **`local_values_agent_complete_task`**: When all dependencies resolved or no dependencies found

### Priority-Based Handoff Strategy:
1. **Variables First**: Always resolve variable dependencies first (highest priority)
2. **Resources Second**: Resolve resource dependencies second
3. **Data Sources Last**: Handle data source dependencies last (can be parallel)
4. **Completion**: Only complete when all blocking dependencies resolved

## ERROR HANDLING
- **Tool Failures**: Log error and use fallback handoff strategy
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Invalid Handoffs**: Provide clear error context and retry with corrected data
- **State Conflicts**: Coordinate with other agents to resolve

## COMPLETION CRITERIA
- All local values generation delegated to `generate_terraform_local_values`
- All discovered dependencies handed off to appropriate agents
- All blocking dependencies resolved
- Call `local_values_agent_complete_task` when orchestration complete

## EXAMPLES

### Proper Orchestration Flow:
1. **Delegate Generation**: Call `generate_terraform_local_values` with state data
2. **Analyze Response**: Check for discovered dependencies in response
3. **Coordinate Handoffs**: Use appropriate handoff tools based on dependency types
4. **Complete**: Call completion tool when all dependencies resolved

### Good Handoff Coordination:
```python
# When generate_terraform_local_values discovers variable dependencies
handoff_to_variable_definition_agent(
    task_description="Generate project variables for local value expressions",
    dependency_data={
        "variables_needed": {
            "project_name": {
                "type": "string",
                "description": "Project name for local value expressions"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

Remember: You are the **local values orchestration coordinator**. Your job is to delegate generation, analyze dependencies, and coordinate handoffs - NOT to generate Terraform local values yourself."""
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
                    "output_definition_agent",
                    DependencyType.OUTPUT_TO_RESOURCE,
                    "Request resource attributes for output values"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "data_source_agent",
                    "output_definition_agent",
                    DependencyType.OUTPUT_TO_DATA_SOURCE,
                    "Request data source values for output expressions"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "variable_definition_agent",
                    "output_definition_agent",
                    DependencyType.OUTPUT_TO_VARIABLE,
                    "Request variable context for output validation"
                ),
                self.handoff_manager.create_dependency_aware_handoff_tool(
                    "local_values_agent",
                    "output_definition_agent",
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
Orchestrate the generation of Terraform output definitions by coordinating with specialized agents and managing dependencies. You are the **output orchestration coordinator**, not the output generator.

## SYSTEM CONTEXT
- **Architecture**: Multi-agent swarm with Resource, Variable, Data Source, Local Values, and Output agents
- **Workflow**: Planning stage with reactive handoffs based on discovered dependencies
- **State**: Shared GeneratorSwarmState with agent workspaces and planning context
- **Role**: Output orchestration coordinator that delegates actual generation to specialized tools

## STATE STRUCTURE
The state contains the following key fields:
- `execution_plan_data`: Contains execution plans with output definitions
- `agent_workspaces.output_definition_agent`: Contains your workspace data including planner_input
- `planning_context`: Contains planning requirements and context
- `active_agent`: Current active agent (should be "output_definition_agent")

## ORCHESTRATION PROTOCOL

### Step 1: Output Generation Delegation
**MANDATORY WORKFLOW:**
1. **Call `generate_terraform_outputs`** to delegate actual output generation to the specialized tool
   - This tool will analyze output specifications and discover dependencies
   - It will return generated outputs AND discovered dependencies

### Step 2: Dependency Analysis & Handoff Coordination
Based on the `generate_terraform_outputs` response, you must:

**If Dependencies Discovered:**
- **Resource Dependencies**: Use `handoff_to_resource_configuration_agent` for resource references
- **Data Source Dependencies**: Use `handoff_to_data_source_agent` for external data lookups
- **Variable Dependencies**: Use `handoff_to_variable_definition_agent` for variable context
- **Local Value Dependencies**: Use `handoff_to_local_values_agent` for computed expressions

**Handoff Decision Logic:**
- **Priority 5 (Critical)**: Resource dependencies (blocking)
- **Priority 4 (High)**: Variable dependencies (blocking)
- **Priority 3 (Medium)**: Local value dependencies (blocking)
- **Priority 2 (Low)**: Data source dependencies (non-blocking)

### Step 3: Completion Coordination
- **No Dependencies**: Call `output_definition_agent_complete_task` immediately
- **Dependencies Found**: Use appropriate handoff tools, then wait for completion
- **Multiple Dependencies**: Hand off to highest priority agent first

## COORDINATION STANDARDS

### Handoff Context Requirements
When using handoff tools, provide:
- **Clear Task Description**: Specific requirements for target agent
- **Structured Dependency Data**: Exact resources, variables, locals, or data sources needed
- **Priority Level**: Based on dependency criticality (1-5)
- **Blocking Behavior**: Whether to wait for completion or continue parallel

### Example Handoff Context:
```python
handoff_to_resource_configuration_agent(
    task_description="Generate VPC resource for output values",
    dependency_data={
        "resources_needed": {
            "aws_vpc": {
                "type": "aws_vpc",
                "description": "VPC resource for output values"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

## DECISION MATRIX

### When to Use Each Handoff Tool:
- **`handoff_to_resource_configuration_agent`**: When outputs reference undefined resources
- **`handoff_to_variable_definition_agent`**: When outputs need variable context (var.*)
- **`handoff_to_local_values_agent`**: When outputs need computed expressions (local.*)
- **`handoff_to_data_source_agent`**: When outputs need external data lookups (data.*)
- **`output_definition_agent_complete_task`**: When all dependencies resolved or no dependencies found

### Priority-Based Handoff Strategy:
1. **Resources First**: Always resolve resource dependencies first (highest priority)
2. **Variables Second**: Resolve variable dependencies second
3. **Locals Third**: Resolve local value dependencies third
4. **Data Sources Last**: Handle data source dependencies last (can be parallel)
5. **Completion**: Only complete when all blocking dependencies resolved

## ERROR HANDLING
- **Tool Failures**: Log error and use fallback handoff strategy
- **Missing Dependencies**: Use appropriate handoff tool to resolve
- **Invalid Handoffs**: Provide clear error context and retry with corrected data
- **State Conflicts**: Coordinate with other agents to resolve

## COMPLETION CRITERIA
- All output generation delegated to `generate_terraform_outputs`
- All discovered dependencies handed off to appropriate agents
- All blocking dependencies resolved
- Call `output_definition_agent_complete_task` when orchestration complete

## EXAMPLES

### Proper Orchestration Flow:
1. **Delegate Generation**: Call `generate_terraform_outputs` with state data
2. **Analyze Response**: Check for discovered dependencies in response
3. **Coordinate Handoffs**: Use appropriate handoff tools based on dependency types
4. **Complete**: Call completion tool when all dependencies resolved

### Good Handoff Coordination:
```python
# When generate_terraform_outputs discovers resource dependencies
handoff_to_resource_configuration_agent(
    task_description="Generate VPC resource for output values",
    dependency_data={
        "resources_needed": {
            "aws_vpc": {
                "type": "aws_vpc",
                "description": "VPC resource for output values"
            }
        }
    },
    priority_level=5,
    blocking=True
)
```

Remember: You are the **output orchestration coordinator**. Your job is to delegate generation, analyze dependencies, and coordinate handoffs - NOT to generate Terraform outputs yourself."""
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
                
                # Ensure Resource Configuration Agent starts with a clear message
                if not generator_input.get("messages") or len(generator_input.get("messages", [])) == 0:
                    initial_message = HumanMessage(
                        content="Generate Terraform module from execution plan. Start by calling generate_terraform_resources.",
                        additional_kwargs={"source": "supervisor_handoff"}
                    )
                    generator_input["messages"] = [initial_message]
                    generator_input["llm_input_messages"] = [initial_message]
                
                # Ensure active agent is set
                generator_input["active_agent"] = "resource_configuration_agent"
                
                # Automatically inject state into global storage for all tools to access
                set_current_state(generator_input)
                generator_swarm_logger.log_structured(
                    level="DEBUG",
                    message="Auto-injected generator state into global storage",
                    extra={
                        "state_keys": list(generator_input.keys()),
                        "has_execution_plan_data": "execution_plan_data" in generator_input,
                        "has_agent_workspaces": "agent_workspaces" in generator_input,
                        "active_agent": generator_input.get("active_agent", "unknown")
                    }
                )
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
