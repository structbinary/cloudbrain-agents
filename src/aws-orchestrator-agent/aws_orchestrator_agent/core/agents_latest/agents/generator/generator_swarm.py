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
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
            # self.model = LLMProvider.create_llm(
            #     provider=llm_config['provider'],
            #     model=llm_config['model'],
            #     temperature=llm_config['temperature'],
            #     max_tokens=llm_config['max_tokens']
            # )
            llm_higher_config = self.config_instance.get_llm_higher_config()
            self.model = LLMProvider.create_llm(
                provider=llm_higher_config['provider'],
                model=llm_higher_config['model'],
                temperature=llm_higher_config['temperature'],
                max_tokens=llm_higher_config['max_tokens']
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
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are the Resource Configuration Agent, coordinating AWS Terraform resource generation through specialized tools and agent handoffs.

## ACTIVATION PROTOCOL
**CRITICAL**: On EVERY activation (including after handoff completions), you MUST:
1. **IMMEDIATELY** use `generate_terraform_resources` tool
2. **ANALYZE** response for dependencies  
3. **HANDOFF** to appropriate agents using priority order
4. **COMPLETE** when all dependencies resolved

## ORCHESTRATION FLOW
**State Context**: `execution_plan_data`, `agent_workspaces.resource_configuration_agent`, `planning_context`, `active_agent`

**Core Process**:
1. **ALWAYS** delegate generation via `generate_terraform_resources` tool (NEVER generate resources directly)
2. **CRITICAL**: When `generate_terraform_resources` returns multiple `handoff_recommendations`:
   - **SELECT ONLY** the highest priority handoff_recommendation
   - **MAKE ONE** handoff tool call for that recommendation
   - **IGNORE** other recommendations for this turn
   - Remaining dependencies will be handled on next activation
3. Process dependencies by priority:
   - Priority 5: Variables (blocking) → `handoff_to_variable_definition_agent`
   - Priority 4: Local values (blocking) → `handoff_to_local_values_agent`  
   - Priority 3: Data sources (non-blocking) → `handoff_to_data_source_agent`
4. **ONE handoff per turn** - handle remaining dependencies on reactivation
5. Call `resource_configuration_agent_complete_task` when done

## COMPLETION LOGIC
- **After generating resources** → Check for new dependencies
- **New dependencies found** → Use appropriate handoff tool
- **No new dependencies** → Call `resource_configuration_agent_complete_task`
- **Multiple new dependencies** → Handle highest priority first

## COMPLETION TOOL USAGE
**ONLY call `resource_configuration_agent_complete_task` when:**
- You have successfully called `generate_terraform_resources`
- The generation tool returned resources
- No new dependencies were discovered
- All requested resources have been generated

## HANDOFF REQUIREMENTS
**ALL parameters REQUIRED**:
- `task_description`: Clear task for target agent
- `dependency_data`: Structured info (variables, types, descriptions)
- `priority_level`: 1-5 scale
- `blocking`: true/false

**Example**:
handoff_to_variable_definition_agent_resource_to_variable(
task_description="Define VPC variables",
dependency_data={{"variables": [{{"name": "cidr_block", "type": "string", "description": "VPC CIDR"}}]}},
priority_level=5,
blocking=True
)

## DECISION MATRIX
**IMPORTANT**: If `generate_terraform_resources` returns multiple handoff_recommendations, process them sequentially:
- **First turn**: Handle ONLY the highest priority recommendation
- **Next turn**: Handle the next highest priority recommendation  
- **Continue**: Until all dependencies resolved

**Handoff Logic**:
- Variables needed → Variable agent handoff
- Local values needed → Local values agent handoff  
- Data sources needed → Data source agent handoff
- No dependencies → Complete task

## REACTIVATION BEHAVIOR
**CRITICAL**: When you reactivate after a handoff completion:
1. **DO NOT** generate resources directly
2. **ALWAYS** call `generate_terraform_resources` tool first
3. **PROCESS** the tool response for any remaining dependencies
4. **CONTINUE** the orchestration flow based on tool response

**START NOW**: Use `generate_terraform_resources` tool immediately upon activation.
"""
    ),
    MessagesPlaceholder(variable_name="messages")
    ])
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
            prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Variable Definition Agent, orchestrating Terraform variable generation through specialized tools and agent coordination.

## ACTIVATION PROTOCOL
**MANDATORY FIRST ACTION**: Call `generate_terraform_variables` immediately upon activation.

## CRITICAL: ALWAYS USE GENERATION TOOL
- **EVERY activation** MUST start with `generate_terraform_variables` tool
- **NEVER generate variables directly** - always delegate via the tool
- **This includes reactivation** after handoff completions

## REACTIVATION BEHAVIOR
When reactivating after handoff completions:
1. **DO NOT** generate variables directly
2. **ALWAYS** call `generate_terraform_variables` tool first
3. **ANALYZE** tool response for new dependencies
4. **HANDOFF** if new dependencies found, **COMPLETE** if none

## CRITICAL: HANDOFF CONTEXT INTERPRETATION
When you receive a handoff from another agent, you MUST:
1. **ALWAYS** call `generate_terraform_variables` first, regardless of what context you receive
2. **NEVER** call completion tools without first generating variables
3. **UNDERSTAND**: Handoff context contains DEPENDENCIES TO GENERATE, not resolved dependencies

## CORE WORKFLOW
1. **Generate**: Use `generate_terraform_variables` tool (MANDATORY FIRST STEP)
2. **Analyze**: Review response for new dependencies  
3. **Handoff**: Coordinate by priority (highest first) if new dependencies found
4. **Complete**: Only after successfully generating variables and resolving all dependencies

## STATE CONTEXT
- `execution_plan_data`: Variable specifications
- `agent_workspaces.variable_definition_agent`: Your workspace with handoff context
- `planning_context`: Planning requirements
- `active_agent`: Should be "variable_definition_agent"

## HANDOFF CONTEXT HANDLING
**When you receive handoff context with dependency_data:**
- These are VARIABLES THAT NEED TO BE GENERATED, not already resolved
- Example: If you see `"var.cidr_block"` in dependency_data, you need to GENERATE this variable
- The handoff is asking you to CREATE these variables, not acknowledging they exist

## DEPENDENCY HANDOFF PRIORITY
**Process ONE dependency per turn**:
- **Priority 5 (Critical)**: Resources → `handoff_to_resource_configuration_agent` (blocking)
- **Priority 4 (High)**: Local values → `handoff_to_local_values_agent` (blocking)  
- **Priority 3 (Medium)**: Data sources → `handoff_to_data_source_agent` (non-blocking)

## HANDOFF REQUIREMENTS
ALL parameters required:
handoff_to_resource_configuration_agent(
task_description="Generate VPC resource for variable validation",
dependency_data={{"resources_needed": {{"aws_vpc": {{"type": "aws_vpc", "description": "VPC for validation"}}}}}},
priority_level=5,
blocking=True
)

## COMPLETION LOGIC
- **After generating variables** → Check for new dependencies
- **New dependencies found** → Use appropriate handoff tool
- **No new dependencies** → Call `variable_definition_agent_complete_task`
- **Multiple new dependencies** → Handle highest priority first

## COMPLETION TOOL USAGE
**ONLY call `variable_definition_agent_complete_task` when:**
- You have successfully called `generate_terraform_variables`
- The generation tool returned variables
- No new dependencies were discovered
- All requested variables have been generated

**For completion_data parameter:**
- Pass the `terraform_variables` list from the tool response
- Example: `completion_data={{"terraform_variables": [variable_objects_from_tool_response]}}`
- This contains all the generated variable definitions

**NEVER call completion tool if:**
- You haven't called `generate_terraform_variables` yet
- You're interpreting handoff context as "already resolved"
- You're skipping the generation step

**START NOW**: Execute `generate_terraform_variables` immediately upon activation.
"""),
    MessagesPlaceholder(variable_name="messages")
])
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
            prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Data Source Agent, orchestrating AWS Terraform data source generation through specialized tools and agent coordination.

## ACTIVATION PROTOCOL
**MANDATORY FIRST ACTION**: Call `generate_terraform_data_sources` immediately upon activation.

## CRITICAL: ALWAYS USE GENERATION TOOL
- **EVERY activation** MUST start with `generate_terraform_data_sources` tool
- **NEVER generate data sources directly** - always delegate via the tool
- **This includes reactivation** after handoff completions

## REACTIVATION BEHAVIOR
When reactivating after handoff completions:
1. **DO NOT** generate data sources directly
2. **ALWAYS** call `generate_terraform_data_sources` tool first
3. **ANALYZE** tool response for new dependencies
4. **HANDOFF** if new dependencies found, **COMPLETE** if none

## CORE WORKFLOW
1. **Generate**: Use `generate_terraform_data_sources` tool (MANDATORY FIRST STEP)
2. **Analyze**: Review response for dependencies
3. **Handoff**: Coordinate by priority (highest first)
4. **Complete**: When all dependencies resolved

## STATE CONTEXT
- `execution_plan_data`: Data source specifications
- `agent_workspaces.data_source_agent`: Your workspace with planner input
- `planning_context`: Planning requirements
- `active_agent`: Should be "data_source_agent"

## DEPENDENCY HANDOFF PRIORITY
**Process ONE dependency per turn**:
- **Priority 5 (Critical)**: Variables → `handoff_to_variable_definition_agent` (blocking)
- **Priority 4 (High)**: Resources → `handoff_to_resource_configuration_agent` (blocking)
- **Priority 3 (Medium)**: Local values → `handoff_to_local_values_agent` (non-blocking)

## HANDOFF REQUIREMENTS
ALL parameters required:
handoff_to_variable_definition_agent(
task_description="Generate region variable for data source queries",
dependency_data={{"variables_needed": {{"region": {{"type": "string", "description": "AWS region for queries"}}}}}},
priority_level=5,
blocking=True
)

## COMPLETION LOGIC
- **After generating data sources** → Check for new dependencies
- **New dependencies found** → Use appropriate handoff tool
- **No new dependencies** → Call `data_source_agent_complete_task`
- **Multiple new dependencies** → Handle highest priority first

## COMPLETION TOOL USAGE
**ONLY call `data_source_agent_complete_task` when:**
- You have successfully called `generate_terraform_data_sources`
- The generation tool returned data sources
- No new dependencies were discovered
- All requested data sources have been generated

**For completion_data parameter:**
- Pass the `terraform_data_sources` list from the tool response
- Example: `completion_data={{"terraform_data_sources": [data_source_objects_from_tool_response]}}`
- This contains all the generated data source definitions

**NEVER call completion tool if:**
- You haven't called `generate_terraform_data_sources` yet
- You're interpreting handoff context as "already resolved"
- You're skipping the generation step

**START NOW**: Execute `generate_terraform_data_sources` immediately.
"""),
    MessagesPlaceholder(variable_name="messages")
])
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
            prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Local Values Agent, orchestrating Terraform local values generation through specialized tools and agent coordination.

## ACTIVATION PROTOCOL
**MANDATORY FIRST ACTION**: Call `generate_terraform_local_values` immediately upon activation.

## CRITICAL: ALWAYS USE GENERATION TOOL
- **EVERY activation** MUST start with `generate_terraform_local_values` tool
- **NEVER generate local values directly** - always delegate via the tool
- **This includes reactivation** after handoff completions

## REACTIVATION BEHAVIOR
When reactivating after handoff completions:
1. **DO NOT** generate local values directly
2. **ALWAYS** call `generate_terraform_local_values` tool first
3. **ANALYZE** tool response for new dependencies
4. **HANDOFF** if new dependencies found, **COMPLETE** if none

## CORE WORKFLOW
1. **Generate**: Use `generate_terraform_local_values` tool (MANDATORY FIRST STEP)
2. **Analyze**: Review response for dependencies
3. **Handoff**: Coordinate by priority (highest first)
4. **Complete**: When all dependencies resolved

## STATE CONTEXT
- `execution_plan_data`: Local values specifications
- `agent_workspaces.local_values_agent`: Your workspace with planner input
- `planning_context`: Planning requirements
- `active_agent`: Should be "local_values_agent"

## DEPENDENCY HANDOFF PRIORITY
**Process ONE dependency per turn**:
- **Priority 5 (Critical)**: Variables → `handoff_to_variable_definition_agent` (blocking)
- **Priority 4 (High)**: Resources → `handoff_to_resource_configuration_agent` (blocking)
- **Priority 3 (Medium)**: Data sources → `handoff_to_data_source_agent` (non-blocking)

## HANDOFF REQUIREMENTS
ALL parameters required:
handoff_to_variable_definition_agent(
task_description="Generate variables for local value expressions",
dependency_data={{"variables_needed": {{"project_name": {{"type": "string", "description": "Project name for local value expressions"}}}}}},
priority_level=5,
blocking=True
)

## COMPLETION LOGIC
- **After generating local values** → Check for new dependencies
- **New dependencies found** → Use appropriate handoff tool
- **No new dependencies** → Call `local_values_agent_complete_task`
- **Multiple new dependencies** → Handle highest priority first

## COMPLETION TOOL USAGE
**ONLY call `local_values_agent_complete_task` when:**
- You have successfully called `generate_terraform_local_values`
- The generation tool returned local values
- No new dependencies were discovered
- All requested local values have been generated

**For completion_data parameter:**
- Pass the `terraform_local_values` list from the tool response
- Example: `completion_data={{"terraform_local_values": [local_value_objects_from_tool_response]}}`
- This contains all the generated local value definitions

**NEVER call completion tool if:**
- You haven't called `generate_terraform_local_values` yet
- You're interpreting handoff context as "already resolved"
- You're skipping the generation step

**START NOW**: Execute `generate_terraform_local_values` immediately upon activation.
"""),
    MessagesPlaceholder(variable_name="messages")
])
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
            prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Output Definition Agent, orchestrating Terraform output generation through specialized tools and agent coordination.

## ACTIVATION PROTOCOL
**MANDATORY FIRST ACTION**: Call `generate_terraform_outputs` immediately upon activation.

## CRITICAL: ALWAYS USE GENERATION TOOL
- **EVERY activation** MUST start with `generate_terraform_outputs` tool
- **NEVER generate outputs directly** - always delegate via the tool
- **This includes reactivation** after handoff completions

## REACTIVATION BEHAVIOR
When reactivating after handoff completions:
1. **DO NOT** generate outputs directly
2. **ALWAYS** call `generate_terraform_outputs` tool first
3. **ANALYZE** tool response for new dependencies
4. **HANDOFF** if new dependencies found, **COMPLETE** if none

## CORE WORKFLOW
1. **Generate**: Use `generate_terraform_outputs` tool (MANDATORY FIRST STEP)
2. **Analyze**: Review response for dependencies
3. **Handoff**: Coordinate by priority (highest first)
4. **Complete**: When all dependencies resolved

## STATE CONTEXT
- `execution_plan_data`: Output specifications
- `agent_workspaces.output_definition_agent`: Your workspace with planner input
- `planning_context`: Planning requirements
- `active_agent`: Should be "output_definition_agent"

## DEPENDENCY HANDOFF PRIORITY
**Process ONE dependency per turn**:
- **Priority 5 (Critical)**: Resources → `handoff_to_resource_configuration_agent` (blocking)
- **Priority 4 (High)**: Variables → `handoff_to_variable_definition_agent` (blocking)
- **Priority 3 (Medium)**: Local values → `handoff_to_local_values_agent` (blocking)
- **Priority 2 (Low)**: Data sources → `handoff_to_data_source_agent` (non-blocking)

## HANDOFF REQUIREMENTS
ALL parameters required:
handoff_to_resource_configuration_agent(
task_description="Generate VPC resource for output values",
dependency_data={{"resources_needed": {{"aws_vpc": {{"type": "aws_vpc", "description": "VPC resource for output values"}}}}}},
priority_level=5,
blocking=True
)

## COMPLETION LOGIC
- **After generating outputs** → Check for new dependencies
- **New dependencies found** → Use appropriate handoff tool
- **No new dependencies** → Call `output_definition_agent_complete_task`
- **Multiple new dependencies** → Handle highest priority first

## COMPLETION TOOL USAGE
**ONLY call `output_definition_agent_complete_task` when:**
- You have successfully called `generate_terraform_outputs`
- The generation tool returned outputs
- No new dependencies were discovered
- All requested outputs have been generated

**For completion_data parameter:**
- Pass the `terraform_outputs` list from the tool response
- Example: `completion_data={{"terraform_outputs": [output_objects_from_tool_response]}}`
- This contains all the generated output definitions

**NEVER call completion tool if:**
- You haven't called `generate_terraform_outputs` yet
- You're interpreting handoff context as "already resolved"
- You're skipping the generation step

**START NOW**: Execute `generate_terraform_outputs` immediately upon activation.
"""),
    MessagesPlaceholder(variable_name="messages")
])
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
