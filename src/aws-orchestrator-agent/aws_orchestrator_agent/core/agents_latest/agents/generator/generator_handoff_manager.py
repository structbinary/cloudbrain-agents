from typing import Annotated, Dict, Any, List
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId, tool
from .generator_state import GeneratorSwarmState, DependencyType, GeneratorAgentStatus
from .generator_state_controller import GeneratorStageController
from aws_orchestrator_agent.utils.logger import AgentLogger
import datetime


class GeneratorStageHandoffManager:
    """Manages sophisticated handoffs with dependency resolution and context filtering"""
    
    def __init__(self):
        self.logger = AgentLogger("GeneratorStageHandoffManager")
    
    def create_dependency_aware_handoff_tool(
        self, 
        target_agent: str, 
        dependency_type: DependencyType,
        description: str
    ):
        """Create intelligent handoff tools that understand dependencies"""
        
        @tool(f"handoff_to_{target_agent}_{dependency_type.value}", description=description)
        def dependency_handoff_tool(
            task_description: Annotated[str, "Specific task for target agent"],
            dependency_data: Annotated[Dict[str, Any], "Structured dependency data"],
            state: Annotated[Any, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            priority_level: Annotated[int, "Priority: 1=low, 5=critical"] = 3,
            blocking: Annotated[bool, "Whether source agent should wait for completion"] = True
        ) -> Command:
            try:
                source_agent = state["active_agent"]
                
                self.logger.log_structured(
                    level="INFO",
                    message="Creating dependency handoff",
                    extra={
                        "source_agent": source_agent,
                        "target_agent": target_agent,
                        "dependency_type": dependency_type.value,
                        "priority_level": priority_level,
                        "blocking": blocking,
                        "task_description": task_description[:100] + "..." if len(task_description) > 100 else task_description
                    }
                )
                
                # Create dependency request
                dependency_request = {
                    "id": f"{source_agent}_{target_agent}_{int(datetime.datetime.now().timestamp())}",
                    "source_agent": source_agent,
                    "target_agent": target_agent,
                    "dependency_type": dependency_type.value,
                    "task_description": task_description,
                    "dependency_data": dependency_data,
                    "priority_level": priority_level,
                    "blocking": blocking,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": "pending"
                }
            
                # Update dependency tracking
                updated_pending_deps = {
                    **state["pending_dependencies"],
                    target_agent: [
                        *state["pending_dependencies"].get(target_agent, []),
                        dependency_request
                    ]
                }
                
                # Update dependency graph
                updated_dep_graph = {
                    **state["dependency_graph"],
                    target_agent: {
                        *state["dependency_graph"].get(target_agent, set()),
                        source_agent
                    }
                }
                
                # Update agent status
                updated_status_matrix = {
                    **state["agent_status_matrix"],
                    target_agent: GeneratorAgentStatus.ACTIVE
                }
                
                if blocking:
                    updated_status_matrix[source_agent] = GeneratorAgentStatus.WAITING
                
                # Create context for target agent
                target_context = self.create_target_agent_context(
                    state, dependency_request, target_agent
                )
                
                # Create tool message
                tool_message = ToolMessage(
                    content=f"Dependency handoff to {target_agent}: {task_description}",
                    name=f"handoff_to_{target_agent}_{dependency_type.value}",
                    tool_call_id=tool_call_id
                )
                
                self.logger.log_structured(
                    level="INFO",
                    message="Dependency handoff completed",
                    extra={
                        "dependency_id": dependency_request["id"],
                        "source_agent": source_agent,
                        "target_agent": target_agent,
                        "blocking": blocking,
                        "pending_deps_count": len(updated_pending_deps.get(target_agent, []))
                    }
                )
                
                return Command(
                    goto=target_agent,
                    update={
                        "messages": [*state["messages"], tool_message],
                        "active_agent": target_agent,
                        "agent_status_matrix": updated_status_matrix,
                        "pending_dependencies": updated_pending_deps,
                        "dependency_graph": updated_dep_graph,
                        "agent_workspaces": {
                            **state["agent_workspaces"],
                            target_agent: {
                                **state["agent_workspaces"].get(target_agent, {}),
                                "current_task": dependency_request,
                                "context": target_context
                            }
                        },
                        "handoff_queue": [
                            *state["handoff_queue"],
                            dependency_request
                        ]
                    },
                    graph=Command.PARENT
                )
                
            except Exception as e:
                self.logger.log_structured(
                    level="ERROR",
                    message="Dependency handoff failed",
                    extra={
                        "source_agent": state.get("active_agent", "unknown"),
                        "target_agent": target_agent,
                        "dependency_type": dependency_type.value,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                # Return safe fallback command
                return Command(
                    goto="resource_configuration_agent",
                    update={"active_agent": "resource_configuration_agent"},
                    graph=Command.PARENT
                )
            
        return dependency_handoff_tool
    
    def create_target_agent_context(
        self, 
        state: GeneratorSwarmState, 
        dependency_request: Dict[str, Any], 
        target_agent: str
    ) -> Dict[str, Any]:
        """Create filtered, relevant context for target agent"""
        
        context = {
            "task": dependency_request,
            "relevant_artifacts": {},
            "execution_plan_excerpt": {},
            "cross_agent_data": {}
        }
        
        # Filter relevant artifacts based on agent type and dependency
        if target_agent == "variable_definition_agent":
            context["relevant_artifacts"] = {
                "resources_needing_variables": dependency_request["dependency_data"].get("resources", []),
                "variable_requirements": dependency_request["dependency_data"].get("variable_specs", [])
            }
        elif target_agent == "data_source_agent":
            context["relevant_artifacts"] = {
                "external_references": dependency_request["dependency_data"].get("external_refs", []),
                "lookup_requirements": dependency_request["dependency_data"].get("lookups", [])
            }
        elif target_agent == "local_values_agent":
            context["relevant_artifacts"] = {
                "computation_requirements": dependency_request["dependency_data"].get("computations", []),
                "expression_specs": dependency_request["dependency_data"].get("expressions", [])
            }
        
        # Add cross-agent data that might be relevant
        context["cross_agent_data"] = {
            agent: workspace.get("generated_resources", []) + 
                   workspace.get("generated_variables", []) +
                   workspace.get("generated_data_sources", []) +
                   workspace.get("generated_locals", [])
            for agent, workspace in state["agent_workspaces"].items()
            if agent != target_agent
        }
        
        return context

    def create_completion_handoff_tool(self, agent_name: str):
        """Create a completion handoff tool for a specific agent"""
        return create_completion_handoff_tool(agent_name)


## Completion and Resolution Handoff

def create_completion_handoff_tool(source_agent: str):
    """Tool for agents to signal completion and resolve dependencies"""
    
    logger = AgentLogger("GeneratorStageHandoffManager")
    
    @tool(f"{source_agent}_complete_task", description=f"Signal completion of {source_agent}'s task")
    def completion_handoff_tool(
        completion_data: Annotated[Dict[str, Any], "Generated artifacts and results"],
        resolved_dependencies: Annotated[List[str], "List of dependency IDs resolved"],
        next_recommendations: Annotated[List[str], "Recommended next agents to activate"],
        state: Annotated[GeneratorSwarmState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        try:
            logger.log_structured(
                level="INFO",
                message="Processing task completion",
                extra={
                    "source_agent": source_agent,
                    "resolved_dependencies_count": len(resolved_dependencies),
                    "next_recommendations": next_recommendations,
                    "completion_data_keys": list(completion_data.keys()) if completion_data else []
                }
            )
        
            # Update agent status to completed
            updated_status_matrix = {
                **state["agent_status_matrix"],
                source_agent: GeneratorAgentStatus.COMPLETED
            }
            
            # Move resolved dependencies from pending to resolved
            updated_resolved_deps = {
                **state["resolved_dependencies"],
                source_agent: [
                    *state["resolved_dependencies"].get(source_agent, []),
                    *resolved_dependencies
                ]
            }
            
            # Remove resolved dependencies from pending
            updated_pending_deps = {}
            for agent, deps in state["pending_dependencies"].items():
                remaining_deps = [
                    dep for dep in deps 
                    if dep["id"] not in resolved_dependencies
                ]
                updated_pending_deps[agent] = remaining_deps
            
            # Update agent workspace with completion data
            updated_workspaces = {
                **state["agent_workspaces"],
                source_agent: {
                    **state["agent_workspaces"].get(source_agent, {}),
                    **completion_data,
                    "status": "completed",
                    "completion_timestamp": datetime.datetime.now().isoformat()
                }
            }
            
            # Update progress
            updated_progress = {
                **state["planning_progress"],
                source_agent: 1.0
            }
            
            # Determine next agent based on recommendations and dependencies
            controller = GeneratorStageController()
            next_agent = controller.determine_next_active_agent({
                **state,
                "agent_status_matrix": updated_status_matrix,
                "pending_dependencies": updated_pending_deps
            })
            
            # Check if stage is complete
            stage_complete = controller.check_stage_completion_conditions({
                **state,
                "agent_status_matrix": updated_status_matrix,
                "pending_dependencies": updated_pending_deps,
                "planning_progress": updated_progress
            })

            tool_message = ToolMessage(
                content=f"Completion handoff to {source_agent}",
                name=f"handoff_to_{source_agent}_complete_task",
                tool_call_id=tool_call_id
            )

            if stage_complete:
                logger.log_structured(
                    level="INFO",
                    message="Planning stage completed",
                    extra={
                        "source_agent": source_agent,
                        "next_action": "check_planning_stage_completion",
                        "resolved_deps_count": len(resolved_dependencies)
                    }
                )
                
                return Command(
                    goto="check_planning_stage_completion",  # Transition checker
                    update={
                        "messages": [*state["messages"], tool_message],
                        "agent_status_matrix": updated_status_matrix,
                        "resolved_dependencies": updated_resolved_deps,
                        "pending_dependencies": updated_pending_deps,
                        "agent_workspaces": updated_workspaces,
                        "planning_progress": updated_progress,
                        "stage_status": "planning_complete"
                    },
                    graph=Command.PARENT
                )
            else:
                logger.log_structured(
                    level="INFO",
                    message="Task completed, transitioning to next agent",
                    extra={
                        "source_agent": source_agent,
                        "next_agent": next_agent,
                        "resolved_deps_count": len(resolved_dependencies),
                        "stage_complete": False
                    }
                )
                
                return Command(
                    goto=next_agent,
                    update={
                        "messages": [*state["messages"], tool_message],
                        "active_agent": next_agent,
                        "agent_status_matrix": {
                            **updated_status_matrix,
                            next_agent: GeneratorAgentStatus.ACTIVE
                        },
                        "resolved_dependencies": updated_resolved_deps,
                        "pending_dependencies": updated_pending_deps,
                        "agent_workspaces": updated_workspaces,
                        "planning_progress": updated_progress
                    },
                    graph=Command.PARENT
                )
                
        except Exception as e:
            logger.log_structured(
                level="ERROR",
                message="Task completion handoff failed",
                extra={
                    "source_agent": source_agent,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Return safe fallback command
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    return completion_handoff_tool

