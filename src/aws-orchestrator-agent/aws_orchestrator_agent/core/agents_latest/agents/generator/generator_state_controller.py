from typing import Annotated
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from .generator_state import GeneratorStageState, GeneratorAgentStatus
from aws_orchestrator_agent.utils.logger import AgentLogger    


class GeneratorStageController:
    """Orchestrates the Planning Stage with parallel execution and dependency management"""
    
    def __init__(self):
        self.completion_threshold = 0.95  # 95% completion required
        self.max_dependency_wait_time = 300  # 5 minutes max wait
        self.parallel_execution_slots = 4  # All 4 agents can run in parallel
        self.logger = AgentLogger("GeneratorStageController")
        
    def initialize_planning_stage(self, state: GeneratorStageState) -> Command:
        """Initialize planning stage with Resource Configuration Agent as entry point"""
        return Command(
            goto="resource_configuration_agent",
            update={
                "stage_status": "planning_active",
                "agent_status_matrix": {
                    **state["agent_status_matrix"],
                    "resource_configuration_agent": GeneratorAgentStatus.ACTIVE
                },
                "planning_progress": {
                    "resource_configuration_agent": 0.0,
                    "variable_definition_agent": 0.0,
                    "data_source_agent": 0.0,
                    "local_values_agent": 0.0
                }
            },
            graph=Command.PARENT
        )
    
    def calculate_stage_completion(self, state: GeneratorStageState) -> float:
        """Calculate overall planning stage completion percentage"""
        progress_values = list(state["planning_progress"].values())
        if not progress_values:
            return 0.0
            
        # Weighted completion based on criticality
        weights = {
            "resource_configuration_agent": 0.4,  # Most critical
            "variable_definition_agent": 0.3,
            "data_source_agent": 0.2,
            "local_values_agent": 0.1
        }
        
        weighted_sum = sum(
            weights.get(agent, 0.25) * progress 
            for agent, progress in state["planning_progress"].items()
        )
        
        return min(weighted_sum, 1.0)
    
    def check_stage_completion_conditions(self, state: GeneratorStageState) -> bool:
        """Comprehensive stage completion check"""
        # Check 1: All agents must be completed
        required_agents = [
            "resource_configuration_agent",
            "variable_definition_agent", 
            "data_source_agent",
            "local_values_agent"
        ]
        
        agents_completed = all(
            state["agent_status_matrix"].get(agent) == GeneratorAgentStatus.COMPLETED
            for agent in required_agents
        )
        
        # Check 2: No pending dependencies
        no_pending_deps = all(
            len(deps) == 0 for deps in state["pending_dependencies"].values()
        )
        
        # Check 3: Overall completion threshold met
        completion_threshold_met = self.calculate_stage_completion(state) >= self.completion_threshold
        
        # Check 4: No agents in error state (unless recovered)
        no_error_agents = all(
            status != GeneratorAgentStatus.ERROR 
            for status in state["agent_status_matrix"].values()
        )
        
        return agents_completed and no_pending_deps and completion_threshold_met and no_error_agents

    @tool("check_planning_stage_completion")
    def check_and_transition_stage(
        self, 
        state: Annotated[GeneratorStageState, InjectedState]
    ) -> Command:
        """Check completion and transition to Enhancement stage if ready"""
        try:
            # Validate state integrity first
            if not self.validate_state_integrity(state):
                self.logger.log_structured(
                    level="ERROR",
                    message="State integrity validation failed, defaulting to resource agent",
                    extra={"fallback_agent": "resource_configuration_agent"}
                )
                return Command(
                    goto="resource_configuration_agent",
                    update={"active_agent": "resource_configuration_agent"},
                    graph=Command.PARENT
                )
            
            if self.check_stage_completion_conditions(state):
                # Prepare transition to Enhancement Stage
                self.logger.log_structured(
                    level="INFO",
                    message="Planning stage completed, transitioning to Enhancement stage",
                    extra={
                        "stage_completion": self.calculate_stage_completion(state),
                        "next_stage": "enhancement",
                        "next_agent": "security_compliance_agent"
                    }
                )
                return Command(
                    goto="security_compliance_agent",  # First agent in Enhancement stage
                    update={
                        "stage_status": "planning_complete",
                        "current_stage": "enhancement",
                        "stage_progress": {
                            **state.get("stage_progress", {}),
                            "planning": 1.0,
                            "enhancement": 0.0
                        }
                    },
                    graph=Command.PARENT
                )
            else:
                # Determine next action based on current state
                next_agent = self.determine_next_active_agent(state)
                self.logger.log_structured(
                    level="DEBUG",
                    message="Planning stage not complete, selecting next agent",
                    extra={
                        "selected_agent": next_agent,
                        "stage_completion": self.calculate_stage_completion(state)
                    }
                )
                return Command(
                    goto=next_agent,
                    update={
                        "active_agent": next_agent
                    },
                    graph=Command.PARENT
                )
                
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in check_and_transition_stage",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            # Fallback to safe default
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    def determine_next_active_agent(self, state: GeneratorStageState) -> str:
        """Intelligent agent selection based on dependencies and priorities"""
        try:
            # Priority 1: Agents with resolved dependencies
            ready_agents = []
            for agent, status in state["agent_status_matrix"].items():
                if status in [GeneratorAgentStatus.INACTIVE, GeneratorAgentStatus.WAITING]:
                    dependencies_met = self.check_agent_dependencies_met(agent, state)
                    if dependencies_met:
                        priority = self.get_agent_priority(agent, state)
                        ready_agents.append((agent, priority))
            
            if ready_agents:
                # Return highest priority agent
                selected_agent = max(ready_agents, key=lambda x: x[1])[0]
                self.logger.log_structured(
                    level="DEBUG",
                    message="Selected next agent from ready agents",
                    extra={
                        "selected_agent": selected_agent,
                        "ready_agents_count": len(ready_agents),
                        "ready_agents": [agent for agent, _ in ready_agents]
                    }
                )
                return selected_agent
            
            # Priority 2: Currently active agent (continue execution)
            current_active = state.get("active_agent")
            if (current_active and 
                state["agent_status_matrix"].get(current_active) == GeneratorAgentStatus.ACTIVE):
                self.logger.log_structured(
                    level="DEBUG",
                    message="Continuing with current active agent",
                    extra={"current_agent": current_active}
                )
                return current_active
                
            # Priority 3: Default to Resource Configuration Agent
            self.logger.log_structured(
                level="DEBUG",
                message="Defaulting to resource_configuration_agent",
                extra={"reason": "no_ready_agents_or_active_agent"}
            )
            return "resource_configuration_agent"
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in determine_next_active_agent",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            # Fallback to safe default
            return "resource_configuration_agent"
    
    def check_agent_dependencies_met(self, agent_name: str, state: GeneratorStageState) -> bool:
        """Check if agent's dependencies are satisfied"""
        required_deps = state["dependency_graph"].get(agent_name, set())
        
        for dep_agent in required_deps:
            dep_status = state["agent_status_matrix"].get(dep_agent)
            if dep_status != GeneratorAgentStatus.COMPLETED:
                return False
                
        return True
    
    def get_agent_priority(self, agent_name: str, state: GeneratorStageState) -> int:
        """Calculate agent priority based on criticality and current state"""
        try:
            # Define priority weights (higher = more important)
            priority_weights = {
                "resource_configuration_agent": 4,  # Highest priority
                "variable_definition_agent": 3,
                "data_source_agent": 2,
                "local_values_agent": 1
            }
            
            base_priority = priority_weights.get(agent_name, 0)
            
            # Boost priority if agent has been waiting longer
            # This helps prevent starvation of lower-priority agents
            waiting_time = state.get("agent_waiting_times", {}).get(agent_name, 0)
            time_boost = min(waiting_time // 60, 2)  # Max 2 point boost for waiting 1+ minutes
            
            total_priority = base_priority + time_boost
            self.logger.log_structured(
                level="DEBUG",
                message="Calculated agent priority",
                extra={
                    "agent_name": agent_name,
                    "base_priority": base_priority,
                    "time_boost": time_boost,
                    "total_priority": total_priority,
                    "waiting_time": waiting_time
                }
            )
            
            return total_priority
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error calculating priority for agent",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_priority": 1
                }
            )
            # Return default priority for unknown agents
            return 1
    
    def validate_state_integrity(self, state: GeneratorStageState) -> bool:
        """Validate that the state has all required fields and is in a valid condition"""
        try:
            # Check required fields exist
            required_fields = [
                "agent_status_matrix", 
                "planning_progress", 
                "dependency_graph",
                "pending_dependencies"
            ]
            
            for field in required_fields:
                if field not in state:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing required state field",
                        extra={"missing_field": field, "required_fields": required_fields}
                    )
                    return False
            
            # Check agent status matrix has all required agents
            required_agents = [
                "resource_configuration_agent",
                "variable_definition_agent", 
                "data_source_agent",
                "local_values_agent"
            ]
            
            for agent in required_agents:
                if agent not in state["agent_status_matrix"]:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing agent in status matrix",
                        extra={"missing_agent": agent, "required_agents": required_agents}
                    )
                    return False
            
            # Check progress tracking has all agents
            for agent in required_agents:
                if agent not in state["planning_progress"]:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing agent in progress tracking",
                        extra={"missing_agent": agent, "required_agents": required_agents}
                    )
                    return False
            
            self.logger.log_structured(
                level="DEBUG",
                message="State integrity validation passed",
                extra={
                    "validated_fields": len(required_fields),
                    "validated_agents": len(required_agents)
                }
            )
            return True
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error validating state integrity",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return False
    
    def test_check_and_transition_stage(self, state: GeneratorStageState) -> bool:
        """Test version of check_and_transition_stage without @tool decorator"""
        try:
            # Validate state integrity first
            if not self.validate_state_integrity(state):
                self.logger.log_structured(
                    level="ERROR",
                    message="State integrity validation failed, defaulting to resource agent",
                    extra={"fallback_agent": "resource_configuration_agent"}
                )
                return False
            
            return self.check_stage_completion_conditions(state)
                
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in test_check_and_transition_stage",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            return False
