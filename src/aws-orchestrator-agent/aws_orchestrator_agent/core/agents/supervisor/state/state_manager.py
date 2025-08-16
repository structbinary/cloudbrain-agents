"""
State Manager for AWS Orchestrator Agent Supervisor.

This module provides the StateManager class that orchestrates all state
transformations, provides validation, state snapshotting for audit trails,
error recovery with state rollback, and integration points for the Supervisor Agent.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

from aws_orchestrator_agent.core.agents.supervisor.state.schemas import (
    SupervisorState,
    AnalysisState,
    GenerationState,
    ValidationState,
    EditorState,
    AgentType,
    WorkflowStatus
)
from aws_orchestrator_agent.core.agents.supervisor.state.transformers import (
    get_transformation_functions,
    supervisor_to_analysis_state,
    analysis_to_supervisor_state,
    supervisor_to_generation_state,
    generation_to_supervisor_state,
    supervisor_to_validation_state,
    validation_to_supervisor_state,
    supervisor_to_editor_state,
    editor_to_supervisor_state
)

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Represents a snapshot of the supervisor state for rollback purposes."""
    timestamp: datetime
    workflow_id: str
    state_data: Dict[str, Any]
    description: str
    step: str


class StateManager:
    """
    Manages state transformations and validation for the Supervisor Agent.
    
    This class provides a unified interface for state management, including
    transformation coordination, validation, snapshotting, and error recovery.
    """
    
    def __init__(self, max_snapshots: int = 10):
        """
        Initialize the StateManager.
        
        Args:
            max_snapshots: Maximum number of snapshots to keep for rollback
        """
        self.max_snapshots = max_snapshots
        self.snapshots: List[StateSnapshot] = []
        
        # Initialize transformation function mapping
        self.transformers: Dict[AgentType, Tuple] = {
            AgentType.ANALYSIS: (supervisor_to_analysis_state, analysis_to_supervisor_state),
            AgentType.GENERATION: (supervisor_to_generation_state, generation_to_supervisor_state),
            AgentType.VALIDATION: (supervisor_to_validation_state, validation_to_supervisor_state),
            AgentType.EDITOR: (supervisor_to_editor_state, editor_to_supervisor_state),
        }
        
        logger.info("StateManager initialized successfully")
    
    def transform_to_subgraph(
        self, 
        agent_type: AgentType, 
        supervisor_state: SupervisorState
    ) -> Union[AnalysisState, GenerationState, ValidationState, EditorState]:
        """
        Transform supervisor state to subgraph input state.
        
        Args:
            agent_type: The type of agent to transform for
            supervisor_state: The current supervisor state
            
        Returns:
            The appropriate subgraph state for the agent type
            
        Raises:
            ValueError: If agent type is not supported or transformation fails
        """
        try:
            # Validate agent type
            if agent_type not in self.transformers:
                raise ValueError(f"Unsupported agent type: {agent_type}")
            
            # Get transformation function
            input_transformer, _ = self.transformers[agent_type]
            
            # Validate supervisor state before transformation
            self._validate_supervisor_state(supervisor_state, agent_type)
            
            # Create snapshot before transformation
            self.create_snapshot(supervisor_state, f"Before {str(agent_type)} transformation")
            
            # Perform transformation
            subgraph_state = input_transformer(supervisor_state)
            
            # Validate subgraph state
            self._validate_subgraph_state(subgraph_state, agent_type)
            
            logger.info(f"Successfully transformed supervisor state to {str(agent_type)} state")
            return subgraph_state
            
        except Exception as e:
            logger.error(f"Failed to transform supervisor state to {str(agent_type)} state: {e}")
            # Add error context to supervisor state
            supervisor_state.error_context[f"{str(agent_type)}_transformation_error"] = str(e)
            raise
    
    def merge_from_subgraph(
        self, 
        agent_type: AgentType, 
        supervisor_state: SupervisorState, 
        subgraph_state: Union[AnalysisState, GenerationState, ValidationState, EditorState]
    ) -> SupervisorState:
        """
        Merge subgraph output state back into supervisor state.
        
        Args:
            agent_type: The type of agent that produced the subgraph state
            supervisor_state: The current supervisor state
            subgraph_state: The subgraph state to merge
            
        Returns:
            Updated SupervisorState with subgraph results merged
            
        Raises:
            ValueError: If agent type is not supported or merge fails
        """
        try:
            # Validate agent type
            if agent_type not in self.transformers:
                raise ValueError(f"Unsupported agent type: {agent_type}")
            
            # Get transformation function
            _, output_transformer = self.transformers[agent_type]
            
            # Validate subgraph state before merge
            self._validate_subgraph_state(subgraph_state, agent_type)
            
            # Perform merge
            updated_supervisor = output_transformer(supervisor_state, subgraph_state)
            
            # Validate updated supervisor state
            self._validate_supervisor_state(updated_supervisor, agent_type)
            
            # Create snapshot after successful merge
            self.create_snapshot(updated_supervisor, f"After {str(agent_type)} merge")
            
            logger.info(f"Successfully merged {str(agent_type)} state back to supervisor state")
            return updated_supervisor
            
        except Exception as e:
            logger.error(f"Failed to merge {str(agent_type)} state to supervisor state: {e}")
            # Add error context to supervisor state
            supervisor_state.error_context[f"{str(agent_type)}_merge_error"] = str(e)
            raise
    
    def validate_state(self, state: Union[SupervisorState, AnalysisState, GenerationState, ValidationState, EditorState]) -> bool:
        """
        Validate a state object for schema compliance and business rules.
        
        Args:
            state: The state object to validate
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValueError: If validation fails with specific error details
        """
        try:
            # Basic schema validation (Pydantic handles this automatically)
            if not isinstance(state, (SupervisorState, AnalysisState, GenerationState, ValidationState, EditorState)):
                raise ValueError(f"Invalid state type: {type(state)}")
            
            # Business rule validation based on state type
            if isinstance(state, SupervisorState):
                return self._validate_supervisor_business_rules(state)
            elif isinstance(state, AnalysisState):
                return self._validate_analysis_business_rules(state)
            elif isinstance(state, GenerationState):
                return self._validate_generation_business_rules(state)
            elif isinstance(state, ValidationState):
                return self._validate_validation_business_rules(state)
            elif isinstance(state, EditorState):
                return self._validate_editor_business_rules(state)
            
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            raise ValueError(f"State validation failed: {e}")
    
    def create_snapshot(self, supervisor_state: SupervisorState, description: str) -> StateSnapshot:
        """
        Create a snapshot of the supervisor state for rollback purposes.
        
        Args:
            supervisor_state: The supervisor state to snapshot
            description: Description of the snapshot
            
        Returns:
            StateSnapshot object containing the snapshot data
        """
        try:
            # Create snapshot
            snapshot = StateSnapshot(
                timestamp=datetime.utcnow(),
                workflow_id=supervisor_state.workflow_id,
                state_data=supervisor_state.model_dump(),
                description=description,
                step=supervisor_state.current_step
            )
            
            # Add to snapshots list
            self.snapshots.append(snapshot)
            
            # Maintain maximum snapshot limit
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots.pop(0)  # Remove oldest snapshot
            
            # Add audit entry
            supervisor_state.add_audit_entry("snapshot_created", {
                "description": description,
                "step": supervisor_state.current_step,
                "snapshot_count": len(self.snapshots)
            })
            
            logger.info(f"Created state snapshot: {description}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create state snapshot: {e}")
            raise
    
    def rollback_to_snapshot(self, supervisor_state: SupervisorState, snapshot_index: int = -1) -> SupervisorState:
        """
        Rollback supervisor state to a previous snapshot.
        
        Args:
            supervisor_state: The current supervisor state
            snapshot_index: Index of snapshot to rollback to (default: latest)
            
        Returns:
            SupervisorState restored from the snapshot
            
        Raises:
            ValueError: If snapshot index is invalid or rollback fails
        """
        try:
            # Validate snapshot index
            if not self.snapshots:
                raise ValueError("No snapshots available for rollback")
            
            if abs(snapshot_index) > len(self.snapshots):
                raise ValueError(f"Invalid snapshot index: {snapshot_index}")
            
            # Get target snapshot
            target_snapshot = self.snapshots[snapshot_index]
            
            # Create snapshot of current state before rollback
            self.create_snapshot(supervisor_state, f"Before rollback to {target_snapshot.description}")
            
            # Restore state from snapshot
            restored_state = SupervisorState(**target_snapshot.state_data)
            
            # Add rollback audit entry
            restored_state.add_audit_entry("state_rollback", {
                "rollback_to": target_snapshot.description,
                "rollback_timestamp": target_snapshot.timestamp.isoformat(),
                "current_step": restored_state.current_step
            })
            
            # Update error context
            restored_state.error_context["last_rollback"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_snapshot": target_snapshot.description,
                "reason": "Manual rollback or error recovery"
            }
            
            logger.info(f"Successfully rolled back to snapshot: {target_snapshot.description}")
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to rollback to snapshot: {e}")
            raise
    
    def get_snapshot_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available snapshots.
        
        Returns:
            List of snapshot information dictionaries
        """
        return [
            {
                "index": i,
                "timestamp": snapshot.timestamp.isoformat(),
                "description": snapshot.description,
                "step": snapshot.step,
                "workflow_id": snapshot.workflow_id
            }
            for i, snapshot in enumerate(self.snapshots)
        ]
    
    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()
        logger.info("All snapshots cleared")
    
    def get_workflow_progress(self, supervisor_state: SupervisorState) -> Dict[str, Any]:
        """
        Get detailed workflow progress information.
        
        Args:
            supervisor_state: The supervisor state to analyze
            
        Returns:
            Dictionary containing progress information
        """
        progress = {
            "workflow_id": supervisor_state.workflow_id,
            "current_step": supervisor_state.current_step,
            "status": supervisor_state.status,
            "completion_percentage": supervisor_state.get_completion_percentage(),
            "total_execution_time": supervisor_state.total_execution_time,
            "agent_execution_times": supervisor_state.agent_execution_times,
            "routing_history_length": len(supervisor_state.routing_history),
            "audit_log_length": len(supervisor_state.audit_log),
            "agent_states": {}
        }
        
        # Add agent state information
        for agent_type in AgentType:
            agent_state = supervisor_state.get_agent_state(agent_type)
            if agent_state:
                progress["agent_states"][agent_type.value] = {
                    "exists": True,
                    "complete": getattr(agent_state, f"{agent_type.value}_complete", False)
                }
            else:
                progress["agent_states"][agent_type.value] = {
                    "exists": False,
                    "complete": False
                }
        
        return progress
    
    def export_state_for_debugging(self, supervisor_state: SupervisorState) -> Dict[str, Any]:
        """
        Export supervisor state for debugging purposes.
        
        Args:
            supervisor_state: The supervisor state to export
            
        Returns:
            Dictionary containing exported state data
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": supervisor_state.workflow_id,
            "state_data": supervisor_state.model_dump(),
            "progress": self.get_workflow_progress(supervisor_state),
            "snapshots": self.get_snapshot_info(),
            "error_context": supervisor_state.error_context
        }
    
    def _validate_supervisor_state(self, supervisor_state: SupervisorState, agent_type: AgentType) -> None:
        """Validate supervisor state before transformation."""
        if not supervisor_state.user_request:
            raise ValueError("Supervisor state missing required user_request")
        
        # Agent-specific validation
        if agent_type == AgentType.GENERATION:
            if not supervisor_state.analysis_state:
                raise ValueError("Analysis state required before generation")
            if not supervisor_state.analysis_state.analysis_complete:
                raise ValueError("Analysis must be complete before generation")
        
        elif agent_type == AgentType.VALIDATION:
            if not supervisor_state.generation_state and not supervisor_state.editor_state:
                raise ValueError("Generation or editor state required before validation")
        
        elif agent_type == AgentType.EDITOR:
            if not supervisor_state.generation_state and not supervisor_state.editor_state:
                raise ValueError("Generation or previous editor state required before editing")
    
    def _validate_subgraph_state(self, subgraph_state: Union[AnalysisState, GenerationState, ValidationState, EditorState], agent_type: AgentType) -> None:
        """Validate subgraph state after transformation."""
        if agent_type == AgentType.ANALYSIS:
            if not isinstance(subgraph_state, AnalysisState):
                raise ValueError("Invalid analysis state type")
            if not subgraph_state.query:
                raise ValueError("Analysis state missing required query field")
        
        elif agent_type == AgentType.GENERATION:
            if not isinstance(subgraph_state, GenerationState):
                raise ValueError("Invalid generation state type")
            if not subgraph_state.module_name:
                raise ValueError("Generation state missing required module_name field")
        
        elif agent_type == AgentType.VALIDATION:
            if not isinstance(subgraph_state, ValidationState):
                raise ValueError("Invalid validation state type")
            if not subgraph_state.terraform_code:
                raise ValueError("Validation state missing required terraform_code field")
        
        elif agent_type == AgentType.EDITOR:
            if not isinstance(subgraph_state, EditorState):
                raise ValueError("Invalid editor state type")
            if not subgraph_state.original_code:
                raise ValueError("Editor state missing required original_code field")
    
    def _validate_supervisor_business_rules(self, state: SupervisorState) -> bool:
        """Validate supervisor state business rules."""
        # Check workflow ID format
        if not state.workflow_id or len(state.workflow_id) < 10:
            return False
        
        # Check user request
        if not state.user_request:
            return False
        
        # Check status consistency
        if state.status == WorkflowStatus.COMPLETED and not state.is_workflow_complete():
            return False
        
        # Check execution time consistency
        if state.total_execution_time < 0:
            return False
        
        return True
    
    def _validate_analysis_business_rules(self, state: AnalysisState) -> bool:
        """Validate analysis state business rules."""
        # Check confidence score range
        if not (0.0 <= state.confidence_score <= 1.0):
            return False
        
        # Check completion consistency
        if state.analysis_complete and not state.requirements:
            return False
        
        return True
    
    def _validate_generation_business_rules(self, state: GenerationState) -> bool:
        """Validate generation state business rules."""
        # Check module name format
        if not state.module_name or len(state.module_name) < 3:
            return False
        
        # Check completion consistency
        if state.generation_complete and not state.terraform_code:
            return False
        
        return True
    
    def _validate_validation_business_rules(self, state: ValidationState) -> bool:
        """Validate validation state business rules."""
        # Check overall score range
        if not (0.0 <= state.overall_score <= 100.0):
            return False
        
        # Check completion consistency
        if state.validation_complete and not state.validation_reports:
            return False
        
        return True
    
    def _validate_editor_business_rules(self, state: EditorState) -> bool:
        """Validate editor state business rules."""
        # Check completion consistency
        if state.editor_complete and not state.modified_code:
            return False
        
        # Check backup creation
        if state.backup_created and not state.original_code:
            return False
        
        return True 