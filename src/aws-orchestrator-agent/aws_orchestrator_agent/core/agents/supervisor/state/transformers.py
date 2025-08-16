"""
State transformation functions for AWS Orchestrator Agent Supervisor.

This module provides transformation functions that map between Supervisor state
and specialized agent subgraph states, ensuring seamless data flow and context
sharing across the multi-agent orchestration system.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from aws_orchestrator_agent.core.agents.supervisor.state.schemas import (
    SupervisorState,
    AnalysisState,
    GenerationState,
    ValidationState,
    EditorState,
    AgentType,
    WorkflowStatus
)

logger = logging.getLogger(__name__)


# Supervisor ↔ Analysis Agent Transformations
def supervisor_to_analysis_state(supervisor_state: SupervisorState) -> AnalysisState:
    """
    Transform supervisor state to analysis agent input state.
    
    This function extracts relevant information from the supervisor state
    and creates an AnalysisState suitable for the Analysis Agent subgraph.
    
    Args:
        supervisor_state: The current supervisor state
        
    Returns:
        AnalysisState configured for analysis agent input
        
    Raises:
        ValueError: If required supervisor state fields are missing
    """
    try:
        # Validate required supervisor state fields
        if not supervisor_state.user_request:
            raise ValueError("Supervisor state missing required user_request")
        
        # Extract AWS context from MCP context
        aws_context = supervisor_state.mcp_context.get("aws_context", {})
        
        # Create analysis state with supervisor data
        analysis_state = AnalysisState(
            query=supervisor_state.user_request,
            conversation_history=supervisor_state.routing_history.copy(),
            aws_context=aws_context,
            analysis_started_at=datetime.utcnow()
        )
        
        logger.info(f"Transformed supervisor state to analysis state for workflow {supervisor_state.workflow_id}")
        return analysis_state
        
    except Exception as e:
        logger.error(f"Failed to transform supervisor state to analysis state: {e}")
        raise


def analysis_to_supervisor_state(supervisor_state: SupervisorState, analysis_state: AnalysisState) -> SupervisorState:
    """
    Merge analysis results back into supervisor state.
    
    This function takes the completed analysis state and merges it back
    into the supervisor state, updating routing history and audit logs.
    
    Args:
        supervisor_state: The current supervisor state
        analysis_state: The completed analysis state
        
    Returns:
        Updated SupervisorState with analysis results merged
        
    Raises:
        ValueError: If analysis state is invalid or incomplete
    """
    try:
        # Validate analysis state
        if not analysis_state.query:
            raise ValueError("Analysis state missing required query field")
        
        # Create a copy of supervisor state to avoid mutations
        updated_supervisor = supervisor_state.model_copy(deep=True)
        
        # Set analysis state
        updated_supervisor.analysis_state = analysis_state
        
        # Update current step and status
        updated_supervisor.current_step = "analysis"
        if analysis_state.analysis_complete:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        else:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        
        # Add routing history entry
        routing_details = {
            "agent_type": AgentType.ANALYSIS,
            "confidence_score": analysis_state.confidence_score,
            "requirements_count": len(analysis_state.requirements),
            "errors": analysis_state.analysis_errors
        }
        updated_supervisor.add_routing_entry(
            "analysis", 
            "completed" if analysis_state.analysis_complete else "in_progress",
            routing_details
        )
        
        # Add audit entry
        audit_details = {
            "agent_type": AgentType.ANALYSIS,
            "analysis_complete": analysis_state.analysis_complete,
            "confidence_score": analysis_state.confidence_score,
            "requirements_extracted": len(analysis_state.requirements)
        }
        updated_supervisor.add_audit_entry("analysis_completed", audit_details)
        
        # Update execution time if analysis is complete
        if analysis_state.analysis_complete and analysis_state.analysis_started_at:
            execution_time = (datetime.utcnow() - analysis_state.analysis_started_at).total_seconds()
            updated_supervisor.agent_execution_times[AgentType.ANALYSIS] = execution_time
        
        logger.info(f"Merged analysis state back to supervisor state for workflow {supervisor_state.workflow_id}")
        return updated_supervisor
        
    except Exception as e:
        logger.error(f"Failed to merge analysis state to supervisor state: {e}")
        raise


# Supervisor ↔ Generation Agent Transformations
def supervisor_to_generation_state(supervisor_state: SupervisorState) -> GenerationState:
    """
    Transform supervisor state to generation agent input state.
    
    This function extracts requirements from analysis state and creates
    a GenerationState suitable for the Generation Agent subgraph.
    
    Args:
        supervisor_state: The current supervisor state
        
    Returns:
        GenerationState configured for generation agent input
        
    Raises:
        ValueError: If analysis state is missing or incomplete
    """
    try:
        # Validate that analysis state exists and is complete
        if not supervisor_state.analysis_state:
            raise ValueError("Analysis state required before generation")
        
        if not supervisor_state.analysis_state.analysis_complete:
            raise ValueError("Analysis must be complete before generation")
        
        # Extract requirements from analysis state
        requirements = supervisor_state.analysis_state.requirements
        
        # Determine module name from user request or analysis
        module_name = supervisor_state.user_request
        if requirements.get("module_name"):
            module_name = requirements["module_name"]
        
        # Create generation state
        generation_state = GenerationState(
            module_name=module_name,
            requirements=requirements,
            generation_started_at=datetime.utcnow()
        )
        
        logger.info(f"Transformed supervisor state to generation state for workflow {supervisor_state.workflow_id}")
        return generation_state
        
    except Exception as e:
        logger.error(f"Failed to transform supervisor state to generation state: {e}")
        raise


def generation_to_supervisor_state(supervisor_state: SupervisorState, generation_state: GenerationState) -> SupervisorState:
    """
    Merge generation results back into supervisor state.
    
    This function takes the completed generation state and merges it back
    into the supervisor state, updating routing history and audit logs.
    
    Args:
        supervisor_state: The current supervisor state
        generation_state: The completed generation state
        
    Returns:
        Updated SupervisorState with generation results merged
        
    Raises:
        ValueError: If generation state is invalid
    """
    try:
        # Validate generation state
        if not generation_state.module_name:
            raise ValueError("Generation state missing required module_name field")
        
        # Create a copy of supervisor state to avoid mutations
        updated_supervisor = supervisor_state.model_copy(deep=True)
        
        # Set generation state
        updated_supervisor.generation_state = generation_state
        
        # Update current step and status
        updated_supervisor.current_step = "generation"
        if generation_state.generation_complete:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        else:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        
        # Add routing history entry
        routing_details = {
            "agent_type": AgentType.GENERATION,
            "module_name": generation_state.module_name,
            "files_generated": len(generation_state.generated_files),
            "errors": generation_state.generation_errors,
            "best_practices_applied": len(generation_state.best_practices_applied)
        }
        updated_supervisor.add_routing_entry(
            "generation",
            "completed" if generation_state.generation_complete else "in_progress",
            routing_details
        )
        
        # Add audit entry
        audit_details = {
            "agent_type": AgentType.GENERATION,
            "generation_complete": generation_state.generation_complete,
            "module_name": generation_state.module_name,
            "files_generated": len(generation_state.generated_files),
            "best_practices_applied": generation_state.best_practices_applied
        }
        updated_supervisor.add_audit_entry("generation_completed", audit_details)
        
        # Update execution time if generation is complete
        if generation_state.generation_complete and generation_state.generation_started_at:
            execution_time = (datetime.utcnow() - generation_state.generation_started_at).total_seconds()
            updated_supervisor.agent_execution_times[AgentType.GENERATION] = execution_time
        
        logger.info(f"Merged generation state back to supervisor state for workflow {supervisor_state.workflow_id}")
        return updated_supervisor
        
    except Exception as e:
        logger.error(f"Failed to merge generation state to supervisor state: {e}")
        raise


# Supervisor ↔ Validation Agent Transformations
def supervisor_to_validation_state(supervisor_state: SupervisorState) -> ValidationState:
    """
    Transform supervisor state to validation agent input state.
    
    This function extracts Terraform code from either generation or editor state
    and creates a ValidationState suitable for the Validation Agent subgraph.
    
    Args:
        supervisor_state: The current supervisor state
        
    Returns:
        ValidationState configured for validation agent input
        
    Raises:
        ValueError: If no Terraform code is available for validation
    """
    try:
        # Determine source of Terraform code
        terraform_code = {}
        
        if supervisor_state.generation_state and supervisor_state.generation_state.generation_complete:
            terraform_code = supervisor_state.generation_state.terraform_code
            logger.info("Using Terraform code from generation state for validation")
        elif supervisor_state.editor_state and supervisor_state.editor_state.editor_complete:
            terraform_code = supervisor_state.editor_state.modified_code
            logger.info("Using Terraform code from editor state for validation")
        else:
            raise ValueError("No completed Terraform code available for validation")
        
        if not terraform_code:
            raise ValueError("Terraform code is empty")
        
        # Create validation state
        validation_state = ValidationState(
            terraform_code=terraform_code,
            validation_started_at=datetime.utcnow()
        )
        
        logger.info(f"Transformed supervisor state to validation state for workflow {supervisor_state.workflow_id}")
        return validation_state
        
    except Exception as e:
        logger.error(f"Failed to transform supervisor state to validation state: {e}")
        raise


def validation_to_supervisor_state(supervisor_state: SupervisorState, validation_state: ValidationState) -> SupervisorState:
    """
    Merge validation results back into supervisor state.
    
    This function takes the completed validation state and merges it back
    into the supervisor state, updating routing history and audit logs.
    
    Args:
        supervisor_state: The current supervisor state
        validation_state: The completed validation state
        
    Returns:
        Updated SupervisorState with validation results merged
        
    Raises:
        ValueError: If validation state is invalid
    """
    try:
        # Validate validation state
        if not validation_state.terraform_code:
            raise ValueError("Validation state missing required terraform_code field")
        
        # Create a copy of supervisor state to avoid mutations
        updated_supervisor = supervisor_state.model_copy(deep=True)
        
        # Set validation state
        updated_supervisor.validation_state = validation_state
        
        # Update current step and status
        updated_supervisor.current_step = "validation"
        if validation_state.validation_complete:
            # Check if validation passed
            if validation_state.overall_score >= 80.0 and not validation_state.validation_errors:
                updated_supervisor.status = WorkflowStatus.COMPLETED
            else:
                updated_supervisor.status = WorkflowStatus.FAILED
        else:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        
        # Add routing history entry
        routing_details = {
            "agent_type": AgentType.VALIDATION,
            "overall_score": validation_state.overall_score,
            "validation_errors": len(validation_state.validation_errors),
            "validation_warnings": len(validation_state.validation_warnings),
            "security_issues": len(validation_state.security_scan_results.get("issues", [])),
            "compliance_status": validation_state.compliance_results.get("overall_status", "unknown")
        }
        updated_supervisor.add_routing_entry(
            "validation",
            "completed" if validation_state.validation_complete else "in_progress",
            routing_details
        )
        
        # Add audit entry
        audit_details = {
            "agent_type": AgentType.VALIDATION,
            "validation_complete": validation_state.validation_complete,
            "overall_score": validation_state.overall_score,
            "validation_errors": validation_state.validation_errors,
            "security_issues": len(validation_state.security_scan_results.get("issues", [])),
            "compliance_status": validation_state.compliance_results.get("overall_status", "unknown")
        }
        updated_supervisor.add_audit_entry("validation_completed", audit_details)
        
        # Update execution time if validation is complete
        if validation_state.validation_complete and validation_state.validation_started_at:
            execution_time = (datetime.utcnow() - validation_state.validation_started_at).total_seconds()
            updated_supervisor.agent_execution_times[AgentType.VALIDATION] = execution_time
        
        logger.info(f"Merged validation state back to supervisor state for workflow {supervisor_state.workflow_id}")
        return updated_supervisor
        
    except Exception as e:
        logger.error(f"Failed to merge validation state to supervisor state: {e}")
        raise


# Supervisor ↔ Editor Agent Transformations
def supervisor_to_editor_state(supervisor_state: SupervisorState) -> EditorState:
    """
    Transform supervisor state to editor agent input state.
    
    This function extracts existing Terraform code and modification requirements
    to create an EditorState suitable for the Editor Agent subgraph.
    
    Args:
        supervisor_state: The current supervisor state
        
    Returns:
        EditorState configured for editor agent input
        
    Raises:
        ValueError: If no existing code or modifications are available
    """
    try:
        # Determine source of original code
        original_code = {}
        
        if supervisor_state.generation_state and supervisor_state.generation_state.generation_complete:
            original_code = supervisor_state.generation_state.terraform_code
            logger.info("Using Terraform code from generation state for editing")
        elif supervisor_state.editor_state and supervisor_state.editor_state.editor_complete:
            original_code = supervisor_state.editor_state.modified_code
            logger.info("Using Terraform code from previous editor state for editing")
        else:
            raise ValueError("No existing Terraform code available for editing")
        
        if not original_code:
            raise ValueError("Original Terraform code is empty")
        
        # Extract modifications from user request or analysis
        modifications = {}
        if supervisor_state.analysis_state and supervisor_state.analysis_state.requirements:
            modifications = supervisor_state.analysis_state.requirements.get("modifications", {})
        
        if not modifications:
            # Fallback to user request as modification context
            modifications = {"user_request": supervisor_state.user_request}
        
        # Create editor state
        editor_state = EditorState(
            original_code=original_code,
            modifications=modifications,
            editing_started_at=datetime.utcnow()
        )
        
        logger.info(f"Transformed supervisor state to editor state for workflow {supervisor_state.workflow_id}")
        return editor_state
        
    except Exception as e:
        logger.error(f"Failed to transform supervisor state to editor state: {e}")
        raise


def editor_to_supervisor_state(supervisor_state: SupervisorState, editor_state: EditorState) -> SupervisorState:
    """
    Merge editor results back into supervisor state.
    
    This function takes the completed editor state and merges it back
    into the supervisor state, updating routing history and audit logs.
    
    Args:
        supervisor_state: The current supervisor state
        editor_state: The completed editor state
        
    Returns:
        Updated SupervisorState with editor results merged
        
    Raises:
        ValueError: If editor state is invalid
    """
    try:
        # Validate editor state
        if not editor_state.original_code:
            raise ValueError("Editor state missing required original_code field")
        
        # Create a copy of supervisor state to avoid mutations
        updated_supervisor = supervisor_state.model_copy(deep=True)
        
        # Set editor state
        updated_supervisor.editor_state = editor_state
        
        # Update current step and status
        updated_supervisor.current_step = "editor"
        if editor_state.editor_complete:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        else:
            updated_supervisor.status = WorkflowStatus.IN_PROGRESS
        
        # Add routing history entry
        routing_details = {
            "agent_type": AgentType.EDITOR,
            "files_modified": len(editor_state.modified_code),
            "surgical_changes": len(editor_state.surgical_changes),
            "errors": editor_state.editor_errors,
            "backup_created": editor_state.backup_created
        }
        updated_supervisor.add_routing_entry(
            "editor",
            "completed" if editor_state.editor_complete else "in_progress",
            routing_details
        )
        
        # Add audit entry
        audit_details = {
            "agent_type": AgentType.EDITOR,
            "editor_complete": editor_state.editor_complete,
            "files_modified": len(editor_state.modified_code),
            "surgical_changes": editor_state.surgical_changes,
            "backup_created": editor_state.backup_created,
            "change_summary": editor_state.change_summary
        }
        updated_supervisor.add_audit_entry("editor_completed", audit_details)
        
        # Update execution time if editing is complete
        if editor_state.editor_complete and editor_state.editing_started_at:
            execution_time = (datetime.utcnow() - editor_state.editing_started_at).total_seconds()
            updated_supervisor.agent_execution_times[AgentType.EDITOR] = execution_time
        
        logger.info(f"Merged editor state back to supervisor state for workflow {supervisor_state.workflow_id}")
        return updated_supervisor
        
    except Exception as e:
        logger.error(f"Failed to merge editor state to supervisor state: {e}")
        raise


# Utility function to get transformation function pairs
def get_transformation_functions(agent_type: AgentType) -> tuple:
    """
    Get the input and output transformation functions for a given agent type.
    
    Args:
        agent_type: The type of agent to get transformations for
        
    Returns:
        Tuple of (input_transformer, output_transformer) functions
        
    Raises:
        ValueError: If agent type is not supported
    """
    transformation_map = {
        AgentType.ANALYSIS: (supervisor_to_analysis_state, analysis_to_supervisor_state),
        AgentType.GENERATION: (supervisor_to_generation_state, generation_to_supervisor_state),
        AgentType.VALIDATION: (supervisor_to_validation_state, validation_to_supervisor_state),
        AgentType.EDITOR: (supervisor_to_editor_state, editor_to_supervisor_state),
    }
    
    if agent_type not in transformation_map:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return transformation_map[agent_type] 