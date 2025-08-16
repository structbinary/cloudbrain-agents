"""
State schema definitions for AWS Orchestrator Agent Supervisor.

This module defines Pydantic-based state schemas for the Supervisor Agent
and all specialized agent subgraphs, ensuring type safety and validation
across the multi-agent orchestration system.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from langgraph.graph.message import add_messages


# Enums for type safety
class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_APPROVAL = "human_approval"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Agent type enumeration."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    EDITOR = "editor"


# Base state model with common configuration
class BaseState(BaseModel):
    """Base state model with common configuration."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Prevent additional fields
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


# Specialized Agent State Schemas
class AnalysisState(BaseState):
    """State schema for Analysis Agent subgraph."""
    
    # Core analysis data
    query: str = Field(description="User query or request to analyze")
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of conversation interactions"
    )
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted requirements from analysis"
    )
    aws_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="AWS context and resource information"
    )
    
    # Analysis metadata
    analysis_complete: bool = Field(
        default=False,
        description="Whether analysis is complete"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of analysis results"
    )
    analysis_errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during analysis"
    )
    
    # Timestamps
    analysis_started_at: Optional[datetime] = Field(
        default=None,
        description="When analysis started"
    )
    analysis_completed_at: Optional[datetime] = Field(
        default=None,
        description="When analysis completed"
    )


class GenerationState(BaseState):
    """State schema for Generation Agent subgraph."""
    
    # Core generation data
    module_name: str = Field(description="Name of the Terraform module to generate")
    requirements: Dict[str, Any] = Field(
        description="Requirements for module generation"
    )
    generated_files: List[str] = Field(
        default_factory=list,
        description="List of generated file paths"
    )
    terraform_code: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated Terraform code by file path"
    )
    
    # Generation metadata
    generation_complete: bool = Field(
        default=False,
        description="Whether generation is complete"
    )
    generation_errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during generation"
    )
    best_practices_applied: List[str] = Field(
        default_factory=list,
        description="Best practices applied during generation"
    )
    
    # Timestamps
    generation_started_at: Optional[datetime] = Field(
        default=None,
        description="When generation started"
    )
    generation_completed_at: Optional[datetime] = Field(
        default=None,
        description="When generation completed"
    )


class ValidationState(BaseState):
    """State schema for Validation Agent subgraph."""
    
    # Core validation data
    terraform_code: Dict[str, str] = Field(
        description="Terraform code to validate"
    )
    validation_reports: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation reports from different validation steps"
    )
    security_scan_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Security scanning results"
    )
    compliance_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Compliance checking results"
    )
    
    # Validation metadata
    validation_complete: bool = Field(
        default=False,
        description="Whether validation is complete"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during validation"
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from validation process"
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall validation score (0-100)"
    )
    
    # Timestamps
    validation_started_at: Optional[datetime] = Field(
        default=None,
        description="When validation started"
    )
    validation_completed_at: Optional[datetime] = Field(
        default=None,
        description="When validation completed"
    )


class EditorState(BaseState):
    """State schema for Editor Agent subgraph."""
    
    # Core editing data
    original_code: Dict[str, str] = Field(
        description="Original Terraform code before modifications"
    )
    modifications: Dict[str, Any] = Field(
        description="Requested modifications to apply"
    )
    modified_code: Dict[str, str] = Field(
        default_factory=dict,
        description="Modified Terraform code after applying changes"
    )
    surgical_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed list of surgical changes applied"
    )
    
    # Editing metadata
    editor_complete: bool = Field(
        default=False,
        description="Whether editing is complete"
    )
    editor_errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during editing"
    )
    change_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of changes made"
    )
    backup_created: bool = Field(
        default=False,
        description="Whether backup of original code was created"
    )
    
    # Timestamps
    editing_started_at: Optional[datetime] = Field(
        default=None,
        description="When editing started"
    )
    editing_completed_at: Optional[datetime] = Field(
        default=None,
        description="When editing completed"
    )


class PlannerState(BaseState):
    """State for Planner Agent subgraph (dummy implementation for testing)."""
    query: str = Field(default="")
    plan: Dict[str, Any] = Field(default_factory=dict)
    steps: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    status: str = Field(default="pending")
    error_context: Dict[str, Any] = Field(default_factory=dict)


# Main Supervisor State Schema
class SupervisorState(BaseState):
    """Main state schema for Supervisor Agent orchestration."""
    
    # Message tracking for LangGraph
    messages: Annotated[List[Any], add_messages] = Field(
        default_factory=list,
        description="Message history for LangGraph state tracking"
    )
    
    # Core workflow state
    workflow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the workflow"
    )
    current_step: str = Field(
        default="initialized",
        description="Current step in the workflow"
    )
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Current workflow status"
    )
    
    # Request context
    user_request: str = Field(
        description="Original user request or query"
    )
    context_id: Optional[str] = Field(
        default=None,
        description="Context identifier for tracking and state transfer"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task identifier for tracking and state transfer"
    )
    mcp_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="MCP context and external tool information"
    )
    
    # Agent subgraph states (namespaced)
    analysis_state: Optional[AnalysisState] = Field(
        default=None,
        description="Analysis Agent subgraph state"
    )
    generation_state: Optional[GenerationState] = Field(
        default=None,
        description="Generation Agent subgraph state"
    )
    validation_state: Optional[ValidationState] = Field(
        default=None,
        description="Validation Agent subgraph state"
    )
    editor_state: Optional[EditorState] = Field(
        default=None,
        description="Editor Agent subgraph state"
    )
    
    # Orchestration metadata
    routing_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of routing decisions and agent invocations"
    )
    error_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Error context and recovery information"
    )
    audit_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audit log for compliance and debugging"
    )
    
    # Human-in-the-loop
    human_approval_required: bool = Field(
        default=False,
        description="Whether human approval is required"
    )
    approval_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Context for human approval decision"
    )
    
    # Performance and monitoring
    total_execution_time: float = Field(
        default=0.0,
        description="Total execution time in seconds"
    )
    agent_execution_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution times for each agent"
    )
    
    # Timestamps
    workflow_started_at: Optional[datetime] = Field(
        default=None,
        description="When workflow started"
    )
    workflow_completed_at: Optional[datetime] = Field(
        default=None,
        description="When workflow completed"
    )
    
    def get_agent_state(self, agent_type: AgentType) -> Optional[BaseState]:
        """Get the state for a specific agent type."""
        state_map = {
            AgentType.ANALYSIS: self.analysis_state,
            AgentType.GENERATION: self.generation_state,
            AgentType.VALIDATION: self.validation_state,
            AgentType.EDITOR: self.editor_state,
        }
        return state_map.get(agent_type)
    
    def set_agent_state(self, agent_type: AgentType, state: BaseState) -> None:
        """Set the state for a specific agent type."""
        if agent_type == AgentType.ANALYSIS:
            self.analysis_state = state
        elif agent_type == AgentType.GENERATION:
            self.generation_state = state
        elif agent_type == AgentType.VALIDATION:
            self.validation_state = state
        elif agent_type == AgentType.EDITOR:
            self.editor_state = state
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def add_routing_entry(self, step: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add a routing history entry."""
        entry = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
        }
        if details:
            entry.update(details)
        self.routing_history.append(entry)
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add an audit log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **details
        }
        self.audit_log.append(entry)
    
    def is_workflow_complete(self) -> bool:
        """Check if the workflow is complete."""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
    
    def get_completion_percentage(self) -> float:
        """Get workflow completion percentage."""
        if self.is_workflow_complete():
            return 100.0
        
        # Calculate based on completed agent states
        completed_agents = 0
        total_agents = 4  # analysis, generation, validation, editor
        
        if self.analysis_state and self.analysis_state.analysis_complete:
            completed_agents += 1
        if self.generation_state and self.generation_state.generation_complete:
            completed_agents += 1
        if self.validation_state and self.validation_state.validation_complete:
            completed_agents += 1
        if self.editor_state and self.editor_state.editor_complete:
            completed_agents += 1
        
        return (completed_agents / total_agents) * 100.0 