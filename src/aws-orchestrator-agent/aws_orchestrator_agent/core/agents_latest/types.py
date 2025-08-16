"""
State schema definitions for the custom supervisor agent system.

This module defines the Pydantic-based state schemas for the Supervisor Agent
and all specialized agent subgraphs, following LangGraph multi-agent best practices
with minimal state overlap and external artifact references.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional, Any, Union, Annotated, AsyncGenerator
from enum import Enum
from datetime import datetime, timezone
import uuid

# LangGraph imports for proper message handling
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages




class AgentResponse(BaseModel):
    """
    Response from an agent during execution.
    
    This represents a single response item from the agent's stream,
    containing the content, metadata, and control flags.
    """
    
    model_config = ConfigDict(extra="allow")
    
    content: Any = Field(..., description="The response content (text or data)")
    response_type: str = Field(default="text", description="Type of response: 'text' or 'data'")
    is_task_complete: bool = Field(default=False, description="Whether this response indicates task completion")
    require_user_input: bool = Field(default=False, description="Whether this response requires user input to continue")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
    root: Optional[Any] = Field(default=None, description="Root object for A2A protocol integration")


class BaseAgent(ABC):
    """
    Base interface for all AWS Orchestrator Agent implementations.
    
    This abstract base class defines the contract that all agent implementations
    must follow to work with the A2A protocol integration.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            The agent's name
        """
        pass
    
    @abstractmethod
    async def stream(
        self, 
        query: str, 
        context_id: str, 
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Stream responses for a given query.
        
        This method should implement the core agent logic and yield
        AgentResponse objects as the agent processes the query.
        
        Args:
            query: The user query to process
            context_id: The A2A context ID
            task_id: The A2A task ID
            
        Yields:
            AgentResponse objects representing the agent's progress
        """
        pass
    
    async def initialize(self) -> None:
        """
        Initialize the agent.
        
        This method can be overridden to perform any initialization
        required by the agent implementation.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the agent.
        
        This method can be overridden to perform any cleanup
        required by the agent implementation.
        """
        pass


# ============================================================================
# ENUMS
# ============================================================================

class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_APPROVAL = "human_approval"
    ROLLED_BACK = "rolled_back"
    INTERRUPTED = "interrupted"

class AgentType(str, Enum):
    """Agent type enumeration."""
    PLANNER = "planner"
    GENERATION = "generation"
    VALIDATION = "validation"
    EDITOR = "editor"
    SECURITY = "security"
    COST = "cost"

class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WARNING = "warning"

class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# SUPERVISOR STATE SCHEMA
# ============================================================================

class SupervisorState(BaseModel):
    """
    Supervisor state following LangGraph best practices.
    
    Uses Annotated[list, add_messages] for proper message handling and
    extends with infrastructure orchestration fields.
    """
    
    # Core LangGraph-style state with proper message handling
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    
    # Minimal workflow metadata
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_agent: Optional[AgentType] = None
    
    # User context (essential for our use case)
    user_request: str
    
    # Infrastructure artifacts (essential for Terraform orchestration)
    workspace_ref: Optional[str] = None  # URI to workspace
    terraform_context: Optional[Dict[str, Any]] = None  # Terraform-specific context
    generated_module_ref: Optional[str] = None  # URI to generated module
    validation_report_ref: Optional[str] = None  # URI to validation report
    security_report_ref: Optional[str] = None  # URI to security scan
    cost_report_ref: Optional[str] = None  # URI to cost analysis
    
    # Human-in-the-loop (essential for approval workflow)
    human_approval_required: bool = False
    approval_context: Optional[Dict[str, Any]] = None
    approval_timeout: Optional[datetime] = None
    
    # Error handling (simplified)
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Minimal audit trail (for compliance)
    workflow_started_at: Optional[datetime] = None
    workflow_completed_at: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# PLANNER AGENT STATE SCHEMA
# ============================================================================

class PlannerState(BaseModel):
    """
    State schema for the Planner Agent subgraph.
    
    Responsible for requirements analysis, dependency mapping, and execution planning.
    Extends MessagesState pattern for proper message handling.
    """
    
    # Core LangGraph-style state with proper message handling
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    
    # Input from supervisor
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    user_request: str
    mcp_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Planning process
    requirements_analysis: Dict[str, Any] = Field(default_factory=dict)
    infrastructure_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    architectural_patterns: List[str] = Field(default_factory=list)
    security_requirements: List[str] = Field(default_factory=list)
    compliance_requirements: List[str] = Field(default_factory=list)
    
    # Dependency mapping
    dependency_mapping_complete: bool = False
    dependency_questions: List[str] = Field(default_factory=list)
    dependency_answers: Dict[str, Any] = Field(default_factory=dict)
    waiting_for_dependency_input: bool = False
    current_dependency_question: Optional[str] = None
    
    # Execution plan
    execution_plan: List[Dict[str, Any]] = Field(default_factory=list)
    complexity_score: int = 0
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    
    # Internal working state
    planning_metadata: Dict[str, Any] = Field(default_factory=dict)
    planning_started_at: Optional[datetime] = None
    planning_completed_at: Optional[datetime] = None
    planning_duration: Optional[float] = None
    
    # Status and workflow tracking
    status: str = "initialized"
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    request_type: Optional[str] = None
    error: Optional[str] = None
    
    # Additional fields for planning process
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    cost_analysis: Dict[str, Any] = Field(default_factory=dict)
    validation_criteria: List[str] = Field(default_factory=list)
    
    # Approval and interrupt handling
    requires_approval: bool = False
    approval_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    interrupt_required: bool = False
    interrupt_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Output to supervisor
    planning_complete: bool = False
    next_agent: Optional[AgentType] = None
    handoff_context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# GENERATION AGENT STATE SCHEMA
# ============================================================================

class GenerationState(BaseModel):
    """
    State schema for the Generation Agent subgraph.
    
    Responsible for creating new Terraform modules from scratch with best practices.
    """
    
    # Input from supervisor/planner
    requirements: Dict[str, Any] = Field(default_factory=dict)
    provider_versions: Dict[str, str] = Field(default_factory=dict)
    registry_schemas_ref: Optional[str] = None  # URI to registry schemas
    standards_profile: Dict[str, Any] = Field(default_factory=dict)
    
    # Generation process
    design_rationale: Optional[str] = None
    file_plan: List[Dict[str, Any]] = Field(default_factory=list)
    template_params: Dict[str, Any] = Field(default_factory=dict)
    generated_files_ref: Optional[str] = None  # URI to generated files
    
    # Module metadata
    module_name: str
    module_version: str = "1.0.0"
    module_manifest: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality checks
    warnings: List[str] = Field(default_factory=list)
    best_practices_applied: List[str] = Field(default_factory=list)
    
    # Output to supervisor
    generation_complete: bool = False
    generated_module_ref: Optional[str] = None  # URI to final module
    module_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# VALIDATION AGENT STATE SCHEMA
# ============================================================================

class ValidationState(BaseModel):
    """
    State schema for the Validation Agent subgraph.
    
    Responsible for comprehensive validation including syntax, plan, security, and compliance.
    """
    
    # Input from supervisor
    module_ref: Optional[str] = None  # URI to module to validate
    workspace_ref: Optional[str] = None  # URI to workspace
    policy_sets: List[str] = Field(default_factory=list)
    quota_profile: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation stages (parallel execution)
    static_analysis: Dict[str, Any] = Field(default_factory=dict)
    terraform_validate: Dict[str, Any] = Field(default_factory=dict)
    terraform_plan: Dict[str, Any] = Field(default_factory=dict)
    security_scans: Dict[str, Any] = Field(default_factory=dict)
    compliance_checks: Dict[str, Any] = Field(default_factory=dict)
    quota_evaluation: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation results
    validation_report_ref: Optional[str] = None  # URI to full report
    status: ValidationStatus = ValidationStatus.PENDING
    blockers: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Parallel execution tracking
    completed_stages: List[str] = Field(default_factory=list)
    failed_stages: List[str] = Field(default_factory=list)
    
    # Output to supervisor
    validation_complete: bool = False
    validation_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# EDITOR AGENT STATE SCHEMA
# ============================================================================

class EditorState(BaseModel):
    """
    State schema for the Editor Agent subgraph.
    
    Responsible for modifying existing Terraform configurations with surgical precision.
    """
    
    # Input from supervisor
    target_config_ref: Optional[str] = None  # URI to config to edit
    change_request: Dict[str, Any] = Field(default_factory=dict)
    state_diff_context: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph: Dict[str, Any] = Field(default_factory=dict)
    
    # Editing process
    ast_notes: Dict[str, Any] = Field(default_factory=dict)
    formatting_profile: Dict[str, Any] = Field(default_factory=dict)
    compatibility_findings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Surgical changes
    surgical_changes: List[Dict[str, Any]] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    patch_ref: Optional[str] = None  # URI to patch/diff
    
    # Migration and risk
    migration_notes: List[str] = Field(default_factory=list)
    apply_risk: RiskLevel = RiskLevel.LOW
    rollback_strategy: Optional[str] = None
    
    # Output to supervisor
    editor_complete: bool = False
    modified_config_ref: Optional[str] = None  # URI to modified config
    change_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# SECURITY AGENT STATE SCHEMA
# ============================================================================

class SecurityState(BaseModel):
    """
    State schema for the Security Agent subgraph.
    
    Responsible for security analysis, IAM analysis, and compliance checking.
    """
    
    # Input from supervisor
    module_ref: Optional[str] = None  # URI to module to analyze
    security_policies: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)
    
    # Security analysis
    iam_analysis: Dict[str, Any] = Field(default_factory=dict)
    network_security: Dict[str, Any] = Field(default_factory=dict)
    data_protection: Dict[str, Any] = Field(default_factory=dict)
    vulnerability_scan: Dict[str, Any] = Field(default_factory=dict)
    
    # Compliance checks
    compliance_results: Dict[str, Any] = Field(default_factory=dict)
    policy_violations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Security findings
    security_findings: List[Dict[str, Any]] = Field(default_factory=list)
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    required_actions: List[str] = Field(default_factory=list)
    
    # Output to supervisor
    security_complete: bool = False
    security_report_ref: Optional[str] = None  # URI to security report
    security_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# COST AGENT STATE SCHEMA
# ============================================================================

class CostState(BaseModel):
    """
    State schema for the Cost Agent subgraph.
    
    Responsible for cost estimation, forecasting, and optimization.
    """
    
    # Input from supervisor
    plan_json_ref: Optional[str] = None  # URI to Terraform plan JSON
    region: str = "us-east-1"
    pricing_cache_hint: Optional[str] = None
    
    # Cost analysis
    monthly_estimate: Optional[float] = None
    annual_forecast: Optional[float] = None
    cost_breakdown: Dict[str, Any] = Field(default_factory=dict)
    resource_costs: Dict[str, float] = Field(default_factory=dict)
    
    # Optimization
    optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    potential_savings: Optional[float] = None
    budget_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cost tracking
    cost_history: List[Dict[str, Any]] = Field(default_factory=list)
    trend_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Output to supervisor
    cost_complete: bool = False
    cost_report_ref: Optional[str] = None  # URI to cost report
    cost_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# ============================================================================
# TYPE ALIASES AND UTILITIES
# ============================================================================

# Type aliases for common patterns
AgentState = Union[PlannerState, GenerationState, ValidationState, EditorState, SecurityState, CostState]
StateDict = Dict[str, Any]

# LangGraph-compatible state type
MessagesState = Dict[str, Any]  # Simple alias for LangGraph's MessagesState

# Utility function to get state class by agent type
def get_state_class(agent_type: AgentType) -> type:
    """Get the state class for a given agent type."""
    state_classes = {
        AgentType.PLANNER: PlannerState,
        AgentType.GENERATION: GenerationState,
        AgentType.VALIDATION: ValidationState,
        AgentType.EDITOR: EditorState,
        AgentType.SECURITY: SecurityState,
        AgentType.COST: CostState,
    }
    return state_classes.get(agent_type, SupervisorState)

# Utility function to create state reference URI
def create_state_ref(agent_type: AgentType, workflow_id: str) -> str:
    """Create a state reference URI for an agent."""
    return f"{agent_type.value}_state_{workflow_id}"

# LangGraph-style utility functions
def create_supervisor_state(
    user_request: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    mcp_context: Optional[Dict[str, Any]] = None
) -> SupervisorState:
    """Create a new supervisor state following LangGraph patterns."""
    return SupervisorState(
        user_request=user_request,
        session_id=session_id,
        task_id=task_id,
        mcp_context=mcp_context or {},
        workflow_started_at=datetime.now(timezone.utc)
    )

def add_message_to_state(state: SupervisorState, role: str, content: str, **kwargs) -> SupervisorState:
    """Add a message to the supervisor state (LangGraph-style)."""
    message = {"role": role, "content": content, **kwargs}
    state.messages.append(message)
    return state

def get_last_message(state: SupervisorState) -> Optional[Dict[str, Any]]:
    """Get the last message from the state."""
    return state.messages[-1] if state.messages else None

def is_human_approval_required(state: SupervisorState) -> bool:
    """Check if human approval is required."""
    return state.human_approval_required

def set_approval_required(state: SupervisorState, context: Dict[str, Any]) -> SupervisorState:
    """Set human approval as required."""
    state.human_approval_required = True
    state.approval_context = context
    state.approval_timeout = datetime.utcnow()
    return state

def clear_approval_required(state: SupervisorState) -> SupervisorState:
    """Clear human approval requirement."""
    state.human_approval_required = False
    state.approval_context = None
    state.approval_timeout = None
    return state

# Export all state classes
__all__ = [
    # Enums
    "WorkflowStatus",
    "AgentType", 
    "ValidationStatus",
    "RiskLevel",
    
    # State schemas
    "SupervisorState",
    "PlannerState",
    "GenerationState", 
    "ValidationState",
    "EditorState",
    "SecurityState",
    "CostState",
    
    # Type aliases
    "AgentState",
    "StateDict",
    "MessagesState",
    
    # Utility functions
    "get_state_class",
    "create_state_ref",
    "create_supervisor_state",
    "add_message_to_state",
    "get_last_message",
    "is_human_approval_required",
    "set_approval_required",
    "clear_approval_required",
]
