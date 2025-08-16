"""
Planner Agent State Schema.

This module defines the state schema for the Planner Agent, which is responsible for:
- Analyzing user requirements and existing infrastructure
- Creating detailed implementation plans
- Identifying AWS services and their interdependencies
- Planning architectural patterns and security considerations
- Generating step-by-step execution plans
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InfrastructureRequirement(BaseModel):
    """Represents a specific infrastructure requirement."""
    service: str = Field(description="AWS service name (e.g., 'ec2', 'rds', 's3')")
    purpose: str = Field(description="Purpose of this service in the architecture")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Service-specific configuration")
    dependencies: List[str] = Field(default_factory=list, description="List of dependent services")
    security_requirements: List[str] = Field(default_factory=list, description="Security requirements for this service")
    cost_estimate: Optional[float] = Field(default=None, description="Estimated monthly cost in USD")


class ArchitecturalPattern(BaseModel):
    """Represents an architectural pattern identified in the requirements."""
    pattern_type: str = Field(description="Type of pattern (e.g., 'microservices', 'event-driven', 'layered')")
    description: str = Field(description="Description of the pattern")
    components: List[str] = Field(description="Components that implement this pattern")
    benefits: List[str] = Field(description="Benefits of using this pattern")
    considerations: List[str] = Field(description="Important considerations for implementation")


class SecurityRequirement(BaseModel):
    """Represents a security requirement for the infrastructure."""
    category: str = Field(description="Security category (e.g., 'authentication', 'encryption', 'network')")
    requirement: str = Field(description="Specific security requirement")
    implementation: str = Field(description="How to implement this requirement")
    priority: str = Field(description="Priority level (high, medium, low)")


class ExecutionStep(BaseModel):
    """Represents a step in the execution plan."""
    step_number: int = Field(description="Sequential step number")
    action: str = Field(description="Action to perform")
    description: str = Field(description="Detailed description of the step")
    dependencies: List[int] = Field(default_factory=list, description="Steps that must be completed first")
    estimated_time: Optional[str] = Field(default=None, description="Estimated time to complete")
    resources: List[str] = Field(default_factory=list, description="Resources needed for this step")
    validation_criteria: List[str] = Field(default_factory=list, description="Criteria to validate step completion")


class PlannerState(BaseModel):
    """
    State schema for the Planner Agent.
    
    This state tracks the planning process, including requirements analysis,
    architectural decisions, and execution planning.
    """
    
    # Core messages for LangGraph state tracking
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Message history for LangGraph state tracking"
    )
    
    # Agent identification
    agent_name: str = Field(default="planner_agent", description="Name of the planner agent")
    status: str = Field(default="initialized", description="Current status of the planner")
    
    # Input and context
    user_request: Optional[str] = Field(default="", description="Original user request for infrastructure planning")
    context_id: Optional[str] = Field(default=None, description="Context identifier for tracking")
    task_id: Optional[str] = Field(default=None, description="Task identifier for tracking")
    
    # Request classification
    request_type: str = Field(default="new", description="Type of request: 'new' or 'modify'")
    request_classification_confidence: Optional[float] = Field(default=None, description="Confidence score for request classification")
    
    # Analysis results
    requirements_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results of requirements analysis"
    )
    
    # Infrastructure planning
    infrastructure_requirements: List[InfrastructureRequirement] = Field(
        default_factory=list,
        description="List of identified infrastructure requirements"
    )
    
    architectural_patterns: List[ArchitecturalPattern] = Field(
        default_factory=list,
        description="Identified architectural patterns"
    )
    
    security_requirements: List[SecurityRequirement] = Field(
        default_factory=list,
        description="Security requirements for the infrastructure"
    )
    
    # Execution planning
    execution_plan: List[ExecutionStep] = Field(
        default_factory=list,
        description="Step-by-step execution plan"
    )
    
    # Cost and resource analysis
    cost_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cost analysis and estimates"
    )
    
    resource_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource requirements and constraints"
    )
    
    # Validation and compliance
    compliance_requirements: List[str] = Field(
        default_factory=list,
        description="Compliance requirements (e.g., SOC2, HIPAA, GDPR)"
    )
    
    validation_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for validating the planned infrastructure"
    )
    
    # Planning metadata
    planning_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the planning process"
    )
    
    # Existing state analysis (for modifications)
    existing_infrastructure: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Current infrastructure state for modification requests"
    )
    affected_resources: List[str] = Field(
        default_factory=list, 
        description="Resources that will be modified"
    )
    unchanged_resources: List[str] = Field(
        default_factory=list, 
        description="Resources that remain unchanged"
    )
    
    # Change impact analysis
    change_impact: Dict[str, Any] = Field(
        default_factory=dict,
        description="Impact analysis of proposed changes"
    )
    downtime_required: bool = Field(
        default=False, 
        description="Whether changes require downtime"
    )
    rollback_strategy: Optional[str] = Field(
        default=None, 
        description="Rollback strategy for changes"
    )
    risk_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Risk assessment for the proposed changes"
    )
    
    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if planning fails")
    error_context: Optional[Dict[str, Any]] = Field(default=None, description="Context information for errors")
    
    # Execution tracking
    current_step: Optional[str] = Field(default=None, description="Current step in execution plan")
    completed_steps: List[str] = Field(default_factory=list, description="List of completed steps")
    
    # Approval and human-in-the-loop
    requires_approval: bool = Field(default=False, description="Whether human approval is required")
    approval_context: Optional[Dict[str, Any]] = Field(default=None, description="Context for approval requests")
    
    # Dependency mapping HITL
    dependency_questions: List[str] = Field(default_factory=list, description="Follow-up questions from dependency mapping")
    dependency_answers: Dict[str, str] = Field(default_factory=dict, description="User answers to dependency questions")
    waiting_for_dependency_input: bool = Field(default=False, description="Whether waiting for user input on dependencies")
    current_dependency_question: Optional[str] = Field(default=None, description="Current question being asked")
    dependency_mapping_complete: bool = Field(default=False, description="Whether dependency mapping is complete")
    
    # Timestamps
    planning_started_at: Optional[str] = Field(default=None, description="When planning started")
    planning_completed_at: Optional[str] = Field(default=None, description="When planning completed")
    
    # Performance metrics
    planning_duration: Optional[float] = Field(default=None, description="Planning duration in seconds")
    complexity_score: Optional[int] = Field(default=None, description="Complexity score (1-10)")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow" 