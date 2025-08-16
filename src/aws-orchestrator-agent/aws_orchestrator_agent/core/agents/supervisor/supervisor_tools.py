"""
Tool definitions for the Supervisor Agent.

This module contains tool definitions that enable the Supervisor Agent
to route requests to specialized agent subgraphs using LangGraph's
tool-calling pattern.
"""

from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

@tool
def call_analysis_agent(
    query: str, 
    context: Optional[Dict[str, Any]] = None,
    workflow_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    Route to Analysis Agent for requirements analysis and conversation management.
    
    Args:
        query: The user query or request to analyze
        context: Additional context information (AWS resources, existing infrastructure, etc.)
        workflow_state: Current workflow state and conversation history
        
    Returns:
        Analysis Agent response with requirements, context, and next steps
    """
    logger.info(f"Routing to Analysis Agent: {query[:100]}...")
    
    # This will be replaced with actual subgraph invocation
    # For now, return a placeholder response
    return f"Analysis Agent will process: {query}. Context: {context}, State: {workflow_state}"

@tool
def call_generation_agent(
    requirements: Dict[str, Any],
    aws_context: Optional[Dict[str, Any]] = None,
    patterns: Optional[List[str]] = None
) -> str:
    """
    Route to Generation Agent for creating new Terraform modules.
    
    Args:
        requirements: Detailed requirements for the Terraform module
        aws_context: AWS-specific context and constraints
        patterns: Enterprise patterns and best practices to apply
        
    Returns:
        Generated Terraform code and module structure
    """
    logger.info(f"Routing to Generation Agent with requirements: {list(requirements.keys())}")
    
    # This will be replaced with actual subgraph invocation
    # For now, return a placeholder response
    return f"Generation Agent will create Terraform module based on requirements: {requirements}"

@tool
def call_validation_agent(
    terraform_code: str,
    validation_stages: Optional[List[str]] = None,
    compliance_policies: Optional[Dict[str, Any]] = None
) -> str:
    """
    Route to Validation Agent for comprehensive validation.
    
    Args:
        terraform_code: The Terraform code to validate
        validation_stages: Specific validation stages to run (syntax, plan, security, compliance)
        compliance_policies: Organizational compliance policies to check against
        
    Returns:
        Validation results and recommendations
    """
    logger.info(f"Routing to Validation Agent for {len(terraform_code)} characters of Terraform code")
    
    # This will be replaced with actual subgraph invocation
    # For now, return a placeholder response
    return f"Validation Agent will validate Terraform code with stages: {validation_stages}"

@tool
def call_editor_agent(
    terraform_code: str,
    modifications: Dict[str, Any],
    state_analysis: Optional[Dict[str, Any]] = None
) -> str:
    """
    Route to Editor Agent for modifying existing Terraform configurations.
    
    Args:
        terraform_code: Existing Terraform code to modify
        modifications: Specific modifications to apply
        state_analysis: Analysis of current state and implications
        
    Returns:
        Modified Terraform code with surgical updates
    """
    logger.info(f"Routing to Editor Agent for modifications: {list(modifications.keys())}")
    
    # This will be replaced with actual subgraph invocation
    # For now, return a placeholder response
    return f"Editor Agent will apply modifications: {modifications} to existing code"

@tool
def request_human_approval(
    action: str,
    details: Dict[str, Any],
    risks: Optional[List[str]] = None,
    alternatives: Optional[List[str]] = None
) -> str:
    """
    Request human approval for critical actions.
    
    Args:
        action: The action requiring approval (e.g., "terraform_apply", "security_violation")
        details: Detailed information about the action
        risks: Potential risks and implications
        alternatives: Alternative approaches if available
        
    Returns:
        Human approval decision and feedback
    """
    logger.info(f"Requesting human approval for action: {action}")
    
    approval_request = {
        "action": action,
        "details": details,
        "risks": risks or [],
        "alternatives": alternatives or [],
        "status": "pending_approval"
    }
    
    return f"Human approval requested for {action}. Details: {details}"

@tool
def handle_error_recovery(
    error_type: str,
    error_details: Dict[str, Any],
    failed_agent: str,
    retry_count: int = 0
) -> str:
    """
    Handle error recovery and fallback logic.
    
    Args:
        error_type: Type of error encountered
        error_details: Detailed error information
        failed_agent: Name of the agent that failed
        retry_count: Number of retry attempts made
        
    Returns:
        Recovery action and next steps
    """
    logger.warning(f"Error recovery for {failed_agent}: {error_type}")
    
    if retry_count < 2:
        return f"Retrying {failed_agent} operation (attempt {retry_count + 1})"
    else:
        return f"Escalating to human: {failed_agent} failed after {retry_count} retries"

@tool
def log_routing_decision(
    decision: str,
    agent_name: str,
    reasoning: str,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log routing decisions for audit and debugging.
    
    Args:
        decision: The routing decision made
        agent_name: Name of the agent selected
        reasoning: Reasoning behind the decision
        context: Additional context for the decision
        
    Returns:
        Confirmation of logging
    """
    logger.info(f"Routing decision: {decision} -> {agent_name}. Reasoning: {reasoning}")
    
    return f"Logged routing decision: {decision} -> {agent_name}"

# Tool registry for easy access
SUPERVISOR_TOOLS = {
    "call_analysis_agent": call_analysis_agent,
    "call_generation_agent": call_generation_agent,
    "call_validation_agent": call_validation_agent,
    "call_editor_agent": call_editor_agent,
    "request_human_approval": request_human_approval,
    "handle_error_recovery": handle_error_recovery,
    "log_routing_decision": log_routing_decision
} 