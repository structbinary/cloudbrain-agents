"""
State Transformation Functions for Supervisor Agent.

This module provides comprehensive state transformation functions for transferring
state between the Supervisor Agent and all specialized agent subgraphs, ensuring
proper context preservation and data flow.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .schemas import (
    SupervisorState, AnalysisState, GenerationState, 
    ValidationState, EditorState, PlannerState
)


class StateTransformer:
    """
    Comprehensive state transformation system for supervisor-agent communication.
    
    This class provides bidirectional state transformation between the Supervisor
    and all specialized agent subgraphs, ensuring proper context preservation,
    message handling, and data flow.
    """
    
    @staticmethod
    def supervisor_to_analysis_state(supervisor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor state to Analysis Agent state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            
        Returns:
            Dictionary containing Analysis Agent state fields
        """
        return {
            "messages": supervisor_state.get("messages", []),
            "user_request": supervisor_state.get("user_request", ""),
            "context_id": supervisor_state.get("context_id"),
            "task_id": supervisor_state.get("task_id"),
            "query": supervisor_state.get("user_request", ""),
            "conversation_history": supervisor_state.get("messages", []),
            "requirements": supervisor_state.get("analysis_state", {}).get("requirements", {}),
            "aws_context": supervisor_state.get("analysis_state", {}).get("aws_context", {}),
            "analysis_complete": supervisor_state.get("analysis_state", {}).get("analysis_complete", False),
            "confidence_score": supervisor_state.get("analysis_state", {}).get("confidence_score", 0.0),
            "analysis_errors": supervisor_state.get("analysis_state", {}).get("analysis_errors", []),
            "analysis_started_at": supervisor_state.get("analysis_state", {}).get("analysis_started_at"),
            "analysis_completed_at": supervisor_state.get("analysis_state", {}).get("analysis_completed_at"),
            "status": "analysis_started",
            "analysis_started_at": datetime.utcnow().isoformat() if not supervisor_state.get("analysis_state", {}).get("analysis_started_at") else None
        }
    
    @staticmethod
    def analysis_to_supervisor_state(supervisor_state: Dict[str, Any], analysis_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Analysis Agent state back to supervisor state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            analysis_state: Analysis Agent state dictionary
            
        Returns:
            Updated supervisor state dictionary
        """
        updated_state = supervisor_state.copy()
        
        # Update analysis-specific fields
        updated_state["analysis_state"] = {
            "query": analysis_state.get("query", ""),
            "conversation_history": analysis_state.get("conversation_history", []),
            "requirements": analysis_state.get("requirements", {}),
            "aws_context": analysis_state.get("aws_context", {}),
            "analysis_complete": analysis_state.get("analysis_complete", False),
            "confidence_score": analysis_state.get("confidence_score", 0.0),
            "analysis_errors": analysis_state.get("analysis_errors", []),
            "analysis_started_at": analysis_state.get("analysis_started_at"),
            "analysis_completed_at": analysis_state.get("analysis_completed_at")
        }
        
        # Update common fields
        updated_state["messages"] = analysis_state.get("messages", [])
        updated_state["user_request"] = analysis_state.get("user_request", "")
        updated_state["context_id"] = analysis_state.get("context_id")
        updated_state["task_id"] = analysis_state.get("task_id")
        
        return updated_state
    
    @staticmethod
    def supervisor_to_planner_state(supervisor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor state to Planner Agent state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            
        Returns:
            Dictionary containing Planner Agent state fields
        """
        return {
            "messages": supervisor_state.get("messages", []),
            "user_request": supervisor_state.get("user_request", ""),
            "context_id": supervisor_state.get("context_id"),
            "task_id": supervisor_state.get("task_id"),
            "agent_name": "planner_agent",
            "status": "planning_started",
            "request_type": "new",  # Will be classified by planner
            "requirements_analysis": supervisor_state.get("planner_state", {}).get("requirements_analysis", {}),
            "infrastructure_requirements": supervisor_state.get("planner_state", {}).get("infrastructure_requirements", []),
            "architectural_patterns": supervisor_state.get("planner_state", {}).get("architectural_patterns", []),
            "security_requirements": supervisor_state.get("planner_state", {}).get("security_requirements", []),
            "execution_plan": supervisor_state.get("planner_state", {}).get("execution_plan", []),
            "cost_analysis": supervisor_state.get("planner_state", {}).get("cost_analysis", {}),
            "resource_requirements": supervisor_state.get("planner_state", {}).get("resource_requirements", {}),
            "compliance_requirements": supervisor_state.get("planner_state", {}).get("compliance_requirements", []),
            "validation_criteria": supervisor_state.get("planner_state", {}).get("validation_criteria", []),
            "planning_metadata": supervisor_state.get("planner_state", {}).get("planning_metadata", {}),
            "existing_infrastructure": supervisor_state.get("planner_state", {}).get("existing_infrastructure"),
            "affected_resources": supervisor_state.get("planner_state", {}).get("affected_resources", []),
            "unchanged_resources": supervisor_state.get("planner_state", {}).get("unchanged_resources", []),
            "change_impact": supervisor_state.get("planner_state", {}).get("change_impact", {}),
            "downtime_required": supervisor_state.get("planner_state", {}).get("downtime_required", False),
            "rollback_strategy": supervisor_state.get("planner_state", {}).get("rollback_strategy"),
            "risk_assessment": supervisor_state.get("planner_state", {}).get("risk_assessment", {}),
            "error": supervisor_state.get("planner_state", {}).get("error"),
            "error_context": supervisor_state.get("planner_state", {}).get("error_context"),
            "current_step": supervisor_state.get("planner_state", {}).get("current_step"),
            "completed_steps": supervisor_state.get("planner_state", {}).get("completed_steps", []),
            "requires_approval": supervisor_state.get("planner_state", {}).get("requires_approval", False),
            "approval_context": supervisor_state.get("planner_state", {}).get("approval_context"),
            "planning_started_at": datetime.utcnow().isoformat(),
            "planning_completed_at": supervisor_state.get("planner_state", {}).get("planning_completed_at"),
            "planning_duration": supervisor_state.get("planner_state", {}).get("planning_duration"),
            "complexity_score": supervisor_state.get("planner_state", {}).get("complexity_score")
        }
    
    @staticmethod
    def planner_to_supervisor_state(supervisor_state: Dict[str, Any], planner_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Planner Agent state back to supervisor state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            planner_state: Planner Agent state dictionary
            
        Returns:
            Updated supervisor state dictionary
        """
        updated_state = supervisor_state.copy()
        
        # Update planner-specific fields
        updated_state["planner_state"] = {
            "requirements_analysis": planner_state.get("requirements_analysis", {}),
            "infrastructure_requirements": planner_state.get("infrastructure_requirements", []),
            "architectural_patterns": planner_state.get("architectural_patterns", []),
            "security_requirements": planner_state.get("security_requirements", []),
            "execution_plan": planner_state.get("execution_plan", []),
            "cost_analysis": planner_state.get("cost_analysis", {}),
            "resource_requirements": planner_state.get("resource_requirements", {}),
            "compliance_requirements": planner_state.get("compliance_requirements", []),
            "validation_criteria": planner_state.get("validation_criteria", []),
            "planning_metadata": planner_state.get("planning_metadata", {}),
            "existing_infrastructure": planner_state.get("existing_infrastructure"),
            "affected_resources": planner_state.get("affected_resources", []),
            "unchanged_resources": planner_state.get("unchanged_resources", []),
            "change_impact": planner_state.get("change_impact", {}),
            "downtime_required": planner_state.get("downtime_required", False),
            "rollback_strategy": planner_state.get("rollback_strategy"),
            "risk_assessment": planner_state.get("risk_assessment", {}),
            "error": planner_state.get("error"),
            "error_context": planner_state.get("error_context"),
            "current_step": planner_state.get("current_step"),
            "completed_steps": planner_state.get("completed_steps", []),
            "requires_approval": planner_state.get("requires_approval", False),
            "approval_context": planner_state.get("approval_context"),
            "planning_started_at": planner_state.get("planning_started_at"),
            "planning_completed_at": planner_state.get("planning_completed_at"),
            "planning_duration": planner_state.get("planning_duration"),
            "complexity_score": planner_state.get("complexity_score")
        }
        
        # Update common fields
        updated_state["messages"] = planner_state.get("messages", [])
        updated_state["user_request"] = planner_state.get("user_request", "")
        updated_state["context_id"] = planner_state.get("context_id")
        updated_state["task_id"] = planner_state.get("task_id")
        
        return updated_state
    
    @staticmethod
    def supervisor_to_generation_state(supervisor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor state to Generation Agent state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            
        Returns:
            Dictionary containing Generation Agent state fields
        """
        return {
            "messages": supervisor_state.get("messages", []),
            "user_request": supervisor_state.get("user_request", ""),
            "context_id": supervisor_state.get("context_id"),
            "task_id": supervisor_state.get("task_id"),
            "module_name": supervisor_state.get("generation_state", {}).get("module_name", ""),
            "requirements": supervisor_state.get("generation_state", {}).get("requirements", {}),
            "module_type": supervisor_state.get("generation_state", {}).get("module_type", ""),
            "aws_services": supervisor_state.get("generation_state", {}).get("aws_services", []),
            "generated_files": supervisor_state.get("generation_state", {}).get("generated_files", []),
            "terraform_code": supervisor_state.get("generation_state", {}).get("terraform_code", {}),
            "generation_complete": supervisor_state.get("generation_state", {}).get("generation_complete", False),
            "generation_errors": supervisor_state.get("generation_state", {}).get("generation_errors", []),
            "best_practices_applied": supervisor_state.get("generation_state", {}).get("best_practices_applied", []),
            "generation_started_at": datetime.utcnow().isoformat(),
            "generation_completed_at": supervisor_state.get("generation_state", {}).get("generation_completed_at"),
            "status": "generation_started"
        }
    
    @staticmethod
    def generation_to_supervisor_state(supervisor_state: Dict[str, Any], generation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Generation Agent state back to supervisor state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            generation_state: Generation Agent state dictionary
            
        Returns:
            Updated supervisor state dictionary
        """
        updated_state = supervisor_state.copy()
        
        # Update generation-specific fields
        updated_state["generation_state"] = {
            "module_name": generation_state.get("module_name", ""),
            "requirements": generation_state.get("requirements", {}),
            "module_type": generation_state.get("module_type", ""),
            "aws_services": generation_state.get("aws_services", []),
            "generated_files": generation_state.get("generated_files", []),
            "terraform_code": generation_state.get("terraform_code", {}),
            "generation_complete": generation_state.get("generation_complete", False),
            "generation_errors": generation_state.get("generation_errors", []),
            "best_practices_applied": generation_state.get("best_practices_applied", []),
            "generation_started_at": generation_state.get("generation_started_at"),
            "generation_completed_at": generation_state.get("generation_completed_at")
        }
        
        # Update common fields
        updated_state["messages"] = generation_state.get("messages", [])
        updated_state["user_request"] = generation_state.get("user_request", "")
        updated_state["context_id"] = generation_state.get("context_id")
        updated_state["task_id"] = generation_state.get("task_id")
        
        return updated_state
    
    @staticmethod
    def supervisor_to_validation_state(supervisor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor state to Validation Agent state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            
        Returns:
            Dictionary containing Validation Agent state fields
        """
        return {
            "messages": supervisor_state.get("messages", []),
            "user_request": supervisor_state.get("user_request", ""),
            "context_id": supervisor_state.get("context_id"),
            "task_id": supervisor_state.get("task_id"),
            "terraform_code": supervisor_state.get("validation_state", {}).get("terraform_code", {}),
            "validation_reports": supervisor_state.get("validation_state", {}).get("validation_reports", {}),
            "security_scan_results": supervisor_state.get("validation_state", {}).get("security_scan_results", {}),
            "compliance_results": supervisor_state.get("validation_state", {}).get("compliance_results", {}),
            "validation_complete": supervisor_state.get("validation_state", {}).get("validation_complete", False),
            "validation_errors": supervisor_state.get("validation_state", {}).get("validation_errors", []),
            "validation_warnings": supervisor_state.get("validation_state", {}).get("validation_warnings", []),
            "overall_score": supervisor_state.get("validation_state", {}).get("overall_score", 0.0),
            "validation_started_at": datetime.utcnow().isoformat(),
            "validation_completed_at": supervisor_state.get("validation_state", {}).get("validation_completed_at"),
            "status": "validation_started"
        }
    
    @staticmethod
    def validation_to_supervisor_state(supervisor_state: Dict[str, Any], validation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Validation Agent state back to supervisor state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            validation_state: Validation Agent state dictionary
            
        Returns:
            Updated supervisor state dictionary
        """
        updated_state = supervisor_state.copy()
        
        # Update validation-specific fields
        updated_state["validation_state"] = {
            "terraform_code": validation_state.get("terraform_code", {}),
            "validation_reports": validation_state.get("validation_reports", {}),
            "security_scan_results": validation_state.get("security_scan_results", {}),
            "compliance_results": validation_state.get("compliance_results", {}),
            "validation_complete": validation_state.get("validation_complete", False),
            "validation_errors": validation_state.get("validation_errors", []),
            "validation_warnings": validation_state.get("validation_warnings", []),
            "overall_score": validation_state.get("overall_score", 0.0),
            "validation_started_at": validation_state.get("validation_started_at"),
            "validation_completed_at": validation_state.get("validation_completed_at")
        }
        
        # Update common fields
        updated_state["messages"] = validation_state.get("messages", [])
        updated_state["user_request"] = validation_state.get("user_request", "")
        updated_state["context_id"] = validation_state.get("context_id")
        updated_state["task_id"] = validation_state.get("task_id")
        
        return updated_state
    
    @staticmethod
    def supervisor_to_editor_state(supervisor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform supervisor state to Editor Agent state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            
        Returns:
            Dictionary containing Editor Agent state fields
        """
        return {
            "messages": supervisor_state.get("messages", []),
            "user_request": supervisor_state.get("user_request", ""),
            "context_id": supervisor_state.get("context_id"),
            "task_id": supervisor_state.get("task_id"),
            "original_code": supervisor_state.get("editor_state", {}).get("original_code", {}),
            "modifications": supervisor_state.get("editor_state", {}).get("modifications", {}),
            "modified_code": supervisor_state.get("editor_state", {}).get("modified_code", {}),
            "surgical_changes": supervisor_state.get("editor_state", {}).get("surgical_changes", []),
            "editor_complete": supervisor_state.get("editor_state", {}).get("editor_complete", False),
            "editor_errors": supervisor_state.get("editor_state", {}).get("editor_errors", []),
            "change_summary": supervisor_state.get("editor_state", {}).get("change_summary", {}),
            "backup_created": supervisor_state.get("editor_state", {}).get("backup_created", False),
            "editing_started_at": datetime.utcnow().isoformat(),
            "editing_completed_at": supervisor_state.get("editor_state", {}).get("editing_completed_at"),
            "status": "editing_started"
        }
    
    @staticmethod
    def editor_to_supervisor_state(supervisor_state: Dict[str, Any], editor_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Editor Agent state back to supervisor state.
        
        Args:
            supervisor_state: Current supervisor state dictionary
            editor_state: Editor Agent state dictionary
            
        Returns:
            Updated supervisor state dictionary
        """
        updated_state = supervisor_state.copy()
        
        # Update editor-specific fields
        updated_state["editor_state"] = {
            "original_code": editor_state.get("original_code", {}),
            "modifications": editor_state.get("modifications", {}),
            "modified_code": editor_state.get("modified_code", {}),
            "surgical_changes": editor_state.get("surgical_changes", []),
            "editor_complete": editor_state.get("editor_complete", False),
            "editor_errors": editor_state.get("editor_errors", []),
            "change_summary": editor_state.get("change_summary", {}),
            "backup_created": editor_state.get("backup_created", False),
            "editing_started_at": editor_state.get("editing_started_at"),
            "editing_completed_at": editor_state.get("editing_completed_at")
        }
        
        # Update common fields
        updated_state["messages"] = editor_state.get("messages", [])
        updated_state["user_request"] = editor_state.get("user_request", "")
        updated_state["context_id"] = editor_state.get("context_id")
        updated_state["task_id"] = editor_state.get("task_id")
        
        return updated_state
    
    @staticmethod
    def get_state_transformer_for_agent(agent_name: str) -> Dict[str, Any]:
        """
        Get the appropriate state transformer functions for a given agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'planner_agent', 'generation_agent')
            
        Returns:
            Dictionary containing transform_to and transform_from functions
        """
        transformers = {
            "planner_agent": {
                "transform_to": StateTransformer.supervisor_to_planner_state,
                "transform_from": StateTransformer.planner_to_supervisor_state
            },
            "generation_agent": {
                "transform_to": StateTransformer.supervisor_to_generation_state,
                "transform_from": StateTransformer.generation_to_supervisor_state
            },
            "validation_agent": {
                "transform_to": StateTransformer.supervisor_to_validation_state,
                "transform_from": StateTransformer.validation_to_supervisor_state
            },
            "editor_agent": {
                "transform_to": StateTransformer.supervisor_to_editor_state,
                "transform_from": StateTransformer.editor_to_supervisor_state
            },
            "analysis_agent": {
                "transform_to": StateTransformer.supervisor_to_analysis_state,
                "transform_from": StateTransformer.analysis_to_supervisor_state
            }
        }
        
        return transformers.get(agent_name, {
            "transform_to": lambda state: state,  # Default: pass through
            "transform_from": lambda supervisor_state, agent_state: agent_state
        }) 