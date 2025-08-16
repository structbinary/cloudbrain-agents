"""
Reducer functions for state aggregation in AWS Orchestrator Agent Supervisor.

This module provides reducer functions that define how state updates are
merged and aggregated across the multi-agent orchestration system.
"""

from typing import Dict, List, Any, Union
import logging

logger = logging.getLogger(__name__)


def merge_validation_reports(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge validation reports from multiple validation steps.
    
    This reducer intelligently merges validation reports, handling different
    data types (lists, dicts, primitives) and preserving historical data
    while adding new validation results.
    
    Args:
        old: Existing validation reports
        new: New validation reports to merge
        
    Returns:
        Merged validation reports
    """
    merged = old.copy()
    
    for key, value in new.items():
        if key in merged:
            existing_value = merged[key]
            
            # Handle different data types
            if isinstance(existing_value, list):
                if isinstance(value, list):
                    # Merge lists, avoiding duplicates
                    merged[key] = existing_value + [item for item in value if item not in existing_value]
                else:
                    # Append single value to list
                    if value not in existing_value:
                        merged[key].append(value)
                        
            elif isinstance(existing_value, dict):
                if isinstance(value, dict):
                    # Recursively merge dictionaries
                    merged[key] = merge_validation_reports(existing_value, value)
                else:
                    # Replace dict with new value (log warning)
                    logger.warning(f"Replacing dict value for key '{key}' with non-dict value")
                    merged[key] = value
                    
            elif isinstance(existing_value, (int, float)):
                if isinstance(value, (int, float)):
                    # For numeric values, take the maximum (assuming higher is better)
                    merged[key] = max(existing_value, value)
                else:
                    # Replace numeric with new value
                    merged[key] = value
                    
            else:
                # For other types (strings, booleans), replace with new value
                merged[key] = value
        else:
            # New key, add directly
            merged[key] = value
    
    return merged


def append_audit_log(old: List[Dict[str, Any]], new: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Append new audit entry to existing audit log.
    
    This reducer ensures audit entries are properly appended to the log
    while maintaining chronological order and data integrity.
    
    Args:
        old: Existing audit log entries
        new: New audit entry to append
        
    Returns:
        Updated audit log with new entry appended
    """
    if not isinstance(new, dict):
        logger.error(f"Invalid audit entry type: {type(new)}, expected dict")
        return old
    
    # Validate required fields
    required_fields = ["action"]
    for field in required_fields:
        if field not in new:
            logger.warning(f"Audit entry missing required field: {field}")
            new[field] = "unknown"
    
    # Ensure timestamp is present
    if "timestamp" not in new:
        from datetime import datetime
        new["timestamp"] = datetime.utcnow().isoformat()
    
    # Append new entry
    return old + [new]


def merge_conversation_history(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge conversation history from multiple sources.
    
    This reducer merges conversation history while maintaining chronological
    order and avoiding duplicate entries.
    
    Args:
        old: Existing conversation history
        new: New conversation entries to merge
        
    Returns:
        Merged conversation history
    """
    if not isinstance(new, list):
        logger.error(f"Invalid conversation history type: {type(new)}, expected list")
        return old
    
    # Create a set of existing message IDs to avoid duplicates
    existing_ids = set()
    for entry in old:
        if isinstance(entry, dict) and "message_id" in entry:
            existing_ids.add(entry["message_id"])
    
    # Filter out duplicates from new entries
    unique_new_entries = []
    for entry in new:
        if isinstance(entry, dict):
            message_id = entry.get("message_id")
            if message_id is None or message_id not in existing_ids:
                unique_new_entries.append(entry)
                if message_id:
                    existing_ids.add(message_id)
        else:
            # Non-dict entries are added as-is
            unique_new_entries.append(entry)
    
    # Merge and maintain chronological order
    merged = old + unique_new_entries
    
    # Sort by timestamp if available
    try:
        merged.sort(key=lambda x: x.get("timestamp", ""))
    except (TypeError, AttributeError):
        # If sorting fails, keep original order
        pass
    
    return merged


def merge_error_context(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge error context information.
    
    This reducer merges error context while preserving historical error
    information and adding new error details.
    
    Args:
        old: Existing error context
        new: New error context to merge
        
    Returns:
        Merged error context
    """
    merged = old.copy()
    
    for key, value in new.items():
        if key in merged:
            existing_value = merged[key]
            
            if isinstance(existing_value, list) and isinstance(value, list):
                # Merge error lists
                merged[key] = existing_value + value
            elif isinstance(existing_value, dict) and isinstance(value, dict):
                # Recursively merge error dictionaries
                merged[key] = merge_error_context(existing_value, value)
            else:
                # Replace with new value
                merged[key] = value
        else:
            # New error context
            merged[key] = value
    
    return merged


def merge_execution_times(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """
    Merge agent execution times.
    
    This reducer merges execution time data, taking the maximum time
    for each agent (assuming longer times indicate more comprehensive execution).
    
    Args:
        old: Existing execution times
        new: New execution times to merge
        
    Returns:
        Merged execution times
    """
    merged = old.copy()
    
    for agent_name, execution_time in new.items():
        if agent_name in merged:
            # Take the maximum execution time
            merged[agent_name] = max(merged[agent_name], execution_time)
        else:
            # New agent execution time
            merged[agent_name] = execution_time
    
    return merged


def merge_requirements(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge requirements from multiple sources.
    
    This reducer intelligently merges requirements, handling conflicts
    and preserving the most specific or recent information.
    
    Args:
        old: Existing requirements
        new: New requirements to merge
        
    Returns:
        Merged requirements
    """
    merged = old.copy()
    
    for key, value in new.items():
        if key in merged:
            existing_value = merged[key]
            
            if isinstance(existing_value, dict) and isinstance(value, dict):
                # Recursively merge requirement dictionaries
                merged[key] = merge_requirements(existing_value, value)
            elif isinstance(existing_value, list) and isinstance(value, list):
                # Merge requirement lists, avoiding duplicates
                merged[key] = existing_value + [item for item in value if item not in existing_value]
            else:
                # For conflicts, prefer the new value (more recent)
                merged[key] = value
        else:
            # New requirement
            merged[key] = value
    
    return merged


def merge_terraform_code(old: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """
    Merge Terraform code from multiple sources.
    
    This reducer merges Terraform code files, with new code taking precedence
    over existing code for the same file paths.
    
    Args:
        old: Existing Terraform code
        new: New Terraform code to merge
        
    Returns:
        Merged Terraform code
    """
    merged = old.copy()
    
    for file_path, code_content in new.items():
        if file_path in merged:
            # New code takes precedence (assumes it's more up-to-date)
            logger.info(f"Updating existing Terraform file: {file_path}")
        merged[file_path] = code_content
    
    return merged


def merge_security_results(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge security scanning results.
    
    This reducer merges security results while maintaining severity levels
    and avoiding duplicate findings.
    
    Args:
        old: Existing security results
        new: New security results to merge
        
    Returns:
        Merged security results
    """
    merged = old.copy()
    
    for key, value in new.items():
        if key in merged:
            existing_value = merged[key]
            
            if isinstance(existing_value, list) and isinstance(value, list):
                # Merge security findings, avoiding duplicates by ID
                existing_ids = {finding.get("id") for finding in existing_value if isinstance(finding, dict)}
                unique_new_findings = [
                    finding for finding in value 
                    if isinstance(finding, dict) and finding.get("id") not in existing_ids
                ]
                merged[key] = existing_value + unique_new_findings
            elif isinstance(existing_value, dict) and isinstance(value, dict):
                # Recursively merge security result dictionaries
                merged[key] = merge_security_results(existing_value, value)
            else:
                # Replace with new value
                merged[key] = value
        else:
            # New security result
            merged[key] = value
    
    return merged


def merge_compliance_results(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge compliance checking results.
    
    This reducer merges compliance results while maintaining compliance
    status and rule information.
    
    Args:
        old: Existing compliance results
        new: New compliance results to merge
        
    Returns:
        Merged compliance results
    """
    merged = old.copy()
    
    for key, value in new.items():
        if key in merged:
            existing_value = merged[key]
            
            if isinstance(existing_value, dict) and isinstance(value, dict):
                # Recursively merge compliance result dictionaries
                merged[key] = merge_compliance_results(existing_value, value)
            elif isinstance(existing_value, list) and isinstance(value, list):
                # Merge compliance rule results
                merged[key] = existing_value + value
            else:
                # For compliance status, prefer the more restrictive result
                if isinstance(existing_value, str) and isinstance(value, str):
                    # Priority: FAILED > WARNING > PASSED
                    priority = {"FAILED": 3, "WARNING": 2, "PASSED": 1}
                    existing_priority = priority.get(existing_value.upper(), 0)
                    new_priority = priority.get(value.upper(), 0)
                    
                    if new_priority > existing_priority:
                        merged[key] = value
                else:
                    # Replace with new value
                    merged[key] = value
        else:
            # New compliance result
            merged[key] = value
    
    return merged 