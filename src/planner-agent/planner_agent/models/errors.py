# Copyright (C) 2025 StructBinary
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Custom error models for message validation and JSON-RPC 2.0 protocol.

This module defines validation errors and exceptions specific to the 
A2A protocol implementation with JSON-RPC 2.0 compliance.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class ValidationError(Exception):
    """Base validation error for message formatting issues."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


class MessageFormatError(ValidationError):
    """Error for invalid message format or structure."""
    
    def __init__(self, message: str, expected_format: str, received_format: str):
        self.expected_format = expected_format
        self.received_format = received_format
        super().__init__(message)


class JsonRpcValidationError(ValidationError):
    """Error for JSON-RPC 2.0 specification violations."""
    
    def __init__(self, message: str, rpc_error_code: int, field: Optional[str] = None):
        self.rpc_error_code = rpc_error_code
        super().__init__(message, field)


class A2AProtocolError(ValidationError):
    """Error for A2A protocol specific violations."""
    
    def __init__(self, message: str, protocol_version: str, agent_id: Optional[str] = None):
        self.protocol_version = protocol_version
        self.agent_id = agent_id
        super().__init__(message)


class ValidationResult(BaseModel):
    """Result of message validation with detailed feedback."""
    
    is_valid: bool = Field(..., description="Whether the message is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0


class ErrorContext(BaseModel):
    """Context information for error tracking and debugging."""
    
    timestamp: str = Field(..., description="Error timestamp")
    agent_id: Optional[str] = Field(None, description="Agent identifier where error occurred")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_id: Optional[str] = Field(None, description="Request identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")


def create_validation_error(
    message: str,
    error_type: str = "validation",
    field: Optional[str] = None,
    value: Any = None,
    context: Optional[ErrorContext] = None
) -> Dict[str, Any]:
    """
    Create a standardized validation error dictionary.
    
    Args:
        message: Error message
        error_type: Type of error (validation, format, protocol)
        field: Field name where error occurred
        value: Value that caused the error
        context: Additional error context
        
    Returns:
        Standardized error dictionary
    """
    error_dict = {
        "message": message,
        "type": error_type,
        "field": field,
        "value": str(value) if value is not None else None,
    }
    
    if context:
        error_dict["context"] = context.model_dump()
    
    return error_dict


# Standard error message templates
ERROR_MESSAGES = {
    "missing_required_field": "Required field '{field}' is missing",
    "invalid_jsonrpc_version": "Invalid JSON-RPC version. Expected '2.0', got '{version}'",
    "invalid_method": "Invalid method '{method}'. Supported methods: {supported_methods}",
    "invalid_params": "Invalid parameters for method '{method}': {details}",
    "missing_request_id": "Request ID is required for tracking",
    "invalid_response_format": "Response must have either 'result' or 'error', not both",
    "protocol_version_mismatch": "A2A protocol version mismatch. Expected '{expected}', got '{actual}'",
    "authentication_failed": "Agent authentication failed for agent ID '{agent_id}'",
    "agent_not_available": "Target agent '{agent_id}' is not available",
    "task_processing_failed": "Task processing failed: {details}",
    "capability_mismatch": "Agent capability mismatch for requested operation '{operation}'",
}


def get_error_message(error_key: str, **kwargs) -> str:
    """
    Get a formatted error message from the templates.
    
    Args:
        error_key: Key from ERROR_MESSAGES
        **kwargs: Values to format into the message template
        
    Returns:
        Formatted error message
    """
    template = ERROR_MESSAGES.get(error_key, "Unknown error: {error_key}")
    try:
        return template.format(error_key=error_key, **kwargs)
    except KeyError as e:
        return f"Error formatting message template '{error_key}': missing key {e}" 