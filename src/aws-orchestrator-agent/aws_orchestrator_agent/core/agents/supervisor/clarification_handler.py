"""
Clarification Loop Handler implementation.

This module implements the clarification loop pattern from flow.md,
handling needs_clarification workflow and human escalation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from aws_orchestrator_agent.core.supervisor.state import SupervisorState, AgentType
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync

# Create agent logger for clarification handler
clarification_logger = AgentLogger("CLARIFICATION_HANDLER")


class ClarificationRequest(BaseModel):
    """Model for clarification requests."""
    fields: List[str]
    context: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ClarificationResponse(BaseModel):
    """Model for human clarification responses."""
    provided_fields: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ClarificationHandler:
    """
    Handles the clarification loop pattern from flow.md.
    
    Pattern:
    1. Analysis Agent emits needs_clarification fields
    2. Supervisor calls escalate_to_human tool
    3. Human Agent replies with structured data
    4. Supervisor merges feedback and re-runs analysis
    """
    
    def __init__(self):
        """Initialize the clarification handler."""
        clarification_logger.log_structured(
            level="INFO",
            message="Initialized Clarification Handler"
        )
    
    @log_sync
    def check_clarification_needs(self, state: Dict[str, Any]) -> Optional[ClarificationRequest]:
        """
        Check if clarification is needed based on state.
        
        Args:
            state: Current workflow state
            
        Returns:
            ClarificationRequest if clarification needed, None otherwise
        """
        try:
            needs_clarification = state.get("needs_clarification", [])
            
            if not needs_clarification:
                return None
            
            # Extract context for clarification
            context = self._build_clarification_context(state)
            
            clarification_request = ClarificationRequest(
                fields=needs_clarification,
                context=context
            )
            
            clarification_logger.log_structured(
                level="INFO",
                message=f"Clarification needed for {len(needs_clarification)} fields",
                extra={"fields": needs_clarification, "context_length": len(context)}
            )
            
            return clarification_request
            
        except Exception as e:
            clarification_logger.log_structured(
                level="ERROR",
                message=f"Failed to check clarification needs: {e}"
            )
            raise
    
    @log_sync
    def process_human_response(self, state: Dict[str, Any], human_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process human response and merge into state (per flow.md pattern).
        
        Args:
            state: Current workflow state
            human_response: Response from human agent
            
        Returns:
            Updated state with merged feedback
        """
        try:
            updated_state = state.copy()
            
            # Merge human response into plan (per flow.md)
            if "plan" in updated_state and human_response:
                updated_state["plan"].update(human_response)
            
            # Clear needs_clarification (per flow.md)
            updated_state["needs_clarification"] = []
            
            # Store human response for audit
            updated_state["human_response"] = human_response
            
            clarification_logger.log_structured(
                level="INFO",
                message="Merged human response into state",
                extra={
                    "response_keys": list(human_response.keys()) if human_response else [],
                    "cleared_clarification": True
                }
            )
            
            return updated_state
            
        except Exception as e:
            clarification_logger.log_structured(
                level="ERROR",
                message=f"Failed to process human response: {e}"
            )
            raise
    
    @log_sync
    def should_rerun_analysis(self, state: Dict[str, Any]) -> bool:
        """
        Determine if analysis should be re-run after clarification.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if analysis should be re-run
        """
        try:
            # Check if we have human response and cleared clarification needs
            has_human_response = bool(state.get("human_response"))
            needs_clarification_cleared = not state.get("needs_clarification", [])
            
            should_rerun = has_human_response and needs_clarification_cleared
            
            clarification_logger.log_structured(
                level="DEBUG",
                message=f"Analysis rerun decision: {should_rerun}",
                extra={
                    "has_human_response": has_human_response,
                    "needs_clarification_cleared": needs_clarification_cleared
                }
            )
            
            return should_rerun
            
        except Exception as e:
            clarification_logger.log_structured(
                level="ERROR",
                message=f"Failed to determine analysis rerun: {e}"
            )
            return False
    
    @log_sync
    def build_escalation_prompt(self, clarification_request: ClarificationRequest) -> str:
        """
        Build escalation prompt for human agent.
        
        Args:
            clarification_request: The clarification request
            
        Returns:
            Formatted prompt for human escalation
        """
        try:
            fields_str = ", ".join(clarification_request.fields)
            
            prompt = f"""
Human intervention required for AWS Terraform Orchestrator.

CONTEXT:
{clarification_request.context}

MISSING INFORMATION:
The following fields need clarification: {fields_str}

Please provide the missing information in a structured format.
For example:
- For CIDR blocks: "cidr_block": "10.0.0.0/16"
- For subnet CIDRs: "subnet_cidr_blocks": ["10.0.1.0/24", "10.0.2.0/24"]
- For tags: "tags": {{"Environment": "production", "Project": "my-project"}}

Please respond with the missing information:
"""
            
            clarification_logger.log_structured(
                level="INFO",
                message="Built escalation prompt",
                extra={"fields_count": len(clarification_request.fields)}
            )
            
            return prompt
            
        except Exception as e:
            clarification_logger.log_structured(
                level="ERROR",
                message=f"Failed to build escalation prompt: {e}"
            )
            raise
    
    def _build_clarification_context(self, state: Dict[str, Any]) -> str:
        """Build context for clarification request."""
        try:
            context_parts = []
            
            # Add user request
            if "user_request" in state:
                context_parts.append(f"User Request: {state['user_request']}")
            
            # Add current plan if available
            if "plan" in state and state["plan"]:
                plan = state["plan"]
                context_parts.append(f"Current Plan: {plan.get('action', 'Unknown action')}")
                
                # Add existing parameters
                existing_params = plan.get("parameters", {})
                if existing_params:
                    context_parts.append(f"Existing Parameters: {existing_params}")
            
            # Add agent context
            if "active_agent" in state:
                context_parts.append(f"Current Agent: {state['active_agent']}")
            
            return "\n".join(context_parts) if context_parts else "No context available"
            
        except Exception as e:
            clarification_logger.log_structured(
                level="WARNING",
                message=f"Failed to build clarification context: {e}"
            )
            return "Error building context"


# Factory function for easy creation
@log_sync
def create_clarification_handler() -> ClarificationHandler:
    """
    Factory function to create a Clarification Handler.
    
    Returns:
        Configured ClarificationHandler instance
    """
    return ClarificationHandler() 