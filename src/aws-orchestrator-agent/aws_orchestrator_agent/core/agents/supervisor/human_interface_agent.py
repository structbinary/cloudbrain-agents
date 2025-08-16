"""
Human Interface Agent implementation.

This module provides the Human Interface Agent as specified in flow.md,
responsible for presenting changes for human review and collecting feedback.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync
from aws_orchestrator_agent.config.config import Config

# Create agent logger for human interface
human_logger = AgentLogger("HUMAN_INTERFACE")


class ReviewRequest(BaseModel):
    """Model for presenting changes to human for review."""
    type: str = "review_request"
    terraform_diff: str
    security_summary: str
    cost_impact: str
    options: List[str] = Field(default_factory=lambda: ["approve", "modify", "reject"])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HumanFeedback(BaseModel):
    """Model for structured human feedback."""
    action: str  # "approve", "modify", "reject"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    feedback: str
    modifications: Optional[Dict[str, Any]] = None


class HumanInterfaceAgent:
    """
    Human Interface Agent for presenting changes for human review and collecting feedback.
    
    Responsibilities (per flow.md):
    - Display diffs (HCL, JSON) with highlights for issues
    - Summarize security findings and cost impacts
    - Collect "approve"/"modify" responses
    - Return structured feedback for loopback
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Human Interface Agent."""
        self.config = config or Config()
        
        # Get LLM configuration
        llm_config = self.config.get_llm_config()
        
        # Initialize the LLM model
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            human_logger.log_structured(
                level="INFO",
                message=f"Initialized Human Interface Agent with LLM: {llm_config['provider']}:{llm_config['model']}"
            )
        except Exception as e:
            human_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM for Human Interface Agent: {e}"
            )
            raise
    
    @log_sync
    def present_for_review(
        self, 
        terraform_code: str, 
        validation_report: Dict[str, Any], 
        cost_estimate: float,
        original_code: Optional[str] = None
    ) -> ReviewRequest:
        """
        Present changes for human review.
        
        Args:
            terraform_code: The generated/modified Terraform code
            validation_report: Validation results from Validation Agent
            cost_estimate: Estimated monthly cost
            original_code: Original code (for diffs)
            
        Returns:
            ReviewRequest with formatted presentation
        """
        try:
            # Generate diff if original code provided
            terraform_diff = self._generate_diff(original_code, terraform_code) if original_code else terraform_code
            
            # Summarize security findings
            security_summary = self._summarize_security(validation_report)
            
            # Format cost impact
            cost_impact = f"Estimated monthly cost: ${cost_estimate:.2f}"
            
            review_request = ReviewRequest(
                terraform_diff=terraform_diff,
                security_summary=security_summary,
                cost_impact=cost_impact
            )
            
            human_logger.log_structured(
                level="INFO",
                message="Presented changes for human review",
                extra={
                    "cost_estimate": cost_estimate,
                    "security_issues": len(validation_report.get("security_issues", [])),
                    "validation_errors": len(validation_report.get("errors", []))
                }
            )
            
            return review_request
            
        except Exception as e:
            human_logger.log_structured(
                level="ERROR",
                message=f"Failed to present for review: {e}"
            )
            raise
    
    @log_sync
    def collect_feedback(self, user_response: str) -> HumanFeedback:
        """
        Collect and structure human feedback.
        
        Args:
            user_response: Raw user response
            
        Returns:
            Structured HumanFeedback object
        """
        try:
            # Parse user response to determine action
            response_lower = user_response.lower()
            
            if "approve" in response_lower or "yes" in response_lower or "ok" in response_lower:
                action = "approve"
            elif "modify" in response_lower or "change" in response_lower or "edit" in response_lower:
                action = "modify"
            elif "reject" in response_lower or "no" in response_lower or "cancel" in response_lower:
                action = "reject"
            else:
                # Default to modify if unclear
                action = "modify"
            
            # Extract modifications if action is modify
            modifications = None
            if action == "modify":
                modifications = self._extract_modifications(user_response)
            
            feedback = HumanFeedback(
                action=action,
                feedback=user_response,
                modifications=modifications
            )
            
            human_logger.log_structured(
                level="INFO",
                message=f"Collected human feedback: {action}",
                extra={"action": action, "feedback_length": len(user_response)}
            )
            
            return feedback
            
        except Exception as e:
            human_logger.log_structured(
                level="ERROR",
                message=f"Failed to collect feedback: {e}"
            )
            raise
    
    def _generate_diff(self, original_code: str, modified_code: str) -> str:
        """Generate a diff between original and modified code."""
        try:
            # Simple diff generation - in production, use proper diff library
            if original_code == modified_code:
                return "No changes detected"
            
            # For now, return a simple comparison
            return f"""
=== DIFF SUMMARY ===
Original code length: {len(original_code)} characters
Modified code length: {len(modified_code)} characters
Changes detected: {'Yes' if original_code != modified_code else 'No'}

=== MODIFIED CODE ===
{modified_code}
"""
        except Exception as e:
            human_logger.log_structured(
                level="WARNING",
                message=f"Failed to generate diff: {e}"
            )
            return f"Error generating diff: {e}"
    
    def _summarize_security(self, validation_report: Dict[str, Any]) -> str:
        """Summarize security findings from validation report."""
        try:
            security_issues = validation_report.get("security_issues", [])
            errors = validation_report.get("errors", [])
            warnings = validation_report.get("warnings", [])
            
            summary_parts = []
            
            if security_issues:
                summary_parts.append(f"🚨 {len(security_issues)} security issues found")
                for issue in security_issues[:3]:  # Show first 3
                    summary_parts.append(f"  - {issue.get('title', 'Unknown issue')}")
            
            if errors:
                summary_parts.append(f"❌ {len(errors)} validation errors")
            
            if warnings:
                summary_parts.append(f"⚠️ {len(warnings)} warnings")
            
            if not summary_parts:
                summary_parts.append("✅ No issues detected")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            human_logger.log_structured(
                level="WARNING",
                message=f"Failed to summarize security: {e}"
            )
            return f"Error summarizing security: {e}"
    
    def _extract_modifications(self, user_response: str) -> Optional[Dict[str, Any]]:
        """Extract modification requests from user response."""
        try:
            # Simple extraction - in production, use LLM to parse
            modifications = {}
            
            # Look for common modification patterns
            if "cidr" in user_response.lower():
                # Extract CIDR block
                import re
                cidr_match = re.search(r'(\d+\.\d+\.\d+\.\d+/\d+)', user_response)
                if cidr_match:
                    modifications["cidr_block"] = cidr_match.group(1)
            
            if "tags" in user_response.lower():
                modifications["tags"] = "user_requested_tags"
            
            if "name" in user_response.lower():
                modifications["name"] = "user_requested_name"
            
            return modifications if modifications else None
            
        except Exception as e:
            human_logger.log_structured(
                level="WARNING",
                message=f"Failed to extract modifications: {e}"
            )
            return None


# Factory function for easy creation
@log_sync
def create_human_interface_agent(config: Optional[Config] = None) -> HumanInterfaceAgent:
    """
    Factory function to create a Human Interface Agent.
    
    Args:
        config: Configuration instance
        
    Returns:
        Configured HumanInterfaceAgent instance
    """
    return HumanInterfaceAgent(config=config) 