"""
Supervisor Agent module for AWS Orchestrator Agent.

This module provides the core Supervisor Agent implementation using LangGraph's
supervisor (tool-calling) pattern for orchestrating specialized agent subgraphs.
"""

from aws_orchestrator_agent.core.agents.supervisor.supervisor_prompts import (
    SUPERVISOR_PROMPT,
    ANALYSIS_ROUTING_PROMPT,
    GENERATION_ROUTING_PROMPT,
    VALIDATION_ROUTING_PROMPT,
    EDITOR_ROUTING_PROMPT
)


__all__ = [
    # Prompts
    "SUPERVISOR_PROMPT",
    "ANALYSIS_ROUTING_PROMPT",
    "GENERATION_ROUTING_PROMPT",
    "VALIDATION_ROUTING_PROMPT",
    "EDITOR_ROUTING_PROMPT",

] 