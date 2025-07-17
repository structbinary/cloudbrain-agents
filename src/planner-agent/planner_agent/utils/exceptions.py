"""Custom exceptions for the planner agent."""


class PlannerAgentError(Exception):
    """Base exception for planner agent errors."""
    pass

class UnsupportedProviderError(PlannerAgentError):
    """Raised when an unsupported LLM provider is requested."""
    pass

class LLMConfigurationError(PlannerAgentError):
    """Raised when LLM configuration is invalid."""
    pass
