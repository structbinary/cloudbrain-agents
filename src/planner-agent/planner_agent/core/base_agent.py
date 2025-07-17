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
Abstract Base Agent Architecture

This module provides a robust abstract base class for all agent implementations,
following Python architectural best practices and design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterable, TypeVar, Generic, Union, Protocol, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from planner_agent.utils.exceptions import ConfigError
from planner_agent.utils.logger import AgentLogger
import asyncio
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

# Type variables for generic agent responses
T = TypeVar('T')
ConfigType = TypeVar('ConfigType')


class AgentStatus(Enum):
    """Agent lifecycle status."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentCapability(Enum):
    """Standard agent capabilities."""
    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"
    TOOL_CALLING = "tool_calling"
    MEMORY_PERSISTENCE = "memory_persistence"
    MULTI_MODAL = "multi_modal"


@dataclass
class AgentMetrics:
    """Agent performance and usage metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_activity: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


@dataclass
class AgentConfig:
    """Standardized agent configuration."""
    name: str
    description: str = ""
    capabilities: List['AgentCapability'] = field(default_factory=list)
    content_types: List[str] = field(default_factory=lambda: ['text/plain'])
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name.strip():
            raise ConfigError("Agent name cannot be empty")
        if self.max_retries < 0:
            raise ConfigError("max_retries must be non-negative")
        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be positive")


class StreamingProtocol(Protocol):
    """Protocol for streaming response objects."""
    response_type: str
    is_task_complete: bool
    require_user_input: bool
    content: Any


class AgentResponse(BaseModel):
    """Standardized agent response format."""
    response_type: str = "text"
    is_task_complete: bool = False
    require_user_input: bool = False
    content: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return self.error is None


class BaseAgent(ABC, Generic[ConfigType]):
    """
    Abstract base class for all agent implementations.

    Enforced interface:
    - def _initialize_agent(self, **kwargs: Any) -> None:  # Abstract, must be implemented by all agents
    - async def stream(self, query: str, session_id: str, task_id: str) -> AsyncIterable[AgentResponse]:  # Abstract, must be implemented by all agents

    All other methods are optional hooks or template methods for convenience and extension.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]], **kwargs: Any) -> None:
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration (AgentConfig instance or dict)
            **kwargs: Additional initialization parameters
        """
        # Normalize config to AgentConfig instance
        if isinstance(config, dict):
            self._config = AgentConfig(**config)
        else:
            self._config = config
            
        # Core agent state
        self._status = AgentStatus.INITIALIZING
        self._metrics = AgentMetrics() if self._config.enable_metrics else None
        self._session_data: Dict[str, Any] = {}
        
        # Set up centralized logger
        self._logger = AgentLogger(self.name)
            
        # Initialize agent-specific components
        self._initialize_agent(**kwargs)
        
        # Mark as ready
        self._status = AgentStatus.READY
        self._logger.log_structured(
            level="INFO",
            message=f"Agent '{self.name}' initialized successfully",
            extra={"agent_name": self.name}
        )
    
    # =========================================================================
    # PROPERTIES - Clean interface for agent information
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name
    
    @property 
    def description(self) -> str:
        """Agent description."""
        return self._config.description
    
    @property
    def capabilities(self) -> List['AgentCapability']:
        """Agent capabilities."""
        return self._config.capabilities
    
    @property
    def status(self) -> 'AgentStatus':
        """Current agent status."""
        return self._status
    
    @property
    def metrics(self) -> Optional[AgentMetrics]:
        """Agent performance metrics."""
        return self._metrics
    
    @property
    def config(self) -> AgentConfig:
        """Agent configuration (read-only)."""
        return self._config
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================
    @abstractmethod
    def _initialize_agent(self, **kwargs: Any) -> None:
        """
        Initialize agent-specific components.
        
        This method is called during __init__ and should set up any
        agent-specific resources, models, tools, etc.
        
        Args:
            **kwargs: Additional initialization parameters
        Must be implemented by all agent subclasses.
        """
        pass
    
    @abstractmethod
    async def stream(self, query: str, session_id: str, task_id: str) -> AsyncIterable[AgentResponse]:
        """
        Stream agent response for the given query.
        
        Args:
            query: User query or input
            session_id: Unique session identifier
            task_id: Unique task identifier
            
        Yields:
            AgentResponse: Streaming response objects
        Must be implemented by all agent subclasses.
        """
        pass
    
    
    # =========================================================================
    # TEMPLATE METHODS - Common workflow with customizable steps
    # =========================================================================
    # @log_async
    async def process_request(
        self, 
        query: str, 
        session_id: str = "",
        task_id: str = "",
        streaming: bool = True
    ) -> Union[AgentResponse, AsyncIterable[AgentResponse]]:
        """
        Template method for processing requests with error handling and metrics.
        
        Args:
            query: User query
            session_id: Session identifier
            task_id: Task identifier (optional)
            streaming: Whether to stream response
            
        Returns:
            Response or async iterator of responses
        """
        if not task_id:
            task_id = f"task_{session_id}_{asyncio.get_event_loop().time()}"
        
        # Update metrics
        if self._metrics:
            self._metrics.total_requests += 1
            
        try:
            self._status = AgentStatus.PROCESSING
            
            # Pre-processing hook
            await self._pre_process(query, session_id, task_id)
            
            # Main processing
            if streaming and AgentCapability.STREAMING in self.capabilities:
                result = self.stream(query, session_id, task_id)
            else:
                result = self.invoke(query, session_id)  # type: ignore
                
            # Post-processing hook
            await self._post_process(query, session_id, task_id)
            
            # Update success metrics
            if self._metrics:
                self._metrics.successful_requests += 1
                
            self._status = AgentStatus.READY
            return result  # type: ignore
            
        except Exception as e:
            # Update failure metrics
            if self._metrics:
                self._metrics.failed_requests += 1
                
            self._status = AgentStatus.ERROR
            self._logger.log_structured(
                level="ERROR",
                message=f"Agent {self.name} processing failed: {e}",
                extra={"agent_name": self.name}
            )
            
            # Return error response
            return AgentResponse(
                response_type="error",
                is_task_complete=True,
                error=str(e)
            )
    
    # =========================================================================
    # LIFECYCLE HOOKS - Optional customization points
    # =========================================================================
    
    async def _pre_process(self, query: str, session_id: str, task_id: str) -> None:
        """
        Hook called before main processing.
        
        Override to add custom pre-processing logic like:
        - Query validation
        - Session setup
        - Resource allocation
        """
        pass
    
    async def _post_process(self, query: str, session_id: str, task_id: str) -> None:
        """
        Hook called after successful processing.
        
        Override to add custom post-processing logic like:
        - Cleanup
        - Logging
        - Cache updates
        """
        pass
    
    # =========================================================================
    # UTILITY METHODS - Common functionality
    # =========================================================================
    
    def has_capability(self, capability: 'AgentCapability') -> bool:
        """Check if agent has specific capability."""
        return capability in self.capabilities
    
    def get_session_data(self, session_id: str = "", key: Optional[str] = None) -> Any:
        """Get session-specific data."""
        session = self._session_data.get(session_id, {})
        return session.get(key) if key else session
    
    def set_session_data(self, session_id: str, key: str, value: Any) -> None:
        """Set session-specific data."""
        if session_id not in self._session_data:
            self._session_data[session_id] = {}
        self._session_data[session_id][key] = value
    
    def clear_session(self, session_id: str) -> None:
        """Clear session data."""
        self._session_data.pop(session_id, None)
    
    @asynccontextmanager
    async def error_handling(self, operation: str = "operation") -> AsyncIterator[None]:
        """
        Context manager for consistent error handling.
        
        Usage:
            async with self.error_handling("my_operation"):
                # risky code here
                pass
        """
        try:
            yield
        except Exception as e:
            self._logger.log_structured(
                level="ERROR",
                message=f"Error in {operation}: {e}",
                extra={"agent_name": self.name}
            )
            if self._metrics:
                self._metrics.failed_requests += 1
            raise
    
    def _log_handler(self, message: Any) -> None:
        """Handle log messages (can be overridden for custom logging)."""
        self._logger.log_structured(
            level="INFO",
            message=str(message),
            extra={"agent_name": self.name}
        )
    
    def get_logging_context(self) -> Dict[str, Any]:
        """
        Get current logging context and capabilities.
        
        Returns:
            Dict containing logging setup information
        """
        return {
            "agent_name": self.name,
            "standard_logging": True,
            "enhanced_logging": False, # Enhanced logging is now handled by AgentLogger
            "websocket_streaming": False, # Websocket streaming is now handled by AgentLogger
            "log_level": self._config.log_level if self._config.enable_logging else "DISABLED"
        }
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent.
        
        Override to add custom cleanup logic.
        """
        self._status = AgentStatus.SHUTDOWN
        self._logger.log_structured(
            level="INFO",
            message=f"Agent '{self.name}' shutting down",
            extra={"agent_name": self.name}
        )
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"status='{self.status.value}', "
                f"capabilities={[c.value for c in self.capabilities]})")


# =========================================================================
# SPECIALIZED BASE CLASSES - For common agent patterns
# =========================================================================

class LLMAgent(BaseAgent[AgentConfig]):
    """
    Base class for LLM-powered agents.
    
    Provides common functionality for agents that use language models.
    """
    
    def _initialize_agent(self, **kwargs: Any) -> None:
        """Initialize LLM-specific components."""
        self._model = None  # To be set by subclasses
        self._capabilities = [
            AgentCapability.STREAMING,
            AgentCapability.TOOL_CALLING,
            AgentCapability.BATCH_PROCESSING
        ]
        # Update config capabilities
        self._config.capabilities.extend(self._capabilities)


class MultiAgentCoordinator(BaseAgent[AgentConfig]):
    """
    Base class for agents that coordinate multiple sub-agents.
    
    Provides common functionality for supervisor/coordinator patterns.
    """
    
    def _initialize_agent(self, **kwargs: Any) -> None:
        """Initialize multi-agent coordination components."""
        self._sub_agents: Dict[str, BaseAgent] = {}
        self._capabilities = [
            AgentCapability.STREAMING,
            AgentCapability.BATCH_PROCESSING,
            AgentCapability.TOOL_CALLING
        ]
        # Update config capabilities
        self._config.capabilities.extend(self._capabilities)
    
    def register_agent(self, name: str, agent: 'BaseAgent') -> None:
        """Register a sub-agent."""
        self._sub_agents[name] = agent
        self._logger.log_structured(
            level="INFO",
            message=f"Registered sub-agent '{name}' in coordinator '{self.name}'",
            extra={"agent_name": self.name, "sub_agent": name}
        )
    
    def get_agent(self, name: str) -> Optional['BaseAgent']:
        """Get a registered sub-agent."""
        return self._sub_agents.get(name)
    
    @property
    def sub_agents(self) -> List[str]:
        """List of registered sub-agent names."""
        return list(self._sub_agents.keys())
        
        
