"""
Base Agent Class for LangGraph-based agents following LangChain/LangGraph best practices.

This module provides a comprehensive base agent class that encapsulates:
- State management with Pydantic schemas
- Async/sync node execution patterns
- Error handling and recovery mechanisms
- Observability and instrumentation
- Subgraph composition patterns
- Human-in-the-loop integration
- MCP tool integration patterns

All specialized agents (Planner, Generation, Validation, Editor) should inherit from this base class.
"""

import asyncio
import inspect
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any, 
    AsyncGenerator, 
    Callable, 
    Dict, 
    List, 
    Optional, 
    Type, 
    TypeVar, 
    Union,
    TypedDict,
    Annotated
)
from functools import wraps

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field, ValidationError

from aws_orchestrator_agent.utils.logger import AgentLogger
from aws_orchestrator_agent.utils.mcp_client import MCPAdapterClient
from aws_orchestrator_agent.core.agents.supervisor.state.state_manager import StateManager

# Type variables for generic state types
StateType = TypeVar('StateType', bound=BaseModel)
NodeFunction = Callable[[Dict[str, Any]], Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]]

# Configure structured logging
logger = AgentLogger("BASE_AGENT")


class BaseAgentState(TypedDict):
    """Base state schema for all agents following LangGraph patterns."""
    messages: Annotated[List[BaseMessage], add_messages]
    agent_name: str
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base agent class following LangChain/LangGraph best practices.
    
    This class provides a comprehensive foundation for all specialized agents,
    implementing patterns for state management, error handling, observability,
    and integration with the LangGraph ecosystem.
    """
    
    def __init__(
        self,
        name: str,
        state_schema: Type[StateType],
        config: Optional[Dict[str, Any]] = None,
        mcp_client: Optional[MCPAdapterClient] = None,
        state_manager: Optional[StateManager] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for this agent
            state_schema: Pydantic model defining the agent's state structure
            config: Configuration dictionary for the agent
            mcp_client: Optional MCP client for external tool integration
            state_manager: Optional state manager for advanced state operations
        """
        self.name = name
        self.state_schema = state_schema
        self.config = config or {}
        self.mcp_client = mcp_client
        self.state_manager = state_manager or StateManager()
        
        # Core components
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[Any] = None
        self.nodes: Dict[str, NodeFunction] = {}
        self.tools: Dict[str, Any] = {}
        
        # State and execution tracking
        self.current_state: Optional[StateType] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.is_initialized = False
        
        # Observability
        self.logger = AgentLogger(f"{self.__class__.__name__.upper()}")
        self.metrics = {
            "executions": 0,
            "errors": 0,
            "total_execution_time": 0.0
        }
        
        # Human-in-the-loop support
        self.hitl_enabled = self.config.get("human_in_the_loop", False)
        self.approval_required = self.config.get("approval_required", False)
        
    @abstractmethod
    def define_nodes(self) -> Dict[str, NodeFunction]:
        """
        Define the agent's nodes (functions) that will be added to the StateGraph.
        
        Returns:
            Dictionary mapping node names to their corresponding functions
        """
        pass
    
    @abstractmethod
    def define_edges(self) -> Dict[str, List[str]]:
        """
        Define the edges (transitions) between nodes in the StateGraph.
        
        Returns:
            Dictionary mapping node names to lists of target node names
        """
        pass
    
    @abstractmethod
    def create_initial_state(self, input_data: Dict[str, Any]) -> StateType:
        """
        Create the initial state for the agent based on input data.
        
        Args:
            input_data: Input data to initialize the state
            
        Returns:
            Initialized state object
        """
        pass
    
    def add_node(self, name: str, node_function: NodeFunction) -> None:
        """
        Add a node to the agent's graph.
        
        Args:
            name: Name of the node
            node_function: Function to execute for this node
        """
        self.nodes[name] = node_function
        self.logger.log_structured(
            level="DEBUG",
            message=f"Added node: {name}",
            extra={"agent_name": self.name, "node_name": name}
        )
    
    def add_tool(self, name: str, tool: Any) -> None:
        """
        Add a tool to the agent's available tools.
        
        Args:
            name: Name of the tool
            tool: Tool object (LangChain tool, MCP tool, etc.)
        """
        self.tools[name] = tool
        self.logger.log_structured(
            level="DEBUG",
            message=f"Added tool: {name}",
            extra={"agent_name": self.name, "tool_name": name}
        )
    
    def build_graph(self) -> StateGraph:
        """
        Build the StateGraph for this agent.
        
        Returns:
            Configured StateGraph instance
        """
        if not self.nodes:
            raise ValueError(f"No nodes defined for agent {self.name}")
        
        # Create StateGraph with the agent's state schema
        graph = StateGraph(self.state_schema)
        
        # Add all nodes to the graph
        for name, node_function in self.nodes.items():
            graph.add_node(name, node_function)
        
        # Add edges based on the agent's definition
        edges = self.define_edges()
        for source, targets in edges.items():
            for target in targets:
                graph.add_edge(source, target)
        
        # Set entry and finish points automatically
        # Find nodes that are not targets of any edge (start nodes)
        all_targets = set()
        for targets in edges.values():
            all_targets.update(targets)
        
        start_nodes = [name for name in self.nodes.keys() if name not in all_targets]
        if start_nodes:
            graph.set_entry_point(start_nodes[0])
        
        # Find nodes that don't have any targets (end nodes)
        end_nodes = [name for name in self.nodes.keys() if not edges.get(name)]
        if end_nodes:
            graph.set_finish_point(end_nodes[-1])
        
        self.graph = graph
        self.logger.log_structured(
            level="INFO",
            message=f"Built StateGraph for agent {self.name} with {len(self.nodes)} nodes",
            extra={"agent_name": self.name, "node_count": len(self.nodes)}
        )
        return graph
    
    def compile_graph(self, **kwargs) -> Any:
        """
        Compile the StateGraph into an executable agent.
        
        Args:
            **kwargs: Additional compilation options
            
        Returns:
            Compiled graph object
        """
        if not self.graph:
            self.build_graph()
        
        # Add memory saver for state persistence
        memory = MemorySaver()
        
        # Compile with memory, name, and additional options
        self.compiled_graph = self.graph.compile(
            checkpointer=memory,
            name=self.name,  # Add name for langgraph-supervisor compatibility
            **kwargs
        )
        
        self.is_initialized = True
        self.logger.log_structured(
            level="INFO",
            message=f"Compiled graph for agent {self.name}",
            extra={"agent_name": self.name}
        )
        return self.compiled_graph
    
    def get_compiled_graph(self) -> Any:
        """
        Get the compiled graph for this agent.
        
        Returns:
            Compiled graph object ready for execution by supervisor
        """
        if not self.is_initialized:
            self.compile_graph()
        
        return self.compiled_graph
    
    def create_state_for_execution(self, input_data: Dict[str, Any]) -> StateType:
        """
        Create initial state for execution by supervisor.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Initialized state object ready for execution
        """
        return self.create_initial_state(input_data)
    
    def get_interrupt_info(self) -> Dict[str, Any]:
        """Get information about interrupt handling for supervisor integration."""
        return {
            "agent_name": self.name,
            "supports_interrupts": self.hitl_enabled,
            "approval_required": self.approval_required,
            "interrupt_nodes": [name for name, func in self.nodes.items() if hasattr(func, '__wrapped__') and hasattr(func.__wrapped__, '_require_approval')]
        }
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log errors with proper context for supervisor integration."""
        self.metrics["errors"] += 1
        
        error_context = {
            "agent": self.name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.logger.log_structured(
            level="ERROR",
            message=f"Error in agent {self.name}",
            extra=error_context
        )
        
        # Record error in execution history
        self.execution_history.append({
            "timestamp": time.time(),
            "type": "error",
            "error": error_context
        })
    
    def record_metric(self, metric_name: str, value: Any) -> None:
        """Record custom metrics for observability."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "timestamp": time.time(),
            "value": value
        })
        
        self.logger.log_structured(
            level="DEBUG",
            message=f"Recorded metric for agent {self.name}",
            extra={
                "agent_name": self.name,
                "metric_name": metric_name,
                "value": value
            }
        )
    
    def get_state(self) -> Optional[StateType]:
        """Get the current state of the agent."""
        return self.current_state
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history of the agent."""
        return self.execution_history
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the agent's execution metrics."""
        return self.metrics.copy()
    
    def reset(self) -> None:
        """Reset the agent's state and execution history."""
        self.current_state = None
        self.execution_history = []
        self.metrics = {
            "executions": 0,
            "errors": 0,
            "total_execution_time": 0.0
        }
        self.logger.log_structured(
            level="INFO",
            message=f"Reset agent {self.name}",
            extra={"agent_name": self.name}
        )
    
    def is_ready(self) -> bool:
        """Check if the agent is ready for execution."""
        return self.is_initialized and self.compiled_graph is not None
    
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the agent's graph for supervisor integration.
        
        Returns:
            Dictionary with graph information
        """
        if not self.is_initialized:
            self.compile_graph()
        
        return {
            "agent_name": self.name,
            "node_count": len(self.nodes),
            "tool_count": len(self.tools),
            "is_compiled": self.compiled_graph is not None,
            "state_schema": self.state_schema.__name__,
            "entry_points": list(self.nodes.keys())[:1] if self.nodes else [],
            "finish_points": list(self.nodes.keys())[-1:] if self.nodes else []
        }
    
    def add_mcp_tool(self, tool_name: str, mcp_tool: Any) -> None:
        """
        Add an MCP tool to the agent's available tools.
        
        Args:
            tool_name: Name of the MCP tool
            mcp_tool: MCP tool object
        """
        if self.mcp_client:
            self.mcp_client.register_tool(tool_name, mcp_tool)
            self.logger.log_structured(
                level="INFO",
                message=f"Added MCP tool: {tool_name}",
                extra={"agent_name": self.name, "tool_name": tool_name}
            )
        else:
            self.logger.log_structured(
                level="WARNING",
                message=f"Cannot add MCP tool {tool_name}: no MCP client configured",
                extra={"agent_name": self.name, "tool_name": tool_name}
            )
    
    def create_tool_node(self, tool_name: str) -> ToolNode:
        """
        Create a LangGraph ToolNode for a specific tool.
        
        Args:
            tool_name: Name of the tool to create a node for
            
        Returns:
            ToolNode instance
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in agent {self.name}")
        
        return ToolNode([self.tools[tool_name]])
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate a state dictionary against the agent's state schema.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.state_schema(**state)
            return True
        except ValidationError as e:
            self.logger.log_structured(
                level="ERROR",
                message=f"State validation failed for agent {self.name}: {e}",
                extra={"agent_name": self.name, "error": str(e)}
            )
            return False
    
    def transform_state(
        self, 
        source_state: Dict[str, Any], 
        target_schema: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Transform state from one schema to another.
        
        Args:
            source_state: Source state dictionary
            target_schema: Target state schema
            
        Returns:
            Transformed state dictionary
        """
        try:
            # Create intermediate state object
            intermediate = self.state_schema(**source_state)
            
            # Convert to target schema
            target_state = target_schema(**intermediate.dict())
            
            return target_state.dict()
        except ValidationError as e:
            self.logger.log_structured(
                level="ERROR",
                message=f"State transformation failed: {e}",
                extra={"agent_name": self.name, "error": str(e)}
            )
            raise


def agent_node(func: NodeFunction) -> NodeFunction:
    """
    Decorator for agent node functions that adds observability and error handling.
    
    Args:
        func: Node function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    async def wrapper(state: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        node_name = func.__name__
        node_logger = AgentLogger(f"NODE_{node_name.upper()}")
        
        start_time = time.time()
        
        # Handle both dictionary and Pydantic object states
        if hasattr(state, 'model_dump'):
            # It's a Pydantic model
            state_keys = list(state.model_dump().keys())
            state_type = type(state).__name__
        else:
            # It's a dictionary
            state_keys = list(state.keys()) if isinstance(state, dict) else []
            state_type = type(state).__name__
        
        node_logger.log_structured(
            level="DEBUG",
            message=f"Executing node {node_name}",
            extra={"node_name": node_name, "state_keys": state_keys, "state_type": state_type}
        )
        
        try:
            if inspect.iscoroutinefunction(func):
                # Handle async nodes
                result = await func(state)
                execution_time = time.time() - start_time
                node_logger.log_structured(
                    level="DEBUG",
                    message=f"Completed async node {node_name}",
                    extra={"node_name": node_name, "execution_time": execution_time}
                )
                return result
            else:
                # Handle sync nodes
                result = func(state)
                execution_time = time.time() - start_time
                node_logger.log_structured(
                    level="DEBUG",
                    message=f"Completed sync node {node_name}",
                    extra={"node_name": node_name, "execution_time": execution_time}
                )
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Check if this is a GraphInterrupt (LangGraph's interrupt mechanism)
            # If it is, we should NOT log it as an error - it's normal flow control
            if hasattr(e, '__class__') and 'GraphInterrupt' in e.__class__.__name__:
                # This is a LangGraph interrupt - let it propagate naturally
                # Don't log it as an error, just re-raise it
                raise
            
            # For all other exceptions, log as error and re-raise
            node_logger.log_structured(
                level="ERROR",
                message=f"Error in node {node_name}",
                extra={
                    "node_name": node_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "execution_time": execution_time
                }
            )
            
            # Re-raise the exception to be handled by the caller
            raise
    
    return wrapper


def require_approval(func: NodeFunction) -> NodeFunction:
    """
    Decorator for nodes that require human approval.
    
    Args:
        func: Node function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(state: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        # Check if approval is required - handle both dict and Pydantic objects
        approval_required = False
        if hasattr(state, 'model_dump'):
            # It's a Pydantic model
            state_dict = state.model_dump()
            approval_required = state_dict.get("approval_required", False)
        else:
            # It's a dictionary
            approval_required = state.get("approval_required", False)
        
        # Let the node function handle approval logic naturally
        # The node function should call interrupt() if needed
        return func(state)
    
    return wrapper 