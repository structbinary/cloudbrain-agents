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
Multi-Agent Planner: Supervisor Tool-Calling Architecture

This module implements a multi-agent architecture with:
1. Agent Discovery & Prompt Refinement Node (with discovery tools)
2. MCP Server Orchestration Node (with orchestration tools)

Each node has specific tool bindings appropriate to its function.
Inherits from MultiAgentCoordinator for robust agent lifecycle management.
"""

from typing import Any, AsyncIterable, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from planner_agent.config import Config
from planner_agent.utils.exceptions import ConfigError
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentResponse, AgentCapability
from planner_agent.core.llm.llm_provider import LLMProvider
from planner_agent.models.agent_state import MultiAgentState
from planner_agent.utils.logger import log_sync, log_async, AgentLogger
from planner_agent.core.nodes import A2AAgentCardMapper, MCPNodeMapper, TaskDecompositionNode
import uuid

# Memory saver for conversation state
memory = MemorySaver()
load_dotenv()


class MultiAgentPlanner(MultiAgentCoordinator):
    """
    Multi-Agent Planner using supervisor tool-calling pattern with specific node tool bindings.
    
    Inherits from MultiAgentCoordinator to leverage:
    - Robust lifecycle management
    - Built-in metrics and monitoring
    - Standardized error handling
    - Session management
    - Template method patterns
    """

    @log_sync
    def __init__(
        self,
        llm_model: Any = None,
        a2a_agent_mapper: Optional['A2AAgentCardMapper'] = None,
        mcp_server_mapper: Optional['MCPNodeMapper'] = None,
        config: Optional[Config] = None,
        logger: Optional[AgentLogger] = None,
        enable_visual_logging: bool = True,
        websocket: Optional[Any] = None,
        stream_output: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        self.model = llm_model
        self.a2a_agent_mapper = a2a_agent_mapper
        self.mcp_server_mapper = mcp_server_mapper
        self._logger = logger or AgentLogger("multi_agent_planner")
        self._planner_config = config or Config()
        self._enable_visual_logging = enable_visual_logging
        self._websocket = websocket
        self._stream_output = stream_output
        # REMOVE: self.task_decomposition_node = TaskDecompositionNode(self.model, logger=self._logger)
        agent_config = AgentConfig(
            name="MULTI_AGENT_PLANNER",
            description="Multi-agent DevOps task planning with intelligent routing",
            capabilities=[
                AgentCapability.STREAMING,
                AgentCapability.TOOL_CALLING,
                AgentCapability.BATCH_PROCESSING
            ],
            content_types=['text/plain', 'application/json'],
            enable_metrics=True,
            enable_logging=True,
            log_level="INFO"
        )
        super().__init__(config=agent_config, **kwargs)

    @log_sync
    def _initialize_agent(self, **kwargs: Any) -> None:
        """Initialize multi-agent planner specific components using injected dependencies."""
        # Initialize LLM if not injected
        if self.model is None:
            try:
                self.model = LLMProvider.create_llm(**self._planner_config.get_llm_config())
            except Exception as e:
                raise ConfigError(f"LLM initialization failed: {e}")
        # Now instantiate the task decomposition node with a valid model
        self.task_decomposition_node = TaskDecompositionNode(self.model, logger=self._logger)
        # Initialize agent mappers if not injected
        if self.a2a_agent_mapper is None:
            self.a2a_agent_mapper = A2AAgentCardMapper(self.model, config=self._planner_config, logger=self._logger)
        if self.mcp_server_mapper is None:
            self.mcp_server_mapper = MCPNodeMapper(self.model, config=self._planner_config, logger=self._logger)
        self.graph = self._build_graph()

    def generic_branch(self, state: MultiAgentState, *args: Any, **kwargs: Any) -> str:
        return state.next if state.next is not None else "__end__"

    @log_sync
    def _build_graph(self):
        """Build the multi-agent state graph with node-specific tool bindings. Requires agent mappers to be set."""
        if self.a2a_agent_mapper is None or self.mcp_server_mapper is None:
            raise RuntimeError("Both a2a_agent_mapper and mcp_server_mapper must be set before building the graph.")
        graph = StateGraph(MultiAgentState)
        # Use the new node_stream method from TaskDecompositionNode
        graph.add_node("task_decomposition", self.task_decomposition_node.node_stream)
        graph.add_node("a2a_agent_mapper", self.a2a_agent_mapper.node_stream)
        graph.add_node("mcp_server_mapper", self.mcp_server_mapper.node_stream)
        graph.add_edge(START, "task_decomposition")
        graph.add_conditional_edges("task_decomposition", self.generic_branch)
        graph.add_conditional_edges("a2a_agent_mapper", self.generic_branch)
        graph.add_conditional_edges("mcp_server_mapper", self.generic_branch)
        return graph.compile(checkpointer=memory)


    @log_async
    async def stream(self, query_or_command, session_id: str, task_id: str) -> AsyncIterable[AgentResponse]:
        self._logger.log_structured(
            level="INFO",
            message=f"[stream] START",
            task_id=task_id,
            context_id=session_id,
            extra={"agent_name": self.__class__.__name__, "query_or_command": str(query_or_command)}
        )
        """
        Stream method supporting both initial and resume calls for human-in-the-loop (HITL).
        - If query_or_command is a string: initial call (user query)
        - If query_or_command is a Command: resume call (user feedback)
        """
        self._logger.log_structured(
            level="INFO",
            message=f"Starting Multi-Agent Stream",
            task_id=task_id,
            context_id=session_id,
            extra={"agent_name": self.__class__.__name__, "query_or_command": str(query_or_command)}
        )

        # For each new user-initiated task, generate a unique thread_id
        if isinstance(query_or_command, Command):
            # Resume call: extract resume value and inject into state, reuse session_id as thread_id
            resume_value = query_or_command.resume
            graph_input = MultiAgentState(
                messages=[],
                user_query="",
                context_id=session_id,
                task_id=task_id,
                resume_value=resume_value
            )
            thread_id = session_id  # Reuse for HITL resume
        else:
            # Initial call: build input state, use a new thread_id for state isolation
            user_query = query_or_command
            graph_input = MultiAgentState(
                messages=[HumanMessage(content=user_query)],
                user_query=user_query,
                context_id=session_id,
                task_id=task_id,
            )
            thread_id = str(uuid.uuid4())  # Unique per new user-initiated task

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
        step_count = 0
        try:
            async for item in self.graph.astream(graph_input, config, stream_mode='values'):
                step_count += 1
                self._logger.log_structured(
                    level="INFO",
                    message=f"[stream] step={step_count} item={item}",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "step_count": step_count, "item": str(item)}
                )

                # 1. Handle human-in-the-loop interrupt
                if '__interrupt__' in item:
                    interrupt_payload = item['__interrupt__'][0].value  # dict passed to interrupt()
                    yield AgentResponse(
                        response_type='human_input',
                        is_task_complete=False,
                        require_user_input=True,
                        content=interrupt_payload.get('question', 'Input required'),
                        metadata={
                            'session_id': session_id,
                            'task_id': task_id,
                            'agent_name': self.name,
                            'step_count': step_count,
                            'logging_context': self.get_logging_context(),
                            'status': 'input_required'
                        }
                    )
                    # Pause streaming until client resumes with feedback
                    break

                # 2. Existing logic for normal state updates
                status = item.get('status')
                question = item.get('question')
                mcp_server_details = item.get('mcp_servers')
                final_response = item.get('final_response')

                if status is not None:
                    if status == 'input_required':
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=True,
                            content=question or 'More information needed to proceed.',
                            metadata={
                                'session_id': session_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'logging_context': self.get_logging_context(),
                                'status': 'input_required'
                            }
                        )
                    elif status == 'error' or status == 'failed':
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=True,
                            content=question or 'An error occurred while processing your request.',
                            metadata={
                                'session_id': session_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'logging_context': self.get_logging_context(),
                                'status': 'failed'
                            }
                        )
                    elif status == 'completed':
                        if mcp_server_details is not None:
                            content_data = {
                                'status': status,
                                'mcp_server_details': mcp_server_details,
                                'question': question,
                                'final_response': final_response
                            }
                            yield AgentResponse(
                                response_type='data',
                                is_task_complete=True,
                                require_user_input=False,
                                content=content_data,
                                metadata={
                                    'session_id': session_id,
                                    'task_id': task_id,
                                    'agent_name': self.name,
                                    'step_count': step_count,
                                    'logging_context': self.get_logging_context(),
                                    'status': 'completed'
                                }
                            )
                        else:
                            yield AgentResponse(
                                response_type='text',
                                is_task_complete=False,
                                require_user_input=False,
                                content=f'Task decomposition completed. Moving to agent selection...',
                                metadata={
                                    'session_id': session_id,
                                    'task_id': task_id,
                                    'agent_name': self.name,
                                    'step_count': step_count,
                                    'logging_context': self.get_logging_context(),
                                    'status': 'working'
                                }
                            )
                    else:
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=False,
                            content=f'Processing... Status: {status}',
                            metadata={
                                'session_id': session_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'logging_context': self.get_logging_context(),
                                'status': 'working'
                            }
                        )
                else:
                    yield AgentResponse(
                        response_type='text',
                        is_task_complete=False,
                        require_user_input=False,
                        content='Processing...',
                        metadata={
                            'session_id': session_id,
                            'task_id': task_id,
                            'agent_name': self.name,
                            'step_count': step_count,
                            'logging_context': self.get_logging_context()
                        }
                    )
        except Exception as e:
            self._logger.log_structured(
                level="ERROR",
                message=f"\uD83D\uDCA5 Stream Error: {str(e)}",
                task_id=task_id,
                context_id=session_id,
                extra={"agent_name": self.__class__.__name__, "error_type": type(e).__name__}
            )
            yield AgentResponse(
                response_type='error',
                is_task_complete=True,
                require_user_input=False,
                content=f'Error during streaming: {str(e)}',
                error=str(e),
                metadata={
                    'session_id': session_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'error_type': type(e).__name__,
                    'step_count': step_count
                }
            )
        self._logger.log_structured(
            level="INFO",
            message=f"DEBUG: [stream] END session_id={session_id}, task_id={task_id}",
            task_id=task_id,
            context_id=session_id,
            extra={"agent_name": self.__class__.__name__}
        )

