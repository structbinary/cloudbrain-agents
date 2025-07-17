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

from typing import Any, AsyncIterable, Dict, List, Literal, Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
import json
from planner_agent.config import Config
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentResponse, AgentCapability
from planner_agent.core.llm.llm_provider import LLMProvider
from planner_agent.models.agent_state import MultiAgentState, TaskDecomposition
from planner_agent.prompts.prompts import PLANNER_TASK_DECOMPOSITION_PROMPT
from planner_agent.utils.logger import log_sync, log_async
from planner_agent.core.agents.a2a_agent_mapper import A2AAgentCardMapper
from planner_agent.core.agents.mcp_server_mapper import MCPNodeMapper
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
    def __init__(self, enable_visual_logging=True, websocket=None, stream_output=None, **kwargs):
        # logger.info('Initializing MultiAgentPlanner with node-specific tool bindings')

        # Create agent configuration
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
        
        # Store enhanced logging parameters
        self._enable_visual_logging = enable_visual_logging
        self._websocket = websocket
        self._stream_output = stream_output
        
        # Initialize the base coordinator
        super().__init__(config=agent_config, **kwargs)

        # logger.info('MultiAgentPlanner initialization complete')

    @log_sync
    def _initialize_agent(self, **kwargs) -> None:
        """Initialize multi-agent planner specific components."""
        # logger.info("Initializing MultiAgentPlanner components")
        
        # Set up enhanced logging for visual debugging and streaming
        if self._enable_visual_logging:
            self.setup_enhanced_logging(
                websocket=self._websocket,
                stream_output=self._stream_output
            )
        
        # Initialize LLM
        try:
            self._planner_config = Config()
            self.model = LLMProvider.create_llm(**self._planner_config.get_llm_config())
            # logger.info('LLM initialized successfully')
        except Exception as e:
            # logger.error(f'Failed to initialize LLM: {e}')
            raise ValueError(f"LLM initialization failed: {e}")
        
        self.a2a_agent_mapper = A2AAgentCardMapper(self.model)
        self.mcp_server_mapper = MCPNodeMapper(self.model)
        # Build multi-agent graph
        self.graph = self._build_graph()

    def generic_branch(self, state: MultiAgentState, *args, **kwargs) -> str:
        return state.next if state.next is not None else "__end__"

    @log_sync
    def _build_graph(self):
        """Build the multi-agent state graph with node-specific tool bindings."""
        
        # Create the graph
        graph = StateGraph(MultiAgentState)
        
        # Add nodes with specific purposes
        graph.add_node("task_decomposition", self._task_decomposition_node)
        graph.add_node("a2a_agent_mapper", self.a2a_agent_mapper.node_stream)
        graph.add_node("mcp_server_mapper", self.mcp_server_mapper.node_stream)
        
        # Add edges with conditional routing
        graph.add_edge(START, "task_decomposition")
        graph.add_conditional_edges("task_decomposition", self.generic_branch)
        graph.add_conditional_edges("a2a_agent_mapper", self.generic_branch)
        graph.add_conditional_edges("mcp_server_mapper", self.generic_branch)
        
        # Compile with memory
        return graph.compile(checkpointer=memory)


    @log_async
    async def _task_decomposition_node(self, state: MultiAgentState) -> MultiAgentState:
        await self.log_enhanced(f"DEBUG: [task_decomposition_node] START state={state}", "INFO")
        await self.log_enhanced("🔍 Starting Task Decomposition Phase", "INFO")

        # If resuming from human input
        if state.status == "input_required":
            if state.resume_value is not None:
                # Use the human's answer and continue
                human_answer = state.resume_value
                state.user_query = human_answer
                state.resume_value = None  # Clear after use
                state.status = None  # Reset status to continue
                # Optionally, you may want to re-run the decomposition with the new input
                # (fall through to normal logic below)
            else:
                # Pause for human input
                interrupt({"question": state.question})
                await self.log_enhanced(f"DEBUG: [task_decomposition_node] END (interrupt) state={state}", "INFO")
                return state

        prompt = ChatPromptTemplate.from_template(PLANNER_TASK_DECOMPOSITION_PROMPT)
        decomposition_agent = create_react_agent(
            self.model,
            checkpointer=memory,
            prompt=PLANNER_TASK_DECOMPOSITION_PROMPT,
            tools=[],  # No tools needed for basic task decomposition
        )
        user_query = getattr(state, "user_query", None)
        if user_query is None:
            user_query = ""
        await self.log_enhanced(f"DEBUG: user_query={user_query}", "INFO")
        response = await decomposition_agent.ainvoke({'messages': [('user', user_query)]})
        await self.log_enhanced(f"DEBUG: LLM raw response: {response}", "INFO")
        message = response['messages'][-1]
        if isinstance(message, AIMessage):
            content = message.content
            content_str = str(content) if not isinstance(content, str) else content
            content_data = json.loads(content_str)
            structured_response = TaskDecomposition(**content_data)
        else:
            content = response
            content_str = str(content) if not isinstance(content, str) else content
            content_data = json.loads(content_str)
            structured_response = TaskDecomposition(**content_data)
        await self.log_enhanced(f"DEBUG: structured_response={structured_response}", "INFO")
        # --- Set next node ---
        if structured_response.status == "completed":
            state.next = "a2a_agent_mapper"
            state.refined_task_list = structured_response.task_list
            state.status = "working"
        elif structured_response.status == "input_required":
            state.next = "__end__"
            state.question = structured_response.question
            state.status = "input_required"
        else:
            state.next = "__end__"
            state.status = "failed"
            state.question = structured_response.question
        await self.log_enhanced(f"DEBUG: [task_decomposition_node] END state={state}", "INFO")
        return state


    @log_async
    async def stream(self, query_or_command, session_id: str, task_id: str) -> AsyncIterable[AgentResponse]:
        await self.log_enhanced(f"DEBUG: [stream] START session_id={session_id}, task_id={task_id}, query_or_command={query_or_command}", "INFO")
        """
        Stream method supporting both initial and resume calls for human-in-the-loop (HITL).
        - If query_or_command is a string: initial call (user query)
        - If query_or_command is a Command: resume call (user feedback)
        """
        await self.log_enhanced(f"🚀 Starting Multi-Agent Stream | Session: {session_id}... | Task: {task_id} | Query/Command: {query_or_command}", "INFO")

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
                await self.log_enhanced(f"DEBUG: [stream] step={step_count} item={item}", "INFO")

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
            await self.log_enhanced(f"💥 Stream Error: {str(e)}", "ERROR")
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
        await self.log_enhanced(f"DEBUG: [stream] END session_id={session_id}, task_id={task_id}", "INFO")

