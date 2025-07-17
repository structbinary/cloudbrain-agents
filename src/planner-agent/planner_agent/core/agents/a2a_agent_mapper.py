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

import traceback
from typing import Literal, cast, List, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import json
from langgraph.types import Command
from planner_agent.models.agent_state import MultiAgentState
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentCapability
from planner_agent.utils.logger import log_sync, log_async, AgentLogger
from planner_agent.prompts.prompts import TASK_TO_AGENT_MAPPER_PROMPT
from planner_agent.utils.mcp_agent_client import create_mcp_client
from planner_agent.config import Config
from planner_agent.models.agent_state import A2AAgentCardResponse
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

class A2AAgentCardMapper(MultiAgentCoordinator):
    """
    This node is responsible for mapping the right a2a agent card for a given task.
    """
    def __init__(self, llm_model: Any, enable_visual_logging: bool = True, websocket: Optional[Any] = None, stream_output: Optional[Any] = None, **kwargs: Any) -> None:
        agent_config = AgentConfig(
            name="a2a_agent_mapper",
            description="A2A Agent Mapper",
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
        self._enable_visual_logging = enable_visual_logging
        self._websocket = websocket
        self._stream_output = stream_output
        self._llm_model = llm_model
        self._logger = AgentLogger("a2a_agent_mapper")
        
        super().__init__(config=agent_config, **kwargs)

    @log_sync
    def _initialize_agent(self, **kwargs: Any) -> None:
        """Initialize multi-agent planner specific components."""
        if self._enable_visual_logging:
            self.setup_enhanced_logging(
                websocket=self._websocket,
                stream_output=self._stream_output
            )
        self._a2a_planner_config = Config()
        # self.available_tools = self._mcp_client.get_available_tools()



    @log_sync
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template for the agent selection chain of thought."""
        return ChatPromptTemplate.from_template(TASK_TO_AGENT_MAPPER_PROMPT)

    
    @log_async
    async def node_stream(self, state: MultiAgentState) -> MultiAgentState:
        self._logger.log_structured(
            level="INFO",
            message="[a2a_agent_mapper.node_stream] START",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__, "state": str(state)}
        )
        # HITL resume logic
        if state.status == "input_required":
            if state.resume_value is not None:
                human_answer = state.resume_value
                state.user_query = human_answer
                state.resume_value = None
                state.status = None
            else:
                interrupt({"question": state.question})
                self._logger.log_structured(
                    level="INFO",
                    message="[a2a_agent_mapper.node_stream] END (interrupt)",
                    task_id=getattr(state, 'task_id', None),
                    context_id=getattr(state, 'context_id', None),
                    extra={"agent_name": self.__class__.__name__, "state": str(state)}
                )
                return state
        query = cast(str, getattr(state, "user_query", ""))
        session_id = cast(str, getattr(state, "context_id", ""))
        task_id = cast(str, getattr(state, "task_id", ""))
        task_list = cast(List[str], getattr(state, "refined_task_list", []))
        result = await self.stream(query, session_id, task_id, task_list, state)
        self._logger.log_structured(
            level="INFO",
            message="[a2a_agent_mapper.node_stream] END",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__, "result": str(result)}
        )
        return result

    @log_async
    async def stream(self, query: str, session_id: str, task_id: str, task_list: List[str], state: MultiAgentState) -> MultiAgentState:
        self._logger.log_structured(
            level="INFO",
            message="[a2a_agent_mapper.stream] START",
            task_id=task_id,
            context_id=session_id,
            extra={"agent_name": self.__class__.__name__, "query": query, "task_list": str(task_list), "state": str(state)}
        )
        try:
            prompt_template = self._create_prompt_template()
            async with create_mcp_client(
                host=self._a2a_planner_config._config["AGENTS_MCP_SERVER_HOST"],
                port=self._a2a_planner_config._config.get("AGENTS_MCP_SERVER_PORT", 8080),
                transport=self._a2a_planner_config._config.get("AGENTS_MCP_SERVER_TRANSPORT", "sse")
            ) as mcp_client:
                tools = mcp_client.get_tools()
                self._logger.log_structured(
                    level="INFO",
                    message="Agent selection Partial",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "query": query}
                )
                self._logger.log_structured(
                    level="INFO",
                    message="Tools list",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "tools": str(tools)}
                )

                # Fill the prompt template with variables at runtime (static prompt)
                system_content = prompt_template.format(tasks_list=task_list)
                if not isinstance(system_content, str) and hasattr(system_content, 'to_string'):
                    system_content = system_content.to_string()
                if not isinstance(system_content, str):
                    system_content = str(system_content)

                agent = create_react_agent(
                    model=self._llm_model,
                    tools=tools,
                    prompt=system_content  # Pass the filled prompt as a static string
                )

                agent_state = {
                    "tasks_list": task_list
                }
                response = await agent.ainvoke(agent_state)
                self._logger.log_structured(
                    level="INFO",
                    message="Agent selection response",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "response": str(response)}
                )
                message = response['messages'][-1]
                if isinstance(message, AIMessage):
                    content = message.content
                    content_str = str(content) if not isinstance(content, str) else content
                    content_data = json.loads(content_str)
                    structured_response = A2AAgentCardResponse(**content_data)
                else:
                    content = response
                    content_str = str(content) if not isinstance(content, str) else content
                    content_data = json.loads(content_str)
                    structured_response = A2AAgentCardResponse(**content_data)
                refined_task_list = structured_response.refined_task_list
                selected_agent = structured_response.selected_agent
                status = structured_response.status
                question = structured_response.question
                self._logger.log_structured(
                    level="INFO",
                    message="[a2a_agent_mapper.stream] structured_response",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "structured_response": str(structured_response)}
                )
                # --- Set next node ---
                if status == "completed" and not question:
                    state.next = "mcp_server_mapper"
                    state.status = "working"
                    state.selected_agent = selected_agent
                elif status == "completed" and question:
                    state.next = "__end__"
                    state.status = "input_required"
                    state.question = question
                    state.final_response = response
                else:
                    state.next = "__end__"
                    state.status = "failed"
                    state.final_response = response
                self._logger.log_structured(
                    level="INFO",
                    message="[a2a_agent_mapper.stream] END",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "state": str(state)}
                )
                return state
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.log_structured(
                level="ERROR",
                message="Stream Error",
                task_id=task_id,
                context_id=session_id,
                extra={"agent_name": self.__class__.__name__, "error": str(e), "traceback": tb}
            )
            state.next = "__end__"
            state.status = "failed"
            state.final_response = {
                "error": str(e),
                "traceback": tb,
                "original_query": query,
                "status": "failed"
            }
            return state

