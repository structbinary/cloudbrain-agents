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
from typing import Literal, cast, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import json
from langgraph.types import Command
from planner_agent.models.agent_state import MultiAgentState
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentCapability
from planner_agent.utils.logger import log_sync, log_async
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
    def __init__(self, llm_model, enable_visual_logging=True, websocket=None, stream_output=None, **kwargs):
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
        
        super().__init__(config=agent_config, **kwargs)

    @log_sync
    def _initialize_agent(self, **kwargs):
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
        await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.node_stream] START state={state}", "INFO")
        # HITL resume logic
        if state.status == "input_required":
            if state.resume_value is not None:
                human_answer = state.resume_value
                state.user_query = human_answer
                state.resume_value = None
                state.status = None
            else:
                interrupt({"question": state.question})
                await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.node_stream] END (interrupt) state={state}", "INFO")
                return state
        query = cast(str, getattr(state, "user_query", ""))
        session_id = cast(str, getattr(state, "context_id", ""))
        task_id = cast(str, getattr(state, "task_id", ""))
        task_list = cast(List[str], getattr(state, "refined_task_list", []))
        result = await self.stream(query, session_id, task_id, task_list, state)
        await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.node_stream] END state={result}", "INFO")
        return result

    @log_async
    async def stream(self, query: str, session_id: str, task_id: str, task_list: List[str], state: MultiAgentState) -> MultiAgentState:
        await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.stream] START query={query}, session_id={session_id}, task_id={task_id}, task_list={task_list}, state={state}", "INFO")
        try:
            prompt_template = self._create_prompt_template()
            async with create_mcp_client(
                host=self._a2a_planner_config._config["AGENTS_MCP_SERVER_HOST"],
                port=self._a2a_planner_config._config.get("AGENTS_MCP_SERVER_PORT", 8080),
                transport=self._a2a_planner_config._config.get("AGENTS_MCP_SERVER_TRANSPORT", "sse")
            ) as mcp_client:
                tools = mcp_client.get_tools()
                await self.log_enhanced(f"Agent selection Partial: {query}", "INFO")
                await self.log_enhanced(f"Tools list: {tools}", "INFO")

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
                await self.log_enhanced(f"🔍 Agent selection response: {response}", "INFO")
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
                await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.stream] structured_response={structured_response}", "INFO")
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
                await self.log_enhanced(f"DEBUG: [a2a_agent_mapper.stream] END state={state}", "INFO")
                return state
        except Exception as e:
            tb = traceback.format_exc()
            await self.log_enhanced(f"💥 Stream Error: {str(e)}\nTraceback:\n{tb}", "ERROR")
            state.next = "__end__"
            state.status = "failed"
            state.final_response = {
                "error": str(e),
                "traceback": tb,
                "original_query": query,
                "status": "failed"
            }
            return state

