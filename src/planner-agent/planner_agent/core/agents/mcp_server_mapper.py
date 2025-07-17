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
from typing import Literal, cast, List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import json
from langgraph.types import Command
from planner_agent.models.agent_state import MultiAgentState
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentCapability
from planner_agent.utils.logger import log_sync, log_async, AgentLogger
from planner_agent.prompts.prompts import AGENT_SKILL_TO_MCP_SERVER_MAPPER_PROMPT
from planner_agent.utils.mcp_agent_client import create_mcp_client
from planner_agent.config import Config
from planner_agent.models.agent_state import MCPNodeResponse
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


def extract_json_from_code_block(s: str) -> str:
    # Remove triple backticks and optional 'json' after them
    s = s.strip()
    if s.startswith("```"):
        # Remove the first line (```json or ```)
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove the last line if it's a code block marker
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()

class MCPNodeMapper(MultiAgentCoordinator):
    """
    This node is responsible for mapping the right mcp server for a given a2a agent card.
    """
    def __init__(self, llm_model: Any, enable_visual_logging: bool = True, websocket: Optional[Any] = None, stream_output: Optional[Any] = None, **kwargs: Any) -> None:
        agent_config = AgentConfig(
            name="mcp_server_mapper",
            description="MCP Server Mapper",
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
        self._logger = AgentLogger("mcp_server_mapper")
        
        super().__init__(config=agent_config, **kwargs)

    @log_sync
    def _initialize_agent(self, **kwargs: Any) -> None:
        """Initialize multi-agent planner specific components."""
        if self._enable_visual_logging:
            self.setup_enhanced_logging(
                websocket=self._websocket,
                stream_output=self._stream_output
            )
        self._mcp_planner_config = Config()
        # self.available_tools = self._mcp_client.get_available_tools()



    @log_sync
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template for the agent selection chain of thought."""
        return ChatPromptTemplate.from_template(AGENT_SKILL_TO_MCP_SERVER_MAPPER_PROMPT)

    async def node_stream(self, state: MultiAgentState) -> MultiAgentState:
        self._logger.log_structured(
            level="INFO",
            message="[mcp_server_mapper.node_stream] START",
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
                    message="[mcp_server_mapper.node_stream] END (interrupt)",
                    task_id=getattr(state, 'task_id', None),
                    context_id=getattr(state, 'context_id', None),
                    extra={"agent_name": self.__class__.__name__, "state": str(state)}
                )
                return state
        state_dict = state.model_dump() if hasattr(state, 'model_dump') else dict(state)
        for k, v in state_dict.items():
            self._logger.log_structured(
                level="INFO",
                message=f"[MCP_SERVER_MAPPER] state[{k}] = {v}",
                task_id=getattr(state, 'task_id', None),
                context_id=getattr(state, 'context_id', None),
                extra={"agent_name": self.__class__.__name__}
            )
        query = cast(str, getattr(state, "user_query", ""))
        session_id = cast(str, getattr(state, "context_id", ""))
        task_id = cast(str, getattr(state, "task_id", ""))
        selected_agent = cast(List[Dict[str, Any]], getattr(state, "selected_agent", []))
        task_list = cast(List[str], getattr(state, "refined_task_list", []))
        result = await self.stream(query, session_id, task_id, task_list, selected_agent, state)
        self._logger.log_structured(
            level="INFO",
            message="[mcp_server_mapper.node_stream] END",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__, "result": str(result)}
        )
        return result

    @log_async
    async def stream(self, query: str, session_id: str, task_id: str, task_list: List[str], selected_agent: List[Dict[str, Any]], state: MultiAgentState) -> MultiAgentState:
        self._logger.log_structured(
            level="INFO",
            message="[mcp_server_mapper.stream] START",
            task_id=task_id,
            context_id=session_id,
            extra={"agent_name": self.__class__.__name__, "query": query, "task_list": str(task_list), "selected_agent": str(selected_agent), "state": str(state)}
        )
        try:
            prompt_template = self._create_prompt_template()
            async with create_mcp_client(
                host=self._mcp_planner_config._config["AGENTS_MCP_SERVER_HOST"],
                port=self._mcp_planner_config._config.get("AGENTS_MCP_SERVER_PORT", 8080),
                transport=self._mcp_planner_config._config.get("AGENTS_MCP_SERVER_TRANSPORT", "sse")
            ) as mcp_client:
                tools = mcp_client.get_tools()
                self._logger.log_structured(
                    level="INFO",
                    message="MCP Server selection Partial",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "query": query}
                )
                self._logger.log_structured(
                    level="INFO",
                    message="MCP Server tools list",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "tools": str(tools)}
                )

                # Build agent_list with agent_name and their capability_list (skill ids)
                agent_list = []
                for agent_dict in selected_agent:
                    agent_card = agent_dict.get("agent_card", {})
                    agent_name = agent_dict.get("agent_name")
                    skills = agent_card.get("skills", [])
                    capability_list = []
                    for skill in skills:
                        skill_id = skill.get("id")
                        if skill_id:
                            capability_list.append(skill_id)
                    agent_list.append({
                        "agent_name": agent_name,
                        "capability_list": capability_list
                    })

                self._logger.log_structured(
                    level="INFO",
                    message="Extracted agent_list",
                    task_id=task_id,
                    context_id=session_id,
                    extra={"agent_name": self.__class__.__name__, "agent_list": str(agent_list)}
                )
                system_content = prompt_template.format(agent_list=agent_list)
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
                    "agent_list": agent_list
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
                    content_str = extract_json_from_code_block(content_str)
                    content_data = json.loads(content_str)
                    structured_response = MCPNodeResponse(**content_data)
                else:
                    content = response
                    content_str = str(content) if not isinstance(content, str) else content
                    content_data = json.loads(content_str)
                    structured_response = MCPNodeResponse(**content_data)
                agent_responses = structured_response.agent_responses
                status = structured_response.status
                question = structured_response.question

                # Combine selected_agent and agent_responses into a final mapping list
                final_list = []
                if selected_agent and agent_responses:
                    for sel, resp in zip(selected_agent, agent_responses):
                        final_list.append({
                            "task": sel.get("task"),
                            "selected_agent": sel.get("agent_name"),
                            "agent_card": sel.get("agent_card"),
                            "mcp_server_details": resp.get("mcp_servers_details"),
                        })
                # state.final_agent_server_mapping = final_list  # type: ignore[attr-defined]
                # Debug: Print the agent_responses structure
                # print(f"[DEBUG] agent_responses: {agent_responses}")
                # print(f"[DEBUG] agent_responses type: {type(agent_responses)}")
                # if agent_responses and isinstance(agent_responses, list):
                #     print(f"[DEBUG] First response: {agent_responses[0]}")
                #     print(f"[DEBUG] First response keys: {agent_responses[0].keys() if hasattr(agent_responses[0], 'keys') else 'No keys method'}")
                # Extract mcp_server_details from the first agent response if present
                if status == "completed" and not question:
                    state.next = "__end__"
                    state.status = "completed"
                    state.mcp_servers = final_list
                elif status == "completed" and question:
                    state.next = "__end__"
                    state.status = "input_required"
                    state.question = question
                    state.final_response = structured_response.model_dump()
                else:
                    state.next = "__end__"
                    state.status = "failed"
                    state.final_response = structured_response.model_dump()
                self._logger.log_structured(
                    level="INFO",
                    message="[mcp_server_mapper.stream] END",
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
            self._logger.log_structured(
                level="INFO",
                message="[mcp_server_mapper.stream] END (exception)",
                task_id=task_id,
                context_id=session_id,
                extra={"agent_name": self.__class__.__name__, "state": str(state)}
            )
            return state
