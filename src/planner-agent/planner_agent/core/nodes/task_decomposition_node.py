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

import json
from typing import Any, Optional
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from planner_agent.models.agent_state import MultiAgentState, TaskDecomposition
from planner_agent.prompts.prompts import PLANNER_TASK_DECOMPOSITION_PROMPT
from planner_agent.utils.logger import log_async, AgentLogger
from planner_agent.core.base_agent import MultiAgentCoordinator, AgentConfig, AgentCapability
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

class TaskDecompositionNode(MultiAgentCoordinator):
    """
    Node responsible for task decomposition in the multi-agent planning workflow.
    Extracted from MultiAgentPlanner for separation of business logic.
    """
    def __init__(self, llm_model: Any, logger: Optional[AgentLogger] = None, **kwargs):
        self._llm_model = llm_model
        self._logger = logger if logger is not None else AgentLogger("task_decomposition_node")
        agent_config = AgentConfig(
            name="task_decomposition_node",
            description="Task Decomposition Node",
            capabilities=[AgentCapability.STREAMING],
            content_types=["text/plain", "application/json"],
            enable_metrics=True,
            enable_logging=True,
            log_level="INFO"
        )
        super().__init__(config=agent_config, **kwargs)

    @log_async
    async def node_stream(self, state: MultiAgentState) -> MultiAgentState:
        self._logger.log_structured(
            level="INFO",
            message="[task_decomposition_node] START",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__, "state": str(state)}
        )
        self._logger.log_structured(
            level="INFO",
            message="Starting Task Decomposition Phase",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__}
        )

        # If resuming from human input
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
                    message="[task_decomposition_node] END (interrupt)",
                    task_id=getattr(state, 'task_id', None),
                    context_id=getattr(state, 'context_id', None),
                    extra={"agent_name": self.__class__.__name__, "state": str(state)}
                )
                return state

        prompt = ChatPromptTemplate.from_template(PLANNER_TASK_DECOMPOSITION_PROMPT)
        decomposition_agent = create_react_agent(
            self._llm_model,
            checkpointer=memory,
            prompt=PLANNER_TASK_DECOMPOSITION_PROMPT,
            tools=[],
        )
        user_query = getattr(state, "user_query", None) or ""
        self._logger.log_structured(
            level="INFO",
            message=f"user_query={user_query}",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__}
        )
        response = await decomposition_agent.ainvoke({'messages': [('user', user_query)]})
        self._logger.log_structured(
            level="INFO",
            message=f"LLM raw response: {response}",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__}
        )
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
        self._logger.log_structured(
            level="INFO",
            message=f"structured_response={structured_response}",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__}
        )
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
        self._logger.log_structured(
            level="INFO",
            message="[task_decomposition_node] END",
            task_id=getattr(state, 'task_id', None),
            context_id=getattr(state, 'context_id', None),
            extra={"agent_name": self.__class__.__name__, "state": str(state)}
        )
        return state

    async def stream(self, *args, **kwargs):
        raise NotImplementedError("stream is not implemented for TaskDecompositionNode. Use node_stream instead.") 