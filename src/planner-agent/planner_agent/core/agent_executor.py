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

import logging
import inspect

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
    Part,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from planner_agent.core.base_agent import BaseAgent
from typing import cast, Any


logger = logging.getLogger(__name__)


class GenericAgentExecutor(AgentExecutor):
    """AgentExecutor used by the tragel agents with JSON-RPC 2.0 validation support."""

    def __init__(self, agent: BaseAgent) -> None:
        self.agent: BaseAgent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.info(f'Executing agent {self.agent.name}')
        error = self._validate_request(context)
        if error:
            logger.error('Validation error in request context')
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        logger.debug(f'User query: {query}')

        task = context.current_task

        if not task:
            logger.info('No current task found, creating new task')
            if context.message is None:
                logger.error('No message provided for new task')
                raise ServerError(error=InvalidParamsError())
            task = new_task(context.message)
            logger.info(f'Created new task with id: {task.id}, contextId: {task.contextId}')  # DEBUG log
            await event_queue.enqueue_event(task)
            logger.debug(f'Enqueued new task: {task}')

        updater = TaskUpdater(event_queue, task.id, task.contextId)
        logger.info(f'Starting agent stream for task_id={task.id}, context_id={task.contextId}')

        try:
            # Ensure self.agent.stream is an async generator, or await if it's a coroutine returning one
            agent_stream = self.agent.stream(query, task.contextId, task.id)
            if not inspect.isasyncgen(agent_stream):
                agent_stream = await agent_stream  # type: ignore
            async for item in agent_stream:  # type: ignore
                # logger.debug(f'Received item from agent stream: {item}')
                # Forward agent-to-agent events directly to the event queue
                root = getattr(item, 'root', None)
                if root is not None and isinstance(root, SendStreamingMessageSuccessResponse):
                    event = root.result
                    if isinstance(
                        event,
                        (TaskStatusUpdateEvent, TaskArtifactUpdateEvent),
                    ):
                        logger.info(f'Enqueuing event from agent: {event}')
                        await event_queue.enqueue_event(event)
                    continue
                
                is_task_complete = item.is_task_complete
                require_user_input = item.require_user_input
                # logger.debug(f'is_task_complete={is_task_complete}, require_user_input={require_user_input}')
                
                # Map custom status to A2A TaskState enum
                custom_status = getattr(item, 'metadata', {}).get('status', 'working')
                task_state = self._map_status_to_task_state(custom_status)
                # logger.debug(f'Mapped custom status "{custom_status}" to task_state {task_state}')

                if is_task_complete:
                    logger.info('Task is marked as complete by agent')
                    if item.response_type == 'data':
                        data_part: Part = cast(Part, DataPart(data=item.content))
                    else:
                        text_part: Part = cast(Part, TextPart(text=item.content))

                    logger.info('Adding artifact to updater')
                    if item.response_type == 'data':
                        await updater.add_artifact(
                            [data_part],
                            name=f'{self.agent.name}-result',
                        )
                    else:
                        await updater.add_artifact(
                            [text_part],
                            name=f'{self.agent.name}-result',
                        )
                    logger.info('Sending final status update: TaskState.completed')
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            "Task completed successfully.",
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    logger.info('Calling updater.complete()')
                    await updater.complete()
                    logger.info('Updater.complete() finished, breaking stream loop')
                    break
                if require_user_input:
                    logger.info('Agent requires user input, updating status to input_required')
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item.content,
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    logger.info('Status updated to input_required, breaking stream loop')
                    break
                logger.info(f'Updating status to {task_state}')
                await updater.update_status(
                    task_state,
                    new_agent_text_message(
                        item.content,
                        task.contextId,
                        task.id,
                    ),
                )
                # logger.debug('Status update sent')
        except Exception as e:
            logger.error(f'Exception in agent executor stream: {e}', exc_info=True)
            raise


    def _validate_request(self, context: RequestContext) -> bool:
        return False

    def _map_status_to_task_state(self, custom_status: str) -> TaskState:
        """Map custom status strings to A2A TaskState enum values."""
        status_mapping: dict[str, TaskState] = {
            'working': TaskState.working,
            'input_required': TaskState.input_required,
            'completed': TaskState.completed,
            'failed': TaskState.failed,
            'error': TaskState.failed,
            'submitted': TaskState.submitted,
            'canceled': TaskState.canceled,
            'rejected': TaskState.rejected,
            'auth_required': TaskState.auth_required,
            'unknown': TaskState.unknown
        }
        return status_mapping.get(custom_status, TaskState.working)

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())