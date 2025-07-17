
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
Planner Agent - A2A Server Entry Point

This is the main entry point for the Planner Agent service.
It initializes the A2A server with proper configuration and logging.
"""

import json
import logging
import sys

from pathlib import Path
from planner_agent.utils.exceptions import ConfigError
from planner_agent.utils.logger import AgentLogger

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCard
from planner_agent.core.agent_manager import MultiAgentPlanner
from planner_agent.core.agent_executor import GenericAgentExecutor


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10101)
@click.option('--agent-card', 'agent_card')
def main(host: str, port: int, agent_card: str) -> None:
    """
    Main entry point for the Planner Agent server.
    """
    logger = AgentLogger("PLANNER_AGENT")
    try:
        if not agent_card:
            raise ConfigError('Agent card is required')
        with Path(agent_card).open() as file:
            data = json.load(file)
        agent_card_obj: AgentCard = AgentCard(**data)
        client: httpx.AsyncClient = httpx.AsyncClient()
        request_handler: DefaultRequestHandler = DefaultRequestHandler(
            agent_executor=GenericAgentExecutor(agent=MultiAgentPlanner()),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(client),
        )
        server: A2AStarletteApplication = A2AStarletteApplication(
            agent_card=agent_card_obj, http_handler=request_handler
        )
        logger._log_to_console(f'Starting Planner Agent server on {host}:{port}', level="INFO")
        logger._log_to_file(f'Starting Planner Agent server on {host}:{port}', level="INFO")
        uvicorn.run(server.build(), host=host, port=port)
    except FileNotFoundError:
        logger._log_to_console(f"Error: File '{agent_card}' not found.", level="ERROR")
        logger._log_to_file(f"Error: File '{agent_card}' not found.", level="ERROR")
        sys.exit(1)
    except json.JSONDecodeError:
        logger._log_to_console(f"Error: File '{agent_card}' contains invalid JSON.", level="ERROR")
        logger._log_to_file(f"Error: File '{agent_card}' contains invalid JSON.", level="ERROR")
        sys.exit(1)
    except Exception as e:
        logger._log_to_console(f'An error occurred during server startup: {e}', level="ERROR")
        logger._log_to_file(f'An error occurred during server startup: {e}', level="ERROR")
        sys.exit(1)
if __name__ == '__main__':
    main()