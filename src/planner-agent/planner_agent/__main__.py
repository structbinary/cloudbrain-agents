#!/usr/bin/env python3
"""
Planner Agent - A2A Server Entry Point

This is the main entry point for the Planner Agent service.
It initializes the A2A server with proper configuration and logging.
"""

import json
import logging
import sys

from pathlib import Path

import click
import httpx
from loguru import logger
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCard
from planner_agent.core.agents.multi_agent_planner import MultiAgentPlanner
from planner_agent.core.agent_executor import GenericAgentExecutor


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10101)
@click.option('--agent-card', 'agent_card')
def main(host, port, agent_card):
    """
    Main entry point for the Planner Agent server.
    """
    try:
        if not agent_card:
            raise ValueError('Agent card is required')
        with Path.open(agent_card) as file:
            data = json.load(file)
        agent_card = AgentCard(**data)
        client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=GenericAgentExecutor(agent=MultiAgentPlanner()),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        logger.info(f'Starting Planner Agent server on {host}:{port}')
        uvicorn.run(server.build(), host=host, port=port)
    except FileNotFoundError:
        logger.error(f"Error: File '{agent_card}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Error: File '{agent_card}' contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)
if __name__ == '__main__':
    main()