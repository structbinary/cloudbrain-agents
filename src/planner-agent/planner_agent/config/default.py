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

class DefaultConfig:
    """Default configuration for the planner agent."""
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1000
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "planner_agent.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    AGENTS_MCP_SERVER_HOST: str = "localhost"
    AGENTS_MCP_SERVER_PORT: int = 8080
    AGENTS_MCP_SERVER_TRANSPORT: str = "sse"
    AGENTS_MCP_SERVER_DISABLED: bool = False
    AGENTS_MCP_SERVER_AUTO_APPROVE: list = []