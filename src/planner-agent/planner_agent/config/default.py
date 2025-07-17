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
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0.0
    LLM_MAX_TOKENS = 1000
    LOG_LEVEL = "INFO"
    LOG_FILE = "planner_agent.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    AGENTS_MCP_SERVER_HOST = "localhost"
    AGENTS_MCP_SERVER_PORT = 8080
    AGENTS_MCP_SERVER_TRANSPORT = "sse"
    AGENTS_MCP_SERVER_DISABLED = False
    AGENTS_MCP_SERVER_AUTO_APPROVE = []