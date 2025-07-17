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

from colorama import Fore, Style
from enum import Enum
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from functools import wraps
import functools
import inspect
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentColor(Enum):
    CLOUD_CONFIG_GENERATOR = Fore.LIGHTBLUE_EX
    GENERATOR = Fore.YELLOW
    WEBSEARCH = Fore.LIGHTGREEN_EX
    ROUTER = Fore.MAGENTA
    REVIEWER = Fore.CYAN
    REVISOR = Fore.LIGHTWHITE_EX
    MASTER = Fore.LIGHTYELLOW_EX
    SEARCHER = Fore.LIGHTGREEN_EX
    BASE = Fore.WHITE

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AgentLogger:
    """Logger class for agent output with color encoding and multiple output methods."""
    
    def __init__(self, agent_name: str = "BASE") -> None:
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self.websocket = None
        self.stream_output = None
        
    def set_websocket(self, websocket: Any, stream_output: Callable) -> None:
        """Set websocket and stream output function."""
        self.websocket = websocket
        self.stream_output = stream_output
        
    def _get_color(self, agent: str) -> str:
        """Get color for agent."""
        try:
            return AgentColor[agent].value
        except KeyError:
            return AgentColor.BASE.value
            
    def _format_log_entry(self, message: str, level: str = "INFO") -> Dict[str, Any]:
        """Format log entry."""
        return {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "message": message,
            "level": level
        }
        
    def _log_to_console(self, message: str, level: str = "INFO") -> None:
        """Log to console with color."""
        color = self._get_color(self.agent_name)
        print(f"{color}{self.agent_name}: {message}{Style.RESET_ALL}")
        
    def _log_to_file(self, message: str, level: str = "INFO") -> None:
        """Log to file."""
        log_method = getattr(self.logger, level.lower())
        log_method(f"[{self.agent_name}] {message}")
        
    async def _log_to_websocket(self, message: str) -> None:
        """Log to websocket."""
        if self.websocket and self.stream_output:
            await self.stream_output("logs", self.agent_name, message, self.websocket)
            
    async def log(self, message: str, level: str = "INFO") -> None:
        """Log message to all configured outputs."""
        # Log to console
        self._log_to_console(message, level)
        
        # Log to file
        self._log_to_file(message, level)
        
        # Log to websocket if available
        await self._log_to_websocket(message)
        
    @staticmethod
    def format_log_message(message: str, level: str = "INFO") -> str:
        """Format a log message."""
        return f"[{level}] {message}"
        
    @classmethod
    def create_logger(cls, agent_name: str) -> 'AgentLogger':
        """Create a new logger instance."""
        return cls(agent_name)

def get_log_context(args: Any, kwargs: Any, func: Optional[Callable] = None) -> tuple[Optional[str], Optional[Any], Optional[Any]]:
    agent_name = None
    task_id = None
    context_id = None
    session_id = None

    # Try to get from self
    if args:
        self_obj = args[0]
        agent_name = getattr(self_obj, 'name', None) or getattr(self_obj, 'agent_name', None)
        task_id = getattr(self_obj, 'task_id', None)
        context_id = getattr(self_obj, 'context_id', None)
        session_id = getattr(self_obj, 'session_id', None)

    # Try to get from kwargs
    task_id = kwargs.get('task_id', task_id)
    context_id = kwargs.get('context_id', context_id)
    session_id = kwargs.get('session_id', session_id)

    # Try to get from positional args using function signature
    if func is not None:
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        for i, param in enumerate(params):
            if param == 'task_id' and len(args) > i:
                task_id = args[i]
            if param == 'context_id' and len(args) > i:
                context_id = args[i]
            if param == 'session_id' and len(args) > i:
                session_id = args[i]

    # Try to get from dict or Pydantic model argument (e.g., state)
    if args:
        for arg in args[1:]:  # skip self
            # If it's a dict
            if isinstance(arg, dict):
                task_id = arg.get('task_id', task_id)
                context_id = arg.get('context_id', context_id)
                session_id = arg.get('session_id', session_id)
            # If it's a Pydantic model
            elif hasattr(arg, '__fields__'):
                task_id = getattr(arg, 'task_id', task_id)
                context_id = getattr(arg, 'context_id', context_id)
                session_id = getattr(arg, 'session_id', session_id)

    # If context_id is still None, but session_id is present, use session_id as context_id
    if context_id is None and session_id is not None:
        context_id = session_id

    return agent_name, task_id, context_id

def log_json(log_type: str, message: str, agent_name: Optional[str] = None, task_id: Optional[Any] = None, context_id: Optional[Any] = None) -> None:
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent_name": agent_name,
        "log_type": log_type,
        "message": message,
        "task_id": task_id,
        "context_id": context_id
    }
    print(json.dumps(log_entry))

def log_sync(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        agent_name, task_id, context_id = get_log_context(args, kwargs, func)
        log_json("INFO", f"Starting {func.__name__}", agent_name, task_id, context_id)
        try:
            result = func(*args, **kwargs)
            log_json("INFO", f"Completed {func.__name__}", agent_name, task_id, context_id)
            return result
        except Exception as e:
            log_json("ERROR", f"Error in {func.__name__}: {str(e)}", agent_name, task_id, context_id)
            raise
    return wrapper

def log_async(func: Callable) -> Callable:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        agent_name, task_id, context_id = get_log_context(args, kwargs, func)
        log_json("INFO", f"Starting {func.__name__}", agent_name, task_id, context_id)
        try:
            result = await func(*args, **kwargs)
            log_json("INFO", f"Completed {func.__name__}", agent_name, task_id, context_id)
            return result
        except Exception as e:
            log_json("ERROR", f"Error in {func.__name__}: {str(e)}", agent_name, task_id, context_id)
            raise

    @functools.wraps(func)
    async def asyncgen_wrapper(*args, **kwargs):
        agent_name, task_id, context_id = get_log_context(args, kwargs, func)
        log_json("INFO", f"Starting {func.__name__}", agent_name, task_id, context_id)
        try:
            async for value in func(*args, **kwargs):
                yield value
            log_json("INFO", f"Completed {func.__name__}", agent_name, task_id, context_id)
        except Exception as e:
            log_json("ERROR", f"Error in {func.__name__}: {str(e)}", agent_name, task_id, context_id)
            raise

    if inspect.isasyncgenfunction(func):
        return asyncgen_wrapper
    else:
        return async_wrapper