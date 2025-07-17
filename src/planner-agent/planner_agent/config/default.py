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