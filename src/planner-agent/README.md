# Planner Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-2.0](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](https://opensource.org/licenses/GPL-2.0)
[![A2A Protocol](https://img.shields.io/badge/Protocol-A2A-green.svg)](https://github.com/google/a2a)
[![MCP Integration](https://img.shields.io/badge/Integration-MCP-orange.svg)](https://modelcontextprotocol.io/)

> **AI-Powered DevOps Orchestrator** - Intelligent task decomposition and multi-agent coordination for complex DevOps automation

The Planner Agent serves as the strategic brain in a multi-agent DevOps ecosystem, receiving complex queries from supervisor agents, decomposing them into actionable tasks, and intelligently mapping these tasks to specialized executor agents (AWS orchestrator, CI/CD agent, monitoring agent, etc.).

**Example Use Case:**  
A DevOps engineer submits a high-level request:  
“Deploy our new microservice to production with blue-green deployment, set up monitoring, and configure CI/CD.”  
The Planner Agent automatically:
- Breaks down the request into actionable tasks (e.g., infrastructure provisioning, Kubernetes deployment, monitoring setup, CI/CD pipeline configuration)
- Assigns each task to the most suitable specialized agent, which it identifies from the central registry service running on the MCP server
- Depending on the tasks and agent configuration, it also dynamically fetches the details of the MCP servers that will ultimately handle the corresponding requests
- Forwards the detailed execution plan to the Supervisor Agent
- The Supervisor Agent then coordinates with the appropriate executor agents to carry out each task

This enables teams to automate complex, multi-step DevOps workflows with minimal manual intervention, while maintaining a clear separation of planning and execution responsibilities.

## 🧩 Dependencies

To enable the Planner Agent to map user requests to the appropriate executor agents—and to determine which MCP server should supplement each executor agent for specific tasks—a central registry service is required. This registry holds details about available executor agents and which MCP servers are available for which tasks.

You can find setup and usage details in the agents-mcp-server README: <placeholder>

**Setup Steps:**
1. **Spin up the central registry server (agents-mcp-server)** before running the Planner Agent.
2. **Configure the MCP server connection details** in your `.env` file (see example variables below):
   - `AGENTS_MCP_SERVER_HOST`
   - `AGENTS_MCP_SERVER_PORT`
   - `AGENTS_MCP_SERVER_TRANSPORT`
   - `AGENTS_MCP_SERVER_DISABLED`
   - `AGENTS_MCP_SERVER_AUTO_APPROVE`

---

The second dependency is the LLM model provider. By default, the Planner Agent uses OpenAI models, but if you prefer to use a different LLM provider, please follow the onboarding guide:

- [Onboarding Guide: Adding a New LLM Provider](docs/ONBOARDING_LLM_PROVIDER.md)

This guide explains how to add and configure a new LLM provider for the Planner Agent.

---

## 🚀 Quick Start

1. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
2. **Create and activate a virtual environment with Python 3.12:**
   ```sh
   uv venv --python=3.12
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies from pyproject.toml:**
   ```sh
   uv pip install -e .
   ```
4. **Create a `.env` file and add the following environment variables:**
   ```sh
   OPENAI_API_KEY=XXXXXXXXX
   ```
   > **Note:** All available configuration options can be found in [`planner_agent/config/default.py`](planner_agent/config/default.py). You can set any of these options via your `.env` file to customize the Planner Agent's behavior.

---

This agent runs on the Google A2A protocol to communicate with other A2A agents. To enable this, you will need an agent card so that other clients or servers can interact with this service. 

**Run the server with your agent card (you can also specify host and port):**
```sh
uv run planner_agent --agent-card planner_agent/card/planner_agent.json --host 0.0.0.0 --port 10101
```
- `--host`: The host address to bind the server (default: `localhost`)
- `--port`: The port to run the server on (default: `10101`)

---

## 🧑‍💻 Interacting with the Planner Agent

To interact with the Planner Agent server, use the provided A2A client in [`planner_client`](planner_client). While further enhancements are planned, you can use it to communicate with the server as follows:

Start the client:
```sh
python planner_client/__main__.py
```

> **Note:** You can override the following options when running the client:
> - `--agent`: The URL of the Planner Agent server to connect to (default: `http://localhost:10101`)
> - `--session`: The session ID to use (default: `0`)
> - `--history`: Whether to show session history (default: `False`)
>
> **Example:**
> ```sh
> python planner_client/__main__.py --agent http://localhost:10101 --session 1 --history True
> ```

## 🎯 What is Planner Agent?

Planner Agent is a specialized agent designed for highly interactive, multi-agent frameworks. Its core expertise is to break down human requests into multiple actionable tasks and intelligently map each task to the most suitable executor agent. By offloading the responsibility of task decomposition and agent mapping from the supervisor or coordinator agent, Planner Agent enables a more scalable, modular, and efficient automation ecosystem.

### The Problem
In a robust multi-agent system, the supervisor or coordinator agent is often burdened with both orchestrating task execution and interpreting complex human requests. This dual responsibility can lead to bottlenecks, reduced modularity, and less effective human-in-the-loop interactions. There is a need for a dedicated agent that specializes in understanding human intent, decomposing it into actionable steps, and determining which agent should handle each task.

### The Solution
Planner Agent acts as the “strategic brain” of the framework. It:
- Receives high-level human statements or requests.
- Decomposes them into clear, actionable tasks.
- Maps each task to the most appropriate executor agent using a central registry and MCP server.
- Involves humans in the loop whenever clarification or feedback is needed, ensuring accuracy and adaptability.
- Frees the supervisor/coordinator agent to focus solely on orchestrating execution and communication between executor agents and clients.

This separation of concerns leads to a more interactive, scalable, and maintainable multi-agent system.

## ✨ Key Features

### 🤖 **Query Decomposition Engine**
- Receives complex DevOps queries from supervisor agents
- Uses AI to break down queries into specific, high-level actionable tasks (never subtasks or implementation steps)
- Always returns results in a strict, machine-parseable JSON format for reliability
- Proactively asks for clarification if queries are vague or missing context
- Categorizes tasks by domain (cloud provisioning, Kubernetes deployment, monitoring, CI/CD)
- Maintains context and dependencies between related tasks

### 🔄 **Dynamic Agent Registry Integration**
- Fetches live agent cards from central MCP server registry
- Maintains up-to-date knowledge of available executor agents and their capabilities
- Performs intelligent agent-to-task matching based on agent specializations
- Handles agent availability and capability changes dynamically

### 🛠️ **MCP Server Orchestration**
- Dynamically discovers available MCP servers and their tool capabilities
- Maps specific tasks to appropriate MCP server tools (terraform, kubectl, AWS CLI, etc.)
- Provides detailed tool configuration and execution instructions
- Ensures proper tool-to-agent-to-task alignment

### 🧠 **Context & Memory Management**
- Maintains comprehensive A2A context throughout multi-agent conversations
- Tracks task execution history and outcomes
- Preserves user preferences and common patterns
- Enables learning from previous successful task decompositions

### 👥 **Human-in-the-Loop Integration**
- Identifies when human clarification or approval is needed
- Proactively requests additional information from users if requirements are ambiguous or incomplete
- Escalates complex decisions to users via supervisor agent
- Provides clear explanations for task decomposition rationale
- Supports iterative refinement based on user feedback

### 🧩 **Extensible & Modular Design**
- Built with strict output models and modular prompts, making it easy to add new agent types, task categories, or integrations

## 🏗️ Architecture Overview

```
┌──────────────┐
│    User      │
└─────┬────────┘
      │  (1. High-level request)
      ▼
┌──────────────┐
│ Planner      │
│  Agent       │
└─────┬────────┘
      │  (2. Decompose, map tasks, fetch agent & MCP info, clarify if needed)
      ▼
┌──────────────┐
│ Supervisor   │
│  Agent       │
└─────┬────────┘
      │  (3. Orchestrate execution)
      ▼
┌──────────────┐
│ Executor     │────────────┐
│  Agents      │            │
└─────┬────────┘            │
      │ (4. Use MCP server) │
      ▼                     │
┌──────────────┐            │
│ MCP Servers  │◀───────────┘
└──────────────┘

[Supporting Components]
┌──────────────────────────────┐
│ Central Registry (on MCP)    │
│ - Agent cards                │
│ - MCP server inventory       │
└──────────────────────────────┘

[Human-in-the-Loop]
- Planner Agent may interact with User for clarification/feedback at any step before plan is finalized.
```

## 📋 Prerequisites

- **Python 3.12+** (for latest typing features and performance)
- **PostgreSQL 14+** with JSON/JSONB support
- **2GB+ RAM** for LLM operations and context management
- **Network access** to central MCP server registry
- **A2A protocol** communication capabilities

## 🛠️ Advanced Installation (Optional)

### Docker (Work in Progress)
> **Note:** Docker container installation is currently a work in progress. Please check back soon for updates or contribute to the project if you'd like to help!


## 🎮 Usage Examples

> **Tip:** From the client CLI, you can ask anything related to DevOps—such as:
> - "Help me in writing a Terraform module for S3"
> - "Can you run the Jenkins pipeline?"
> - "Set up monitoring for my application"
> - ...and more! The Planner Agent will decompose your request and inform you about which executor agents will execute each task and which MCP servers will be involved in performing the work.


## 🤖 Agent Ecosystem

Planner Agent can map tasks to any executor agent registered in the central registry. Common types of supported agents include:

- **Cloud & Infrastructure Agents** (e.g., AWS Orchestrator, Kubernetes Agent, Terraform Agent)
- **CI/CD Agents** (e.g., Build Agent, Deploy Agent, Test Agent)
- **Monitoring & Observability Agents** (e.g., Metrics Agent, Logging Agent, Alerting Agent)
- **Security Agents** (e.g., Scan Agent, IAM Agent, Secrets Agent)

> The actual list of available agents is dynamic and discovered from the central registry at runtime.

## 🔧 Configuration

Planner Agent can be customized via environment variables or a `.env` file.  
Below are the main configuration options (with their default values):

```python
# planner_agent/config/default.py

LLM_PROVIDER = "openai"           # LLM provider to use (e.g., openai, anthropic, etc.)
LLM_MODEL = "gpt-4o"              # Model name for the LLM provider
LLM_TEMPERATURE = 0.0             # Sampling temperature for LLM responses
LLM_MAX_TOKENS = 1000             # Maximum tokens for LLM responses

LOG_LEVEL = "INFO"                # Logging level (e.g., INFO, DEBUG)
LOG_FILE = "planner_agent.log"    # Log file path
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_TO_CONSOLE = True             # Output logs to console
LOG_TO_FILE = True                # Output logs to file
LOG_STRUCTURED_JSON = False       # Output logs in structured JSON format

AGENTS_MCP_SERVER_HOST = "localhost"   # MCP server host for agent registry
AGENTS_MCP_SERVER_PORT = 8080          # MCP server port
AGENTS_MCP_SERVER_TRANSPORT = "sse"    # Transport protocol (e.g., sse)
AGENTS_MCP_SERVER_DISABLED = False     # Disable MCP server integration
AGENTS_MCP_SERVER_AUTO_APPROVE = []    # List of agent IDs to auto-approve
```

> **Tip:**  
> You can override any of these settings by adding them to your `.env` file or exporting them as environment variables before starting the Planner Agent.

**Example `.env` file:**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.2
LOG_LEVEL=DEBUG
AGENTS_MCP_SERVER_HOST=localhost
AGENTS_MCP_SERVER_PORT=8080
```

For a full list of options and their descriptions, see [`planner_agent/config/default.py`](planner_agent/config/default.py).


## 🤝 Contributing

Contributions are always welcome. Please follow the workflow below:

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 📈 Roadmap

### Phase 1: Core Planning Engine (MVP) ✅
- [x] Basic query parsing and task decomposition
- [x] Simple agent registry client
- [x] Foundational A2A protocol communication
- [x] Basic task-to-agent mapping logic

### Phase 2: Dynamic Discovery & MCP Integration 🚧
- [x] Dynamic agent registry fetching
- [x] MCP server discovery and capability enumeration
- [x] Intelligent tool-to-task matching algorithms
- [ ] Enhanced task decomposition with domain-specific knowledge

### Phase 3: Intelligence & Learning 📋
- [ ] Advanced LLM capabilities for better task decomposition
- [ ] Learning from successful task execution patterns
- [ ] Predictive agent and tool recommendations

### Phase 4: Production Hardening & Scaling 📋
- [x] Comprehensive monitoring and observability
- [ ] Security hardening and credential management
- [ ] Advanced caching and performance optimization


## 📄 License

This project is licensed under the GNU General Public License v2 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google A2A Protocol** for robust agent-to-agent communication
- **Anthropic MCP** for standardized tool and server integration
- **LangChain/LangGraph** for powerful workflow orchestration

---

