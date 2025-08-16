"""
LangGraph Supervisor Adapter implementation.

This module provides an adapter class that properly implements the langgraph-supervisor
pattern with create_supervisor and auto-generated handoff tools as documented in flow.md.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from langgraph_supervisor import create_supervisor, create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseLanguageModel

from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async

# Create agent logger for adapter
adapter_logger = AgentLogger("LANGGRAPH_SUPERVISOR_ADAPTER")


class AgentDefinition:
    """Definition for an agent that will be registered with the supervisor."""
    
    def __init__(
        self,
        name: str,
        prompt: str,
        display_name: Optional[str] = None,
        tools: Optional[List[Any]] = None
    ):
        """
        Initialize agent definition.
        
        Args:
            name: Internal name for the agent (used for auto-generated handoff tools)
            prompt: System prompt for the agent
            display_name: Human-readable name for the agent
            tools: Optional list of tools for the agent
        """
        self.name = name
        self.prompt = prompt
        self.display_name = display_name or name
        self.tools = tools or []
    
    def create_agent(self, model: BaseLanguageModel) -> Any:
        """Create the actual agent using create_react_agent."""
        return create_react_agent(
            model,
            name=self.name,
            prompt=self.prompt,
            tools=self.tools
        )


class LangGraphSupervisorAdapter:
    """
    Adapter class that properly implements the langgraph-supervisor pattern.
    
    This adapter follows the flow.md specifications:
    - Uses create_supervisor() to auto-generate handoff tools
    - Implements the clarification loop pattern
    - Provides human-in-the-loop support
    - Maintains proper state management
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LangGraph Supervisor Adapter.
        
        Args:
            config: Configuration instance
            custom_config: Optional custom configuration to override defaults
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Get LLM configuration
        llm_config = self.config_instance.get_llm_config()
        
        # Initialize the LLM model
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            adapter_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model: {llm_config['provider']}:{llm_config['model']}"
            )
        except Exception as e:
            adapter_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model: {e}"
            )
            raise
        
        # Agent definitions and instances
        self.agent_definitions: List[AgentDefinition] = []
        self.agents: List[Any] = []
        self.agent_names: Dict[str, str] = {}
        
        # Supervisor and memory
        self.supervisor = None
        self.memory = MemorySaver()
        
        # Custom handoff tools
        self.custom_tools: List[Any] = []
        
        adapter_logger.log_structured(
            level="INFO",
            message="LangGraph Supervisor Adapter initialized"
        )
    
    @log_sync
    def add_agent(
        self,
        name: str,
        prompt: str,
        display_name: Optional[str] = None,
        tools: Optional[List[Any]] = None
    ) -> None:
        """
        Add an agent definition to the supervisor.
        
        Args:
            name: Internal name for the agent (triggers auto-generated handoff tools)
            prompt: System prompt for the agent
            display_name: Human-readable name for the agent
            tools: Optional list of tools for the agent
        """
        agent_def = AgentDefinition(name, prompt, display_name, tools)
        self.agent_definitions.append(agent_def)
        
        adapter_logger.log_structured(
            level="INFO",
            message=f"Added agent definition: {name} -> {agent_def.display_name}"
        )
    
    @log_sync
    def add_custom_handoff_tool(
        self,
        agent_name: str,
        name: str,
        description: str
    ) -> None:
        """
        Add a custom handoff tool (like escalate_to_human).
        
        Args:
            agent_name: Name of the target agent
            name: Tool name
            description: Tool description
        """
        custom_tool = create_handoff_tool(
            agent_name=agent_name,
            name=name,
            description=description
        )
        self.custom_tools.append(custom_tool)
        
        adapter_logger.log_structured(
            level="INFO",
            message=f"Added custom handoff tool: {name} -> {agent_name}"
        )
    
    @log_sync
    def create_supervisor(
        self,
        prompt: Optional[str] = None,
        add_handoff_back_messages: bool = True,
        output_mode: str = "full_history"
    ) -> None:
        """
        Create the supervisor using langgraph-supervisor create_supervisor.
        
        This method:
        1. Creates all agent instances from definitions
        2. Calls create_supervisor() to auto-generate handoff tools
        3. Compiles the supervisor with memory checkpointer
        
        Args:
            prompt: Custom supervisor prompt
            add_handoff_back_messages: Whether to add handoff back messages
            output_mode: Output mode for the supervisor
        """
        if not self.agent_definitions:
            raise ValueError("No agent definitions added. Call add_agent() first.")
        
        try:
            # Step 1: Create agent instances from definitions
            adapter_logger.log_structured(
                level="INFO",
                message="Creating agent instances from definitions"
            )
            
            for agent_def in self.agent_definitions:
                agent = agent_def.create_agent(self.model)
                self.agents.append(agent)
                self.agent_names[agent_def.name] = agent_def.display_name
                
                adapter_logger.log_structured(
                    level="DEBUG",
                    message=f"Created agent: {agent_def.name}"
                )
            
            # Step 2: Use default supervisor prompt if none provided
            if not prompt:
                prompt = self._get_default_supervisor_prompt()
            
            # Step 3: Create supervisor with auto-generated handoff tools
            adapter_logger.log_structured(
                level="INFO",
                message="Creating supervisor with auto-generated handoff tools"
            )
            
            self.supervisor = create_supervisor(
                agents=self.agents,
                model=self.model,
                tools=self.custom_tools,
                prompt=prompt,
                add_handoff_back_messages=add_handoff_back_messages,
                output_mode=output_mode
            )
            
            # Step 4: Compile with memory checkpointer
            self.supervisor = self.supervisor.compile(checkpointer=self.memory)
            
            adapter_logger.log_structured(
                level="INFO",
                message=f"Successfully created supervisor with {len(self.agents)} agents",
                extra={
                    "agent_count": len(self.agents),
                    "custom_tools": len(self.custom_tools),
                    "auto_generated_tools": len(self.agents)  # One per agent
                }
            )
            
        except Exception as e:
            adapter_logger.log_structured(
                level="ERROR",
                message=f"Failed to create supervisor: {e}"
            )
            raise
    
    @log_sync
    def invoke(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the supervisor with messages.
        
        Args:
            messages: List of message dictionaries
            config: Optional configuration for the invocation
            
        Returns:
            Supervisor response
        """
        if not self.supervisor:
            raise RuntimeError("Supervisor not created. Call create_supervisor() first.")
        
        try:
            # Prepare input state
            input_state = {"messages": messages}
            
            # Add thread_id for human-in-the-loop support
            if config and "thread_id" in config:
                input_state["configurable"] = {"thread_id": config["thread_id"]}
            
            adapter_logger.log_structured(
                level="INFO",
                message=f"Invoking supervisor with {len(messages)} messages"
            )
            
            # Invoke the supervisor
            result = self.supervisor.invoke(input_state)
            
            adapter_logger.log_structured(
                level="INFO",
                message="Supervisor invocation completed",
                extra={"messages_count": len(result.get("messages", []))}
            )
            
            return result
            
        except Exception as e:
            adapter_logger.log_structured(
                level="ERROR",
                message=f"Supervisor invocation failed: {e}"
            )
            raise
    
    @log_async
    async def astream(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Stream the supervisor execution for human-in-the-loop support.
        
        Args:
            messages: List of message dictionaries
            config: Optional configuration for the invocation
            
        Yields:
            Streaming results from supervisor execution
        """
        if not self.supervisor:
            raise RuntimeError("Supervisor not created. Call create_supervisor() first.")
        
        try:
            # Prepare input state
            input_state = {"messages": messages}
            
            # Add thread_id for human-in-the-loop support
            if config and "thread_id" in config:
                input_state["configurable"] = {"thread_id": config["thread_id"]}
            
            adapter_logger.log_structured(
                level="INFO",
                message=f"Starting supervisor stream with {len(messages)} messages"
            )
            
            # Stream the supervisor execution
            async for item in self.supervisor.astream(input_state):
                yield item
                
        except Exception as e:
            adapter_logger.log_structured(
                level="ERROR",
                message=f"Supervisor streaming failed: {e}"
            )
            raise
    
    @log_sync
    def get_agent_info(self) -> Dict[str, str]:
        """Get information about registered agents."""
        return self.agent_names.copy()
    
    @log_sync
    def get_auto_generated_tools(self) -> List[str]:
        """Get list of auto-generated handoff tool names."""
        if not self.agents:
            return []
        
        # Auto-generated tool names follow the pattern: delegate_to_{agent_name}_agent()
        tool_names = []
        for agent_name in self.agent_names.keys():
            tool_name = f"delegate_to_{agent_name}_agent"
            tool_names.append(tool_name)
        
        return tool_names
    
    @log_sync
    def get_custom_tools(self) -> List[str]:
        """Get list of custom tool names."""
        return [tool.name for tool in self.custom_tools]
    
    def _get_default_supervisor_prompt(self) -> str:
        """Get the default supervisor prompt following flow.md specifications."""
        return f"""You are the AWS Terraform Orchestrator Supervisor.

AGENTS:
{self._format_agent_descriptions()}

ROUTING RULES:
{self._format_routing_rules()}

CONSTRAINTS:
- Always route to Validation Agent before finalizing any Terraform changes
- Use escalate_to_human when clarification is needed
- Do not perform work yourself, only delegate to appropriate agents
- Assign work to one agent at a time, do not call agents in parallel

AUTO-GENERATED HANDOFF TOOLS:
The create_supervisor function has automatically generated these handoff tools:
{self._format_auto_generated_tools()}

CUSTOM TOOLS:
{self._format_custom_tools()}

Use these tools to route work to the appropriate agents based on the user's request.
"""
    
    def _format_agent_descriptions(self) -> str:
        """Format agent descriptions for the prompt."""
        descriptions = []
        for name, display_name in self.agent_names.items():
            descriptions.append(f"- {display_name}: Handles {name.replace('_', ' ')} tasks")
        return "\n".join(descriptions)
    
    def _format_routing_rules(self) -> str:
        """Format routing rules for the prompt."""
        rules = []
        for i, (name, display_name) in enumerate(self.agent_names.items(), 1):
            if "analysis" in name:
                rules.append(f"{i}. Route to {display_name} for: requirements gathering, planning, analysis")
            elif "generation" in name:
                rules.append(f"{i}. Route to {display_name} for: creating new Terraform modules")
            elif "validation" in name:
                rules.append(f"{i}. Route to {display_name} for: validating Terraform code")
            elif "editor" in name:
                rules.append(f"{i}. Route to {display_name} for: modifying existing configurations")
            elif "human" in name:
                rules.append(f"{i}. Route to {display_name} for: review, approval, clarification")
            else:
                rules.append(f"{i}. Route to {display_name} for: {name.replace('_', ' ')} tasks")
        return "\n".join(rules)
    
    def _format_auto_generated_tools(self) -> str:
        """Format auto-generated tool names for the prompt."""
        tools = self.get_auto_generated_tools()
        return "\n".join([f"- {tool}()" for tool in tools])
    
    def _format_custom_tools(self) -> str:
        """Format custom tool names for the prompt."""
        tools = self.get_custom_tools()
        if not tools:
            return "- None"
        return "\n".join([f"- {tool}()" for tool in tools])


# Factory function for easy creation
@log_sync
def create_langgraph_supervisor_adapter(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> LangGraphSupervisorAdapter:
    """
    Factory function to create a LangGraph Supervisor Adapter.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Configured LangGraphSupervisorAdapter instance
    """
    return LangGraphSupervisorAdapter(config=config, custom_config=custom_config) 