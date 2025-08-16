"""
New Infrastructure Subgraph for Planner Agent.

This module implements the NewInfrastructureSubgraph, which handles new infrastructure
planning workflows within the main Planner Agent. It specializes in:
- Dependency mapping for new AWS services
- Greenfield infrastructure requirements analysis
- New service selection and architectural design
- Deployment strategies for new infrastructure
"""

import json
import traceback
from datetime import datetime, timezone
import re
from typing import Dict, Any, List, Optional
from functools import wraps
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.utils.mcp_client import MCPAdapterClient
from ...types import PlannerState

# Create agent logger for new infrastructure subgraph
new_infra_logger = AgentLogger("NEW_INFRA_SUBGRAPH")


class NewInfrastructureRequirements(BaseModel):
    """Output schema for new infrastructure requirements analysis."""
    business_requirements: List[str] = Field(description="Business requirements and objectives")
    technical_requirements: List[str] = Field(description="Technical requirements and specifications")
    constraints: List[str] = Field(description="Constraints and limitations")
    assumptions: List[str] = Field(description="Assumptions and prerequisites")
    risk_factors: List[str] = Field(description="Risk factors and mitigation strategies")
    scalability_requirements: List[str] = Field(description="Scalability requirements for future growth")
    performance_requirements: List[str] = Field(description="Performance requirements and SLAs")


class NewInfrastructurePlan(BaseModel):
    """Output schema for new infrastructure planning."""
    services: List[Dict[str, Any]] = Field(description="Required AWS services with configurations")
    patterns: List[Dict[str, Any]] = Field(description="Architectural patterns")
    security_requirements: List[Dict[str, Any]] = Field(description="Security requirements")
    cost_estimates: Dict[str, Any] = Field(description="Cost estimates and breakdown")
    dependencies: Dict[str, List[str]] = Field(description="Service dependencies")
    scalability_considerations: List[str] = Field(description="Scalability considerations")
    performance_optimizations: List[str] = Field(description="Performance optimizations")


class NewInfrastructureExecutionPlan(BaseModel):
    """Output schema for new infrastructure execution planning."""
    steps: List[Dict[str, Any]] = Field(description="Step-by-step execution plan")
    total_estimated_time: str = Field(description="Total estimated execution time")
    critical_path: List[int] = Field(description="Critical path steps")
    resource_requirements: Dict[str, Any] = Field(description="Resource requirements")
    deployment_strategy: str = Field(description="Deployment strategy")
    testing_phases: List[str] = Field(description="Testing phases")


class NewInfrastructureValidationResult(BaseModel):
    """Output schema for new infrastructure validation."""
    validation_results: Dict[str, List[str]] = Field(description="Validation results by category")
    overall_assessment: str = Field(description="Overall assessment")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    risk_level: str = Field(description="Overall risk level")


class NewInfrastructureDependencyMapping(BaseModel):
    """Output schema for dependency mapping analysis."""
    primary_service: str = Field(description="The main AWS service being requested")
    mandatory_dependencies: List[Dict[str, Any]] = Field(
        description="Dependencies that must be set up for the primary service to work"
    )
    optional_dependencies: List[Dict[str, Any]] = Field(
        description="Dependencies that may be needed depending on use case"
    )
    dependency_categories: Dict[str, List[str]] = Field(
        description="Dependencies grouped by category (networking, security, monitoring, etc.)"
    )
    setup_prerequisites: List[str] = Field(
        description="Prerequisites that must be in place before creating the main service"
    )
    terraform_provider_requirements: List[str] = Field(
        description="Required Terraform providers and their versions"
    )
    dependency_explanations: Dict[str, str] = Field(
        description="Explanations for why each dependency is needed"
    )
    follow_up_questions: List[str] = Field(
        description="Questions to clarify requirements and dependencies"
    )

# New, service-agnostic dependency planning schema
class DependencyPlan(BaseModel):
    action: str = Field(description="High-level action to perform, e.g., 'create_vpc', 'create_subnet', 'create_cluster'")
    parameters: Dict[str, Any] = Field(description="Service-agnostic parameters; set null/[] for unknowns")
    needs_clarification: List[str] = Field(description="List of parameter keys requiring clarification")

# Structured response schema handled via response_format using existing mapping model


class NewInfrastructureSubgraph:
    """
    New Infrastructure Subgraph for handling new infrastructure planning workflows.
    
    This subgraph specializes in creating new AWS infrastructure from scratch,
    focusing on dependency mapping, architectural design, and deployment planning.
    """
    
    def __init__(
        self,
        model,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        mcp_client_factory: Optional[Any] = None,
        mcp_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the New Infrastructure Subgraph.
        
        Args:
            model: LLM model instance
            config: Configuration instance
            custom_config: Optional custom configuration
        """
        self.model = model
        self.config_instance = config or Config(custom_config or {})
        self.mcp_client_factory = mcp_client_factory
        self.mcp_params = mcp_params or {}
        # Initialize output parsers
        self.requirements_parser = JsonOutputParser(pydantic_object=NewInfrastructureRequirements)
        self.infrastructure_parser = JsonOutputParser(pydantic_object=NewInfrastructurePlan)
        self.execution_parser = JsonOutputParser(pydantic_object=NewInfrastructureExecutionPlan)
        self.validation_parser = JsonOutputParser(pydantic_object=NewInfrastructureValidationResult)
        # Parsers
        self.dependency_parser = JsonOutputParser(pydantic_object=NewInfrastructureDependencyMapping)
        self.dependency_plan_parser = JsonOutputParser(pydantic_object=DependencyPlan)
        # Define prompts
        self._define_prompts()
        # Defer MCP client usage to nodes; avoid calling methods here if client is not ready
        # Initialize dependency mapping tool and React agent wrapper
        self.dependency_mapping_tool = self._create_dependency_mapping_tool()
        self.dependency_mapping_lc_tool = StructuredTool.from_function(
            func=self.dependency_mapping_tool,
            name="analyze_dependencies",
            description="Analyze user_request to map Terraform dependencies. Returns a JSON string with primary_service, dependencies, and follow_up_questions."
        )
        # Resolve final tools list once at init (optional MCP tools)
        self.dependency_mapping_tools: List[Any] = [self.dependency_mapping_lc_tool]
        if self.mcp_client_factory is not None:
            try:
                async def _load_mcp():
                    async with self.mcp_client_factory(**self.mcp_params) as client:
                        return client.get_tools()
                mcp_tools = asyncio.run(_load_mcp())
                if mcp_tools:
                    self.dependency_mapping_tools.extend(mcp_tools)
            except Exception:
                pass

        self.dependency_mapping_agent = create_react_agent(
            model=self.model,
            tools=self.dependency_mapping_tools,
            prompt=(
                "You are a dependency mapping agent for Terraform planning.\n\n"
                "INSTRUCTIONS:\n"
                "- Always call the tool analyze_dependencies with the user's request content.\n"
                "- Do not produce extra commentary.\n"
                "- The tool returns JSON as a string.\n"
                "- After calling the tool, output a final structured response matching the declared response schema."
            ),
            # Keep response_format off for now to avoid schema errors; rely on tool JSON
            name="dependency_mapping_agent",
        )
        
        new_infra_logger.log_structured(
            level="INFO",
            message="New Infrastructure Subgraph initialized",
            extra={
                "model_provider": config.get_llm_config()['provider'] if config else 'unknown',
                "model_name": config.get_llm_config()['model'] if config else 'unknown'
            }
        )
    
    def _define_prompts(self) -> None:
        """Define the prompts used by the new infrastructure subgraph."""
        
        # Dependency planning prompt (AWS-focused, structured steps, few-shot)
        self.dependency_prompt = ChatPromptTemplate.from_messages([
            ("system", """
[ROLE]
You are an advanced AWS Terraform Dependency Planner. Analyze the user's request, identify the primary AWS service and dependencies, and produce a concise plan for Terraform module work.

[INPUT]
You will receive user request specific to terraform for aws services.

[PROCESS — EXECUTE THE FOLLOWING STEPS EXACTLY IN ORDER]

Step 1: Identify Primary AWS Service (action)
- Choose a single AWS action (e.g., "create_vpc", "create_subnet", "create_db", "create_bucket").

Step 2: Derive Parameters
- Build a parameters object for the chosen action.
- Unknown or ambiguous values MUST be null or [].
- If the action depends on another AWS service (e.g., subnets depend on a VPC), include explicit dependency keys (e.g., "provision_vpc": true|false, or "vpc_id" when using an existing VPC).

Step 3: Clarification Planning
- Create a needs_clarification list of parameter keys that must be clarified.
- Order clarifications so dependency decisions come FIRST (e.g., decide new vs. existing VPC before subnet CIDR/AZ questions), followed by sizing and CIDR/AZ details.

Step 4: Output Assembly (Strict JSON Constraint)
- Output ONLY a single JSON object matching this Pydantic model:

class DependencyPlan(BaseModel):
    action: str
    parameters: Dict[str, Any]
    needs_clarification: List[str]

[STRICT RESPONSE RULES]
- Output MUST be ONLY the JSON object; no prose, markdown, or comments.
- Use AWS-only actions and parameters.
- Unknowns MUST be null or [].
- Prefer stable parameter keys when relevant:
  - VPC: cidr_block, azs, num_private_subnets, num_public_subnets, enable_nat_gateways, dns_hostnames, dns_support, private_subnet_cidr_blocks
  - Subnet: vpc_id, provision_vpc, azs, num_subnets, subnet_type, subnet_cidr_blocks, associate_route_tables, create_nat_gateway
  - RDS: engine, engine_version, instance_class, multi_az, storage_gb, encryption, kms_key_id, backup_retention_days, subnet_group_name, security_group_ids
  - S3: bucket_name, versioning, encryption.type, encryption.kms_key_id, block_public_access, lifecycle_rules

[EXAMPLES]

Example: Please plan a VPC with high-availability across three AZs and private subnets.
{{
  "action": "create_vpc",
  "parameters": {{
    "cidr_block": null,
    "azs": [],
    "subnet_strategy": "private_only",
    "num_private_subnets": 3,
    "num_public_subnets": 0,
    "private_subnet_cidr_blocks": [],
    "enable_nat_gateways": true,
    "dns_hostnames": true,
    "dns_support": true
  }},
  "needs_clarification": ["cidr_block", "azs", "private_subnet_cidr_blocks"]
}}

Example: Can you help me in crating a an aws subnet module 
{{
  "action": "create_subnet",
  "parameters": {{
    "vpc_id": null,
    "provision_vpc": null,
    "azs": [],
    "num_subnets": null,
    "subnet_type": null,
    "subnet_cidr_blocks": [],
    "associate_route_tables": true,
    "create_nat_gateway": null
  }},
  "needs_clarification": [
    "provision_vpc",
    "vpc_id",
    "azs",
    "num_subnets",
    "subnet_type",
    "subnet_cidr_blocks",
    "create_nat_gateway"
  ]
}}

Example: Provision a production-grade PostgreSQL RDS with Multi-AZ and encryption.
{{
  "action": "create_db",
  "parameters": {{
    "engine": "postgres",
    "engine_version": null,
    "instance_class": null,
    "multi_az": true,
    "storage_gb": 100,
    "encryption": true,
    "kms_key_id": null,
    "backup_retention_days": 7,
    "subnet_group_name": null,
    "security_group_ids": []
  }},
  "needs_clarification": [
    "engine_version",
    "instance_class",
    "kms_key_id",
    "subnet_group_name",
    "security_group_ids"
  ]
}}

Example: Create an S3 bucket for logs with KMS encryption and versioning.
{{
  "action": "create_bucket",
  "parameters": {{
    "bucket_name": null,
    "versioning": true,
    "encryption": {{"type": "SSE-KMS", "kms_key_id": null}},
    "block_public_access": true,
    "lifecycle_rules": []
  }},
  "needs_clarification": ["bucket_name", "encryption.kms_key_id"]
}}
"""),
            ("human", "Analyze the following request and produce ONLY the plan JSON: {user_request}")
        ])
        
        # Requirements analysis prompt for new infrastructure
        self.requirements_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS infrastructure planner specializing in new infrastructure projects. Analyze the user request to identify comprehensive requirements for a greenfield deployment.

Analyze the user request and identify:
1. Business requirements and objectives
2. Technical requirements and specifications
3. Constraints and limitations
4. Assumptions and prerequisites
5. Risk factors and mitigation strategies
6. Scalability requirements for future growth
7. Performance requirements and SLAs

IMPORTANT: You must respond with a valid JSON object that matches the following structure exactly:
{{
  "business_requirements": ["requirement1", "requirement2", ...],
  "technical_requirements": ["requirement1", "requirement2", ...],
  "constraints": ["constraint1", "constraint2", ...],
  "assumptions": ["assumption1", "assumption2", ...],
  "risk_factors": ["risk1", "risk2", ...],
  "scalability_requirements": ["requirement1", "requirement2", ...],
  "performance_requirements": ["requirement1", "requirement2", ...]
}}

Each field should be an array of strings. Do not include any markdown formatting, explanations, or additional text outside the JSON structure."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Analyze the following new infrastructure request: {user_request}")
        ])
        
        # Infrastructure planning prompt for new deployments
        self.infrastructure_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AWS architect specializing in new infrastructure design. Create a comprehensive infrastructure plan for a greenfield deployment.

Design:
1. Required AWS services with optimal configurations
2. Modern architectural patterns (microservices, event-driven, serverless, etc.)
3. Security requirements and best practices
4. Cost estimates and optimization strategies
5. Service dependencies and integration points
6. Scalability considerations for future growth
7. Performance optimizations and monitoring

IMPORTANT: You must respond with a valid JSON object that matches the following structure exactly:
{{
  "services": [{{"service_name": "string", "configuration": "object", ...}}],
  "patterns": [{{"pattern_name": "string", "description": "string", ...}}],
  "security_requirements": [{{"requirement": "string", "implementation": "string", ...}}],
  "cost_estimates": {{"monthly_cost": "string", "breakdown": "object", ...}},
  "dependencies": {{"service1": ["dependency1", "dependency2"], ...}},
  "scalability_considerations": ["consideration1", "consideration2", ...],
  "performance_optimizations": ["optimization1", "optimization2", ...]
}}

Follow AWS best practices and consider modern cloud-native patterns for maximum efficiency and scalability. Do not include any markdown formatting, explanations, or additional text outside the JSON structure."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Create new infrastructure plan based on: {requirements_analysis}")
        ])
        
        # Execution planning prompt for new deployments
        self.execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert DevOps engineer specializing in new infrastructure deployments. Create a detailed step-by-step execution plan for implementing new AWS infrastructure.

The plan should include:
1. Sequential steps with clear actions and commands
2. Dependencies between steps and services
3. Time estimates for each phase
4. Resource requirements and provisioning
5. Validation criteria for each step
6. Deployment strategy (blue-green, canary, etc.)
7. Testing phases and validation procedures
8. Monitoring and alerting setup

IMPORTANT: You must respond with a valid JSON object that matches the following structure exactly:
{{
  "steps": [{{"step_number": 1, "action": "string", "commands": ["string"], "estimated_time": "string", ...}}],
  "total_estimated_time": "string",
  "critical_path": [1, 2, 3, ...],
  "resource_requirements": {{"cpu": "string", "memory": "string", "storage": "string", ...}},
  "deployment_strategy": "string",
  "testing_phases": ["phase1", "phase2", ...]
}}

Ensure the plan follows infrastructure-as-code best practices and includes proper testing and validation. Do not include any markdown formatting, explanations, or additional text outside the JSON structure."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Create execution plan for new infrastructure: {infrastructure_plan}")
        ])
        
        # Validation prompt for new infrastructure
        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert infrastructure validator specializing in new deployments. Review the complete new infrastructure plan and identify any issues, risks, or improvements.

Check for:
1. Security vulnerabilities and compliance gaps
2. Cost optimization opportunities
3. Scalability and performance concerns
4. Best practice violations
5. Architectural design issues
6. Deployment strategy problems
7. Testing and validation gaps

IMPORTANT: You must respond with a valid JSON object that matches the following structure exactly:
{{
  "validation_results": {{
    "security_issues": ["issue1", "issue2", ...],
    "cost_optimizations": ["optimization1", "optimization2", ...],
    "scalability_concerns": ["concern1", "concern2", ...],
    "best_practice_violations": ["violation1", "violation2", ...],
    "architectural_issues": ["issue1", "issue2", ...],
    "deployment_issues": ["issue1", "issue2", ...],
    "testing_gaps": ["gap1", "gap2", ...]
  }},
  "overall_assessment": "string",
  "recommendations": ["recommendation1", "recommendation2", ...],
  "risk_level": "low|medium|high"
}}

Provide actionable recommendations for improvement and optimization. Do not include any markdown formatting, explanations, or additional text outside the JSON structure."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Validate the new infrastructure plan: {complete_plan}")
        ])
    
    def _create_dependency_mapping_tool(self):
        """Create the dependency mapping tool."""
        
        def dependency_mapping_tool(user_request: str) -> str:
            """
            Analyze an AWS Terraform user request and return a concise dependency plan.

            Identifies the primary AWS action (for example: create_vpc, create_subnet, create_db, create_bucket),
            derives action-specific parameters (including dependency keys such as provision_vpc or vpc_id), and
            lists missing items requiring human clarification. Returns a JSON string matching DependencyPlan:
            {"action": str, "parameters": Dict[str, Any], "needs_clarification": List[str]}.
            """
            
            # Create the chain (new schema)
            chain = self.dependency_prompt | self.model | self.dependency_plan_parser
            
            # Execute the analysis
            result = chain.invoke({"user_request": user_request})

            # Return the result as a JSON string
            return json.dumps(result, indent=2)
        
        return dependency_mapping_tool
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for new infrastructure planning.
        
        Returns:
            StateGraph: The compiled graph for new infrastructure planning
        """
        # Create the graph with PlannerState
        graph = StateGraph(PlannerState)
        
        # Add nodes for the new infrastructure workflow
        graph.add_node("dependency_mapping", self._dependency_mapping_agent_node)
        graph.add_node("analyze_requirements", self._analyze_requirements_node)
        graph.add_node("plan_infrastructure", self._plan_infrastructure_node)
        graph.add_node("create_execution_plan", self._create_execution_plan_node)
        graph.add_node("validate_plan", self._validate_plan_node)
        
        # Add conditional routing after dependency mapping to check for interrupts
        graph.add_conditional_edges(
            "dependency_mapping",
            self._route_after_dependency_mapping,
            {
                "continue": "analyze_requirements",
                "interrupt": END  # End the subgraph if interrupt is needed
            }
        )
        
        # Add linear workflow for the rest of new infrastructure
        graph.add_edge("analyze_requirements", "plan_infrastructure")
        graph.add_edge("plan_infrastructure", "create_execution_plan")
        graph.add_edge("create_execution_plan", "validate_plan")
        
        # Set entry and exit points
        graph.set_entry_point("dependency_mapping")
        graph.set_finish_point("validate_plan")
        
        return graph
    
    # ============================================================================
    # ROUTING FUNCTIONS
    # ============================================================================
    
    def _route_after_dependency_mapping(self, state: PlannerState) -> str:
        """
        Route after dependency mapping based on whether interrupt is needed.
        
        Args:
            state: Current planner state
            
        Returns:
            str: Route destination - "continue" or "interrupt"
        """
        # Check if dependency mapping needs human input
        if getattr(state, 'waiting_for_dependency_input', False):
            new_infra_logger.log_structured(
                level="INFO",
                message="Dependency mapping requires human input - routing to interrupt",
                extra={
                    "current_question": getattr(state, 'current_dependency_question', 'unknown'),
                    "waiting_for_dependency_input": True
                }
            )
            return "interrupt"
        
        # Check if dependency mapping is complete
        if getattr(state, 'dependency_mapping_complete', False):
            new_infra_logger.log_structured(
                level="INFO",
                message="Dependency mapping complete - continuing to requirements analysis",
                extra={
                    "dependency_mapping_complete": True,
                    "answers_count": len(getattr(state, 'dependency_answers', {}))
                }
            )
            return "continue"
        
        # Default to continue if no clear interrupt flag
        new_infra_logger.log_structured(
            level="WARNING",
            message="No clear routing decision - defaulting to continue",
            extra={
                "waiting_for_dependency_input": getattr(state, 'waiting_for_dependency_input', False),
                "dependency_mapping_complete": getattr(state, 'dependency_mapping_complete', False)
            }
        )
        return "continue"
    
    # ============================================================================
    # GRAPH NODES
    # ============================================================================
    
    def _dependency_mapping_agent_node(self, state: PlannerState) -> PlannerState:
        """Map dependencies using a prebuilt React agent + single tool; handle HITL via interrupt."""
        # Ensure containers
        if not hasattr(state, 'dependency_answers') or state.dependency_answers is None:
            state.dependency_answers = {}
        if not hasattr(state, 'dependency_questions') or state.dependency_questions is None:
            state.dependency_questions = []
        # Always reset completion at node start; we'll mark complete only after successful Q&A
        state.dependency_mapping_complete = False

        # Initialize questions from dependency mapping analysis if needed or mapping missing/incomplete
        existing_mapping = (getattr(state, 'planning_metadata', {}) or {}).get("dependency_mapping", {})
        has_questions = bool(state.dependency_questions)
        has_mapping = bool(existing_mapping)
        has_action = bool(existing_mapping.get("action")) if has_mapping else False
        should_run = (not has_questions) or (not has_mapping) or (not has_action)
        try:
            new_infra_logger.log_structured(
                level="DEBUG",
                message="Dependency mapping run decision",
                extra={
                    "has_questions": has_questions,
                    "has_mapping": has_mapping,
                    "has_action": has_action,
                    "should_run": should_run,
                }
            )
        except Exception:
            pass
        if should_run:
            # Prepare follow-up container upfront to avoid UnboundLocalError on exceptions
            ai_followups: List[str] = []
            # Invoke the React agent which will call the tool(s) and return output
            try:
                # Use a clean, minimal chat history to avoid invalid tool-call history
                # observed in upstream supervisor messages
                messages = [HumanMessage(content=state.user_request)]
                # If MCP tools are available, bind them dynamically for this call
                agent_resp = self.dependency_mapping_agent.invoke({"messages": messages})
                # Extract the tool's JSON payload from ToolMessage and the AI follow-ups
                mapping_json = None
                last_ai_after_tool = None
                msgs = agent_resp.get("messages", [])
                for msg in reversed(msgs):
                    if mapping_json is None and isinstance(msg, ToolMessage) and getattr(msg, 'name', None) == 'analyze_dependencies':
                        mapping_json = msg.content
                    if last_ai_after_tool is None and isinstance(msg, AIMessage) and getattr(msg, 'content', None):
                        last_ai_after_tool = msg.content
                mapping = json.loads(mapping_json) if mapping_json else {}
                # Optionally parse AI follow-up questions if present
                if last_ai_after_tool:
                    try:
                        lines = [ln.strip() for ln in str(last_ai_after_tool).splitlines() if ln.strip()]
                        for ln in lines:
                            if re.match(r"^(?:\d+\.|-)\s+", ln):
                                # Remove leading list marker
                                q = re.sub(r"^(?:\d+\.|-)\s+", "", ln).strip()
                                # If bold headings like **CIDR Block**: ... keep full sentence
                                # Normalize to end at question mark when present
                                q_end = q.find('?')
                                ai_followups.append(q if q_end == -1 else q[:q_end+1])
                    except Exception:
                        ai_followups = []
                try:
                    new_infra_logger.log_structured(
                        level="DEBUG",
                        message="Dependency mapping tool parsed",
                        extra={
                            "action": (mapping.get("action") or mapping.get("primary_service")) if isinstance(mapping, dict) else None,
                            "follow_up_len": len(mapping.get("follow_up_questions", [])) if isinstance(mapping, dict) else 0,
                            "needs_clarification_len": len(mapping.get("needs_clarification", [])) if isinstance(mapping, dict) else 0
                        }
                    )
                except Exception:
                    pass
            except Exception as e:
                mapping = {"follow_up_questions": []}
                # Add a lightweight trace to messages for debugging
                try:
                    state.messages.append(AIMessage(content=f"Dependency mapping failed: {str(e)}"))
                except Exception:
                    pass
            # Persist mapping context for later nodes
            if not hasattr(state, 'planning_metadata') or state.planning_metadata is None:
                state.planning_metadata = {}
            state.planning_metadata["dependency_mapping"] = mapping
            # Support both legacy and new schema
            dep_questions = ai_followups or mapping.get("follow_up_questions")
            if not dep_questions:
                dep_questions = mapping.get("needs_clarification", [])
            state.dependency_questions = dep_questions
            try:
                new_infra_logger.log_structured(
                    level="DEBUG",
                    message="Prepared dependency questions",
                    extra={
                        "questions_count": len(state.dependency_questions or []),
                        "first_question": (state.dependency_questions[0] if state.dependency_questions else None)
                    }
                )
            except Exception:
                pass
            # Add a concise structured summary to messages for traceability
            try:
                primary = mapping.get("primary_service") or mapping.get("action")
                q_cnt = len(state.dependency_questions or [])
                summary = f"Dependency plan: action={primary}, clarifications={q_cnt}"
                state.messages.append(AIMessage(content=summary))
            except Exception:
                pass

        # Find the next unanswered question
        next_q = None
        for q in state.dependency_questions:
            if q not in state.dependency_answers:
                next_q = q
                break
        try:
            new_infra_logger.log_structured(
                level="DEBUG",
                message="Selecting next dependency question",
                extra={
                    "has_next": bool(next_q),
                    "answered_count": len(state.dependency_answers or {}),
                    "total_questions": len(state.dependency_questions or [])
                }
            )
        except Exception:
            pass

        if next_q:
            state.current_dependency_question = next_q
            
            # Set interrupt flags in state so the interrupt gate can detect them
            state.waiting_for_dependency_input = True
            state.dependency_mapping_complete = False
            
            # Store interrupt data in state for the interrupt gate
            if not hasattr(state, 'planning_metadata'):
                state.planning_metadata = {}
            state.planning_metadata['dependency_mapping_partial'] = {
                'primary_service': (
                    state.planning_metadata.get('dependency_mapping', {}).get('primary_service')
                    or state.planning_metadata.get('dependency_mapping', {}).get('action')
                ),
                'mandatory_dependencies_count': len(state.planning_metadata.get('dependency_mapping', {}).get('mandatory_dependencies', [])),
                'optional_dependencies_count': len(state.planning_metadata.get('dependency_mapping', {}).get('optional_dependencies', [])),
            }
            
            payload = {
                'question': next_q,
                'context': 'dependency_mapping',
                'available_questions': state.dependency_questions,
                'partial_analysis': state.planning_metadata['dependency_mapping_partial']
            }
            
            try:
                new_infra_logger.log_structured(
                    level="INFO",
                    message="Setting interrupt flags and raising interrupt for dependency clarification",
                    extra={
                        "question": payload.get('question'),
                        "available_count": len(payload.get('available_questions') or []),
                        "partial_primary": payload.get('partial_analysis', {}).get('primary_service'),
                        "waiting_for_dependency_input": True,
                        "dependency_mapping_complete": False
                    }
                )
            except Exception:
                pass
            
            # Add interrupt message to state
            state.messages.append(
                AIMessage(
                    content=f"INTERRUPT: {next_q}",
                    additional_kwargs={
                        "interrupt_data": payload,
                        "interrupt_type": "dependency_mapping",
                        "waiting_for_dependency_input": True
                    }
                )
            )
            
            # Return state with interrupt flags set - the interrupt gate will handle the actual interrupt
            return state

        # All questions answered: finalize
        state.dependency_mapping_complete = True
        state.current_dependency_question = None
        state.messages.append(
            AIMessage(content="Dependency mapping completed. Proceeding to requirements analysis.")
        )
        try:
            new_infra_logger.log_structured(
                level="INFO",
                message="Dependency mapping finalized (no pending questions)",
                extra={
                    "answers_count": len(state.dependency_answers or {}),
                    "questions_count": len(state.dependency_questions or [])
                }
            )
        except Exception:
            pass
        return state
    
    def _complete_dependency_mapping(self, state: PlannerState, final_result: Dict[str, Any]) -> PlannerState:
        """Complete the dependency mapping process."""
        
        # Update state with final dependency mapping results
        if not hasattr(state, 'planning_metadata'):
            state.planning_metadata = {}
        state.planning_metadata["dependency_mapping"] = final_result
        state.status = "dependencies_mapped"
        state.current_step = "dependencies_mapped"
        if not hasattr(state, 'completed_steps'):
            state.completed_steps = []
        state.completed_steps.append("dependency_mapping")
        state.dependency_mapping_complete = True
        state.waiting_for_dependency_input = False
        state.current_dependency_question = None
        
        # Add dependency mapping result to messages
        state.messages.append(
            AIMessage(content=f"Dependency mapping completed for {final_result.get('primary_service', 'infrastructure')}. Identified {len(final_result.get('mandatory_dependencies', []))} mandatory and {len(final_result.get('optional_dependencies', []))} optional dependencies.")
        )
        
        new_infra_logger.log_structured(
            level="INFO",
            message="New infrastructure dependency mapping completed",
            extra={
                "primary_service": final_result.get('primary_service', 'unknown'),
                "mandatory_dependencies_count": len(final_result.get('mandatory_dependencies', [])),
                "optional_dependencies_count": len(final_result.get('optional_dependencies', [])),
                "categories_count": len(final_result.get('dependency_categories', {})),
                "follow_up_questions_count": len(final_result.get('follow_up_questions', []))
            }
        )
        
        return state
    
    def _analyze_requirements_node(self, state: PlannerState) -> PlannerState:
        """Analyze user requirements for new infrastructure."""
        
        try:
            # Get dependency mapping from previous step
            dependency_mapping = state.planning_metadata.get("dependency_mapping", {})
            
            # Create prompt chain
            chain = self.requirements_prompt | self.model | self.requirements_parser
            
            # Enhance the requirements analysis with dependency context
            enhanced_user_request = f"{state.user_request}\n\nDependency Context: {str(dependency_mapping)}"
            
            # Execute analysis with enhanced context
            result = chain.invoke({
                "messages": state.messages,
                "user_request": enhanced_user_request
            })
            
            # Update state with analysis results
            state.requirements_analysis = result
            state.status = "requirements_analyzed"
            state.current_step = "requirements_analyzed"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("analyze_requirements")
            
            # Add analysis result to messages
            state.messages.append(
                AIMessage(content=f"New infrastructure requirements analysis completed. Found {len(result['business_requirements'])} business requirements, {len(result['technical_requirements'])} technical requirements, and {len(result['scalability_requirements'])} scalability requirements.")
            )
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure requirements analysis completed",
                extra={
                    "business_requirements_count": len(result['business_requirements']),
                    "technical_requirements_count": len(result['technical_requirements']),
                    "scalability_requirements_count": len(result['scalability_requirements']),
                    "constraints_count": len(result['constraints'])
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Requirements analysis failed: {str(e)}"
            state.status = "error"
            raise
    
    def _plan_infrastructure_node(self, state: PlannerState) -> PlannerState:
        """Plan new infrastructure based on requirements analysis."""
        
        try:
            # Get dependency mapping and requirements analysis from state
            dependency_mapping = state.planning_metadata.get("dependency_mapping", {})
            requirements_analysis = state.requirements_analysis or {}
            
            # Create enhanced infrastructure plan context
            enhanced_context = {
                "requirements": requirements_analysis,
                "dependencies": dependency_mapping
            }
            
            # Create prompt chain
            chain = self.infrastructure_prompt | self.model | self.infrastructure_parser
            
            # Execute infrastructure planning with dependency context
            result = chain.invoke({
                "messages": state.messages,
                "requirements_analysis": str(enhanced_context)
            })
            
            # Update state
            state.infrastructure_requirements = result['services']
            # Convert patterns to strings to match PlannerState schema
            state.architectural_patterns = [str(pattern) for pattern in result['patterns']]
            # Convert security requirements to strings to match PlannerState schema
            state.security_requirements = [str(req) for req in result['security_requirements']]
            state.cost_analysis = result['cost_estimates']
            state.status = "infrastructure_planned"
            state.current_step = "infrastructure_planned"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("plan_infrastructure")
            
            # Add planning result to messages
            planning_summary = f"New infrastructure planning completed. Identified {len(result['services'])} services, {len(result['patterns'])} architectural patterns, and {len(result['scalability_considerations'])} scalability considerations."
            state.messages.append(AIMessage(content=planning_summary))
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure planning completed",
                extra={
                    "services_count": len(result['services']),
                    "patterns_count": len(result['patterns']),
                    "security_requirements_count": len(result['security_requirements']),
                    "scalability_considerations_count": len(result['scalability_considerations'])
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Infrastructure planning failed: {str(e)}"
            state.status = "error"
            raise
    
    def _create_execution_plan_node(self, state: PlannerState) -> PlannerState:
        """Create a detailed execution plan for new infrastructure."""
        
        try:
            # Get infrastructure plan from state
            infrastructure_plan = {
                "services": state.infrastructure_requirements or [],
                "patterns": state.architectural_patterns or [],
                "security": state.security_requirements or []
            }
            
            # Create prompt chain
            chain = self.execution_prompt | self.model | self.execution_parser
            
            # Execute execution planning
            result = chain.invoke({
                "messages": state.messages,
                "infrastructure_plan": str(infrastructure_plan)
            })
            
            # Update state
            state.execution_plan = result['steps']
            state.resource_requirements = result['resource_requirements']
            state.status = "execution_planned"
            state.current_step = "execution_planned"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("create_execution_plan")
            
            # Add execution plan to messages
            execution_summary = f"New infrastructure execution plan created with {len(result['steps'])} steps. Estimated time: {result['total_estimated_time']}. Deployment strategy: {result['deployment_strategy']}"
            state.messages.append(AIMessage(content=execution_summary))
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure execution plan created",
                extra={
                    "steps_count": len(result['steps']),
                    "estimated_time": result['total_estimated_time'],
                    "deployment_strategy": result['deployment_strategy'],
                    "testing_phases_count": len(result['testing_phases'])
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Execution planning failed: {str(e)}"
            state.status = "error"
            raise
    
    def _validate_plan_node(self, state: PlannerState) -> PlannerState:
        """Validate the complete new infrastructure plan."""
        
        try:
            # Prepare complete plan for validation
            complete_plan = {
                "requirements": state.requirements_analysis or {},
                "infrastructure": {
                    "services": state.infrastructure_requirements or [],
                    "patterns": state.architectural_patterns or [],
                    "security": state.security_requirements or []
                },
                "execution": {
                    "steps": state.execution_plan or [],
                    "resources": state.resource_requirements or {}
                }
            }
            
            # Create prompt chain
            chain = self.validation_prompt | self.model | self.validation_parser
            
            # Execute validation
            result = chain.invoke({
                "messages": state.messages,
                "complete_plan": str(complete_plan)
            })
            
            # Update state with validation results
            state.validation_criteria = [result]
            state.status = "plan_validated"
            state.current_step = "plan_validated"
            if not hasattr(state, 'completed_steps'):
                state.completed_steps = []
            state.completed_steps.append("validate_plan")
            
            # Set planning complete flag for supervisor detection
            state.planning_complete = True
            
            # Add validation result to messages
            state.messages.append(
                AIMessage(content=f"New infrastructure plan validation completed. Review the validation results and recommendations.")
            )
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure plan validation completed",
                extra={
                    "validation_content_length": len(str(result))
                }
            )
            
            return state
            
        except Exception as e:
            state.error = f"Plan validation failed: {str(e)}"
            state.status = "error"
            raise
    
    # Removed hardcoded VPC-specific helper methods; questions are driven by LLM output
