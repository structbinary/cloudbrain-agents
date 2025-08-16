"""
New Infrastructure Planner Implementation.

This module implements the NewInfrastructurePlanner, which is specialized for:
- Creating new AWS infrastructure from scratch
- Greenfield project planning
- Service selection and architectural design
- Cost estimation for new deployments

The NewInfrastructurePlanner inherits from BaseAgent and focuses on new infrastructure scenarios.
"""

import time
import uuid
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from functools import wraps

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


from aws_orchestrator_agent.core.agents.base_agent import BaseAgent, agent_node, require_approval
from aws_orchestrator_agent.core.agents.planner.planner_state import (
    PlannerState,
    InfrastructureRequirement,
    ArchitecturalPattern,
    SecurityRequirement,
    ExecutionStep
)
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config


# Create agent logger for new infrastructure planner
new_infra_logger = AgentLogger("NEW_INFRA_PLANNER")


class NewInfrastructureRequirements(BaseModel):
    """Output schema for new infrastructure requirements analysis."""
    business_requirements: List[str]
    technical_requirements: List[str]
    constraints: List[str]
    assumptions: List[str]
    risk_factors: List[str]
    scalability_requirements: List[str]
    performance_requirements: List[str]


class NewInfrastructurePlan(BaseModel):
    """Output schema for new infrastructure planning."""
    services: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    security_requirements: List[Dict[str, Any]]
    cost_estimates: Dict[str, Any]
    dependencies: Dict[str, List[str]]
    scalability_considerations: List[str]
    performance_optimizations: List[str]


class NewInfrastructureExecutionPlan(BaseModel):
    """Output schema for new infrastructure execution planning."""
    steps: List[Dict[str, Any]]
    total_estimated_time: str
    critical_path: List[int]
    resource_requirements: Dict[str, Any]
    deployment_strategy: str
    testing_phases: List[str]


class NewInfrastructureValidationResult(BaseModel):
    """Output schema for new infrastructure validation."""
    validation_results: Dict[str, List[str]]
    overall_assessment: str
    recommendations: List[str]
    risk_level: str


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


class NewInfrastructurePlanner(BaseAgent):
    """
    New Infrastructure Planner for AWS greenfield projects.
    
    This agent specializes in creating new AWS infrastructure from scratch,
    focusing on architectural design, service selection, and deployment planning.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "new_infrastructure_planner"
    ):
        """
        Initialize the New Infrastructure Planner.
        
        Args:
            config: Configuration instance
            custom_config: Optional custom configuration
            name: Agent name
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Get LLM configuration
        llm_config = self.config_instance.get_llm_config()
        
        # Initialize LLM model
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            new_infra_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model for new infrastructure planner: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model for new infrastructure planner: {e}",
                extra={"error": str(e)}
            )
            raise
        
        # Initialize base agent
        super().__init__(
            name=name,
            state_schema=PlannerState,
            config=custom_config or {},
            mcp_client=None,  # Will be set by supervisor if needed
            state_manager=None  # Will be set by supervisor if needed
        )
        
        # Initialize output parsers
        self.requirements_parser = JsonOutputParser(pydantic_object=NewInfrastructureRequirements)
        self.infrastructure_parser = JsonOutputParser(pydantic_object=NewInfrastructurePlan)
        self.execution_parser = JsonOutputParser(pydantic_object=NewInfrastructureExecutionPlan)
        self.validation_parser = JsonOutputParser(pydantic_object=NewInfrastructureValidationResult)
        
        # Initialize the dependency mapping tool (no agent executor)
        self.dependency_mapping_tool = self._create_dependency_mapping_tool()
        
        # Define prompts
        self._define_prompts()
        
        # Build the agent graph
        self._build_planner_graph()
    
    def _define_prompts(self) -> None:
        """Define the prompts used by the new infrastructure planner."""
        
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
    
    def _build_planner_graph(self) -> None:
        """Build the new infrastructure planner's StateGraph."""
        
        # Define nodes
        nodes = self.define_nodes()
        for name, node_func in nodes.items():
            self.add_node(name, node_func)
        
        # Build and compile the graph
        self.build_graph()
        self.compile_graph()
        
        new_infra_logger.log_structured(
            level="INFO",
            message="New infrastructure planner graph built and compiled successfully",
            extra={"agent_name": self.name, "node_count": len(nodes)}
        )
    
    def _create_dependency_mapping_tool(self):
        """Create the dependency mapping tool."""
        
        @tool
        def dependency_mapping_tool(user_request: str) -> str:
            """
            Analyze a new infrastructure request and map all dependencies for Terraform module creation.
            
            This tool examines the user's request, identifies the primary AWS service, and creates a comprehensive
            checklist of both mandatory and optional dependencies. It acts as an expert checklist maker that
            explains what needs to be in place before the main service can work.
            
            Args:
                user_request: The user's infrastructure request (e.g., "Create an S3 bucket for storing application data")
            
            Returns:
                A JSON string containing the dependency mapping analysis with the following structure:
                {
                    "primary_service": "string",
                    "mandatory_dependencies": [...],
                    "optional_dependencies": [...],
                    "dependency_categories": {...},
                    "setup_prerequisites": [...],
                    "terraform_provider_requirements": [...],
                    "dependency_explanations": {...},
                    "follow_up_questions": [...]
                }
            """
            
            # Create the dependency mapping prompt
            dependency_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AWS infrastructure architect specializing in dependency mapping for Terraform modules. Your role is to analyze user requests and create comprehensive dependency checklists.

When analyzing a request, you must:

1. **Identify the Primary Service**: Determine the main AWS service the user wants to create
2. **Map Mandatory Dependencies**: List all resources that MUST be created first for the primary service to function
3. **Map Optional Dependencies**: List resources that MAY be needed depending on specific use cases
4. **Categorize Dependencies**: Group dependencies by category (networking, security, monitoring, etc.)
5. **Explain Dependencies**: Provide clear explanations for why each dependency is needed
6. **Identify Prerequisites**: List any setup requirements before creating the main service
7. **Specify Terraform Requirements**: Identify required Terraform providers and versions
8. **Generate Follow-up Questions**: Create questions to clarify requirements and dependencies

Additionally, when the user's request concerns a resource that inherently depends on other resources (for example: Subnet depends on VPC; NAT Gateway depends on Elastic IP and Route Tables), you must:

- Explicitly include those dependencies under "mandatory_dependencies" with clear Terraform resources and configuration notes
- Include a follow-up question that asks whether the user wants you to provision the dependency resources as part of this plan OR use existing ones
- If using existing resources is an option, also ask for the exact identifiers required (e.g., VPC ID, Subnet IDs, Route Table IDs, Security Group IDs)
- Phrase these follow-ups generically so they work for any resource, not just the provided examples
- Also include a generic follow-up to clarify INTENT and SCOPE, asking whether the user wants only the requested primary service module, the mandatory dependency modules, or a minimal functional stack required for a working setup

IMPORTANT: You must respond with a valid JSON object that matches the following structure exactly:
{{
  "primary_service": "string",
  "mandatory_dependencies": [
    {{
      "service": "string",
      "purpose": "string",
      "category": "string",
      "terraform_resource": "string",
      "configuration_notes": "string"
    }}
  ],
  "optional_dependencies": [
    {{
      "service": "string",
      "purpose": "string",
      "category": "string",
      "terraform_resource": "string",
      "configuration_notes": "string",
      "when_needed": "string"
    }}
  ],
  "dependency_categories": {{
    "networking": ["dependency1", "dependency2"],
    "security": ["dependency1", "dependency2"],
    "monitoring": ["dependency1", "dependency2"],
    "storage": ["dependency1", "dependency2"]
  }},
  "setup_prerequisites": ["prerequisite1", "prerequisite2"],
  "terraform_provider_requirements": ["provider1", "provider2"],
  "dependency_explanations": {{
    "dependency_name": "explanation of why this is needed"
  }},
  "follow_up_questions": [
    "question1",
    "question2"
  ]
}}

Focus on practical, implementation-ready dependency mapping that will guide Terraform module creation. Do not include any markdown formatting, explanations, or additional text outside the JSON structure."""),
                ("human", "Map dependencies for the following new infrastructure request: {user_request}")
            ])
            
            # Create the parser
            parser = JsonOutputParser(pydantic_object=NewInfrastructureDependencyMapping)
            
            # Create the chain
            chain = dependency_prompt | self.model | parser
            
            # Execute the analysis
            result = chain.invoke({"user_request": user_request})
            
            # Ensure generic clarifying follow-up about provisioning vs. existing dependencies
            try:
                if isinstance(result, dict):
                    followups = result.get("follow_up_questions") or []
                    mandatory = result.get("mandatory_dependencies") or []
                    dependency_services = [
                        d.get("service") for d in mandatory if isinstance(d, dict) and d.get("service")
                    ]
                    if dependency_services:
                        deps_str = ", ".join(sorted(set(dependency_services)))
                        generic_q = (
                            f"For mandatory dependencies [{deps_str}], should I provision these as part of this plan "
                            f"or use existing resources? If using existing, please provide exact identifiers (e.g., IDs)."
                        )
                        if generic_q not in followups:
                            followups.append(generic_q)
                            result["follow_up_questions"] = followups
            except Exception:
                # Non-fatal; return original result
                pass

            # Return the result as a JSON string
            return json.dumps(result, indent=2)
        
        return dependency_mapping_tool
    
    def _perform_dependency_mapping_analysis(self, user_request: str) -> Dict[str, Any]:
        """Perform dependency mapping analysis directly using the tool."""
        
        # Call the dependency mapping tool directly
        result_json = self.dependency_mapping_tool.invoke(user_request)
        
        # Parse the JSON result
        if isinstance(result_json, str):
            return json.loads(result_json)
        else:
            return result_json
    

    
    async def dependency_mapping_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Execute dependency mapping using ReAct agent with interactive follow-up."""
        
        # Get current state - handle both dict and Pydantic objects
        if hasattr(state, 'model_dump'):
            current_state = state
        else:
            current_state = PlannerState(**state)
        
        # Check if we're resuming from a supervisor interrupt
        resume_value = getattr(current_state, 'resume_value', None)
        if resume_value and isinstance(resume_value, dict):
            # Extract the answer from the resume value
            user_answer = resume_value.get('answer', '')
            question = resume_value.get('question', '')
            
            if user_answer and question:
                # Add user's answer to the conversation
                current_state.messages.append(
                    HumanMessage(content=f"Answer to '{question}': {user_answer}")
                )
                
                # Store the answer in the state
                current_state.dependency_answers[question] = user_answer
                
                # Clear the resume value
                current_state.resume_value = None
                
                # Continue with the dependency mapping using the new information
                return await self._continue_dependency_mapping(current_state, user_answer)
        
        # Check if we're waiting for user input
        if current_state.waiting_for_dependency_input:
            # Process user's answer to the current question
            if current_state.current_dependency_question and current_state.dependency_answers:
                # Get the answer for the current question
                question_key = current_state.current_dependency_question
                user_answer = current_state.dependency_answers.get(question_key, "")
                
                if user_answer:
                    # Add user's answer to the conversation
                    current_state.messages.append(
                        HumanMessage(content=f"Answer to '{question_key}': {user_answer}")
                    )
                    
                    # Clear the current question and continue processing
                    current_state.current_dependency_question = None
                    current_state.waiting_for_dependency_input = False
                    
                    # Continue with the dependency mapping using the new information
                    return await self._continue_dependency_mapping(current_state, user_answer)
                else:
                    # Still waiting for input
                    interrupt({
                        "question": current_state.current_dependency_question,
                        "context": "dependency_mapping",
                        "available_questions": current_state.dependency_questions
                    })
                    return current_state.model_dump()
            else:
                # No current question, but waiting for input - this shouldn't happen
                current_state.waiting_for_dependency_input = False
        
        # Initial dependency mapping or continuation
        return await self._perform_initial_dependency_mapping(current_state)
    
    async def _perform_initial_dependency_mapping(self, current_state: PlannerState) -> Dict[str, Any]:
        """Perform initial dependency mapping analysis."""
        
        # Get the user request
        user_request = current_state.user_request
        
        # Execute the dependency mapping tool directly (no AgentExecutor)
        # Note: No try-catch here to allow interrupts to flow naturally
        # Perform the dependency mapping analysis
        final_result = self._perform_dependency_mapping_analysis(user_request)
        
        # Check if there are follow-up questions
        follow_up_questions = final_result.get('follow_up_questions', [])
        
        if follow_up_questions:
            # Store questions and ask the first one
            current_state.dependency_questions = follow_up_questions
            current_state.current_dependency_question = follow_up_questions[0]
            current_state.waiting_for_dependency_input = True
            current_state.status = "waiting_for_dependency_input"
            
            # Store partial results
            current_state.planning_metadata["dependency_mapping_partial"] = final_result
            
            # Use LangGraph interrupt properly - this will pause the graph and await input
            interrupt_data = {
                "question": follow_up_questions[0],
                "context": "dependency_mapping",
                "available_questions": follow_up_questions,
                "partial_analysis": {
                    "primary_service": final_result.get('primary_service', 'unknown'),
                    "mandatory_dependencies_count": len(final_result.get('mandatory_dependencies', [])),
                    "optional_dependencies_count": len(final_result.get('optional_dependencies', []))
                }
            }
            
            # Update state to indicate we're waiting for dependency input
            current_state.waiting_for_dependency_input = True
            current_state.current_dependency_question = interrupt_data['question']
            current_state.dependency_questions = interrupt_data['available_questions']
            current_state.planning_metadata = {
                "dependency_mapping_partial": interrupt_data['partial_analysis'],
                "interrupt_data": interrupt_data
            }
            
            # Create interrupt message to be added to state
            interrupt_message = AIMessage(
                content=f"INTERRUPT: {interrupt_data['question']}",
                additional_kwargs={
                    "interrupt_data": interrupt_data,
                    "interrupt_type": "dependency_mapping",
                    "waiting_for_dependency_input": True,
                    "current_dependency_question": interrupt_data['question'],
                    "dependency_questions": interrupt_data['available_questions'],
                    "planning_metadata": {
                        "dependency_mapping_partial": interrupt_data['partial_analysis'],
                        "interrupt_data": interrupt_data
                    }
                }
            )
            
            # Add the interrupt message to state
            current_state.messages.append(interrupt_message)
            
            # Use LangGraph interrupt for human-in-the-loop
            # This will pause the graph and surface the question to the supervisor
            
            
            # Call interrupt() with the interrupt data
            # This will pause execution and surface the question
            interrupt(interrupt_data)
            
            # Return the updated state
            return current_state.model_dump()
        else:
            # No questions, complete the mapping
            return await self._complete_dependency_mapping(current_state, final_result)
    
    async def _continue_dependency_mapping(self, current_state: PlannerState, user_answer: str) -> Dict[str, Any]:
        """Continue dependency mapping with user's answer."""
        
        # Add user's answer to the conversation
        enhanced_context = f"User provided additional information: {user_answer}"
        
        # Get the next question or complete the mapping
        remaining_questions = [q for q in current_state.dependency_questions if q not in current_state.dependency_answers]
        
        if remaining_questions:
            # Ask the next question
            next_question = remaining_questions[0]
            current_state.current_dependency_question = next_question
            current_state.waiting_for_dependency_input = True
            
            # Use LangGraph interrupt properly - this will pause the graph and await input
            interrupt_data = {
                "question": next_question,
                "context": "dependency_mapping",
                "available_questions": remaining_questions,
                "previous_answer": user_answer
            }
            
            # Update state to indicate we're waiting for dependency input
            current_state.waiting_for_dependency_input = True
            current_state.current_dependency_question = interrupt_data['question']
            current_state.dependency_questions = interrupt_data['available_questions']
            current_state.planning_metadata.update({
                "dependency_mapping_partial": current_state.planning_metadata.get("dependency_mapping_partial", {}),
                "interrupt_data": interrupt_data
            })
            
            # Create interrupt message to be added to state
            interrupt_message = AIMessage(
                content=f"INTERRUPT: {interrupt_data['question']}",
                additional_kwargs={
                    "interrupt_data": interrupt_data,
                    "interrupt_type": "dependency_mapping",
                    "waiting_for_dependency_input": True,
                    "current_dependency_question": interrupt_data['question'],
                    "dependency_questions": interrupt_data['available_questions'],
                    "planning_metadata": {
                        "dependency_mapping_partial": current_state.planning_metadata.get("dependency_mapping_partial", {}),
                        "interrupt_data": interrupt_data
                    }
                }
            )
            
            # Add the interrupt message to state
            current_state.messages.append(interrupt_message)
            
            # Use LangGraph interrupt for human-in-the-loop
            # This will pause the graph and surface the question to the supervisor
            # Call interrupt() with the interrupt data
            # This will pause execution and surface the question
            interrupt(interrupt_data)
            
            # Return the updated state
            return current_state.model_dump()
        else:
            # All questions answered, complete the mapping
            partial_result = current_state.planning_metadata.get("dependency_mapping_partial", {})
            
            # Enhance the result with user answers
            enhanced_result = self._enhance_dependency_mapping_with_answers(partial_result, current_state.dependency_answers)
            
            return await self._complete_dependency_mapping(current_state, enhanced_result)
    
    def _enhance_dependency_mapping_with_answers(self, partial_result: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """Enhance dependency mapping with user answers."""
        
        # Create a new result that incorporates user answers
        enhanced_result = partial_result.copy()
        
        # Add user answers to the result
        enhanced_result["user_answers"] = answers
        enhanced_result["enhanced_analysis"] = True
        
        # You could add logic here to refine dependencies based on user answers
        # For example, if user says they need high availability, add more redundancy dependencies
        
        return enhanced_result
    
    async def _complete_dependency_mapping(self, current_state: PlannerState, final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Complete the dependency mapping process."""
        
        # Update state with final dependency mapping results
        current_state.planning_metadata["dependency_mapping"] = final_result
        current_state.status = "dependencies_mapped"
        current_state.current_step = "dependencies_mapped"
        current_state.completed_steps.append("dependency_mapping")
        current_state.dependency_mapping_complete = True
        current_state.waiting_for_dependency_input = False
        current_state.current_dependency_question = None
        
        # Add dependency mapping result to messages
        current_state.messages.append(
            AIMessage(content=f"Dependency mapping completed for {final_result.get('primary_service', 'infrastructure')}. Identified {len(final_result.get('mandatory_dependencies', []))} mandatory and {len(final_result.get('optional_dependencies', []))} optional dependencies.")
        )
        
        new_infra_logger.log_structured(
            level="INFO",
            message="New infrastructure dependency mapping completed",
            extra={
                "agent_name": self.name,
                "primary_service": final_result.get('primary_service', 'unknown'),
                "mandatory_dependencies_count": len(final_result.get('mandatory_dependencies', [])),
                "optional_dependencies_count": len(final_result.get('optional_dependencies', [])),
                "categories_count": len(final_result.get('dependency_categories', {})),
                "follow_up_questions_count": len(final_result.get('follow_up_questions', [])),
                "user_answers_count": len(current_state.dependency_answers)
            }
        )
        
        return current_state.model_dump()
    
    def define_nodes(self) -> Dict[str, Any]:
        """Define the new infrastructure planner's nodes."""
        from aws_orchestrator_agent.core.agents.base_agent import agent_node
        
        # Create wrapper functions with proper names for async methods
        async def dependency_mapping_wrapper(state):
            return await self.dependency_mapping_node(state)
        
        async def analyze_requirements_wrapper(state):
            return await self.analyze_requirements_node(state)
        
        async def plan_infrastructure_wrapper(state):
            return await self.plan_infrastructure_node(state)
        
        async def create_execution_plan_wrapper(state):
            return await self.create_execution_plan_node(state)
        
        async def validate_plan_wrapper(state):
            return await self.validate_plan_node(state)
        
        async def finalize_plan_wrapper(state):
            return await self.finalize_plan_node(state)
        
        return {
            "dependency_mapping": agent_node(dependency_mapping_wrapper),  # NEW - FIRST NODE
            "analyze_requirements": agent_node(analyze_requirements_wrapper),
            "plan_infrastructure": agent_node(plan_infrastructure_wrapper),
            "create_execution_plan": agent_node(create_execution_plan_wrapper),
            "validate_plan": agent_node(validate_plan_wrapper),
            "finalize_plan": agent_node(finalize_plan_wrapper)
        }
    
    def define_edges(self) -> Dict[str, List[str]]:
        """Define the edges between nodes."""
        return {
            "dependency_mapping": ["analyze_requirements"],  # NEW - FIRST EDGE
            "analyze_requirements": ["plan_infrastructure"],
            "plan_infrastructure": ["create_execution_plan"],
            "create_execution_plan": ["validate_plan"],
            "validate_plan": ["finalize_plan"],
            "finalize_plan": []
        }
    
    def create_initial_state(self, input_data: Dict[str, Any]) -> PlannerState:
        """Create the initial state for the new infrastructure planner."""
        
        # Extract user request
        user_request = input_data.get("user_request", "")
        context_id = input_data.get("context_id", str(uuid.uuid4()))
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        
        # Create initial messages
        messages = [
            SystemMessage(content="You are an expert AWS infrastructure planner specializing in new infrastructure projects. Create a comprehensive plan for the new deployment."),
            HumanMessage(content=user_request)
        ]
        
        # Create initial state
        state = PlannerState(
            messages=messages,
            user_request=user_request,
            context_id=context_id,
            task_id=task_id,
            request_type="new",  # Always new for this planner
            status="planning_started",
            planning_started_at=datetime.utcnow().isoformat()
        )
        
        new_infra_logger.log_structured(
            level="INFO",
            message="Created initial new infrastructure planner state",
            extra={
                "agent_name": self.name,
                "context_id": context_id,
                "task_id": task_id,
                "user_request_length": len(user_request)
            }
        )
        
        return state
    
    async def analyze_requirements_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Analyze user requirements for new infrastructure."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Get dependency mapping from previous step
            dependency_mapping = current_state.planning_metadata.get("dependency_mapping", {})
            
            # Create prompt chain
            chain = self.requirements_prompt | self.model | self.requirements_parser
            
            # Enhance the requirements analysis with dependency context
            enhanced_user_request = f"{current_state.user_request}\n\nDependency Context: {str(dependency_mapping)}"
            
            # Execute analysis with enhanced context
            result = chain.invoke({
                "messages": current_state.messages,
                "user_request": enhanced_user_request
            })
            
            # Update state with analysis results
            current_state.requirements_analysis = result
            current_state.status = "requirements_analyzed"
            current_state.current_step = "requirements_analyzed"
            current_state.completed_steps.append("analyze_requirements")
            
            # Add analysis result to messages
            current_state.messages.append(
                AIMessage(content=f"New infrastructure requirements analysis completed. Found {len(result['business_requirements'])} business requirements, {len(result['technical_requirements'])} technical requirements, and {len(result['scalability_requirements'])} scalability requirements.")
            )
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure requirements analysis completed",
                extra={
                    "agent_name": self.name,
                    "business_requirements_count": len(result['business_requirements']),
                    "technical_requirements_count": len(result['technical_requirements']),
                    "scalability_requirements_count": len(result['scalability_requirements']),
                    "constraints_count": len(result['constraints'])
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error in new infrastructure requirements analysis: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "requirements_analysis"
            
            return current_state.model_dump()
    
    async def plan_infrastructure_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Plan new infrastructure based on requirements analysis."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Get dependency mapping and requirements analysis from state
            dependency_mapping = current_state.planning_metadata.get("dependency_mapping", {})
            requirements_analysis = current_state.requirements_analysis or {}
            
            # Create enhanced infrastructure plan context
            enhanced_context = {
                "requirements": requirements_analysis,
                "dependencies": dependency_mapping
            }
            
            # Create prompt chain
            chain = self.infrastructure_prompt | self.model | self.infrastructure_parser
            
            # Execute infrastructure planning with dependency context
            result = chain.invoke({
                "messages": current_state.messages,
                "requirements_analysis": str(enhanced_context)
            })
            
            # Convert results to Pydantic models
            infrastructure_requirements = [
                InfrastructureRequirement(**service) for service in result['services']
            ]
            
            architectural_patterns = [
                ArchitecturalPattern(**pattern) for pattern in result['patterns']
            ]
            
            security_requirements = [
                SecurityRequirement(**req) for req in result['security_requirements']
            ]
            
            # Update state
            current_state.infrastructure_requirements = infrastructure_requirements
            current_state.architectural_patterns = architectural_patterns
            current_state.security_requirements = security_requirements
            current_state.cost_analysis = result['cost_estimates']
            current_state.status = "infrastructure_planned"
            current_state.current_step = "infrastructure_planned"
            current_state.completed_steps.append("plan_infrastructure")
            
            # Add planning result to messages
            planning_summary = f"New infrastructure planning completed. Identified {len(infrastructure_requirements)} services, {len(architectural_patterns)} architectural patterns, and {len(result['scalability_considerations'])} scalability considerations."
            current_state.messages.append(AIMessage(content=planning_summary))
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure planning completed",
                extra={
                    "agent_name": self.name,
                    "services_count": len(infrastructure_requirements),
                    "patterns_count": len(architectural_patterns),
                    "security_requirements_count": len(security_requirements),
                    "scalability_considerations_count": len(result['scalability_considerations'])
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error in new infrastructure planning: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "infrastructure_planning"
            
            return current_state.model_dump()
    
    async def create_execution_plan_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Create a detailed execution plan for new infrastructure."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Get infrastructure plan from state
            infrastructure_plan = {
                "services": [req.dict() for req in current_state.infrastructure_requirements or []],
                "patterns": [pattern.dict() for pattern in current_state.architectural_patterns or []],
                "security": [req.dict() for req in current_state.security_requirements or []]
            }
            
            # Create prompt chain
            chain = self.execution_prompt | self.model | self.execution_parser
            
            # Execute execution planning
            result = chain.invoke({
                "messages": current_state.messages,
                "infrastructure_plan": str(infrastructure_plan)
            })
            
            # Convert results to Pydantic models
            execution_steps = [
                ExecutionStep(**step) for step in result['steps']
            ]
            
            # Update state
            current_state.execution_plan = execution_steps
            current_state.resource_requirements = result['resource_requirements']
            current_state.status = "execution_planned"
            current_state.current_step = "execution_planned"
            current_state.completed_steps.append("create_execution_plan")
            
            # Add execution plan to messages
            execution_summary = f"New infrastructure execution plan created with {len(execution_steps)} steps. Estimated time: {result['total_estimated_time']}. Deployment strategy: {result['deployment_strategy']}"
            current_state.messages.append(AIMessage(content=execution_summary))
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure execution plan created",
                extra={
                    "agent_name": self.name,
                    "steps_count": len(execution_steps),
                    "estimated_time": result['total_estimated_time'],
                    "deployment_strategy": result['deployment_strategy'],
                    "testing_phases_count": len(result['testing_phases'])
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error in new infrastructure execution planning: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "execution_planning"
            
            return current_state.model_dump()
    
    async def validate_plan_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Validate the complete new infrastructure plan."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Prepare complete plan for validation
            complete_plan = {
                "requirements": current_state.requirements_analysis or {},
                "infrastructure": {
                    "services": [req.dict() for req in current_state.infrastructure_requirements or []],
                    "patterns": [pattern.dict() for pattern in current_state.architectural_patterns or []],
                    "security": [req.dict() for req in current_state.security_requirements or []]
                },
                "execution": {
                    "steps": [step.dict() for step in current_state.execution_plan or []],
                    "resources": current_state.resource_requirements or {}
                }
            }
            
            # Create prompt chain
            chain = self.validation_prompt | self.model | self.validation_parser
            
            # Execute validation
            result = chain.invoke({
                "messages": current_state.messages,
                "complete_plan": str(complete_plan)
            })
            
            # Extract validation results
            validation_content = result
            
            # Update state with validation results
            current_state.validation_criteria = [validation_content]
            current_state.status = "plan_validated"
            current_state.current_step = "plan_validated"
            current_state.completed_steps.append("validate_plan")
            
            # Add validation result to messages
            current_state.messages.append(
                AIMessage(content=f"New infrastructure plan validation completed. Review the validation results and recommendations.")
            )
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure plan validation completed",
                extra={
                    "agent_name": self.name,
                    "validation_content_length": len(validation_content)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error in new infrastructure plan validation: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "plan_validation"
            
            return current_state.model_dump()
    
    @require_approval
    async def finalize_plan_node(self, state: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Finalize the new infrastructure plan and prepare for execution."""
        
        try:
            # Get current state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(current_state.model_dump())
            
            # Update final state
            current_state.complexity_score = complexity_score
            current_state.status = "plan_completed"
            current_state.planning_completed_at = datetime.utcnow().isoformat()
            current_state.current_step = "plan_completed"
            current_state.completed_steps.append("finalize_plan")
            
            # Calculate planning duration
            if current_state.planning_started_at:
                start_time = datetime.fromisoformat(current_state.planning_started_at)
                end_time = datetime.fromisoformat(current_state.planning_completed_at)
                current_state.planning_duration = (end_time - start_time).total_seconds()
            
            # Add finalization message
            current_state.messages.append(
                AIMessage(content=f"New infrastructure planning completed successfully! Complexity score: {complexity_score}/10. The plan is ready for execution.")
            )
            
            new_infra_logger.log_structured(
                level="INFO",
                message="New infrastructure plan finalized",
                extra={
                    "agent_name": self.name,
                    "complexity_score": complexity_score,
                    "planning_duration": current_state.planning_duration,
                    "total_steps": len(current_state.completed_steps)
                }
            )
            
            return current_state.model_dump()
            
        except Exception as e:
            # Handle both dict and Pydantic object states
            user_request = ""
            if hasattr(state, 'user_request'):
                # It's a Pydantic object
                user_request = state.user_request
            elif isinstance(state, dict):
                # It's a dictionary
                user_request = state.get("user_request", "")
            
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error in new infrastructure plan finalization: {e}",
                extra={"error": str(e), "user_request": user_request}
            )
            
            # Set error state - handle both dict and Pydantic objects
            if hasattr(state, 'model_dump'):
                # It's already a Pydantic object
                current_state = state
            else:
                # It's a dictionary, convert to Pydantic object
                current_state = PlannerState(**state)
            
            current_state.error = str(e)
            current_state.error_context = "plan_finalization"
            
            return current_state.model_dump()
    
    def _calculate_complexity_score(self, state: Dict[str, Any]) -> int:
        """Calculate complexity score for new infrastructure plan."""
        try:
            # Base complexity
            complexity = 3
            
            # Add complexity based on number of services
            services_count = len(state.get("infrastructure_requirements", []))
            if services_count > 10:
                complexity += 3
            elif services_count > 5:
                complexity += 2
            elif services_count > 2:
                complexity += 1
            
            # Add complexity based on architectural patterns
            patterns_count = len(state.get("architectural_patterns", []))
            if patterns_count > 3:
                complexity += 2
            elif patterns_count > 1:
                complexity += 1
            
            # Add complexity based on execution steps
            steps_count = len(state.get("execution_plan", []))
            if steps_count > 20:
                complexity += 2
            elif steps_count > 10:
                complexity += 1
            
            # Cap at 10
            return min(complexity, 10)
            
        except Exception as e:
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error calculating complexity score: {e}",
                extra={"error": str(e)}
            )
            return 5  # Default complexity
    
    @log_sync
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the new infrastructure plan."""
        try:
            state = self.get_state()
            if not state:
                return {"error": "No plan state available"}
            
            return {
                "plan_type": "new_infrastructure",
                "status": state.get("status"),
                "complexity_score": state.get("complexity_score"),
                "services_count": len(state.get("infrastructure_requirements", [])),
                "patterns_count": len(state.get("architectural_patterns", [])),
                "execution_steps_count": len(state.get("execution_plan", [])),
                "planning_duration": state.get("planning_duration"),
                "completed_steps": state.get("completed_steps", [])
            }
        except Exception as e:
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error getting plan summary: {e}",
                extra={"error": str(e)}
            )
            return {"error": str(e)}
    
    @log_sync
    def export_plan(self, format: str = "json") -> Dict[str, Any]:
        """Export the new infrastructure plan in the specified format."""
        try:
            state = self.get_state()
            if not state:
                return {"error": "No plan state available"}
            
            if format.lower() == "json":
                return {
                    "plan_type": "new_infrastructure",
                    "exported_at": datetime.utcnow().isoformat(),
                    "plan_data": state.model_dump() if hasattr(state, 'model_dump') else state
                }
            else:
                return {"error": f"Unsupported format: {format}"}
                
        except Exception as e:
            new_infra_logger.log_structured(
                level="ERROR",
                message=f"Error exporting plan: {e}",
                extra={"error": str(e), "format": format}
            )
            return {"error": str(e)}


@log_sync
def create_new_infrastructure_planner(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "new_infrastructure_planner"
) -> NewInfrastructurePlanner:
    """
    Factory function to create a NewInfrastructurePlanner instance.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured NewInfrastructurePlanner instance
    """
    return NewInfrastructurePlanner(config=config, custom_config=custom_config, name=name) 