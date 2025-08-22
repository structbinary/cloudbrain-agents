"""
Dependency Mapper React Agent for Planner Sub-Supervisor.

This module implements the Dependency Mapper as a React agent with tools:
- map_dependencies_tool: Maps AWS service dependencies based on requirements
- generate_questions_tool: Generates dependency questions for user clarification
- process_user_input_tool: Processes user responses for dependency refinement
"""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger

# Create logger
dependency_logger = AgentLogger("DEPENDENCY_MAPPER_REACT")

class DependencyMapping(BaseModel):
    """Output schema for dependency mapping analysis."""
    primary_service: str = Field(description="The main AWS service being requested")
    mandatory_dependencies: List[Dict[str, Any]] = Field(description="Dependencies that must be set up")
    optional_dependencies: List[Dict[str, Any]] = Field(description="Dependencies that may be needed")
    dependency_categories: Dict[str, List[str]] = Field(description="Dependencies grouped by category")
    setup_prerequisites: List[str] = Field(description="Prerequisites that must be in place")
    terraform_provider_requirements: List[str] = Field(description="Required Terraform providers")
    dependency_explanations: Dict[str, str] = Field(description="Explanations for why each dependency is needed")
    follow_up_questions: List[str] = Field(description="Questions to ask the user for clarification")

# Global variables for LLM and parsers
_model = None
_dependency_parser = None
_dependency_prompt = None

def _initialize_dependency_tools(config: Config):
    """Initialize LLM and parsers for dependency tools."""
    global _model, _dependency_parser, _dependency_prompt
    
    if _model is None:
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        _dependency_parser = JsonOutputParser(pydantic_object=DependencyMapping)
        
        _dependency_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an advanced AWS Terraform Dependency Planner. Analyze the user's request, identify the primary AWS service and dependencies, and produce a comprehensive dependency mapping.

[ROLE]
- Identify the primary AWS service being requested
- Map mandatory and optional dependencies
- Categorize dependencies by type (networking, security, monitoring, etc.)
- Identify setup prerequisites
- Determine Terraform provider requirements
- Generate follow-up questions for user clarification

[OUTPUT]
Provide a comprehensive dependency mapping including:
1. Primary Service: The main AWS service being requested
2. Mandatory Dependencies: Dependencies that must be set up
3. Optional Dependencies: Dependencies that may be needed
4. Dependency Categories: Grouped by type
5. Setup Prerequisites: What must be in place first
6. Terraform Provider Requirements: Required providers and versions
7. Dependency Explanations: Why each dependency is needed
8. Follow-up Questions: Questions for user clarification

[RESPONSE FORMAT]
Return a JSON object with the following structure:
{
    "primary_service": "service_name",
    "mandatory_dependencies": [{"service": "name", "reason": "explanation"}],
    "optional_dependencies": [{"service": "name", "reason": "explanation"}],
    "dependency_categories": {"category": ["service1", "service2"]},
    "setup_prerequisites": ["prerequisite1", "prerequisite2"],
    "terraform_provider_requirements": ["provider1", "provider2"],
    "dependency_explanations": {"service": "explanation"},
    "follow_up_questions": ["question1", "question2"]
}
            """),
            MessagesPlaceholder(variable_name="messages")
        ])

@tool
async def map_dependencies_tool(user_request: str, requirements_analysis: str) -> str:
    """
    Map AWS service dependencies based on user request and requirements.
    
    Args:
        user_request: The user's infrastructure request
        requirements_analysis: JSON string of requirements analysis
        
    Returns:
        JSON string containing dependency mapping
    """
    try:
        if _model is None:
            raise ValueError("Dependency tools not initialized. Call _initialize_dependency_tools first.")
        
        dependency_logger.log_structured(
            level="INFO",
            message="Starting async dependency mapping",
            extra={
                "user_request": user_request[:100] + "..." if len(user_request) > 100 else user_request,
                "requirements_analysis_length": len(requirements_analysis)
            }
        )
        
        # Prepare context for LLM
        context = f"User Request: {user_request}\nRequirements Analysis: {requirements_analysis}\n"
        
        # Prepare messages for LLM
        messages = [{"role": "user", "content": context}]
        
        # Get dependency mapping from LLM - now using await
        response = await _model.ainvoke(
            _dependency_prompt.format_messages(messages=messages)
        )
        
        dependency_logger.log_structured(
            level="DEBUG",
            message="Dependency mapping LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Parse the response
        try:
            dependency_data = _dependency_parser.parse(response.content)
            result = json.dumps(dependency_data.dict(), indent=2)
            
            dependency_logger.log_structured(
                level="INFO",
                message="Dependency mapping completed successfully",
                extra={
                    "primary_service": dependency_data.get("primary_service"),
                    "mandatory_dependencies_count": len(dependency_data.get("mandatory_dependencies", [])),
                    "optional_dependencies_count": len(dependency_data.get("optional_dependencies", [])),
                    "follow_up_questions_count": len(dependency_data.get("follow_up_questions", []))
                }
            )
            
        except Exception as parse_error:
            dependency_logger.log_structured(
                level="WARNING",
                message="Failed to parse dependency response, using fallback",
                extra={"error": str(parse_error), "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            # Fallback to basic dependency mapping
            fallback_data = {
                "primary_service": "aws_service",
                "mandatory_dependencies": [{"service": "vpc", "reason": "Networking foundation required"}],
                "optional_dependencies": [],
                "dependency_categories": {"networking": ["vpc"]},
                "setup_prerequisites": ["AWS account access"],
                "terraform_provider_requirements": ["aws"],
                "dependency_explanations": {"vpc": "Provides networking foundation"},
                "follow_up_questions": ["What is your preferred VPC CIDR block?"]
            }
            result = json.dumps(fallback_data, indent=2)
        
        return result
        
    except Exception as e:
        dependency_logger.log_structured(
            level="ERROR",
            message=f"Async dependency mapping failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Dependency mapping failed: {str(e)}"})

@tool
async def generate_questions_tool(dependency_mapping: str) -> str:
    """
    Generate dependency questions for user clarification.
    
    Args:
        dependency_mapping: JSON string of dependency mapping
        
    Returns:
        List of questions to ask the user
    """
    try:
        if _model is None:
            raise ValueError("Dependency tools not initialized. Call _initialize_dependency_tools first.")
        
        dependency_logger.log_structured(
            level="INFO",
            message="Starting async dependency question generation",
            extra={"dependency_mapping_length": len(dependency_mapping)}
        )
        
        # Parse the dependency mapping
        try:
            dependency_data = json.loads(dependency_mapping)
        except json.JSONDecodeError:
            dependency_logger.log_structured(
                level="ERROR",
                message="Invalid JSON format in dependency mapping",
                extra={"dependency_mapping_preview": dependency_mapping[:200] + "..." if len(dependency_mapping) > 200 else dependency_mapping}
            )
            return json.dumps({"error": "Invalid JSON format in dependency mapping"})
        
        # Create question generation prompt
        question_prompt = f"""
Based on the following dependency mapping, generate specific questions to ask the user for clarification:

{dependency_mapping}

Generate questions that will help clarify:
1. Specific configuration preferences
2. Resource sizing requirements
3. Security and compliance needs
4. Performance requirements
5. Integration requirements

Return a JSON array of questions:
["question1", "question2", "question3"]
"""
        
        # Get questions from LLM - now using await
        response = await _model.ainvoke([{"role": "user", "content": question_prompt}])
        
        dependency_logger.log_structured(
            level="DEBUG",
            message="Question generation LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Try to parse as JSON, fallback to text if needed
        try:
            questions = json.loads(response.content)
            
            dependency_logger.log_structured(
                level="INFO",
                message="Dependency questions generated successfully",
                extra={"questions_count": len(questions)}
            )
            
        except json.JSONDecodeError:
            dependency_logger.log_structured(
                level="WARNING",
                message="Failed to parse questions as JSON, using fallback",
                extra={"response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            questions = ["What is your preferred AWS region?", "What is your expected traffic volume?"]
        
        return json.dumps(questions, indent=2)
        
    except Exception as e:
        dependency_logger.log_structured(
            level="ERROR",
            message=f"Async question generation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Question generation failed: {str(e)}"})

@tool
async def process_user_input_tool(user_response: str, current_question: str, dependency_context: str) -> str:
    """
    Process user input for dependency clarification.
    
    Args:
        user_response: User's answer to the question
        current_question: The question that was asked
        dependency_context: Current dependency mapping context
        
    Returns:
        Updated dependency mapping with user input incorporated
    """
    try:
        if _model is None:
            raise ValueError("Dependency tools not initialized. Call _initialize_dependency_tools first.")
        
        dependency_logger.log_structured(
            level="INFO",
            message="Starting async user input processing for dependency clarification",
            extra={
                "user_response": user_response[:100] + "..." if len(user_response) > 100 else user_response,
                "current_question": current_question,
                "dependency_context_length": len(dependency_context)
            }
        )
        
        # Create processing prompt
        processing_prompt = f"""
Process the user's response to update the dependency mapping:

Question Asked: {current_question}
User Response: {user_response}
Current Dependency Context: {dependency_context}

Update the dependency mapping based on the user's response. Consider:
1. How the user's answer affects mandatory dependencies
2. Whether new optional dependencies should be added
3. Updates to setup prerequisites
4. Changes to Terraform provider requirements

Return the updated dependency mapping in JSON format.
"""
        
        # Get updated mapping from LLM - now using await
        response = await _model.ainvoke([{"role": "user", "content": processing_prompt}])
        
        dependency_logger.log_structured(
            level="DEBUG",
            message="User input processing LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Try to parse as JSON, fallback to original if needed
        try:
            updated_mapping = json.loads(response.content)
            
            dependency_logger.log_structured(
                level="INFO",
                message="User input processed successfully",
                extra={"mapping_updated": True, "updated_mapping_keys": list(updated_mapping.keys()) if isinstance(updated_mapping, dict) else "not_dict"}
            )
            
        except json.JSONDecodeError:
            dependency_logger.log_structured(
                level="WARNING",
                message="Failed to parse updated mapping as JSON, using original mapping",
                extra={"response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            # Fallback to original mapping
            updated_mapping = json.loads(dependency_context)
        
        return json.dumps(updated_mapping, indent=2)
        
    except Exception as e:
        dependency_logger.log_structured(
            level="ERROR",
            message=f"Async user input processing failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"User input processing failed: {str(e)}"})

def create_dependency_mapper_react_agent(config: Config):
    """
    Create a React agent for dependency mapping.
    
    Args:
        config: Configuration instance
        
    Returns:
        React agent for dependency mapping
    """
    try:
        dependency_logger.log_structured(
            level="INFO",
            message="=== CREATING DEPENDENCY MAPPER REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        # Initialize tools
        dependency_logger.log_structured(
            level="DEBUG",
            message="Initializing dependency tools",
            extra={}
        )
        
        _initialize_dependency_tools(config)
        
        # Get LLM from config
        llm_config = config.get_llm_config()
        
        dependency_logger.log_structured(
            level="DEBUG",
            message="Creating LLM for dependency mapper",
            extra={
                "llm_provider": llm_config.get('provider'),
                "llm_model": llm_config.get('model'),
                "llm_temperature": llm_config.get('temperature'),
                "llm_max_tokens": llm_config.get('max_tokens')
            }
        )
        
        llm = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        # Create React agent with async tools
        dependency_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 3,
                "tool_names": ["map_dependencies_tool", "generate_questions_tool", "process_user_input_tool"]
            }
        )
        
        dependency_mapper = create_react_agent(
            model=llm,
            tools=[map_dependencies_tool, generate_questions_tool, process_user_input_tool],
            name="dependency_mapper",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert AWS Dependency Mapper. Your role is to analyze infrastructure requirements and map AWS service dependencies.

[ROLE]
- Map AWS service dependencies based on requirements
- Generate questions for user clarification
- Process user responses for dependency refinement
- Identify mandatory and optional dependencies

[WORKFLOW]
1. Use map_dependencies_tool to analyze dependencies from requirements
2. Use generate_questions_tool to create clarification questions
3. Use process_user_input_tool to incorporate user responses
4. Provide comprehensive dependency mapping summary

[OUTPUT]
Provide a clear, structured response that includes:
- Primary AWS service being requested
- Mandatory and optional dependencies
- Dependency categories and explanations
- Setup prerequisites and Terraform requirements
- Follow-up questions for user clarification
            """),
                ("user", "{input}")
            ])
        )
        
        dependency_logger.log_structured(
            level="INFO",
            message="=== DEPENDENCY MAPPER REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(dependency_mapper).__name__,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "tools_count": 3
            }
        )
        
        return dependency_mapper
        
    except Exception as e:
        dependency_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE DEPENDENCY MAPPER REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
