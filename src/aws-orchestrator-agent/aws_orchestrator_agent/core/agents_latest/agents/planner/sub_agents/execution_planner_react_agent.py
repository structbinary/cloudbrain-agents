"""
Execution Planner React Agent for Planner Sub-Supervisor.

This module implements the Execution Planner as a React agent with tools:
- create_execution_plan_tool: Creates execution plans based on requirements and dependencies
- assess_risks_tool: Assesses risks associated with execution plans
- calculate_complexity_score_tool: Calculates complexity scores
- validate_execution_plan_tool: Validates execution plans for feasibility
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
execution_logger = AgentLogger("EXECUTION_PLANNER_REACT")

class ExecutionPlan(BaseModel):
    """Output schema for execution planning."""
    steps: List[Dict[str, Any]] = Field(description="Step-by-step execution plan")
    total_estimated_time: str = Field(description="Total estimated execution time")
    critical_path: List[int] = Field(description="Critical path steps")
    resource_requirements: Dict[str, Any] = Field(description="Resource requirements")
    deployment_strategy: str = Field(description="Deployment strategy")
    testing_phases: List[str] = Field(description="Testing phases")
    rollback_plan: Dict[str, Any] = Field(description="Rollback plan")
    success_criteria: List[str] = Field(description="Success criteria for each step")

class RiskAssessment(BaseModel):
    """Output schema for risk assessment."""
    overall_risk_level: str = Field(description="Overall risk level (low, medium, high, critical)")
    risk_factors: List[Dict[str, Any]] = Field(description="Identified risk factors")
    mitigation_strategies: List[str] = Field(description="Risk mitigation strategies")
    contingency_plans: List[str] = Field(description="Contingency plans")
    dependencies_risks: List[str] = Field(description="Risks related to dependencies")

# Global variables for LLM and parsers
_model = None
_execution_parser = None
_risk_parser = None
_execution_prompt = None
_risk_prompt = None

def _initialize_execution_tools(config: Config):
    """Initialize LLM and parsers for execution tools."""
    global _model, _execution_parser, _risk_parser, _execution_prompt, _risk_prompt
    
    if _model is None:
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        _execution_parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        _risk_parser = JsonOutputParser(pydantic_object=RiskAssessment)
        
        _execution_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert AWS Infrastructure Execution Planner. Your role is to create comprehensive execution plans for infrastructure deployment based on requirements and dependencies.

[ROLE]
- Create step-by-step execution plans
- Define deployment strategies
- Estimate execution time and resource requirements
- Identify critical path steps
- Plan testing phases and rollback strategies
- Define success criteria for each step

[OUTPUT]
Provide a comprehensive execution plan including:
1. Steps: Detailed step-by-step execution plan
2. Time Estimation: Total estimated execution time
3. Critical Path: Steps that are critical for success
4. Resource Requirements: What resources are needed
5. Deployment Strategy: How to deploy (blue-green, rolling, etc.)
6. Testing Phases: Testing strategy for each phase
7. Rollback Plan: How to rollback if issues occur
8. Success Criteria: Criteria for successful completion

[RESPONSE FORMAT]
Return a JSON object with the following structure:
{
    "steps": [
        {
            "step_number": 1,
            "description": "Step description",
            "estimated_time": "30 minutes",
            "dependencies": ["step1", "step2"],
            "resources": ["resource1", "resource2"],
            "success_criteria": ["criteria1", "criteria2"]
        }
    ],
    "total_estimated_time": "2 hours",
    "critical_path": [1, 3, 5],
    "resource_requirements": {
        "compute": "t3.medium",
        "storage": "100GB",
        "network": "VPC with subnets"
    },
    "deployment_strategy": "Blue-green deployment",
    "testing_phases": ["Unit testing", "Integration testing", "End-to-end testing"],
    "rollback_plan": {
        "trigger_conditions": ["condition1", "condition2"],
        "rollback_steps": ["step1", "step2"]
    },
    "success_criteria": ["criteria1", "criteria2"]
}
            """),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        _risk_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert Risk Assessment Specialist for AWS Infrastructure. Your role is to assess risks associated with infrastructure deployment plans.

[ROLE]
- Assess overall risk level
- Identify specific risk factors
- Propose mitigation strategies
- Create contingency plans
- Evaluate dependency risks

[OUTPUT]
Provide a comprehensive risk assessment including:
1. Overall Risk Level: Low, Medium, High, or Critical
2. Risk Factors: Specific risks identified
3. Mitigation Strategies: How to reduce risks
4. Contingency Plans: Backup plans if risks materialize
5. Dependency Risks: Risks related to service dependencies

[RESPONSE FORMAT]
Return a JSON object with the following structure:
{
    "overall_risk_level": "medium",
    "risk_factors": [
        {
            "risk": "Service dependency failure",
            "probability": "medium",
            "impact": "high",
            "description": "Risk description"
        }
    ],
    "mitigation_strategies": ["strategy1", "strategy2"],
    "contingency_plans": ["plan1", "plan2"],
    "dependencies_risks": ["risk1", "risk2"]
}
            """),
            MessagesPlaceholder(variable_name="messages")
        ])

@tool
async def create_execution_plan_tool(requirements_analysis: str, dependency_mapping: str) -> str:
    """
    Create execution plan based on requirements and dependencies.
    
    Args:
        requirements_analysis: JSON string of requirements analysis
        dependency_mapping: JSON string of dependency mapping
        
    Returns:
        JSON string containing execution plan
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting async execution plan creation",
            extra={
                "requirements_analysis_length": len(requirements_analysis),
                "dependency_mapping_length": len(dependency_mapping)
            }
        )
        
        # Prepare context for LLM
        context = f"Requirements Analysis: {requirements_analysis}\nDependency Mapping: {dependency_mapping}\n"
        
        # Prepare messages for LLM
        messages = [{"role": "user", "content": context}]
        
        # Get execution plan from LLM - now using await
        response = await _model.ainvoke(
            _execution_prompt.format_messages(messages=messages)
        )
        
        execution_logger.log_structured(
            level="DEBUG",
            message="Execution plan LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Parse the response
        try:
            execution_data = _execution_parser.parse(response.content)
            result = json.dumps(execution_data.dict(), indent=2)
            
            execution_logger.log_structured(
                level="INFO",
                message="Execution plan created successfully",
                extra={
                    "steps_count": len(execution_data.get("steps", [])),
                    "total_estimated_time": execution_data.get("total_estimated_time"),
                    "deployment_strategy": execution_data.get("deployment_strategy")
                }
            )
            
        except Exception as parse_error:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse execution plan response, using fallback",
                extra={"error": str(parse_error), "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            # Fallback to basic execution plan
            fallback_data = {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Validate requirements and dependencies",
                        "estimated_time": "15 minutes",
                        "dependencies": [],
                        "resources": ["terraform"],
                        "success_criteria": ["All requirements validated"]
                    },
                    {
                        "step_number": 2,
                        "description": "Deploy infrastructure",
                        "estimated_time": "45 minutes",
                        "dependencies": ["step1"],
                        "resources": ["aws_account", "terraform"],
                        "success_criteria": ["Infrastructure deployed successfully"]
                    }
                ],
                "total_estimated_time": "1 hour",
                "critical_path": [1, 2],
                "resource_requirements": {
                    "compute": "terraform",
                    "storage": "aws_resources",
                    "network": "aws_vpc"
                },
                "deployment_strategy": "Direct deployment",
                "testing_phases": ["Validation testing"],
                "rollback_plan": {
                    "trigger_conditions": ["Deployment failure"],
                    "rollback_steps": ["Destroy infrastructure"]
                },
                "success_criteria": ["Infrastructure operational"]
            }
            result = json.dumps(fallback_data, indent=2)
        
        return result
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Async execution plan creation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Execution plan creation failed: {str(e)}"})

@tool
async def assess_risks_tool(execution_plan: str, requirements_analysis: str) -> str:
    """
    Assess risks associated with the execution plan.
    
    Args:
        execution_plan: JSON string of execution plan
        requirements_analysis: JSON string of requirements analysis
        
    Returns:
        JSON string containing risk assessment
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting async risk assessment",
            extra={
                "execution_plan_length": len(execution_plan),
                "requirements_analysis_length": len(requirements_analysis)
            }
        )
        
        # Prepare context for LLM
        context = f"Execution Plan: {execution_plan}\nRequirements Analysis: {requirements_analysis}\n"
        
        # Prepare messages for LLM
        messages = [{"role": "user", "content": context}]
        
        # Get risk assessment from LLM - now using await
        response = await _model.ainvoke(
            _risk_prompt.format_messages(messages=messages)
        )
        
        execution_logger.log_structured(
            level="DEBUG",
            message="Risk assessment LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Parse the response
        try:
            risk_data = _risk_parser.parse(response.content)
            result = json.dumps(risk_data.dict(), indent=2)
            
            execution_logger.log_structured(
                level="INFO",
                message="Risk assessment completed successfully",
                extra={
                    "overall_risk_level": risk_data.get("overall_risk_level"),
                    "risk_factors_count": len(risk_data.get("risk_factors", []))
                }
            )
            
        except Exception as parse_error:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse risk assessment response, using fallback",
                extra={"error": str(parse_error), "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            # Fallback to basic risk assessment
            fallback_data = {
                "overall_risk_level": "medium",
                "risk_factors": [
                    {
                        "risk": "Infrastructure deployment failure",
                        "probability": "medium",
                        "impact": "high",
                        "description": "Risk of deployment failure"
                    }
                ],
                "mitigation_strategies": ["Test in staging environment", "Have rollback plan ready"],
                "contingency_plans": ["Rollback to previous state", "Manual intervention"],
                "dependencies_risks": ["Service availability", "Configuration errors"]
            }
            result = json.dumps(fallback_data, indent=2)
        
        return result
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Async risk assessment failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Risk assessment failed: {str(e)}"})

@tool
async def calculate_complexity_score_tool(execution_plan: str, risk_assessment: str, dependency_mapping: str) -> str:
    """
    Calculate complexity score based on execution plan, risks, and dependencies.
    
    Args:
        execution_plan: JSON string of execution plan
        risk_assessment: JSON string of risk assessment
        dependency_mapping: JSON string of dependency mapping
        
    Returns:
        Complexity score (1-10) with explanation
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting async complexity score calculation",
            extra={
                "execution_plan_length": len(execution_plan),
                "risk_assessment_length": len(risk_assessment),
                "dependency_mapping_length": len(dependency_mapping)
            }
        )
        
        # Parse inputs
        try:
            plan_data = json.loads(execution_plan)
            risk_data = json.loads(risk_assessment)
            dependency_data = json.loads(dependency_mapping)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
        
        # Calculate complexity score
        score = 1  # Base score
        
        # Add points for number of execution steps
        if "steps" in plan_data:
            score += min(len(plan_data["steps"]), 3)
        
        # Add points for risk level
        risk_level = risk_data.get("overall_risk_level", "low")
        risk_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        score += risk_scores.get(risk_level, 1)
        
        # Add points for dependencies
        if "mandatory_dependencies" in dependency_data:
            score += min(len(dependency_data["mandatory_dependencies"]), 2)
        
        # Cap at 10
        score = min(score, 10)
        
        complexity_result = {
            "complexity_score": score,
            "explanation": f"Score {score}/10 based on {len(plan_data.get('steps', []))} steps, {risk_level} risk level, and {len(dependency_data.get('mandatory_dependencies', []))} dependencies",
            "factors": {
                "execution_steps": len(plan_data.get("steps", [])),
                "risk_level": risk_level,
                "dependencies_count": len(dependency_data.get("mandatory_dependencies", []))
            }
        }
        
        execution_logger.log_structured(
            level="INFO",
            message="Complexity score calculated successfully",
            extra={"complexity_score": score}
        )
        
        return json.dumps(complexity_result, indent=2)
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Async complexity score calculation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Complexity score calculation failed: {str(e)}"})

@tool
async def validate_execution_plan_tool(execution_plan: str, risk_assessment: str, complexity_score: str) -> str:
    """
    Validate the execution plan for completeness and feasibility.
    
    Args:
        execution_plan: JSON string of execution plan
        risk_assessment: JSON string of risk assessment
        complexity_score: Complexity score and explanation
        
    Returns:
        Validation result with any issues or confirmations
    """
    try:
        if _model is None:
            raise ValueError("Execution tools not initialized. Call _initialize_execution_tools first.")
        
        execution_logger.log_structured(
            level="INFO",
            message="Starting async execution plan validation",
            extra={
                "execution_plan_length": len(execution_plan),
                "risk_assessment_length": len(risk_assessment),
                "complexity_score_length": len(complexity_score)
            }
        )
        
        # Create validation prompt
        validation_prompt = f"""
Validate the following execution plan for completeness and feasibility:

Execution Plan: {execution_plan}
Risk Assessment: {risk_assessment}
Complexity Score: {complexity_score}

Check for:
1. Missing critical steps
2. Unrealistic time estimates
3. Missing dependencies between steps
4. Inadequate resource requirements
5. Insufficient testing phases
6. Missing rollback procedures
7. Unclear success criteria

Provide validation results in JSON format:
{{
    "is_valid": true/false,
    "issues": ["issue1", "issue2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "feasibility_score": 1-10,
    "completeness_score": 1-10
}}
"""
        
        # Get validation from LLM - now using await
        response = await _model.ainvoke([{"role": "user", "content": validation_prompt}])
        
        execution_logger.log_structured(
            level="DEBUG",
            message="Execution plan validation LLM response received",
            extra={
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_length": len(response.content) if hasattr(response, 'content') else 0
            }
        )
        
        # Try to parse as JSON, fallback to text if needed
        try:
            validation_result = json.loads(response.content)
            
            execution_logger.log_structured(
                level="INFO",
                message="Execution plan validation completed successfully",
                extra={
                    "is_valid": validation_result.get("is_valid"),
                    "feasibility_score": validation_result.get("feasibility_score"),
                    "completeness_score": validation_result.get("completeness_score"),
                    "issues_count": len(validation_result.get("issues", []))
                }
            )
            
        except json.JSONDecodeError:
            execution_logger.log_structured(
                level="WARNING",
                message="Failed to parse validation response as JSON, using fallback",
                extra={"response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content}
            )
            validation_result = {
                "is_valid": True,
                "issues": [],
                "recommendations": ["Manual review recommended"],
                "feasibility_score": 7,
                "completeness_score": 7,
                "raw_response": response.content
            }
        
        return json.dumps(validation_result, indent=2)
        
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message=f"Async execution plan validation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Execution plan validation failed: {str(e)}"})

def create_execution_planner_react_agent(config: Config):
    """
    Create a React agent for execution planning.
    
    Args:
        config: Configuration instance
        
    Returns:
        React agent for execution planning
    """
    try:
        execution_logger.log_structured(
            level="INFO",
            message="=== CREATING EXECUTION PLANNER REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        # Initialize tools
        execution_logger.log_structured(
            level="DEBUG",
            message="Initializing execution tools",
            extra={}
        )
        
        _initialize_execution_tools(config)
        
        # Get LLM from config
        llm_config = config.get_llm_config()
        
        execution_logger.log_structured(
            level="DEBUG",
            message="Creating LLM for execution planner",
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
        execution_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 4,
                "tool_names": ["create_execution_plan_tool", "assess_risks_tool", "calculate_complexity_score_tool", "validate_execution_plan_tool"]
            }
        )
    
        # Create React agent
        execution_planner = create_react_agent(
            model=llm,
            tools=[
                create_execution_plan_tool,
                assess_risks_tool,
                calculate_complexity_score_tool,
                validate_execution_plan_tool
            ],
            name="execution_planner",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert AWS Execution Planner. Your role is to create comprehensive execution plans for infrastructure deployment.

[ROLE]
- Create step-by-step execution plans
- Assess risks and calculate complexity scores
- Define deployment strategies and rollback plans
- Validate execution plans for feasibility

[WORKFLOW]
1. Use create_execution_plan_tool to create detailed execution plan
2. Use assess_risks_tool to evaluate risks and mitigation strategies
3. Use calculate_complexity_score_tool to determine complexity
4. Use validate_execution_plan_tool to validate the complete plan

[OUTPUT]
Provide a clear, structured response that includes:
- Step-by-step execution plan with time estimates
- Risk assessment and mitigation strategies
- Complexity score and explanation
- Deployment strategy and rollback plan
- Success criteria and testing phases
                """),
                ("user", "{input}")
            ])
        )
        
        execution_logger.log_structured(
            level="INFO",
            message="=== EXECUTION PLANNER REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(execution_planner).__name__,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "tools_count": 4
            }
        )
        
        return execution_planner
    
    except Exception as e:
        execution_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE EXECUTION PLANNER REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
