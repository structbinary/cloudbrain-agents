"""
Requirements Analyzer React Agent for Planner Sub-Supervisor.

This module implements the Requirements Analyzer as a React agent with tools:
- analyze_requirements_tool: Analyzes user requests for infrastructure requirements
- validate_requirements_tool: Validates extracted requirements for completeness
"""

import json
from typing import Dict, Any, List, Literal
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator, field_validator
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from enum import Enum
from .requirement_analyser_prompts import AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT, AWS_SERVICE_DISCOVERY_HUMAN_PROMPT

# Create logger
requirements_logger = AgentLogger("REQUIREMENTS_ANALYZER_REACT")

# Global variables for LLM and parsers
_model = None
_requirements_parser_prompt = None
_infra_requirements_parser = None
_service_discovery_parser = None
_service_discovery_system_prompt = None
_service_discovery_human_prompt = None
_service_discovery_prompt = None

# Define the output schema for structured extraction
class InfrastructureRequirements(BaseModel):
    """Structured representation of AWS infrastructure requirements"""
    primary_services: List[str] = Field(description="Main AWS services explicitly mentioned")
    secondary_services: List[str] = Field(description="Supporting/dependent AWS services identified")
    scope_classification: str = Field(description="One of: 'single_service', 'multi_service', 'full_application_stack'")
    business_requirements: Dict[str, str] = Field(description="Business needs mapped to technical specifications")
    technical_specifications: Dict[str, Any] = Field(description="Technical details and constraints")
    deployment_context: str = Field(description="Context like development, production, compliance requirements")

# Enhanced AWS service category enumeration based on official AWS documentation
class AWSServiceCategory(str, Enum):
    COMPUTE = "Compute"
    CONTAINERS = "Containers" 
    STORAGE = "Storage"
    DATABASE = "Database"
    NETWORKING = "Networking and Content Delivery"
    SECURITY_IDENTITY = "Security, Identity, and Compliance"
    MANAGEMENT_GOVERNANCE = "Management and Governance"
    ANALYTICS = "Analytics"
    MACHINE_LEARNING = "Machine Learning"
    APPLICATION_INTEGRATION = "Application Integration"
    DEVELOPER_TOOLS = "Developer Tools"
    MIGRATION_TRANSFER = "Migration and Transfer"
    MEDIA_SERVICES = "Media Services"
    IOT = "Internet of Things"
    GAME_TECH = "Game Tech"
    BLOCKCHAIN = "Blockchain"
    SERVERLESS = "Serverless"
    EDGE_COMPUTING = "Edge and Hybrid"


# Dependency relationship types based on AWS Config service relationships
class DependencyType(str, Enum):
    REQUIRED = "required"          # Critical dependency - cannot function without
    RECOMMENDED = "recommended"    # Best practice dependency
    OPTIONAL = "optional"         # Enhancement dependency
    CONDITIONAL = "conditional"   # Depends on specific configuration
    IMPLICIT = "implicit"         # Automatically created/managed


class DependencyNature(str, Enum):
    CREATION_ORDER = "creation_order"      # Must be created before
    ACCESS_CONTROL = "access_control"      # Provides permissions/access
    ENCRYPTION = "encryption"              # Provides encryption services
    MONITORING = "monitoring"              # Provides observability
    NETWORKING = "networking"              # Network connectivity/security
    DATA_FLOW = "data_flow"               # Data processing/storage
    CONFIGURATION = "configuration"        # Configuration management


# Enhanced relationship types for multi-service intelligence
class ServiceRelationshipType(str, Enum):
    ENABLES = "enables"                    # VPC enables EKS deployment
    REQUIRES = "requires"                  # EKS requires VPC
    ENHANCES = "enhances"                  # CloudWatch enhances EKS observability
    INTEGRATES_WITH = "integrates_with"    # EKS integrates with ECR
    DEPENDS_ON = "depends_on"             # Node groups depend on EC2
    PROVIDES_FOR = "provides_for"         # IAM provides access control for EKS


class DependencyLayer(str, Enum):
    FOUNDATION = "foundation"         # VPC, IAM - foundational services
    CORE = "core"                    # EKS cluster - core application service
    INTEGRATION = "integration"      # ECR, ELB - integration services
    OPERATIONAL = "operational"      # CloudWatch, Systems Manager - ops services
    SECURITY = "security"           # KMS, GuardDuty - security services


# Enhanced service discovery models with validation
class TerraformResource(BaseModel):
    """Terraform resource specification with validation"""
    resource_type: str = Field(..., description="Exact Terraform resource type (e.g., aws_s3_bucket)")
    purpose: str = Field(..., description="Purpose of this resource in the infrastructure")
    required: bool = Field(..., description="Whether this resource is required for basic functionality")
    depends_on: List[str] = Field(default_factory=list, description="List of resource types this depends on")
    configuration_priority: int = Field(default=1, description="Configuration priority (1=highest, 5=lowest)")
    
    @field_validator('resource_type')
    @classmethod
    def validate_terraform_resource_type(cls, v):
        if not v.startswith('aws_'):
            raise ValueError('Resource type must start with "aws_"')
        return v


class ServiceDependency(BaseModel):
    """Enhanced service dependency specification"""
    service_name: str = Field(..., description="Name of the dependent service")
    aws_service_type: str = Field(..., description="AWS service identifier")
    category: AWSServiceCategory = Field(..., description="AWS service category")
    terraform_resources: List[str] = Field(..., description="List of Terraform resource types")
    dependency_reason: str = Field(..., description="Detailed explanation of why this dependency exists")
    dependency_type: DependencyType = Field(..., description="Type of dependency relationship")
    dependency_nature: DependencyNature = Field(..., description="Nature of the dependency")
    well_architected_pillar: List[str] = Field(..., description="Which AWS Well-Architected pillars this addresses")
    configuration_details: Dict[str, Any] = Field(default_factory=dict, description="Specific configuration requirements")


class SecurityRequirement(BaseModel):
    """Security-specific dependencies and requirements"""
    service: str = Field(..., description="Security service name")
    terraform_resources: List[str] = Field(..., description="Required Terraform security resources")
    purpose: str = Field(..., description="Security purpose (encryption, access control, etc.)")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Supported compliance frameworks")
    well_architected_controls: List[str] = Field(..., description="Security controls addressed")


class NetworkingDependency(BaseModel):
    """Network-specific dependencies"""
    component: str = Field(..., description="Networking component")
    terraform_resources: List[str] = Field(..., description="Required networking resources")
    purpose: str = Field(..., description="Networking purpose")
    network_tier: Literal["public", "private", "isolated"] = Field(..., description="Network tier placement")


class ServiceRelationship(BaseModel):
    """Service relationship mapping with enhanced details"""
    source_service: str = Field(..., description="Source service in the relationship")
    target_services: List[str] = Field(..., description="Target services this depends on")
    relationship_type: DependencyNature = Field(..., description="Nature of the relationship")
    creation_order_priority: int = Field(..., description="Order priority for resource creation")


# Enhanced service specification with relationships for multi-service intelligence
class ServiceSpecification(BaseModel):
    """Enhanced service specification with relationship mapping"""
    service_name: str = Field(..., description="AWS service name")
    aws_service_type: str = Field(..., description="AWS service identifier")
    category: AWSServiceCategory = Field(..., description="AWS service category")
    dependency_layer: DependencyLayer = Field(..., description="Dependency layer classification")
    terraform_resources: List[str] = Field(..., description="Complete Terraform resource list")
    relationship_to_primary: ServiceRelationshipType = Field(..., description="How this service relates to primary services")
    enables_services: List[str] = Field(default_factory=list, description="Services this enables")
    requires_services: List[str] = Field(default_factory=list, description="Services this requires")
    configuration_priority: int = Field(..., description="Configuration order (1=first, 5=last)")
    production_criticality: Literal["critical", "recommended", "optional"] = Field(..., description="Production deployment criticality")
    well_architected_pillars: List[str] = Field(default_factory=list, description="AWS Well-Architected pillars addressed")


class DeploymentPhase(BaseModel):
    """Deployment sequence phase with detailed rationale"""
    sequence: int = Field(..., description="Deployment sequence number")
    services: List[str] = Field(..., description="Services to be deployed in this phase")
    rationale: str = Field(..., description="Why these services are deployed together in this sequence")
    layer: DependencyLayer = Field(..., description="Dependency layer for this phase")
    estimated_duration: str = Field(default="5-10 minutes", description="Estimated deployment time")
    rollback_strategy: str = Field(default="Terraform destroy in reverse order", description="Rollback approach if deployment fails")


# Main output schema with comprehensive validation and multi-service intelligence
class AWSServiceMapping(BaseModel):
    """Production-grade AWS service discovery and dependency mapping with multi-service intelligence"""
    
    # Enhanced primary services with full relationship mapping
    primary_services: List[ServiceSpecification] = Field(
        ..., description="Primary services with complete relationship mapping and dependencies"
    )
    
    # Foundation services (those that enable primary services)
    foundation_services: List[ServiceSpecification] = Field(
        ..., description="Foundation services that enable primary services (VPC, IAM, KMS)"
    )
    
    # Integration services (those that enhance primary services)
    integration_services: List[ServiceSpecification] = Field(
        ..., description="Services that integrate with or enhance primary services"
    )
    
    # Operational services (monitoring, management, etc.)
    operational_services: List[ServiceSpecification] = Field(
        ..., description="Operational services for monitoring, logging, management"
    )
    
    # Security services (additional security layers)
    security_services: List[ServiceSpecification] = Field(
        ..., description="Security services beyond foundation security"
    )
    
    # Legacy field maintained for backward compatibility
    implicit_dependencies: List[ServiceDependency] = Field(
        ..., description="Comprehensive implicit/supporting services with detailed analysis"
    )
    
    # Terraform resource mapping
    terraform_resources: List[TerraformResource] = Field(
        ..., description="Complete Terraform resource specification with dependencies"
    )
    
    # Enhanced service relationship matrix
    service_relationships: List[ServiceRelationship] = Field(
        ..., description="Detailed service dependency relationships with creation order"
    )
    
    # Service relationship matrix for multi-service intelligence
    service_relationship_matrix: Dict[str, Dict[str, ServiceRelationshipType]] = Field(
        ..., description="Matrix showing how each service relates to others (enables, requires, integrates_with)"
    )
    
    # Deployment sequence with rationale
    deployment_sequence: List[DeploymentPhase] = Field(
        ..., description="Ordered deployment sequence with dependencies and rationale explained"
    )
    
    # Multi-service architecture patterns
    architecture_patterns: Dict[str, List[str]] = Field(
        ..., description="Architecture patterns identified (e.g., 'container_platform', 'network_foundation')"
    )
    
    # Categorization
    category_mapping: Dict[str, AWSServiceCategory] = Field(
        ..., description="AWS service category classification for each service"
    )
    
    # Security analysis
    security_dependencies: List[SecurityRequirement] = Field(
        ..., description="Security-related services with compliance mapping"
    )
    
    # Monitoring and observability
    monitoring_dependencies: List[Dict[str, Any]] = Field(
        ..., description="Monitoring, logging, and observability services"
    )
    
    # Network architecture
    networking_dependencies: List[NetworkingDependency] = Field(
        ..., description="Network architecture requirements and dependencies"
    )
    
    # Well-Architected Framework alignment
    well_architected_alignment: Dict[str, List[str]] = Field(
        ..., description="Mapping to AWS Well-Architected Framework pillars"
    )
    
    # Cost optimization insights
    cost_optimization_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Cost optimization recommendations"
    )
    
    @model_validator(mode='after')
    def validate_service_mapping_consistency(self):
        """Ensure consistency across all mapping components"""
        # Validate primary services
        all_services = []
        all_services.extend([svc.aws_service_type for svc in self.primary_services])
        all_services.extend([svc.aws_service_type for svc in self.foundation_services])
        all_services.extend([svc.aws_service_type for svc in self.integration_services])
        all_services.extend([svc.aws_service_type for svc in self.operational_services])
        all_services.extend([svc.aws_service_type for svc in self.security_services])
        
        # Validate that all services have category mappings
        category_mapping = self.category_mapping
        for service_type in all_services:
            if service_type not in category_mapping:
                raise ValueError(f"Missing category mapping for service: {service_type}")
        
        # Validate relationship matrix consistency
        relationship_matrix = self.service_relationship_matrix
        for source_service, relationships in relationship_matrix.items():
            if source_service not in all_services:
                raise ValueError(f"Service in relationship matrix not found in service lists: {source_service}")
            for target_service in relationships.keys():
                if target_service not in all_services:
                    raise ValueError(f"Target service in relationship matrix not found in service lists: {target_service}")
        
        # Validate deployment sequence references valid services
        for phase in self.deployment_sequence:
            for service in phase.services:
                if service not in all_services:
                    raise ValueError(f"Service in deployment sequence not found in service lists: {service}")
        
        return self
    
    @model_validator(mode='after') 
    def validate_no_empty_service_arrays(self):
        """Ensure no service has empty terraform_resources arrays"""
        all_service_lists = [
            self.primary_services,
            self.foundation_services, 
            self.integration_services,
            self.operational_services,
            self.security_services
        ]
        
        for service_list in all_service_lists:
            for service in service_list:
                if not service.terraform_resources:
                    raise ValueError(f"Service {service.service_name} has empty terraform_resources array")
        
        return self


def _initialize_requirements_tools(config: Config):
    """Initialize LLM and parsers for requirements tools."""
    global _model, _infra_requirements_parser, _requirements_parser_prompt, _service_discovery_parser, _service_discovery_prompt, _service_discovery_system_prompt, _service_discovery_human_prompt
    
    if _model is None:
        llm_config = config.get_llm_config()
        _model = LLMProvider.create_llm(
            provider=llm_config['provider'],
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_tokens']
        )
        
        _infra_requirements_parser = JsonOutputParser(pydantic_object=InfrastructureRequirements)
        _service_discovery_parser = JsonOutputParser(pydantic_object=AWSServiceMapping)
        _service_discovery_system_prompt = AWS_SERVICE_DISCOVERY_SYSTEM_PROMPT
        _service_discovery_human_prompt = AWS_SERVICE_DISCOVERY_HUMAN_PROMPT
        _service_discovery_prompt = ChatPromptTemplate.from_messages([
            ("system", _service_discovery_system_prompt),
            ("human", _service_discovery_human_prompt)
        ])
        
    # ChatPromptTemplate for Requirements Parser Tool
        _requirements_parser_prompt = ChatPromptTemplate.from_messages([("system", """
You are an expert AWS Infrastructure Requirements Analyst specialized in parsing natural language queries into structured technical specifications for Terraform module development.

Your primary responsibilities:
1. Extract infrastructure requirements from user queries with precision
2. Identify primary and secondary AWS services mentioned or implied
3. Determine the scope and complexity of the infrastructure request
4. Map business requirements to specific technical specifications
5. Classify the deployment context and constraints

ANALYSIS FRAMEWORK:

**Step 1: Service Identification**
- PRIMARY SERVICES: Explicitly mentioned AWS services in the user query
- SECONDARY SERVICES: Dependent/supporting services required (IAM, KMS, CloudWatch, VPC components, etc.)
- Use your knowledge of AWS service dependencies and best practices

**Step 2: Scope Classification**
- SINGLE_SERVICE: One main AWS service (e.g., "S3 bucket module")  
- MULTI_SERVICE: Multiple related services (e.g., "web application with RDS and ALB")
- FULL_APPLICATION_STACK: Complete application infrastructure (e.g., "3-tier web application")

**Step 3: Requirements Mapping**
- BUSINESS REQUIREMENTS: What the user wants to achieve (storage, compute, networking, etc.)
- TECHNICAL SPECIFICATIONS: How it should be implemented (encryption, scaling, networking, etc.)
- DEPLOYMENT CONTEXT: Environment type, compliance needs, security requirements

**Step 4: Inference and Best Practices**
- Identify implied services based on AWS best practices
- Consider security, monitoring, and compliance requirements
- Think about resource dependencies and deployment order

**Output Format:**
Provide your analysis in the structured JSON format specified by the schema.
Be thorough but concise. If information is not explicitly provided, use AWS best practices to make reasonable inferences.

**CRITICAL: Return ONLY the JSON object without any markdown formatting, code blocks, or additional text.**
- DO NOT wrap the response in ```json or ``` blocks
- DO NOT add any explanatory text before or after the JSON
- Return ONLY the raw JSON object that matches the schema
"""),
    
    ("human", """
Analyze the following infrastructure request and extract structured requirements:

USER QUERY: {user_query}

Please provide a comprehensive analysis following the framework above. Consider:
- What AWS services are explicitly mentioned?
- What supporting services would be needed?
- What's the scope and complexity?
- What business goals are implied?
- What technical specifications can be inferred?
- What deployment context clues are present?

Your final output MUST be a JSON object matching this Pydantic model: `InfrastructureRequirements`:

class InfrastructureRequirements(BaseModel):
    primary_services: List[str] = Field(description="Main AWS services explicitly mentioned")
    secondary_services: List[str] = Field(description="Supporting/dependent AWS services identified")
    scope_classification: str = Field(description="One of: 'single_service', 'multi_service', 'full_application_stack'")
    business_requirements: Dict[str, str] = Field(description="Business needs mapped to technical specifications")
    technical_specifications: Dict[str, Any] = Field(description="Technical details and constraints")
    deployment_context: str = Field(description="Context like development, production, compliance requirements")

""")
])

@tool
async def infra_requirements_parser_tool(user_query: str) -> InfrastructureRequirements:
    """
    Extracts infrastructure requirements from natural language queries.
    
    This tool:
    - Identifies primary and secondary AWS services mentioned
    - Determines scope (single service, multi-service, full application stack)  
    - Maps business requirements to technical specifications
    - Provides structured output for downstream processing
    
    Args:
        user_query: Natural language description of infrastructure needs
        
    Returns:
        InfrastructureRequirements: Structured analysis of the request
    """
    try:
        if _model is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
        requirements_logger.log_structured(
            level="INFO",
            message="Starting async Infra requirements parser tool",
            extra={"user_request": user_query[:100] + "..." if len(user_query) > 100 else user_query}
        )
        formatted_prompt = _requirements_parser_prompt.format(user_query=user_query)
        llm_response = await _model.ainvoke(formatted_prompt)
        if isinstance(llm_response, AIMessage):
            response = llm_response.content
        else:
            response = llm_response
            
        requirements_logger.log_structured(
            level="DEBUG",
            message="Infra requirements parser tool response received",
            extra={
                "response_type": type(llm_response).__name__,
                "has_content": hasattr(llm_response, 'content'),
                "content_length": len(response) if response else 0
            }
        )
        
        # Clean the response content to remove markdown code blocks if present
        content = response.strip()
        
        parsed_response = _infra_requirements_parser.parse(content)
        return parsed_response

    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Failed to parse Infra requirements: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Infra requirements parser tool failed: {str(e)}"})

@tool
async def aws_service_discovery_tool(requirements_analysis: str) -> AWSServiceMapping:
    """
    Production-grade AWS service discovery tool with comprehensive dependency mapping.
    
    This tool provides:
    - Complete AWS service ecosystem mapping with all dependencies
    - Production-ready Terraform resource specifications with proper ordering
    - Security-first approach with comprehensive security dependencies
    - Well-Architected Framework alignment and compliance mapping
    - Cost optimization recommendations and best practices integration
    - Network architecture planning with proper segmentation
    - Comprehensive monitoring and observability stack recommendations
    
    Args:
        requirements_analysis: Structured requirements from the Requirements Parser Tool
        
    Returns:
        AWSServiceMapping: Comprehensive service discovery with validation and dependencies
        
    Raises:
        ValidationError: If output doesn't meet production quality standards
    """
    try:
        if _model is None:
            raise ValueError("Requirements tools not initialized. Call _initialize_requirements_tools first.")
        
        requirements_logger.log_structured(
            level="INFO",
            message="Starting async AWS service discovery",
            extra={"requirements_analysis_length": len(requirements_analysis)}
        )
        
        # Debug the input and prompt template
        requirements_logger.log_structured(
            level="DEBUG",
            message="Debugging prompt formatting",
            extra={
                "requirements_analysis_type": type(requirements_analysis).__name__,
                "requirements_analysis_preview": requirements_analysis[:200] + "..." if len(requirements_analysis) > 200 else requirements_analysis,
                "prompt_template_type": type(_service_discovery_prompt).__name__
            }
        )
        
        # Use a different approach to avoid format() issues with JSON
        try:
            # Try direct format first
            formatted_prompt = _service_discovery_prompt.format(requirements_input=requirements_analysis)
        except (KeyError, ValueError) as format_error:
            # If format fails, manually construct the prompt
            requirements_logger.log_structured(
                level="WARNING",
                message="Format method failed, using manual prompt construction",
                extra={"format_error": str(format_error)}
            )
            
            # Manually construct the prompt by replacing the placeholder
            system_prompt = _service_discovery_system_prompt
            human_prompt = _service_discovery_human_prompt.replace("{requirements_input}", requirements_analysis)
            
            # Create messages directly instead of using ChatPromptTemplate
            formatted_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        
        # Format the prompt before invoking the LLM
        if 'formatted_messages' not in locals():
            # This means we used the try/except path and already have formatted_messages
            formatted_messages = formatted_prompt
        llm_response = await _model.ainvoke(formatted_messages)
        
        if isinstance(llm_response, AIMessage):
            response = llm_response.content
        else:
            response = llm_response
            
        # Log the raw response for debugging
        requirements_logger.log_structured(
            level="DEBUG",
            message="Raw LLM response received",
            extra={
                "response_type": type(response).__name__,
                "response_length": len(response) if response else 0,
                "response_preview": response[:200] + "..." if response and len(response) > 200 else response
            }
        )
        
        # Clean the response content to remove markdown code blocks if present
        content = response.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.endswith('```'):
            content = content[:-3]  # Remove ```
        content = content.strip()
        
        # Parse the response using the service discovery parser
        parsed_response = _service_discovery_parser.parse(content)
        
        # CRITICAL: Set completion flag after successful parsing
        # Note: In React agent context, we can't directly modify state here
        # The completion flag will be set by the supervisor when this tool completes successfully
        
        requirements_logger.log_structured(
            level="INFO",
            message="AWS service discovery completed successfully",
            extra={
                "completion_ready": True,
                "services_discovered": len(parsed_response.get('primary_services', [])),
                "workflow_phase": "requirements_analysis"
            }
        )
        
        return parsed_response
        
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message=f"Async requirements validation failed: {e}",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return json.dumps({"error": f"Requirements validation failed: {str(e)}"})

def create_requirements_analyzer_react_agent(config: Config):
    """
    Create a React agent for requirements analysis.
    
    Args:
        config: Configuration instance
        
    Returns:
        React agent for requirements analysis
    """
    try:
        requirements_logger.log_structured(
            level="INFO",
            message="=== CREATING REQUIREMENTS ANALYZER REACT AGENT ===",
            extra={"config_type": type(config).__name__}
        )
        
        # Initialize tools
        requirements_logger.log_structured(
            level="DEBUG",
            message="Initializing requirements tools",
            extra={}
        )
        
        _initialize_requirements_tools(config)
        
        # Get LLM from config
        llm_config = config.get_llm_config()
        
        requirements_logger.log_structured(
            level="DEBUG",
            message="Creating LLM for requirements analyzer",
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
        requirements_logger.log_structured(
            level="DEBUG",
            message="Creating React agent with async tools",
            extra={
                "tools_count": 2,
                "tool_names": ["analyze_requirements_tool", "validate_requirements_tool"]
            }
        )
        
        requirements_analyzer = create_react_agent(
            model=llm,
            tools=[infra_requirements_parser_tool, aws_service_discovery_tool],
            name="requirements_analyzer",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are an expert AWS Infrastructure Requirements Analyst. Your role is to analyze user requests and extract comprehensive infrastructure requirements.

[ROLE]
- Analyze user requests for AWS infrastructure needs
- Extract business and technical requirements
- Identify constraints, assumptions, and risk factors
- Define scalability and performance requirements

[COMPLETION REQUIREMENTS]
- ALWAYS use both tools: infra_requirements_parser_tool THEN aws_service_discovery_tool
- After aws_service_discovery_tool completes successfully, provide final summary
- DO NOT continue if either tool fails - report the failure clearly
- Provide clear completion status in your final response

[WORKFLOW - CRITICAL]
1. Use infra_requirements_parser_tool with the user's request
2. Use aws_service_discovery_tool with the results from step 1
3. Verify both tools completed successfully
4. Provide comprehensive summary and mark analysis complete

[IMPORTANT INSTRUCTIONS]
- NEVER ask the user for more information directly - use the tools first
- ALWAYS use infra_requirements_parser_tool to analyze the request
- If the user request is empty, unclear, or lacks detail, still use the tool
- Extract what requirements you can from the available information
- Make reasonable assumptions based on common infrastructure patterns
- If critical information is missing, note it in the analysis but don't stop the workflow
- Always provide a structured analysis even with limited information

[HANDLING INCOMPLETE REQUESTS]
- For empty requests: Use infra_requirements_parser_tool with "AWS infrastructure deployment"
- For unclear requests: Use infra_requirements_parser_tool with the available information
- For partial requests: Use infra_requirements_parser_tool to extract what you can
- Always use the tools before making any assumptions

[TOOL USAGE]
- First tool call: infra_requirements_parser_tool with the user's request
- Second tool call: aws_service_discovery_tool with the results from first tool
- Then provide a summary based on the tool outputs

[COMPLETION DETECTION]
- Only mark complete when AWS service mapping is successfully generated
- If any tool fails, do not set completion flag
- Always provide clear completion status in final response

[OUTPUT]
Provide a clear, structured response that includes:
- Summary of business and technical requirements (from tool output)
- Key constraints and assumptions (from tool output)
- Risk factors and mitigation strategies (from tool output)
- Scalability and performance considerations (from tool output)
- Any missing information that should be clarified
- Clear completion status: "Requirements analysis completed successfully"
            """),
                MessagesPlaceholder(variable_name="messages")
            ])
        )
        
        requirements_logger.log_structured(
            level="INFO",
            message="=== REQUIREMENTS ANALYZER REACT AGENT CREATED SUCCESSFULLY ===",
            extra={
                "agent_type": type(requirements_analyzer).__name__,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "tools_count": 2
            }
        )
        
        return requirements_analyzer
        
    except Exception as e:
        requirements_logger.log_structured(
            level="ERROR",
            message="=== FAILED TO CREATE REQUIREMENTS ANALYZER REACT AGENT ===",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "config_type": type(config).__name__ if config else "None"
            }
        )
        raise
