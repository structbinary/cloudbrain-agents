import json
import uuid
import re
from typing import Dict, List, Any, Optional, Annotated, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger
from .data_generator_prompts import DATA_SOURCE_AGENT_SYSTEM_PROMPT, DATA_SOURCE_AGENT_USER_PROMPT_TEMPLATE

# Create agent logger for data source generator
data_generator_logger = AgentLogger("DATA_GENERATOR")


class DataSourceType(str, Enum):
    """AWS data source types supported - extensible for dynamic discovery"""
    # Core AWS data sources
    AWS_AMI = "aws_ami"
    AWS_VPC = "aws_vpc"
    AWS_SUBNET = "aws_subnet"
    AWS_SECURITY_GROUP = "aws_security_group"
    AWS_AVAILABILITY_ZONES = "aws_availability_zones"
    AWS_CALLER_IDENTITY = "aws_caller_identity"
    AWS_REGION = "aws_region"
    AWS_S3_BUCKET = "aws_s3_bucket"
    AWS_IAM_ROLE = "aws_iam_role"
    AWS_ROUTE53_ZONE = "aws_route53_zone"
    AWS_INSTANCE = "aws_instance"
    AWS_EBS_VOLUME = "aws_ebs_volume"
    AWS_DB_INSTANCE = "aws_db_instance"
    AWS_EKS_CLUSTER = "aws_eks_cluster"
    AWS_RDS_CLUSTER = "aws_rds_cluster"
    
    # Extended AWS data sources
    AWS_SUBNETS = "aws_subnets"
    AWS_INTERNET_GATEWAY = "aws_internet_gateway"
    AWS_NAT_GATEWAY = "aws_nat_gateway"
    AWS_ROUTE_TABLE = "aws_route_table"
    AWS_ELASTIC_IP = "aws_eip"
    AWS_KEY_PAIR = "aws_key_pair"
    AWS_LAUNCH_TEMPLATE = "aws_launch_template"
    AWS_AUTOSCALING_GROUP = "aws_autoscaling_group"
    AWS_LOAD_BALANCER = "aws_lb"
    AWS_TARGET_GROUP = "aws_lb_target_group"
    AWS_CLOUDFRONT_DISTRIBUTION = "aws_cloudfront_distribution"
    AWS_LAMBDA_FUNCTION = "aws_lambda_function"
    AWS_API_GATEWAY = "aws_api_gateway_rest_api"
    AWS_SQS_QUEUE = "aws_sqs_queue"
    AWS_SNS_TOPIC = "aws_sns_topic"
    AWS_CLOUDWATCH_LOG_GROUP = "aws_cloudwatch_log_group"
    AWS_KMS_KEY = "aws_kms_key"
    AWS_SECRETS_MANAGER_SECRET = "aws_secretsmanager_secret"
    
    # Dynamic data source type support
    @classmethod
    def from_string(cls, data_source_type: str) -> 'DataSourceType':
        """Create DataSourceType from string, supporting dynamic types"""
        try:
            return cls(data_source_type)
        except ValueError:
            # For dynamic data source types not in enum, create a new instance
            return cls._create_dynamic_type(data_source_type)
    
    @classmethod
    def _create_dynamic_type(cls, data_source_type: str) -> 'DataSourceType':
        """Create a dynamic data source type for unknown AWS data source types"""
        # This allows the system to handle any AWS data source type dynamically
        return data_source_type  # Return as string for dynamic types

class DataSourceFilter(BaseModel):
    """Individual filter for data source queries"""
    name: str = Field(..., description="Filter attribute name")
    values: List[str] = Field(..., description="Filter values")
    
    @field_validator('name')
    @classmethod
    def validate_filter_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Filter name cannot be empty')
        return v.strip()

class TerraformDataSourceBlock(BaseModel):
    """Individual Terraform data source block specification"""
    
    data_source_type: Union[DataSourceType, str] = Field(..., description="AWS data source type (from planner or dynamic discovery)")
    data_source_name: str = Field(..., description="Terraform data source name")
    logical_name: str = Field(..., description="Human-readable data source name")
    
    # Core configuration
    configuration_attributes: Dict[str, Any] = Field(..., description="Data source configuration")
    filters: List[DataSourceFilter] = Field(default_factory=list, description="Query filters")
    
    # Query characteristics
    most_recent: Optional[bool] = Field(None, description="Get most recent match")
    owners: Optional[List[str]] = Field(None, description="Resource owners filter")
    tags: Optional[Dict[str, str]] = Field(None, description="Tag-based filtering")
    
    # Meta-arguments
    meta_arguments: Dict[str, Any] = Field(default_factory=dict, description="count, for_each, provider")
    
    # Dependencies and references
    depends_on_resources: List[str] = Field(default_factory=list, description="Resources this data source depends on")
    referenced_by_resources: List[str] = Field(default_factory=list, description="Resources that will use this data")
    
    # Generated HCL
    hcl_block: str = Field(..., description="Complete HCL data source block")
    
    # Validation and compliance
    validation_rules: List[str] = Field(default_factory=list, description="Applied validation rules")
    expected_attributes: List[str] = Field(default_factory=list, description="Attributes this data source will expose")
    
    @field_validator('data_source_name')
    @classmethod
    def validate_data_source_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Data source name must be alphanumeric with underscores/hyphens only')
        if v[0].isdigit():
            raise ValueError('Data source name cannot start with a number')
        return v

class DiscoveredDataDependency(BaseModel):
    """Dependency discovered that requires handoff to another agent"""
    
    dependency_id: str = Field(..., description="Unique dependency identifier")
    dependency_type: str = Field(..., description="Type of dependency discovered")
    target_agent: str = Field(..., description="Agent that should handle this dependency")
    
    # Context for handoff
    source_data_source: str = Field(..., description="Data source that triggered this dependency")
    requirement_details: Dict[str, Any] = Field(..., description="Specific details about what's needed")
    priority_level: int = Field(default=3, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    
    # Handoff payload
    handoff_context: Dict[str, Any] = Field(..., description="Context to pass to target agent")
    expected_response: Dict[str, str] = Field(..., description="Expected response format from target agent")
    
    # Timing and blocking
    is_blocking: bool = Field(default=True, description="Whether data source generation should wait")
    timeout_minutes: int = Field(default=30, description="Maximum time to wait for dependency resolution")

class DataSourceGenerationMetrics(BaseModel):
    """Metrics and performance data for data source generation"""
    
    total_data_sources_generated: int = Field(..., description="Number of data sources successfully generated")
    generation_duration_seconds: float = Field(..., description="Time taken for generation")
    dependencies_discovered: int = Field(..., description="Total dependencies found")
    handoffs_required: int = Field(..., description="Number of handoffs needed")
    
    # Data source type breakdown
    data_source_type_counts: Dict[Union[DataSourceType, str], int] = Field(default_factory=dict)
    complexity_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Complexity score (0-10)")
    
    # Filter statistics
    total_filters_applied: int = Field(default=0, description="Total number of filters across all data sources")
    dynamic_lookups_count: int = Field(default=0, description="Number of dynamic lookups required")
    
    # Error tracking
    validation_errors: List[str] = Field(default_factory=list)
    warning_messages: List[str] = Field(default_factory=list)

class DataSourceHandoffRecommendation(BaseModel):
    """Recommendation for agent handoff with data source context"""
    
    target_agent: str = Field(..., description="Recommended target agent")
    handoff_reason: str = Field(..., description="Reason for handoff")
    handoff_priority: int = Field(default=3, ge=1, le=5)
    
    # Handoff data
    context_payload: Dict[str, Any] = Field(..., description="Data to pass to target agent")
    expected_deliverables: List[str] = Field(..., description="What the target agent should produce")
    
    # Data source specific context
    data_source_context: Dict[str, Any] = Field(..., description="Specific data source context for handoff")
    
    # Coordination
    should_wait_for_completion: bool = Field(default=True)
    can_continue_parallel: bool = Field(default=False)

class TerraformDataSourceGenerationResponse(BaseModel):
    """Complete response from generate_terraform_data_sources tool"""
    
    # Generation results
    generated_data_sources: List[TerraformDataSourceBlock] = Field(..., description="Successfully generated data source blocks")
    discovered_dependencies: List[DiscoveredDataDependency] = Field(default_factory=list, description="Dependencies requiring handoffs")
    
    # Agent coordination
    handoff_recommendations: List[DataSourceHandoffRecommendation] = Field(default_factory=list, description="Recommended handoffs to other agents")
    completion_status: str = Field(..., description="Current completion status")
    next_recommended_action: str = Field(..., description="Recommended next action")
    
    # Generation metadata
    generation_metadata: DataSourceGenerationMetrics = Field(..., description="Generation performance metrics")
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    
    # State updates
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="Updates to apply to swarm state")
    workspace_updates: Dict[str, Any] = Field(..., description="Updates for agent workspace")
    
    # Error handling
    critical_errors: List[str] = Field(default_factory=list, description="Critical errors that block progress")
    recoverable_warnings: List[str] = Field(default_factory=list, description="Warnings that don't block progress")
    
    # Checkpoint data
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, description="Data for checkpoint creation")
    
    @field_validator('completion_status')
    @classmethod
    def validate_completion_status(cls, v):
        valid_statuses = ['in_progress', 'completed', 'blocked', 'error', 'waiting_for_dependencies']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v

@tool("generate_terraform_data_sources")
def generate_terraform_data_sources(
    data_source_requirements: Annotated[List[Dict[str, Any]], "Data source requirements from execution plan or agent requests"],
    generation_context: Annotated[Dict[str, Any], "Context from execution plan and previous agents"], 
    state: Annotated[Dict[str, Any], InjectedState]
) -> TerraformDataSourceGenerationResponse:
    """
    Generate Terraform AWS data source blocks from execution plan specifications and agent requests.
    
    This tool analyzes external infrastructure reference needs, generates HCL blocks, 
    identifies dependencies, and provides handoff recommendations to other agents. It supports
    both planner specifications and dynamic agent communication.
    """
    
    try:
        start_time = datetime.now()
        
        data_generator_logger.log_structured(
            level="INFO",
            message="Starting Terraform data source generation",
            extra={
                "data_source_requirements_count": len(data_source_requirements),
                "generation_id": state.get('generation_id', 'unknown'),
                "current_stage": state.get('current_stage', 'unknown'),
                "active_agent": state.get('active_agent', 'unknown')
            }
        )
        
        # Extract context for prompt formatting
        exec_plan = generation_context.get('execution_plan', {})
        workspace = state.get('agent_workspaces', {}).get('data_source_agent', {})
        
        # Format user prompt with actual data, escaping curly braces in JSON
        def escape_json_for_template(json_str):
            """Escape curly braces in JSON strings for template compatibility"""
            return json_str.replace('{', '{{').replace('}', '}}')
        
        formatted_user_prompt = DATA_SOURCE_AGENT_USER_PROMPT_TEMPLATE.format(
            service_name=exec_plan.get('service_name', 'unknown'),
            module_name=exec_plan.get('module_name', 'unknown'),
            target_environment=exec_plan.get('target_environment', 'development'),
            generation_id=state.get('generation_id', str(uuid.uuid4())),
            data_source_requirements=escape_json_for_template(json.dumps(data_source_requirements, indent=2)),
            current_stage=state.get('current_stage', 'planning'),
            active_agent=state.get('active_agent', 'data_source_agent'),
            previous_agent_results=escape_json_for_template(json.dumps(state.get('resolved_dependencies', {}), indent=2)),
            generation_context=escape_json_for_template(json.dumps(generation_context, indent=2)),
            specific_requirements=extract_specific_requirements(generation_context),
            handoff_context=escape_json_for_template(json.dumps(state.get('handoff_context', {}), indent=2)),
            agent_workspace=escape_json_for_template(json.dumps(workspace, indent=2))
        )
        
        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=TerraformDataSourceGenerationResponse)
        
        # Build complete prompt, escaping curly braces in system prompt
        escaped_system_prompt = DATA_SOURCE_AGENT_SYSTEM_PROMPT.replace('{', '{{').replace('}', '}}')
        prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_system_prompt),
            ("user", formatted_user_prompt),
            ("user", "Please respond with valid JSON matching the TerraformDataSourceGenerationResponse schema:\n{format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())
        
        # Create and execute chain using centralized LLM
        try:
            # Get LLM configuration from centralized config
            config_instance = Config()
            llm_config = config_instance.get_llm_config()
            
            data_generator_logger.log_structured(
                level="DEBUG",
                message="Initializing LLM for data source generation",
                extra={
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model'),
                    "llm_temperature": llm_config.get('temperature'),
                    "llm_max_tokens": llm_config.get('max_tokens')
                }
            )
            
            model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            
            data_generator_logger.log_structured(
                level="DEBUG",
                message="LLM initialized successfully for data source generation",
                extra={
                    "model_type": type(model).__name__
                }
            )
        except Exception as e:
            data_generator_logger.log_structured(
                level="ERROR",
                message="Failed to initialize LLM for data source generation",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
        
        chain = prompt | model | parser
        
        data_generator_logger.log_structured(
            level="DEBUG",
            message="Executing LLM chain for data source generation",
            extra={
                "prompt_length": len(formatted_user_prompt),
                "generation_id": state.get('generation_id', 'unknown')
            }
        )
        
        # Execute the chain
        llm_response = chain.invoke({})
        
        data_generator_logger.log_structured(
            level="DEBUG",
            message="LLM response received, starting post-processing",
            extra={
                "generated_data_sources_count": len(llm_response.generated_data_sources),
                "discovered_dependencies_count": len(llm_response.discovered_dependencies),
                "generation_id": state.get('generation_id', 'unknown')
            }
        )
        
        # Post-process and enhance response
        enhanced_response = post_process_data_source_response(
            llm_response, 
            state, 
            generation_context, 
            start_time
        )
        
        data_generator_logger.log_structured(
            level="INFO",
            message="Terraform data source generation completed successfully",
            extra={
                "final_data_sources_count": len(enhanced_response.generated_data_sources),
                "final_dependencies_count": len(enhanced_response.discovered_dependencies),
                "generation_duration_seconds": enhanced_response.generation_metadata.generation_duration_seconds,
                "completion_status": enhanced_response.completion_status,
                "generation_id": state.get('generation_id', 'unknown')
            }
        )
        
        return enhanced_response
        
    except Exception as e:
        data_generator_logger.log_structured(
            level="ERROR",
            message="Terraform data source generation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "generation_id": state.get('generation_id', 'unknown'),
                "current_stage": state.get('current_stage', 'unknown')
            }
        )
        return create_data_source_error_response(e, state, datetime.now())

def extract_specific_requirements(context: Dict[str, Any]) -> str:
    """Extract specific requirements from context"""
    requirements = []
    
    exec_plan = context.get('execution_plan', {})
    
    if 'external_references' in exec_plan:
        requirements.append(f"External References: {exec_plan['external_references']}")
    
    if 'network_discovery' in exec_plan:
        requirements.append(f"Network Discovery: {exec_plan['network_discovery']}")
    
    if 'service_discovery' in exec_plan:
        requirements.append(f"Service Discovery: {exec_plan['service_discovery']}")
    
    if 'multi_region_requirements' in exec_plan:
        requirements.append(f"Multi-Region: {exec_plan['multi_region_requirements']}")
    
    return '\n'.join(requirements) if requirements else "No specific requirements specified"

def post_process_data_source_response(
    llm_response: TerraformDataSourceGenerationResponse,
    state: Dict[str, Any], 
    context: Dict[str, Any],
    start_time: datetime
) -> TerraformDataSourceGenerationResponse:
    """Post-process LLM response with additional validation and enhancements"""
    
    # Calculate actual generation duration
    generation_duration = (datetime.now() - start_time).total_seconds()
    llm_response.generation_metadata.generation_duration_seconds = generation_duration
    
    # Validate generated HCL syntax and data source configurations
    validated_data_sources = []
    validation_errors = []
    
    for data_source in llm_response.generated_data_sources:
        validation_result = validate_terraform_data_source(data_source)
        if validation_result['valid']:
            validated_data_sources.append(data_source)
        else:
            validation_errors.extend(validation_result['errors'])
            # Attempt to fix common issues
            fixed_data_source = attempt_data_source_fix(data_source, validation_result['errors'])
            if fixed_data_source:
                validated_data_sources.append(fixed_data_source)
                llm_response.recoverable_warnings.append(
                    f"Fixed validation issues for {data_source.data_source_name}"
                )
    
    # Update response with validated data sources
    llm_response.generated_data_sources = validated_data_sources
    llm_response.generation_metadata.validation_errors.extend(validation_errors)
    
    # Enhance dependencies with additional context
    enhanced_dependencies = enhance_data_source_dependencies(
        llm_response.discovered_dependencies, 
        validated_data_sources,
        context
    )
    llm_response.discovered_dependencies = enhanced_dependencies
    
    # Create comprehensive handoff recommendations
    llm_response.handoff_recommendations = create_data_source_handoff_recommendations(
        enhanced_dependencies,
        validated_data_sources
    )
    
    # Add comprehensive state updates
    llm_response.state_updates = create_data_source_state_updates(
        validated_data_sources,
        enhanced_dependencies,
        state,
        llm_response.completion_status
    )
    
    # Add workspace updates
    llm_response.workspace_updates = create_data_source_workspace_updates(
        validated_data_sources,
        enhanced_dependencies,
        llm_response.generation_metadata,
        llm_response.completion_status
    )
    
    # Add checkpoint data
    llm_response.checkpoint_data = create_data_source_checkpoint_data(
        validated_data_sources,
        enhanced_dependencies,
        llm_response.completion_status
    )
    
    return llm_response

def validate_terraform_data_source(data_source: TerraformDataSourceBlock) -> Dict[str, Any]:
    """Validate individual Terraform data source"""
    errors = []
    
    # Validate HCL syntax
    if not validate_data_source_hcl_syntax(data_source.hcl_block):
        errors.append(f"Invalid HCL syntax in {data_source.data_source_name}")
    
    # Validate AWS data source type
    if not validate_aws_data_source_type(data_source.data_source_type, data_source.configuration_attributes):
        errors.append(f"Invalid AWS data source configuration for {data_source.data_source_type}")
    
    # Validate naming conventions
    if not validate_data_source_naming_convention(data_source.data_source_name):
        errors.append(f"Data source name {data_source.data_source_name} doesn't follow conventions")
    
    # Validate filters
    filter_validation = validate_data_source_filters(data_source)
    if filter_validation['errors']:
        errors.extend(filter_validation['errors'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_data_source_hcl_syntax(hcl_block: str) -> bool:
    """Basic HCL syntax validation for data sources"""
    try:
        # Check for balanced braces
        if hcl_block.count('{') != hcl_block.count('}'):
            return False
        
        # Check for valid data source block structure
        if not re.match(r'data\s+"[^"]+"\s+"[^"]+"\s*{', hcl_block):
            return False
        
        # Check for proper attribute formatting
        lines = hcl_block.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line in ['{', '}']:
                if 'filter' in line and '{' in line:
                    continue  # Filter blocks are valid
                elif '=' not in line and 'data' not in line and '{' not in line and '}' not in line and 'filter' not in line:
                    return False
        
        return True
    except Exception:
        return False

def validate_aws_data_source_type(data_source_type: Union[DataSourceType, str], attributes: Dict[str, Any]) -> bool:
    """Validate AWS data source type and basic attributes with support for dynamic types"""
    
    # Convert to string for validation
    if isinstance(data_source_type, DataSourceType):
        data_source_type_str = data_source_type.value
    else:
        data_source_type_str = str(data_source_type)
    
    # Define common attributes for data source types
    common_attrs = {
        'aws_ami': ['most_recent', 'owners', 'filter'],
        'aws_vpc': ['filter', 'id'],
        'aws_subnet': ['filter', 'vpc_id'],
        'aws_subnets': ['filter', 'vpc_id'],
        'aws_security_group': ['filter', 'vpc_id'],
        'aws_availability_zones': ['state'],
        'aws_caller_identity': [],  # No required attributes
        'aws_region': ['current'],
        'aws_s3_bucket': ['bucket'],
        'aws_iam_role': ['name'],
        'aws_route53_zone': ['name'],
        'aws_instance': ['filter', 'instance_id'],
        'aws_ebs_volume': ['filter', 'volume_id'],
        'aws_db_instance': ['db_instance_identifier'],
        'aws_eks_cluster': ['name'],
        'aws_rds_cluster': ['cluster_identifier'],
        'aws_internet_gateway': ['filter', 'internet_gateway_id'],
        'aws_nat_gateway': ['filter', 'nat_gateway_id'],
        'aws_route_table': ['filter', 'route_table_id'],
        'aws_eip': ['filter', 'public_ip'],
        'aws_key_pair': ['key_name'],
        'aws_launch_template': ['name'],
        'aws_autoscaling_group': ['name'],
        'aws_lb': ['name'],
        'aws_lb_target_group': ['name'],
        'aws_cloudfront_distribution': ['id'],
        'aws_lambda_function': ['function_name'],
        'aws_api_gateway_rest_api': ['name'],
        'aws_sqs_queue': ['name'],
        'aws_sns_topic': ['name'],
        'aws_cloudwatch_log_group': ['name'],
        'aws_kms_key': ['key_id'],
        'aws_secretsmanager_secret': ['name']
    }
    
    if data_source_type_str not in common_attrs:
        # For dynamic data source types, perform basic validation
        return validate_dynamic_data_source_type(data_source_type_str, attributes)
    
    # At least one common attribute should be present for most data sources
    expected_attrs = common_attrs[data_source_type_str]
    if expected_attrs and not any(attr in attributes for attr in expected_attrs):
        return False
    
    return True

def validate_dynamic_data_source_type(data_source_type: str, attributes: Dict[str, Any]) -> bool:
    """Validate dynamic data source types discovered during agent communication"""
    # Basic validation for dynamic data source types
    if not data_source_type or not isinstance(data_source_type, str):
        return False
    
    # Check if it follows AWS data source naming convention
    if not re.match(r'^aws_[a-z_]+$', data_source_type):
        return False
    
    # Basic attribute validation - should have at least some configuration
    if not attributes or not isinstance(attributes, dict):
        return False
    
    # For dynamic types, we're more lenient - just ensure it has some basic structure
    return True

def validate_data_source_naming_convention(data_source_name: str) -> bool:
    """Validate Terraform data source naming convention"""
    # Check if name follows snake_case convention
    if not re.match(r'^[a-z][a-z0-9_]*$', data_source_name):
        return False
    
    # Check if name is not too long
    if len(data_source_name) > 64:
        return False
    
    # Check for reserved words
    reserved_words = ['data', 'resource', 'variable', 'output', 'module', 'provider']
    if data_source_name in reserved_words:
        return False
    
    return True

def validate_data_source_filters(data_source: TerraformDataSourceBlock) -> Dict[str, Any]:
    """Validate data source filters for efficiency and correctness"""
    errors = []
    warnings = []
    
    for filter_item in data_source.filters:
        # Check filter name is valid
        if not filter_item.name or not filter_item.name.strip():
            errors.append(f"Empty filter name in {data_source.data_source_name}")
        
        # Check filter values are provided
        if not filter_item.values:
            errors.append(f"Empty filter values for {filter_item.name} in {data_source.data_source_name}")
        
        # Check for overly broad filters
        if filter_item.name == 'state' and 'available' in filter_item.values:
            warnings.append(f"Broad 'state=available' filter in {data_source.data_source_name} may return many results")
        
        # Check for potentially inefficient wildcard patterns
        for value in filter_item.values:
            if isinstance(value, str) and value.count('*') > 2:
                warnings.append(f"Complex wildcard pattern '{value}' in {data_source.data_source_name} may be inefficient")
    
    return {
        'errors': errors,
        'warnings': warnings
    }

def attempt_data_source_fix(
    data_source: TerraformDataSourceBlock, 
    errors: List[str]
) -> Optional[TerraformDataSourceBlock]:
    """Attempt to fix common data source issues"""
    
    fixed_data_source = data_source.copy(deep=True)
    
    # Try to fix HCL syntax issues
    if any("Invalid HCL syntax" in error for error in errors):
        fixed_hcl = fix_data_source_hcl_syntax(data_source.hcl_block)
        if fixed_hcl != data_source.hcl_block:
            fixed_data_source.hcl_block = fixed_hcl
    
    # Try to fix naming issues
    if any("doesn't follow conventions" in error for error in errors):
        fixed_name = fix_data_source_name(data_source.data_source_name)
        if fixed_name != data_source.data_source_name:
            fixed_data_source.data_source_name = fixed_name
            # Update HCL block with new name
            fixed_data_source.hcl_block = re.sub(
                f'data\\s+"{data_source.data_source_type}"\\s+"{data_source.data_source_name}"',
                f'data "{data_source.data_source_type}" "{fixed_name}"',
                fixed_data_source.hcl_block
            )
    
    # Validate the fixed data source
    validation_result = validate_terraform_data_source(fixed_data_source)
    if validation_result['valid']:
        return fixed_data_source
    
    return None

def fix_data_source_hcl_syntax(hcl_block: str) -> str:
    """Attempt to fix common HCL syntax issues in data sources"""
    
    # Fix common formatting issues
    fixed = hcl_block
    
    # Ensure proper spacing around equals signs
    fixed = re.sub(r'\s*=\s*', ' = ', fixed)
    
    # Ensure proper indentation (2 spaces)
    lines = fixed.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
        
        if stripped == '}':
            indent_level = max(0, indent_level - 1)
        
        formatted_lines.append('  ' * indent_level + stripped)
        
        if stripped.endswith('{'):
            indent_level += 1
    
    return '\n'.join(formatted_lines)

def fix_data_source_name(data_source_name: str) -> str:
    """Fix data source naming convention issues"""
    
    # Convert to lowercase
    fixed = data_source_name.lower()
    
    # Replace invalid characters with underscores
    fixed = re.sub(r'[^a-z0-9_]', '_', fixed)
    
    # Ensure it starts with a letter
    if fixed[0].isdigit():
        fixed = 'd_' + fixed
    
    # Remove multiple consecutive underscores
    fixed = re.sub(r'_+', '_', fixed)
    
    # Remove leading/trailing underscores
    fixed = fixed.strip('_')
    
    # Ensure it's not too long
    if len(fixed) > 64:
        fixed = fixed[:64].rstrip('_')
    
    return fixed

def enhance_data_source_dependencies(
    dependencies: List[DiscoveredDataDependency],
    data_sources: List[TerraformDataSourceBlock],
    context: Dict[str, Any]
) -> List[DiscoveredDataDependency]:
    """Enhance dependencies with additional context and validation"""
    
    enhanced_deps = []
    
    for dep in dependencies:
        enhanced_dep = dep.copy(deep=True)
        
        # Add data source context
        source_data_source = next(
            (ds for ds in data_sources if ds.data_source_name == dep.source_data_source),
            None
        )
        
        if source_data_source:
            enhanced_dep.handoff_context.update({
                'source_data_source_type': source_data_source.data_source_type,
                'source_data_source_config': source_data_source.configuration_attributes,
                'source_filters': [f.dict() for f in source_data_source.filters]
            })
        
        # Add execution plan context
        enhanced_dep.handoff_context.update({
            'execution_plan_excerpt': context.get('execution_plan', {}),
            'generation_context': context
        })
        
        enhanced_deps.append(enhanced_dep)
    
    return enhanced_deps

def create_data_source_handoff_recommendations(
    dependencies: List[DiscoveredDataDependency],
    data_sources: List[TerraformDataSourceBlock]
) -> List[DataSourceHandoffRecommendation]:
    """Create comprehensive handoff recommendations for data sources"""
    
    recommendations = []
    
    # Group dependencies by target agent
    agent_groups = {}
    for dep in dependencies:
        if dep.target_agent not in agent_groups:
            agent_groups[dep.target_agent] = []
        agent_groups[dep.target_agent].append(dep)
    
    # Create recommendations for each target agent
    for agent, deps in agent_groups.items():
        # Determine handoff characteristics
        max_priority = max(dep.priority_level for dep in deps)
        has_blocking = any(dep.is_blocking for dep in deps)
        
        # Create expected deliverables based on agent type
        if agent == 'variable_definition_agent':
            deliverables = ['Variable definitions for filter parameters', 'Variable validation rules for data source queries']
        elif agent == 'local_values_agent':
            deliverables = ['Local value definitions for filter expressions', 'Computed filter values']
        elif agent == 'resource_configuration_agent':
            deliverables = ['Resource dependency coordination', 'Managed resource references']
        else:
            deliverables = ['Required configurations']
        
        recommendation = DataSourceHandoffRecommendation(
            target_agent=agent,
            handoff_reason=f"Resolve {len(deps)} data source dependencies",
            handoff_priority=max_priority,
            context_payload={
                'dependencies': [dep.dict() for dep in deps],
                'total_count': len(deps),
                'priority_levels': [dep.priority_level for dep in deps],
                'affected_data_sources': [dep.source_data_source for dep in deps]
            },
            expected_deliverables=deliverables,
            data_source_context={
                'data_source_types': list(set(ds.data_source_type for ds in data_sources)),
                'filter_requirements': [ds.filters for ds in data_sources],
                'external_references': [ds.logical_name for ds in data_sources]
            },
            should_wait_for_completion=has_blocking,
            can_continue_parallel=not has_blocking
        )
        
        recommendations.append(recommendation)
    
    return recommendations

def create_data_source_state_updates(
    data_sources: List[TerraformDataSourceBlock],
    dependencies: List[DiscoveredDataDependency],
    current_state: Dict[str, Any],
    completion_status: str
) -> Dict[str, Any]:
    """Create comprehensive state updates for the swarm"""
    
    updates = {
        'terraform_data_sources': [ds.dict() for ds in data_sources],
        'pending_dependencies': {
            **current_state.get('pending_dependencies', {}),
            'data_source_agent': [dep.dict() for dep in dependencies]
        },
        'agent_status_matrix': {
            **current_state.get('agent_status_matrix', {}),
            'data_source_agent': completion_status
        },
        'planning_progress': {
            **current_state.get('planning_progress', {}),
            'data_source_agent': 1.0 if completion_status == 'completed' else 0.6
        }
    }
    
    return updates

def create_data_source_workspace_updates(
    data_sources: List[TerraformDataSourceBlock],
    dependencies: List[DiscoveredDataDependency],
    metrics: DataSourceGenerationMetrics,
    completion_status: str
) -> Dict[str, Any]:
    """Create workspace updates for the data source agent"""
    
    return {
        'generated_data_sources': [ds.dict() for ds in data_sources],
        'pending_dependencies': [dep.dict() for dep in dependencies],
        'generation_metrics': metrics.dict(),
        'completion_status': completion_status,
        'completion_timestamp': datetime.now().isoformat(),
        'data_source_summary': {
            'total_data_sources': len(data_sources),
            'data_source_types': list(set(ds.data_source_type for ds in data_sources)),
            'dependencies_discovered': len(dependencies),
            'complexity_score': metrics.complexity_score,
            'external_references': [ds.logical_name for ds in data_sources]
        }
    }

def create_data_source_checkpoint_data(
    data_sources: List[TerraformDataSourceBlock],
    dependencies: List[DiscoveredDataDependency],
    completion_status: str
) -> Dict[str, Any]:
    """Create checkpoint data for recovery"""
    
    return {
        'stage': 'planning',
        'agent': 'data_source_agent',
        'checkpoint_type': 'data_source_generation_complete',
        'data_sources_generated': len(data_sources),
        'dependencies_discovered': len(dependencies),
        'completion_status': completion_status,
        'timestamp': datetime.now().isoformat(),
        'data_source_names': [ds.data_source_name for ds in data_sources],
        'dependency_types': [d.dependency_type for d in dependencies],
        'external_references': [ds.logical_name for ds in data_sources]
    }

def create_data_source_error_response(
    error: Exception, 
    state: Dict[str, Any], 
    start_time: datetime
) -> TerraformDataSourceGenerationResponse:
    """Create error response when tool execution fails"""
    
    generation_duration = (datetime.now() - start_time).total_seconds()
    
    return TerraformDataSourceGenerationResponse(
        generated_data_sources=[],
        discovered_dependencies=[],
        handoff_recommendations=[],
        completion_status='error',
        next_recommended_action='escalate_to_human_or_retry',
        generation_metadata=DataSourceGenerationMetrics(
            total_data_sources_generated=0,
            generation_duration_seconds=generation_duration,
            dependencies_discovered=0,
            handoffs_required=0,
            validation_errors=[f"Tool execution failed: {str(error)}"]
        ),
        state_updates={
            'agent_status_matrix': {
                **state.get('agent_status_matrix', {}),
                'data_source_agent': 'error'
            }
        },
        workspace_updates={
            'error': str(error),
            'completion_status': 'error',
            'error_timestamp': datetime.now().isoformat()
        },
        critical_errors=[f"Critical tool failure: {str(error)}"],
        checkpoint_data={
            'stage': 'planning',
            'agent': 'data_source_agent',
            'checkpoint_type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    )