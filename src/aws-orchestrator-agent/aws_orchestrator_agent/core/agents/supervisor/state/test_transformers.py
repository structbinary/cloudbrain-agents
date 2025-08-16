"""
Test for state transformation functions.

This test verifies that all transformation functions work correctly
for mapping between Supervisor and specialized agent subgraph states.
"""

import pytest
from datetime import datetime
from aws_orchestrator_agent.core.supervisor.state.schemas import (
    SupervisorState,
    AnalysisState,
    GenerationState,
    ValidationState,
    EditorState,
    WorkflowStatus,
    AgentType
)
from aws_orchestrator_agent.core.supervisor.state.transformers import (
    supervisor_to_analysis_state,
    analysis_to_supervisor_state,
    supervisor_to_generation_state,
    generation_to_supervisor_state,
    supervisor_to_validation_state,
    validation_to_supervisor_state,
    supervisor_to_editor_state,
    editor_to_supervisor_state,
    get_transformation_functions
)


class TestTransformationFunctions:
    """Test cases for state transformation functions."""
    
    def test_supervisor_to_analysis_state(self):
        """Test supervisor to analysis state transformation."""
        # Create supervisor state
        supervisor_state = SupervisorState(
            user_request="Create a VPC with subnets",
            mcp_context={"aws_context": {"region": "us-west-2"}}
        )
        
        # Transform to analysis state
        analysis_state = supervisor_to_analysis_state(supervisor_state)
        
        assert analysis_state.query == "Create a VPC with subnets"
        assert analysis_state.aws_context == {"region": "us-west-2"}
        assert analysis_state.analysis_started_at is not None
        assert analysis_state.analysis_complete is False
    
    def test_analysis_to_supervisor_state(self):
        """Test analysis to supervisor state transformation."""
        # Create supervisor state
        supervisor_state = SupervisorState(user_request="Create a VPC")
        
        # Create analysis state
        analysis_state = AnalysisState(
            query="Create a VPC",
            requirements={"vpc": True, "subnets": 2},
            analysis_complete=True,
            confidence_score=0.9
        )
        
        # Transform back to supervisor state
        updated_supervisor = analysis_to_supervisor_state(supervisor_state, analysis_state)
        
        assert updated_supervisor.analysis_state == analysis_state
        assert updated_supervisor.current_step == "analysis"
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "analysis"
        assert len(updated_supervisor.audit_log) == 1
        assert updated_supervisor.audit_log[0]["action"] == "analysis_completed"
    
    def test_supervisor_to_generation_state(self):
        """Test supervisor to generation state transformation."""
        # Create supervisor state with analysis
        supervisor_state = SupervisorState(user_request="web_app_infrastructure")
        supervisor_state.analysis_state = AnalysisState(
            query="Create web app infrastructure",
            requirements={"vpc": True, "subnets": 2, "security_groups": True},
            analysis_complete=True
        )
        
        # Transform to generation state
        generation_state = supervisor_to_generation_state(supervisor_state)
        
        assert generation_state.module_name == "web_app_infrastructure"
        assert generation_state.requirements == {"vpc": True, "subnets": 2, "security_groups": True}
        assert generation_state.generation_started_at is not None
        assert generation_state.generation_complete is False
    
    def test_generation_to_supervisor_state(self):
        """Test generation to supervisor state transformation."""
        # Create supervisor state
        supervisor_state = SupervisorState(user_request="web_app_infrastructure")
        
        # Create generation state
        generation_state = GenerationState(
            module_name="web_app_infrastructure",
            requirements={"vpc": True},
            generated_files=["main.tf", "variables.tf"],
            terraform_code={"main.tf": "# VPC resource", "variables.tf": "# Variables"},
            generation_complete=True
        )
        
        # Transform back to supervisor state
        updated_supervisor = generation_to_supervisor_state(supervisor_state, generation_state)
        
        assert updated_supervisor.generation_state == generation_state
        assert updated_supervisor.current_step == "generation"
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "generation"
        assert len(updated_supervisor.audit_log) == 1
        assert updated_supervisor.audit_log[0]["action"] == "generation_completed"
    
    def test_supervisor_to_validation_state(self):
        """Test supervisor to validation state transformation."""
        # Create supervisor state with generation
        supervisor_state = SupervisorState(user_request="web_app_infrastructure")
        supervisor_state.generation_state = GenerationState(
            module_name="web_app_infrastructure",
            requirements={},
            terraform_code={"main.tf": "# VPC resource"},
            generation_complete=True
        )
        
        # Transform to validation state
        validation_state = supervisor_to_validation_state(supervisor_state)
        
        assert validation_state.terraform_code == {"main.tf": "# VPC resource"}
        assert validation_state.validation_started_at is not None
        assert validation_state.validation_complete is False
    
    def test_validation_to_supervisor_state(self):
        """Test validation to supervisor state transformation."""
        # Create supervisor state
        supervisor_state = SupervisorState(user_request="web_app_infrastructure")
        
        # Create validation state
        validation_state = ValidationState(
            terraform_code={"main.tf": "# VPC resource"},
            validation_reports={"syntax": "passed"},
            overall_score=95.0,
            validation_complete=True
        )
        
        # Transform back to supervisor state
        updated_supervisor = validation_to_supervisor_state(supervisor_state, validation_state)
        
        assert updated_supervisor.validation_state == validation_state
        assert updated_supervisor.current_step == "validation"
        assert updated_supervisor.status == WorkflowStatus.COMPLETED
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "validation"
        assert len(updated_supervisor.audit_log) == 1
        assert updated_supervisor.audit_log[0]["action"] == "validation_completed"
    
    def test_supervisor_to_editor_state(self):
        """Test supervisor to editor state transformation."""
        # Create supervisor state with generation
        supervisor_state = SupervisorState(user_request="Add security group")
        supervisor_state.generation_state = GenerationState(
            module_name="web_app_infrastructure",
            requirements={},
            terraform_code={"main.tf": "# VPC resource"},
            generation_complete=True
        )
        
        # Transform to editor state
        editor_state = supervisor_to_editor_state(supervisor_state)
        
        assert editor_state.original_code == {"main.tf": "# VPC resource"}
        assert editor_state.modifications == {"user_request": "Add security group"}
        assert editor_state.editing_started_at is not None
        assert editor_state.editor_complete is False
    
    def test_editor_to_supervisor_state(self):
        """Test editor to supervisor state transformation."""
        # Create supervisor state
        supervisor_state = SupervisorState(user_request="Add security group")
        
        # Create editor state
        editor_state = EditorState(
            original_code={"main.tf": "# VPC resource"},
            modifications={"add_security_group": True},
            modified_code={"main.tf": "# VPC resource", "security.tf": "# Security group"},
            surgical_changes=[{"file": "security.tf", "action": "added"}],
            editor_complete=True
        )
        
        # Transform back to supervisor state
        updated_supervisor = editor_to_supervisor_state(supervisor_state, editor_state)
        
        assert updated_supervisor.editor_state == editor_state
        assert updated_supervisor.current_step == "editor"
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "editor"
        assert len(updated_supervisor.audit_log) == 1
        assert updated_supervisor.audit_log[0]["action"] == "editor_completed"
    
    def test_get_transformation_functions(self):
        """Test getting transformation function pairs."""
        # Test for each agent type
        for agent_type in AgentType:
            input_func, output_func = get_transformation_functions(agent_type)
            assert input_func is not None
            assert output_func is not None
        
        # Test invalid agent type
        with pytest.raises(ValueError, match="Unsupported agent type"):
            get_transformation_functions("invalid_agent")
    
    def test_error_handling_missing_analysis_state(self):
        """Test error handling when analysis state is missing for generation."""
        supervisor_state = SupervisorState(user_request="Create VPC")
        
        with pytest.raises(ValueError, match="Analysis state required before generation"):
            supervisor_to_generation_state(supervisor_state)
    
    def test_error_handling_missing_terraform_code(self):
        """Test error handling when no Terraform code is available for validation."""
        supervisor_state = SupervisorState(user_request="Validate code")
        
        with pytest.raises(ValueError, match="No completed Terraform code available for validation"):
            supervisor_to_validation_state(supervisor_state)
    
    def test_error_handling_missing_user_request(self):
        """Test error handling when user request is missing."""
        supervisor_state = SupervisorState(user_request="")
        
        with pytest.raises(ValueError, match="Supervisor state missing required user_request"):
            supervisor_to_analysis_state(supervisor_state)


if __name__ == "__main__":
    # Run basic tests
    print("Testing transformation functions...")
    test_transformers = TestTransformationFunctions()
    
    test_transformers.test_supervisor_to_analysis_state()
    print("✅ supervisor_to_analysis_state test passed!")
    
    test_transformers.test_analysis_to_supervisor_state()
    print("✅ analysis_to_supervisor_state test passed!")
    
    test_transformers.test_supervisor_to_generation_state()
    print("✅ supervisor_to_generation_state test passed!")
    
    test_transformers.test_generation_to_supervisor_state()
    print("✅ generation_to_supervisor_state test passed!")
    
    test_transformers.test_supervisor_to_validation_state()
    print("✅ supervisor_to_validation_state test passed!")
    
    test_transformers.test_validation_to_supervisor_state()
    print("✅ validation_to_supervisor_state test passed!")
    
    test_transformers.test_supervisor_to_editor_state()
    print("✅ supervisor_to_editor_state test passed!")
    
    test_transformers.test_editor_to_supervisor_state()
    print("✅ editor_to_supervisor_state test passed!")
    
    test_transformers.test_get_transformation_functions()
    print("✅ get_transformation_functions test passed!")
    
    test_transformers.test_error_handling_missing_analysis_state()
    print("✅ error handling test passed!")
    
    print("\n🎉 All transformation function tests passed!") 