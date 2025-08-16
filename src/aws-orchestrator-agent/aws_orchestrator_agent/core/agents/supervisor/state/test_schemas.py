"""
Simple test for state schema definitions.

This test verifies that all state schemas can be created and validated
correctly with proper type checking and default values.
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
from aws_orchestrator_agent.core.supervisor.state.reducers import (
    merge_validation_reports,
    append_audit_log,
    merge_conversation_history
)


class TestStateSchemas:
    """Test cases for state schema definitions."""
    
    def test_supervisor_state_creation(self):
        """Test that SupervisorState can be created with minimal data."""
        state = SupervisorState(user_request="Create a VPC")
        
        assert state.user_request == "Create a VPC"
        assert state.status == WorkflowStatus.PENDING
        assert state.workflow_id is not None
        assert state.current_step == "initialized"
        assert state.analysis_state is None
        assert state.generation_state is None
        assert state.validation_state is None
        assert state.editor_state is None
    
    def test_analysis_state_creation(self):
        """Test that AnalysisState can be created."""
        state = AnalysisState(query="What resources do I need for a web application?")
        
        assert state.query == "What resources do I need for a web application?"
        assert state.analysis_complete is False
        assert state.confidence_score == 0.0
        assert state.conversation_history == []
        assert state.requirements == {}
        assert state.aws_context == {}
    
    def test_generation_state_creation(self):
        """Test that GenerationState can be created."""
        requirements = {"vpc": True, "subnets": 2, "security_groups": True}
        state = GenerationState(
            module_name="web_app_infrastructure",
            requirements=requirements
        )
        
        assert state.module_name == "web_app_infrastructure"
        assert state.requirements == requirements
        assert state.generation_complete is False
        assert state.generated_files == []
        assert state.terraform_code == {}
    
    def test_validation_state_creation(self):
        """Test that ValidationState can be created."""
        terraform_code = {"main.tf": "# Terraform code here"}
        state = ValidationState(terraform_code=terraform_code)
        
        assert state.terraform_code == terraform_code
        assert state.validation_complete is False
        assert state.validation_reports == {}
        assert state.security_scan_results == {}
        assert state.compliance_results == {}
        assert state.overall_score == 0.0
    
    def test_editor_state_creation(self):
        """Test that EditorState can be created."""
        original_code = {"main.tf": "# Original code"}
        modifications = {"add_security_group": True}
        state = EditorState(
            original_code=original_code,
            modifications=modifications
        )
        
        assert state.original_code == original_code
        assert state.modifications == modifications
        assert state.editor_complete is False
        assert state.modified_code == {}
        assert state.surgical_changes == []
    
    def test_supervisor_state_methods(self):
        """Test SupervisorState utility methods."""
        state = SupervisorState(user_request="Test request")
        
        # Test agent state management
        analysis_state = AnalysisState(query="Test query")
        state.set_agent_state(AgentType.ANALYSIS, analysis_state)
        assert state.get_agent_state(AgentType.ANALYSIS) == analysis_state
        
        # Test routing history
        state.add_routing_entry("analysis", "completed", {"agent": "analysis"})
        assert len(state.routing_history) == 1
        assert state.routing_history[0]["step"] == "analysis"
        assert state.routing_history[0]["status"] == "completed"
        
        # Test audit log
        state.add_audit_entry("test_action", {"details": "test"})
        assert len(state.audit_log) == 1
        assert state.audit_log[0]["action"] == "test_action"
        
        # Test completion percentage
        assert state.get_completion_percentage() == 0.0
        
        # Test workflow completion
        assert state.is_workflow_complete() is False
        state.status = WorkflowStatus.COMPLETED
        assert state.is_workflow_complete() is True
        assert state.get_completion_percentage() == 100.0


class TestReducerFunctions:
    """Test cases for reducer functions."""
    
    def test_merge_validation_reports(self):
        """Test validation report merging."""
        old = {"syntax": "passed", "security": ["warning1"]}
        new = {"syntax": "passed", "security": ["warning2"], "compliance": "failed"}
        
        merged = merge_validation_reports(old, new)
        
        assert merged["syntax"] == "passed"
        assert "warning1" in merged["security"]
        assert "warning2" in merged["security"]
        assert merged["compliance"] == "failed"
    
    def test_append_audit_log(self):
        """Test audit log appending."""
        old = [{"action": "start", "timestamp": "2023-01-01T00:00:00"}]
        new = {"action": "complete", "details": "test"}
        
        merged = append_audit_log(old, new)
        
        assert len(merged) == 2
        assert merged[0]["action"] == "start"
        assert merged[1]["action"] == "complete"
        assert "timestamp" in merged[1]
    
    def test_merge_conversation_history(self):
        """Test conversation history merging."""
        old = [{"message_id": "1", "content": "Hello"}]
        new = [{"message_id": "2", "content": "World"}]
        
        merged = merge_conversation_history(old, new)
        
        assert len(merged) == 2
        assert merged[0]["message_id"] == "1"
        assert merged[1]["message_id"] == "2"


if __name__ == "__main__":
    # Run basic tests
    print("Testing state schema creation...")
    test_schemas = TestStateSchemas()
    
    test_schemas.test_supervisor_state_creation()
    print("✅ SupervisorState creation test passed!")
    
    test_schemas.test_analysis_state_creation()
    print("✅ AnalysisState creation test passed!")
    
    test_schemas.test_generation_state_creation()
    print("✅ GenerationState creation test passed!")
    
    test_schemas.test_validation_state_creation()
    print("✅ ValidationState creation test passed!")
    
    test_schemas.test_editor_state_creation()
    print("✅ EditorState creation test passed!")
    
    test_schemas.test_supervisor_state_methods()
    print("✅ SupervisorState methods test passed!")
    
    print("\nTesting reducer functions...")
    test_reducers = TestReducerFunctions()
    
    test_reducers.test_merge_validation_reports()
    print("✅ merge_validation_reports test passed!")
    
    test_reducers.test_append_audit_log()
    print("✅ append_audit_log test passed!")
    
    test_reducers.test_merge_conversation_history()
    print("✅ merge_conversation_history test passed!")
    
    print("\n🎉 All state schema tests passed!") 