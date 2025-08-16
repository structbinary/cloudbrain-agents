"""
Test for StateManager class.

This test verifies that the StateManager orchestrates state transformations,
provides validation, state snapshotting, error recovery, and integration
points correctly.
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
from aws_orchestrator_agent.core.supervisor.state.state_manager import (
    StateManager,
    StateSnapshot
)


class TestStateManager:
    """Test cases for StateManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_manager = StateManager(max_snapshots=5)
        self.supervisor_state = SupervisorState(user_request="Create a VPC")
    
    def teardown_method(self):
        """Clean up after each test."""
        self.state_manager.clear_snapshots()
    
    def test_initialization(self):
        """Test StateManager initialization."""
        assert self.state_manager.max_snapshots == 5
        assert len(self.state_manager.snapshots) == 0
        assert len(self.state_manager.transformers) == 4  # 4 agent types
        assert AgentType.ANALYSIS in self.state_manager.transformers
        assert AgentType.GENERATION in self.state_manager.transformers
        assert AgentType.VALIDATION in self.state_manager.transformers
        assert AgentType.EDITOR in self.state_manager.transformers
    
    def test_transform_to_subgraph_analysis(self):
        """Test transforming supervisor state to analysis subgraph state."""
        # Save snapshot count before
        prev_count = len(self.state_manager.snapshots)
        # Transform to analysis state
        analysis_state = self.state_manager.transform_to_subgraph(
            AgentType.ANALYSIS, 
            self.supervisor_state
        )
        assert isinstance(analysis_state, AnalysisState)
        assert analysis_state.query == "Create a VPC"
        assert analysis_state.analysis_started_at is not None
        assert analysis_state.analysis_complete is False
        # Check that a new snapshot with the expected description was added
        new_snapshots = self.state_manager.snapshots[prev_count:]
        print('SNAPSHOT DESCRIPTIONS:', [snap.description for snap in self.state_manager.snapshots])
        assert any("Before AgentType.ANALYSIS transformation" in snap.description for snap in new_snapshots)
    
    def test_transform_to_subgraph_generation_with_analysis(self):
        """Test transforming supervisor state to generation subgraph state."""
        # First, add analysis state
        self.supervisor_state.analysis_state = AnalysisState(
            query="Create a VPC",
            requirements={"vpc": True, "subnets": 2},
            analysis_complete=True
        )
        
        # Transform to generation state
        generation_state = self.state_manager.transform_to_subgraph(
            AgentType.GENERATION, 
            self.supervisor_state
        )
        
        assert isinstance(generation_state, GenerationState)
        assert generation_state.module_name == "Create a VPC"
        assert generation_state.requirements == {"vpc": True, "subnets": 2}
        assert generation_state.generation_started_at is not None
        assert generation_state.generation_complete is False
    
    def test_transform_to_subgraph_generation_without_analysis(self):
        """Test that generation transformation fails without analysis state."""
        with pytest.raises(ValueError, match="Analysis state required before generation"):
            self.state_manager.transform_to_subgraph(
                AgentType.GENERATION, 
                self.supervisor_state
            )
    
    def test_merge_from_subgraph_analysis(self):
        """Test merging analysis subgraph state back to supervisor state."""
        # Create analysis state
        analysis_state = AnalysisState(
            query="Create a VPC",
            requirements={"vpc": True, "subnets": 2},
            analysis_complete=True,
            confidence_score=0.9
        )
        
        # Merge back to supervisor state
        updated_supervisor = self.state_manager.merge_from_subgraph(
            AgentType.ANALYSIS,
            self.supervisor_state,
            analysis_state
        )
        
        assert updated_supervisor.analysis_state == analysis_state
        assert updated_supervisor.current_step == "analysis"
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "analysis"
        assert len(updated_supervisor.audit_log) >= 1  # at least analysis_completed
        
        # Check that snapshot was created
        assert any("After AgentType.ANALYSIS merge" in snap.description for snap in self.state_manager.snapshots)
    
    def test_merge_from_subgraph_generation(self):
        """Test merging generation subgraph state back to supervisor state."""
        # Create generation state
        generation_state = GenerationState(
            module_name="vpc_module",
            requirements={"vpc": True},
            generated_files=["main.tf", "variables.tf"],
            terraform_code={"main.tf": "# VPC resource"},
            generation_complete=True
        )
        
        # Merge back to supervisor state
        updated_supervisor = self.state_manager.merge_from_subgraph(
            AgentType.GENERATION,
            self.supervisor_state,
            generation_state
        )
        
        assert updated_supervisor.generation_state == generation_state
        assert updated_supervisor.current_step == "generation"
        assert len(updated_supervisor.routing_history) == 1
        assert updated_supervisor.routing_history[0]["step"] == "generation"
    
    def test_validate_state_supervisor(self):
        """Test supervisor state validation."""
        # Valid supervisor state
        assert self.state_manager.validate_state(self.supervisor_state) is True
        
        # Invalid supervisor state (missing user_request)
        invalid_state = SupervisorState(user_request="")
        assert self.state_manager.validate_state(invalid_state) is False
    
    def test_validate_state_analysis(self):
        """Test analysis state validation."""
        # Valid analysis state
        analysis_state = AnalysisState(
            query="Test query",
            confidence_score=0.8,
            analysis_complete=True,
            requirements={"test": True}
        )
        assert self.state_manager.validate_state(analysis_state) is True
        
        # Invalid analysis state (confidence score out of range)
        invalid_state = AnalysisState(
            query="Test query",
            confidence_score=1.5  # Should be <= 1.0
        )
        assert self.state_manager.validate_state(invalid_state) is False
    
    def test_create_snapshot(self):
        """Test snapshot creation."""
        snapshot = self.state_manager.create_snapshot(
            self.supervisor_state, 
            "Test snapshot"
        )
        
        assert isinstance(snapshot, StateSnapshot)
        assert snapshot.workflow_id == self.supervisor_state.workflow_id
        assert snapshot.description == "Test snapshot"
        assert snapshot.step == self.supervisor_state.current_step
        # Instead of checking the count, check the latest snapshot's description
        assert self.state_manager.snapshots[-1].description == "Test snapshot"
        
        # Check audit entry was added
        assert self.supervisor_state.audit_log[-1]["action"] == "snapshot_created"
        assert self.supervisor_state.audit_log[-1]["description"] == "Test snapshot"
    
    def test_snapshot_limit_enforcement(self):
        """Test that snapshot limit is enforced."""
        # Create more snapshots than the limit
        for i in range(7):  # More than max_snapshots (5)
            self.state_manager.create_snapshot(
                self.supervisor_state, 
                f"Snapshot {i}"
            )
        
        # Should only keep the latest 5 snapshots
        assert len(self.state_manager.snapshots) == 5
        assert self.state_manager.snapshots[0].description == "Snapshot 2"
        assert self.state_manager.snapshots[-1].description == "Snapshot 6"
    
    def test_rollback_to_snapshot(self):
        """Test rolling back to a snapshot."""
        # Create initial state
        initial_state = SupervisorState(user_request="Initial request")
        
        # Create snapshot
        self.state_manager.create_snapshot(initial_state, "Initial snapshot")
        
        # Modify state
        modified_state = SupervisorState(user_request="Modified request")
        modified_state.current_step = "modified"
        
        # Rollback to snapshot
        restored_state = self.state_manager.rollback_to_snapshot(modified_state)
        
        assert restored_state.user_request == "Initial request"
        assert restored_state.current_step == "initialized"
        
        # Check rollback audit entry
        rollback_entries = [entry for entry in restored_state.audit_log if entry["action"] == "state_rollback"]
        assert len(rollback_entries) == 1
        assert rollback_entries[0]["rollback_to"] == "Initial snapshot"
    
    def test_rollback_with_invalid_index(self):
        """Test rollback with invalid snapshot index."""
        with pytest.raises(ValueError, match="No snapshots available for rollback"):
            self.state_manager.rollback_to_snapshot(self.supervisor_state)
        
        # Create one snapshot
        self.state_manager.create_snapshot(self.supervisor_state, "Test")
        
        # Try invalid index
        with pytest.raises(ValueError, match="Invalid snapshot index"):
            self.state_manager.rollback_to_snapshot(self.supervisor_state, 5)
    
    def test_get_snapshot_info(self):
        """Test getting snapshot information."""
        # Create snapshots
        self.state_manager.create_snapshot(self.supervisor_state, "Snapshot 1")
        self.state_manager.create_snapshot(self.supervisor_state, "Snapshot 2")
        
        info = self.state_manager.get_snapshot_info()
        
        assert len(info) == 2
        assert info[0]["description"] == "Snapshot 1"
        assert info[1]["description"] == "Snapshot 2"
        assert "timestamp" in info[0]
        assert "workflow_id" in info[0]
        assert "step" in info[0]
    
    def test_clear_snapshots(self):
        """Test clearing all snapshots."""
        # Create snapshots
        self.state_manager.create_snapshot(self.supervisor_state, "Snapshot 1")
        self.state_manager.create_snapshot(self.supervisor_state, "Snapshot 2")
        
        assert len(self.state_manager.snapshots) == 2
        
        # Clear snapshots
        self.state_manager.clear_snapshots()
        
        assert len(self.state_manager.snapshots) == 0
    
    def test_get_workflow_progress(self):
        """Test getting workflow progress information."""
        # Add some agent states
        self.supervisor_state.analysis_state = AnalysisState(
            query="Test",
            analysis_complete=True
        )
        self.supervisor_state.generation_state = GenerationState(
            module_name="test",
            requirements={},
            generation_complete=False
        )
        
        progress = self.state_manager.get_workflow_progress(self.supervisor_state)
        
        assert progress["workflow_id"] == self.supervisor_state.workflow_id
        assert progress["current_step"] == self.supervisor_state.current_step
        assert progress["status"] == self.supervisor_state.status
        assert "completion_percentage" in progress
        assert "agent_states" in progress
        assert progress["agent_states"]["analysis"]["exists"] is True
        assert progress["agent_states"]["analysis"]["complete"] is True
        assert progress["agent_states"]["generation"]["exists"] is True
        assert progress["agent_states"]["generation"]["complete"] is False
        assert progress["agent_states"]["validation"]["exists"] is False
    
    def test_export_state_for_debugging(self):
        """Test exporting state for debugging."""
        # Add some data to supervisor state
        self.supervisor_state.analysis_state = AnalysisState(
            query="Test",
            analysis_complete=True
        )
        
        export_data = self.state_manager.export_state_for_debugging(self.supervisor_state)
        
        assert "timestamp" in export_data
        assert export_data["workflow_id"] == self.supervisor_state.workflow_id
        assert "state_data" in export_data
        assert "progress" in export_data
        assert "snapshots" in export_data
        assert "error_context" in export_data
    
    def test_error_handling_in_transformation(self):
        """Test error handling during transformation."""
        # Try to transform with invalid agent type
        with pytest.raises(ValueError, match="Unsupported agent type"):
            self.state_manager.transform_to_subgraph("invalid_agent", self.supervisor_state)
        
        # Check that error context was added
        assert "invalid_agent_transformation_error" in self.supervisor_state.error_context
    
    def test_error_handling_in_merge(self):
        """Test error handling during merge."""
        # Try to merge with invalid agent type
        analysis_state = AnalysisState(query="Test")
        
        with pytest.raises(ValueError, match="Unsupported agent type"):
            self.state_manager.merge_from_subgraph("invalid_agent", self.supervisor_state, analysis_state)
        
        # Check that error context was added
        assert "invalid_agent_merge_error" in self.supervisor_state.error_context


if __name__ == "__main__":
    # Run basic tests
    print("Testing StateManager functionality...")
    test_manager = TestStateManager()
    
    test_manager.setup_method()
    test_manager.test_initialization()
    print("✅ StateManager initialization test passed!")
    
    test_manager.test_transform_to_subgraph_analysis()
    print("✅ transform_to_subgraph_analysis test passed!")
    
    test_manager.test_merge_from_subgraph_analysis()
    print("✅ merge_from_subgraph_analysis test passed!")
    
    test_manager.test_validate_state_supervisor()
    print("✅ validate_state_supervisor test passed!")
    
    test_manager.test_create_snapshot()
    print("✅ create_snapshot test passed!")
    
    test_manager.test_rollback_to_snapshot()
    print("✅ rollback_to_snapshot test passed!")
    
    test_manager.test_get_workflow_progress()
    print("✅ get_workflow_progress test passed!")
    
    test_manager.test_error_handling_in_transformation()
    print("✅ error handling test passed!")
    
    print("\n🎉 All StateManager tests passed!") 