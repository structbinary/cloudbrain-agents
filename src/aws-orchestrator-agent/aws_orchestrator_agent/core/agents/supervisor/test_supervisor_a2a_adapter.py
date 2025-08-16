"""
Test for Supervisor Agent A2A Adapter.

This test verifies that the SupervisorAgentAdapter correctly implements
the BaseAgent interface and integrates with the A2A executor flow.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from aws_orchestrator_agent.core.supervisor.supervisor_a2a_adapter import (
    SupervisorAgentAdapter,
    create_supervisor_agent_adapter
)
from aws_orchestrator_agent.core.supervisor.supervisor_agent import SupervisorAgent
from aws_orchestrator_agent.core.supervisor.state import AgentType, WorkflowStatus
from aws_orchestrator_agent.types import AgentResponse


class TestSupervisorAgentAdapter:
    """Test cases for SupervisorAgentAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock supervisor agent
        self.mock_supervisor = Mock(spec=SupervisorAgent)
        self.mock_supervisor.name = "test-supervisor"
        self.mock_supervisor.is_ready.return_value = True
        self.mock_supervisor.supervisor_state = None
        
        # Create the adapter
        self.adapter = SupervisorAgentAdapter(
            supervisor_agent=self.mock_supervisor,
            name="test-adapter"
        )

    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.name == "test-adapter"
        assert self.adapter.supervisor_agent == self.mock_supervisor
        assert self.adapter.config == {}
        assert self.adapter.active_workflows == {}

    def test_name_property(self):
        """Test name property."""
        assert self.adapter.name == "test-adapter"
        
        # Test with custom name
        adapter_with_name = SupervisorAgentAdapter(
            supervisor_agent=self.mock_supervisor,
            name="custom-name"
        )
        assert adapter_with_name.name == "custom-name"

    @pytest.mark.asyncio
    async def test_stream_method_signature(self):
        """Test that stream method has correct signature."""
        # Verify the method exists and is async
        assert asyncio.iscoroutinefunction(self.adapter.stream)
        
        # Test method signature
        import inspect
        sig = inspect.signature(self.adapter.stream)
        params = list(sig.parameters.keys())
        assert params == ['self', 'query', 'context_id', 'task_id']

    @pytest.mark.asyncio
    async def test_stream_initialization(self):
        """Test stream method workflow initialization."""
        query = "Create a VPC"
        context_id = "test-context"
        task_id = "test-task"
        
        # Mock supervisor state
        mock_state = Mock()
        mock_state.workflow_id = "test-workflow-id"
        self.mock_supervisor.supervisor_state = mock_state
        
        # Mock the workflow execution
        with patch.object(self.adapter, '_execute_supervisor_workflow') as mock_execute:
            mock_execute.return_value = None
            
            # Collect responses
            responses = []
            async for response in self.adapter.stream(query, context_id, task_id):
                responses.append(response)
            
            # Verify initialization
            self.mock_supervisor.initialize_workflow.assert_called_once_with(
                query, {"a2a_context_id": context_id}
            )
            
            # Verify initial response
            assert len(responses) >= 1
            initial_response = responses[0]
            assert isinstance(initial_response, AgentResponse)
            assert "Supervisor Agent initialized workflow" in initial_response.content
            assert initial_response.response_type == "text"
            assert not initial_response.is_task_complete
            assert not initial_response.require_user_input
            assert initial_response.metadata["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_determine_agent_type(self):
        """Test agent type determination logic."""
        # Test analysis queries
        assert await self.adapter._determine_agent_type("Analyze my requirements") == AgentType.ANALYSIS
        assert await self.adapter._determine_agent_type("What do I need?") == AgentType.ANALYSIS
        
        # Test generation queries
        assert await self.adapter._determine_agent_type("Create a new VPC") == AgentType.GENERATION
        assert await self.adapter._determine_agent_type("Generate Terraform code") == AgentType.GENERATION
        
        # Test validation queries
        assert await self.adapter._determine_agent_type("Validate my code") == AgentType.VALIDATION
        assert await self.adapter._determine_agent_type("Check security") == AgentType.VALIDATION
        
        # Test editor queries
        assert await self.adapter._determine_agent_type("Modify my VPC") == AgentType.EDITOR
        assert await self.adapter._determine_agent_type("Update configuration") == AgentType.EDITOR
        
        # Test default case
        assert await self.adapter._determine_agent_type("Random query") == AgentType.ANALYSIS

    @pytest.mark.asyncio
    async def test_execute_agent_subgraph(self):
        """Test agent subgraph execution."""
        query = "Create a VPC"
        task_id = "test-task"
        
        # Test each agent type
        for agent_type in AgentType:
            result = await self.adapter._execute_agent_subgraph(agent_type, query, task_id)
            
            assert isinstance(result, dict)
            assert "output" in result
            assert "execution_time" in result
            assert result["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_execute_supervisor_workflow(self):
        """Test supervisor workflow execution."""
        query = "Create a VPC"
        task_id = "test-task"
        context_id = "test-context"
        
        # Mock supervisor state
        mock_state = Mock()
        mock_state.current_step = "completed"
        self.mock_supervisor.supervisor_state = mock_state
        
        # Mock agent subgraph execution
        with patch.object(self.adapter, '_execute_agent_subgraph') as mock_execute:
            mock_execute.return_value = {
                "output": "Test result",
                "execution_time": 1.5
            }
            
            # Collect responses
            responses = []
            async for response in self.adapter._execute_supervisor_workflow(query, task_id, context_id):
                responses.append(response)
            
            # Verify responses
            assert len(responses) >= 2  # routing + completion
            
            # Check routing response
            routing_response = responses[0]
            assert "Routing to" in routing_response.content
            assert routing_response.metadata["status"] == "routing"
            
            # Check completion response
            completion_response = responses[1]
            assert "Successfully completed" in completion_response.content
            assert completion_response.is_task_complete
            assert completion_response.metadata["status"] == "completed"

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """Test error handling in stream method."""
        query = "Create a VPC"
        context_id = "test-context"
        task_id = "test-task"
        
        # Mock supervisor to raise an exception
        self.mock_supervisor.initialize_workflow.side_effect = Exception("Test error")
        
        # Collect responses
        responses = []
        async for response in self.adapter.stream(query, context_id, task_id):
            responses.append(response)
        
        # Verify error response
        assert len(responses) == 1
        error_response = responses[0]
        assert "Supervisor workflow failed" in error_response.content
        assert error_response.is_task_complete
        assert error_response.metadata["status"] == "failed"
        assert "Test error" in error_response.metadata["error"]

    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self):
        """Test initialize and cleanup methods."""
        # Test initialize
        await self.adapter.initialize()
        
        # Test cleanup
        await self.adapter.cleanup()
        
        # Verify cleanup actions
        self.mock_supervisor.reset_state.assert_called_once()
        assert len(self.adapter.active_workflows) == 0

    def test_get_workflow_progress(self):
        """Test workflow progress retrieval."""
        task_id = "test-task"
        
        # Test with no active workflow
        progress = self.adapter.get_workflow_progress(task_id)
        assert progress is None
        
        # Test with active workflow
        mock_state = Mock()
        self.adapter.active_workflows[task_id] = mock_state
        self.mock_supervisor.get_workflow_progress.return_value = {"status": "working"}
        
        progress = self.adapter.get_workflow_progress(task_id)
        assert progress == {"status": "working"}

    def test_get_active_workflows(self):
        """Test active workflows retrieval."""
        # Add some mock workflows
        mock_state1 = Mock()
        mock_state1.workflow_id = "workflow-1"
        mock_state2 = Mock()
        mock_state2.workflow_id = "workflow-2"
        
        self.adapter.active_workflows["task-1"] = mock_state1
        self.adapter.active_workflows["task-2"] = mock_state2
        
        active_workflows = self.adapter.get_active_workflows()
        assert active_workflows == {
            "task-1": "workflow-1",
            "task-2": "workflow-2"
        }

    def test_factory_function(self):
        """Test the factory function."""
        adapter = create_supervisor_agent_adapter(
            supervisor_agent=self.mock_supervisor,
            name="factory-test",
            config={"test": "config"}
        )
        
        assert isinstance(adapter, SupervisorAgentAdapter)
        assert adapter.name == "factory-test"
        assert adapter.config == {"test": "config"}


if __name__ == "__main__":
    # Run basic tests
    print("Testing SupervisorAgentAdapter...")
    
    # Create test instance
    mock_supervisor = Mock(spec=SupervisorAgent)
    mock_supervisor.name = "test-supervisor"
    mock_supervisor.is_ready.return_value = True
    
    adapter = SupervisorAgentAdapter(mock_supervisor, "test-adapter")
    
    # Test basic properties
    assert adapter.name == "test-adapter"
    print("✅ Basic properties test passed!")
    
    # Test agent type determination
    async def test_agent_types():
        assert await adapter._determine_agent_type("Create VPC") == AgentType.GENERATION
        assert await adapter._determine_agent_type("Analyze requirements") == AgentType.ANALYSIS
        print("✅ Agent type determination test passed!")
    
    asyncio.run(test_agent_types())
    
    print("\n🎉 All SupervisorAgentAdapter tests passed!") 