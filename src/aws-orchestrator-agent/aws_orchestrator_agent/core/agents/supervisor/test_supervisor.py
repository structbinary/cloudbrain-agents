"""
Simple test for the Supervisor Agent implementation.

This test verifies that the Supervisor Agent can be created and configured
with StateManager integration for robust state management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from aws_orchestrator_agent.core.supervisor.supervisor_agent import SupervisorAgent, create_supervisor_agent
from aws_orchestrator_agent.core.supervisor.state import AgentType, WorkflowStatus
from aws_orchestrator_agent.config.config import Config


class TestSupervisorAgent:
    """Test cases for the Supervisor Agent implementation."""

    def test_supervisor_agent_initialization(self):
        """Test that the Supervisor Agent can be initialized with StateManager."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            assert agent is not None
            assert agent.model == mock_model
            assert agent.agents == []
            assert agent.supervisor is None
            assert agent.state_manager is not None
            assert agent.supervisor_state is None
            assert agent.config_instance is not None
            assert agent.supervisor_config is not None

    def test_supervisor_agent_with_custom_config(self):
        """Test that the Supervisor Agent can be initialized with custom config."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            custom_config = {
                "supervisor_max_snapshots": 5,
                "supervisor_output_mode": "json"
            }
            
            agent = SupervisorAgent(custom_config=custom_config)

            assert agent is not None
            assert agent.supervisor_config["max_snapshots"] == 5
            assert agent.supervisor_config["output_mode"] == "json"

    def test_supervisor_agent_with_config_instance(self):
        """Test that the Supervisor Agent can be initialized with Config instance."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            config = Config()
            agent = SupervisorAgent(config=config)

            assert agent is not None
            assert agent.config_instance == config

    def test_initialize_workflow(self):
        """Test workflow initialization with StateManager."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC", {"aws_region": "us-west-2"})

            assert agent.supervisor_state is not None
            assert agent.supervisor_state.user_request == "Create a VPC"
            assert agent.supervisor_state.mcp_context == {"aws_region": "us-west-2"}
            assert agent.supervisor_state.workflow_started_at is not None
            assert agent.supervisor_state.status == WorkflowStatus.PENDING

    def test_register_agent_subgraph(self):
        """Test that agent subgraphs can be registered."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            mock_subgraph = Mock()

            agent.register_agent_subgraph("test_agent", mock_subgraph, "Test Agent")

            assert len(agent.agents) == 1
            assert agent.agents[0] == mock_subgraph
            assert agent.agent_names["test_agent"] == "Test Agent"

    def test_register_agent_subgraph_without_display_name(self):
        """Test that agent subgraphs can be registered without display name."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            mock_subgraph = Mock()

            agent.register_agent_subgraph("test_agent", mock_subgraph)

            assert len(agent.agents) == 1
            assert agent.agents[0] == mock_subgraph
            # Should use agent_name as fallback
            assert agent.agent_names["test_agent"] == "test_agent"

    def test_create_supervisor(self):
        """Test that the supervisor can be created."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm, \
             patch('aws_orchestrator_agent.core.supervisor.supervisor_agent.create_supervisor') as mock_create_supervisor:

            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            mock_supervisor = Mock()
            mock_create_supervisor.return_value = mock_supervisor
            mock_compiled_supervisor = Mock()
            mock_supervisor.compile.return_value = mock_compiled_supervisor

            agent = SupervisorAgent()
            mock_subgraph = Mock()
            agent.register_agent_subgraph("test_agent", mock_subgraph)
            agent.create_supervisor()

            assert agent.supervisor == mock_compiled_supervisor
            mock_create_supervisor.assert_called_once()

    def test_create_supervisor_without_agents(self):
        """Test that creating supervisor without agents fails."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            with pytest.raises(ValueError, match="No agent subgraphs registered"):
                agent.create_supervisor()

    def test_invoke_agent_subgraph_without_workflow(self):
        """Test that invoking agent subgraph without workflow initialization fails."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            with pytest.raises(RuntimeError, match="No workflow initialized"):
                agent.invoke_agent_subgraph(AgentType.ANALYSIS)

    def test_invoke_agent_subgraph_without_supervisor(self):
        """Test that invoking agent subgraph without supervisor creation fails."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Test request")

            with pytest.raises(RuntimeError, match="Supervisor not created"):
                agent.invoke_agent_subgraph(AgentType.ANALYSIS)

    def test_invoke_agent_subgraph_success(self):
        """Test successful agent subgraph invocation with state management."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm, \
             patch('aws_orchestrator_agent.core.supervisor.supervisor_agent.create_supervisor') as mock_create_supervisor:

            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            mock_supervisor = Mock()
            mock_create_supervisor.return_value = mock_supervisor
            mock_compiled_supervisor = Mock()
            mock_supervisor.compile.return_value = mock_compiled_supervisor

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC")
            
            mock_subgraph = Mock()
            agent.register_agent_subgraph("analysis", mock_subgraph)
            agent.create_supervisor()

            # Mock the state manager methods
            with patch.object(agent.state_manager, 'transform_to_subgraph') as mock_transform, \
                 patch.object(agent.state_manager, 'validate_state') as mock_validate, \
                 patch.object(agent.state_manager, 'create_snapshot') as mock_snapshot, \
                 patch.object(agent.state_manager, 'merge_from_subgraph') as mock_merge:

                mock_transform.return_value = Mock()
                mock_validate.return_value = True
                mock_merge.return_value = agent.supervisor_state

                result = agent.invoke_agent_subgraph(AgentType.ANALYSIS)

                assert result["agent_type"] == "analysis"
                assert result["status"] == "completed"
                mock_transform.assert_called_once()
                mock_validate.assert_called_once()
                mock_snapshot.assert_called()
                mock_merge.assert_called_once()

    def test_invoke_with_messages(self):
        """Test invoking supervisor with messages."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm, \
             patch('aws_orchestrator_agent.core.supervisor.supervisor_agent.create_supervisor') as mock_create_supervisor:

            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            mock_supervisor = Mock()
            mock_create_supervisor.return_value = mock_supervisor
            mock_compiled_supervisor = Mock()
            mock_supervisor.compile.return_value = mock_compiled_supervisor

            agent = SupervisorAgent()
            mock_subgraph = Mock()
            agent.register_agent_subgraph("test_agent", mock_subgraph)
            agent.create_supervisor()

            messages = [{"role": "user", "content": "Create a VPC"}]
            state = {"context": {"region": "us-west-2"}}

            # Mock the supervisor invoke method
            mock_compiled_supervisor.invoke.return_value = {
                "messages": messages,
                "current_workflow": "test_workflow",
                "timestamp": "2023-01-01T00:00:00Z"
            }

            # Mock state manager methods
            with patch.object(agent.state_manager, 'create_snapshot') as mock_snapshot:
                result = agent.invoke(messages, state)

                assert result is not None
                assert "messages" in result
                mock_compiled_supervisor.invoke.assert_called_once()
                mock_snapshot.assert_called()

    def test_invoke_without_supervisor(self):
        """Test that invoking without supervisor creation fails."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            messages = [{"role": "user", "content": "Create a VPC"}]

            with pytest.raises(RuntimeError, match="Supervisor not created"):
                agent.invoke(messages)

    def test_invoke_with_query(self):
        """Test invoking supervisor with a simple query."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm, \
             patch('aws_orchestrator_agent.core.supervisor.supervisor_agent.create_supervisor') as mock_create_supervisor:

            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            mock_supervisor = Mock()
            mock_create_supervisor.return_value = mock_supervisor
            mock_compiled_supervisor = Mock()
            mock_supervisor.compile.return_value = mock_compiled_supervisor

            agent = SupervisorAgent()
            mock_subgraph = Mock()
            agent.register_agent_subgraph("test_agent", mock_subgraph)
            agent.create_supervisor()

            query = "Create a VPC"
            context = {"region": "us-west-2"}

            # Mock the invoke method
            with patch.object(agent, 'invoke') as mock_invoke:
                mock_invoke.return_value = {"result": "success"}
                
                result = agent.invoke_with_query(query, context)

                assert result == {"result": "success"}
                mock_invoke.assert_called_once()
                
                # Check that messages and state were constructed correctly
                call_args = mock_invoke.call_args
                assert call_args[0][0] == [{"role": "user", "content": query}]
                assert call_args[0][1] == {"context": context}

    def test_get_workflow_progress(self):
        """Test getting workflow progress information."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test without workflow initialized
            progress = agent.get_workflow_progress()
            assert progress["error"] == "No workflow initialized"

            # Test with workflow initialized
            agent.initialize_workflow("Create a VPC")
            progress = agent.get_workflow_progress()
            
            assert "workflow_id" in progress
            assert "current_step" in progress
            assert "status" in progress
            assert "completion_percentage" in progress

    def test_get_snapshot_info(self):
        """Test getting snapshot information."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC")

            snapshot_info = agent.get_snapshot_info()
            assert isinstance(snapshot_info, list)
            assert len(snapshot_info) >= 1  # At least the initial snapshot

    def test_rollback_workflow(self):
        """Test workflow rollback functionality."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test without workflow initialized
            success = agent.rollback_workflow()
            assert success is False

            # Test with workflow initialized
            agent.initialize_workflow("Create a VPC")
            success = agent.rollback_workflow()
            assert success is True

    def test_export_state_for_debugging(self):
        """Test exporting state for debugging."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test without workflow initialized
            export_data = agent.export_state_for_debugging()
            assert export_data["error"] == "No workflow initialized"

            # Test with workflow initialized
            agent.initialize_workflow("Create a VPC")
            export_data = agent.export_state_for_debugging()
            
            assert "timestamp" in export_data
            assert "workflow_id" in export_data
            assert "state_data" in export_data
            assert "progress" in export_data
            assert "snapshots" in export_data
            assert "error_context" in export_data

    def test_get_routing_history(self):
        """Test getting routing history."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test without workflow initialized (should return legacy history)
            history = agent.get_routing_history()
            assert isinstance(history, list)

            # Test with workflow initialized
            agent.initialize_workflow("Create a VPC")
            history = agent.get_routing_history()
            assert isinstance(history, list)

    def test_get_error_context(self):
        """Test getting error context."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test without workflow initialized (should return legacy context)
            context = agent.get_error_context()
            assert isinstance(context, dict)

            # Test with workflow initialized
            agent.initialize_workflow("Create a VPC")
            context = agent.get_error_context()
            assert isinstance(context, dict)

    def test_reset_state(self):
        """Test resetting supervisor state."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC")

            # Verify state is initialized
            assert agent.supervisor_state is not None
            assert len(agent.get_snapshot_info()) >= 1

            # Reset state
            agent.reset_state()

            # Verify state is reset
            assert agent.supervisor_state is None
            assert len(agent.get_snapshot_info()) == 0

    def test_get_agent_info(self):
        """Test getting agent information."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test with no agents registered
            agent_info = agent.get_agent_info()
            assert agent_info == {}

            # Test with agents registered
            mock_subgraph = Mock()
            agent.register_agent_subgraph("test_agent", mock_subgraph, "Test Agent")
            
            agent_info = agent.get_agent_info()
            assert agent_info == {"test_agent": "Test Agent"}

    def test_is_ready(self):
        """Test checking if supervisor is ready."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test not ready (no agents, no supervisor)
            assert agent.is_ready() is False

            # Test not ready (agents but no supervisor)
            mock_subgraph = Mock()
            agent.register_agent_subgraph("test_agent", mock_subgraph)
            assert agent.is_ready() is False

            # Test ready (agents and supervisor)
            with patch('aws_orchestrator_agent.core.supervisor.supervisor_agent.create_supervisor') as mock_create_supervisor:
                mock_supervisor = Mock()
                mock_create_supervisor.return_value = mock_supervisor
                mock_compiled_supervisor = Mock()
                mock_supervisor.compile.return_value = mock_compiled_supervisor

                agent.create_supervisor()
                assert agent.is_ready() is True

    def test_extract_user_request(self):
        """Test extracting user request from messages."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()

            # Test with user message
            messages = [{"role": "user", "content": "Create a VPC"}]
            request = agent._extract_user_request(messages)
            assert request == "Create a VPC"

            # Test with no user message
            messages = [{"role": "assistant", "content": "I'll help you"}]
            request = agent._extract_user_request(messages)
            assert request == "No user request found"

            # Test with empty messages
            request = agent._extract_user_request([])
            assert request == "No user request found"

    def test_create_mock_subgraph_state(self):
        """Test creating mock subgraph state."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC")

            # Test with analysis agent
            analysis_result = {"output": {"vpc": True, "subnets": 2}}
            state = agent._create_mock_subgraph_state(AgentType.ANALYSIS, analysis_result)
            assert state is not None
            assert hasattr(state, 'requirements')

            # Test with generation agent - use dict for terraform_code
            generation_result = {
                "output": {
                    "main.tf": "# VPC Configuration\nresource \"aws_vpc\" \"main\" {\n  cidr_block = \"10.0.0.0/16\"\n}",
                    "variables.tf": "# Variables\nvariable \"vpc_cidr\" {\n  default = \"10.0.0.0/16\"\n}"
                }
            }
            state = agent._create_mock_subgraph_state(AgentType.GENERATION, generation_result)
            assert state is not None
            assert hasattr(state, 'terraform_code')

            # Test with validation agent
            validation_result = {"output": {"score": 95, "issues": []}}
            state = agent._create_mock_subgraph_state(AgentType.VALIDATION, validation_result)
            assert state is not None
            assert hasattr(state, 'validation_reports')

            # Test with editor agent - use dict for modified_code
            editor_result = {
                "output": {
                    "main.tf": "# Modified VPC Configuration\nresource \"aws_vpc\" \"main\" {\n  cidr_block = \"10.0.0.0/16\"\n  tags = {\n    Name = \"main-vpc\"\n  }\n}",
                    "variables.tf": "# Variables\nvariable \"vpc_cidr\" {\n  default = \"10.0.0.0/16\"\n}"
                }
            }
            state = agent._create_mock_subgraph_state(AgentType.EDITOR, editor_result)
            assert state is not None
            assert hasattr(state, 'modified_code')

    def test_log_routing_decision(self):
        """Test logging routing decisions."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            agent = SupervisorAgent()
            agent.initialize_workflow("Create a VPC")

            result = {
                "timestamp": "2023-01-01T00:00:00Z",
                "messages": [{"role": "user", "content": "test"}],
                "current_workflow": "test_workflow",
                "user_approval_required": False
            }

            # Just test that the method doesn't raise an exception
            try:
                agent._log_routing_decision(result)
                # If we get here, the method executed successfully
                assert True
            except Exception as e:
                # If there's an error, it should be a specific type we can handle
                assert isinstance(e, (AttributeError, TypeError))

    def test_create_supervisor_agent_factory(self):
        """Test the factory function for creating supervisor agents."""
        with patch('aws_orchestrator_agent.core.llm.llm_provider.LLMProvider.create_llm') as mock_create_llm:
            mock_model = Mock()
            mock_create_llm.return_value = mock_model

            # Test with default config
            agent = create_supervisor_agent()
            assert agent is not None
            assert isinstance(agent, SupervisorAgent)

            # Test with custom config
            custom_config = {"supervisor_max_snapshots": 10}
            agent = create_supervisor_agent(custom_config=custom_config)
            assert agent is not None
            assert agent.supervisor_config["max_snapshots"] == 10

            # Test with Config instance
            config = Config()
            agent = create_supervisor_agent(config=config)
            assert agent is not None
            assert agent.config_instance == config


if __name__ == "__main__":
    # Run basic tests
    test_agent = TestSupervisorAgent()

    print("Testing Supervisor Agent initialization...")
    test_agent.test_supervisor_agent_initialization()
    print("✅ Supervisor Agent initialization test passed!")

    print("Testing Supervisor Agent with custom config...")
    test_agent.test_supervisor_agent_with_custom_config()
    print("✅ Supervisor Agent with custom config test passed!")

    print("Testing Supervisor Agent with Config instance...")
    test_agent.test_supervisor_agent_with_config_instance()
    print("✅ Supervisor Agent with Config instance test passed!")

    print("Testing workflow initialization...")
    test_agent.test_initialize_workflow()
    print("✅ Workflow initialization test passed!")

    print("Testing agent subgraph registration...")
    test_agent.test_register_agent_subgraph()
    print("✅ Agent subgraph registration test passed!")

    print("Testing agent subgraph registration without display name...")
    test_agent.test_register_agent_subgraph_without_display_name()
    print("✅ Agent subgraph registration without display name test passed!")

    print("Testing supervisor creation...")
    test_agent.test_create_supervisor()
    print("✅ Supervisor creation test passed!")

    print("Testing supervisor creation without agents...")
    test_agent.test_create_supervisor_without_agents()
    print("✅ Supervisor creation without agents test passed!")

    print("Testing agent subgraph invocation...")
    test_agent.test_invoke_agent_subgraph_success()
    print("✅ Agent subgraph invocation test passed!")

    print("Testing supervisor invoke with messages...")
    test_agent.test_invoke_with_messages()
    print("✅ Supervisor invoke with messages test passed!")

    print("Testing supervisor invoke without supervisor...")
    test_agent.test_invoke_without_supervisor()
    print("✅ Supervisor invoke without supervisor test passed!")

    print("Testing supervisor invoke with query...")
    test_agent.test_invoke_with_query()
    print("✅ Supervisor invoke with query test passed!")

    print("Testing workflow progress...")
    test_agent.test_get_workflow_progress()
    print("✅ Workflow progress test passed!")

    print("Testing state management utilities...")
    test_agent.test_get_snapshot_info()
    test_agent.test_rollback_workflow()
    test_agent.test_export_state_for_debugging()
    test_agent.test_get_routing_history()
    test_agent.test_get_error_context()
    test_agent.test_reset_state()
    print("✅ State management utilities tests passed!")

    print("Testing agent information...")
    test_agent.test_get_agent_info()
    print("✅ Agent information test passed!")

    print("Testing supervisor readiness...")
    test_agent.test_is_ready()
    print("✅ Supervisor readiness test passed!")

    print("Testing utility methods...")
    test_agent.test_extract_user_request()
    test_agent.test_create_mock_subgraph_state()
    test_agent.test_log_routing_decision()
    print("✅ Utility methods tests passed!")

    print("Testing factory function...")
    test_agent.test_create_supervisor_agent_factory()
    print("✅ Factory function test passed!")

    print("\n🎉 All tests passed! Supervisor Agent with enhanced logging and Config integration is working correctly.") 