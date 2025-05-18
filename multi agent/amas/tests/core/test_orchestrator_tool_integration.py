import unittest
from unittest.mock import patch, Mock, MagicMock, call
import datetime
import json
from typing import Dict, Any, Optional

# Assuming the amas package is in the python path or installed
from amas.core.orchestrator_tool_integration import OrchestratorToolIntegration
from amas.core.tools.execution_engine import ToolExecutionEngine
from amas.core.tools.protocol import ToolRequestProtocol, ToolRequest

# Mock the datetime object to control timestamps
class MockDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        # Return a fixed UTC datetime for consistent testing
        return datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

class TestOrchestratorToolIntegration(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tool_engine = Mock(spec=ToolExecutionEngine)
        self.integration_with_mock_engine = OrchestratorToolIntegration(tool_engine=self.mock_tool_engine)
        self.integration_default_engine = OrchestratorToolIntegration() # Test default engine creation
        self.initial_context = {"user_request": "test request", "history": []}
        self.fixed_timestamp = "2024-01-01T12:00:00+00:00" # Expected timestamp from MockDateTime

    @patch('amas.core.orchestrator_tool_integration.ToolExecutionEngine')
    def test_initialization_default_engine(self, MockToolEngine):
        """Test initialization uses a default ToolExecutionEngine if none provided."""
        integration = OrchestratorToolIntegration()
        self.assertIsInstance(integration.tool_engine, MockToolEngine)
        MockToolEngine.assert_called_once()

    def test_initialization_provided_engine(self):
        """Test initialization with a provided ToolExecutionEngine."""
        integration = OrchestratorToolIntegration(tool_engine=self.mock_tool_engine)
        self.assertEqual(integration.tool_engine, self.mock_tool_engine)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    @patch('amas.core.orchestrator_tool_integration.datetime', MockDateTime)
    def test_process_agent_output_valid_tool_request(self, MockToolRequestProtocol):
        """Test processing valid tool requests and context updates."""
        agent_output = "<tool_request><tool>file_read</tool><params><filename>test.txt</filename></params></tool_request>"
        mock_request: ToolRequest = {"tool": "file_read", "params": {"filename": "test.txt"}}
        mock_result = {"status": "success", "content": "file content"}
        formatted_result_for_agent = "<tool_result>...</tool_result>" # Simplified

        MockToolRequestProtocol.parse_agent_output.return_value = mock_request
        MockToolRequestProtocol.format_result_for_agent.return_value = formatted_result_for_agent
        self.mock_tool_engine.execute_tool.return_value = mock_result

        original_context = self.initial_context.copy()
        new_context = self.integration_with_mock_engine.process_agent_output(agent_output, original_context)

        # Verify mocks were called
        MockToolRequestProtocol.parse_agent_output.assert_called_once_with(agent_output)
        self.mock_tool_engine.execute_tool.assert_called_once_with(mock_request)
        MockToolRequestProtocol.format_result_for_agent.assert_called_once_with(mock_result)

        # Verify context update
        self.assertNotEqual(id(new_context), id(original_context), "Context should be a new dictionary (immutability)")
        self.assertEqual(original_context, self.initial_context, "Original context should not be modified") # Immutability check
        self.assertIn("tool_results", new_context)
        self.assertEqual(len(new_context["tool_results"]), 1)
        expected_tool_result_entry = {
            "request": mock_request,
            "result": mock_result,
            "timestamp": self.fixed_timestamp
        }
        self.assertEqual(new_context["tool_results"][0], expected_tool_result_entry)

        # Verify file content update for file_read
        self.assertIn("file_contents", new_context)
        self.assertEqual(new_context["file_contents"], {"test.txt": "file content"})

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_process_agent_output_no_tool_request(self, MockToolRequestProtocol):
        """Test processing output without a tool request."""
        agent_output = "This is just a regular message."
        MockToolRequestProtocol.parse_agent_output.return_value = None

        original_context = self.initial_context.copy()
        new_context = self.integration_with_mock_engine.process_agent_output(agent_output, original_context)

        MockToolRequestProtocol.parse_agent_output.assert_called_once_with(agent_output)
        self.mock_tool_engine.execute_tool.assert_not_called()
        # Check that the context returned is the *new* copy, even if unchanged content-wise beyond the copy
        self.assertNotEqual(id(new_context), id(original_context))
        # Check content equality (no tool results added)
        self.assertEqual(new_context, original_context)


    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    @patch('amas.core.orchestrator_tool_integration.datetime', MockDateTime)
    @patch('amas.core.orchestrator_tool_integration.logger')
    def test_process_agent_output_tool_execution_error(self, mock_logger, MockToolRequestProtocol):
        """Test handling of errors during tool execution."""
        agent_output = "<tool_request><tool>error_tool</tool><params></params></tool_request>"
        mock_request: ToolRequest = {"tool": "error_tool", "params": {}}
        error_message = "Something went wrong"
        expected_error_result = {"status": "error", "error": f"Tool execution failed: {error_message}"}
        formatted_result_for_agent = "<tool_result status='error'>...</tool_result>" # Simplified

        MockToolRequestProtocol.parse_agent_output.return_value = mock_request
        MockToolRequestProtocol.format_result_for_agent.return_value = formatted_result_for_agent
        self.mock_tool_engine.execute_tool.side_effect = Exception(error_message)

        original_context = self.initial_context.copy()
        new_context = self.integration_with_mock_engine.process_agent_output(agent_output, original_context)

        # Verify mocks
        MockToolRequestProtocol.parse_agent_output.assert_called_once_with(agent_output)
        self.mock_tool_engine.execute_tool.assert_called_once_with(mock_request)
        MockToolRequestProtocol.format_result_for_agent.assert_called_once_with(expected_error_result)
        mock_logger.error.assert_called_once() # Check that error was logged

        # Verify context update with error
        self.assertNotEqual(id(new_context), id(original_context))
        self.assertEqual(original_context, self.initial_context) # Immutability
        self.assertIn("tool_results", new_context)
        self.assertEqual(len(new_context["tool_results"]), 1)
        expected_tool_result_entry = {
            "request": mock_request,
            "result": expected_error_result,
            "timestamp": self.fixed_timestamp
        }
        self.assertEqual(new_context["tool_results"][0], expected_tool_result_entry)

    @patch('amas.core.orchestrator_tool_integration.logger')
    def test_process_agent_output_empty_input(self, mock_logger):
        """Test processing empty agent output."""
        agent_output = ""
        original_context = self.initial_context.copy()
        new_context = self.integration_with_mock_engine.process_agent_output(agent_output, original_context)

        # Verify no tool processing occurred
        self.mock_tool_engine.execute_tool.assert_not_called()
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Received empty agent output. Skipping tool processing.")
        # Verify original context is returned (by identity in this specific case as per implementation)
        self.assertEqual(id(new_context), id(original_context))
        self.assertEqual(new_context, self.initial_context)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    @patch('amas.core.orchestrator_tool_integration.datetime', MockDateTime)
    def test_context_update_file_write(self, MockToolRequestProtocol):
        """Test context update for successful file_write operation."""
        agent_output = "<tool_request><tool>file_write</tool><params><filename>output.log</filename><content>data</content></params></tool_request>"
        mock_request: ToolRequest = {"tool": "file_write", "params": {"filename": "output.log", "content": "data"}}
        mock_result = {"status": "success"}
        formatted_result_for_agent = "<tool_result status='success'></tool_result>"

        MockToolRequestProtocol.parse_agent_output.return_value = mock_request
        MockToolRequestProtocol.format_result_for_agent.return_value = formatted_result_for_agent
        self.mock_tool_engine.execute_tool.return_value = mock_result

        original_context = self.initial_context.copy()
        new_context = self.integration_with_mock_engine.process_agent_output(agent_output, original_context)

        self.assertNotEqual(id(new_context), id(original_context))
        self.assertEqual(original_context, self.initial_context) # Immutability

        # Verify tool_results update
        self.assertIn("tool_results", new_context)
        self.assertEqual(len(new_context["tool_results"]), 1)
        self.assertEqual(new_context["tool_results"][0]["request"], mock_request)
        self.assertEqual(new_context["tool_results"][0]["result"], mock_result)
        self.assertEqual(new_context["tool_results"][0]["timestamp"], self.fixed_timestamp)

        # Verify modified_files update
        self.assertIn("modified_files", new_context)
        self.assertEqual(new_context["modified_files"], ["output.log"])
        # Ensure file_contents is not incorrectly added for write
        self.assertNotIn("file_contents", new_context)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_get_tool_documentation_for_agent_planner(self, MockToolRequestProtocol):
        """Test getting tool documentation for 'planner' agent."""
        expected_tools = ["file_read", "file_exists"]
        expected_docs = "Formatted docs for planner"
        MockToolRequestProtocol.format_for_agent_prompt.return_value = expected_docs

        docs = self.integration_with_mock_engine.get_tool_documentation_for_agent("planner")

        MockToolRequestProtocol.format_for_agent_prompt.assert_called_once_with(expected_tools)
        self.assertEqual(docs, expected_docs)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_get_tool_documentation_for_agent_coder(self, MockToolRequestProtocol):
        """Test getting tool documentation for 'coder' agent."""
        expected_tools = ["file_read", "file_write", "file_append", "file_delete", "file_exists"]
        expected_docs = "Formatted docs for coder"
        MockToolRequestProtocol.format_for_agent_prompt.return_value = expected_docs

        docs = self.integration_with_mock_engine.get_tool_documentation_for_agent("coder")

        MockToolRequestProtocol.format_for_agent_prompt.assert_called_once_with(expected_tools)
        self.assertEqual(docs, expected_docs)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_get_tool_documentation_for_agent_tester(self, MockToolRequestProtocol):
        """Test getting tool documentation for 'tester' agent."""
        expected_tools = ["file_read", "file_write", "file_append", "file_exists"]
        expected_docs = "Formatted docs for tester"
        MockToolRequestProtocol.format_for_agent_prompt.return_value = expected_docs

        docs = self.integration_with_mock_engine.get_tool_documentation_for_agent("tester")

        MockToolRequestProtocol.format_for_agent_prompt.assert_called_once_with(expected_tools)
        self.assertEqual(docs, expected_docs)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_get_tool_documentation_for_agent_unknown(self, MockToolRequestProtocol):
        """Test getting tool documentation for an unknown agent type (should use default)."""
        expected_tools = ["file_read", "file_exists"] # Default tools
        expected_docs = "Formatted docs for default"
        MockToolRequestProtocol.format_for_agent_prompt.return_value = expected_docs

        docs = self.integration_with_mock_engine.get_tool_documentation_for_agent("researcher") # Unknown type

        MockToolRequestProtocol.format_for_agent_prompt.assert_called_once_with(expected_tools)
        self.assertEqual(docs, expected_docs)

    @patch('amas.core.orchestrator_tool_integration.ToolRequestProtocol')
    def test_get_tool_documentation_case_insensitive(self, MockToolRequestProtocol):
        """Test agent type matching is case-insensitive."""
        expected_tools = ["file_read", "file_write", "file_append", "file_delete", "file_exists"]
        expected_docs = "Formatted docs for coder"
        MockToolRequestProtocol.format_for_agent_prompt.return_value = expected_docs

        docs = self.integration_with_mock_engine.get_tool_documentation_for_agent("CoDeR") # Mixed case

        MockToolRequestProtocol.format_for_agent_prompt.assert_called_once_with(expected_tools)
        self.assertEqual(docs, expected_docs)


if __name__ == '__main__':
    unittest.main()