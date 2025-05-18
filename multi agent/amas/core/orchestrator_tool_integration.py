"""
Orchestrator integration for AMAS Tool Execution Engine.
This module provides integration between the Orchestrator and Tool Execution Engine.
"""

import logging
import json
import datetime
from typing import Dict, Any, Optional, List, Union

from .tools.execution_engine import ToolExecutionEngine
from .tools.protocol import ToolRequestProtocol, ToolRequest

# Configure logging
logger = logging.getLogger(__name__)

class OrchestratorToolIntegration:
    """Integrates the Tool Execution Engine with the Orchestrator."""
    
    def __init__(self, tool_engine: Optional[ToolExecutionEngine] = None):
        """Initialize the integration.
        
        Args:
            tool_engine: Optional Tool Execution Engine instance.
        """
        self.tool_engine = tool_engine or ToolExecutionEngine()
        logger.info("Orchestrator Tool Integration initialized")
    
    def process_agent_output(self, agent_output: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent output to identify and execute tool requests.

        Args:
            agent_output: String output from an agent. Must be non-empty.
            context: Current context dictionary.

        Returns:
            A new context dictionary updated with tool results, or the original context
            if no tool request is found or input is invalid.
        """
        # Input Validation
        if not agent_output:
            logger.warning("Received empty agent output. Skipping tool processing.")
            return context

        # Create a copy of the context to maintain immutability
        new_context = context.copy()

        # Parse agent output for tool requests
        tool_request: Optional[ToolRequest] = ToolRequestProtocol.parse_agent_output(agent_output)

        if tool_request:
            logger.info(f"Tool request identified: {tool_request}")

            # Execute the tool request with error handling
            try:
                result = self.tool_engine.execute_tool(tool_request)
            except Exception as e:
                logger.error(f"Tool execution failed for request {tool_request}: {str(e)}", exc_info=True)
                result = {"status": "error", "error": f"Tool execution failed: {str(e)}"}

            # Update context with the result
            self._update_context_with_result(new_context, tool_request, result)

            # Format the result for agent consumption
            formatted_result = ToolRequestProtocol.format_result_for_agent(result)
            if logger.isEnabledFor(logging.DEBUG):
                 # Use json.dumps for potentially complex results
                 try:
                     result_json = json.dumps(result)
                     logger.debug(f"Tool execution result details: {result_json}")
                 except TypeError: # Handle non-serializable data if necessary
                     logger.debug(f"Tool execution result (non-serializable): {result}")

            logger.info(f"Tool execution finished. Formatted result: {formatted_result}")

            return new_context
        else:
            logger.info("No tool request identified in agent output")
            return new_context # Return the original context if no request found
    
    def _update_context_with_result(self, context: Dict[str, Any],
                                   request: ToolRequest, # Use the specific type
                                   result: Dict[str, Any]) -> None:
        """Update the context with tool execution result.
        
        Args:
            context: Current context dictionary.
            request: Tool request dictionary.
            result: Tool execution result dictionary.
        """
        # Initialize tool results in context if not present
        if "tool_results" not in context:
            context["tool_results"] = []
        
        # Add the result to the context
        context["tool_results"].append({
            "request": request,
            "result": result,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        
        # For file operations, update file-specific context
        tool_name = request.get("tool", "")
        params = request.get("params", {})
        
        if tool_name == "file_read" and result.get("status") == "success":
            filename = params.get("filename", "")
            if "file_contents" not in context:
                context["file_contents"] = {}
            context["file_contents"][filename] = result.get("content", "")
        
        elif tool_name in ["file_write", "file_append"] and result.get("status") == "success":
            filename = params.get("filename", "")
            if "modified_files" not in context:
                context["modified_files"] = []
            if filename not in context["modified_files"]:
                context["modified_files"].append(filename)
    
    def get_tool_documentation_for_agent(self, agent_type: str) -> str:
        """Get tool documentation formatted for inclusion in agent prompts.

        Retrieves the names of tools available to the specified agent type
        and formats their documentation (schema, description) as a string suitable
        for inclusion in the agent's system prompt or context.

        Args:
            agent_type: Type of agent (e.g., "planner", "coder").

        Returns:
            A formatted string containing the documentation for available tools,
            or an empty string if no tools are available or the agent type is unknown.
        """
        # Define available tools based on agent type
        available_tools = self._get_available_tools_for_agent(agent_type)
        
        # Get formatted documentation
        return ToolRequestProtocol.format_for_agent_prompt(available_tools)
    
    # Consider moving this mapping to a configuration file or class variable for easier management
    _AGENT_TOOL_MAPPING: Dict[str, List[str]] = {
        "planner": ["file_read", "file_exists"],
        "coder": ["file_read", "file_write", "file_append", "file_delete", "file_exists"],
        "tester": ["file_read", "file_write", "file_append", "file_exists"],
        # Add other agent types and their tools here
    }
    _DEFAULT_TOOLS: List[str] = ["file_read", "file_exists"] # Default for unknown types

    def _get_available_tools_for_agent(self, agent_type: str) -> List[str]:
        """Get list of available tools for a specific agent type using a mapping.

        Args:
            agent_type: Type of agent (e.g., "planner", "coder"). Case-insensitive.

        Returns:
            List of available tool names based on the mapping, or default tools if type is unknown.
        """
        return self._AGENT_TOOL_MAPPING.get(agent_type.lower(), self._DEFAULT_TOOLS)
