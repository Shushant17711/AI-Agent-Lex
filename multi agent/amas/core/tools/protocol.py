"""
Tool Request Protocol for AMAS.
This module defines the protocol for agents to request tool usage.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger(__name__)

class ToolRequestProtocol:
    """Defines the protocol for agents to request tool usage."""
    
    @staticmethod
    def create_request(tool_name: str, **params) -> Dict[str, Any]:
        """Create a standardized tool request.
        
        Args:
            tool_name: Name of the tool to request.
            **params: Parameters for the tool operation.
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return {
            "tool": tool_name,
            "params": params
        }
    
    @staticmethod
    def create_file_read_request(filename: str) -> Dict[str, Any]:
        """Create a standardized file read request.
        
        Args:
            filename: Path to the file to read.
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return ToolRequestProtocol.create_request("file_read", filename=filename)
    
    @staticmethod
    def create_file_write_request(filename: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """Create a standardized file write request.
        
        Args:
            filename: Path to the file to write.
            content: Content to write to the file.
            mode: File open mode ('w' for write, 'a' for append).
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return ToolRequestProtocol.create_request("file_write", filename=filename, content=content, mode=mode)
    
    @staticmethod
    def create_file_append_request(filename: str, content: str) -> Dict[str, Any]:
        """Create a standardized file append request.
        
        Args:
            filename: Path to the file to append to.
            content: Content to append to the file.
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return ToolRequestProtocol.create_request("file_append", filename=filename, content=content)
    
    @staticmethod
    def create_file_delete_request(filename: str) -> Dict[str, Any]:
        """Create a standardized file delete request.
        
        Args:
            filename: Path to the file to delete.
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return ToolRequestProtocol.create_request("file_delete", filename=filename)
    
    @staticmethod
    def create_file_exists_request(filename: str) -> Dict[str, Any]:
        """Create a standardized file exists request.
        
        Args:
            filename: Path to the file to check.
            
        Returns:
            Dictionary containing the standardized tool request.
        """
        return ToolRequestProtocol.create_request("file_exists", filename=filename)
    
    @staticmethod
    def format_for_agent_prompt(available_tools: Optional[List[str]] = None) -> str:
        """Format tool request protocol documentation for inclusion in agent prompts.
        
        Args:
            available_tools: Optional list of available tools to include in the documentation.
            
        Returns:
            Formatted string containing tool request protocol documentation.
        """
        if available_tools is None:
            available_tools = ["file_read", "file_write", "file_append", "file_delete", "file_exists"]
        
        docs = """
## Tool Request Protocol

To use tools, output a JSON object with the following format:

```json
{
  "tool": "tool_name",
  "params": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### Available Tools:
"""
        
        if "file_read" in available_tools:
            docs += """
#### file_read
Reads content from a file.

```json
{
  "tool": "file_read",
  "params": {
    "filename": "path/to/file.py"
  }
}
```
"""
        
        if "file_write" in available_tools:
            docs += """
#### file_write
Writes content to a file (overwrites existing content).

```json
{
  "tool": "file_write",
  "params": {
    "filename": "path/to/file.py",
    "content": "file content here",
    "mode": "w"  // Optional, defaults to "w"
  }
}
```
"""
        
        if "file_append" in available_tools:
            docs += """
#### file_append
Appends content to a file.

```json
{
  "tool": "file_append",
  "params": {
    "filename": "path/to/file.py",
    "content": "content to append"
  }
}
```
"""
        
        if "file_delete" in available_tools:
            docs += """
#### file_delete
Deletes a file.

```json
{
  "tool": "file_delete",
  "params": {
    "filename": "path/to/file.py"
  }
}
```
"""
        
        if "file_exists" in available_tools:
            docs += """
#### file_exists
Checks if a file exists.

```json
{
  "tool": "file_exists",
  "params": {
    "filename": "path/to/file.py"
  }
}
```
"""
        
        return docs
    
    @staticmethod
    def parse_agent_output(agent_output: str) -> Optional[Dict[str, Any]]:
        """Parse agent output to extract tool requests.
        
        This method looks for JSON-formatted tool requests in agent output.
        
        Args:
            agent_output: String output from an agent.
            
        Returns:
            Extracted tool request dictionary or None if no request found.
        """
        try:
            # Try to parse the entire output as JSON
            try:
                data = json.loads(agent_output)
                if "tool" in data and "params" in data:
                    return data
            except json.JSONDecodeError:
                pass
            
            # Look for JSON-like patterns in the text
            import re
            pattern = r'\{[\s\S]*?"tool"[\s\S]*?"params"[\s\S]*?\}'
            matches = re.findall(pattern, agent_output)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    if "tool" in data and "params" in data:
                        return data
                except json.JSONDecodeError:
                    continue
            
            return None
        except Exception as e:
            logger.error(f"Error parsing agent output: {str(e)}")
            return None
    
    @staticmethod
    def format_result_for_agent(result: Dict[str, Any]) -> str:
        """Format a tool execution result for agent consumption.
        
        Args:
            result: Dictionary containing the tool execution result.
            
        Returns:
            Formatted string containing the tool execution result.
        """
        try:
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error formatting result: {str(e)}")
            return str(result)
