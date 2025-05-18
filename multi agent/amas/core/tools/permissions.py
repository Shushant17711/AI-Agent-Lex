"""
Permissions management for AMAS tools.
This module provides permission management for tool operations.
"""

import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class PermissionsManager:
    """Manages permissions for tool usage in the multi-agent system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the permissions manager.
        
        Args:
            config: Optional configuration dictionary for permissions.
        """
        self.config = config or {}
        self.pre_approved_tools = set(self.config.get('pre_approved_tools', [
            'file_read', 
            'file_exists'
        ]))
        self.sensitive_tools = set(self.config.get('sensitive_tools', [
            'file_write',
            'file_append',
            'file_delete',
            'execute_command' # Added execute_command as sensitive by default
        ]))
        logger.info(f"PermissionsManager initialized with {len(self.pre_approved_tools)} pre-approved tools")
    
    def check_permission(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if a tool operation is permitted.
        
        Args:
            tool_name: Name of the tool to check.
            params: Parameters for the tool operation.
            
        Returns:
            True if the operation is permitted, False otherwise.
        """
        # Pre-approved tools don't need explicit permission
        if tool_name in self.pre_approved_tools:
            logger.info(f"Tool {tool_name} is pre-approved")
            return True
        
        # Check if this is a sensitive operation
        if self.is_sensitive_operation(tool_name, params):
            logger.info(f"Tool {tool_name} requires explicit permission")
            return False
        
        # Default to requiring permission for unknown tools
        logger.warning(f"Unknown tool {tool_name}, requiring permission")
        return False
    
    def request_permission(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Request user permission for a tool operation.
        
        In a real implementation, this would prompt the user for permission.
        For this implementation, we'll simulate user approval for demonstration.
        
        Args:
            tool_name: Name of the tool to request permission for.
            params: Parameters for the tool operation.
            
        Returns:
            True if permission is granted, False otherwise.
        """
        # In a real implementation, this would prompt the user
        # For now, we'll simulate user approval
        operation_desc = self._get_operation_description(tool_name, params)
        logger.info(f"Requesting permission for: {operation_desc}")
        
        # Simulate user approval (always approve for demonstration)
        # In a real implementation, this would wait for user input
        approved = True
        
        if approved:
            logger.info(f"Permission granted for: {operation_desc}")
        else:
            logger.info(f"Permission denied for: {operation_desc}")
        
        return approved
    
    def register_pre_approved_tool(self, tool_name: str) -> None:
        """Register a tool as pre-approved.
        
        Args:
            tool_name: Name of the tool to pre-approve.
        """
        self.pre_approved_tools.add(tool_name)
        logger.info(f"Tool {tool_name} registered as pre-approved")
    
    def is_sensitive_operation(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Determine if an operation is sensitive and requires permission.
        
        Args:
            tool_name: Name of the tool.
            params: Parameters for the tool operation.
            
        Returns:
            True if the operation is sensitive, False otherwise.
        """
        # Check if the tool is in the sensitive tools list
        if tool_name in self.sensitive_tools:
            return True
        
        # Additional checks could be added here
        # For example, checking if a file path is in a protected directory
        
        return False
    
    def _get_operation_description(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Get a human-readable description of an operation.
        
        Args:
            tool_name: Name of the tool.
            params: Parameters for the tool operation.
            
        Returns:
            Human-readable description of the operation.
        """
        if tool_name == 'file_write':
            filename = params.get('filename', 'unknown')
            content_length = len(params.get('content', ''))
            return f"Write {content_length} bytes to file '{filename}'"
        
        elif tool_name == 'file_append':
            filename = params.get('filename', 'unknown')
            content_length = len(params.get('content', ''))
            return f"Append {content_length} bytes to file '{filename}'"
        
        elif tool_name == 'file_delete':
            filename = params.get('filename', 'unknown')
            return f"Delete file '{filename}'"
        
        elif tool_name == 'file_read':
            filename = params.get('filename', 'unknown')
            return f"Read file '{filename}'"
        
        else:
            return f"{tool_name} with parameters {params}"
