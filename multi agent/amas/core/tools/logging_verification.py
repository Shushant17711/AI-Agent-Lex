"""
Logging and verification utilities for AMAS.
This module provides enhanced logging and verification for tool operations.
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ToolLogger:
    """Enhanced logging for tool operations."""
    
    def __init__(self, log_dir: Optional[str] = None, log_to_file: bool = True):
        """Initialize the tool logger.
        
        Args:
            log_dir: Directory to store log files. If None, uses 'logs' in current directory.
            log_to_file: Whether to log to file in addition to console.
        """
        self.logger = logging.getLogger("amas.tools")
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self.log_to_file = log_to_file
        
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create a file handler for the log file
            log_file = os.path.join(self.log_dir, f'tool_operations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Add the file handler to the logger
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Tool Logger initialized with log directory: {self.log_dir}")
    
    def log_tool_request(self, tool_request: Dict[str, Any], agent_name: Optional[str] = None) -> None:
        """Log a tool request.
        
        Args:
            tool_request: The tool request dictionary.
            agent_name: Optional name of the agent making the request.
        """
        tool_name = tool_request.get("tool", "unknown")
        params = tool_request.get("params", {})
        
        agent_info = f" from agent '{agent_name}'" if agent_name else ""
        self.logger.info(f"Tool request{agent_info}: {tool_name} with params: {json.dumps(params)}")
    
    def log_tool_execution(self, tool_request: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log a tool execution.
        
        Args:
            tool_request: The tool request dictionary.
            result: The result of the tool execution.
        """
        tool_name = tool_request.get("tool", "unknown")
        status = result.get("status", "unknown")
        
        if status == "success":
            self.logger.info(f"Tool execution successful: {tool_name}")
            
            # Log additional details for specific tools
            if tool_name == "file_write" or tool_name == "file_append":
                filename = tool_request.get("params", {}).get("filename", "unknown")
                content_length = len(tool_request.get("params", {}).get("content", ""))
                self.logger.info(f"  Wrote {content_length} bytes to file: {filename}")
            
            elif tool_name == "file_read":
                filename = tool_request.get("params", {}).get("filename", "unknown")
                content_length = len(result.get("content", ""))
                self.logger.info(f"  Read {content_length} bytes from file: {filename}")
        else:
            self.logger.error(f"Tool execution failed: {tool_name}")
            self.logger.error(f"  Error: {result.get('message', 'Unknown error')}")
    
    def log_permission_check(self, tool_name: str, params: Dict[str, Any], granted: bool) -> None:
        """Log a permission check.
        
        Args:
            tool_name: Name of the tool.
            params: Parameters for the tool operation.
            granted: Whether permission was granted.
        """
        if granted:
            self.logger.info(f"Permission granted for tool: {tool_name}")
        else:
            self.logger.warning(f"Permission denied for tool: {tool_name}")
    
    def log_verification(self, verification_type: str, result: bool, details: Dict[str, Any]) -> None:
        """Log a verification result.
        
        Args:
            verification_type: Type of verification performed.
            result: Whether verification was successful.
            details: Additional details about the verification.
        """
        if result:
            self.logger.info(f"Verification successful: {verification_type}")
        else:
            self.logger.warning(f"Verification failed: {verification_type}")
            self.logger.warning(f"  Details: {json.dumps(details)}")


class OperationVerifier:
    """Verifies the success of tool operations."""
    
    @staticmethod
    def verify_file_operation(operation_type: str, filename: str, expected_content: Optional[str] = None) -> Dict[str, Any]:
        """Verify a file operation.
        
        Args:
            operation_type: Type of operation ('write', 'append', 'delete', 'read').
            filename: Path to the file.
            expected_content: Optional expected content for write/append operations.
            
        Returns:
            Dictionary containing verification result.
        """
        try:
            file_path = Path(filename)
            
            if operation_type in ["write", "append"]:
                if not file_path.exists():
                    return {
                        "success": False,
                        "message": f"File {filename} does not exist after {operation_type} operation"
                    }
                
                file_size = file_path.stat().st_size
                
                # If expected content is provided, verify it
                if expected_content is not None:
                    with open(filename, 'r', encoding='utf-8') as file:
                        actual_content = file.read()
                    
                    if operation_type == "write":
                        content_match = actual_content == expected_content
                    else:  # append
                        content_match = actual_content.endswith(expected_content)
                    
                    if not content_match:
                        return {
                            "success": False,
                            "message": f"File content does not match expected content after {operation_type} operation"
                        }
                
                return {
                    "success": True,
                    "message": f"File {filename} exists with size {file_size} bytes",
                    "size": file_size
                }
            
            elif operation_type == "delete":
                if file_path.exists():
                    return {
                        "success": False,
                        "message": f"File {filename} still exists after delete operation"
                    }
                
                return {
                    "success": True,
                    "message": f"File {filename} successfully deleted"
                }
            
            elif operation_type == "read":
                if not file_path.exists():
                    return {
                        "success": False,
                        "message": f"File {filename} does not exist for read operation"
                    }
                
                file_size = file_path.stat().st_size
                return {
                    "success": True,
                    "message": f"File {filename} exists and is readable",
                    "size": file_size
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unknown operation type: {operation_type}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error verifying {operation_type} operation: {str(e)}"
            }
    
    @staticmethod
    def verify_tool_result(tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a tool execution result.
        
        Args:
            tool_name: Name of the tool.
            result: Result of the tool execution.
            
        Returns:
            Dictionary containing verification result.
        """
        if result.get("status") != "success":
            return {
                "success": False,
                "message": f"Tool {tool_name} execution failed: {result.get('message', 'Unknown error')}"
            }
        
        return {
            "success": True,
            "message": f"Tool {tool_name} execution successful"
        }
