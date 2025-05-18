"""
Integration of logging and verification with the Tool Execution Engine.
This module enhances the Tool Execution Engine with logging and verification capabilities.
"""

import logging
from typing import Dict, Any, Optional

from .execution_engine import ToolExecutionEngine
from .logging_verification import ToolLogger, OperationVerifier

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedToolExecutionEngine:
    """Tool Execution Engine with enhanced logging and verification."""
    
    def __init__(self, base_engine: Optional[ToolExecutionEngine] = None, 
                 log_dir: Optional[str] = None,
                 log_to_file: bool = True):
        """Initialize the enhanced tool execution engine.
        
        Args:
            base_engine: Optional base Tool Execution Engine to enhance.
            log_dir: Directory to store log files.
            log_to_file: Whether to log to file in addition to console.
        """
        self.base_engine = base_engine or ToolExecutionEngine()
        self.logger = ToolLogger(log_dir, log_to_file)
        self.verifier = OperationVerifier()
        
        logger.info("Enhanced Tool Execution Engine initialized with logging and verification")
    
    def execute_tool(self, tool_request: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute a tool with enhanced logging and verification.
        
        Args:
            tool_request: Dictionary containing the tool request.
            agent_name: Optional name of the agent making the request.
                
        Returns:
            Dictionary containing the result of the tool execution.
        """
        # Log the tool request
        self.logger.log_tool_request(tool_request, agent_name)
        
        # Execute the tool using the base engine
        result = self.base_engine.execute_tool(tool_request)
        
        # Log the tool execution
        self.logger.log_tool_execution(tool_request, result)
        
        # Verify the result if successful
        if result.get("status") == "success":
            tool_name = tool_request.get("tool", "")
            params = tool_request.get("params", {})
            
            # Perform additional verification for file operations
            if tool_name in ["file_write", "file_append", "file_delete", "file_read"]:
                filename = params.get("filename", "")
                expected_content = params.get("content", None) if tool_name in ["file_write", "file_append"] else None
                
                verification_result = self.verifier.verify_file_operation(
                    operation_type=tool_name.replace("file_", ""),
                    filename=filename,
                    expected_content=expected_content
                )
                
                # Log the verification result
                self.logger.log_verification(
                    verification_type=f"{tool_name} operation",
                    result=verification_result.get("success", False),
                    details=verification_result
                )
                
                # Add verification info to the result
                result["verification"] = verification_result
        
        return result
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the base engine.
        
        Args:
            name: Name of the attribute.
            
        Returns:
            The attribute from the base engine.
        """
        return getattr(self.base_engine, name)
