import sys
import logging
import json
import subprocess
import os
import shlex
import time # Added for execution time logging
from typing import Dict, Any, Callable, Optional, Union, List # Added Union, List

# Define specific exception types for clarity
class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found."""
    pass

class PermissionDeniedError(Exception):
    """Raised when permission to execute a tool is denied."""
    pass

class InvalidToolRequestError(Exception):
    """Raised when a tool request has an invalid format."""
    pass

from .permissions import PermissionsManager
from .file_system_tool import FileOperationTools

# Configure logging
logger = logging.getLogger(__name__)

class ToolExecutionEngine:
    """Central engine for executing tool requests from agents."""

    def __init__(self, permissions_manager: Optional[PermissionsManager] = None, base_dir: Optional[str] = None, file_focus_callback: Optional[callable] = None, file_content_callback: Optional[callable] = None, default_command_timeout: int = 60):
        """
        Initialize the Tool Execution Engine.

        Args:
            permissions_manager: Manages permissions for tool execution. Defaults to a new instance.
            base_dir: The base directory for file operations. Defaults to the current working directory.
            file_focus_callback: Optional callback function to focus on a file in the UI.
            file_content_callback: Optional callback function to get file content from the UI.
            default_command_timeout: Default timeout in seconds for execute_command.
        """
        self.permissions_manager = permissions_manager or PermissionsManager()
        self.file_tools = FileOperationTools(
            base_dir=base_dir,
            file_focus_callback=file_focus_callback,
            file_content_callback=file_content_callback
        )
        self.default_command_timeout = default_command_timeout

        # Register available tools
        self.tools: Dict[str, Callable] = {
            "file_read": self.file_tools.file_read,
            "file_write": self.file_tools.file_write,
            "file_append": self.file_tools.file_append,
            "file_delete": self.file_tools.file_delete,
            "file_exists": self.file_tools.file_exists,
            "file_list": self.file_tools.file_list,
            "apply_diff": self.file_tools.apply_diff,
            "execute_command": self.execute_command, # Updated method
        }

        logger.info(f"Tool Execution Engine initialized with {len(self.tools)} tools.")

    def execute_tool(self, tool_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool based on a structured request, handling validation, permissions, and execution.

        Args:
            tool_request: A dictionary containing 'tool' name and 'params' dictionary.
                          Example: {"tool": "file_read", "params": {"path": "my_file.txt"}}

        Returns:
            A dictionary containing the execution result, including 'status' ('success' or 'error'),
            and other relevant information (e.g., 'content', 'message', 'return_code').

        Raises:
            (Internally catches and converts to error dicts):
            InvalidToolRequestError: If the request format is invalid.
            ToolNotFoundError: If the requested tool is not registered.
            PermissionDeniedError: If permission for the tool/parameters is denied.
            TypeError: If tool called with incorrect parameters.
            Exception: For other unexpected errors during tool execution.
        """
        start_time = time.monotonic()
        tool_name = "unknown" # Initialize for logging in case of early failure
        try:
            # 1. Validate Request Format
            self.validate_request(tool_request) # Raises InvalidToolRequestError on failure

            tool_name = tool_request.get("tool") # Safe to get after validation
            params = tool_request.get("params", {})

            logger.debug(f"Attempting to execute tool: {tool_name} with params: {params}")

            # 2. Check if Tool Exists
            if tool_name not in self.tools:
                raise ToolNotFoundError(f"Unknown tool: {tool_name}")

            # 3. Check Permissions
            if not self.permissions_manager.check_permission(tool_name, params):
                if not self.permissions_manager.request_permission(tool_name, params):
                    # Log clearly why permission was denied if possible (e.g., user declined)
                    logger.warning(f"Permission explicitly denied by manager/user for tool: {tool_name}")
                    raise PermissionDeniedError(f"Permission denied for tool: {tool_name}")
                else:
                    logger.info(f"Permission granted upon request for tool: {tool_name}")
            else:
                 logger.debug(f"Permission already granted for tool: {tool_name}")


            # 4. Execute Tool
            logger.info(f"Executing tool: {tool_name}")
            logger.debug(f"Calling tool '{tool_name}' with params: {params}")
            result = self.tools[tool_name](**params)

            # Ensure result is always a dictionary with status for consistency downstream
            if not isinstance(result, dict) or "status" not in result:
                 logger.warning(f"Tool '{tool_name}' returned non-standard result: {result}. Wrapping for consistency.")
                 # Attempt to infer status or default to success if no obvious error indicated
                 status = "success" # Default assumption
                 message = f"Tool '{tool_name}' completed."
                 if isinstance(result, dict):
                     if 'error' in result or 'exception' in result:
                         status = "error"
                         message = result.get('error', result.get('exception', f"Tool '{tool_name}' failed."))
                     elif 'message' in result:
                          message = result['message'] # Use tool's message if available
                 # Wrap the original result under a 'data' key if it wasn't already structured well
                 result_data = result if isinstance(result, dict) and 'data' in result else result
                 result = {"status": status, "message": message, "data": result_data}


            logger.info(f"Tool '{tool_name}' execution completed with status: {result.get('status')}")
            logger.debug(f"Tool '{tool_name}' result: {result}")
            return result

        except InvalidToolRequestError as e:
            logger.error(f"Invalid tool request: {str(e)}")
            return {"status": "error", "message": f"Invalid tool request: {str(e)}"}
        except ToolNotFoundError as e:
            logger.error(str(e))
            return {"status": "error", "message": str(e)}
        except PermissionDeniedError as e:
            logger.warning(str(e)) # Log permission denial as warning
            return {"status": "error", "message": str(e)}
        except TypeError as e: # Catch errors from calling tool with wrong params
            logger.error(f"Type error executing tool '{tool_name}' with params {params}: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Incorrect parameters provided for tool '{tool_name}': {str(e)}"}
        except Exception as e:
            # Catch other potential errors during tool execution
            logger.error(f"Unexpected error executing tool '{tool_name}': {str(e)}", exc_info=True) # Use exc_info for traceback
            return {"status": "error", "message": f"Unexpected tool execution error for '{tool_name}': {str(e)}"}
        finally:
            execution_time = time.monotonic() - start_time
            logger.info(f"Tool '{tool_name}' execution finished in {execution_time:.4f} seconds.")

    def register_tool(self, tool_name: str, tool_function: Callable) -> None:
        """
        Register a new tool with the engine.

        Args:
            tool_name: The name to register the tool under. Must be a non-empty string.
            tool_function: The callable function that implements the tool.

        Raises:
            ValueError: If tool_name is empty or not a string.
            TypeError: If tool_function is not callable.
        """
        if not tool_name or not isinstance(tool_name, str):
             raise ValueError("Tool name must be a non-empty string.")
        if not callable(tool_function):
            raise TypeError(f"Tool function for '{tool_name}' must be callable.")

        if tool_name in self.tools:
             logger.warning(f"Overwriting existing tool registration for: {tool_name}")
        self.tools[tool_name] = tool_function
        logger.info(f"Registered tool: {tool_name}")

    def get_available_tools(self) -> Dict[str, str]:
        """Get descriptions of available tools."""
        # TODO: Consider dynamically generating descriptions from docstrings or annotations.
        # TODO: Add info about whether docker is required/available for execute_command if applicable.
        descriptions = {
            "file_read": "Read content from a file. Params: 'path' (str), 'start_line' (Optional[int]), 'end_line' (Optional[int]).",
            "file_write": "Write content to a file, overwriting if it exists. Params: 'path' (str), 'content' (str).",
            "file_append": "Append content to a file. Params: 'path' (str), 'content' (str).",
            "file_delete": "Delete a file. Params: 'path' (str).",
            "file_exists": "Check if a file or directory exists. Params: 'path' (str).",
            "file_list": "List files and directories. Params: 'path' (str), 'recursive' (Optional[bool]).",
            "apply_diff": "Apply changes to a file using unified diff format. Params: 'path' (str), 'diff' (str).",
            "execute_command": f"Execute a shell command. Params: 'command' (str), 'cwd' (Optional[str]), 'timeout' (Optional[int], default: {self.default_command_timeout}s). Warning: Direct execution, use with caution.",
        }
        # Add descriptions for any dynamically registered tools if needed
        # for name, func in self.tools.items():
        #     if name not in descriptions:
        #         descriptions[name] = func.__doc__ or "No description available."
        return descriptions

    def validate_request(self, tool_request: Dict[str, Any]) -> None:
        """
        Validate the basic structure of a tool request.

        Args:
            tool_request: The incoming tool request dictionary.

        Raises:
            InvalidToolRequestError: If the validation fails (e.g., not a dict, missing 'tool',
                                     'tool' not a string, 'params' not a dict if present).
        """
        logger.debug(f"Validating tool request structure: {tool_request}")
        if not isinstance(tool_request, dict):
            raise InvalidToolRequestError("Tool request must be a dictionary")
        if "tool" not in tool_request:
            raise InvalidToolRequestError("Tool request must contain a 'tool' field")
        tool_name = tool_request.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
             raise InvalidToolRequestError("'tool' field must be a non-empty string")
        if "params" in tool_request and not isinstance(tool_request.get("params"), dict):
            # Allow params to be missing, but if present, must be a dict
            raise InvalidToolRequestError("Tool request 'params' must be a dictionary if present")
        # Further parameter validation specific to the tool should happen within the tool function itself
        # or could be added here if a schema definition system is implemented.
        logger.debug(f"Tool request structure for '{tool_name}' validated successfully.")

    # Removed parse_agent_output method.
    # This engine executes structured requests, it doesn't parse agent text output.
    # Parsing logic belongs elsewhere if needed (e.g., Orchestrator or dedicated parser).

    def execute_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None # Use instance default if None
    ) -> Dict[str, Any]:
        """
        Executes a shell command directly using subprocess.

        Args:
            command: The command string to execute.
            cwd: Optional working directory relative to the project's base_dir. If None, uses base_dir.
            timeout: Optional timeout in seconds. If None, uses the engine's default_command_timeout.

        Returns:
            Dict containing status, return_code, stdout, stderr, and a message.
        """
        # Directly call the subprocess execution method
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        logger.debug(f"Executing command '{command}' with effective timeout {effective_timeout}s")
        # Note: Consider adding asynchronous support in the future for long-running commands.
        return self._execute_command_subprocess(command, cwd, effective_timeout)

    # --- Subprocess Fallback Method ---
    def _execute_command_subprocess(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Internal method to execute a command using subprocess.

        Args:
            command: The command string.
            cwd: Optional working directory.
            timeout: Timeout in seconds (should always have a value passed from execute_command).

        Returns:
            A dictionary with execution results: status, return_code, stdout, stderr, message.

        Warning:
            Direct subprocess execution can be risky, especially with shell=True.
            Consider security implications and alternatives like sandboxing if possible.
            A whitelist approach for allowed commands is recommended for production systems.
        """
        # Ensure timeout has a value; fall back to default if somehow None reaches here (shouldn't happen)
        effective_timeout = timeout if timeout is not None else self.default_command_timeout
        # Determine the correct working directory, defaulting to the engine's base directory
        # Ensure base_dir is handled correctly if None
        base_dir = self.file_tools.base_dir if self.file_tools.base_dir else os.getcwd()
        effective_cwd = os.path.abspath(os.path.join(base_dir, cwd)) if cwd else os.path.abspath(base_dir)
        logger.info(f"[Subprocess] Executing command: '{command}' in directory: {effective_cwd} with timeout: {effective_timeout}s")

        # Ensure the target directory exists
        if not os.path.isdir(effective_cwd):
             logger.error(f"[Subprocess] Working directory does not exist: {effective_cwd}")
             # Use FileNotFoundError semantics for consistency
             return {"status": "error", "return_code": -1, "stdout": "", "stderr": f"Working directory not found: {effective_cwd}", "message": f"[Subprocess] Working directory does not exist: {effective_cwd}"}

        use_shell = False
        # Union and List are now imported at the top
        cmd_list_or_str: Union[List[str], str]
        try:
            # Attempt to split the command safely for non-shell execution on POSIX-like systems
            # On Windows, shlex might behave differently, and shell=True is often more reliable for complex commands.
            # However, shell=False with a list is generally safer if the command is simple.
            if sys.platform != "win32":
                 # Only use shlex if the command doesn't contain obvious shell metacharacters
                 # This is a basic check; more robust parsing might be needed for complex cases
                 if not any(c in command for c in ['|', ';', '&', '>', '<', '`', '$', '\\']):
                     cmd_list_or_str = shlex.split(command)
                 else:
                     # Log clearly that shell=True is being used due to potential shell syntax
                     logger.warning(f"[Subprocess] Command '{command}' contains shell characters or complex syntax, using shell=True (Security Risk).")
                     use_shell = True
                     cmd_list_or_str = command
            else:
                # On Windows, shell=True is often needed for built-ins or complex paths.
                logger.debug("[Subprocess] Using shell=True on Windows for command execution.")
                use_shell = True
                cmd_list_or_str = command
        except ValueError:
            # If shlex fails (e.g., unmatched quotes), fall back to shell=True
            logger.warning(f"[Subprocess] Could not parse command '{command}' with shlex (e.g., unmatched quotes), falling back to shell=True (Security Risk).")
            use_shell = True
            cmd_list_or_str = command # Use the raw command string with shell=True

        try:
            process = subprocess.run(
                cmd_list_or_str,
                shell=use_shell, # Use shell=True if determined above
                capture_output=True,
                text=True, # Ensure stdout/stderr are strings
                cwd=effective_cwd,
                timeout=effective_timeout, # Use the determined timeout
                check=False, # We handle non-zero exit codes manually
                # Set environment variables for the subprocess if needed (example)
                # env=os.environ.copy() # Pass parent environment
            )
            logger.info(f"[Subprocess] Command finished with return code: {process.returncode}")
            stdout_content = process.stdout.strip()
            stderr_content = process.stderr.strip()
            stdout_log = f"[Subprocess] stdout:\n{stdout_content}" if stdout_content else "[Subprocess] stdout: (empty)"
            stderr_log = f"[Subprocess] stderr:\n{stderr_content}" if stderr_content else "[Subprocess] stderr: (empty)"
            logger.debug(stdout_log)
            if stderr_content: logger.warning(stderr_log) # Log non-empty stderr as warning

            result_status = "success" if process.returncode == 0 else "error"
            error_message = ""
            if result_status == "error":
                 error_message = f"[Subprocess] Command failed with exit code {process.returncode}."
                 if stderr_content:
                     error_message += f" Stderr: {stderr_content}"

            return {
                 "status": result_status,
                 "return_code": process.returncode,
                 "stdout": stdout_content,
                 "stderr": stderr_content,
                 "message": f"[Subprocess] Command executed successfully." if result_status == "success" else error_message
             }
        except subprocess.TimeoutExpired:
             logger.error(f"[Subprocess] Command '{command}' timed out after {effective_timeout} seconds.")
             return {"status": "error", "return_code": -1, "stdout": "", "stderr": f"Command timed out after {effective_timeout} seconds.", "message": f"[Subprocess] Command timed out after {effective_timeout} seconds."}
        except FileNotFoundError:
              # This usually happens if the command itself isn't found (often when shell=False)
              cmd_name = cmd_list_or_str[0] if isinstance(cmd_list_or_str, list) and cmd_list_or_str else command.split(' ', 1)[0]
              logger.error(f"[Subprocess] Command or executable not found: '{cmd_name}' in CWD '{effective_cwd}'")
              return {"status": "error", "return_code": -1, "stdout": "", "stderr": f"Command not found: '{cmd_name}'", "message": f"[Subprocess] Command or executable not found: '{cmd_name}'"}
        except PermissionError as e:
              logger.error(f"[Subprocess] Permission denied executing command '{command}' in '{effective_cwd}': {e}", exc_info=True)
              return {"status": "error", "return_code": -1, "stdout": "", "stderr": str(e), "message": f"[Subprocess] Permission denied: {e}"}
        except Exception as e:
             # Catch other potential OS or subprocess errors
             logger.error(f"[Subprocess] Unexpected error executing command '{command}': {str(e)}", exc_info=True)
             return {"status": "error", "return_code": -1, "stdout": "", "stderr": str(e), "message": f"[Subprocess] Unexpected command execution error: {str(e)}"}
    # --- End Fallback Method ---
