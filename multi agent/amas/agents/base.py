"""Base classes and decorators for AMAS agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import os # Added for path manipulation
from amas.core.llm_service_gemini import BaseLLMService # Keep specific import if only Gemini is used, or create a base interface file
import logging # Import logging module
from logging import Handler, LogRecord # Import Handler and LogRecord for custom handler
from amas.core.context_manager import ContextManager
from amas.core.tools.execution_engine import ToolExecutionEngine # Import the correct class
from amas.core.communication_bus import CommunicationBus # Import CommunicationBus
# Setup logger for this module
logger = logging.getLogger(__name__)

# Custom Exception for LLM errors
class LLMServiceError(Exception):
    """Custom exception for errors occurring during LLM service calls."""
    pass

class BaseAgent(ABC):
    """Abstract base class for all agents in the AMAS system."""

    def __init__(self, name: str, config: Dict[str, Any], llm_service: BaseLLMService, context: ContextManager, base_dir: str, log_callback: Optional[Callable[[str, str], None]] = None, communication_bus: Optional[CommunicationBus] = None): # Added base_dir
        """
        Initializes the BaseAgent.

        Args:
            name: The unique name of the agent instance (e.g., 'planner', 'coder').
            config: Agent-specific configuration dictionary from the main config file.
            llm_service: An initialized instance of a BaseLLMService subclass.
            context: The shared ContextManager instance.
            base_dir: The base working directory for file operations. # Added base_dir doc
            log_callback: Optional function to send log messages to (e.g., GUI).
            communication_bus: Optional instance of the CommunicationBus.

        Example:
            >>> class MyAgent(BaseAgent):
            ...     def execute_task(self, task_input: str) -> str:
            ...         self.log_callback(f"Executing task: {task_input}")
            ...         prompt = self._get_prompt("my_prompt", input=task_input)
            ...         response = self._call_llm(prompt)
            ...         return f"Processed: {response}"
            ...
            >>> # Assuming llm_service, context, bus are initialized
            >>> my_agent = MyAgent(name="my_agent_instance", config={"some_setting": True}, llm_service=llm_service, context=context, communication_bus=bus)
        """
        self.name = name
        self.config = config
        self.llm_service = llm_service # Use generic name
        self.context = context
        self.base_dir = base_dir # Store the base directory
        self.communication_bus = communication_bus # Store the communication bus
        self._log_callback = log_callback # Store the raw callback

        # Initialize tool executor
        self.tool_executor: Optional[ToolExecutionEngine] = None # Correct type hint
        self.available_tools: List[Dict[str, Any]] = []

        # --- Logging Setup ---
        # Add a custom handler if a callback is provided
        if self._log_callback:
            # Check if handler already exists to prevent duplicates if re-initialized
            handler_exists = any(isinstance(h, _CallbackHandler) and h.callback == self._log_callback for h in logger.handlers)
            if not handler_exists:
                callback_handler = _CallbackHandler(self._log_callback, self.name)
                logger.addHandler(callback_handler)
                # Ensure logger level is low enough for handler to receive messages
                if logger.level > logging.DEBUG: # Or use a configurable level
                     logger.setLevel(logging.DEBUG)
                     
        # Log initialization using the standard logger
        logger.info(f"Initializing BaseAgent '{self.name}'...")
        logger.info(f"BaseAgent '{self.name}' initialized with LLM: {self.llm_service.__class__.__name__}")

    # Removed _setup_agent_communication, _on_plan_update, _on_error methods as they relied on non-existent subscribe functionality

    @abstractmethod
    def execute_task(self, task_input: Any) -> Any:
        """
        The main method for an agent to perform its specific task.

        Args:
            task_input: The input required for the agent's task (can be string, dict, etc.).

        Returns:
            The result of the agent's task execution.

        Example:
            >>> # In a derived agent class:
            >>> def execute_task(self, user_query: str) -> Dict[str, Any]:
            ...     # ... implementation using self._call_llm, self.use_tool etc. ...
            ...     result = {"summary": "...", "actions_taken": []}
            ...     return result
            >>>
            >>> # Called by orchestrator or another agent:
            >>> agent_result = my_agent.execute_task("Analyze recent sales data.")
        """
        pass

    async def request_help(self, target_agent: str, request: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        """
        Sends a help request message to another agent via the communication bus.

        Args:
            target_agent: The name of the agent to request help from.
            request: A short description of the help needed (used as message type).
            payload: Optional dictionary containing details of the request.

        Returns:
            True if the message was sent successfully (bus available), False otherwise.

        Example:
            >>> # Inside an agent's method:
            >>> async def perform_complex_analysis(self, data):
            ...     # ... some processing ...
            ...     if self.needs_coding_help(data):
            ...         success = await self.request_help(
            ...             target_agent="coder",
            ...             request="generate_python_script",
            ...             payload={"description": "Create script for data transformation", "data_schema": data.schema}
            ...         )
            ...         if not success:
            ...             self.log_callback("Failed to request help from coder agent.", "warning")
            ...     # ... continue processing ...
        """
        if not self.communication_bus:
            logger.error("Communication bus not available, cannot send help request.")
            return False

        logger.info(f"Sending help request '{request}' to agent '{target_agent}' via bus.")
        try:
            await self.communication_bus.send_message(
                sender=self.name,
                recipient=target_agent,
                message_type=f"help_request:{request}", # Example message type
                payload=payload
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send help request via bus: {e}")
            return False

    def _get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Helper method to retrieve and format a prompt from the 'amas/prompts' directory.

        Args:
            prompt_name: The relative path name of the prompt template within 'amas/prompts'
                         (e.g., 'planner/plan_creation'). The '.prompt' extension is assumed.
            **kwargs: Variables to format the prompt template using str.format().

        Returns:
            The formatted prompt string.

        Raises:
            FileNotFoundError: If the prompt template file (e.g., amas/prompts/planner/plan_creation.prompt) is not found.
            KeyError: If a required variable for formatting is missing in kwargs.
            IOError: If there's an error reading the prompt file.
            ValueError: If there's an error during formatting (other than KeyError).

        Example:
            >>> # Assuming 'amas/prompts/summarizer/basic.prompt' exists and contains:
            >>> # "Summarize the following text:\n{text_to_summarize}"
            >>> try:
            ...     prompt_text = self._get_prompt(
            ...         "summarizer/basic",
            ...         text_to_summarize="This is a long document..."
            ...     )
            ...     print(prompt_text)
            ... except (FileNotFoundError, KeyError) as e:
            ...     logger.error(f"Failed to get prompt: {e}") # Use logger
        """
        base_prompt_dir = "amas/prompts" # Define base directory
        prompt_filename = f"{prompt_name}.prompt"
        # Ensure prompt_name doesn't try to escape the base directory (basic security)
        # Ensure prompt_name doesn't try to escape the base directory (basic security)
        # Note: os.path.join already handles path separators correctly. normpath cleans '..' etc.
        # We construct the full path first, then normalize and check against the base dir.
        prompt_filepath = os.path.join(base_prompt_dir, prompt_filename)
        abs_prompt_filepath = os.path.abspath(prompt_filepath)
        abs_base_prompt_dir = os.path.abspath(base_prompt_dir)

        # Security check: Ensure the final absolute path is within the intended base directory
        if not abs_prompt_filepath.startswith(abs_base_prompt_dir):
             raise ValueError(f"Invalid prompt path attempting directory traversal: {prompt_filepath}")
             
        # The security check using abspath and startswith (lines 181-183) is sufficient.
        # Removing the redundant/old logic below.

        logger.debug(f"Attempting to load prompt from '{abs_prompt_filepath}' with args: {kwargs}")

        try:
            # Use absolute path based on this file's location if needed, assuming relative to workspace for now
            # script_dir = os.path.dirname(__file__)
            # abs_prompt_filepath = os.path.join(script_dir, '..', '..', prompt_filepath) # Adjust relative path as needed
            # Using absolute path derived from relative path:
            abs_prompt_filepath = os.path.abspath(prompt_filepath)
            
            with open(abs_prompt_filepath, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {abs_prompt_filepath}")
            raise FileNotFoundError(f"Prompt template file not found: {abs_prompt_filepath}")
        except IOError as e:
            logger.error(f"Error reading prompt file {abs_prompt_filepath}: {e}")
            raise IOError(f"Could not read prompt file {abs_prompt_filepath}: {e}") from e

        try:
            formatted_prompt = prompt_template.format(**kwargs)
            logger.debug(f"Successfully loaded and formatted prompt '{prompt_name}'.")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Missing key '{e}' for formatting prompt '{prompt_name}' from {abs_prompt_filepath}")
            raise KeyError(f"Missing variable '{e}' required for formatting prompt template '{prompt_name}'") from e
        except Exception as format_e: # Catch other potential formatting errors
             logger.error(f"Error formatting prompt '{prompt_name}' from {abs_prompt_filepath}: {format_e}")
             raise ValueError(f"Failed to format prompt '{prompt_name}': {format_e}") from format_e

    def _call_llm(self, prompt: str, **generation_kwargs) -> str: # Added return type hint
        """
        Helper method to call the agent's configured LLM service.

        Args:
            prompt: The formatted prompt string to send to the LLM.
            **generation_kwargs: Additional keyword arguments for the LLM's
                                 generate_response method (e.g., temperature).

        Returns:
            The raw text response from the LLM.

        Example:
            >>> prompt = "Translate 'hello' to French."
            >>> try:
            ...     translation = self._call_llm(prompt, temperature=0.7, max_tokens=50)
            ...     print(f"LLM Response: {translation}")
            ... except LLMServiceError as e:
            ...     logger.error(f"LLM call failed: {e}") # Use logger
        """
        logger.debug(f"Calling LLM service {self.llm_service.__class__.__name__} with {len(prompt)} char prompt...")
        # Combine base config generation params with call-specific overrides
        final_kwargs = {**self.config.get('generation_params', {}), **generation_kwargs}
        try:
            response = self.llm_service.generate_response(prompt, **final_kwargs)
            if not isinstance(response, str):
                 logger.warning(f"LLM returned non-string type: {type(response)}. Returning empty string.")
                 return "" # Return empty string if LLM gives non-string
            logger.debug(f"Received LLM response (length {len(response)}). First 100 chars: '{response[:100]}...'")
            return response
        except Exception as llm_e:
            error_msg = f"Error during LLM call: {str(llm_e)}"
            logger.error(error_msg) # Already logged by logger.exception below, but keep for consistency if exception logging is removed
            logger.exception("LLM Call Traceback:") # Log the full traceback for debugging
            raise LLMServiceError(error_msg) from llm_e # Raise specific exception
        
    def setup_tools(self, tool_executor: ToolExecutionEngine) -> None: 
        """
        Set up the tool executor for this agent.
        
        Args:
            tool_executor: The tool executor to use.

        Example:
            >>> # Typically called during initialization or by the orchestrator
            >>> from amas.core.tools.execution_engine import ToolExecutionEngine # Example import
            >>> tool_engine = ToolExecutionEngine(permission_manager=...) # Initialize engine
            >>> # Register tools with the engine...
            >>> my_agent.setup_tools(tool_engine)
        """
        self.tool_executor = tool_executor
        # Get available tools from the executor itself
        if hasattr(tool_executor, 'get_available_tools'):
             self.available_tools = tool_executor.get_available_tools() # Get dict {name: description}
             # Use direct logger to bypass potential callback issues during setup
             logger.info(f"Agent '{self.name}' has access to {len(self.available_tools)} tools: {list(self.available_tools.keys())}")
        else:
             # Use direct logger
             logger.warning(f"Agent '{self.name}': Tool executor setup, but cannot list available tools.")
             self.available_tools = [] # Or handle differently
        
    # Removed request_permission_for_tool method.
    # Permission handling should be managed by the ToolExecutionEngine/PermissionsManager.
    # Agents should simply call use_tool.
    def use_tool(self, tool_name: str, **tool_args) -> Any: # Added return type hint
        """
        Use a tool, potentially requesting permission first.
        
        Args:
            tool_name: The name of the tool to use.
            **tool_args: Arguments to pass to the tool.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ValueError: If the tool executor is not set up or the tool is not found.
            PermissionError: If permission to use the tool is denied.
            RuntimeError: If the tool execution itself fails.

        Example:
            >>> # Assuming 'file_writer' tool is available and agent has executor
            >>> try:
            ...     result = self.use_tool(
            ...         "file_writer",
            ...         path="./output.txt",
            ...         content="This is the content to write."
            ...     )
            ...     if result.get("status") == "success":
            ...         logger.info(f"File written successfully: {result.get('details')}") # Use logger
            ... except (ValueError, PermissionError, RuntimeError) as e:
            ...     logger.error(f"Failed to use tool 'file_writer': {e}") # Use logger
        """
        if not self.tool_executor:
            raise ValueError(f"Agent '{self.name}' does not have a tool executor set up.")
            
        # Construct the tool request dictionary expected by ToolExecutionEngine
        tool_request = {
            "tool": tool_name,
            "params": tool_args
        }
            
        # The ToolExecutionEngine should handle permission checks internally now
        # based on its PermissionsManager
        try:
            logger.info(f"Attempting to use tool '{tool_name}'...")
            result = self.tool_executor.execute_tool(tool_request) # Pass the structured request
            
            if result.get("status") == "error":
                 # Handle tool execution errors reported by the engine
                 error_message = result.get("message", "Unknown tool execution error")
                 logger.error(f"Tool '{tool_name}' execution failed: {error_message}")
                 # Decide whether to raise an exception or return the error dict
                 # Raising allows the agent's main loop to catch it
                 raise RuntimeError(f"Tool '{tool_name}' failed: {error_message}")
                 
            logger.info(f"Tool '{tool_name}' executed successfully.")
            return result # Return the full result dictionary from the engine
            
        except PermissionError as e: # Should be caught by engine, but as fallback
            logger.error(f"Tool use denied: {e}")
            raise
        except Exception as e:
            logger.error(f"Error using tool '{tool_name}': {e}")
            raise

# Local registry for agent classes
_AGENT_REGISTRY = {}

def register_agent(name):
    """Decorator to register agent classes."""
    def decorator(cls):
        if not issubclass(cls, BaseAgent):
            raise TypeError("Registered class must be a subclass of BaseAgent")
        if name in _AGENT_REGISTRY:
             logger.warning(f"Overwriting registration for agent '{name}'. Previous: {_AGENT_REGISTRY[name].__name__}, New: {cls.__name__}")
        _AGENT_REGISTRY[name] = cls
        logger.info(f"Registered agent '{name}' as {cls.__name__}") # Use logger
        return cls
    return decorator

def get_agent_registry():
    """Returns the agent registry dictionary."""
    return _AGENT_REGISTRY


# --- Custom Logging Handler ---
class _CallbackHandler(Handler):
    """A logging handler that directs logs to a callback function."""
    def __init__(self, callback: Callable[[str, str], None], agent_name: str):
        """
        Initializes the handler.

        Args:
            callback: The function to call with log messages (msg, level_name).
            agent_name: The name of the agent this handler is associated with.
        """
        super().__init__()
        self.callback = callback
        self.agent_name = agent_name
        # You might want to add a formatter here for consistency
        # self.setFormatter(logging.Formatter(...))

    def emit(self, record: LogRecord):
        """
        Formats the log record and sends it to the callback.

        Args:
            record: The log record.
        """
        try:
            # Format the message using the handler's formatter if set, else default
            msg = self.format(record)
            # Add agent name prefix if not already present (optional)
            # if not msg.startswith(f"Agent {self.agent_name}"):
            #     msg = f"Agent {self.agent_name}: {msg}"
            level_name = record.levelname.lower()
            self.callback(msg, level_name)
        except Exception:
            self.handleError(record)