"""
Orchestrator module for AMAS.
This module provides the orchestrator functionality for the multi-agent system.
"""

import logging
import yaml
import os
import re  # Add this import for regular expressions
from pathlib import Path
from typing import Dict, Any, Optional, Type, List, TypedDict, Set
import asyncio # Import asyncio

from .context_manager import ContextManager
from amas.agents.base import BaseAgent, get_agent_registry # Import agent base and registry
from .llm_service_gemini import GeminiLLMService # Import specific LLM service
# Assuming a BaseLLMService exists for type hinting and ToolExecutionEngine for type hinting
from amas.core.llm_service_gemini import BaseLLMService
from .tools.execution_engine import ToolExecutionEngine # Added for type hint
from .communication_bus import CommunicationBus # Added for inter-agent communication
# Removed unused import of the instance registry: from .agent_registry import agent_registry


# Define a TypedDict for the internal task structure
class TaskDict(TypedDict):
    id: Any
    description: str
    dependencies: List[Any]
    priority: int
    status: str # Literal["pending", "running", "completed", "failed", "cancelled"] ?
    result: Optional[Dict[str, Any]]
    async_task: Optional[asyncio.Task]


# Configure logging
logger = logging.getLogger(__name__)

class Orchestrator:
    """Main orchestrator for the multi-agent system."""

    
    def __init__(self, config: Dict[str, Any], context: Optional[ContextManager] = None, llm_service_gemini: Optional[BaseLLMService] = None, base_dir: Optional[str] = None, log_callback: Optional[callable] = None): # Added base_dir
        """Initialize the orchestrator, LLM service, and agents.

        Args:
            config: Configuration dictionary.
            context: Optional context manager.
            llm_service_gemini: Optional pre-initialized LLM service. If None, it will be initialized based on config.
            base_dir: Optional base working directory for agents. If None, uses current working directory. # Added base_dir doc
            log_callback: Optional function to send log messages to (e.g., GUI).
        """
        self.config = config
        self.context = context or ContextManager()
        self.base_dir = base_dir or os.getcwd() # Store base_dir, default to cwd
        self._validate_config()

        self.log_callback = log_callback or (lambda msg, level="info", **kwargs: logger.log(logging.getLevelName(level.upper()), f"[{level.upper()}] {msg}", **kwargs)) # Default logger that handles kwargs
        self.llm_service_gemini = llm_service_gemini or self._initialize_llm_service_gemini() # Pass callback here if LLM service supports it
        self.communication_bus = CommunicationBus() # Initialize the communication bus FIRST
        self.agents: Dict[str, BaseAgent] = self._initialize_agents() # Initialize agents AFTER bus
        self.tool_engine: Optional[ToolExecutionEngine] = None # Added to hold the tool engine
        
        self.log_callback(f"Orchestrator initialized with {len(self.agents)} agents, {self.llm_service_gemini.__class__.__name__}, Communication Bus, and base_dir '{self.base_dir}'", "info") # Updated log

    def _validate_config(self) -> None:
        """
        Validates the essential structure of the configuration dictionary.

        Checks for the presence and correct type of key sections like
        'llm_service_gemini', 'agents', 'primary_agent', and 'max_concurrent_tasks'.
        Logs warnings or raises ValueErrors for invalid configurations.
        """
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary.")

        # Validate LLM service config (presence, basic type)
        if 'llm_service_gemini' not in self.config:
            self.log_callback("Configuration missing 'llm_service_gemini' section.", "warning")
            # Allow proceeding if LLM service is provided directly, but log warning.
            # If not provided directly, _initialize_llm_service_gemini will handle it.
        elif not isinstance(self.config.get('llm_service_gemini'), dict):
             error_msg = "Configuration section 'llm_service_gemini' must be a dictionary."
             self.log_callback(error_msg, "error")
             current_errors = self.context.get("errors", [])
             current_errors.append(error_msg)
             self.context.set("errors", current_errors)
             raise ValueError(error_msg)

        # Validate agents config (presence, basic type)
        if 'agents' not in self.config:
            # This might be acceptable if no agents are needed, but log a warning.
            self.log_callback("Configuration missing 'agents' section. No agents will be initialized.", "warning")
        elif not isinstance(self.config.get('agents'), dict):
            error_msg = "Configuration section 'agents' must be a dictionary."
            self.log_callback(error_msg, "error")
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            raise ValueError(error_msg)

        # Validate primary_agent if agents are defined
        if 'agents' in self.config and isinstance(self.config.get('agents'), dict):
            if 'primary_agent' not in self.config:
                 self.log_callback("Configuration missing 'primary_agent' key. Defaulting to 'planner'.", "warning")
            elif not isinstance(self.config.get('primary_agent'), str):
                 error_msg = "Configuration key 'primary_agent' must be a string."
                 self.log_callback(error_msg, "error")
                 current_errors = self.context.get("errors", [])
                 current_errors.append(error_msg)
                 self.context.set("errors", current_errors)
                 raise ValueError(error_msg)

        # Validate max_concurrent_tasks if present
        if 'max_concurrent_tasks' in self.config:
            max_tasks = self.config.get('max_concurrent_tasks')
            if not isinstance(max_tasks, int) or max_tasks <= 0:
                 error_msg = f"Configuration key 'max_concurrent_tasks' must be a positive integer. Found: {max_tasks}"
                 self.log_callback(error_msg, "error")
                 current_errors = self.context.get("errors", [])
                 current_errors.append(error_msg)
                 self.context.set("errors", current_errors)
                 raise ValueError(error_msg)

    
    @classmethod
    def from_config(cls, config_path: Path, context: Optional[ContextManager] = None, base_dir: Optional[str] = None, log_callback: Optional[callable] = None) -> 'Orchestrator': # Added base_dir
        """Create an orchestrator from a configuration file.

        Args:
            config_path: Path to the configuration file.
            context: Optional context manager.
            base_dir: Optional base working directory for agents. # Added base_dir doc
            log_callback: Optional function to send log messages to.
            
        Returns:
            Orchestrator instance.
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the configuration file is invalid.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Use callback if provided, otherwise use standard logger for this initial message
            effective_log = log_callback or (lambda msg, level="info": logger.info(f"[{level.upper()}] {msg}"))
            effective_log(f"Configuration loaded from {config_path}", "info")
            # Pass the log_callback and base_dir to the constructor
            return cls(config, context, base_dir=base_dir, log_callback=log_callback) # Added base_dir
        except FileNotFoundError:
            # Use standard logger here as callback might not be available if init fails early
            logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_path}: {str(e)}")
            raise
    
    def _initialize_llm_service_gemini(self) -> BaseLLMService:
        """Initializes the LLM service based on configuration."""
        llm_config = self.config.get('llm_service_gemini', {})
        llm_type = llm_config.get('type', 'gemini') # Default to gemini
        api_key = llm_config.get('api_key') # Can be None, Gemini service checks env var
        model_name = llm_config.get('model', 'gemini-2.0-flash')
        generation_params = llm_config.get('generation_params', {})
        
        self.log_callback(f"Initializing LLM service of type '{llm_type}' with model '{model_name}'", "info")
        
        if llm_type.lower() == 'gemini':
            try:
                # TODO: Modify GeminiLLMService to accept log_callback if needed
                return GeminiLLMService(api_key=api_key, model_name=model_name, **generation_params)
            except (ValueError, ConnectionError) as e:
                error_msg = f"Failed to initialize GeminiLLMService: {e}"
                self.log_callback(error_msg, "error")
                # Append error to context
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                raise RuntimeError(f"LLM Initialization failed: {e}") from e
        else:
            # Placeholder for other LLM types
            raise NotImplementedError(f"LLM service type '{llm_type}' is not supported.")
            
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initializes agents based on configuration."""
        agents_config = self.config.get('agents', {})
        initialized_agents = {}
        class_registry = get_agent_registry() # Renamed variable for clarity
        
        if not agents_config:
            self.log_callback("No agents defined in the configuration.", "warning")
            return {}
            
        if not self.llm_service_gemini:
             raise RuntimeError("LLM Service must be initialized before agents.")

        for agent_name, agent_conf in agents_config.items():
            agent_type = agent_conf.get('type')
            if not agent_type:
                self.log_callback(f"Agent '{agent_name}' missing 'type' in configuration. Skipping.", "warning")
                continue
                
            agent_class: Optional[Type[BaseAgent]] = class_registry.get(agent_type)
            if not agent_class:
                self.log_callback(f"Agent type '{agent_type}' for agent '{agent_name}' not found in registry. Skipping.", "warning")
                continue
                
            try:
                self.log_callback(f"Initializing agent '{agent_name}' of type '{agent_type}'", "info")
                # Pass agent-specific config, llm_service_gemini, and context
                agent_instance = agent_class(
                    name=agent_name,
                    config=agent_conf,
                    llm_service=self.llm_service_gemini, # Corrected keyword argument
                    context=self.context,
                    base_dir=self.base_dir, # Pass base_dir to agent constructor
                    log_callback=self.log_callback, # Pass the orchestrator's callback
                    communication_bus=self.communication_bus # Pass the bus instance
                )
                initialized_agents[agent_name] = agent_instance
                self.communication_bus.register_agent(agent_name) # Register agent with the bus
            except Exception as e:
                error_msg = f"Failed to initialize agent '{agent_name}': {e}"
                self.log_callback(error_msg, "error")
                # Append error to context
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                # Log traceback separately if needed logger.exception(...)
                # Decide whether to continue or raise
                # raise RuntimeError(f"Agent Initialization failed for '{agent_name}': {e}") from e
                
        return initialized_agents

    def _select_agent_for_task(self, step_index: int, step: Dict[str, Any]) -> Optional[str]:
        """
        Selects the agent specified in the plan step dictionary.

        Args:
            step_index: The index of the current step (for logging).
            step: The plan step dictionary.

        Returns:
            The name (string) of the specified agent if found and initialized,
            otherwise None. Errors are logged and added to context.
        """
        target_agent_name = step.get("agent")

        if not target_agent_name or not isinstance(target_agent_name, str):
            error_msg = f"Step {step.get('step_id', step_index + 1)} is missing a valid 'agent' key."
            self.log_callback(error_msg, "error")
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            return None

        target_agent_name = target_agent_name.lower() # Ensure lowercase for comparison

        if target_agent_name not in self.agents:
            error_msg = f"Agent '{target_agent_name}' specified in step {step.get('step_id', step_index + 1)} is not initialized or available."
            self.log_callback(error_msg, "error")
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            return None

        self.log_callback(f"Selected agent '{target_agent_name}' for step {step.get('step_id', step_index + 1)}.", "debug")
        return target_agent_name


    async def _execute_single_step(self, step_index: int, step: Dict[str, Any], total_steps: int) -> Dict[str, Any]:
        """
        Executes a single step dictionary from the plan.
        (Docstring remains the same)
        """
        step_id = step.get("step_id", step_index + 1)
        step_action = step.get("action", "No action specified")
        step_params = step.get("parameters", {}) # Original, unresolved params
        step_description_for_log = f"Action: {step_action}, Params: {step_params}"
        self.log_callback(f"Preparing step {step_index + 1}/{total_steps} (ID: {step_id}): {step_description_for_log}", "info")

        step_results_for_this = []
        max_retries = self.config.get('step_max_retries', 1)
        retry_count = 0
        step_succeeded = False

        # --- Agent Selection ---
        target_agent_name = self._select_agent_for_task(step_index, step)
        if target_agent_name is None:
            return {"step_id": step_id, "action": step_action, "status": "error", "message": "No suitable agent found or selected agent not initialized."}
        target_agent = self.agents[target_agent_name]

        while retry_count <= max_retries and not step_succeeded:
            step_failed_this_attempt = False
            context_version_before_step = self.context.get_version()
            self.log_callback(f"Context version before step {step_index + 1} (Attempt {retry_count + 1}): {context_version_before_step}", "debug")

            try:
                # --- Prepare the final input dictionary for the agent ---
                # Start with original action and parameters for this attempt
                task_for_final_agent = {
                    "action": step_action,
                    "parameters": step_params.copy() # Use a copy of original params
                }

                # --- Placeholder Resolution (INSIDE LOOP) ---
                # Modify the 'parameters' within task_for_final_agent directly
                placeholder_pattern = re.compile(r"\[([\w\s]+) from step (\d+)\]")
                previous_step_results = self.context.get("execution.results", [])

                # Use .items() on the copied dictionary to allow modification
                params_to_resolve = task_for_final_agent["parameters"]
                for key, value in list(params_to_resolve.items()): # Iterate over list of items
                    if isinstance(value, str):
                        match = placeholder_pattern.search(value)
                        if match:
                            placeholder_desc = match.group(1).strip()
                            source_step_id = int(match.group(2))
                            self.log_callback(f"Attempt {retry_count + 1}: Resolving placeholder '{value}' for parameter '{key}'", "debug")

                            source_result = None
                            for res in previous_step_results:
                                if res.get("step_id") == source_step_id:
                                    source_result = res
                                    break

                            if source_result:
                                actual_data = None
                                placeholder_desc_lower = placeholder_desc.lower()
                                actual_data = None # Initialize actual_data

                                # --- NEW: Specific Placeholder Logic ---
                                if "line number" in placeholder_desc_lower and source_result.get("action") == "analyze_content" and "analysis_result" in source_result:
                                    analysis_text = source_result.get("analysis_result", "")
                                    # Try to extract a number following "line"
                                    line_num_match = re.search(r'(?:line|error on line)\s+(\d+)', analysis_text, re.IGNORECASE)
                                    if line_num_match:
                                        actual_data = line_num_match.group(1) # Extract the number as string
                                        self.log_callback(f"Attempt {retry_count + 1}: Extracted line number '{actual_data}' from analysis result for placeholder '{value}'.", "debug")
                                    else:
                                        self.log_callback(f"Attempt {retry_count + 1}: Could not extract line number from analysis result for placeholder '{value}'. Analysis: '{analysis_text}'", "warning")
                                # --- END NEW ---
                                else:
                                    # --- Existing Placeholder Logic (as fallback) ---
                                    # Prioritize specific keys based on placeholder description
                                    if ("diff" in placeholder_desc_lower or "code content" in placeholder_desc_lower or "full code" in placeholder_desc_lower) and "result_or_code" in source_result:
                                        actual_data = source_result["result_or_code"]
                                    elif "analysis result" in placeholder_desc_lower and "analysis_result" in source_result:
                                        actual_data = source_result["analysis_result"]
                                    elif "strategy" in placeholder_desc_lower and "strategy" in source_result:
                                        actual_data = source_result["strategy"]
                                    elif "content" in placeholder_desc_lower and "content" in source_result:
                                        actual_data = source_result["content"]
                                    # Fallbacks if specific keys not found or description is generic
                                    if actual_data is None:
                                        if "result_or_code" in source_result: actual_data = source_result["result_or_code"]
                                        elif "analysis_result" in source_result: actual_data = source_result["analysis_result"]
                                        elif "content" in source_result: actual_data = source_result["content"]
                                        elif "message" in source_result: actual_data = source_result["message"]
                                        elif "result" in source_result: actual_data = source_result["result"]
                                    # --- End Existing Placeholder Logic ---

                                # --- Assign resolved data or keep placeholder ---
                                if actual_data is not None:
                                    # Perform substitution within the string if it's the target key
                                    if key == "change_description":
                                         original_string = params_to_resolve[key]
                                         # Use re.sub to replace only the placeholder part
                                         new_value = re.sub(placeholder_pattern, str(actual_data), original_string, count=1)
                                         if new_value != original_string: # Check if substitution happened
                                             self.log_callback(f"Attempt {retry_count + 1}: Substituted placeholder in '{key}'. Original: '{original_string}', New: '{new_value}'", "debug")
                                             params_to_resolve[key] = new_value
                                         else:
                                             self.log_callback(f"Attempt {retry_count + 1}: Placeholder pattern not found for substitution in '{key}'. Keeping original value: '{original_string}'", "warning")
                                    else:
                                         # For other keys, replace the whole value
                                         self.log_callback(f"Attempt {retry_count + 1}: Successfully extracted/resolved data for placeholder '{value}'. Assigning to key '{key}'.", "debug")
                                         params_to_resolve[key] = actual_data
                                else:
                                    self.log_callback(f"Attempt {retry_count + 1}: Extraction failed for placeholder '{value}'. Keeping original placeholder for key '{key}'.", "warning")
                                    # Keep original value (placeholder)
                            else:
                                self.log_callback(f"Attempt {retry_count + 1}: Could not find result for source step {source_step_id} to resolve placeholder '{value}'. Keeping original.", "warning")
                                # Keep original value (placeholder)
                # --- End Placeholder Resolution ---

                # --- Inject Specific Error Context if Applicable ---
                if task_for_final_agent.get("action") == "generate_code_diff":
                    original_task_desc = self.context.get("task", "")
                    # Try to extract specific error info from the original task
                    line_match = re.search(r'line (\d+)', original_task_desc)
                    error_match = re.search(r'(\w+Error):', original_task_desc)
                    file_match = re.search(r'File "([^"]+)"', original_task_desc) # Optional: extract filename too

                    if line_match and error_match:
                        line_num = line_match.group(1)
                        error_type = error_match.group(1)
                        specific_desc = f"Fix {error_type} on line {line_num}"
                        if file_match:
                             # Extract base filename
                             filepath = file_match.group(1)
                             filename = os.path.basename(filepath)
                             specific_desc += f" in {filename}"
                        # Overwrite the planner's potentially generic description
                        current_desc = task_for_final_agent["parameters"].get("change_description", "")
                        if specific_desc not in current_desc: # Avoid redundant appending
                             self.log_callback(f"Injecting specific error context into change_description: '{specific_desc}'", "debug")
                             task_for_final_agent["parameters"]["change_description"] = specific_desc
                # --- End Error Context Injection ---


                # Add retry context if needed
                if retry_count > 0:
                    last_error = "Unknown error"
                    if step_results_for_this:
                        last_error = step_results_for_this[-1].get("message", "Unknown error")
                    task_for_final_agent["retry_context"] = { # Add directly to the final dict
                        "attempt": retry_count + 1,
                        "max_attempts": max_retries + 1,
                        "last_error": last_error
                    }
                    self.log_callback(f"Retrying step {step_index + 1} (ID: {step_id}) (Attempt {retry_count + 1}/{max_retries + 1})", "warning")
                    if step_results_for_this: step_results_for_this.pop()

                # --- Execute Step via Target Agent ---
                log_params_snippet = str(task_for_final_agent.get('parameters', {}))[:200]
                self.log_callback(f"DEBUG PRE-CALL: Passing to agent '{target_agent_name}': action='{task_for_final_agent.get('action')}', params_snippet='{log_params_snippet}...'", "debug")

                step_result: Dict[str, Any] = {}
                if asyncio.iscoroutinefunction(target_agent.execute_task):
                    task_timeout = self.config.get('task_timeout_seconds', 300)
                    self.log_callback(f"Setting timeout for task {step_index + 1} to {task_timeout} seconds.", "debug")
                    step_result = await asyncio.wait_for(
                        target_agent.execute_task(task_for_final_agent), # Pass the correctly prepared dict
                        timeout=task_timeout
                    )
                else:
                    step_result = target_agent.execute_task(task_for_final_agent) # Pass the correctly prepared dict

                self.log_callback(f"Step {step_index + 1} (Attempt {retry_count + 1}) by '{target_agent_name}'. Raw Result Status: {step_result.get('status', 'unknown')}", "debug")

                # Combine step info with agent result
                full_step_result = {
                    "step_id": step_id,
                    "action": step_action,
                    "parameters": step_params, # Log original params for clarity
                    "agent": target_agent_name,
                    "attempt": retry_count + 1, # Log 1-based attempt
                    **(step_result or {})
                }
                step_results_for_this.append(full_step_result)

                # Check status from the agent's result
                if step_result.get("status") == "error":
                    error_message = step_result.get('message', 'No details provided by agent.')
                    self.log_callback(f"Step {step_index + 1} attempt failed (Agent Error): {error_message}", "error")
                    step_failed_this_attempt = True
                    if self.context.rollback(context_version_before_step):
                        self.log_callback(f"Context rolled back to version {context_version_before_step} due to agent-reported step failure.", "info")
                    else:
                        self.log_callback(f"Failed to rollback context to version {context_version_before_step} after agent error.", "error")
                elif step_result.get("status") == "success":
                    self.log_callback(f"Step {step_index + 1} attempt succeeded.", "info")
                    step_succeeded = True
                else:
                    self.log_callback(f"Step {step_index + 1} completed with non-standard status '{step_result.get('status')}' from agent '{target_agent_name}'. Assuming success.", "warning")
                    step_succeeded = True

            except asyncio.TimeoutError:
                self.log_callback(f"Timeout during step {step_index + 1} execution (Attempt {retry_count + 1}) via '{target_agent_name}'.", "error")
                error_msg = f"Task timed out for agent {target_agent_name}"
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                step_results_for_this.append({
                    "step_id": step_id, "action": step_action, "agent": target_agent_name,
                    "attempt": retry_count + 1, "status": "error", "message": error_msg
                })
                step_failed_this_attempt = True
                if self.context.rollback(context_version_before_step):
                    self.log_callback(f"Context rolled back to version {context_version_before_step} due to timeout.", "info")
                else:
                    self.log_callback(f"Failed to rollback context to version {context_version_before_step} after timeout.", "error")

            except Exception as e:
                self.log_callback(f"Unexpected exception during step {step_index + 1} execution (Attempt {retry_count + 1}) via '{target_agent_name}': {e}", "error", exc_info=True)
                error_msg = f"Unhandled exception in agent {target_agent_name} or step processing: {e}"
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                step_results_for_this.append({
                    "step_id": step_id, "action": step_action, "agent": target_agent_name,
                    "attempt": retry_count + 1, "status": "error", "message": error_msg
                })
                step_failed_this_attempt = True
                if self.context.rollback(context_version_before_step):
                    self.log_callback(f"Context rolled back to version {context_version_before_step} due to exception.", "info")
                else:
                    self.log_callback(f"Failed to rollback context to version {context_version_before_step} after exception.", "error")

            if step_failed_this_attempt:
                retry_count += 1

        # --- End Retry Loop ---

        if step_results_for_this:
            return step_results_for_this[-1]
        else:
            self.log_callback(f"Error: Step {step_index + 1} failed to produce any result after {max_retries + 1} attempts.", "critical")
            return {
                 "step_id": step_id, "action": step_action, "agent": target_agent_name,
                 "attempt": retry_count, "status": "error",
                 "message": "Step execution failed critically without producing results."
            }

    def _update_directory_context(self) -> None:
         """Updates the context with current directory listing and base path."""
         if self.tool_engine and self.tool_engine.file_tools.base_dir:
             base_dir_str = str(self.tool_engine.file_tools.base_dir)
             self.context.set("workspace.base_dir", base_dir_str)
             try:
                 self.log_callback(f"Listing files in working directory: {base_dir_str}", "info")
                 list_result = self.tool_engine.file_tools.file_list(path=".", recursive=False)
                 if list_result["status"] == "success":
                     files = list_result.get("files", [])
                     self.context.set("workspace.directory_listing", files)
                     if not files:
                          self.log_callback(f"Working directory {base_dir_str} is empty.", "info")
                 else:
                     error_msg = f"Could not list files in {base_dir_str}: {list_result.get('message', 'Unknown error')}"
                     self.log_callback(error_msg, "warning")
                     current_errors = self.context.get("errors", [])
                     current_errors.append(error_msg)
                     self.context.set("errors", current_errors)
                     self.context.set("workspace.directory_listing", None) # Indicate failure
             except Exception as e:
                 error_msg = f"Critical error during file listing in {base_dir_str}: {e}"
                 self.log_callback(error_msg, "error")
                 current_errors = self.context.get("errors", [])
                 current_errors.append(error_msg)
                 self.context.set("errors", current_errors)
                 self.context.set("workspace.directory_listing", None) # Indicate failure
         elif self.tool_engine:
              warn_msg = "Working directory not set for Tool Engine. Cannot get directory context."
              self.log_callback(warn_msg, "warning")
              current_errors = self.context.get("errors", [])
              current_errors.append(warn_msg)
              self.context.set("errors", current_errors)
              self.context.set("workspace.base_dir", None)
              self.context.set("workspace.directory_listing", None)
         else:
              warn_msg = "Tool Engine not available. Cannot get directory context."
              self.log_callback(warn_msg, "warning")
              current_errors = self.context.get("errors", [])
              current_errors.append(warn_msg)
              self.context.set("errors", current_errors)
              self.context.set("workspace.base_dir", None)
              self.context.set("workspace.directory_listing", None)
         # This method updates context directly, doesn't return the string


    # Removed _cancel_dependent_tasks as sequential execution stops on first failure.

    def _detect_dependency_cycle(self, tasks: List[Dict[str, Any]]) -> bool:
        """Detects cycles in task dependencies using Depth First Search."""
        task_map = {task["id"]: task for task in tasks}
        path = set()  # Tracks nodes currently in the recursion stack
        visited = set() # Tracks nodes that have been fully explored

        def visit(task_id):
            path.add(task_id)
            visited.add(task_id)
            task = task_map.get(task_id)
            if task: # Check if task exists
                for dep_id in task.get("dependencies", []):
                    if dep_id not in task_map: # Skip if dependency doesn't exist (already validated)
                        continue
                    if dep_id in path:
                        # Cycle detected
                        return True
                    if dep_id not in visited:
                        if visit(dep_id):
                            return True
            path.remove(task_id)
            return False

        for task in tasks:
            task_id = task["id"]
            if task_id not in visited:
                if visit(task_id):
                    return True # Cycle found
        return False # No cycles found
    async def _generate_and_validate_plan(self, task: str) -> Optional[List[Dict[str, Any]]]:
        """
        Generates a plan using the primary planner agent and validates its basic structure.

        Args:
            task: The initial task description provided by the user or system.

        Returns:
            A list of step dictionaries representing the plan if generation and
            basic validation are successful. Returns an empty list if the planner
            generates a valid but empty plan. Returns None if any critical error
            occurs during planning or validation. Errors are logged and appended
            to the context.
        """
        primary_agent_name = self.config.get('primary_agent', 'planner')
        if not self.agents:
            self.log_callback("No agents initialized.", "error")
            current_errors = self.context.get("errors", [])
            current_errors.append("No agents available for planning.")
            self.context.set("errors", current_errors)
            return None
        if primary_agent_name not in self.agents:
            error_msg = f"Primary agent '{primary_agent_name}' not found."
            self.log_callback(error_msg, "error")
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            return None

        planner_agent = self.agents[primary_agent_name]
        plan_result: Optional[Dict[str, Any]] = None
        plan_steps: List[Dict[str, Any]] = [] # Expect list of step dicts

        try:
            # --- Update and Use Directory Context ---
            self._update_directory_context() # Update context with dir info
            dir_listing = self.context.get("workspace.directory_listing")
            base_dir = self.context.get("workspace.base_dir")
            directory_context_str = "Directory context not available."
            if base_dir and dir_listing is not None:
                if dir_listing:
                    directory_context_str = f"Current working directory ({base_dir}) contains:\n" + "\n".join(f"- {f}" for f in dir_listing)
                else:
                    directory_context_str = f"Current working directory ({base_dir}) is empty."
            elif base_dir:
                directory_context_str = f"Could not retrieve listing for working directory ({base_dir})."

            task_for_planner = f"{directory_context_str}\n\nUser Task: {task}" # Use the passed task
            self.log_callback(f"Task for planner:\n{task_for_planner}", "debug")

            # --- Call Planner ---
            self.log_callback(f"Delegating to '{primary_agent_name}' for planning.", "info")
            if asyncio.iscoroutinefunction(planner_agent.execute_task):
                plan_result = await planner_agent.execute_task(task_for_planner)
            else:
                plan_result = planner_agent.execute_task(task_for_planner)

            # --- Process Plan ---
            if not plan_result or not isinstance(plan_result, dict):
                error_msg = f"Planner '{primary_agent_name}' did not return a valid plan dictionary. Result: {plan_result}"
                self.log_callback(error_msg, "error")
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                return None

            # Expect the plan as a list of dicts under the "steps" key
            plan_steps = plan_result.get("steps")
            if not isinstance(plan_steps, list):
                 error_msg = f"Planner '{primary_agent_name}' result missing 'steps' list or it's not a list. Result: {plan_result}"
                 self.log_callback(error_msg, "error")
                 current_errors = self.context.get("errors", [])
                 current_errors.append(error_msg)
                 self.context.set("errors", current_errors)
                 return None

            # Basic validation of the list content (are they dicts?)
            if not all(isinstance(step, dict) for step in plan_steps):
                 error_msg = f"Planner '{primary_agent_name}' returned 'steps' list containing non-dictionary items."
                 self.log_callback(error_msg, "error")
                 current_errors = self.context.get("errors", [])
                 current_errors.append(error_msg)
                 self.context.set("errors", current_errors)
                 return None

            if not plan_steps:
                self.log_callback("Planner returned an empty plan (zero steps).", "warning")
                # Store empty plan in context before returning
                self.context.set("plan", [])
                return [] # Return an empty list for a valid but empty plan

            # Add status field to each step for tracking during execution
            for step in plan_steps:
                step["status"] = "pending"
                step["result"] = None

            self.log_callback(f"Planner generated {len(plan_steps)} steps.", "info")
            self.context.set("plan", plan_steps) # Store the list of step dicts

            # TODO: Add more detailed validation of step dictionaries here if needed
            # (e.g., check required keys like 'agent', 'action', 'parameters')
            # For now, basic structure check is done above.

            return plan_steps

        except Exception as e:
            error_msg = f"Error during planning or plan processing phase: {e}"
            self.log_callback(error_msg, "error", exc_info=True)
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            self.context.set("execution.status", "failed") # Mark execution as failed
            return None # Indicate failure
            self.context.set("execution.status", "failed") # Mark execution as failed
            return None # Indicate failure


    async def execute(self, task: str) -> str:
        """
        Main entry point to execute a high-level task sequentially.

        Orchestrates the process:
        1. Generates and validates an execution plan (list of step dicts) using `_generate_and_validate_plan`.
        2. Executes the plan steps sequentially using `_execute_single_step`.
        3. Stops execution if a step fails.
        4. Constructs and returns a final summary string detailing the execution results.

        Args:
            task: The high-level task description string.

        Returns:
            A string summarizing the execution results.
        """
        self.log_callback(f"Orchestrator received task: {task}", "info")
        self.context.set("task", task) # Store initial task
        self.context.set("execution.status", "planning") # Set initial status
        execution_summary = [f"Task: {task}"]

        # --- 1. Generate and Validate Plan ---
        plan_steps = await self._generate_and_validate_plan(task) # Now returns List[Dict] or None/[]

        if plan_steps is None:
            # Error already logged and context updated by helper method
            self.context.set("execution.status", "failed")
            last_error = self.context.get("errors", ["Unknown planning error"])[-1]
            return f"Error: Failed to generate or validate plan. Last error: {last_error}"

        if not plan_steps:
             # Planner returned a valid but empty plan
             self.context.set("execution.status", "completed")
             return "Plan generated successfully, but it contains no steps to execute."

        # --- Log the generated plan ---
        execution_summary.append("\nPlan Steps:")
        for i, step in enumerate(plan_steps):
             step_id = step.get("step_id", i + 1)
             agent = step.get("agent", "N/A")
             action = step.get("action", "N/A")
             params = step.get("parameters", {})
             execution_summary.append(f"  - Step {step_id}: Agent={agent}, Action={action}, Params={params}")
        execution_summary.append("\nExecuting Steps Sequentially...")
        self.log_callback("Plan generation and validation successful.", "info")

        # --- 2. Execute Steps Sequentially ---
        self.context.set("execution.status", "running")
        step_results = [] # List to store results of each step (used for placeholder resolution)
        self.context.set("execution.results", step_results) # Store in context for access in _execute_single_step
        execution_failed = False
        failed_step_index = -1 # Track index of failure
        failed_step_id = None # Track ID of failure
        total_steps = len(plan_steps)

        for i, step in enumerate(plan_steps):
            current_step_id = step.get("step_id", i + 1)
            self.log_callback(f"--- Starting Step {i+1}/{total_steps} (ID: {current_step_id}) ---", "info")
            step_result = await self._execute_single_step(step_index=i, step=step, total_steps=total_steps)
            step_results.append(step_result) # Store the result dict locally
            self.context.set("execution.results", step_results) # Update context with the latest results list for placeholder resolution in the *next* step

            # Update the step status in the context plan
            step["status"] = step_result.get("status", "error")
            step["result"] = step_result # Store full result in the step itself
            self.context.set(f"plan.{i}", step) # Update the specific step in context

            # --- Check for Planner Strategy Decision and Enforce Consistency ---
            agent_name = step_result.get("agent")
            action_name = step_result.get("action")
            if agent_name == "planner" and action_name == "decide_edit_strategy" and step_result.get("status") == "success":
                decided_strategy = step_result.get("strategy")
                self.log_callback(f"Planner decided strategy: {decided_strategy}. Checking plan consistency...", "debug")
                # Find the next two steps (generation and application)
                if i + 2 < total_steps: # Ensure there are at least two steps remaining
                    generation_step_index = i + 1
                    application_step_index = i + 2
                    generation_step = plan_steps[generation_step_index]
                    application_step = plan_steps[application_step_index]

                    # Determine expected actions based on strategy
                    expected_generation_action = "generate_full_code" if decided_strategy == "write_file" else "generate_code_diff"
                    expected_application_action = "write_file" if decided_strategy == "write_file" else "apply_diff"

                    # Check and potentially modify the generation step
                    if generation_step.get("agent") == "coder" and generation_step.get("action") != expected_generation_action:
                        self.log_callback(f"Plan Inconsistency: Forcing step {generation_step_index + 1} action from '{generation_step.get('action')}' to '{expected_generation_action}' based on decided strategy '{decided_strategy}'.", "warning")
                        generation_step["action"] = expected_generation_action
                        if "parameters" in generation_step and isinstance(generation_step["parameters"], dict):
                             generation_step["parameters"]["strategy"] = decided_strategy # Also update strategy param
                        self.context.set(f"plan.{generation_step_index}", generation_step) # Update context

                    # Check and potentially modify the application step
                    current_app_action = application_step.get("action")
                    if application_step.get("agent") == "coder" and current_app_action != expected_application_action:
                         self.log_callback(f"Plan Inconsistency: Forcing step {application_step_index + 1} action from '{current_app_action}' to '{expected_application_action}' based on decided strategy '{decided_strategy}'.", "warning")
                         application_step["action"] = expected_application_action
                         # --- Rename parameter key based on action change ---
                         if "parameters" in application_step and isinstance(application_step["parameters"], dict):
                              params = application_step["parameters"]
                              if expected_application_action == "write_file" and "diff" in params:
                                   self.log_callback(f"Plan Inconsistency: Renaming parameter 'diff' to 'content' for step {application_step_index + 1}.", "debug")
                                   params["content"] = params.pop("diff") # Rename diff to content
                              elif expected_application_action == "apply_diff" and "content" in params:
                                   self.log_callback(f"Plan Inconsistency: Renaming parameter 'content' to 'diff' for step {application_step_index + 1}.", "debug")
                                   params["diff"] = params.pop("content") # Rename content to diff
                         # --- End parameter rename ---
                         self.context.set(f"plan.{application_step_index}", application_step) # Update context
                else:
                     self.log_callback(f"Not enough subsequent steps found after decide_edit_strategy (step {i+1}) to enforce consistency.", "warning")


            # --- Check Step Status and Stop on Error ---
            if step_result.get("status") == "error":
                self.log_callback(f"--- Step {i+1}/{total_steps} (ID: {current_step_id}) FAILED. Stopping execution. ---", "error")
                execution_failed = True
                failed_step_index = i
                failed_step_id = current_step_id
                break # Stop execution on first failure
            else:
                 self.log_callback(f"--- Step {i+1}/{total_steps} (ID: {current_step_id}) COMPLETED Successfully ---", "info")

        # Store all results in context under execution.results (optional, as results are in plan steps now)
        # self.context.set("execution.results", {res.get("step_id"): res for res in step_results if "step_id" in res})

    # Removed _process_completed_task as sequential execution is now sequential.
    # Removed _run_execution_loop as execution is now sequential.

        # --- 3. Refinement Check (Placeholder) ---
        if execution_failed:
            # failed_step_index and failed_step_id are set in the loop above
            feedback = "Unknown error"
            if failed_step_index >= 0 and failed_step_index < len(step_results):
                 feedback = step_results[failed_step_index].get("message", "Step failed with no message.")

            self.log_callback(f"Execution stopped due to failure at step {failed_step_index + 1} (ID: {failed_step_id}).", "warning")
            # TODO: Implement plan refinement logic here if desired.
            # Could call planner_agent.refine_plan(...) with plan_steps, feedback, failed_step_index etc.
            # For now, we just report the failure.

        # --- 4. Final Summary ---
        execution_summary.append("\nExecution Results:")
        all_steps_succeeded = not execution_failed

        for i, res in enumerate(step_results):
            # Get info from the result dictionary generated by _execute_single_step
            step_id = res.get("step_id", i + 1)
            agent = res.get("agent", "N/A")
            action = res.get("action", "N/A")
            status = res.get("status", "unknown")
            message = res.get("message", "")

            summary_line = f"- Step {step_id}: Agent={agent}, Action={action} -> Status: {status}"
            if message and status == "error":
                summary_line += f" -> Message: {message}"
            # Add more details from result if needed, e.g., file paths written
            elif status == "success":
                 # Example: Check for specific keys added by agents/tools
                 if "file_path" in res: # Example key from a file tool
                     summary_line += f" -> File: {res['file_path']}"
                 elif "output" in res: # Example key from command execution
                     output_str = str(res['output'])
                     summary_line += f" -> Output: {output_str[:50]}{'...' if len(output_str) > 50 else ''}" # Truncate

            execution_summary.append(summary_line)

        # Add summary for steps that were not executed due to failure
        if execution_failed:
            start_index = len(step_results) # Index of the first skipped step
            for i in range(start_index, total_steps):
                 step = plan_steps[i] # Get the original step dict from the plan
                 step_id = step.get("step_id", i + 1)
                 agent = step.get("agent", "N/A")
                 action = step.get("action", "N/A")
                 execution_summary.append(f"- Step {step_id}: Agent={agent}, Action={action} -> Status: skipped (due to previous failure)")

        # --- Store final execution state ---
        final_status = "completed" if all_steps_succeeded else "failed"
        self.context.set("execution.status", final_status)
        # Store the results directly in the plan steps within the context (already done in the loop)
        # Optionally store failed step ID if needed elsewhere
        if execution_failed:
             self.context.set("execution.failed_step_id", failed_step_id)
        else:
             self.context.set("execution.failed_step_id", None) # Clear if successful

        if all_steps_succeeded:
            execution_summary.append("\nPlan executed successfully.")
        else:
            execution_summary.append(f"\nPlan execution failed at step {failed_step_index + 1} (ID: {failed_step_id}).")
            # TODO: Add refinement logic trigger here if needed

        final_output = "\n".join(execution_summary)
        self.log_callback(f"Orchestration complete. Final Status: {final_status}", "info")
        return final_output

    def _infer_capability_from_task(self, task_description: str) -> Optional[str]:
        """
        Basic inference of required capability based on keywords in the task description.
        This is a simple implementation and can be expanded significantly.
        """
        task_lower = task_description.lower()
        # Order matters: more specific checks first
        if "write" in task_lower and "file" in task_lower: return "file_write"
        if "read" in task_lower and "file" in task_lower: return "file_read"
        if "list" in task_lower and ("files" in task_lower or "directory" in task_lower): return "file_list"
        if "create" in task_lower and "directory" in task_lower: return "directory_create"
        if "delete" in task_lower and ("file" in task_lower or "directory" in task_lower): return "file_delete" # Needs refinement
        if "execute" in task_lower and ("command" in task_lower or "script" in task_lower): return "command_execution"
        if "code" in task_lower or "implement" in task_lower or "function" in task_lower or "class" in task_lower or "script" in task_lower or "python" in task_lower or "javascript" in task_lower: return "coding"
        if "debug" in task_lower or "fix" in task_lower or "error" in task_lower: return "debugging"
        if "plan" in task_lower or "steps" in task_lower or "decompose" in task_lower: return "planning"
        if "verify" in task_lower or "check" in task_lower or "analyze" in task_lower or "validate" in task_lower: return "verification"
        if "decide" in task_lower or "choose" in task_lower or "select" in task_lower: return "decision_making"
        # Add more rules as needed
        self.log_callback(f"Capability inference: No specific capability matched for task: '{task_description}'", "debug")
        return None # Default if no specific capability inferred
    
    def process_agent_output(self, agent_output: str, agent_name: str) -> str:
        """Process output from an agent.
        
        Args:
            agent_output: Output from the agent.
            agent_name: Name of the agent.
            
        Returns:
            Processed output.
        """
        # In a real implementation, this would process the agent's output
        # For this stub implementation, we'll just return the output
        self.log_callback(f"Processing output from agent {agent_name}", "info") # Use callback
        return agent_output
        
    def setup_agent_tools(self, tool_executor: ToolExecutionEngine):
        """Provides the tool executor to all initialized agents and stores it."""
        # Use direct logger for debugging this specific issue
        self.log_callback(f"Orchestrator: Entered setup_agent_tools. tool_executor is {'None' if not tool_executor else 'provided'}", "debug")
        self.tool_engine = tool_executor # Store the tool engine instance
        self.log_callback("Orchestrator: Assigned self.tool_engine.", "debug")
        if not tool_executor:
            self.log_callback("Orchestrator: Tool executor is None, returning.", "warning")
            # self.log_callback("Tool executor not provided, agents will not have tool access.", "warning") # Keep original callback commented for now
            return

        self.log_callback("Orchestrator: Tool executor is valid. About to log 'Setting up tools...'", "debug")
        # Use logger first to see if callback is the issue, then try original callback if needed
        self.log_callback("Orchestrator: Setting up tools for all agents...", "info")
        # self.log_callback("Setting up tools for all agents...", "info") # Keep original callback commented

        self.log_callback("Orchestrator: About to access self.agents.items().", "debug")
        try:
            agent_items = list(self.agents.items()) # Try accessing items outside loop
            self.log_callback(f"Orchestrator: Accessed self.agents. Found {len(agent_items)} agents.", "debug")
        except Exception as access_e:
            self.log_callback(f"Orchestrator: FAILED to access self.agents.items(): {access_e}", "error", exc_info=True)
            # Explicitly set context error here if accessing agents fails critically
            # Append error to the list
            error_msg = f"Failed to access agent list during tool setup: {access_e}"
            current_errors = self.context.get("errors", [])
            current_errors.append(error_msg)
            self.context.set("errors", current_errors)
            return # Stop further execution

        self.log_callback("Orchestrator: Starting loop over agents.", "debug")
        for agent_name, agent in agent_items:
            self.log_callback(f"Orchestrator: Setting up tools for agent '{agent_name}'.", "debug")
            try:
                if hasattr(agent, 'setup_tools') and callable(getattr(agent, 'setup_tools')):
                    agent.setup_tools(tool_executor)
                    self.log_callback(f"Orchestrator: Successfully called setup_tools for '{agent_name}'.", "debug")
                else:
                    self.log_callback(f"Agent '{agent_name}' of type {type(agent).__name__} does not have a setup_tools method.", "warning")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                log_message = f"CRITICAL FAILURE during setup_tools for agent '{agent_name}': {e}\nTraceback:\n{error_details}"
                self.log_callback(log_message, "error")
                # Append error to context if setup fails for an agent
                error_msg = f"Failed to setup tools for agent '{agent_name}': {e}"
                current_errors = self.context.get("errors", [])
                current_errors.append(error_msg)
                self.context.set("errors", current_errors)
                # Decide whether to raise or just log and continue
                # raise RuntimeError(f"Tool setup failed for agent '{agent_name}'") from e
