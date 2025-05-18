"""Coder agent implementation."""

from typing import Any, Dict, Optional, List
from ..core.llm_service_gemini import BaseLLMService
from ..core.context_manager import ContextManager
from .base import BaseAgent, register_agent, CommunicationBus # Import CommunicationBus for type hint
import traceback # For better error reporting
import os # For file operations
import re # For parsing filenames from task description
import logging # Import logging
import json # For parsing LLM response for diff info

logger = logging.getLogger(__name__) # Setup logger for CoderAgent

@register_agent("coder")
class CoderAgent(BaseAgent):
    """Agent responsible for writing and maintaining code, and executing simple tasks."""

    # Regex Constants
    # Removed old regex constants for task parsing
    _RE_KEYWORD = re.compile(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b') # Added missing regex constant

    # --- Prompt Templates ---
    CODE_GENERATION_PROMPT = """
You are an expert Coder Agent in a multi-agent system. Your task is to generate code, execute commands, or provide direct answers based on the given task description.
Target Languages: {languages}

Current Directory Files (if relevant):
{file_list}

Current Plan Context:
{plan_context}

Current Step in Plan: {current_step_number} of {total_steps}
Current Task: {task}

Previous Steps Completed:
{previous_steps}

Code Context (Code read from a file in a previous step, if relevant):
{code_context}

Instructions:
1. Analyze the task description carefully in the context of the overall plan.
2. **For direct questions or calculations:** Provide the direct result or answer without generating code (e.g., for "Calculate 5 + 7", output "12").
3. **For code generation tasks:**
   a. Write clean, efficient, and well-commented code in one of the target languages ({languages}).
   b. Adhere to the plan steps and consider how your current task fits into the overall plan.
   c. Consider the Current Directory Files if the task involves modifying existing code.
   d. Output ONLY the raw code without markdown formatting unless specifically requested.
   e. CRITICAL: When the task involves writing to a specific file (e.g., 'write to main.py', 'save as script.js'), you MUST generate code in the language corresponding to the file extension (.py for Python, .js for JavaScript, .html for HTML, .css for CSS, etc.). This overrides the general target languages if the file extension implies a different language.

4. **For command execution tasks:**
   a. If the task contains "execute command" or "run command" followed by a command in quotes, execute that command directly.
   b. For file operations like deletion, use the appropriate system command.
   c. DO NOT write code to perform the command - instead, identify it as a command to be executed.
5. **For file operations:**
   a. If asked to read, write, or modify files, use the appropriate tools (file_read, file_write, file_append).
   b. Before writing to an existing file, analyze if the task requires minor changes or a complete rewrite. If minor changes are needed, prefer using the 'apply_diff' tool. If a complete rewrite is necessary, use the 'file_write' tool.
   c. When writing code to a file, ensure it's properly formatted for the target language.
   d. For file deletion, use "execute_command" with the appropriate system command (e.g., 'del file.txt' on Windows).
   e. **CRITICAL: Using 'apply_diff' for modifications:**
      - If the task requires modifying an existing file with minor changes, you MUST use the 'apply_diff' tool.
      - **Workflow:**
        1. **Read:** Use the 'read_file' tool to get the *exact* current content and line numbers of the target file. This is MANDATORY before generating the diff.
        2. **Generate Diff Info:** Use 'generate_code_diff' action. The LLM will identify the start/end lines and provide the *new code snippet* in a JSON structure.
        3. **Apply:** Use the 'apply_diff' tool. It will retrieve the diff info from context, read the file again, construct the required `<<<<<<< SEARCH...` block programmatically using the actual file content and the new code snippet, and then apply it.
      - **DO NOT:**
        - Ask the LLM to generate the full `<<<<<<< SEARCH...` block directly.
        - Use 'apply_diff' without first running 'generate_code_diff' to get the structured diff info.
6. **If the task is unclear:** Respond with "Error: Task is unclear. Please provide more details."

Examples:
1. Task: "Calculate 100 / 4"
   Output: 25.0

2. Task: "Write a Python function to add two numbers."
   Output:
   def add_numbers(a, b):
     \"\"\"Adds two numbers together.\"\"\\
     return a + b

3. Task: "Execute command 'echo hello world'"
   Action: [Execute the command directly using the execute_command tool]
   Output: [Command execution result]

4. Task: "Delete the file example.txt"
   Action: [Execute the appropriate system command using execute_command tool]
   Output: [Command execution result]

Now, fulfill the task based on the description and plan context.
Output:
"""

    CODE_REFACTORING_PROMPT = """
You are a Code Refactoring Agent. Your task is to improve the given code based on the provided feedback or general best practices.
Target Languages: {languages}

Original Code:

{original_code}


Feedback/Goal: {feedback}

Instructions:
1. Analyze the original code and the feedback/goal.
2. Refactor the code to address the feedback, improve clarity, efficiency, and maintainability in {languages}.
3. Ensure the core functionality remains the same unless the feedback specifies changes.
4. Output ONLY the refactored code. Do not include explanations, introductions, or markdown formatting like  ... .

Refactored Code:
"""
    # --- End Prompt Templates ---
    def __init__(self, name: str, config: Dict[str, Any], llm_service: BaseLLMService, context: ContextManager, base_dir: str, log_callback: Optional[callable] = None, communication_bus: Optional[CommunicationBus] = None): # Added base_dir
        super().__init__(name, config, llm_service, context, base_dir, log_callback, communication_bus=communication_bus) # Pass base_dir to super
        self.languages = config.get("languages", ["python"])
        self.use_tools = config.get("use_tools", True)
        self._log_callback(f"CoderAgent initialized for languages: {self.languages}, tools enabled: {self.use_tools}", "info")

    async def verify_file_content(self, filename: str, expected_content_type: str) -> Dict[str, Any]:
        """
        Verifies if a file contains the expected type of content.

        Args:
            filename: The name of the file to verify
            expected_content_type: Description of expected content (e.g., "snake game", "calculator")

        Returns:
            Dictionary with verification result and content if available
        """
        self._log_callback(f"Verifying if {filename} contains {expected_content_type}...", "info")

        try:
            # Read the file content
            read_result = self.use_tool("file_read", filename=filename) # CORRECTED

            if read_result.get("status") != "success":
                error_msg = f"Failed to read {filename} for verification: {read_result.get('message', 'Unknown error')}"
                self._log_callback(error_msg, "error")
                raise RuntimeError(error_msg)

            content = read_result.get("content", "")

            # Store content in context
            self.context.set(f"file_content_{filename}", content)

            # If file is empty or very small, it's likely not what we're looking for
            if len(content) < 50:
                self._log_callback(f"File {filename} is too small ({len(content)} bytes) to be a {expected_content_type}", "warning")
                return {
                    "verified": False,
                    "status": "success",
                    "content": content,
                    "message": f"File is too small to be a {expected_content_type}"
                }

            # Use LLM to verify content type
            prompt = f"""
            Analyze the following code and determine if it contains a {expected_content_type} implementation.
            Respond with ONLY "YES" or "NO".


            {content[:2000]}  # Limit to first 2000 chars for LLM

            """

            verification = self._call_llm(prompt, temperature=0.1).strip().upper()
            verified = "YES" in verification

            self._log_callback(f"Verification result for {filename}: {'CONTAINS' if verified else 'DOES NOT CONTAIN'} {expected_content_type}", "info")

            return {
                "verified": verified,
                "status": "success",
                "content": content,
                "message": f"File {'contains' if verified else 'does not contain'} {expected_content_type}"
            }

        except Exception as e:
            error_msg = f"Error during verification of {filename}: {e}"
            self._log_callback(error_msg, "error")
            raise RuntimeError(error_msg) from e

    def _extract_plan_context(self, current_plan: Dict[str, Any], current_task: str) -> Dict[str, Any]:
        """
        Extracts relevant plan context for the current task.

        Args:
            current_plan: The current plan dictionary from context
            current_task: The current task description

        Returns:
            Dictionary with plan context information
        """
        if not current_plan or not isinstance(current_plan, dict):
            return {
                "plan_context": "No plan available",
                "current_step_number": "N/A",
                "total_steps": "N/A",
                "previous_steps": "No previous steps available"
            }

        steps = current_plan.get("steps", [])
        if not steps:
            return {
                "plan_context": "Plan exists but has no steps",
                "current_step_number": "N/A",
                "total_steps": "N/A",
                "previous_steps": "No steps in plan"
            }

        # Find current step in plan
        current_step_index = -1
        for i, step in enumerate(steps):
            # Strip agent prefix for comparison
            step_content = step
            if ":" in step:
                _, step_content = step.split(":", 1)
                step_content = step_content.strip()

            # Check if current task is similar to this step
            # Use a more flexible comparison to handle slight variations
            if self._is_similar_task(current_task, step_content):
                current_step_index = i
                break

        # If we couldn't find the step, default to the last step
        if current_step_index == -1:
            current_step_index = len(steps) - 1

        # Get previous steps (limit to 5 for context)
        previous_steps = []
        for i in range(max(0, current_step_index - 5), current_step_index):
            previous_steps.append(f"{i+1}. {steps[i]}")

        return {
            "plan_context": "\n".join(steps),
            "current_step_number": str(current_step_index + 1),
            "total_steps": str(len(steps)),
            "previous_steps": "\n".join(previous_steps) if previous_steps else "No previous steps"
        }

    def _is_similar_task(self, task1: str, task2: str) -> bool:
        """
        Determines if two task descriptions are similar enough to be considered the same step.

        Args:
            task1: First task description
            task2: Second task description

        Returns:
            True if tasks are similar, False otherwise
        """
        # Remove agent prefixes if present
        if ":" in task1:
            _, task1 = task1.split(":", 1)
            task1 = task1.strip()

        if ":" in task2:
            _, task2 = task2.split(":", 1)
            task2 = task2.strip()

        # Normalize tasks for comparison
        task1 = task1.lower().strip()
        task2 = task2.lower().strip()

        # Check for exact match
        if task1 == task2:
            return True

        # Check if one is a substring of the other
        if task1 in task2 or task2 in task1:
            return True

        # Check for keyword similarity using precompiled regex
        keywords1 = set(self._RE_KEYWORD.findall(task1))
        keywords2 = set(self._RE_KEYWORD.findall(task2))

        # If they share at least 60% of keywords, consider them similar
        if len(keywords1) > 0 and len(keywords2) > 0:
            common_keywords = keywords1.intersection(keywords2)
            similarity = len(common_keywords) / min(len(keywords1), len(keywords2))
            return similarity >= 0.6

        return False

    async def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles code-related tasks based on structured input from the orchestrator.

        Args:
            task_input: A dictionary containing 'action' and 'parameters'.
                        Example: {"action": "read_file", "parameters": {"path": "main.py"}}

        Returns:
            Dictionary containing the result and metadata.

        Raises:
            RuntimeError: If a critical error occurs during task execution.
            ValueError: If the input format is invalid.
        """
        # --- ADDED DEBUG LOG ---
        self._log_callback(f"DEBUG CODER execute_task: Received raw task_input: {str(task_input)[:500]}...", "debug")
        # --- END ADDED DEBUG LOG ---
        if not isinstance(task_input, dict):
            # Log the error before raising
            error_msg = f"CoderAgent expected a dictionary task_input, but received {type(task_input)}"
            self._log_callback(error_msg, "error")
            raise ValueError(error_msg)

        action = task_input.get("action")
        parameters = task_input.get("parameters", {}) # Default to empty dict if missing
        retry_context = task_input.get("retry_context") # Get retry context if present

        if not action or not isinstance(action, str):
             raise ValueError("CoderAgent task_input dictionary must contain a valid 'action' string.")
        if not isinstance(parameters, dict):
             raise ValueError("CoderAgent task_input 'parameters' must be a dictionary.")

        action = action.lower() # Normalize action name
        self._log_callback(f"Executing action: '{action}' with parameters: {parameters}", "info")
        if retry_context:
            self._log_callback(f"Retry context: {retry_context}", "debug") # Log retry context

        # --- Check for incoming messages ---
        await self._process_incoming_messages()

        try:
            # --- Action Dispatching ---
            if not self.use_tools or not self.tool_executor:
                self._log_callback("Tools are disabled or not configured. Cannot execute action.", "error")
                return {"action": action, "status": "error", "message": "Tools are disabled or not configured."}

            # Dispatch based on the 'action' key
            if action == "read_file":
                return await self._handle_file_read_task(parameters)
            elif action == "write_file":
                return await self._handle_file_write_task(parameters)
            elif action == "execute_command":
                return await self._handle_command_execution_task(parameters)
            elif action == "apply_diff":
                return await self._handle_apply_diff_task(parameters)
            elif action == "analyze_content":
                 return await self._handle_analyze_content_task(parameters)
            elif action == "generate_code_diff":
                 return await self._handle_generate_code_task(action, parameters) # Use generic LLM handler
            elif action == "generate_full_code":
                 return await self._handle_generate_code_task(action, parameters) # Use generic LLM handler
            elif action == "file_exists":
                 return await self._handle_file_exists_task(parameters)
            elif action == "list_files":
                 return await self._handle_list_files_task(parameters)
            elif action == "verify_output":
                 return await self._handle_verify_output_task(parameters) # Needs implementation
            # Add more actions as needed...
            else:
                self._log_callback(f"Unknown or unsupported action '{action}'.", "error")
                return {"action": action, "status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            error_msg = f"Coder error processing action '{action}': {e}"
            self._log_callback(error_msg, "error")
            logger.exception("Traceback for coder error:") # Log full traceback
            # Return error dict instead of raising, Orchestrator handles status
            return {"action": action, "status": "error", "message": error_msg}

    # Removed _extract_core_task as input is now structured

    async def _process_incoming_messages(self):
        """Processes any pending messages from the communication bus."""
        if self.communication_bus:
            while True:
                message = await self.communication_bus.get_message(self.name, timeout=0)
                if message:
                    self._log_callback(f"Received message via bus: {message}", "debug")
                    # TODO: Implement message handling logic
                else:
                    break

    async def _handle_file_read_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'read_file' action."""
        filename_to_read = parameters.get("path") # Planner uses 'path'
        if not filename_to_read or not isinstance(filename_to_read, str):
             return {"action": "read_file", "status": "error", "message": "Missing or invalid 'path' parameter for read_file."}
        self._log_callback(f"Identified file read task for: '{filename_to_read}'", "info")
        try:
            # Corrected parameter name from 'path' to 'filename' for use_tool
            read_result = self.use_tool("file_read", filename=filename_to_read) # CORRECTED
            self._log_callback(f"File read result: {read_result}", "info")

            if read_result.get("status") == "success":
                content = read_result.get("content")
                if content is not None:
                    self.context.set(f"file_content_{filename_to_read}", content)
                    self._log_callback(f"Stored content for '{filename_to_read}' in context.", "debug")
                else:
                    self._log_callback(f"File read successful for '{filename_to_read}', but content was None.", "warning")

            # Return the result from the tool directly, adding action info
            return {"action": "read_file", **read_result}
        except Exception as e:
            self._log_callback(f"Error using file_read tool for '{filename_to_read}': {e}", "error")
            # Raise a specific error for file read failure
            raise RuntimeError(f"Failed to read file '{filename_to_read}': {e}") from e

    async def _handle_file_write_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'write_file' action."""
        filename_to_write = parameters.get("path") # Planner uses 'path'
        content_to_write = parameters.get("content") # Expect content directly in parameters

        if not filename_to_write or not isinstance(filename_to_write, str):
             return {"action": "write_file", "status": "error", "message": "Missing or invalid 'path' parameter for write_file."}
        # Allow empty content, but check if it's None
        if content_to_write is None or not isinstance(content_to_write, str):
             return {"action": "write_file", "status": "error", "message": "Missing or invalid 'content' parameter (must be string) for write_file."}

        self._log_callback(f"Identified file write task for: '{filename_to_write}'. Retrieving content...", "debug")

        # Check if content is a placeholder (e.g., "[Generated code from step X]")
        # A simple check for brackets is used here. More robust regex could be added if needed.
        is_placeholder = isinstance(content_to_write, str) and content_to_write.startswith('[') and content_to_write.endswith(']')

        if is_placeholder:
            self._log_callback(f"Content parameter '{content_to_write}' looks like a placeholder. Attempting to retrieve actual content from context key 'latest_generated_code'.", "info")
            generated_artifact = self.context.get("latest_generated_code")
            if generated_artifact and isinstance(generated_artifact, dict) and "result_or_code" in generated_artifact:
                actual_content = generated_artifact.get("result_or_code")
                if isinstance(actual_content, str):
                    content_to_write = actual_content # Replace placeholder with actual content
                    self._log_callback(f"Successfully retrieved actual content (length {len(content_to_write)}) from context.", "debug")
                else:
                    self._log_callback(f"Error: Content retrieved from 'latest_generated_code' is not a string (type: {type(actual_content)}). Using placeholder value.", "error")
                    # Optionally return error here, or proceed with placeholder content
                    return {"action": "write_file", "status": "error", "message": "Failed to retrieve valid string content from context for placeholder."}
            else:
                self._log_callback(f"Error: Could not find valid 'latest_generated_code' artifact in context to resolve placeholder '{content_to_write}'.", "error")
                # Optionally return error here, or proceed with placeholder content
                return {"action": "write_file", "status": "error", "message": f"Failed to resolve content placeholder '{content_to_write}' from context."}
        else:
             self._log_callback(f"Received direct content (length {len(content_to_write)}) for writing to '{filename_to_write}'.", "debug")


        try:
            content_to_write_cleaned = self._clean_code_content(content_to_write) # Clean the potentially resolved content
            self._log_callback(f"Calling file_write for '{filename_to_write}' with cleaned content length: {len(content_to_write_cleaned)}", "debug")
            # Corrected parameter name from 'path' to 'filename' for use_tool
            write_result = self.use_tool("file_write", filename=filename_to_write, content=content_to_write_cleaned) # CORRECTED
            self._log_callback(f"File write result: {write_result}", "info")

            # Return the result from the tool directly, adding action info
            return {"action": "write_file", "filename_written": filename_to_write, **write_result}
        except Exception as e:
            self._log_callback(f"Error during file_write tool call for '{filename_to_write}': {e}", "error")
            # Raise RuntimeError for consistency with other tool handlers
            raise RuntimeError(f"Failed to write file '{filename_to_write}': {e}") from e

    def _clean_code_content(self, content: str) -> str:
        """Cleans code content by removing standard markdown code fences."""
        cleaned = content.strip()

        # Remove starting fence (```python or ```)
        start_fence_removed = False
        if cleaned.startswith("```python"):
            cleaned = cleaned[len("```python"):].lstrip('\n\r') # Remove fence and leading newline
            start_fence_removed = True
        elif cleaned.startswith("```"):
            # Remove ``` and potentially a language identifier on the same line
            first_line_end = cleaned.find('\n')
            if first_line_end != -1:
                 # Check if the first line after ``` seems like just an identifier
                 potential_identifier = cleaned[3:first_line_end].strip()
                 # Only strip if it looks like a simple identifier (alphanumeric, short)
                 if potential_identifier and len(potential_identifier) < 15 and potential_identifier.isalnum():
                     cleaned = cleaned[first_line_end + 1:] # Skip the identifier line
                 else:
                     cleaned = cleaned[3:] # Just remove the fence
            else:
                 cleaned = cleaned[3:] # Only fence was present
            start_fence_removed = True

        # Remove ending fence (```) only if a starting fence was likely removed
        # This prevents accidentally removing ``` if it's part of the actual code
        if start_fence_removed and cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip('\n\r') # Remove fence and trailing newline

        return cleaned.strip() # Final strip for safety

    async def _handle_command_execution_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'execute_command' action."""
        command_to_run = parameters.get("command")
        if not command_to_run or not isinstance(command_to_run, str):
             return {"action": "execute_command", "status": "error", "message": "Missing or invalid 'command' parameter for execute_command."}
        self._log_callback(f"Identified command execution task: '{command_to_run}'", "info")
        try:
            # execute_command tool uses 'command' parameter, which is correct
            command_result = self.use_tool("execute_command", command=command_to_run)
            self._log_callback(f"Command execution result: {command_result}", "info")
            # Return the result from the tool directly, adding action info
            return {"action": "execute_command", **command_result}
        except Exception as e:
            self._log_callback(f"Error using execute_command tool for '{command_to_run}': {e}", "error")
            raise RuntimeError(f"Failed to execute command '{command_to_run}': {e}") from e

    async def _handle_apply_diff_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles the 'apply_diff' action.
        Retrieves structured diff info (start_line, end_line, new_code) from context,
        reads the file, constructs the diff block, and applies it.
        """
        filename_to_apply = parameters.get("path") # Planner uses 'path'
        diff_placeholder = parameters.get("diff") # Expect placeholder like "[Diff info from step X]"

        if not filename_to_apply or not isinstance(filename_to_apply, str):
             return {"action": "apply_diff", "status": "error", "message": "Missing or invalid 'path' parameter for apply_diff."}
        if not diff_placeholder or not isinstance(diff_placeholder, str) or not (diff_placeholder.startswith('[') and diff_placeholder.endswith(']')):
             return {"action": "apply_diff", "status": "error", "message": "Missing or invalid 'diff' placeholder parameter (e.g., '[Diff info from step X]')."}

        self._log_callback(f"Identified apply diff task for: '{filename_to_apply}'. Retrieving diff info...", "info")

        # Retrieve the structured diff info from context
        generated_artifact = self.context.get("latest_generated_code")
        if not generated_artifact or not isinstance(generated_artifact, dict) or generated_artifact.get("status") != "generated_diff_info":
            self._log_callback(f"Error: Could not find valid 'generated_diff_info' artifact in context key 'latest_generated_code' to resolve placeholder '{diff_placeholder}'.", "error")
            return {"action": "apply_diff", "status": "error", "message": f"Failed to resolve diff info placeholder '{diff_placeholder}' from context. Expected status 'generated_diff_info'."}

        diff_info = generated_artifact.get("diff_info")
        if not diff_info or not isinstance(diff_info, dict) or not all(k in diff_info for k in ["start_line", "end_line", "new_code"]):
            self._log_callback(f"Error: Retrieved artifact has missing or invalid 'diff_info' structure: {diff_info}", "error")
            return {"action": "apply_diff", "status": "error", "message": "Retrieved diff info from context is missing required keys (start_line, end_line, new_code)."}

        try:
            start_line = int(diff_info["start_line"])
            end_line = int(diff_info["end_line"])
            new_code = diff_info["new_code"] # Should already be a string
            if not isinstance(new_code, str):
                 raise ValueError("new_code in diff_info is not a string.")
            if start_line <= 0 or end_line < start_line:
                 raise ValueError(f"Invalid line numbers in diff_info: start={start_line}, end={end_line}")

            self._log_callback(f"Retrieved diff info: Replace lines {start_line}-{end_line}.", "debug")

        except (ValueError, TypeError) as e:
            self._log_callback(f"Error: Invalid data types or values in retrieved diff_info: {diff_info}. Error: {e}", "error")
            return {"action": "apply_diff", "status": "error", "message": f"Invalid data in retrieved diff info: {e}"}

        # Read the current file content to construct the SEARCH block
        try:
            self._log_callback(f"Reading current content of '{filename_to_apply}' to construct diff.", "info")
            # Use the file_read tool internally (or direct read if preferred/safer)
            # Using internal read for simplicity here, assuming agent has read access
            full_path = self._get_full_path(filename_to_apply) # Use internal helper to resolve path
            if not os.path.exists(full_path):
                 return {"action": "apply_diff", "status": "error", "message": f"File '{filename_to_apply}' not found for reading."}
            with open(full_path, 'r', encoding='utf-8') as f:
                current_lines = f.readlines()

        except Exception as e:
            self._log_callback(f"Error reading file '{filename_to_apply}' to construct diff: {e}", "error")
            return {"action": "apply_diff", "status": "error", "message": f"Failed to read file '{filename_to_apply}' before applying diff: {e}"}

        # Extract the original content block
        start_index = start_line - 1
        end_index = end_line # Slice index is exclusive
        if start_index < 0 or end_index > len(current_lines) or start_index >= end_index:
             self._log_callback(f"Error: Line numbers {start_line}-{end_line} are out of bounds for file '{filename_to_apply}' (length {len(current_lines)}).", "error")
             return {"action": "apply_diff", "status": "error", "message": f"Line numbers {start_line}-{end_line} out of bounds for file '{filename_to_apply}'."}

        original_content = "".join(current_lines[start_index:end_index])

        # Construct the diff string in the required format
        constructed_diff = f"""<<<<<<< SEARCH
:start_line:{start_line}
:end_line:{end_line}
-------
{original_content}=======
{new_code}
>>>>>>> REPLACE"""

        self._log_callback(f"Constructed diff block for lines {start_line}-{end_line}.", "debug")

        # Apply the constructed diff using the tool
        try:
            self._log_callback(f"Attempting to apply constructed diff to '{filename_to_apply}' using 'apply_diff' tool.", "info")
            apply_result = self.use_tool("apply_diff", filename=filename_to_apply, diff=constructed_diff)
            self._log_callback(f"Apply diff result: {apply_result}", "info")

            # --- ADDED CHECK ---
            if apply_result.get("status") != "success":
                error_msg = f"Apply diff tool failed for '{filename_to_apply}': {apply_result.get('message', 'Unknown error from tool')}"
                self._log_callback(error_msg, "error")
                # Return an error status that the orchestrator should handle
                return {"action": "apply_diff", "status": "error", "message": error_msg, "tool_result": apply_result}
            # --- END ADDED CHECK ---

            # Return the success result from the tool directly, adding action info
            return {"action": "apply_diff", **apply_result}
        except Exception as e:
            self._log_callback(f"Error using apply_diff tool for '{filename_to_apply}' with constructed diff: {e}", "error")
            # Log the constructed diff for debugging if the tool fails
            logger.error(f"Constructed diff that failed:\n{constructed_diff}")
            raise RuntimeError(f"Failed to apply constructed diff to '{filename_to_apply}': {e}") from e


    async def _handle_generate_code_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles 'generate_code_diff' and 'generate_full_code' actions using LLM."""
        self._log_callback(f"Action '{action}' requires LLM intervention...", "debug")

        # Extract relevant parameters for the prompt
        change_description = parameters.get("change_description", "Generate code based on the action.")
        strategy = parameters.get("strategy", action) # Use action name if strategy not specified
        target_file = parameters.get("path") # Planner uses 'path'
        code_context_for_prompt = "[Code context not available]" # Default
        if target_file:
             context_key = f"file_content_{target_file}"
             retrieved_content = self.context.get(context_key)
             if retrieved_content:
                 code_context_for_prompt = retrieved_content
                 self._log_callback(f"Retrieved code context for '{target_file}' from context key '{context_key}'.", "debug")
             else:
                 self._log_callback(f"Could not retrieve code context for '{target_file}' from context key '{context_key}'. LLM might lack context.", "warning")
        else:
             target_file = "[unspecified file]" # Set placeholder if no path provided
             self._log_callback("No target file path specified in parameters for code generation.", "warning")

        # Construct the prompt based on the action
        prompt = "" # Initialize prompt
        if action == "generate_code_diff":
            if code_context_for_prompt == "[Code context not available]":
                 self._log_callback("Cannot generate diff info: Code context is missing.", "error")
                 return {"action": action, "status": "error", "message": "Cannot generate diff info without the current code context. Ensure read_file ran first."}

            prompt = f"""
You are an expert Coder Agent assisting in code modification.
Your task is to identify the *specific line(s)* causing the error described and provide the *minimal corrected code snippet* to fix it.

Target File: {target_file}
Change Description / Error to Fix: {change_description}

Current Code Context (from {target_file}):
--- START CODE CONTEXT ---
{code_context_for_prompt}
--- END CODE CONTEXT ---

Instructions:
1. Analyze the 'Current Code Context' and the 'Change Description / Error to Fix'.
2. **If the description specifies a line number and error type:** Identify the *exact* line number(s) in the 'Current Code Context' that match the description.
3. **If the description is general (e.g., "fix indentation errors"):** Analyze the *entire* 'Current Code Context' to find the *first occurrence* of the described error type (e.g., the first `IndentationError`). Identify the line number(s) for this first occurrence.
4. Determine the *minimal* change needed to fix *only* the identified error instance (whether specified or found). Do NOT replace unrelated code blocks.
5. Determine the starting line number (1-based) and ending line number (1-based, inclusive) for the *minimal* block of code you need to replace to apply the fix. Often, this will be a single line.
6. Generate *only* the new code snippet that should replace the identified block. **CRITICAL: Ensure the indentation of the generated snippet precisely matches the indentation level of the code being replaced or the line immediately preceding the insertion point in the original code context. Use standard Python 4-space indentation.**
7. Output your response as a JSON object containing the following keys:
   - "start_line": The starting line number (integer) of the block to replace.
   - "end_line": The ending line number (integer) of the block to replace.
   - "new_code": A string containing ONLY the corrected code snippet(s) to insert. Preserve original newlines within the snippet using \\n.

Example (Fixing Indentation on line 51):
If the error is "IndentationError: unexpected indent" on line 51 and line 51 is "    y1 = screen_height / 2", the output should target line 51 and fix its indentation:
{{
  "start_line": 51,
  "end_line": 51,
  "new_code": "    y1 = screen_height / 2"
}}

Example (Fixing a typo on line 20):
If the error is "NameError: name 'lenght' is not defined" on line 20 and line 20 is "snake_lenght = 1", the output should target line 20:
{{
  "start_line": 20,
  "end_line": 20,
  "new_code": "snake_length = 1"
}}


CRITICAL: Output ONLY the JSON object. Do not include any other text, explanations, or markdown formatting like ```json. Focus strictly on the minimal fix for the described error.

JSON Output:
"""
        elif action == "generate_full_code":
            prompt = f"""
You are an expert Coder Agent. Your task is to generate the complete code for a file based on the description.

Target File: {target_file}
Change Description: {change_description}
Strategy: {strategy}

Existing Code Context (if available, for reference only):
--- START CODE CONTEXT ---
{code_context_for_prompt}
--- END CODE CONTEXT ---

Instructions:
1. Generate the *complete* Python code required to fulfill the 'Change Description' for the 'Target File'.
2. If 'Existing Code Context' is provided, use it for reference but generate the *entire* new file content.
3. **CRITICAL: Ensure all generated Python code uses standard 4-space indentation consistently. Pay close attention to indentation within functions, loops, conditional statements, and class definitions.**
4. Output ONLY the raw code. Do not include explanations, introductions, or markdown formatting like ```.

Full Code Output:
"""
        else:
            # Fallback for potentially other generation actions
            self._log_callback(f"Warning: Unhandled generation action '{action}'. Using generic prompt.", "warning")
            prompt = f"""
            Action: {action}
            Target File: {target_file}
            Change Description: {change_description}
            Strategy: {strategy}
            Code Context:
            {code_context_for_prompt}

            Generate the required output based on the action.
            Output ONLY the raw result or code.
            """

        if not prompt: # Should not happen if action is valid, but safety check
             return {"action": action, "status": "error", "message": f"Failed to construct prompt for action '{action}'."}

        try:
            self._log_callback(f"Calling LLM for action '{action}'...", "debug")
            response = self._call_llm(
                prompt,
                temperature=self.config.get("temperature", 0.4),
                max_output_tokens=self.config.get("max_tokens", 8000),
            ).strip()
            self._log_callback(f"LLM Raw Response received (length {len(response)}).", "debug")

            if "Error: Task is unclear" in response:
                 raise ValueError("Task is unclear according to LLM.")

            result_artifact = {} # Initialize artifact

            # Process response based on the action
            if action == "generate_code_diff":
                import json
                try:
                    # Clean potential markdown fences around JSON
                    cleaned_response = response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[len("```json"):].strip()
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3].strip()

                    diff_info = json.loads(cleaned_response)
                    if not all(k in diff_info for k in ["start_line", "end_line", "new_code"]):
                        raise ValueError("Missing required keys (start_line, end_line, new_code) in JSON response.")
                    if not isinstance(diff_info["start_line"], int) or not isinstance(diff_info["end_line"], int) or not isinstance(diff_info["new_code"], str):
                         raise ValueError("Incorrect data types in JSON response (start_line: int, end_line: int, new_code: str).")

                    self._log_callback(f"Successfully parsed diff info: Start={diff_info['start_line']}, End={diff_info['end_line']}", "debug")
                    result_artifact = {
                        "action": action,
                        "parameters": parameters,
                        "diff_info": diff_info, # Store parsed JSON
                        "status": "generated_diff_info", # New status indicating structured data
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    self._log_callback(f"Error parsing LLM JSON response for diff info: {e}. Response: {response}", "error")
                    return {"action": action, "status": "error", "message": f"Failed to parse valid JSON diff info from LLM: {e}"}

            elif action == "generate_full_code":
                 # Existing logic for full code generation
                 self._log_callback(f"Response type detected as: Code", "info")
                 result_artifact = {
                     "action": action,
                     "parameters": parameters,
                     "result_or_code": response, # Store raw code
                     "language": self.languages[0], # Assume primary language
                     "status": "generated",
                 }
            else:
                 # Fallback for other actions (e.g., direct answers)
                 self._log_callback(f"Response type detected as: Direct Answer/Calculation", "info")
                 result_artifact = {
                     "action": action,
                     "parameters": parameters,
                     "result_or_code": response,
                     "status": "calculated", # Or answered
                 }

            # Store artifact in context under a predictable key
            context_key = "latest_generated_code" # Use a consistent key
            self._log_callback(f"Storing LLM generation result in context key '{context_key}'", "debug")
            self.context.set(context_key, result_artifact)

            return result_artifact

        except Exception as e:
            # Log and re-raise for the main execute_task handler
            self._log_callback(f"Error during LLM call for action '{action}': {e}", "error")
            # Check for specific unclear errors
            if "unclear" in str(e).lower() or "ambiguous" in str(e).lower():
                 self._log_callback("Task deemed unclear by LLM or during processing.", "warning")
                 # Optionally try to request clarification here if mechanism exists
            raise # Re-raise the original exception to be caught by execute_task

    # Removed _get_file_list_for_context - Orchestrator/Planner should provide context
    # Removed _get_code_context_for_prompt - Context should be passed explicitly if needed
    # Removed _get_plan_steps_str - Plan context handled differently

    async def _handle_analyze_content_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
         """Handles the 'analyze_content' action using LLM."""
         analysis_query = parameters.get("analysis_query")
         target_file = parameters.get("path") # Planner uses 'path'
         content_to_analyze = parameters.get("content") # Expect content directly

         if not analysis_query or not isinstance(analysis_query, str):
             return {"action": "analyze_content", "status": "error", "message": "Missing or invalid 'analysis_query' parameter."}

         # If content not provided directly, try reading from context (assuming read_file ran)
         if content_to_analyze is None and target_file:
             context_key = f"file_content_{target_file}"
             content_to_analyze = self.context.get(context_key)
             if content_to_analyze is None:
                  # Try reading the file if not in context
                  self._log_callback(f"Content for '{target_file}' not in context, attempting read...", "warning")
                  read_result = await self._handle_file_read_task({"path": target_file})
                  if read_result.get("status") == "success":
                      content_to_analyze = read_result.get("content")
                  else:
                      return {"action": "analyze_content", "status": "error", "message": f"Content for '{target_file}' not provided directly and failed to read file: {read_result.get('message')}"}
             self._log_callback(f"Using content from context key '{context_key}' or file read for analysis.", "debug")
         elif content_to_analyze is None:
              return {"action": "analyze_content", "status": "error", "message": "Missing 'content' parameter and no 'path' provided to read from context."}

         if not isinstance(content_to_analyze, str):
              return {"action": "analyze_content", "status": "error", "message": "'content' parameter must be a string."}


         self._log_callback(f"Performing analysis: '{analysis_query}'", "info")
         # Simple prompt for analysis
         prompt = f"""
         Analyze the following content based on the query.
         Content:
         ---
         {content_to_analyze[:4000]} # Limit context size
         ---
         Query: {analysis_query}

         Provide a concise answer to the query based *only* on the provided content.
         Answer:
         """
         try:
             analysis_result = self._call_llm(prompt, temperature=0.2).strip()
             self._log_callback(f"LLM Analysis Result: {analysis_result}", "debug")
             return {
                 "action": "analyze_content",
                 "status": "success",
                 "analysis_result": analysis_result,
                 "query": analysis_query
             }
         except Exception as e:
             self._log_callback(f"Error during LLM call for analysis: {e}", "error")
             return {"action": "analyze_content", "status": "error", "message": f"LLM analysis failed: {e}"}

    async def _handle_file_exists_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'file_exists' action."""
        path_to_check = parameters.get("path") # Planner uses 'path'
        if not path_to_check or not isinstance(path_to_check, str):
             return {"action": "file_exists", "status": "error", "message": "Missing or invalid 'path' parameter."}
        try:
            # Corrected parameter name from 'path' to 'filename' for use_tool
            result = self.use_tool("file_exists", filename=path_to_check) # CORRECTED
            return {"action": "file_exists", **result}
        except Exception as e:
            self._log_callback(f"Error using file_exists tool for '{path_to_check}': {e}", "error")
            return {"action": "file_exists", "status": "error", "message": f"Tool execution failed: {e}"}

    async def _handle_list_files_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'list_files' action."""
        path_to_list = parameters.get("path", ".") # file_list tool uses 'path', this is correct
        recursive = parameters.get("recursive", False)
        if not isinstance(path_to_list, str) or not isinstance(recursive, bool):
             return {"action": "list_files", "status": "error", "message": "Invalid 'path' (string) or 'recursive' (bool) parameter."}
        try:
            # file_list tool uses 'path', which is correct
            result = self.use_tool("file_list", path=path_to_list, recursive=recursive)
            return {"action": "list_files", **result}
        except Exception as e:
            self._log_callback(f"Error using file_list tool for '{path_to_list}': {e}", "error")
            return {"action": "list_files", "status": "error", "message": f"Tool execution failed: {e}"}

    async def _handle_verify_output_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
         """Handles the 'verify_output' action (placeholder)."""
         # TODO: Implement verification logic, potentially using LLM or string matching
         expected_output = parameters.get("expected_output")
         actual_output = parameters.get("actual_output") # Needs to be retrieved from previous step context
         self._log_callback(f"Verification requested: Expect '{expected_output}'. Actual: '{actual_output}'", "info")
         # Placeholder implementation
         verified = False
         if expected_output and actual_output and expected_output in actual_output:
             verified = True
         return {
             "action": "verify_output",
             "status": "success", # Or "error" if verification fails
             "verified": verified,
             "message": f"Verification {'succeeded' if verified else 'failed'}."
         }

    # Removed _analyze_directory_for_context

    # Removed _calculate_arithmetic method (LLM can handle)
    # Removed _validate_code method (LLM can handle)

    def refactor_code(self, code: str, feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Refactors existing code based on feedback or optimization goals using LLM guidance.

        Args:
            code: The existing code to refactor.
            feedback: Optional feedback about improvements needed.

        Returns:
            Dictionary containing the refactored code and metadata.
        """
        self._log_callback("Refactoring code...", "info")

        refactor_goal = feedback or "Improve code quality, clarity, and maintainability."

        prompt = self.CODE_REFACTORING_PROMPT.format(
            original_code=code,
            feedback=refactor_goal,
            languages=", ".join(self.languages)
        )
        self._log_callback("Using CODE_REFACTORING_PROMPT...", "debug")


        refactored_code = self._call_llm(
            prompt,
            temperature=0.3, # Keep low temp for refactoring
            max_output_tokens=self.config.get("max_tokens", 2500) # Renamed from max_tokens
        ).strip()

        self._log_callback(f"Refactored code snippet: {refactored_code[:100]}{'...' if len(refactored_code) > 100 else ''}", "debug")


        artifact = {
            "refactored_code": refactored_code,
            "original_code": code,
            "changes": refactor_goal,
            "status": "refactored",
            "language": self.languages[0] # Assume primary language
        }

        self.context.set("refactored_code", artifact)
        return artifact

    # Added internal helper method
    def _get_full_path(self, filename: str) -> str:
        """Get the full, validated path for a file relative to the base directory."""
        if os.path.isabs(filename):
             raise ValueError(f"Absolute paths are not allowed: {filename}")
        resolved_base_dir = os.path.abspath(self.base_dir)
        combined_path = os.path.join(resolved_base_dir, filename)
        resolved_combined_path = os.path.abspath(combined_path)
        if os.path.commonpath([resolved_base_dir, resolved_combined_path]) != resolved_base_dir:
            raise ValueError(f"Path traversal attempt detected: '{filename}' resolves outside the base directory '{resolved_base_dir}'")
        return resolved_combined_path

    # Removed _try_enhance_with_tools method
