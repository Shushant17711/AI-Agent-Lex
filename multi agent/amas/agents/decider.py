# amas/agents/decider.py
import logging
import time  # Added for performance metrics
from typing import Dict, Any, Optional
# Assuming potential specific exceptions from the LLM service
# from google.api_core.exceptions import ApiException, ServiceUnavailable
# from requests.exceptions import RequestException # If using requests

from amas.agents.base import BaseAgent, register_agent
from amas.core.llm_service_gemini import BaseLLMService # Assuming _call_llm might raise specific errors from here
from amas.core.context_manager import ContextManager
from amas.core.communication_bus import CommunicationBus

logger = logging.getLogger(__name__)

@register_agent("decider")
class DeciderAgent(BaseAgent):
    """
    An agent responsible for analyzing the planner's output and providing
    clear, actionable instructions to the coder.
    """

    def __init__(self, name: str, config: Dict[str, Any], llm_service: BaseLLMService, context: ContextManager, log_callback: Optional[callable] = None, communication_bus: Optional[CommunicationBus] = None):
        """Initializes the DeciderAgent."""
        super().__init__(name, config, llm_service, context, log_callback, communication_bus)
        self.log_callback(f"DeciderAgent '{self.name}' initialized.", "info")
        self.max_log_length = self.config.get('max_log_length', 500) # Configurable log truncation length
        self.default_prompt_template = """
{role}

Your task is to analyze the following instruction received from the planner agent and refine it into a clear, specific, and actionable task for the coder agent. The coder agent needs precise instructions on what code to write, modify, or analyze.

Planner's Instruction:
---
{task_input}
---

Based on this instruction, decide the best course of action (e.g., full rewrite, targeted changes, specific function implementation, analysis needed) and formulate a precise task for the coder. Focus on clarity and actionability. If the instruction is already clear enough, you can state that and pass it along. If the instruction is fundamentally unclear or impossible, state that clearly.

Refined Instruction for Coder:
"""

    def execute_task(self, task_input: Any) -> Any:
        """
        Analyzes the planner's output (task_input) and refines it for the coder.

        Args:
            task_input: The plan or instructions received from the planner.

        Returns:
            Refined instructions for the coder.
        """
        # --- Input Validation ---
        if not task_input:
            self.log_callback("Received empty task input. Cannot proceed.", "warning")
            return "Error: Task input cannot be empty." # Return an error or handle appropriately

        # Truncate for logging if necessary
        log_input = str(task_input)
        if len(log_input) > self.max_log_length:
            log_input = log_input[:self.max_log_length] + "..."
        self.log_callback(f"Received task input for decision: {log_input}", "info")

        # --- LLM Logic ---
        role = self.config.get('role_description', 'You are a helpful assistant.')
        prompt_template = self.config.get('decider_prompt_template', self.default_prompt_template)

        try:
            prompt = prompt_template.format(role=role, task_input=task_input)
        except KeyError as e:
            self.log_callback(f"Error formatting prompt template. Missing key: {e}", "error")
            # Fallback or error handling if template is invalid
            return f"Error: Invalid prompt template configuration. Missing key: {e}"


        try:
            self.log_callback("Calling LLM to refine coder instruction...", "debug")
            start_time = time.time()
            refined_instructions = self._call_llm(prompt)
            duration = time.time() - start_time
            self.log_callback(f"LLM call completed in {duration:.2f} seconds.", "info")

            # Basic check if LLM returned an error message itself (adjust if needed)
            if isinstance(refined_instructions, str) and refined_instructions.startswith("Error during LLM call:"):
                 self.log_callback(f"LLM service reported an internal failure: {refined_instructions}", "error")
                 # Fallback: return original input if LLM fails internally
                 return task_input # Or raise a specific exception

        # --- Specific Exception Handling ---
        # Replace with actual exceptions raised by your LLM service
        # except ApiException as e:
        #     self.log_callback(f"LLM API error in DeciderAgent: {e}", "error")
        #     refined_instructions = f"Error: LLM API interaction failed - {e}" # More specific error
        # except ServiceUnavailable as e:
        #     self.log_callback(f"LLM service unavailable: {e}", "error")
        #     refined_instructions = f"Error: LLM service unavailable - {e}"
        # except RequestException as e: # Example if using requests library
        #     self.log_callback(f"Network error during LLM call: {e}", "error")
        #     refined_instructions = f"Error: Network issue contacting LLM - {e}"
        except Exception as e:
            # Catch-all for unexpected errors during the LLM call itself
            duration = time.time() - start_time # Measure duration even on failure
            self.log_callback(f"Unexpected error calling LLM in DeciderAgent after {duration:.2f}s: {e}", "error", exc_info=True) # Log traceback
            # Fallback: return original input or a specific error message
            refined_instructions = f"Error: Failed to refine instructions due to an unexpected error: {e}"
        # --- End LLM Logic ---

        # Truncate for logging if necessary
        log_output = str(refined_instructions)
        if len(log_output) > self.max_log_length:
            log_output = log_output[:self.max_log_length] + "..."
        self.log_callback(f"Refined instructions for coder: {log_output}", "info")
        return refined_instructions