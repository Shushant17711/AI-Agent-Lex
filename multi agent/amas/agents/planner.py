"""Planner agent implementation."""

import json # Added for JSON parsing
from typing import Any, Dict, Optional, List # Added List
import os
from pathlib import Path
from ..core.llm_service_gemini import BaseLLMService
from ..core.context_manager import ContextManager
from .base import BaseAgent, register_agent, CommunicationBus
import re

@register_agent("planner")
class PlannerAgent(BaseAgent):
    """Agent responsible for creating high-level plans to accomplish tasks."""

    # --- Prompt Loading ---
    def _load_prompt(self, file_path: Path) -> str:
        """Loads a prompt template from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self._log_callback(f"Prompt file not found: {file_path}", "error")
            # Return a default/error message or raise? Returning error message for now.
            return f"ERROR: Prompt file '{file_path.name}' not found."
        except Exception as e:
            self._log_callback(f"Error loading prompt file {file_path}: {e}", "error")
            return f"ERROR: Could not load prompt file '{file_path.name}'."
    # --- End Prompt Loading ---
    def __init__(self, name: str, config: Dict[str, Any], llm_service: BaseLLMService, context: ContextManager, base_dir: str, log_callback: Optional[callable] = None, communication_bus: Optional[CommunicationBus] = None): # Added base_dir
        super().__init__(name, config, llm_service, context, base_dir, log_callback, communication_bus=communication_bus) # Pass base_dir to super
        self.planning_style = config.get("planning_style", "step-by-step")
        self.use_tools = config.get("use_tools", True)

        # Determine the base directory for prompts (relative to this file)
        # Assuming this file is amas/agents/planner.py and prompts are in amas/prompts/planner/
        prompt_dir = Path(__file__).parent.parent / "prompts" / "planner"

        # Load prompts
        self.plan_creation_template = self._load_prompt(prompt_dir / "plan_creation.prompt")
        self.plan_refinement_template = self._load_prompt(prompt_dir / "plan_refinement.prompt")

        self._log_callback(f"PlannerAgent initialized with style: {self.planning_style}, tools enabled: {self.use_tools}", "info")
        if "ERROR:" in self.plan_creation_template:
             self._log_callback("Failed to load plan creation prompt.", "warning")
        if "ERROR:" in self.plan_refinement_template:
             self._log_callback("Failed to load plan refinement prompt.", "warning")

    async def execute_task(self, task_input: Any) -> Dict[str, Any]:
        """
        Handles planner-specific actions or generates a new plan.

        Args:
            task_input: Can be a string (initial task description) or a
                        dictionary containing 'action' and 'parameters' for
                        planner-specific actions within a plan.

        Returns:
            A dictionary containing the result, status, and potentially plan steps.
        """
        # Check if input is a structured action request or an initial task string
        if isinstance(task_input, dict):
            action = task_input.get("action")
            parameters = task_input.get("parameters", {})
            self._log_callback(f"Handling action '{action}' with parameters: {parameters}", "info")

            if action == "decide_edit_strategy":
                return await self._handle_decide_edit_strategy(parameters)
            elif action == "review_plan":
                # Placeholder for future implementation
                self._log_callback(f"Action '{action}' not fully implemented yet.", "warning")
                return {"status": "success", "message": f"Action '{action}' acknowledged but not implemented."}
            else:
                self._log_callback(f"Unknown action '{action}' for PlannerAgent.", "error")
                return {"status": "error", "message": f"PlannerAgent received unknown action: {action}"}

        # --- Original Plan Generation Logic (if task_input is a string) ---
        elif isinstance(task_input, str):
            return await self._generate_initial_plan(task_input)
        else:
             self._log_callback(f"Invalid task_input type for PlannerAgent: {type(task_input)}", "error")
             return {"status": "error", "message": f"Invalid input type: {type(task_input)}"}


    async def _generate_initial_plan(self, task_description: str) -> Dict[str, Any]:
        """Generates the initial plan based on a task description string."""
        self._log_callback(f"Generating initial plan for task: {task_description[:50]}...", "info")

        # Get available agents from context or default
        available_agents = list(self.context.get("agents", {"planner": self, "coder": None}).keys()) # Assume coder exists

        # Check if prompt loading failed
        if "ERROR:" in self.plan_creation_template:
            self._log_callback("Cannot execute task: Plan creation prompt failed to load.", "error")
            return {
                "original_task": task_description, "steps": [], "status": "error",
                "message": "Configuration error: Plan creation prompt not found.",
                "style": self.planning_style
            }

        # Format the prompt using the loaded template
        prompt = self.plan_creation_template.format(
            task=task_description,
            style=self.planning_style,
            agents=available_agents
        )
        self._log_callback(f"Using plan_creation.prompt for task: {task_description[:30]}...", "debug")

        plan_json_str = "" # Initialize
        steps = [] # Initialize steps list
        try:
            plan_json_str = self._call_llm(
                prompt,
                temperature=self.config.get("temperature", 0.7)
            )
            # Attempt to parse the LLM response as JSON
            try:
                # Clean potential markdown fences
                if plan_json_str.startswith("```json"):
                    plan_json_str = plan_json_str[7:]
                if plan_json_str.endswith("```"):
                    plan_json_str = plan_json_str[:-3]
                plan_json_str = plan_json_str.strip()

                parsed_plan = json.loads(plan_json_str)
                if self._validate_plan_json(parsed_plan, available_agents):
                    steps = parsed_plan # Use the validated list of step dicts
                else:
                    # Validation failed, logged within _validate_plan_json
                    raise ValueError("Plan JSON structure validation failed.")

            except json.JSONDecodeError as json_e:
                self._log_callback(f"Failed to parse plan JSON from LLM: {json_e}", "error")
                self._log_callback(f"LLM Raw Output:\n{plan_json_str}", "debug")
                raise ValueError(f"LLM output was not valid JSON: {json_e}") from json_e
            except ValueError as val_e: # Catch validation errors
                 self._log_callback(f"Plan validation failed: {val_e}", "error")
                 raise # Re-raise to be caught by the outer exception handler

        except Exception as e:
            self._log_callback(f"Error during plan creation or LLM call: {e}", "error")
            return {
                "original_task": task_description,
                "steps": [], # Return empty steps on error
                "status": "error",
                "message": f"Failed to generate or validate plan: {e}",
                "style": self.planning_style
            }

        # If steps list is empty after potential errors, return error status
        if not steps:
             self._log_callback(f"Plan generation resulted in zero valid steps for task: {task_description}", "error")
             return {
                 "original_task": task_description,
                 "steps": [],
                 "status": "error",
                 "message": "Failed to generate any valid plan steps.",
                 "style": self.planning_style
             }


        plan = {
            "original_task": task_description,
            "steps": steps,
            "status": "draft",
            "style": self.planning_style
        }

        self.context.set("current_plan", plan)
        first_step_info = f"Agent: {steps[0].get('agent', 'N/A')}, Action: {steps[0].get('action', 'N/A')}" if steps else 'N/A'
        self._log_callback(f"Plan created with {len(plan['steps'])} steps. First step: {first_step_info}", "info")
        return plan


    def refine_plan(self, feedback: str, current_plan: Optional[List[Dict[str, Any]]] = None, failed_step_index: Optional[int] = None, previous_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Refines the current plan (list of step dictionaries) based on feedback.

        Args:
            feedback: Feedback to incorporate into the plan.
            current_plan: The list of step dictionaries in the current plan.
            failed_step_index: The 0-based index of the step that failed.
            previous_results: List of dictionaries summarizing previous step outcomes.

        Returns:
            The updated plan.
        """
        # Use passed plan if available, otherwise try context (though orchestrator should pass it)
        # Ensure current_plan is a list of dicts
        plan_to_refine_list = current_plan
        if not isinstance(plan_to_refine_list, list):
             plan_from_context = self.context.get("current_plan")
             if isinstance(plan_from_context, dict):
                 plan_to_refine_list = plan_from_context.get("steps")
             if not isinstance(plan_to_refine_list, list): # Check again after getting from context
                 plan_to_refine_list = None # Set to None if still not a list

        if not plan_to_refine_list:
            self._log_callback("No current plan steps provided or found in context to refine.", "error")
            # Return an empty plan or raise error? Returning empty for now.
            return {"steps": [], "status": "error", "message": "No plan to refine."}

        self._log_callback(f"Refining plan based on feedback: {feedback[:50]}...", "info")

        # Check for help requests from other agents (optional, keep if needed)
        help_request = self.context.get("help_request")
        if help_request and help_request.get("to") == "planner":
            feedback += f"\nHelp request from {help_request['from']}: {help_request['request']}"
            self.context.set(f"help_response_planner", "Request received", notify=True) # Acknowledge

        available_agents = list(self.context.get("agents", {"planner": self, "coder": None}).keys())
        # Convert the current plan (list of dicts) to a JSON string for the prompt
        try:
            current_plan_steps_str = json.dumps(plan_to_refine_list, indent=2)
        except TypeError as e:
            self._log_callback(f"Error serializing current plan to JSON: {e}", "error")
            current_plan_steps_str = "[]" # Fallback to empty JSON array string
        failed_step_index_human = (failed_step_index + 1) if failed_step_index is not None else "N/A"

        # Summarize previous results for context
        previous_results_summary = "No previous results available."
        if previous_results:
            previous_results_summary = "\n".join([f"- Step {idx+1} ({res.get('agent', 'N/A')}): {res.get('status', 'N/A')} - {res.get('message', '')[:100]}" for idx, res in enumerate(previous_results)])

        # Check if prompt loading failed
        if "ERROR:" in self.plan_refinement_template:
            self._log_callback("Cannot refine plan: Plan refinement prompt failed to load.", "error")
            # Return original steps as fallback
            return {
                "original_task": self.context.get("current_plan", {}).get("original_task", "Unknown"),
                "steps": plan_to_refine_list, # Return original steps
                "status": "error",
                "message": "Configuration error: Plan refinement prompt not found.",
                "version": self.context.get("current_plan", {}).get("version", 0) + 1
            }

        # Format the refinement prompt
        prompt = self.plan_refinement_template.format(
            original_task=self.context.get("current_plan", {}).get("original_task", "Unknown"), # Get original task from context if possible
            current_plan_steps=current_plan_steps_str,
            feedback=feedback,
            agents=available_agents,
            failed_step_index_human=failed_step_index_human,
            previous_results_summary=previous_results_summary
        )
        self._log_callback("Using plan_refinement.prompt...", "debug")


        refined_plan_json_str = "" # Initialize
        steps = [] # Initialize steps list
        try:
            refined_plan_json_str = self._call_llm(
                prompt,
                temperature=self.config.get("temperature", 0.5)
            )
            # Attempt to parse the LLM response as JSON
            try:
                 # Clean potential markdown fences
                if refined_plan_json_str.startswith("```json"):
                    refined_plan_json_str = refined_plan_json_str[7:]
                if refined_plan_json_str.endswith("```"):
                    refined_plan_json_str = refined_plan_json_str[:-3]
                refined_plan_json_str = refined_plan_json_str.strip()

                parsed_plan = json.loads(refined_plan_json_str)
                if self._validate_plan_json(parsed_plan, available_agents):
                    steps = parsed_plan # Use the validated list of step dicts
                else:
                    # Validation failed, logged within _validate_plan_json
                    raise ValueError("Refined plan JSON structure validation failed.")

            except json.JSONDecodeError as json_e:
                self._log_callback(f"Failed to parse refined plan JSON from LLM: {json_e}", "error")
                self._log_callback(f"LLM Raw Output:\n{refined_plan_json_str}", "debug")
                raise ValueError(f"LLM output for refined plan was not valid JSON: {json_e}") from json_e
            except ValueError as val_e: # Catch validation errors
                 self._log_callback(f"Refined plan validation failed: {val_e}", "error")
                 raise # Re-raise to be caught by the outer exception handler

        except Exception as e:
            self._log_callback(f"Error during plan refinement or LLM call: {e}", "error")
            # Fallback: Keep the original plan steps if refinement fails
            steps = plan_to_refine_list # Use original list of dicts
            self._log_callback("Falling back to original plan steps due to refinement error.", "warning")

        # If steps list is empty after potential errors or fallback, return error status
        if not steps:
             self._log_callback("Plan refinement resulted in zero valid steps.", "error")
             # Return original steps if available, otherwise empty
             steps = plan_to_refine_list if plan_to_refine_list else []
             return {
                 "original_task": self.context.get("current_plan", {}).get("original_task", "Unknown"),
                 "steps": steps,
                 "status": "error",
                 "message": "Failed to generate any valid refined plan steps.",
                 "version": self.context.get("current_plan", {}).get("version", 0) + 1
             }


        refined_plan = {
            # Create a new dictionary, don't modify the context one directly if passed
            "original_task": self.context.get("current_plan", {}).get("original_task", "Unknown"),
            "steps": steps,
            "status": "refined",
            "feedback": feedback,
            "version": self.context.get("current_plan", {}).get("version", 0) + 1
        }

        self.context.set("current_plan", refined_plan, notify=True)
        first_step_info = f"Agent: {steps[0].get('agent', 'N/A')}, Action: {steps[0].get('action', 'N/A')}" if steps else 'N/A'
        self._log_callback(f"Plan refined with {len(refined_plan['steps'])} steps. First step: {first_step_info}", "info")
        return refined_plan

    def _validate_plan_json(self, plan: Any, available_agents: List[str]) -> bool:
        """
        Validates the structure of the parsed plan JSON and enforces critical
        sequences like read -> analyze -> modify.
        """
        if not isinstance(plan, list):
            self._log_callback("Plan validation failed: Top level is not a list.", "error")
            return False

        if not plan:
            self._log_callback("Plan validation warning: Plan is an empty list.", "warning")
            return True # Allow empty plans

        required_keys = {"step_id", "agent", "action", "parameters"}
        valid = True
        # Track file states: path -> {"read": step_id, "analyzed": step_id}
        file_states: Dict[str, Dict[str, Optional[int]]] = {}
        # Define actions requiring specific prior steps
        diff_actions = {"apply_diff"}
        analysis_actions = {"analyze_content"}
        # write_file does not strictly require a prior read if creating a new file

        for i, step in enumerate(plan):
            step_id = step.get('step_id', i + 1) # Use index+1 if step_id is missing for logging

            if not isinstance(step, dict):
                self._log_callback(f"Plan validation failed: Step {step_id} is not a dictionary.", "error")
                valid = False
                continue # Skip further checks for this step

            missing_keys = required_keys - set(step.keys())
            if missing_keys:
                self._log_callback(f"Plan validation failed: Step {step_id} missing keys: {missing_keys}", "error")
                valid = False
                # Continue checking other aspects if possible

            # --- Basic Type Checks ---
            if "step_id" in step and not isinstance(step["step_id"], int):
                 self._log_callback(f"Plan validation failed: Step {i+1} 'step_id' is not an integer.", "error")
                 valid = False
            if "agent" in step and not isinstance(step["agent"], str):
                 self._log_callback(f"Plan validation failed: Step {step_id} 'agent' is not a string.", "error")
                 valid = False
            elif "agent" in step and step["agent"] not in available_agents:
                 self._log_callback(f"Plan validation warning: Step {step_id} uses unknown agent '{step['agent']}'. Available: {available_agents}", "warning")
                 # Don't mark as invalid, but warn.
            if "action" in step and not isinstance(step["action"], str):
                 self._log_callback(f"Plan validation failed: Step {step_id} 'action' is not a string.", "error")
                 valid = False
            if "parameters" in step and not isinstance(step["parameters"], dict):
                 self._log_callback(f"Plan validation failed: Step {step_id} 'parameters' is not a dictionary.", "error")
                 valid = False
                 # Cannot proceed with sequence checks if parameters are invalid
                 continue

            # --- Sequence Logic Checks ---
            action = step.get("action")
            params = step.get("parameters", {})
            path = params.get("path") if isinstance(params, dict) else None

            if not path and action in ["read_file", "analyze_content", "apply_diff", "write_file"]:
                 self._log_callback(f"Plan validation warning: Step {step_id} action '{action}' is missing 'path' parameter.", "warning")
                 # Don't necessarily invalidate, but it might cause issues later

            if path: # Only track/check sequences for actions with a path
                if path not in file_states:
                    file_states[path] = {"read": None, "analyzed": None}

                current_state = file_states[path]

                if action == "read_file":
                    current_state["read"] = step_id
                    current_state["analyzed"] = None # Reading resets analysis state for that file

                elif action == "analyze_content":
                    if current_state["read"] is None:
                        self._log_callback(f"Plan validation failed: Step {step_id} ('analyze_content' for '{path}') has no preceding 'read_file' step for the same path.", "error")
                        valid = False
                    # Check if analysis happened *after* the last read for this path
                    elif current_state["analyzed"] is not None and current_state["analyzed"] > current_state["read"]:
                         # This case is less likely but possible if plan has analyze -> read -> analyze
                         self._log_callback(f"Plan validation warning: Step {step_id} ('analyze_content' for '{path}') occurred after a previous analysis (step {current_state['analyzed']}) but before a new read.", "warning")
                         current_state["analyzed"] = step_id # Update to latest analysis
                    else:
                        current_state["analyzed"] = step_id

                # Check apply_diff specifically requires prior read and analysis
                elif action in diff_actions:
                    if current_state["read"] is None:
                        self._log_callback(f"Plan validation failed: Step {step_id} ('{action}' for '{path}') has no preceding 'read_file' step for the same path.", "error")
                        valid = False
                    elif current_state["analyzed"] is None:
                        # apply_diff needs analysis after the read
                        self._log_callback(f"Plan validation failed: Step {step_id} ('{action}' for '{path}') has no preceding 'analyze_content' step after the last 'read_file' (step {current_state.get('read', 'N/A')}) for the same path.", "error")
                        valid = False
                    # Ensure analysis happened *after* the last read
                    elif current_state["read"] is not None and current_state["analyzed"] < current_state["read"]:
                         self._log_callback(f"Plan validation failed: Step {step_id} ('{action}' for '{path}') analysis (step {current_state['analyzed']}) happened before the last read (step {current_state['read']}).", "error")
                         valid = False
                # write_file does not require a prior read/analysis, but if it was read, analysis should precede write
                elif action == "write_file":
                     if current_state["read"] is not None: # If the file *was* read (implying modification)
                          if current_state["analyzed"] is None:
                               self._log_callback(f"Plan validation warning: Step {step_id} ('{action}' for '{path}') modifies a previously read file (step {current_state['read']}) but has no preceding 'analyze_content' step.", "warning")
                               # Don't mark as invalid, but warn about potential overwrite without analysis
                          elif current_state["analyzed"] < current_state["read"]:
                               self._log_callback(f"Plan validation warning: Step {step_id} ('{action}' for '{path}') analysis (step {current_state['analyzed']}) happened before the last read (step {current_state['read']}). Overwriting based on potentially stale analysis.", "warning")
                               # Don't mark as invalid, but warn


        return valid

    async def _handle_decide_edit_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decides whether to use 'apply_diff' or 'write_file' based on task goal and analysis.
        Currently uses a simple heuristic, could be enhanced with LLM.
        """
        analysis_result = parameters.get("analysis_result", "Analysis result missing.")
        task_goal = parameters.get("task_goal", "Task goal missing.")
        self._log_callback(f"Deciding edit strategy. Goal: '{task_goal}'. Analysis: '{analysis_result[:100]}...'", "info")

        # Simple heuristic: If analysis indicates major issues or goal implies large changes, use write_file.
        # Otherwise, default to apply_diff for smaller modifications.
        # TODO: Enhance this logic, potentially with an LLM call for more nuanced decisions.
        strategy = "apply_diff" # Default strategy
        analysis_lower = analysis_result.lower()
        task_goal_lower = task_goal.lower()

        # Conditions to force write_file
        force_write_conditions = [
            "major rewrite" in task_goal_lower,
            "start over" in task_goal_lower,
            "create new" in task_goal_lower,
            "not found" in analysis_lower,
            "does not contain" in analysis_lower,
            "empty file" in analysis_lower,
            "is a diff" in analysis_lower, # Added: If analysis says content is a diff
            "placeholder" in analysis_lower, # Added: If analysis mentions placeholder text
            "corrupted" in analysis_lower, # Added: If analysis mentions corruption
            "not the expected" in analysis_lower, # Added: General mismatch
            "fundamentally different" in analysis_lower # Added: General mismatch
        ]

        if any(force_write_conditions):
            strategy = "write_file"
            self._log_callback("Analysis/Goal suggests major changes, non-existent/corrupted content, or fundamental mismatch. Choosing 'write_file' strategy.", "info")
        else:
            self._log_callback("Analysis/Goal suggests minor changes or compatible content. Choosing 'apply_diff' strategy.", "info")


        return {
            "status": "success",
            "strategy": strategy, # The decided strategy
            "message": f"Edit strategy decided: {strategy}"
        }

    # _on_help_request can remain if inter-agent help is desired
    # def _on_help_request(self, key: str, request: Dict[str, Any]): ...