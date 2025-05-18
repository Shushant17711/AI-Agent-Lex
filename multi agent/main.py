#!/usr/bin/env python3
"""Main entry point for AMAS (Autonomous Multi-Agent System)."""

import argparse
import logging # Added for logging setup
import os # Added for environment variable check
from pathlib import Path
from typing import Optional
import asyncio # Import asyncio
from dotenv import load_dotenv # Import dotenv

from amas.core.enhanced_orchestrator import EnhancedOrchestrator
from amas.core.context_manager import ContextManager

# Configure base logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get main logger

# --- Core Agent Logic Function ---
async def run_agent_task(
    task: str,
    config_path: Path,
    context_path: Optional[Path],
    log_callback: callable,
    working_directory: Optional[Path] = None, # Add working directory parameter
    file_focus_callback: Optional[callable] = None, # Add file focus callback
    file_content_callback: Optional[callable] = None # Add file content callback
):
    """Runs the multi-agent system for a given task."""
    log_callback("Loading environment variables...")
    load_dotenv() # Load environment variables

    # Check for Gemini API Key
    if not os.getenv("GOOGLE_API_KEY"):
         log_callback("WARNING: GOOGLE_API_KEY environment variable not set. Gemini LLM may fail.", "warning")

    log_callback(f"Initializing context from: {context_path or 'new context'}")
    context = ContextManager.load(context_path) if context_path else ContextManager()

    try:
        log_callback(f"Loading configuration from: {config_path}")
        orchestrator = EnhancedOrchestrator.from_config(
            config_path,
            context,
            log_callback=log_callback,
            working_directory=working_directory, # Pass working directory
            file_focus_callback=file_focus_callback, # Pass file focus callback
            file_content_callback=file_content_callback # Pass file content callback
        )

        # --- Prepare Task with Directory Context ---
        effective_wd = working_directory if working_directory else Path.cwd()
        log_callback(f"Gathering directory context for: {effective_wd}", "debug")
        try:
            # List files recursively, ignoring hidden files/dirs and __pycache__
            all_files = []
            for item in effective_wd.rglob('*'):
                if item.is_file():
                     # Basic filtering for common noise
                     parts = item.relative_to(effective_wd).parts
                     if not any(p.startswith('.') for p in parts) and '__pycache__' not in parts:
                          all_files.append(str(item.relative_to(effective_wd)))

            if all_files:
                context_header = "Directory Context (Working Directory Files):\n"
                file_list_str = "\n".join(f"- {f}" for f in sorted(all_files))
                enhanced_task = f"{context_header}{file_list_str}\n\nUser Task:\n{task}"
                log_callback(f"Prepended directory context ({len(all_files)} files) to task.", "info")
            else:
                log_callback("No files found in working directory for context.", "warning")
                enhanced_task = task # Use original task if no files found
        except Exception as list_e:
            log_callback(f"Error listing files in {effective_wd}: {list_e}", "warning")
            enhanced_task = task # Use original task on error

        # --- Execute the task with retry logic ---
        max_attempts = 7
        result = None
        for attempt in range(1, max_attempts + 1):
            try:
                log_callback(f"--- Execution Attempt {attempt}/{max_attempts} ---")
                # Orchestrator.execute is now async - use enhanced_task
                result = await orchestrator.execute(enhanced_task)
                log_callback(f"Attempt {attempt} successful.", "success")
                break # Exit loop on success
            except Exception as e:
                log_callback(f"Attempt {attempt} failed: {e}", "error")
                if attempt == max_attempts:
                    log_callback("Max attempts reached. Task failed.", "error")
                    result = f"Task failed after {max_attempts} attempts. Last error: {e}"
                else:
                    log_callback("Retrying...") # Inform user about retry

        # Save final context
        final_context_path = Path("context.json")
        log_callback(f"Saving final context to: {final_context_path}")
        context.save(final_context_path)
        log_callback("Context saved.")
        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Log the detailed error FIRST using the main logger
        logger.error(f"System initialization or critical error caught in run_agent_task: {e}\nTraceback:\n{error_details}")
        # THEN try to inform the GUI via the callback
        try:
            log_callback(f"System initialization or critical error: {e}", "error")
        except Exception as cb_e:
             logger.error(f"Error occurred WITHIN log_callback itself: {cb_e}", exc_info=True)
        # Save context AFTER logging the error (Consider if saving context on error is desired)
        final_context_path = Path("context.json")
        logger.info(f"Saving context after error to: {final_context_path}") # Log context save attempt
        context.save(final_context_path)
        return f"System error: {e}" # Return the error message

# --- CLI Entry Point ---
def main_cli():
    """Main CLI entry point."""
    load_dotenv() # Load environment variables from .env file at the start
    """Handles command-line arguments and initiates the agent task."""
    parser = argparse.ArgumentParser(description="Autonomous Multi-Agent System")
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="Path to existing context file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbosity
    # Get the root logger for the 'amas' package
    amas_logger = logging.getLogger('amas')
    if args.verbose:
        amas_logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled for 'amas' package.")
    else:
        amas_logger.setLevel(logging.INFO)
        
    # Check for Gemini API Key if Gemini is likely the LLM (best effort check)
    # --- CLI Specific Logging Callback ---
    def cli_log_callback(message: str, level: str = "info"):
        """Prints messages to the console based on level."""
        if level == "error":
            logger.error(message)
            print(f"❌ ERROR: {message}")
        elif level == "warning":
            logger.warning(message)
            print(f"⚠️ WARNING: {message}")
        elif level == "success":
            logger.info(message)
            print(f"✅ SUCCESS: {message}")
        else: # info or other
            logger.info(message)
            print(f"ℹ️ INFO: {message}")

    # Get task from user input
    task = input("Enter task for AI to work on: ").strip()
    while not task:
        print("Task cannot be empty.")
        task = input("Enter task for AI to work on: ").strip()

    # Run the core logic using the CLI logger
    # Pass None for working_directory when running from CLI
    # Run the async function using asyncio.run()
    result = asyncio.run(run_agent_task(task, args.config, args.context, cli_log_callback, working_directory=None, file_focus_callback=None, file_content_callback=None))

    print("\n--- Final Result ---")
    print(result)

    # Note: The result is already printed by the callback/final print statement

if __name__ == "__main__":
    main_cli() # Call the CLI entry point