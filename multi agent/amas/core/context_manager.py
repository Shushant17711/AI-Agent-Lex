"""
Context Manager for AMAS.
This module provides context management for the multi-agent system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import copy
# Configure logging
logger = logging.getLogger(__name__)

# Default structure for a new context
DEFAULT_STRUCTURE = {
    "task": None,
    "plan": [], # List of task dicts from planner
    "execution": {
        # "results": {}, # Removed - Results are now stored within each step in the 'plan' list
        "status": "pending", # Overall status: pending, running, completed, failed
        "failed_step_id": None, # ID of the step that caused failure (if any)
    },
    "workspace": {
        "directory_listing": None, # String or list of files/dirs
        "base_dir": None # Base directory path
    },
    "errors": [] # List of error messages encountered
}

class ContextManager:
    """Manages context for the multi-agent system with versioning and structure."""
    
    def __init__(self, initial_context: Optional[Dict[str, Any]] = None):
        """Initialize the context manager.
        
        Args:
            initial_context: Optional initial context dictionary.
        """
        # Initialize with default structure if no initial context provided
        self.context = initial_context or copy.deepcopy(DEFAULT_STRUCTURE)
        self.history: List[Dict[str, Any]] = [] # History of states
        self.version: int = 0 # Current version number

        # Always start history with the initial state (either loaded or default)
        self.history.append(copy.deepcopy(self.context))

        logger.info(f"Context Manager initialized at version {self.version}")
    
    @classmethod
    def load(cls, context_path: Optional[Union[str, Path]] = None) -> 'ContextManager':
        """Load context from a file.
        
        Args:
            context_path: Path to the context file.
            
        Returns:
            ContextManager instance with loaded context.
        """
        if context_path is None:
            logger.info("No context path provided, initializing empty context")
            return cls()
        
        try:
            context_path = Path(context_path)
            if not context_path.exists():
                logger.warning(f"Context file {context_path} does not exist, initializing empty context")
                return cls()
            
            with open(context_path, 'r', encoding='utf-8') as file:
                context = json.load(file)
            
            logger.info(f"Context loaded from {context_path}, initializing as version 0")
            # When loading, the loaded state becomes the initial version 0
            instance = cls()
            instance.context = context
            instance.history = [copy.deepcopy(context)]
            instance.version = 0
            return instance
        except Exception as e:
            logger.error(f"Error loading context from {context_path}: {str(e)}")
            # Return a clean instance on error
            return cls()
    
    def save(self, context_path: Union[str, Path]) -> bool:
        """Save context to a file.
        
        Args:
            context_path: Path to save the context file.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            context_path = Path(context_path)
            
            # Create directory if it doesn't exist
            context_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(context_path, 'w', encoding='utf-8') as file:
                json.dump(self.context, file, indent=4)
            
            logger.info(f"Context saved to {context_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving context to {context_path}: {str(e)}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value from the context using dot notation for nested keys.

        Args:
            key_path: Dot-separated key path (e.g., "execution.results").
            default: Default value if key path doesn't exist.

        Returns:
            Value from context or default.
        """
        try:
            keys = key_path.split('.')
            value = self.context
            for key in keys:
                # Check if key contains list index notation like "plan[0]"
                if '[' in key and key.endswith(']'):
                    base_key, index_str = key.split('[', 1)
                    index = int(index_str[:-1])  # Remove the closing bracket
                    if base_key: # Accessing dict element then list index e.g. "some_dict[0]"
                         if isinstance(value, dict) and base_key in value and isinstance(value[base_key], list):
                             value = value[base_key][index]
                         else:
                             logger.debug(f"Invalid path segment '{key}' for list indexing in '{key_path}'.")
                             return default
                    else: # Accessing list index directly e.g. "[0]" (less common for root)
                         if isinstance(value, list):
                             value = value[index]
                         else:
                             logger.debug(f"Invalid path segment '{key}' for list indexing in '{key_path}'.")
                             return default
                elif isinstance(value, dict):
                    value = value[key]
                else:
                    # Trying to access a key on a non-dict/non-list element
                    logger.debug(f"Cannot access key '{key}' in non-dict/non-list element for path '{key_path}'")
                    return default
            return value
        except (KeyError, TypeError, IndexError, ValueError):
            logger.debug(f"Key path '{key_path}' not found or invalid index in context.")
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set a value in the context using dot notation for nested keys.
           Creates intermediate dictionaries if they don't exist.

        Args:
            key_path: Dot-separated key path (e.g., "execution.results").
            value: Value to set.
        """
        # Only store history if the value actually changes
        old_value = self.get(key_path, default=object()) # Use a unique sentinel
        if old_value == value:
            logger.debug(f"Skipping version increment - value unchanged for {key_path}")
            return

        try:
            keys = key_path.split('.')
            # Store previous state before modification
            self.history.append(copy.deepcopy(self.context))
            self.version += 1

            current_level = self.context
            for i, key in enumerate(keys[:-1]):
                if key not in current_level or not isinstance(current_level[key], dict):
                    # Create intermediate dict if it doesn't exist or isn't a dict
                    current_level[key] = {}
                current_level = current_level[key]

            final_key = keys[-1]
            current_level[final_key] = value
            logger.info(f"Context updated: {key_path} (Version: {self.version})")

        except Exception as e:
            logger.error(f"Error setting context key '{key_path}': {e}", exc_info=True)
            # Attempt to rollback the failed set operation
            self.version -= 1 # Decrement version counter
            if self.history:
                 self.context = self.history.pop() # Restore previous state
                 logger.warning(f"Rolled back context state due to set error for key '{key_path}'.")
            else:
                 logger.error("Cannot rollback context state: history is empty.")
            # Re-raise the exception or handle it as needed
            raise
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple top-level values in the context.
           Nested updates via dot notation in keys are NOT supported by this method.
           Use multiple `set` calls for nested updates.

        Args:
            updates: Dictionary of top-level updates to apply.
        """
        # Store previous state before modification
        self.history.append(copy.deepcopy(self.context))
        self.version += 1

        try:
            for key, value in updates.items():
                if '.' in key:
                     logger.warning(f"Skipping update for '{key}': dot notation not supported in update(). Use set().")
                     continue
                self.context[key] = value # Apply update directly to top level

            logger.info(f"Context updated with {len(updates)} top-level items (Version: {self.version})")
        except Exception as e:
            logger.error(f"Error during context update: {e}", exc_info=True)
            # Attempt to rollback the failed update operation
            self.version -= 1
            if self.history:
                 self.context = self.history.pop()
                 logger.warning("Rolled back context state due to update error.")
            else:
                 logger.error("Cannot rollback context state: history is empty.")
            raise
    
    def clear(self) -> None:
        """Clear the context."""
        # Store previous state before clearing
        self.history.append(copy.deepcopy(self.context))
        self.version += 1
        
        self.context = copy.deepcopy(DEFAULT_STRUCTURE)
        # Add the cleared state as the new current state in history
        self.history.append(copy.deepcopy(self.context))
        # Increment version for the clear operation itself
        # Note: The previous state was already added before clearing.
        # This adds the *new* cleared state.
        self.version += 1

        logger.info(f"Context cleared and reset to default structure (Version: {self.version})")
    
    def __getitem__(self, key_path: str) -> Any:
        """Get a value from the context using dictionary syntax with dot notation.

        Args:
            key_path: Dot-separated key path to get.

        Returns:
            Value from context.

        Raises:
            KeyError: If key path doesn't exist.
        """
        value = self.get(key_path, default=KeyError) # Use KeyError as sentinel
        if value is KeyError:
            raise KeyError(key_path)
        return value
    
    def __setitem__(self, key_path: str, value: Any) -> None:
        """Set a value in the context using dictionary syntax with dot notation.

        Args:
            key_path: Dot-separated key path to set.
            value: Value to set.
        """
        self.set(key_path, value)
    
    def __contains__(self, key_path: str) -> bool:
        """Check if a key path exists in the context using dot notation.

        Args:
            key_path: Dot-separated key path to check.

        Returns:
            True if key path exists, False otherwise.
        """
        return self.get(key_path, default=KeyError) is not KeyError
    
    def __str__(self) -> str:
        """Get string representation of the context.
        
        Returns:
            String representation.
        """
        return f"Version {self.version}: {str(self.context)}"

    def get_version(self) -> int:
        """Get the current context version.

        Returns:
            Current version number.
        """
        return self.version

    def rollback(self, version: int) -> bool:
        """Rollback context to a specific version.

        Args:
            version: The version number to rollback to (0-based).

        Returns:
            True if rollback was successful, False otherwise.
        """
        if 0 <= version < len(self.history):
            try:
                # The history stores states *before* the change that led to the next version.
                # So, history[version] is the state *at* version 'version'.
                self.context = copy.deepcopy(self.history[version])
                # Trim history to the rollback point
                self.history = self.history[:version + 1]
                self.version = version
                logger.info(f"Context rolled back to version {version}")
                return True
            except Exception as e:
                logger.error(f"Error during rollback to version {version}: {str(e)}")
                return False
        else:
            logger.warning(f"Invalid version {version} for rollback. Max version: {len(self.history) - 1}")
            return False

    def validate(self) -> List[str]:
        """Validate the context structure against expected keys and types.

        Returns:
            List of validation errors, empty if valid.
        """
        errors = []
        required_keys = ["task", "plan", "execution", "workspace", "errors"]

        # Check top-level structure
        for key in required_keys:
            if key not in self.context:
                errors.append(f"Missing required top-level key: {key}")

        # Check execution structure if it exists
        if "execution" in self.context:
            execution = self.context["execution"]
            if not isinstance(execution, dict):
                errors.append("'execution' must be a dictionary")
            else:
                # Removed check for "results" key
                if "status" not in execution:
                    errors.append("'execution' missing required key: 'status'")
                # Optionally check types of results and status

        # Check plan structure if it exists
        if "plan" in self.context and not isinstance(self.context["plan"], list):
             errors.append("'plan' must be a list")

        # Check errors structure if it exists
        if "errors" in self.context and not isinstance(self.context["errors"], list):
             errors.append("'errors' must be a list")

        # Add more specific validation as needed (e.g., types within lists/dicts)

        if errors:
            logger.warning(f"Context validation failed: {errors}")
        else:
            logger.debug("Context validation successful.")

        return errors

    def prune_history(self, max_versions: int = 100) -> None:
        """Prune history to keep only the most recent versions.

        Args:
            max_versions: Maximum number of versions to keep.
        """
        if len(self.history) > max_versions:
            num_pruned = len(self.history) - max_versions
            self.history = self.history[-max_versions:]
            # Adjust version numbers if necessary? No, history indices are relative to the start.
            # The actual version number `self.version` reflects the latest state, not the history length.
            logger.info(f"Pruned {num_pruned} oldest context history versions. Keeping {len(self.history)} versions.")
