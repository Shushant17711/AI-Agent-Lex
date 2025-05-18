"""
Enhanced Orchestrator with Tool Integration for AMAS.
This module extends the base Orchestrator with tool execution capabilities.
"""

import logging
import yaml
import re # Added for task parsing
from pathlib import Path
from typing import Dict, Any, Optional, Union, Awaitable # Added Awaitable
import asyncio # Import asyncio if not already present at top

from .orchestrator import Orchestrator, BaseLLMService # Import BaseLLMService for type hint
from .context_manager import ContextManager # Import ContextManager for type hint
# Removed duplicate import of ContextManager
# Removed OrchestratorToolIntegration as it's agent-focused
from .tools.execution_engine import ToolExecutionEngine
from .tools.permissions import PermissionsManager

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedOrchestrator:
    """Enhanced Orchestrator with integrated tool execution capabilities."""
    
    def __init__(self, base_orchestrator: Orchestrator, config: Optional[Dict[str, Any]] = None, context: Optional[ContextManager] = None, log_callback: Optional[callable] = None, working_directory: Optional[Path] = None, file_focus_callback: Optional[callable] = None, file_content_callback: Optional[callable] = None):
        """Initialize the enhanced orchestrator, linking tools to agents.

        Args:
            base_orchestrator: The initialized base orchestrator instance (contains agents, llm).
            config: Configuration dictionary (used for tool engine setup). If None, uses base_orchestrator's config.
            context: Context manager instance. If None, uses base_orchestrator's context.
            log_callback: Optional function to send log messages to (e.g., GUI).
            working_directory: Optional specific directory to use for file operations, overriding config.
            file_focus_callback: Optional callback when an agent focuses on a file (path: str).
            file_content_callback: Optional callback when file content is updated (path: str, content: str).
        """
        self.base_orchestrator = base_orchestrator
        # Use the config from the base orchestrator if not provided explicitly
        self.config = config if config is not None else self.base_orchestrator.config
        self.context = context if context is not None else self.base_orchestrator.context
        
        self.log_callback = log_callback or (lambda msg, level="info": logger.info(f"[{level.upper()}] {msg}")) # Default logger if no callback
        self.file_focus_callback = file_focus_callback
        self.file_content_callback = file_content_callback

        # Initialize tool components
        self.permissions_manager = PermissionsManager(self.config.get('permissions', {}))
        # Use the config's base_dir if available for the tool engine
        # Determine the base directory for the tool engine
        # Prioritize the explicitly passed working_directory over the config file setting
        config_base_dir = self.config.get('file_operations', {}).get('base_dir')
        effective_base_dir = working_directory if working_directory is not None else config_base_dir
        if effective_base_dir:
             self.log_callback(f"Setting ToolExecutionEngine base directory to: {effective_base_dir}", "info")
        else:
             self.log_callback("ToolExecutionEngine will use the current working directory (no base_dir specified).", "info")

        self.tool_engine = ToolExecutionEngine(
            permissions_manager=self.permissions_manager,
            base_dir=effective_base_dir,
            file_focus_callback=self.file_focus_callback,
            file_content_callback=self.file_content_callback
        )
        
        # Provide the tool engine to the agents initialized in the base orchestrator
        self.base_orchestrator.setup_agent_tools(self.tool_engine)

        # Use direct logger for final init message to bypass potential callback issue
        logger.info("Enhanced Orchestrator initialized and linked tools to agents.")
        # self.log_callback("Enhanced Orchestrator initialized and linked tools to agents.", "info") # Keep original commented
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], context: Optional[ContextManager] = None, log_callback: Optional[callable] = None, working_directory: Optional[Path] = None, file_focus_callback: Optional[callable] = None, file_content_callback: Optional[callable] = None) -> 'EnhancedOrchestrator':
        """Create an enhanced orchestrator from a configuration file.

        Args:
            config_path: Path to the configuration file.
            context: Optional context manager.
            log_callback: Optional function to send log messages to.
            working_directory: Optional specific directory to use for file operations, overriding config.
            file_focus_callback: Optional callback when an agent focuses on a file (path: str).
            file_content_callback: Optional callback when file content is updated (path: str, content: str).

        Returns:
            EnhancedOrchestrator instance.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the configuration file is invalid.
        """
        # Use callback if provided, otherwise use standard logger
        effective_log = log_callback or (lambda msg, level="info": logger.info(f"[{level.upper()}] {msg}"))
        effective_log(f"Creating EnhancedOrchestrator from config: {config_path}", "info")
        
        # Load the config first to determine the base directory
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            effective_log(f"Configuration file {config_path} not found", "error")
            raise
        except yaml.YAMLError as e:
            effective_log(f"Error parsing configuration file {config_path}: {str(e)}", "error")
            raise

        # Determine the effective base directory BEFORE creating the base orchestrator
        config_base_dir = config.get('file_operations', {}).get('base_dir')
        effective_base_dir = str(working_directory) if working_directory is not None else config_base_dir # Convert Path to str if needed
        if effective_base_dir:
             effective_log(f"Determined effective base directory for Orchestrator: {effective_base_dir}", "info")
        else:
             effective_log("No base directory specified in config or arguments for Orchestrator.", "info")

        # Create the base Orchestrator instance, passing the determined base_dir
        base_orchestrator = Orchestrator.from_config(
            config_path,
            context,
            base_dir=effective_base_dir, # Pass the determined base_dir
            log_callback=log_callback
        )
        
        # Create and return the enhanced orchestrator using the loaded config
        # The base_orchestrator already has the correct base_dir set now.
        return cls(
            base_orchestrator=base_orchestrator,
            config=config, # Pass the loaded config
            context=context, # Pass context if provided
            log_callback=log_callback,
            working_directory=working_directory, # Pass original working_directory for EnhancedOrchestrator's own use if needed
            file_focus_callback=file_focus_callback,
            file_content_callback=file_content_callback
        )
    
    async def execute(self, task: str) -> str: # Changed to async def
        """Execute a task by delegating to the base orchestrator's async execute method.

        The base orchestrator now handles agent initialization and delegation.
        This enhanced orchestrator ensures tools are set up for the agents.

        Args:
            task: The task string to execute.

        Returns:
            The result string from the base orchestrator's execution flow.
        """
        self.log_callback(f"Enhanced Orchestrator passing task to base orchestrator: {task}", "info")
        # The base orchestrator's execute method should now handle the agent delegation
        # and ideally use the log_callback passed during its initialization.
        # We assume the base_orchestrator.execute method itself will use the callback.
        return await self.base_orchestrator.execute(task) # Added await
